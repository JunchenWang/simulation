import time
import numpy as np
import mujoco
import mujoco.viewer
import numpy as np
from collections import deque
from py_robotics import *
import osqp
from scipy import sparse
import matplotlib.pyplot as plt


def add_visual_sphere(
    viewer,
    pos: np.ndarray,
    radius: float = 0.03,
    rgba: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0),
) -> None:
    """
    向 viewer.user_scn 添加一个可视化球。
    只显示，不参与动力学，不参与碰撞。
    """
    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
        raise RuntimeError("viewer.user_scn geoms are full.")

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0], dtype=float),
        np.asarray(pos, dtype=float),
        np.eye(3).reshape(-1),
        np.asarray(rgba, dtype=float),
    )
    viewer.user_scn.ngeom += 1


def main():

    model_path = "mujoco_models/universal_robots_ur5e/bph_scene.xml"
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # 禁用 XML 中原本的 actuator，避免 position actuator / velocity actuator 和 qfrc_applied 叠加
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    
    # UR5e 是 6 自由度机械臂
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    init_q = np.array([0.0, -1.7766, -1.9383, -0.9974, 1.5708, 0.0])
    d.ctrl[:] = init_q
    qpos_addrs = []
    qvel_addrs = []

    p1_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "p1")
    p2_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "p2")

    for joint_name, q in zip(joint_names, init_q):
        joint_id = mujoco.mj_name2id(
            m,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name
        )

        if joint_id < 0:
            raise RuntimeError(f"Joint not found: {joint_name}")

        qpos_addrs.append(m.jnt_qposadr[joint_id])
        qvel_addrs.append(m.jnt_dofadr[joint_id])
        d.qpos[m.jnt_qposadr[joint_id]] = q
    
    mujoco.mj_forward(m, d)


    p1 = d.geom_xpos[p1_id]
    p2 = d.geom_xpos[p2_id]
    lamb = 0.95
    prcm = p1 + lamb * (p2 - p1)
    qpos_addrs = np.array(qpos_addrs, dtype=int)
    qvel_addrs = np.array(qvel_addrs, dtype=int)
    incr = 1e-2
    p2_0 = p2.copy()
    dir0 = (p2 - p1) / np.linalg.norm(p2 - p1)
    z = np.array([0.0, 0.0, 1.0])
    r0 = np.cross(dir0, z)
    r0 = r0 / np.linalg.norm(r0)
    up0 = np.cross(r0, dir0)
    
    qp = osqp.OSQP()
    # 位置 P 参数
    kp_pos = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) * 30


    # 速度 PI 参数
    kp_v = np.array([80.0, 80.0, 80.0, 40.0, 40.0, 40.0]) * 1
    ki_v = np.array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0]) * 1

    # 积分项
    vel_error_integral = np.zeros(6)

    # 积分限幅，防止 windup
    integral_limit = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    # 力矩限幅，防止仿真发散
    torque_limit = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])

    velocity_limit = np.array([1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
    d.qfrc_applied[:] = 0.0
    dt = m.opt.timestep
    error = []
    time_step = []
    with mujoco.viewer.launch_passive(m, d) as viewer:
        
        viewer.cam.lookat[:] = m.stat.center
        viewer.cam.distance = 1.2 * m.stat.extent
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -25

        start = time.time()

        while viewer.is_running() and time.time() - start < 60:
            step_start = time.time()
            if d.time == 0.0:
                add_visual_sphere(viewer, prcm, radius=0.005, rgba=(1, 1, 0, 1))

            # mj_step1 后，可以读取当前状态，并设置 qfrc_applied
            mujoco.mj_step1(m, d)
            time_step.append(d.time)
            
            q = d.qpos[qpos_addrs].copy()
            qd = d.qvel[qvel_addrs].copy()
            p1 = d.geom_xpos[p1_id]
            p2 = d.geom_xpos[p2_id]
            J1 = np.zeros((3, m.nv))
            J2 = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))
            mujoco.mj_jacGeom(m, d, J1, jacr, p1_id)
            mujoco.mj_jacGeom(m, d, J2, jacr, p2_id)
            J1 = J1[:, qvel_addrs]
            J2 = J2[:, qvel_addrs]



            Jrcm = np.block([J1 + lamb * (J2 - J1), (p2 - p1).reshape(3, 1)])
            J = np.zeros((6, Jrcm.shape[1]))
            J[0:3] = Jrcm
            J[3:6,0:6] = J2
            
            prcm_act = p1 + lamb * (p2 - p1)
            
            pitch = 0.01
            omega = 1
            radius = 0.03 * np.sin(0.05 * d.time)
            p2_d = p2_0 + np.cos(omega * d.time) * radius * r0 + np.sin(omega * d.time) * radius * up0 + pitch * omega / (2 * np.pi) * d.time * dir0


            xe = np.r_[prcm - prcm_act, p2_d - p2]
            error.append(np.array([np.linalg.norm(xe[0:3]), np.linalg.norm(xe[3:6])]))
            v = kp_pos * xe

           

            JJt = J @ J.T
            N = np.eye(J.shape[1]) - J.T @ np.linalg.solve(JJt, J)
            w = 2 * np.r_[init_q - q, 0]
            # 任务空间优先，null space 中执行二次任务（如关节回中立位置）
            vel_d = J.T @ np.linalg.solve(JJt, v) + N @ w

            
            lamb = lamb + vel_d[-1] * dt
            vel_d = np.clip(vel_d[0:-1], -velocity_limit, velocity_limit)

            # 速度环
            vel_error = vel_d - qd
            # 速度 PI 输出的是关节力矩
            tau_pi = kp_v * vel_error + ki_v * vel_error_integral

            # 补偿 bias force：重力、科氏力、离心力
            tau = d.qfrc_bias[qvel_addrs] + tau_pi
            # tau -= d.qfrc_passive

            # 只对机械臂 6 个关节限幅
            tau = np.clip(
                tau,
                -torque_limit,
                torque_limit
            )
            # 清空后重新写入外加广义力
            d.qfrc_applied[qvel_addrs] = tau

            # 积分速度误差
            vel_error_integral += vel_error * dt

            # anti-windup：积分限幅
            vel_error_integral = np.clip(
                vel_error_integral,
                -integral_limit,
                integral_limit
            )

            mujoco.mj_step2(m, d)
            if int(d.time * 100) % 5 == 0: # 每 0.05 秒画一个小球，显示末端位置
                add_visual_sphere(viewer, p2, radius=0.0005, rgba=(1, 0, 0, 1))

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            # print(time.time() - step_start)

        
        error = np.asarray(error)
        time_step = np.asarray(time_step)
        print(np.max(error, axis=0))
        plt.plot(time_step, error)
        plt.show()


if __name__ == "__main__":
    main()