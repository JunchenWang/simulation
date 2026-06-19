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

ctrl_value_ = 0

def key_callback(keycode):
    global ctrl_value_

    if keycode == 265: # up
        ctrl_value_ = 1

    elif keycode == 264: # down
        ctrl_value_ = 2

    elif keycode == 262: # right
        ctrl_value_ = 3

    elif keycode == 263: # left
        ctrl_value_ = 4

    elif keycode == ord('8'):
        ctrl_value_ = 5

    elif keycode == ord('9'):
        ctrl_value_ = 6


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


def add_line(viewer, p0, p1, rgba=(1.0, 0.0, 0.0, 1.0), width=3):
    """在 viewer.user_scn 中添加一条线段"""

    if viewer.user_scn.ngeom >= len(viewer.user_scn.geoms):
        return

    geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]

    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=np.array(rgba, dtype=np.float32)
    )

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(p0, dtype=np.float64),
        np.asarray(p1, dtype=np.float64)
    )

    viewer.user_scn.ngeom += 1


def draw_trajectory(viewer, points):
    """把末端轨迹画到 viewer 中"""

    viewer.user_scn.ngeom = 1

    if len(points) < 2:
        return

    points_list = list(points)

    max_lines = len(viewer.user_scn.geoms)
    start_index = max(0, len(points_list) - max_lines - 1)

    for i in range(start_index, len(points_list) - 1):
        add_line(
            viewer,
            points_list[i],
            points_list[i + 1],
            rgba=(1.0, 0.0, 0.0, 1.0),
            width=3
        )

def main():
    global ctrl_value_
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
    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        
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
            

            # 第二种计算
            # u = prcm - p1
            # w = p2 - p1
            # w2 = np.linalg.norm(w)**2
            # a = 2 * np.dot(u, w) * w - (u + w) / w2
            # b = u / w2 - 2 * np.dot(u, w) * w
            # Jlamb = a @ J1 + b @ J2
            # lamb = np.dot(u, w) / w2
            # Jrcm = J1 + lamb * (J2 - J1) + w.reshape(-1,1) @ Jlamb.reshape(1, -1)
            # J = np.vstack([Jrcm, J2])

            # u = prcm - p1
            # w = p2 - p1
            # lamb_real = np.dot(u, w) / np.linalg.norm(w)**2
            # prcm_act_real = p1 + lamb_real * (p2 - p1)
            # # print("lambda real:", lamb_real)
            # error_real = np.linalg.norm(prcm_act_real - prcm)

            prcm_act = p1 + lamb * (p2 - p1)

            dir = (p2 - p1) / np.linalg.norm(p2 - p1)
            r = np.cross(dir, z)
            r = r / np.linalg.norm(r)
            up = np.cross(r, dir)
            
            pitch = 0.01
            omega = 1
            radius = 0.03 * np.sin(0.05 * d.time)
            p2_d = p2_0 + np.cos(omega * d.time) * radius * r0 + np.sin(omega * d.time) * radius * up0 + pitch * omega / (2 * np.pi) * d.time * dir0
            # if ctrl_value_ == 1:
            #     p2_d = p2 + incr * dir
            #     ctrl_value_ = 0
            # elif ctrl_value_ == 2:
            #     p2_d = p2 - incr * dir
            #     ctrl_value_ = 0
            # elif ctrl_value_ == 3:
            #     p2_d = p2 + incr * r
            #     ctrl_value_ = 0
            # elif ctrl_value_ == 4:
            #     p2_d = p2 - incr * r    
            #     ctrl_value_ = 0
            # elif ctrl_value_ == 5:
            #     p2_d = p2 + incr * z    
            #     ctrl_value_ = 0
            # elif ctrl_value_ == 6:
            #     p2_d = p2 - incr * z 
            #     ctrl_value_ = 0
  

            xe = np.r_[prcm - prcm_act, p2_d - p2]
            error.append(np.array([np.linalg.norm(xe[0:3]), np.linalg.norm(xe[3:6])]))
            # error.append(np.array([error_real, np.linalg.norm(xe[3:6])]))
            v = kp_pos * xe

            # alpha = 1e-3
            # H = sparse.csc_matrix(J.T @ J + (alpha + 0.01**2) * np.eye(J.shape[1]))
            # # f = -J.T @ (kp_pos * xe) - alpha * np.r_[init_q - q, 0] 
            # f = -J.T @ (kp_pos * xe) - alpha * (init_q - q)   
            # # qp.setup(P = H, q = f, A = sparse.csc_matrix(np.eye(J.shape[1])), l = np.r_[-velocity_limit, 0], u = np.r_[velocity_limit, 1],verbose=False)
            # qp.setup(P = H, q = f, A = sparse.csc_matrix(np.eye(J.shape[1])), l = -velocity_limit, u = velocity_limit,verbose=False)
            # vel_d = qp.solve().x

            # w1 = 1e4
            # w2 = 1
            # P = np.block( [ [w2 * np.eye(J.shape[1]), np.zeros((J.shape[1], J.shape[1]))], [np.zeros((J.shape[0], J.shape[0])), w1 * np.eye(J.shape[0])]])
            # q = np.r_[-2 * (init_q - q), np.zeros(init_q.shape[0])]
            # A = np.block([[J, np.eye(J.shape[0])], [np.eye(J.shape[1]), np.zeros((J.shape[0], J.shape[0]))]])
            # qp.setup(P = sparse.csc_matrix(P), q = q, A = sparse.csc_matrix(A), l = np.r_[v, -velocity_limit], u = np.r_[v, velocity_limit], verbose=False)
            # vel_d = qp.solve().x[0:J.shape[1]]

            JJt = J @ J.T
            # JJt = JJt + 0.01**2 * np.eye(JJt.shape[0])
            # if np.linalg.cond(JJt) > 1e6:
            #     print('singulatirty')
            #     JJt = JJt + 0.01**2 * np.eye(JJt.shape[0])
            N = np.eye(J.shape[1]) - J.T @ np.linalg.solve(JJt, J)
            w = 2 * np.r_[init_q - q, 0]
            # w = 2 * (init_q - q)
            # 任务空间优先，null space 中执行二次任务（如关节回中立位置）
            vel_d = J.T @ np.linalg.solve(JJt, v) + N @ w

            if np.max(np.abs(vel_d)) > 1.5 + 1e-3:
                print("clip")
            # print(np.max(np.abs(vel_d)))
            
            lamb = lamb + vel_d[-1] * dt
            vel_d = np.clip(vel_d[0:-1], -velocity_limit, velocity_limit)
            # vel_d = np.clip(vel_d, -velocity_limit, velocity_limit)

            # 速度环
            vel_error = vel_d - qd
            # 速度 PI 输出的是关节力矩
            tau_pi = kp_v * vel_error + ki_v * vel_error_integral

            # 补偿 bias force：重力、科氏力、离心力
            # 补偿 passive force：关节阻尼、弹簧、frictionloss 等
            # tau_pi_final = np.zeros(m.nv)
            # mujoco.mj_mulM(m, d, tau_pi_final, tau_pi)
            # tau = d.qfrc_bias[qvel_addrs] + tau_pi_final
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

            # trajectory_points_.append(p2)
            # 绘制轨迹
            # draw_trajectory(viewer, trajectory_points_)
            if int(d.time * 100) % 5 == 0: # 每 0.05 秒画一个小球，显示末端位置
                add_visual_sphere(viewer, p2, radius=0.0005, rgba=(1, 0, 0, 1))

            viewer.sync()

            # if int(d.time * 100) % 100 == 0:
            #     print("time:", round(d.time, 3))
            #     print("q:", np.round(q, 3))
            #     print("qd:", np.round(qd, 3))
            #     # print("vel_error:", np.round(vel_error, 3))
            #     # print("tau:", np.round(tau, 3))
            #     # jacp = np.zeros((3, m.nv))
            #     # jacr = np.zeros((3, m.nv))

            #     # mujoco.mj_jacSite(m, d, jacp, jacr, body_end)
            
            #     # J1 = d.site_xmat[body_end].reshape(3, 3).T @ jacr
            #     # J2 = d.site_xmat[body_end].reshape(3, 3).T @ jacp
            #     # J = np.vstack((J1, J2))
            #     # JJ, T = jacobian_matrix(robot, q)
            #     # print(np.linalg.norm(J - JJ))
            #     print()

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