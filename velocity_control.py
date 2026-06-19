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


def main():
    global ctrl_value_
    model_path = "mujoco_models/universal_robots_ur5e/scene.xml"
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



    # 目标关节速度，单位 rad/s
    vel_d = np.array([0.2, -0.2, 0.15, 0.1, 0.1, 0.1])

    # 速度 PI 参数
    kp_v = np.array([40.0, 40.0, 40.0, 20.0, 20.0, 20.0]) * 2
    ki_v = np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0]) * 2

    # 积分项
    vel_error_integral = np.zeros(6)

    # 积分限幅，防止 windup
    integral_limit = np.array([1.0, 1.0, 1.0, 0.5, 0.5, 0.5])

    # 力矩限幅，防止仿真发散
    torque_limit = np.array([150.0, 150.0, 150.0, 50.0, 50.0, 50.0])

    velocity_limit = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0])
    d.qfrc_applied[:] = 0.0
    dt = m.opt.timestep
    error = []
    time_step = []
    with mujoco.viewer.launch_passive(m, d) as viewer:

        start = time.time()

        while viewer.is_running() and time.time() - start < 60:
            step_start = time.time()
  
            # mj_step1 后，可以读取当前状态，并设置 qfrc_applied
            mujoco.mj_step1(m, d)
            q = d.qpos[qpos_addrs].copy()
            qd = d.qvel[qvel_addrs].copy()
            if np.max(q) > 0.8 * np.pi or np.min(q) < -0.8 * np.pi:
                print("break")
                break

            time_step.append(d.time)
            vel_d =  np.ones(m.nv)
            
            # vel_d = np.clip(vel_d, -velocity_limit, velocity_limit)

            # 速度环
            vel_error = vel_d - qd
            error.append(qd)
            # 速度 PI 输出的是关节力矩
            tau_pi = kp_v * vel_error + ki_v * vel_error_integral

            # 补偿 bias force：重力、科氏力、离心力
            # 补偿 passive force：关节阻尼、弹簧等
            # tau_pi_final = np.zeros(m.nv)
            # mujoco.mj_mulM(m, d, tau_pi_final, tau_pi)
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
            mujoco.mj_step2(m, d)

            # 积分速度误差
            vel_error_integral += vel_error * dt

            # anti-windup：积分限幅
            vel_error_integral = np.clip(
                vel_error_integral,
                -integral_limit,
                integral_limit
            )

            viewer.sync()

            # if int(d.time * 100) % 100 == 0:
            #     print("time:", round(d.time, 3))
            #     print("q:", np.round(q, 3))
            #     print("qd:", np.round(qd, 3))
            #     # print("vel_error:", np.round(vel_error, 3))
            #     # print("tau:", np.round(tau, 3))
            #     # jacp = np.zeros((3, m.nv))
            #     # jacr = np.zeros((3, m.nv))


            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

        
        error = np.asarray(error)
        time_step = np.asarray(time_step)
        print(np.max(error, axis=0))
        plt.plot(time_step, error)
        plt.show()


if __name__ == "__main__":
    main()