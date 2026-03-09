import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3 import PPO


class ArmEnvPhase2(gym.Env):

    def __init__(self):
        super().__init__()

        # 动作空间（关节增量）
        self.action_space = spaces.Box(
            low=-0.08,
            high=0.08,
            shape=(5,),
            dtype=np.float32
        )

        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.urdf_path = r"C:\A-MakeBot\SO-ARM100-main\Simulation\SO100\so_arm100.urdf"

        self.max_steps = 200
        self.reset()

    def create_box(self):
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02]*3)
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.02]*3,
            rgbaColor=[1, 0, 0, 1]
        )
        return p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[0.25, 0, 0.05]
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)

        self.joints = []
        for i in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, i)[2] == p.JOINT_REVOLUTE:
                self.joints.append(i)

        self.ee_index = self.joints[-1]
        self.box = self.create_box()

        self.step_counter = 0

        # 初始化 z 距离
        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[0])
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0])
        self.prev_z_dist = ee_pos[2] - box_pos[2]

        return self.get_state(), {}

    def get_state(self):
        joint_states = [
            p.getJointState(self.robot, j)[0]
            for j in self.joints[:5]
        ]

        box_pos, _ = p.getBasePositionAndOrientation(self.box)
        ee_pos = p.getLinkState(self.robot, self.ee_index)[0]

        relative = np.array(box_pos) - np.array(ee_pos)

        return np.array(
            joint_states + list(box_pos) + list(ee_pos) + list(relative),
            dtype=np.float32
        )

    def compute_alignment(self):
        link_state = p.getLinkState(self.robot, self.ee_index)
        rot_matrix = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)
        ee_z = rot_matrix[:, 2]

        box_rot = p.getBasePositionAndOrientation(self.box)[1]
        box_matrix = np.array(p.getMatrixFromQuaternion(box_rot)).reshape(3, 3)
        box_z = box_matrix[:, 2]

        return np.dot(ee_z, box_z)

    def step(self, action):

        for i, j in enumerate(self.joints[:5]):
            current = p.getJointState(self.robot, j)[0]
            target = current + action[i]

            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=200
            )

        p.stepSimulation()

        self.step_counter += 1

        state = self.get_state()

        # =====================
        # Phase2 Reward
        # =====================

        ee_pos = np.array(p.getLinkState(self.robot, self.ee_index)[0])
        box_pos = np.array(p.getBasePositionAndOrientation(self.box)[0])

        xy_dist = np.linalg.norm(ee_pos[:2] - box_pos[:2])
        z_dist = ee_pos[2] - box_pos[2]

        align = self.compute_alignment()

        reward = 0

        # 保留 Phase1 XY 能力
        reward += -4 * xy_dist
        reward += 2 * align

        # 下压门控
        if xy_dist < 0.02:
            reward += -6 * abs(z_dist)
            reward += 10 * (self.prev_z_dist - z_dist)

        # 小动作惩罚
        reward -= 0.002 * np.linalg.norm(action)

        self.prev_z_dist = z_dist

        terminated = False
        truncated = False

        # 成功条件：对齐 + 压到顶部
        if xy_dist < 0.02 and abs(z_dist) < 0.015:
            reward += 30
            terminated = True

        if self.step_counter >= self.max_steps:
            truncated = True

        return state, reward, terminated, truncated, {}

    def close(self):
        p.disconnect()


# =============================
# 训练
# =============================

if __name__ == "__main__":

    env = ArmEnvPhase2()

    # 继续 Phase1
    model = PPO.load("arm_phase1_model", env=env)

    print("Phase2 开始训练...")
    model.learn(
        total_timesteps=300000,
        reset_num_timesteps=False
    )

    model.save("arm_phase2_model")

    print("测试阶段")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()