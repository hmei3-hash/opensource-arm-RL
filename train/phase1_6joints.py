import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO


class ArmEnvPhase1(gym.Env):

    def __init__(self):
        super().__init__()

        # ===== 6关节动作 =====
        self.action_space = spaces.Box(
            low=np.array([-0.05]*6),
            high=np.array([0.05]*6),
            dtype=np.float32
        )

        # ===== 状态空间 15维（现在可以喂6关节）=====
        # 6 joint + 3 box + 3 ee + 3 relative
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(15,),
            dtype=np.float32
        )

        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        self.urdf_path = r"C:\A-MakeBot\SO-ARM100-main\Simulation\SO100\so_arm100.urdf"

        self.max_steps = 200
        self.reset()

    # =========================
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

    # =========================
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

        self.arm_joints = self.joints[:6]
        self.ee_index = self.arm_joints[-1]

        self.box = self.create_box()

        self.step_counter = 0
        self.prev_dist = self.compute_distance()

        return self.get_state(), {}

    # =========================
    def get_state(self):

        joint_states = [
            p.getJointState(self.robot, j)[0]
            for j in self.arm_joints
        ]

        box_pos, _ = p.getBasePositionAndOrientation(self.box)
        ee_pos = p.getLinkState(self.robot, self.ee_index)[0]

        relative = np.array(box_pos) - np.array(ee_pos)

        return np.array(
            joint_states
            + list(box_pos)
            + list(ee_pos)
            + list(relative),
            dtype=np.float32
        )

    # =========================
    def compute_distance(self):

        ee_pos = p.getLinkState(self.robot, self.ee_index)[0]
        box_pos, _ = p.getBasePositionAndOrientation(self.box)

        return np.linalg.norm(np.array(ee_pos) - np.array(box_pos))

    # =========================
    def compute_alignment(self):

        link_state = p.getLinkState(self.robot, self.ee_index)
        rot_matrix = np.array(
            p.getMatrixFromQuaternion(link_state[1])
        ).reshape(3, 3)

        ee_z = rot_matrix[:, 2]

        box_rot = p.getBasePositionAndOrientation(self.box)[1]
        box_matrix = np.array(
            p.getMatrixFromQuaternion(box_rot)
        ).reshape(3, 3)

        box_z = box_matrix[:, 2]

        return np.dot(ee_z, box_z)

    # =========================
    def step(self, action):

        for i, j in enumerate(self.arm_joints):

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
        time.sleep(1./240.)

        self.step_counter += 1

        state = self.get_state()

        # ===== reward =====
        reward = 0

        dist = self.compute_distance()
        align = self.compute_alignment()

        # 距离 shaping
        reward += -5 * dist

        # progress reward
        reward += 2 * (self.prev_dist - dist)
        #model move back due to this reward 
        # 姿态对齐
        reward += 2 * align

        # 动作惩罚
        reward -= 0.01 * np.linalg.norm(action)

        self.prev_dist = dist

        terminated = False
        truncated = False

        if dist < 0.04:
            reward += 10
            terminated = True

        if self.step_counter >= self.max_steps:
            truncated = True

        return state, reward, terminated, truncated, {}

    def close(self):
        p.disconnect()


# =========================================
# 训练
# =========================================
if __name__ == "__main__":

    env = ArmEnvPhase1()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-3,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="./ppo_arm_phase1_v3/"
    )

    print("开始训练 Phase1 v3 (6关节)...")

    model.learn(total_timesteps=300000)

    model.save("arm_phase1_v3_model")

    print("测试阶段")

    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()