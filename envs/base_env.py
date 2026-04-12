"""
Base Gym environment for the SO-ARM100 robotic arm in PyBullet.

Extracts the common simulation setup, joint discovery, target object creation,
state observation, and alignment computation shared by all training phases.
"""

import os
from abc import abstractmethod

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class ArmEnvBase(gym.Env):
    """Abstract base environment for robotic arm RL training.

    Subclasses must implement ``_build_action_space``, ``_build_observation_space``,
    ``_compute_reward``, and ``_check_termination``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        urdf_path: str | None = None,
        max_steps: int = 200,
        render_mode: str = "human",
        target_pos: list[float] | None = None,
        randomize_target: bool = False,
    ):
        super().__init__()

        # --- URDF resolution (configurable, no hardcoded paths) ---
        if urdf_path is None:
            urdf_path = os.environ.get(
                "ARM_URDF_PATH",
                os.path.join(os.path.dirname(__file__), "..", "assets", "so_arm100.urdf"),
            )
        self.urdf_path = urdf_path

        self.max_steps = max_steps
        self.render_mode = render_mode
        self.target_pos = target_pos or [0.25, 0.0, 0.05]
        self.randomize_target = randomize_target

        # Subclasses define these
        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()

        # Connect to physics engine
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.physics_client = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)

        # Will be set in reset()
        self.robot = None
        self.joints: list[int] = []
        self.ee_index: int = 0
        self.box = None
        self.step_counter = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def _build_action_space(self) -> spaces.Space:
        ...

    @abstractmethod
    def _build_observation_space(self) -> spaces.Space:
        ...

    @abstractmethod
    def _compute_reward(self, action: np.ndarray) -> float:
        ...

    @abstractmethod
    def _check_termination(self) -> tuple[bool, bool]:
        """Return (terminated, truncated)."""
        ...

    @abstractmethod
    def _get_active_joints(self) -> list[int]:
        """Return list of joint indices to control."""
        ...

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    def _create_target_box(self) -> int:
        pos = list(self.target_pos)
        if self.randomize_target:
            pos[0] += np.random.uniform(-0.05, 0.05)
            pos[1] += np.random.uniform(-0.05, 0.05)

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.02] * 3)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.02] * 3, rgbaColor=[1, 0, 0, 1]
        )
        return p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
        )

    def _discover_joints(self) -> list[int]:
        """Find all revolute joints in the loaded URDF."""
        joints = []
        for i in range(p.getNumJoints(self.robot)):
            if p.getJointInfo(self.robot, i)[2] == p.JOINT_REVOLUTE:
                joints.append(i)
        return joints

    def get_ee_pos(self) -> np.ndarray:
        return np.array(p.getLinkState(self.robot, self.ee_index)[0])

    def get_box_pos(self) -> np.ndarray:
        return np.array(p.getBasePositionAndOrientation(self.box)[0])

    def compute_distance(self) -> float:
        return float(np.linalg.norm(self.get_ee_pos() - self.get_box_pos()))

    def compute_alignment(self) -> float:
        """Dot product between end-effector z-axis and target z-axis."""
        link_state = p.getLinkState(self.robot, self.ee_index)
        rot = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)
        ee_z = rot[:, 2]

        box_rot = p.getBasePositionAndOrientation(self.box)[1]
        box_mat = np.array(p.getMatrixFromQuaternion(box_rot)).reshape(3, 3)
        box_z = box_mat[:, 2]

        return float(np.dot(ee_z, box_z))

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        self.robot = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.joints = self._discover_joints()
        self.ee_index = self.joints[-1]
        self.box = self._create_target_box()
        self.step_counter = 0

        self._on_reset()  # hook for subclasses

        return self._get_obs(), {}

    def _on_reset(self):
        """Override in subclasses for extra reset logic."""
        pass

    def step(self, action):
        active_joints = self._get_active_joints()
        for i, j in enumerate(active_joints):
            current = p.getJointState(self.robot, j)[0]
            p.setJointMotorControl2(
                self.robot,
                j,
                p.POSITION_CONTROL,
                targetPosition=current + action[i],
                force=200,
            )

        p.stepSimulation()
        self.step_counter += 1

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated, truncated = self._check_termination()

        return obs, reward, terminated, truncated, {}

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
