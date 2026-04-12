"""
Phase 1: 6-joint reaching — move end-effector to target position.

Observation (15-dim): 6 joint angles + 3 box pos + 3 ee pos + 3 relative vec
Action (6-dim): joint angle increments ∈ [-0.05, 0.05]
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from envs.base_env import ArmEnvBase


class ArmEnvPhase1(ArmEnvBase):

    def _build_action_space(self):
        return spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)

    def _build_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

    def _get_active_joints(self):
        return self.joints[:6]

    def _on_reset(self):
        self.prev_dist = self.compute_distance()

    def _get_obs(self):
        joint_angles = [
            p.getJointState(self.robot, j)[0] for j in self.joints[:6]
        ]
        box_pos = self.get_box_pos()
        ee_pos = self.get_ee_pos()
        relative = box_pos - ee_pos
        return np.concatenate([joint_angles, box_pos, ee_pos, relative]).astype(np.float32)

    def _compute_reward(self, action):
        dist = self.compute_distance()
        align = self.compute_alignment()

        reward = (
            -5.0 * dist                          # distance shaping
            + 2.0 * (self.prev_dist - dist)       # progress bonus
            + 2.0 * align                         # orientation alignment
            - 0.01 * np.linalg.norm(action)       # action penalty
        )

        if dist < 0.04:
            reward += 10.0  # success bonus (also triggers termination)

        self.prev_dist = dist
        return reward

    def _check_termination(self):
        terminated = self.compute_distance() < 0.04
        truncated = self.step_counter >= self.max_steps
        return terminated, truncated

