"""
Phase 2: 5-joint press-down — align XY then press end-effector onto target.

Observation (14-dim): 5 joint angles + 3 box pos + 3 ee pos + 3 relative vec
Action (5-dim): joint angle increments ∈ [-0.08, 0.08]
"""

import numpy as np
from gymnasium import spaces
import pybullet as p

from envs.base_env import ArmEnvBase


class ArmEnvPhase2(ArmEnvBase):

    def _build_action_space(self):
        return spaces.Box(low=-0.08, high=0.08, shape=(5,), dtype=np.float32)

    def _build_observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)

    def _get_active_joints(self):
        return self.joints[:5]

    def _on_reset(self):
        ee_pos = self.get_ee_pos()
        box_pos = self.get_box_pos()
        self.prev_z_dist = ee_pos[2] - box_pos[2]

    def _get_obs(self):
        joint_angles = [
            p.getJointState(self.robot, j)[0] for j in self.joints[:5]
        ]
        box_pos = self.get_box_pos()
        ee_pos = self.get_ee_pos()
        relative = box_pos - ee_pos
        return np.concatenate([joint_angles, box_pos, ee_pos, relative]).astype(np.float32)

    def _compute_reward(self, action):
        ee_pos = self.get_ee_pos()
        box_pos = self.get_box_pos()
        xy_dist = float(np.linalg.norm(ee_pos[:2] - box_pos[:2]))
        z_dist = ee_pos[2] - box_pos[2]
        align = self.compute_alignment()

        reward = (
            -4.0 * xy_dist                       # maintain XY accuracy
            + 2.0 * align                         # orientation alignment
            - 0.002 * np.linalg.norm(action)      # action penalty
        )

        # Gated z-press: only encourage downward motion once XY is aligned
        if xy_dist < 0.02:
            reward += -6.0 * abs(z_dist)
            reward += 10.0 * (self.prev_z_dist - z_dist)

        # Success: XY aligned AND pressed onto target
        if xy_dist < 0.02 and abs(z_dist) < 0.015:
            reward += 30.0

        self.prev_z_dist = z_dist
        return reward

    def _check_termination(self):
        ee_pos = self.get_ee_pos()
        box_pos = self.get_box_pos()
        xy_dist = float(np.linalg.norm(ee_pos[:2] - box_pos[:2]))
        z_dist = ee_pos[2] - box_pos[2]

        terminated = xy_dist < 0.02 and abs(z_dist) < 0.015
        truncated = self.step_counter >= self.max_steps
        return terminated, truncated
