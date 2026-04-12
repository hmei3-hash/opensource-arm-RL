#!/usr/bin/env python3
"""
Evaluate a trained model in the PyBullet GUI.

Usage:
    python scripts/evaluate.py --model models/arm_phase1_model.zip --phase 1
    python scripts/evaluate.py --model models/arm_phase2_model.zip --phase 2 --episodes 5
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from envs import ArmEnvPhase1, ArmEnvPhase2


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained arm model")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2])
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes (0 = infinite loop)")
    parser.add_argument("--urdf", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=1.0 / 240,
                        help="Seconds between sim steps for visualization")
    return parser.parse_args()


def main():
    args = parse_args()
    EnvClass = ArmEnvPhase1 if args.phase == 1 else ArmEnvPhase2
    env = EnvClass(urdf_path=args.urdf, render_mode="human")

    model = PPO.load(args.model, env=env)
    print(f"Loaded {args.model} — evaluating phase {args.phase}")

    ep = 0
    obs, _ = env.reset()
    total_reward = 0.0
    successes = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        time.sleep(args.sleep)

        if terminated or truncated:
            ep += 1
            if terminated:
                successes += 1
            print(f"  Episode {ep}: reward={total_reward:.2f}  success={terminated}")
            total_reward = 0.0

            if 0 < args.episodes <= ep:
                break
            obs, _ = env.reset()

    print(f"\nDone — {successes}/{ep} successful episodes")
    env.close()


if __name__ == "__main__":
    main()
