#!/usr/bin/env python3
"""
Train SO-ARM100 robotic arm with PPO.

Usage:
    python scripts/train.py --phase 1
    python scripts/train.py --phase 2 --pretrained models/arm_phase1_model.zip
    python scripts/train.py --phase 1 --timesteps 500000 --render human
"""

import argparse
import os
import sys

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO

from envs import ArmEnvPhase1, ArmEnvPhase2


def parse_args():
    parser = argparse.ArgumentParser(description="Train SO-ARM100 RL agent")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2],
                        help="Training phase (1=reaching, 2=press-down)")
    parser.add_argument("--timesteps", type=int, default=300_000,
                        help="Total training timesteps")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained model .zip (required for phase 2)")
    parser.add_argument("--render", type=str, default="direct",
                        choices=["human", "direct"],
                        help="Render mode: 'human' for GUI, 'direct' for headless")
    parser.add_argument("--urdf", type=str, default=None,
                        help="Path to robot URDF file")
    parser.add_argument("--randomize-target", action="store_true",
                        help="Randomize target box position each episode")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Where to save the trained model")
    parser.add_argument("--log-dir", type=str, default="./logs/",
                        help="TensorBoard log directory")
    return parser.parse_args()


def make_env(args):
    EnvClass = ArmEnvPhase1 if args.phase == 1 else ArmEnvPhase2
    return EnvClass(
        urdf_path=args.urdf,
        render_mode=args.render,
        randomize_target=args.randomize_target,
    )


def main():
    args = parse_args()
    env = make_env(args)

    # Build or load model
    if args.pretrained:
        print(f"Loading pretrained model: {args.pretrained}")
        model = PPO.load(args.pretrained, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            tensorboard_log=os.path.join(args.log_dir, f"phase{args.phase}"),
        )

    # Train
    reset_num = args.phase != 2  # continue timestep counter for phase 2
    print(f"Phase {args.phase} — training for {args.timesteps:,} timesteps ...")
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=reset_num,
    )

    # Save
    save_path = args.save_path or f"models/arm_phase{args.phase}_model"
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
