# SO-ARM100 Reinforcement Learning

Reinforcement learning for a 6-DOF robotic arm ([SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)) using PPO in PyBullet simulation.

The training is split into two curriculum phases:

| Phase | Objective | Action dim | Obs dim |
|-------|-----------|-----------|---------|
| **1 — Reaching** | Move end-effector to target | 6 joints | 15 |
| **2 — Press-down** | Align XY then press onto target | 5 joints | 14 |

## Repository structure

```
├── assets/              # URDF & mesh files (place so_arm100.urdf here)
├── configs/train.yaml   # Hyperparameter defaults
├── envs/                # Gymnasium environments
│   ├── base_env.py      #   Shared simulation logic
│   ├── phase1_env.py    #   Phase 1: 6-joint reaching
│   └── phase2_env.py    #   Phase 2: 5-joint press-down
├── models/              # Saved model checkpoints
├── scripts/
│   ├── train.py         # Training CLI
│   └── evaluate.py      # Inference / visualization CLI
├── requirements.txt
└── README.md
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your URDF
cp /path/to/so_arm100.urdf assets/

# 3. Train Phase 1 (headless)
python scripts/train.py --phase 1 --timesteps 300000

# 4. Train Phase 2 (fine-tune from Phase 1)
python scripts/train.py --phase 2 --pretrained models/arm_phase1_model.zip

# 5. Visualize results
python scripts/evaluate.py --model models/arm_phase2_model.zip --phase 2
```

## Configuration

You can override any setting via CLI flags:

```bash
python scripts/train.py --phase 1 \
    --lr 1e-3 \
    --timesteps 500000 \
    --render human \
    --randomize-target \
    --urdf /custom/path/arm.urdf \
    --log-dir ./my_logs/
```

Or set the URDF path globally: `export ARM_URDF_PATH=/path/to/so_arm100.urdf`

TensorBoard logs are written to `./logs/` by default:

```bash
tensorboard --logdir logs/
```

## Reward design

**Phase 1** combines distance shaping (`−5·d`), a progress bonus for getting closer, orientation alignment, and a small action penalty. Success threshold: distance < 4 cm.

**Phase 2** keeps XY accuracy from Phase 1 and adds a gated z-press reward that only activates once XY error < 2 cm, encouraging the arm to approach from above then push down. Success: XY < 2 cm and z-gap < 1.5 cm.

## Status

🚧 Under active development — contributions and issues welcome.
