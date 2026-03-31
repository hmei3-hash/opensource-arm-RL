# opensource-arm-RL
Project Overview

This project focuses on applying reinforcement learning (RL) to control a 6-DOF robotic arm based on the open-source SO101 model. The goal is to enable the robot to learn task-oriented behaviors such as approaching and interacting with objects through carefully designed action and reward spaces.

Methodology
Action Space Design
Defined a continuous action space corresponding to the 6 joint controls of the robotic arm, enabling smooth and precise motion control.
Reward Function Engineering
Designed task-specific reward functions to guide learning objectives, including:
Minimizing distance between end-effector and target object
Encouraging stable and efficient motion
Penalizing unnecessary movements or instability
Model Training
Leveraged existing pretrained models and fine-tuned them using reinforcement learning to improve task performance and convergence speed.
Current Progress
Implemented RL framework for robotic arm control
Defined action space and initial reward structures
Integrated and fine-tuned existing models for task-specific training
Achieved preliminary results for target-reaching behavior
Future Work
Improve reward shaping for more complex tasks (e.g., grasping, manipulation)
Optimize training stability and sample efficiency
Integrate perception (vision system) for real-world interaction
Deploy on physical robotic arm hardware
Status

🚧 Currently under active development
