# LuckyRobot

# Robot Simulation and Training Environment

This project implements a robot simulation environment using PyBullet physics engine and OpenAI Gym interface, along with a PPO (Proximal Policy Optimization) based training system for robotic control.

## Overview

The project consists of three main components:
- A custom robot environment (`robot_env.py`)
- A training script (`robot_train.py`)
- An evaluation script (`robot_evaluate.py`)

## Environment Features

- 8-joint robot simulation with realistic physics
- Customizable camera views and rendering options
- Configurable physics parameters including gravity and friction
- Comprehensive state observation including:
  - Joint positions and velocities
  - Base position and orientation
  - Linear and angular velocities
- Reward system based on:
  - Forward progress
  - Energy efficiency
  - Stability
  - Movement smoothness

## Requirements

- Python 3.x
- PyBullet
- OpenAI Gym
- Stable-Baselines3
- NumPy
- Matplotlib

## Project Structure

```
.
├── robot_env.py      # Main environment implementation
├── robot_train.py    # Training script using PPO
├── robot_evaluate.py # Evaluation and visualization script
└── urdf/
    └── quad.urdf    # Robot model definition
```

## Usage

## Download URDF file

urdf file: https://drive.google.com/drive/folders/1etM8HWP-oHdVJIvqlKCZ1V-WJAyxhUH5?usp=sharing

### Training

To train the robot:

```bash
python robot_train.py
```

This will:
- Initialize the environment with default parameters
- Train a PPO model for 1,000,000 timesteps
- Save the trained model as "ppo_robot"

Training parameters:
- Learning rate: 0.0005
- Steps per update: 4096
- Batch size: 256

### Evaluation

To evaluate a trained model:

```bash
python robot_evaluate.py
```

This will:
- Load the trained model
- Run a simulation with visualization
- Record joint positions over time
- Optionally save the data to CSV for analysis
