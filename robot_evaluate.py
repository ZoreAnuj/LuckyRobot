import argparse
from stable_baselines3 import PPO
from pybullet_envs.robot_env import RobotEnv
import numpy as np
import matplotlib.pyplot as plt

import time


model_path = 'ppo_robot_bound_best'

# Load the trained model
model = PPO.load(model_path)

# Create the environment
env = RobotEnv("urdf/quad.urdf", render = True)

# Test the trained model
obs = env.reset()

joint_position_list = np.empty((8, 0))
num_steps = 10000

for _ in range(num_steps):
    #time.sleep(2)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    joint_positions  = env.render()

    joint_positions_transposed = (np.array(joint_positions)).reshape((8,1))
    joint_position_list = np.append(joint_position_list, joint_positions_transposed, axis=1)


# time_list = np.linspace(0,num_steps,num_steps)

# np.savetxt('robot_evaluate.csv', joint_position_list, delimiter=',')
