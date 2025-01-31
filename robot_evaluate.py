import argparse
from stable_baselines3 import PPO
from pybullet_envs.robot_env import RobotEnv
import numpy as np
import matplotlib.pyplot as plt
import time

# Path to the pre-trained model
model_path = 'ppo_robot_bound_best'

# Load the trained model
model = PPO.load(model_path)

# Create the robot simulation environment with rendering enabled
env = RobotEnv("urdf/quad.urdf", render=True)

# Reset the environment and get the initial observation
obs = env.reset()

# Initialize an empty NumPy array to store joint position data
joint_position_list = np.empty((8, 0))  # Assuming 8 joints in the robot

# Number of steps to run the simulation
num_steps = 10000

# Run the simulation loop for the specified number of steps
for _ in range(num_steps):
    # time.sleep(2)  # Uncomment this line to slow down the simulation for visualization

    # Use the trained model to predict the next action based on the current observation
    action, _states = model.predict(obs)

    # Apply the action to the environment and receive the next state, reward, and status
    obs, rewards, dones, info = env.step(action)

    # Retrieve the current joint positions from the environment
    joint_positions = env.render()

    # Reshape the joint positions to match the expected data format
    joint_positions_transposed = (np.array(joint_positions)).reshape((8, 1))

    # Append the new joint position data to the existing list
    joint_position_list = np.append(joint_position_list, joint_positions_transposed, axis=1)

# time_list = np.linspace(0, num_steps, num_steps)  # Generate a time array for plotting if needed

# Save the collected joint position data to a CSV file for further analysis
# np.savetxt('robot_evaluate.csv', joint_position_list, delimiter=',')
