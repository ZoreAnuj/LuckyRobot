from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from pybullet_envs.robot_env import RobotEnv

# Create the vectorized environment with a single instance of the robot simulation
# This helps in parallel execution and efficient training
env = make_vec_env(lambda: RobotEnv("urdf/quad.urdf", show_training=True), n_envs=1)

# Initialize the PPO (Proximal Policy Optimization) model with:
# - 'MlpPolicy' (Multi-Layer Perceptron) as the neural network policy
# - The created environment
# - Verbosity set to 1 (prints training progress)
# - Learning rate of 0.0005 for stable training
# - n_steps = 4096 (buffer size before updating the policy)
# - Batch size of 256 (used for training updates)
model = PPO('MlpPolicy', env, verbose=1, learning_rate=5e-4, n_steps=4096, batch_size=256)

# Train the model for 1,000,000 time steps
model.learn(total_timesteps=1000000)

# Save the trained model to a file for later use
model.save("ppo_robot")

# Load the trained model (useful for evaluation or further training)
model = PPO.load("ppo_robot")

# Close the environment to free resources
env.close()
