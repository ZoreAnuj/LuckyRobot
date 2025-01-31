import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class RobotEnv(gym.Env):
    """
    Custom Robot Environment that follows gym interface.
    This environment simulates a robot using PyBullet physics engine and provides
    an interface for reinforcement learning tasks.
    """
    def __init__(self, urdf_path, render=False, show_training=True):
        """
        Initialize the robot environment.
        
        Args:
            urdf_path (str): Path to the robot's URDF file
            render (bool): Whether to render the simulation
            show_training (bool): Whether to show the training process in GUI
        """
        super(RobotEnv, self).__init__()
        self.urdf_path = urdf_path

        # Physics simulation parameters
        self._num_bullet_solver_iterations = 300  # Number of iterations for physics solver
        self._is_render = render
        self._last_frame_time = 0.0
        self._time_step = 0.01  # 10ms timestep for simulation

        # Initialize PyBullet in either GUI or DIRECT mode
        if show_training == True or self.render==True:
            self.physics_client = p.connect(p.GUI)  # For visualization
        else:
            self.physics_client = p.connect(p.DIRECT)  # Headless mode for faster training

        # Camera configuration for visualization
        self.camera_distance = 1.0  
        self.camera_yaw = 85        
        self.camera_pitch = -35     
        self.camera_target_position = [0, 0, 0]  

        # Set up physics environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)  # Set standard gravity
        p.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
        p.setTimeStep(self._time_step)

        # Load the ground plane
        plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)  # Set ground friction

        # Load and position the robot
        basePosition = [0, 0, .2]  # Start slightly above ground
        baseOrientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])  # Initial orientation
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.robot_id = p.loadURDF(self.urdf_path,
                            basePosition=basePosition,
                            baseOrientation=baseOrientation,
                            flags=flags)
        
        # Initialize state tracking variables
        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0]
        self._last_joint_positions = [0, 0, 0, 0, 0, 0, 0, 0]
        self._last_joint_velocities = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # Get initial robot state
        self.base_position, self.base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_index_list = list(range(self.num_joints))

        # Set initial joint configuration (laying flat)
        initial_joint_angles = [np.pi/2, 0, -np.pi/2, 0, np.pi/2, 0, -np.pi/2, 0]
        for joint in range(self.num_joints):
            p.resetJointState(self.robot_id, joint, initial_joint_angles[joint])
            p.enableJointForceTorqueSensor(self.robot_id, joint, 1)
            print('JointInfo' + str(joint) + ": ", p.getJointInfo(self.robot_id, joint))
        
        # Define action and observation spaces
        print("ACTION SPACE SIZE: ", (self.num_joints,))
        print("OBSERVATION SPACE SIZE: ", (self.num_joints*2,))
        obs_dim = 2 * self.num_joints + 3 + 4 + 3 + 3  # joints pos/vel + base pos + orientation + linear/angular vel
        
        # Define joint angle limits in radians
        action_low = np.array([-75, 30, 0, -120, 0, 70, 0, -110]) * np.pi/180
        action_high = np.array([0, 120, 75, -30, 0, 110, 0, -70]) * np.pi/180

        self.action_space = spaces.Box(action_low, action_high, shape=(self.num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.current_step = 0  # Step counter for episode tracking

    def reset(self):
        """
        Reset the environment to initial state.
        Returns:
            observation (array): Initial observation of the environment
        """
        # Reset entire simulation
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setPhysicsEngineParameter(
          numSolverIterations=int(self._num_bullet_solver_iterations))
        p.setTimeStep(self._time_step)
        
        # Recreate ground plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")
        p.changeVisualShape(plane_id, -1, rgbaColor=[1, 1, 1, 0.9])
        p.changeDynamics(plane_id, -1, lateralFriction=1.0)

        # Reload robot in initial position
        basePosition = [0, 0, .2]
        baseOrientation = p.getQuaternionFromEuler([np.pi/2, 0, np.pi/2])
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_INCLUDE_PARENT
        self.robot_id = p.loadURDF(self.urdf_path,
                            basePosition=basePosition,
                            baseOrientation=baseOrientation,
                            flags=flags)
        
        # Reset state tracking variables
        self._last_base_position = [0, 0, 0]
        self._last_base_orientation = [0, 0, 0]
        self._last_joint_positions = [0, 0, 0, 0, 0, 0, 0, 0]
        self._last_joint_velocities = [0, 0, 0, 0, 0, 0, 0, 0]

        # Reset joint positions to initial configuration
        initial_joint_angles = [np.pi/2, 0, -np.pi/2, 0, np.pi/2, 0, -np.pi/2, 0]
        for joint in range(self.num_joints):
            p.resetJointState(self.robot_id, joint, initial_joint_angles[joint])
            p.enableJointForceTorqueSensor(self.robot_id, joint, 1)

        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Args:
            action (array): Joint angles for the robot
            
        Returns:
            observation (array): Current observation of environment
            reward (float): Reward for current action
            done (bool): Whether episode has ended
            info (dict): Additional information
        """
        # Handle rendering timing
        if self._is_render:
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            self._action_repeat = 1
            time_to_sleep = self._action_repeat * self._time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)

        # Update camera position
        self.camera_yaw += .001
        self.camera_target_position = self.base_position
        p.resetDebugVisualizerCamera(self.camera_distance, self.camera_yaw, self.camera_pitch, self.camera_target_position)

        # Set servo parameters
        max_servo_velocity = 8.055  # rad/s
        max_servo_force = 1.177     # Nm

        # Apply joint position control
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot_id, 
                                    i, 
                                    p.POSITION_CONTROL, 
                                    targetPosition=action[i],
                                    maxVelocity=max_servo_velocity, 
                                    force=max_servo_force)
        
        p.stepSimulation()
        
        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._is_done() or self._check_fall()

        self.current_step += 1
        
        return obs, reward, done, {}

    def _get_observation(self):
        """
        Get current observation of environment state.
        
        Returns:
            observation (array): Current state observation including joint positions,
                               velocities, base position, orientation, and velocities
        """
        # Get joint states
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        self.current_joint_positions = [state[0] for state in joint_states]
        self.current_joint_velocities = [state[1] for state in joint_states]

        # Get base pose and velocity
        self.base_position, self.base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        self.base_linear_velocity, self.base_angular_velocity = p.getBaseVelocity(self.robot_id)
    
        # Combine all observations into single array
        observation = np.concatenate([
            self.current_joint_positions,     # 8 values
            self.current_joint_velocities,    # 8 values
            self.base_position,               # 3 values
            self.base_orientation,            # 4 values
            self.base_linear_velocity,        # 3 values
            self.base_angular_velocity        # 3 values
        ])  # Total: 29 observed states

        return observation

    def _compute_reward(self):
        """
        Calculate reward based on multiple factors including forward progress,
        energy consumption, stability, and movement characteristics.
        
        Returns:
            reward (float): Calculated reward value
        """
        # Reward weights
        self._distance_weight = 1.5    # Forward movement
        self._energy_weight = 0.005    # Energy efficiency
        self._shake_weight = 0.0       # Vertical stability
        self._drift_weight = 0.0       # Lateral stability
        self._velocity_weight = .30    # Movement speed
        self._jitter_weight = 0.1      # Smooth motion

        # Get current velocities
        linear_velocity, angular_velocity = p.getBaseVelocity(self.robot_id)
        velocity_magnitude = np.linalg.norm(linear_velocity[:2])

        # Calculate position-based rewards
        self.current_base_position, self.current_base_orientation = p.getBasePositionAndOrientation(self.robot_id)
        forward_reward = self.current_base_position[0] - self._last_base_position[0]  # x-axis progress
        drift_reward = -abs(self.current_base_position[1] - self._last_base_position[1])  # y-axis stability
        shake_reward = -abs(self.current_base_position[2] - self._last_base_position[2])  # z-axis stability

        self._last_base_position = self.current_base_position

        # Calculate joint-based rewards
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        self.current_joint_positions = [state[0] for state in joint_states]
        self.current_joint_torques = [state[3] for state in joint_states]
        self.current_joint_velocities = [state[1] for state in joint_states]

        # Penalize jerky movements
        jitter_penalty = -sum(abs(np.array(self.current_joint_velocities) - np.array(self._last_joint_velocities)))

        self._last_joint_positions = self.current_joint_positions
        self._last_joint_velocities = self.current_joint_velocities
        
        # Calculate energy consumption
        energy_reward = abs(np.dot(self.current_joint_torques, self.current_joint_velocities)) * self._time_step

        # Additional penalties
        fall_weight = -1
        fall_penalty = fall_weight if self._check_fall() else 0
        angular_penalty = -np.linalg.norm(angular_velocity)

        # Velocity reward
        target_velocity = 0.1 
        epsilon = 0.000001  # Prevent division by zero
        velocity_reward = self._velocity_weight * (1 / (abs(target_velocity - velocity_magnitude) + epsilon))

        # Combine all reward components
        reward = (self._distance_weight * forward_reward 
                  - self._energy_weight * energy_reward 
                  + self._drift_weight * drift_reward 
                  + self._shake_weight * shake_reward
                  + fall_penalty)

        print("REWARD:", reward)
        return reward

    def _is_done(self):
        """
        Check if episode is complete.
        
        Returns:
            done (bool): Whether episode has ended
        """
        done = False
        return done
    
    def _check_fall(self):
        """
        Check if robot has fallen over based on orientation.
        
        Returns:
            fallen (bool): Whether robot has fallen
        """
        orientation = p.getBasePositionAndOrientation(self.robot_id)[1]
        roll, pitch, yaw = p.getEulerFromQuaternion(orientation)

        # Check if orientation exceeds safe limits
        if roll < -np.pi/2 or roll > (2/2)*np.pi or pitch < -(1/2)*np.pi or pitch > (1/2)*np.pi:
            return True
        return False

    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
            
        Returns:
            current_joint_positions (list): Current joint positions
        """
        return self.current_joint_positions 

    def close(self):
        """
        Clean up environment resources.
        """
        p.disconnect(self.physics_client)