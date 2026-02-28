import numpy as np
import gymnasium as gym
from gymnasium.wrappers import GrayscaleObservation
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers import ResizeObservation

def create_env(render_mode, stack_size=4, resize_shape=(84, 84)):
    """Create and wrap the MsPacman environment with necessary preprocessing."""
    env = gym.make('ALE/MsPacman-v5', render_mode=render_mode)
    
    # Apply preprocessing wrappers
    grayscale_env = GrayscaleObservation(env)
    resized_env = ResizeObservation(grayscale_env, tuple(resize_shape))
    env = FrameStackObservation(resized_env, stack_size=stack_size)  # Stack frames
    
    return env

def create_envs(num_envs=2, render_mode='human', stack_size=4, resize_shape=(84, 84)):
    """Create multiple environments with the same preprocessing."""
    envs = gym.vector.AsyncVectorEnv([lambda: create_env(render_mode, stack_size, resize_shape) for _ in range(num_envs)])
    return envs

def get_state(env):
    """Get the current state from the environment."""
    state = np.array(env.get_observation())
    return state 