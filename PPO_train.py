"""
Super Mario RL Training Script

This script is adapted from Nicholas Renotte's base implementation of Proximal Policy Optimization (PPO)
for Super Mario Bros. RL training.

Original Source:
- GitHub: https://github.com/nicknochnack/MarioRL
- YouTube Tutorial: https://www.youtube.com/watch?v=2eeYqJ0uBKE

Modifications & Enhancements in this Version:
-Adjusted environment setup and wrappers
-Updated training logic, hyperparameters, and logging
-Integrated model checkpointing and testing structure
-Added custom reward shaping
-Adjusted compatibility with updated OpenAI Gym and SB3

Author: Ashvin Valentine
Date: 01/03/2025
"""

import argparse
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
import os
import numpy as np
from training_callback import TrainAndLoggingCallback
from wrappers import apply_wrappers

# Parse command line arguments - playstyle and continue training option
parser = argparse.ArgumentParser(description='Train Mario with different playstyles')
parser.add_argument('playstyle', type=str, choices=['speedrunner', 'collector', 'enemy_killer'],
                    help='Playstyle to train')
parser.add_argument('--continue_from', type=str, default=None,
                    help='Path to existing model to continue training from')
args = parser.parse_args()

ENV_NAME = 'SuperMarioBros-1-1-v0'

# Setup directories
CHECKPOINT_DIR = f'/content/drive/MyDrive/mario_rl_v2/train/{args.playstyle}/'
LOG_DIR = f'/content/drive/MyDrive/mario_rl_v2/logs/{args.playstyle}/'
METRICS_FILE = f'/content/drive/MyDrive/mario_rl_v2/metrics/{args.playstyle}_metrics.csv'

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)

# Initialize metrics file
with open(METRICS_FILE, 'w') as f:
    f.write("step,avg_reward,completion_rate,frames_per_level,coins,enemies\n")

initial_step = 0
if args.continue_from:
    try:
        # Extract number from filename like PPO_speedrunner_model_800000
        initial_step = int(args.continue_from.split('_')[-1])
        print(f"Continuing from step {initial_step}")
    except:
        print("Could not extract step count from filename, starting from 0")

# Setup callback
callback = TrainAndLoggingCallback(
    check_freq=50000,
    save_path=CHECKPOINT_DIR,
    metrics_file=METRICS_FILE,
    style=args.playstyle,
    initial_step=initial_step,
    verbose=1
)

# Environment setup
env = gym_super_mario_bros.make(
    ENV_NAME, render_mode='rgb_array', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)
env = apply_wrappers(env, style=args.playstyle)  # Use collector reward wrapper
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Load existing model or create new one
if args.continue_from:
    print(f"Loading existing model from {args.continue_from}")
    model = PPO.load(args.continue_from, env=env)
    
    # The key step: don't reset training progress counter
    total_timesteps = 1000000
    print(f"Continuing training {args.playstyle} agent for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False)
else:
    # Create new model
    model = PPO('CnnPolicy', env, verbose=1,
                tensorboard_log=LOG_DIR, learning_rate=0.00001, n_steps=512,
                batch_size=32, n_epochs=10)
                
    total_timesteps = 1000000
    print(f"Training new {args.playstyle} agent for {total_timesteps} timesteps")
    model.learn(total_timesteps=total_timesteps, callback=callback)

print(f"Training complete for {args.playstyle} agent")