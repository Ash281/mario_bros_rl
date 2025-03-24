"""
Super Mario RL Testing Script

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

from stable_baselines3 import PPO
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from wrappers import apply_wrappers
import time
import argparse

parser = argparse.ArgumentParser(description='Test Mario RL agent with different playstyles')
parser.add_argument('playstyle', type=str, choices=['speedrunner', 'collector', 'enemy_killer'],
                    help='Which agent playstyle to test')
args = parser.parse_args()

playstyle = args.playstyle

ENV_NAME = 'SuperMarioBros-1-1-v0'

# environment wrappers
env = gym_super_mario_bros.make(
    ENV_NAME, render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, RIGHT_ONLY)

env = apply_wrappers(env, style=playstyle)
JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

model = PPO.load(f'models/{playstyle}/PPO_{playstyle}_model_4800000', env=env)
vec_env = model.get_env()
obs = vec_env.reset()
while True:
    # time.sleep(0.5)
    action, _ = model.predict(obs)
    action = [int(action)] 
    obs, reward, done, info = model.env.step(action)
    # print(f"X-Position: {info[0]['x_pos']}")  # log x position
    obs = obs.__array__()
    model.env.render()

    if done:
        obs = model.env.reset() 
