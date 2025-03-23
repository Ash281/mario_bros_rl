"""
This file contains the custom reward functions for the different playstyles in the game.

Author: Ashvin Valentine
Date: 01/03/2025
"""

from gym import Wrapper

class RewardWrapper(Wrapper):
    def __init__(self, env, style='speedrunner'):
        """speedrunner playstyle (default rewards)"""
        super().__init__(env)
        self.style = style
        self.prev_coins = None
        self.prev_score = None
        self.prev_status = None
        self.prev_x_pos = None
        self.prev_kills = None
        self.combo_counter = 0  # track consecutive kills

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward = self.custom_reward(reward, info)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.prev_coins = None
        self.prev_score = None
        self.prev_status = None
        self.prev_x_pos = None
        self.prev_kills = None
        self.combo_counter = 0
        return self.env.reset(**kwargs)

    def custom_reward(self, reward, info):
        """calls the correct reward function based on the playstyle"""
        if self.style == 'speedrunner':
            return reward
        elif self.style == 'collector':
            return self.collector_reward(reward, info)
        elif self.style == 'enemy_killer':
            return self.enemy_killer_reward(reward, info)

    def collector_reward(self, reward, info):
        """custom reward function for the collector playstyle"""
        new_reward = 0
        
        # reward for collecting coins increased
        if self.prev_coins is not None:
            coins_gained = info["coins"] - self.prev_coins
            if coins_gained > 0:
                new_reward += coins_gained * 3  # increased reward per coin (from 1)
        
        # reward for collecting powerups (mushrooms, fire flowers) increased (from 1)
        if self.prev_status is not None:
            if info["status"] > self.prev_status:  
                new_reward += 5 
        
        # encourage forward movement to prevent stalling but less than coin rewards (added in)
        if self.prev_x_pos is not None:
            if info["x_pos"] > self.prev_x_pos:
                new_reward += 0.1
        
        # penalise standing still or going backward (encourage forward progression instead)
        if self.prev_x_pos is not None:
            if info["x_pos"] < self.prev_x_pos:
                new_reward -= 0.5
        
        # store previous values for next step
        self.prev_coins = info["coins"]
        self.prev_status = info["status"]
        self.prev_x_pos = info["x_pos"]

        return new_reward

    def enemy_killer_reward(self, reward, info):
      """custom reward function for the enemy killer playstyle"""
      new_reward = 0

      # initialize previous stats
      if self.prev_score is None:
          self.prev_score = info.get("score", 0)
      if self.prev_x_pos is None:
          self.prev_x_pos = info.get("x_pos", 0)

      # reward for defeating enemies (score-based tracking)
      # 100 points = one kill
      current_score = info.get("score", 0)
      new_kills = max(0, (current_score - self.prev_score) // 100)

      if new_kills > 0:
          base_kill_reward = 20  # decrease from 50 to 20
          chain_multiplier = 1.5 ** (self.combo_counter)  # stacking bonus (1.5x per kill)
          kill_bonus = base_kill_reward * chain_multiplier
          new_reward += kill_bonus
          self.combo_counter += new_kills  # track consecutive kills (new)

      # large bonus if the agent gets 3+ kills quickly in a row (new)
      if self.combo_counter >= 3:
          new_reward += 50 

      # stronger shell kill bonus
      if info.get("shell_kills", 0) > 0:
          new_reward += 100

      # reduce movement reward (or) give it only near enemies)
      x_pos = info.get("x_pos", 0)
      if x_pos > self.prev_x_pos:
          nearby_enemy_bonus = 1 if info.get("nearby_enemy", False) else 0 
          new_reward += (x_pos - self.prev_x_pos) * 0.05 * nearby_enemy_bonus  

      # reward Mario for getting closer to an enemy
      if info.get("nearest_enemy_x", None) is not None:
          dist_to_enemy = abs(info["nearest_enemy_x"] - x_pos)
          if dist_to_enemy < 200:
              new_reward += 5

      # time penalty to stop enemy killer skipping enemies
      new_reward -= 0.2  

      # update previous values
      self.prev_score = current_score
      self.prev_x_pos = x_pos

      # reset combo if no kills happen for 1 step
      if new_kills == 0:
          self.combo_counter = 0

      return new_reward
