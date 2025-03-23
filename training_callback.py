import pandas as pd
import json
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    custom callback for logging training metrics and saving model checkpoints
    tracks rewards, completion rate, obstacles, progression, and average x-distance in Super Mario Bros
    """
    def __init__(self, check_freq, save_path, metrics_file, style, initial_step=0, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.metrics_file = metrics_file 
        self.step_count = initial_step
        self.style = style

        # init metrcs
        self.episode_rewards = []
        self.completion_rates = []
        self.frames_per_level = []
        self.coins_collected = []
        self.enemies_defeated = []
        self.x_positions = [] 
        self.obstacle_checkpoints = []

        # file paths for saving logs
        self.csv_file = os.path.join(self.save_path, self.metrics_file)
        self.json_file = os.path.join(self.save_path, "obstacle_checkpoints.json")

    def _init_callback(self):
        """initialize directories and log files"""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        if not os.path.exists(self.csv_file):
            pd.DataFrame(columns=[
                "Step", "Avg_Reward", "Completion_Rate", "Frames_Per_Level", "Coins_Collected", 
                "Enemies_Defeated", "Avg_X_Distance",  
                "First_Pipe", "Second_Pipe", "Third_Pipe", "Fourth_Pipe",
                "First_Goomba", "Second_Goomba",
                "First_Pit", "Second_Pit",
                "Final_Stairs", "Flagpole"
            ]).to_csv(self.csv_file, index=False)

    def _on_step(self):
        """collect metrics during training and save logs every check_freq steps"""
        self.step_count += 1

        # extract rewards and environment info
        reward = self.locals["rewards"]
        info = self.locals["infos"][0]  

        self.episode_rewards.append(reward)
        self.completion_rates.append(1 if info.get("flag_get", False) else 0)
        self.frames_per_level.append(info.get("x_pos", 0))
        self.coins_collected.append(info.get("coins", 0))
        self.enemies_defeated.append(info.get("enemy_kills", 0))
        self.x_positions.append(info.get("x_pos", 0))  # Track X-Distance

        # track checkpoints
        obstacle_data = {
            "Goomba 1": bool(info.get("x_pos", 0) >= 350),
            "Pipe 1": bool(info.get("x_pos", 0) >= 625),
            "Tall Pipe 1": bool(info.get("x_pos", 0) >= 740),
            "Pit 1": bool(info.get("x_pos", 0) >= 1180),
            "Pipe 2": bool(info.get("x_pos", 0) >= 1425),
            "Pit 2": bool(info.get("x_pos", 0) >= 1680),
            "Goombas 2": bool(info.get("x_pos", 0) >= 1800),
            "Koopa": bool(info.get("x_pos", 0) >= 1950),
            "Goombas 3": bool(info.get("x_pos", 0) >= 2100),
            "Goombas 4": bool(info.get("x_pos", 0) >= 2300),
            "Stairs 1": bool(info.get("x_pos", 0) >= 2450),
            "Pit 3": bool(info.get("x_pos", 0) >= 2650),
            "Final Stairs": bool(info.get("x_pos", 0) >= 2800),
            "Flagpole": bool(info.get("x_pos", 0) >= 3072),
        }
        self.obstacle_checkpoints.append(obstacle_data)

        # log & save metrics
        if self.step_count % self.check_freq == 0:
            avg_reward = np.mean(self.episode_rewards[-self.check_freq:])
            completion_rate = np.mean(self.completion_rates[-self.check_freq:]) * 100
            avg_frames = np.mean(self.frames_per_level[-self.check_freq:])
            avg_coins = np.mean(self.coins_collected[-self.check_freq:])
            avg_enemies = np.mean(self.enemies_defeated[-self.check_freq:])
            avg_x_distance = np.mean(self.x_positions[-self.check_freq:]) 
            obstacle_rates = {key: np.mean([1 if x[key] else 0 for x in self.obstacle_checkpoints[-self.check_freq:]]) * 100 for key in obstacle_data}

            print(f"🔹 Step {self.step_count}:")
            print(f"   Avg Reward: {avg_reward:.2f}")
            print(f"   Completion Rate: {completion_rate:.2f}%")
            print(f"   Frames Per Level: {avg_frames:.2f}")
            print(f"   Avg Coins Collected: {avg_coins:.2f}")
            print(f"   Avg Enemies Defeated: {avg_enemies:.2f}")
            print(f"   Avg X-Distance Traveled: {avg_x_distance:.2f} px")

            print(f"   👾 Goomba 1 Reached: {obstacle_rates['Goomba 1']:.2f}%")
            print(f"   🏗️ Pipe 1 Reached: {obstacle_rates['Pipe 1']:.2f}%")
            print(f"   🏗️ Tall Pipe 1 Reached: {obstacle_rates['Tall Pipe 1']:.2f}%")
            print(f"   ⚠️ Pit 1 Cleared: {obstacle_rates['Pit 1']:.2f}%")
            print(f"   🏗️ Pipe 2 Reached: {obstacle_rates['Pipe 2']:.2f}%")
            print(f"   ⚠️ Pit 2 Cleared: {obstacle_rates['Pit 2']:.2f}%")
            print(f"   👾 Goombas 2 Reached: {obstacle_rates['Goombas 2']:.2f}%")
            print(f"   🐢 Koopa Reached: {obstacle_rates['Koopa']:.2f}%")
            print(f"   👾 Goombas 3 Reached: {obstacle_rates['Goombas 3']:.2f}%")
            print(f"   👾 Goombas 4 Reached: {obstacle_rates['Goombas 4']:.2f}%")
            print(f"   🏗️ Stairs 1 Reached: {obstacle_rates['Stairs 1']:.2f}%")
            print(f"   ⚠️ Pit 3 Cleared: {obstacle_rates['Pit 3']:.2f}%")
            print(f"   🏁 Final Stairs Reached: {obstacle_rates['Final Stairs']:.2f}%")
            print(f"   🎉 Flagpole Reached: {obstacle_rates['Flagpole']:.2f}%")

            # append to csv
            df = pd.DataFrame([[self.step_count, avg_reward, completion_rate, avg_frames, avg_coins, avg_enemies, avg_x_distance] + list(obstacle_rates.values())], 
                              columns=["Step", "Avg_Reward", "Completion_Rate", "Frames_Per_Level", "Coins_Collected", 
                                       "Enemies_Defeated", "Avg_X_Distance"] + list(obstacle_data.keys()))
            df.to_csv(self.csv_file, mode='a', header=False, index=False)

            # save to json
            with open(self.json_file, "w") as f:
                json.dump(self.obstacle_checkpoints, f, indent=4)

            # save model checkpoint
            model_path = os.path.join(self.save_path, f'PPO_{self.style}_model_{self.step_count}')
            self.model.save(model_path)

        return True  # continue
