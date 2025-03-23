import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from src.wrappers.wrappers import apply_wrappers

MODEL_FOLDER = "models/test_enemy"
NUM_EPISODES = 250
RESULTS_FILE = "testing_results_enemy.csv"  

# level checkpoints
CHECKPOINTS = {
    "Goomba 1": 350, "Pipe 1": 625, "Tall Pipe 1": 800, "Pit 1": 1150, 
    "Goombas 2": 1550, "Koopa": 1700, "Goombas 3": 1925, "Pit 2": 2600, 
    "Stairs": 2300, "Flagpole": 3000
}

results = []

# reset cv
if os.path.exists(RESULTS_FILE):
    os.remove(RESULTS_FILE)

# iterate through all models to test on
for model_file in sorted(os.listdir(MODEL_FOLDER)):
    if not model_file.endswith(".zip"):
        continue  

    model_path = os.path.join(MODEL_FOLDER, model_file)
    print(f"\n🎮 Testing model: {model_file}")
    model = PPO.load(model_path)

    # init env for testing
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0", render_mode='rgb_array', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = apply_wrappers(env, style="speedrunner")
    env.action_space.seed(42)
    env.observation_space.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # init metrics
    completion_rates = []
    x_distances = []
    coins_collected = []
    enemies_defeated = []
    frames_per_level = []
    checkpoint_reached = {key: [] for key in CHECKPOINTS.keys()}
    frames_to_25 = []
    frames_to_50 = []
    frames_to_75 = []

    for episode in range(NUM_EPISODES):
        obs = env.reset()
        if isinstance(obs, tuple):  
            obs = obs[0]
        if hasattr(obs, '__array__'):
            obs = obs.__array__()

        total_reward = 0
        frames = 0
        done = False
        max_x_pos = 0
        checkpoints_hit = set()
        previous_score = 0  
        enemy_kills = 0  

        frame_25, frame_50, frame_75 = None, None, None

        while not done and frames < 1000:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(int(action))

            if isinstance(obs, tuple):
                obs = obs[0]
            if hasattr(obs, '__array__'):
                obs = obs.__array__()
            if isinstance(info, tuple):
                info = info[0]

            total_reward += reward
            frames += 1
            current_x = info.get("x_pos", 0)
            max_x_pos = max(max_x_pos, current_x)

            # track ceckpoints
            for key, value in CHECKPOINTS.items():
                if current_x >= value and key not in checkpoints_hit:
                    checkpoints_hit.add(key)

            # track frames for 25/50/75 (assuming level is 3000px long)
            if current_x >= 750 and frame_25 is None:
                frame_25 = frames
            if current_x >= 1500 and frame_50 is None:
                frame_50 = frames
            if current_x >= 2250 and frame_75 is None:
                frame_75 = frames

            # track enemy kills (assuming 100 points per kill)
            current_score = info.get("score", 0)
            new_kills = max(0, (current_score - previous_score) // 100)
            enemy_kills += new_kills
            previous_score = current_score


        completion_rates.append(1 if info.get("flag_get", False) else 0)
        x_distances.append(max_x_pos)
        coins_collected.append(info.get("coins", 0))
        enemies_defeated.append(enemy_kills)
        frames_per_level.append(frames)
        frames_to_25.append(frame_25 or 1000)
        frames_to_50.append(frame_50 or 1000)
        frames_to_75.append(frame_75 or 1000)

        for key in CHECKPOINTS.keys():
            checkpoint_reached[key].append(1 if key in checkpoints_hit else 0)

    avg_completion = np.mean(completion_rates) * 100
    avg_x_distance = np.mean(x_distances)
    avg_coins = np.mean(coins_collected)
    avg_enemies = np.mean(enemies_defeated)
    avg_frames = np.mean(frames_per_level)
    avg_frames_25 = np.mean(frames_to_25)
    avg_frames_50 = np.mean(frames_to_50)
    avg_frames_75 = np.mean(frames_to_75)
    avg_checkpoints = {key: np.mean(values) * 100 for key, values in checkpoint_reached.items()}

    results.append({
        "Model": model_file, "Completion Rate (%)": avg_completion, "Avg X-Distance": avg_x_distance,
        "Avg Coins": avg_coins, "Avg Enemies": avg_enemies, "Frames Per Level": avg_frames, 
        "Frames to 25%": avg_frames_25, "Frames to 50%": avg_frames_50, "Frames to 75%": avg_frames_75, 
        **avg_checkpoints
    })

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")
