"""
This script visualizes the performance metrics of the models trained on the Super Mario Bros. environment.

Author: Ashvin Valentine
Date: 01/03/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# load csv
RESULTS_FILE = "testing_results_collector.csv"  # Change as needed

try:
    df = pd.read_csv(RESULTS_FILE)
except FileNotFoundError:
    print(f"{RESULTS_FILE} not found.")
    exit()

# clean up column names
df.columns = df.columns.str.strip()

# extract model number from model name and sort the dataframe by model number
df['Model Number'] = df['Model'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('Model Number')

# convert model number to steps in millions
df['Steps (M)'] = df['Model Number'] / 1e6

# define checkpoints
checkpoints = [
    "Goomba 1", "Pipe 1", "Tall Pipe 1", "Pit 1",
    "Goombas 2", "Koopa", "Goombas 3", "Stairs", "Pit 2", "Flagpole"
]

# convert checkpoint columns to numeric
df[checkpoints] = df[checkpoints].apply(pd.to_numeric, errors='coerce')

# define performance metrics columns
metrics_cols = [
    "Model", "Avg X-Distance", "Avg Coins", "Avg Enemies",
    "Frames Per Level", "Frames to 25%", "Frames to 50%", "Frames to 75%", "Completion Rate (%)"
]

# print model performance metrics
print("\nModel Performance Metrics:")
metrics_table = df[metrics_cols].set_index("Model")
print(metrics_table)

# every 400k steps for speedrunner (till 4.8M), every 300k steps for collector and enemy killer (till 3.6M)

if "speedrunner" in RESULTS_FILE:
    xtick_interval = 0.4
    tick_min = 0.4
    tick_max = 4.8
    expected_steps = np.arange(tick_min, tick_max + 0.001, xtick_interval)
elif "collector" in RESULTS_FILE or "enemy" in RESULTS_FILE:
    xtick_interval = 0.3
    tick_min = 0.3
    tick_max = 2.4
    expected_steps = np.arange(tick_min, tick_max + 0.001, xtick_interval)
else:
    expected_steps = df["Steps (M)"].values

# 1. Checkpoint Heatmap
plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    df[checkpoints].T,
    cmap="YlGnBu",
    annot=True,
    fmt=".1f",
    xticklabels=False,  
    yticklabels=checkpoints
)

plt.xlabel("Training Steps (M)")
plt.title("Checkpoint Completion Rates Across Models (%)")

if "speedrunner" in RESULTS_FILE:
    xtick_interval = 0.4
    tick_min = 0.4
    tick_max = 4.8
elif "collector" in RESULTS_FILE or "enemy" in RESULTS_FILE:
    xtick_interval = 0.3
    tick_min = 0.3
    tick_max = 2.4
else:
    xtick_interval = 0.4
    tick_min = df["Steps (M)"].min()
    tick_max = df["Steps (M)"].max()

desired_ticks = np.arange(tick_min, tick_max + xtick_interval, xtick_interval)
tick_positions = []
tick_labels = []
for dt in desired_ticks:
    closest_idx = (df["Steps (M)"] - dt).abs().idxmin()
    pos = df.index.get_loc(closest_idx)
    tick_positions.append(pos + 0.5)
    tick_labels.append(f"{dt:.1f}")

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45)
plt.tight_layout()
plt.savefig("checkpoint_heatmap.png")
plt.show()

# 2. Performance Metrics Bar Charts (using equal increments)
plt.figure(figsize=(16, 10))

metrics_to_plot = [
    ("Avg X-Distance", "skyblue", "Pixels"),
    ("Avg Coins", "gold", "Coins"),
    ("Avg Enemies", "salmon", "Enemies Defeated"),
    ("Frames Per Level", "lightgreen", "Frames"),
    ("Frames to 25%", "gray", "Frames"),
    ("Frames to 50%", "orange", "Frames")
]

if len(expected_steps) != len(df):
    print("expected steps array length does not match the number of data points")
    
for i, (metric, color, ylabel) in enumerate(metrics_to_plot, 1):
    plt.subplot(2, 3, i)
    x_values = expected_steps  
    y_values = df[metric].values
    plt.bar(x_values, y_values, color=color, width=0.15, align='center')
    plt.title(metric)
    plt.xlabel("Training Steps (M)")
    plt.ylabel(ylabel)
    plt.xticks(expected_steps, [f"{x:.1f}" for x in expected_steps], rotation=45)
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("performance_metrics.png")
plt.show()

# 3. Enemy Kill Progression
plt.figure(figsize=(10, 6))
plt.plot(
    df["Steps (M)"],
    df["Avg Enemies"],
    marker='o',
    linestyle='-',
    color='red',
    label="Average Enemies Defeated"
)
plt.xlabel('Training Steps (M)')
plt.ylabel('Enemies Defeated')
plt.title('Progression of Enemy Kills Over Training')
plt.legend()
plt.grid(alpha=0.3)
plt.xticks(expected_steps, [f"{x:.1f}" for x in expected_steps], rotation=45)
plt.savefig("enemy_progression.png")
plt.show()

# 4. Normalized Performance Metrics Progression
plt.figure(figsize=(12, 8))
metrics = ["Avg X-Distance", "Avg Coins", "Avg Enemies"]

normalized_df = df.copy()
for col in metrics:
    max_val = df[col].max()
    normalized_df[col] = df[col] / max_val if max_val > 0 else 0

plt.plot(
    normalized_df["Steps (M)"],
    normalized_df["Avg X-Distance"],
    'o-',
    linewidth=2,
    markersize=8,
    label='X-Distance'
)
plt.plot(
    normalized_df["Steps (M)"],
    normalized_df["Avg Coins"],
    's-',
    linewidth=2,
    markersize=8,
    label='Coins'
)
plt.plot(
    normalized_df["Steps (M)"],
    normalized_df["Avg Enemies"],
    '^-',
    linewidth=2,
    markersize=8,
    label='Enemies'
)

plt.xlabel('Training Steps (M)')
plt.ylabel('Normalized Performance')
plt.title('Performance Metrics Progression During Training')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(expected_steps, [f"{x:.1f}" for x in expected_steps], rotation=45)
plt.savefig("training_progression.png")
plt.show()

# 5. Additional Performance Metrics: Frames and Level
plt.figure(figsize=(12, 8))
metrics_to_plot2 = [
    ("Frames Per Level", "lightgreen", "Frames"),
    ("Frames to 25%", "gray", "Frames"),
    ("Frames to 50%", "orange", "Frames"),
    ("Frames to 75%", "purple", "Frames")
]

for i, (metric, color, ylabel) in enumerate(metrics_to_plot2, 1):
    plt.subplot(2, 2, i)
    x_values = expected_steps  
    y_values = df[metric].values
    plt.bar(x_values, y_values, color=color, width=0.15, align='center')
    plt.title(metric)
    plt.xlabel("Training Steps (M)")
    plt.ylabel(ylabel)
    plt.xticks(expected_steps, [f"{x:.1f}" for x in expected_steps], rotation=45)
    plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("frames_and_level_metrics.png")
plt.show()

# 6. Coins Collected Metric
plt.figure(figsize=(10, 6))
plt.bar(expected_steps, df["Avg Coins"], color="gold", width=0.15, align="center")
plt.title("Average Coins Collected Over Training")
plt.xlabel("Training Steps (M)")
plt.ylabel("Avg Coins")
plt.xticks(expected_steps, [f"{x:.1f}" for x in expected_steps], rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("avg_coins.png")
plt.show()
