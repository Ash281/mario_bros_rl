"""
This script generates a visualisation of the training metrics from a Tensorboard log file.

Author: Ashvin Valentine
Date: 01/03/2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# check args
if len(sys.argv) != 2:
    print("Usage: python script.py playstyle")
    print("   where playstyle is one of: speedrunner, collector, enemy_killer")
    sys.exit(1)

# get playstyle
playstyle = sys.argv[1]

plt.rcParams.update({
    'figure.facecolor': '#2D2D2D',
    'axes.facecolor': '#2D2D2D',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': 'gray',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11
})

# Load data based on playstyle
if playstyle == 'speedrunner':
    data_4m = pd.read_csv('tensorboard_speedrunner_4M.csv')
    data_5m = pd.read_csv('tensorboard_speedrunner_5M.csv')
    data_5m_filtered = data_5m[data_5m['step'] > 4000000]
    combined_data = pd.concat([data_4m, data_5m_filtered])
    threshold = 4000000 
else:
    file_path = f'tensorboard_{playstyle}.csv'
    combined_data = pd.read_csv(file_path)
    threshold = None  

combined_data = combined_data.sort_values('step')

# define metrics
metrics = [
    'rollout/ep_rew_mean',     # Episode reward mean
    'train/entropy_loss',      # Entropy loss
    'train/explained_variance', # Explained variance
    'rollout/ep_len_mean',     # Episode length
    'train/policy_gradient_loss', # Policy gradient
    'train/approx_kl'          # KL divergence
]

# define line colours
colors = ['#00FFFF', '#FF00FF', '#FFFF00', '#00FF00', '#FF8800', '#FF0000']

# create figure
fig = plt.figure(figsize=(15, 12), facecolor='#2D2D2D')

grid_rows, grid_cols = 2, 3

for i, metric in enumerate(metrics):
    ax = fig.add_subplot(grid_rows, grid_cols, i + 1)
    
    # Apply smoothing to reduce noise
    window_size = 50 
    smoothed = combined_data[metric].rolling(window=window_size, center=True, min_periods=1).mean()
    ax.plot(combined_data['step'], smoothed, 
            color=colors[i], linewidth=1.5, alpha=1.0)
    if threshold is not None:
        ax.axvline(x=threshold, color='white', linestyle='--', alpha=0.3)
    
    # Set title and labels
    title_text = metric.split('/')[-1].replace('_', ' ').title()
    ax.set_title(title_text, color='white', fontsize=10)
    ax.set_xlabel('Steps', color='white', fontsize=8)
    ax.set_ylabel(metric.split('/')[-1], color='white', fontsize=8)
    ax.tick_params(colors='white', labelsize=7)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(6,6))
    ax.grid(True, alpha=0.2)

playstyle_display = playstyle.replace('_', ' ').title()
plt.suptitle(f'Tensorboard Metrics for the {playstyle_display} Agent', color='white', fontsize=16, y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig(f'{playstyle}_metrics.png', dpi=300, bbox_inches='tight')
plt.show()
