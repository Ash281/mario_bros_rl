# Mario Bros RL: Training Agents with Different Playstyles

This repository contains code for training and testing reinforcement learning agents to play Super Mario Bros with different playstyles using PPO algorithms.

## Features

- Train agents with three distinct playstyles:
  - **Speedrunner**: Focuses on completing levels as quickly as possible
  - **Collector**: Prioritizes collecting coins and power-ups
  - **Enemy Killer**: Focuses on defeating enemies
- Custom reward functions for each playstyle
- Visualization tools for training metrics
- TensorBoard integration for monitoring training

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Ash281/mario_bros_rl.git
   cd mario_bros_rl
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate mario_rl
   ```

3. Install dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Training Agents

To train an agent with a specific playstyle, use the `PPO_train.py` script.

### Basic Usage

```bash
python PPO_train.py --playstyle <playstyle>
```

### Training Parameters

The script accepts several optional parameters:

- `--playstyle`: Choose between `speedrunner`, `collector`, or `enemy_killer`
- `--continue_from`: Specify path of model to continue training from, else train from the start

### Example Command

```bash
# Train a collector agent continuing from 300K timesteps
python PPO_train.py --playstyle collector --continue_from your_model_path/300000
```

Training progress will be logged to the console and to TensorBoard. You can change the number of timesteps and the hyperparameters in the script itself.

## Testing Trained Agents

To test a trained agent, use the `PPO_test.py` script. This will open a window with the agent playing the level. This must be run locally since there are no virtual displays available on Colab.

### Choosing an Agent to Test

Pre-trained models are available at:
https://drive.google.com/drive/folders/1L7OcZLMycptXQ4wYct2l7l5AaDikciv3?usp=drive_link

Update the file path in `PPO_test.py`:
```python
model = PPO.load(f'models/{playstyle}/PPO_{playstyle}_model_4800000', env=env)
```
Change `4800000` to the training steps of the desired model.

### Basic Usage

```bash
python PPO_test.py --playstyle speedrunner
```

### Testing Parameters

- `--playstyle`: The playstyle of the agent being tested

## Batch Testing Models

Use `test_models.py` to test the performance of models over multiple episodes:

```bash
python test_models.py
```

You can change the model path and the number of episodes within the script.

## Visualization Tools

### Training Metrics Visualization

Generate graphs of training metrics:

```bash
python visualise_metrics.py
```

This will generate and save graphs showing various metrics like reward, episode length, and loss values.

### TensorBoard Data Visualization

Process and display TensorBoard data:

```bash
python visualise_tensor.py collector
```

## Troubleshooting

### Common Issues

- **Missing dependencies**: Ensure all dependencies are installed through the `environment.yml` file
- **CUDA errors**: For GPU training, ensure CUDA drivers match the PyTorch version
- **Memory issues**: Reduce batch size if encountering out-of-memory errors

### Performance Tips

- Training on GPU is significantly faster than CPU
- For best results, train for at least 2 million timesteps
- Monitor the entropy loss to detect when training stabilizes
