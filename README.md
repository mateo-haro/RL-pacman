# Deep Q-Learning for MsPacman

This project implements a Double DQN agent with soft target updates to play MsPacman using PyTorch and Gymnasium. It is inspired by the work of Minh et al. in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), which introduced DQN agents for Atari games.

Since that original paper, several DQN improvements have been proposed. This repository incorporates a set of those enhancements on top of the baseline DQN setup, including:

- Dueling network head (separate value and advantage streams)
- Smooth target updates
- Modern CNN architecture (deeper network with smaller kernels)
- Imitation learning bootstrapping
- Double DQN updates
- Prioritized experience replay (PER)

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `main.py`: Entry point — training loop, evaluation, and demo recording
- `agent.py`: DQN agent with replay buffer
- `model.py`: Neural network architecture
- `preprocessing.py`: Environment wrappers (grayscale, resize, frame stacking)
- `hyperparameters.yaml`: All tunable parameters

## Usage

### Train

```bash
python main.py train
```

To pre-load human demonstrations into the replay buffer:

```bash
python main.py train --demos demos/human_demos.npz
```

### Evaluate a trained agent

```bash
python main.py eval
python main.py eval --model models/dqn_agent_500.pth
```

Defaults to `models/final_model.pth`. Always renders with a human-visible window.

### Record human demos

Play MsPacman yourself and record transitions for demonstration-based learning:

```bash
python main.py record
python main.py record --save-path demos/session2.npz
```

Control Pac-Man with the **arrow keys** (diagonals work by holding two arrows). Press **Q** to stop and save. The terminal shows a live counter of collected transitions:

```
Transitions: 342  Episode: 2  Score: 180
```

Demos are saved as compressed `.npz` files and can be loaded into training with `--demos`. The current repo includes a `human_demos.npz` file that contains 3795 trainsitions.


## MsPacman Reward System

The official scoring system doesn't implement stay alive score or dying penalty. That's why this are implemented as training feature that can be tuned in the hyperparameters. Furthermore, the rest of the scoring points can also be scaled for training and tuned in the hyperparameters.


| Action                                  | Points  |
|-----------------------------------------|---------|
| Eating a small pellet                   | 10      |
| Eating a power pellet                   | 50      |
| Eating a fruit                          | 100–5000|
| Eating 1st ghost (after power pellet)   | 200     |
| Eating 2nd ghost                        | 400     |
| Eating 3rd ghost                        | 800     |
| Eating 4th ghost                        | 1600    |
| Clearing a level                        | ~5000   |
| Dying (losing a life)                   | 0 (manual penalty in training) |
| Time passing (doing nothing)            | 0       |
