# Deep Q-Learning for MsPacman

This project implements a Deep Q-Learning agent to play the MsPacman-v0 Atari game using PyTorch and OpenAI Gym.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `dqn_agent.py`: Implementation of the DQN agent
- `preprocessing.py`: Frame preprocessing utilities
- `train.py`: Training script
- `utils.py`: Helper functions and utilities

## Usage

To train the agent:
```bash
python train.py
```

The trained model will be saved in the `models` directory. 

## MsPacman Reward System

| Action                        | Points (Reward) |
|------------------------------|-----------------|
| Eating a small pellet        | 10              |
| Eating a power pellet        | 50              |
| Eating a fruit               | 100–5000        |
| Eating 1st ghost (after power pellet) | 200     |
| Eating 2nd ghost             | 400             |
| Eating 3rd ghost             | 800             |
| Eating 4th ghost             | 1600            |
| Clearing a level             | ~5000           |
| Dying (losing a life)        | 0 (manual penalty recommended) |
| Time passing (doing nothing)| 0               |
