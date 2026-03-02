# Flappy Bird — Deep Reinforcement Learning

An AI agent that learns to play Flappy Bird from scratch using deep reinforcement learning.  
Built with **Ray RLlib**, **PyTorch**, and **Gymnasium**.

## Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| **PPO** ★ | Policy Gradient | Best results — fastest learning, achieved infinite gameplay |
| **Rainbow DQN** | Value-Based | Second best — combines 6 DQN improvements (Double, Dueling, PER, Multi-step, C51, Noisy Nets) |
| **APPO** | Policy Gradient | Asynchronous PPO with V-trace correction |
| **IMPALA** | Policy Gradient | Scalable actor-learner architecture |
| **DQN** | Value-Based | Classic Deep Q-Network |
| **CQL** | Value-Based | Conservative Q-Learning |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train a single algorithm:**
```bash
python main.py                          # Train PPO (default)
python train_ppo.py                     # Train PPO
python train_rainbow.py                 # Train Rainbow DQN
python train_dqn.py                     # Train DQN
python train_impala.py                  # Train IMPALA
python train_appo.py                    # Train APPO
python train_cql.py                     # Train CQL
```

**Continue training from checkpoint:**
```bash
python continue_training.py PPO 1000    # Continue PPO for 1000 more iterations
python continue_training.py Rainbow 500 # Continue Rainbow DQN for 500 more
```

**Test a trained model:**
```bash
python load_and_play.py PPO 10          # Play 10 episodes with PPO model
python load_and_play.py Rainbow 5       # Play 5 episodes with Rainbow model
```

## Project Structure

```
flappy_bird/
├── config.py               # Hyperparameters for all algorithms
├── environment.py           # Flappy Bird environment with custom wrappers
├── main.py                  # Main training script
├── load_and_play.py         # Load and test trained models
├── continue_training.py     # Resume training from checkpoint
├── utils.py                 # Plotting, metrics export, comparison
├── train_*.py               # Per-algorithm training scripts
├── trainers/
│   ├── base_trainer.py      # Base class (training, testing, saving)
│   ├── ppo_trainer.py       # PPO configuration
│   ├── rainbow_dqn_trainer.py # Rainbow DQN configuration
│   ├── dqn_trainer.py       # DQN configuration
│   ├── appo_trainer.py      # APPO configuration
│   ├── impala_trainer.py    # IMPALA configuration
│   ├── cql_trainer.py       # CQL configuration
│   └── cql_dqn_policy.py   # Custom CQL loss for discrete actions
├── models/                  # Saved checkpoints (auto-saved every 100 iterations)
└── requirements.txt
```

## Environment Modifications

- **TopTouchPenaltyWrapper** — Penalty of −4 for touching the ceiling
- **MasteryBonusWrapper** — Bonus of +1000 for surviving 10,000 steps
- **TimeLimit** — Cap of 10,000 steps per episode to prevent infinite episodes during training

## Hardware

Trained on NVIDIA GeForce GTX 1650 with CUDA, cuDNN, and TF32 optimizations.

