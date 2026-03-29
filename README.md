# SocialGuard-RL 🛡️ 
RL environment for social media integrity moderation — Meta OpenEnv Hackathon 2026. Gym-compliant agents for spam detection, misinformation, and bot network dismantling.

SocialGuard-RL is a gym-compatible reinforcement learning environment that simulates a live social media platform.
An AI agent plays the role of an automated integrity officer. It observes user behavior and network signals, takes moderation actions, and learns from consequences via a multi-objective reward function.

## Overview

Static classifiers treat every decision independently. Social media moderation is a sequential decision problem under uncertainty. This project implements three tasks as Gym environments:
- **Task 1 (Spam detection):** Feature-based decisions on account queues.
- **Task 2 (Viral Misinformation):** Temporal dynamics and timing bonus for content spread.
- **Task 3 (CIB Takdown):** Graph-level cluster detection and takedown via embeddings.

## Key Features

- **Multi-Objective Rewards:** `correctness`, `fp_cost`, `collateral_damage`, `speed_bonus`, `escalation_penalty`.
- **Config-Driven:** No magic numbers. Everything is YAML.
- **Gymnasium Compatible:** Fully integrates with `stable-baselines3`.
- **Visual Dashboard:** Streamlit UI for visualising the graph, network state, and rewards in real-time.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/vincenzo-afk/SocialGuard-RL.git
cd socialguard-rl

# Install requirements
pip install -r requirements.txt
```

### Running Tasks

```bash
# Evaluate the baseline agent for Task 1
python baseline.py --config configs/task1.yaml

# Train PPO on Task 1
python training/train_ppo.py --config configs/task1.yaml

# Evaluate trained model
python evaluate.py --model models/ppo_task1.zip --config configs/task1.yaml
```

### Dashboard

To run the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

### Docker

```bash
docker build -t socialguard-rl .
docker run -p 8501:8501 socialguard-rl
```

## Structure
- `env/`: Core GYM environment and observation spaces.
- `sim/`: Network and Bot behaviour simulators.
- `tasks/`: Logic for the 3 distinct RL moderation tasks.
- `dashboard/`: Streamlit interactive dashboard.
- `configs/`: Hyperparameters and tuning constraints.
- `graders/`: Metric generation.

## Testing
Run unit and integration tests using `pytest`:
```bash
python -m pytest tests/
```
