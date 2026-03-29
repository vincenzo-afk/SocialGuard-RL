"""
tests/test_training.py — Smoke tests for Phase 5 training scripts.

Ensures that PPO and DQN training entry points can initialize the environment,
build the SB3 model, connect the callbacks, and run at least one optimization
step without crashing.
"""

from __future__ import annotations

import os
import sys
import tempfile
import yaml
import pytest
from unittest.mock import patch

from training.train_ppo import main as ppo_main
from training.train_dqn import main as dqn_main


@pytest.fixture
def tiny_config_path() -> str:
    """Creates a temporary YAML config with minimal steps for fast testing."""
    cfg = {
        "env": {"max_steps": 10, "seed": 42},
        "task": {"name": "task_spam", "bot_ratio": 0.5, "action_space": [0, 1, 2, 3]},
        "reward": {
            "alpha": 1.0, "beta": 1.0, "gamma": 1.0, "delta": 1.0, "epsilon": 0.1,
            "speed_max_hops": 20
        },
        "training": {
            "total_timesteps": 64,   # Very small
            "n_steps": 32,           # For PPO
            "batch_size": 16,
            "n_epochs": 1,
            "eval_freq": 32,
            "n_eval_episodes": 1,
        },
        "training_dqn": {
            "total_timesteps": 64,
            "learning_starts": 10,   # Start learning quickly
            "buffer_size": 100,
            "batch_size": 16,
            "target_update_interval": 10,
            "eval_freq": 32,
            "n_eval_episodes": 1,
        }
    }
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    
    yield path
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


def test_train_ppo_smoke(tiny_config_path: str) -> None:
    """Verify PPO can initialize and run a tiny training loop without crashing."""
    run_name = "test_ppo"
    out_dir = tempfile.mkdtemp()
    test_args = [
        sys.executable, "-m", "training.train_ppo",
        "--config", tiny_config_path,
        "--run_name", run_name,
        "--output_dir", out_dir,
        "--device", "cpu"
    ]
    import subprocess
    result = subprocess.run(test_args, timeout=120)
    assert result.returncode == 0, "PPO training crashed"
    
    # Verify outputs were created
    assert os.path.exists(os.path.join(out_dir, run_name, "final_model.zip"))


def test_train_dqn_smoke(tiny_config_path: str) -> None:
    """Verify DQN can initialize and run a tiny training loop without crashing."""
    run_name = "test_dqn"
    out_dir = tempfile.mkdtemp()
    test_args = [
        sys.executable, "-m", "training.train_dqn",
        "--config", tiny_config_path,
        "--run_name", run_name,
        "--output_dir", out_dir,
        "--device", "cpu"
    ]
    import subprocess
    result = subprocess.run(test_args, timeout=120)
    assert result.returncode == 0, "DQN training crashed"
    
    # Verify outputs were created
    assert os.path.exists(os.path.join(out_dir, run_name, "final_model.zip"))
