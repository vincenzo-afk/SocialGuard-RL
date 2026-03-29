"""
training/train_ppo.py — PPO training entry point for SocialGuard-RL.

Usage:
  python -m training.train_ppo --config configs/default.yaml --run_name ppo_task1
"""

from __future__ import annotations

import argparse
import logging
import os
import yaml
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from env.env import SocialGuardEnv
from training.callbacks import TensorboardCallback, create_eval_callback

logger = logging.getLogger(__name__)


def load_config(default_path: str, task_path: str | None = None) -> dict[str, Any]:
    """Load default config and optionally merge task-specific overrides."""
    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if task_path and os.path.exists(task_path):
        with open(task_path, "r", encoding="utf-8") as f:
            task_cfg = yaml.safe_load(f)
            # Merge task_cfg into base (shallow merge of top-level dicts)
            for k, v in task_cfg.items():
                if isinstance(v, dict) and k in cfg:
                    cfg[k].update(v)
                else:
                    cfg[k] = v
    return cfg


def main() -> None:
    """Run PPO training pipeline."""
    parser = argparse.ArgumentParser(description="Train PPO on SocialGuard-RL")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="ppo_run")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    tb_log_dir = os.path.join(args.output_dir, "tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)

    # 1. Load configuration
    cfg = load_config(args.config, args.task_config)
    merged_config_path = os.path.join(args.output_dir, f"{args.run_name}_config.yaml")
    with open(merged_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
        
    train_cfg = cfg.get("training", {})
    env_cfg = cfg.get("env", {})
    
    # Extract standard hyperparams (no magic numbers!)
    total_timesteps = int(train_cfg.get("total_timesteps", 100000))
    n_steps = int(train_cfg.get("n_steps", 2048))
    batch_size = int(train_cfg.get("batch_size", 64))
    n_epochs = int(train_cfg.get("n_epochs", 10))
    gamma = float(train_cfg.get("gamma", 0.99))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    eval_freq = int(train_cfg.get("eval_freq", 10000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 20))

    # 2. Build vectorized environment
    def make_env() -> SocialGuardEnv:
        return SocialGuardEnv(merged_config_path)

    vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_env_cls([make_env for _ in range(args.n_envs)])
    
    # 3. Build evaluation environment
    eval_env = vec_env_cls([make_env])
    eval_callback = create_eval_callback(
        eval_env,
        eval_freq=eval_freq // args.n_envs,
        n_eval_episodes=n_eval_episodes,
        log_path=tb_log_dir,
        best_model_save_path=os.path.join(args.output_dir, args.run_name),
    )
    
    cbs = [TensorboardCallback(), eval_callback]

    # 4. Initialize PPO
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        tensorboard_log=tb_log_dir,
        device=args.device,
        verbose=1,
    )

    # 5. Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=cbs,
        tb_log_name=args.run_name,
    )

    # 6. Save final model
    final_path = os.path.join(args.output_dir, args.run_name, "final_model.zip")
    model.save(final_path)
    logger.info("Training complete. Model saved to %s", final_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
