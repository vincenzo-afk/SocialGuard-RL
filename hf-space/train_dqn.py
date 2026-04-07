"""
training/train_dqn.py — DQN training entry point for NEMESIS-RL.

Usage:
  python -m training.train_dqn --config configs/default.yaml --run_name dqn_task1
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import yaml
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from env.env import SocialGuardEnv
from training.callbacks import TensorboardCallback, CurriculumCallback, create_eval_callback
from training.curriculum import task_cib_default_schedule

logger = logging.getLogger(__name__)


def _tensorboard_log_dir(output_dir: str) -> str | None:
    if importlib.util.find_spec("tensorboard") is None:
        logger.warning("tensorboard is not installed; disabling SB3 tensorboard logging.")
        return None
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    return tb_dir


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
    """Run DQN training pipeline."""
    parser = argparse.ArgumentParser(description="Train DQN on NEMESIS-RL")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--task_config", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="dqn_run")
    parser.add_argument("--output_dir", type=str, default="models/")
    parser.add_argument("--n_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--curriculum", action="store_true", help="Enable task-specific curriculum (Task 3).")
    args = parser.parse_args()

    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    tb_log_dir = _tensorboard_log_dir(args.output_dir)

    # 1. Load configuration
    cfg = load_config(args.config, args.task_config)
    merged_config_path = os.path.join(run_dir, f"{args.run_name}_config.yaml")
    with open(merged_config_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f)

    train_cfg = cfg.get("training_dqn", cfg.get("training", {}))
    task_name = str(cfg.get("task", {}).get("name", "task_spam"))
    if task_name == "task_cib" and args.n_envs > 1:
        logger.warning("task_cib with n_envs>1 is unsupported; forcing n_envs=1 for consistency.")
        args.n_envs = 1

    if task_name == "task_cib":
        os.environ.setdefault(
            "SOCIALGUARD_NODE2VEC_CACHE_DIR",
            os.path.join(args.output_dir, args.run_name, "node2vec_cache"),
        )
    
    # Extract standard hyperparams (no magic numbers!)
    total_timesteps = int(train_cfg.get("total_timesteps", 100000))
    learning_rate = float(train_cfg.get("learning_rate", 1e-4))
    buffer_size = int(train_cfg.get("buffer_size", 100000))
    learning_starts = int(train_cfg.get("learning_starts", 1000))
    batch_size = int(train_cfg.get("batch_size", 32))
    gamma = float(train_cfg.get("gamma", 0.99))
    target_update_interval = int(train_cfg.get("target_update_interval", 1000))
    exploration_fraction = float(train_cfg.get("exploration_fraction", 0.1))
    
    eval_freq = int(train_cfg.get("eval_freq", 10000))
    n_eval_episodes = int(train_cfg.get("n_eval_episodes", 20))
    checkpoint_freq = int(train_cfg.get("checkpoint_freq", 0))

    # 2. Build vectorized environment
    def make_env(rank: int, config_path: str = merged_config_path) -> callable:
        def _init() -> SocialGuardEnv:
            return SocialGuardEnv(config_path, seed_offset=rank)
        return _init

    if args.n_envs > 1 and task_name == "task_cib":
        logger.warning("task_cib detected: forcing DummyVecEnv to avoid SubprocVecEnv pickling issues.")
        vec_env_cls = DummyVecEnv
    else:
        vec_env_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    env = vec_env_cls([make_env(i) for i in range(args.n_envs)])
    
    # 3. Build evaluation environment
    eval_env = DummyVecEnv([make_env(10_000)])
    eval_callback = create_eval_callback(
        eval_env,
        eval_freq=eval_freq // args.n_envs,
        n_eval_episodes=n_eval_episodes,
        log_path=tb_log_dir or args.output_dir,
        best_model_save_path=os.path.join(args.output_dir, args.run_name),
    )
    
    cbs = [TensorboardCallback(), eval_callback]
    if args.curriculum and task_name == "task_cib":
        schedule = task_cib_default_schedule(
            total_timesteps=total_timesteps,
            final_env_cfg=cfg.get("env", {}),
            final_graph_cfg=cfg.get("graph", {}),
        )
        cbs.insert(0, CurriculumCallback(schedule))
    if checkpoint_freq > 0:
        ckpt_dir = os.path.join(args.output_dir, args.run_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        cbs.insert(0, CheckpointCallback(save_freq=checkpoint_freq, save_path=ckpt_dir, name_prefix=args.run_name))

    # 4. Initialize DQN
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        tensorboard_log=tb_log_dir,
        device=args.device,
        verbose=1,
    )

    # 5. Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=cbs,
            tb_log_name=args.run_name,
        )
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            eval_env.close()
        except Exception:
            pass

    # 6. Save final model
    os.makedirs(os.path.join(args.output_dir, args.run_name), exist_ok=True)
    final_path = os.path.join(args.output_dir, args.run_name, "final_model.zip")
    model.save(final_path)
    logger.info("Training complete. Model saved to %s", final_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
