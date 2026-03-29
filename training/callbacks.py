"""
training/callbacks.py — Callbacks for logging and early stopping.

Includes:
- TensorboardCallback: logs custom metrics from the environment's info dict
  (e.g., reward_breakdown, collateral_damage, true pos/false pos).
- create_eval_callback: helper to build SB3 EvalCallback with early stopping.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecEnv

logger = logging.getLogger(__name__)


class TensorboardCallback(BaseCallback):
    """Custom callback to log info dict metrics to TensorBoard.

    Extracts `reward_breakdown` and task-specific counters from the
    environment's `info` dict at the end of every episode and logs them.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Store episode-wise accumulators if needed, though most
        # metrics from info are already episodic totals (Task 3) or 
        # computed per step. We'll pick up the latest info dict on done.

    def _on_step(self) -> bool:
        """Called by SB3 at each step. Logs metrics on episode boundaries."""
        # SB3 uses `infos` (plural) for VecEnv; keep a fallback for older code.
        infos = self.locals.get("infos", None)
        if infos is None:
            infos = self.locals.get("info", None)
        if infos is None:
            return True

        dones = self.locals.get("dones", None)
        if dones is None:
            dones = self.locals.get("done", None)
        if dones is None:
            return True

        if isinstance(dones, (bool, np.bool_)):
            dones = [bool(dones)]
            infos = [infos]

        for i, done in enumerate(dones):
            if done:
                info = infos[i] if isinstance(infos, (list, tuple)) else infos

                # 1. Log reward breakdown if present
                if "reward_breakdown" in info:
                    for k, v in info["reward_breakdown"].items():
                        self.logger.record(f"reward_components/{k}", v)

                # 2. Log task-specific termination stats
                if "real_removed" in info:
                    self.logger.record("task_cib/real_removed", info["real_removed"])
                if "bots_removed" in info:
                    self.logger.record("task_cib/bots_removed", info["bots_removed"])
                
                # 3. Log misinfo metrics
                if "total_reach" in info:
                    self.logger.record("task_misinfo/total_reach", info["total_reach"])
                if "hop_count" in info:
                    self.logger.record("task_misinfo/terminal_hop_count", info["hop_count"])

        return True


def create_eval_callback(
    eval_env: VecEnv,
    eval_freq: int = 10000,
    n_eval_episodes: int = 20,
    patience: int = 5,
    min_evals: int = 2,
    log_path: str | None = None,
    best_model_save_path: str | None = None,
) -> EvalCallback:
    """Builds an EvalCallback hooked with EarlyStopping.

    Args:
        eval_env: Vectorised environment for evaluation.
        eval_freq: Timesteps between evaluations.
        n_eval_episodes: Episodes per evaluation.
        patience: Number of evals with no improvement before stopping.
        min_evals: Minimum evals before early stopping can trigger.
        log_path: Directory to save evaluation logs.
        best_model_save_path: Directory to save the best model.

    Returns:
        Configured EvalCallback instance.
    """
    early_stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=patience,
        min_evals=min_evals,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_after_eval=early_stop_callback,
        best_model_save_path=best_model_save_path,
        log_path=log_path,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        verbose=1,
    )

    return eval_callback
