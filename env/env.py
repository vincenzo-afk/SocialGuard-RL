"""
env/env.py — Core SocialGuardEnv gymnasium.Env subclass.

This is the central wiring module.  It calls task methods only — never
directly touching sim internals (blueprint rule, Section 8).

Call hierarchy:
    env.step(action)
        → task.step(action)
        → task.get_observation()
        → reward_engine.compute(...)
        → return (obs, reward, terminated, truncated, info)

Supports:
    - Task 1 (task_spam) in this phase — architecture is generic for Phase 3+.
    - gymnasium.utils.env_checker compliance.
    - render_mode="human" for readable stdout output.
    - state() custom method returning a plain Python dict (no numpy arrays).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import yaml

from env.spaces import (
    ObservationSpace,
    ActionSpace,
    ACTION_NAMES,
    OBS_DIM,
)
from env.rewards import RewardEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and return a YAML config dict.

    Args:
        config_path: Path to a YAML file.  Defaults to configs/default.yaml.

    Returns:
        Parsed YAML dict.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core environment
# ---------------------------------------------------------------------------

class SocialGuardEnv(gym.Env):
    """Gym-compatible RL environment for social media integrity moderation.

    The agent plays an automated integrity officer.  It observes user/content
    signals, takes moderation actions, and is rewarded via the multi-objective
    reward function in env/rewards.py.

    Args:
        config_path: Optional path to a YAML config file.  Defaults to
            configs/default.yaml.
        task_override: Optional task name string to override the config value.
            Useful for evaluation scripts.

    Example::

        env = SocialGuardEnv("configs/task1.yaml")
        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    def __init__(
        self,
        config_path: str | Path | None = None,
        task_override: str | None = None,
        seed_offset: int = 0,
    ) -> None:
        """Initialise the environment from a YAML config."""
        super().__init__()

        # ---- load config ------------------------------------------------
        self._cfg: dict[str, Any] = load_config(config_path)
        self._env_cfg: dict[str, Any] = self._cfg["env"]
        self._task_cfg: dict[str, Any] = self._cfg["task"]
        self._reward_cfg: dict[str, Any] = self._cfg["reward"]

        if task_override is not None:
            self._task_cfg["name"] = task_override

        if seed_offset:
            base_seed = int(self._env_cfg.get("seed", 42))
            self._env_cfg["seed"] = base_seed + int(seed_offset)

        # ---- spaces (fixed for all tasks) --------------------------------
        self.observation_space: ObservationSpace = ObservationSpace()
        self.action_space: ActionSpace = ActionSpace()

        # ---- reward engine -----------------------------------------------
        self._reward_engine: RewardEngine = RewardEngine(self._reward_cfg)

        # ---- render mode -------------------------------------------------
        self.render_mode: str | None = self._env_cfg.get("render_mode", None)

        # ---- episode state (initialised properly in reset) ---------------
        self._task: Any = None   # concrete task instance
        self._timestep: int = 0
        self._cumulative_reward: float = 0.0
        self._decision_history: list[dict[str, Any]] = []
        self._episode_step: int = 0

        logger.info(
            "SocialGuardEnv initialised — task=%s render_mode=%s",
            self._task_cfg["name"],
            self.render_mode,
        )

    # ------------------------------------------------------------------
    # gymnasium.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional integer seed for reproducibility.
            options: Optional dict (reserved for future use).

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        effective_seed = seed if seed is not None else int(self._env_cfg.get("seed", 42))

        # Build the active task
        self._task = self._build_task()
        self._task.reset(seed=effective_seed)

        # Reset episode tracking
        self._timestep = 0
        self._cumulative_reward = 0.0
        self._decision_history = []
        self._episode_step = 0

        obs = self._task.get_observation()
        info = self._task.get_info()
        info["episode_step"] = self._episode_step

        if self.render_mode == "human":
            self._render_state(obs, info)

        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one moderation action and advance the environment.

        Args:
            action: Integer action ID in {0, 1, 2, 3, 4}.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).

        Raises:
            RuntimeError: If called before reset().
        """
        if self._task is None:
            raise RuntimeError("reset() must be called before step().")

        # Validate action is in the global action space
        if not self.action_space.contains(np.array(action, dtype=np.int64)):
            raise ValueError(f"Action {action} is not in action_space {self.action_space}")

        # Collect context for reward computation
        gt: int = self._task.get_ground_truth()
        legitimacy: float = self._task.get_legitimacy_score()
        current_hop: int = self._task.get_current_hop()
        allowed: list[int] = self._task.allowed_actions
        escalation_count: int = self._task.get_escalation_count()

        # Advance task state
        self._task.step(action)

        # Compute reward
        breakdown = self._reward_engine.compute(
            action=action,
            is_bot=bool(gt),
            legitimacy_score=legitimacy,
            current_hop=current_hop,
            allowed_actions=allowed,
            escalation_count=escalation_count,
        )
        reward: float = breakdown.total

        # Episode done flags
        self._timestep += 1
        self._episode_step += 1
        self._cumulative_reward += reward

        terminated: bool = self._task.is_done()
        truncated: bool = (
            not terminated
            and self._episode_step >= self._task.max_steps
        )

        # Build info dict (Section 5 contract)
        task_info = self._task.get_info()
        try:
            collateral_count = int(self._task.get_collateral_count())
        except Exception:
            collateral_count = int(self._count_collateral())
        info: dict[str, Any] = {
            "ground_truth": gt,
            "action_taken": action,
            "action_name": ACTION_NAMES.get(action, "unknown"),
            "reward_breakdown": breakdown.to_dict(),
            "collateral_count": collateral_count,
            "episode_step": self._episode_step,
            "task_name": self._task.task_name,
            "cumulative_reward": self._cumulative_reward,
            **task_info,
        }

        # Log decision to history
        self._decision_history.append({
            "step": self._episode_step,
            "action": int(action),
            "ground_truth": int(gt),
            "reward": float(reward),
        })
        if len(self._decision_history) > 10_000:
            self._decision_history = self._decision_history[-10_000:]

        # Next observation (zeros if done)
        if terminated or truncated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            obs = self._task.get_observation()

        if self.render_mode == "human":
            self._render_state(obs, info)

        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        """Render the current environment state to stdout (render_mode='human')."""
        if self.render_mode != "human":
            return
        logger.info(
            "Step %d | Cumulative reward: %.3f",
            self._episode_step,
            self._cumulative_reward,
        )

    def close(self) -> None:
        """Clean up environment resources."""
        logger.debug("SocialGuardEnv.close() called.")

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Merge config overrides into the env for subsequent resets.

        This is used by training curricula to progressively increase difficulty
        without changing the observation/action spaces.
        """
        if not overrides:
            return
        for section, patch in overrides.items():
            if not isinstance(patch, dict):
                self._cfg[section] = patch
                continue
            base = self._cfg.get(section, {})
            if not isinstance(base, dict):
                base = {}
            merged = dict(base)
            merged.update(patch)
            self._cfg[section] = merged

        # Refresh internal references
        self._env_cfg = self._cfg.get("env", self._env_cfg)
        self._task_cfg = self._cfg.get("task", self._task_cfg)
        self._reward_cfg = self._cfg.get("reward", self._reward_cfg)

        # Best-effort: apply relevant overrides to an already-instantiated task.
        if self._task is not None:
            try:
                env_patch = overrides.get("env", {})
                if isinstance(env_patch, dict) and "max_steps" in env_patch and hasattr(self._task, "_max_steps"):
                    self._task._max_steps = int(env_patch["max_steps"])
                task_patch = overrides.get("task", {})
                if isinstance(task_patch, dict) and "action_space" in task_patch and hasattr(self._task, "_allowed_actions"):
                    self._task._allowed_actions = list(task_patch["action_space"])
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Custom public method
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Return a plain Python dict of the full internal environment state.

        All numpy arrays are converted to native Python types so the result
        is JSON-serialisable (required for dashboard integration).

        Returns:
            Dict with timestep, cumulative_reward, decision_history, task info.
        """
        task_info: dict[str, Any] = self._task.get_info() if self._task else {}
        task_name = (
            str(self._task.task_name)
            if self._task is not None and hasattr(self._task, "task_name")
            else str(self._task_cfg.get("name", "unknown"))
        )
        return {
            "timestep": int(self._timestep),
            "episode_step": int(self._episode_step),
            "cumulative_reward": float(self._cumulative_reward),
            "task_name": task_name,
            "active_task": task_name,
            "decision_history": list(self._decision_history),
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in task_info.items()},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_task(self) -> Any:
        """Instantiate the active task from the config task name.

        Returns:
            Concrete task instance (TaskSpam for now; extended in Phase 3).

        Raises:
            ValueError: If the task name is not recognised.
        """
        name: str = str(self._task_cfg.get("name", "task_spam"))
        if name == "task_spam":
            from tasks.task_spam import TaskSpam
            return TaskSpam(self._task_cfg, self._env_cfg)
        if name == "task_misinfo":
            from tasks.task_misinfo import TaskMisinfo
            graph_cfg = self._cfg.get("graph", {})
            return TaskMisinfo(self._task_cfg, self._env_cfg, graph_cfg)
        if name == "task_cib":
            from tasks.task_cib import TaskCIB
            graph_cfg = self._cfg.get("graph", {})
            return TaskCIB(self._task_cfg, self._env_cfg, graph_cfg)
        raise ValueError(
            f"Unknown task '{name}'. Supported: 'task_spam', 'task_misinfo', 'task_cib'."
        )

    def _count_collateral(self) -> int:
        """Count real-user (false positive) removals in the decision history.

        Returns:
            Integer count of steps where a real user was removed.
        """
        from env.spaces import ACTION_REMOVE
        return sum(
            1
            for d in self._decision_history
            if d["action"] == ACTION_REMOVE and d["ground_truth"] == 0
        )

    def _render_state(self, obs: np.ndarray, info: dict[str, Any]) -> None:
        """Print a human-readable summary of the current state.

        Args:
            obs: Current observation vector.
            info: Current step info dict.
        """
        action_name = info.get("action_name", "?")
        gt = info.get("ground_truth", "?")
        reward = info.get("reward_breakdown", {}).get("total", 0.0)
        logger.info(
            "[Step %4d] action=%-13s gt=%-5s reward=%+.3f cumR=%+.3f",
            self._episode_step,
            action_name,
            ("bot" if gt else "human"),
            float(reward),
            float(self._cumulative_reward),
        )
