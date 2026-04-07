"""
tasks/base_task.py — Abstract base class for all NEMESIS-RL tasks.

Every concrete task (task_spam, task_misinfo, task_cib) must subclass
`BaseTask` and implement all abstract methods.  The `env.py` core calls
only the interface defined here — it never reaches into task internals.

Design contract (from blueprint Section 8):
    env.py → task.get_observation()  (never directly calls sim)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """Abstract interface that every NEMESIS-RL task must implement.

    Args:
        task_cfg: Dict with task-specific config keys (name, bot_ratio,
            noise_level, action_space, max_steps, …). Loaded from YAML.
        env_cfg: Dict with environment-level config keys (max_steps, seed).
    """

    def __init__(self, task_cfg: dict[str, Any], env_cfg: dict[str, Any]) -> None:
        """Initialise base task with config dicts."""
        self._task_cfg: dict[str, Any] = task_cfg
        self._env_cfg: dict[str, Any] = env_cfg
        self._step_count: int = 0
        self._max_steps: int = int(env_cfg.get("max_steps", 200))
        self._allowed_actions: list[int] = list(task_cfg.get("action_space", [0, 1, 2, 3, 4]))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_name(self) -> str:
        """Return the task identifier string (e.g. 'task_spam')."""
        return str(self._task_cfg.get("name", "unknown"))

    @property
    def allowed_actions(self) -> list[int]:
        """Return the list of valid action IDs for this task."""
        return self._allowed_actions

    @property
    def step_count(self) -> int:
        """Return the number of steps taken in the current episode."""
        return self._step_count

    @property
    def max_steps(self) -> int:
        """Return the maximum number of steps per episode."""
        return self._max_steps

    # ------------------------------------------------------------------
    # Abstract interface — every task must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Reset the task to a new episode state.

        Args:
            seed: Optional integer seed for reproducibility.
        """

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """Return the current observation vector.

        Returns:
            Float32 numpy array of shape (OBS_DIM,) — padded to full length.
        """

    @abstractmethod
    def get_ground_truth(self) -> int:
        """Return the ground-truth label for the current entity.

        Returns:
            1 if the current entity is malicious/bot, 0 if legitimate.
        """

    @abstractmethod
    def get_legitimacy_score(self) -> float:
        """Return a legitimacy score in [0, 1] for reward computation."""

    @abstractmethod
    def get_current_hop(self) -> int:
        """Return current hop/timestep proxy used for the speed bonus.

        For tasks without a hop concept, return 0.
        """

    @abstractmethod
    def get_escalation_count(self) -> int:
        """Return the number of times ACTION_ESCALATE has been used this episode."""

    @abstractmethod
    def get_collateral_count(self) -> int:
        """Return collateral (false positive removal) count for the episode."""

    @abstractmethod
    def is_done(self) -> bool:
        """Return True if the episode should terminate.

        Termination conditions are task-specific (queue exhausted, all bots
        removed, max hops reached, etc.).
        """

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return a task-specific info dict to include in step() output.

        Returns:
            Plain Python dict (all values must be JSON-serialisable).
        """

    @abstractmethod
    def step(self, action: int) -> None:
        """Advance the task by one step, applying the agent's action.

        Args:
            action: Integer action ID (0–4). The env validates this before
                calling step().
        """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _increment_step(self) -> None:
        """Increment the internal step counter by one."""
        self._step_count += 1

    def _reset_step_count(self) -> None:
        """Reset the step counter to zero (called from reset())."""
        self._step_count = 0
