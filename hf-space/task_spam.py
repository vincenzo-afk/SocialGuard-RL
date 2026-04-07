"""
tasks/task_spam.py — Task 1: Spam Account Detection.

Episode structure:
- At reset(), generate a shuffled queue of N accounts (70% human / 30% bot
  by default; ratio is config-driven).
- Each step() presents one account's 8-feature observation vector.
- The episode terminates when the queue is exhausted or max_steps reached.
- Action space: {allow, warn, reduce_reach, remove} — escalate NOT available.

Feature vector layout matches TASK1_FEATURE_NAMES in env/spaces.py.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from env.spaces import (
    TASK1_FEATURE_NAMES,
    TASK1_OBS_DIM,
    OBS_DIM,
    pad_observation,
    IDX_ACCOUNT_AGE,
    IDX_POSTS_PER_HOUR,
)
from sim.user_behavior import HumanBehavior, BotBehavior
from tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class TaskSpam(BaseTask):
    """Task 1 — Spam account detection.

    Generates a queue of synthetic user accounts and presents them one at a
    time.  The agent must classify each account as legitimate or malicious and
    take an appropriate moderation action.

    Args:
        task_cfg: Task config dict (bot_ratio, noise_level, action_space, …).
        env_cfg: Env config dict (max_steps, seed).

    Raises:
        ValueError: If task_cfg is missing required keys.
    """

    TASK_NAME: str = "task_spam"

    def __init__(self, task_cfg: dict[str, Any], env_cfg: dict[str, Any]) -> None:
        """Initialise TaskSpam with configuration."""
        super().__init__(task_cfg, env_cfg)
        self._bot_ratio: float = float(task_cfg.get("bot_ratio", 0.30))
        self._noise_level: float = float(task_cfg.get("noise_level", 0.15))

        # Episode state — populated in reset()
        self._queue: list[dict[str, Any]] = []
        self._current_idx: int = 0
        self._current_obs: np.ndarray = np.zeros(TASK1_OBS_DIM, dtype=np.float32)
        self._current_gt: int = 0          # ground truth for current account
        self._legitimacy_score: float = 0.5
        self._rng: np.random.RandomState = np.random.RandomState()
        self._escalation_count: int = 0
        self._collateral_count: int = 0

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset the task: generate a new account queue and reset counters.

        Args:
            seed: Optional integer seed for reproducibility.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._reset_step_count()
        self._current_idx = 0
        self._escalation_count = 0
        self._collateral_count = 0

        # Build queue of accounts for this episode
        n_accounts: int = self._max_steps + 1
        n_bots: int = int(round(n_accounts * self._bot_ratio))
        n_humans: int = n_accounts - n_bots

        logger.debug(
            "TaskSpam.reset(): generating %d accounts (%d bots, %d humans)",
            n_accounts, n_bots, n_humans,
        )

        queue: list[dict[str, Any]] = []

        # Generate human accounts
        human_model = HumanBehavior(noise_level=self._noise_level, rng=self._rng)
        for _ in range(n_humans):
            features = human_model.generate()
            queue.append({
                "features": features,
                "is_bot": False,
                "legitimacy_score": self._compute_legitimacy(features, is_bot=False),
            })

        # Generate bot accounts
        bot_model = BotBehavior(noise_level=self._noise_level, rng=self._rng)
        for _ in range(n_bots):
            features = bot_model.generate()
            queue.append({
                "features": features,
                "is_bot": True,
                "legitimacy_score": self._compute_legitimacy(features, is_bot=True),
            })

        # Shuffle so the agent can't learn position
        self._rng.shuffle(queue)
        self._queue = queue

        # Set first observation
        self._load_current()

    def get_observation(self) -> np.ndarray:
        """Return the padded observation vector for the current account.

        Returns:
            Float32 array of shape (OBS_DIM,) — Task 1 features in [0:8],
            zeros in [8:68].
        """
        if self.is_done():
            return self._get_zero_observation()
        return pad_observation(self._current_obs)

    def get_ground_truth(self) -> int:
        """Return 1 if the current account is a bot, 0 if legitimate."""
        return self._current_gt

    def get_legitimacy_score(self) -> float:
        """Return the legitimacy score of the current account (float [0, 1])."""
        return self._legitimacy_score

    def get_current_hop(self) -> int:
        """Return current hop count — always 0 for Task 1 (no spread model)."""
        return 0

    def get_escalation_count(self) -> int:
        """Return number of escalate actions taken this episode."""
        return self._escalation_count

    def get_collateral_count(self) -> int:
        """Return number of real accounts removed this episode."""
        return self._collateral_count

    def is_done(self) -> bool:
        """Return True when the account queue is exhausted."""
        return self._current_idx >= len(self._queue)

    def get_info(self) -> dict[str, Any]:
        """Return task-specific info for the current step.

        Returns:
            Dict with task_name, queue position, ground truth, legitimacy.
        """
        return {
            "task_name": self.TASK_NAME,
            "queue_position": self._current_idx,
            "queue_length": len(self._queue),
            "entity_id": int(self._current_idx),
            "ground_truth": self._current_gt,
            "legitimacy_score": self._legitimacy_score,
        }

    def step(self, action: int) -> None:
        """Advance to the next account in the queue.

        Args:
            action: Integer action ID taken by the agent.
        """
        from env.spaces import ACTION_ESCALATE, ACTION_REMOVE
        if action == ACTION_ESCALATE and ACTION_ESCALATE in self._allowed_actions:
            self._escalation_count += 1
        if action == ACTION_REMOVE and self._current_gt == 0:
            self._collateral_count += 1

        self._increment_step()
        self._current_idx += 1

        if not self.is_done():
            self._load_current()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_current(self) -> None:
        """Load the current queue entry into the observation / GT cache."""
        entry = self._queue[self._current_idx]
        features_dict = entry["features"]

        # Build ordered float array matching TASK1_FEATURE_NAMES
        raw = np.array(
            [features_dict[k] for k in TASK1_FEATURE_NAMES], dtype=np.float32
        )
        # Normalise large-range features into [0, 1] to match ObservationSpace bounds.
        raw[IDX_ACCOUNT_AGE] = float(np.clip(raw[IDX_ACCOUNT_AGE] / 3650.0, 0.0, 1.0))
        raw[IDX_POSTS_PER_HOUR] = float(np.clip(raw[IDX_POSTS_PER_HOUR] / 200.0, 0.0, 1.0))
        self._current_obs = raw
        self._current_gt = int(entry["is_bot"])
        self._legitimacy_score = float(entry["legitimacy_score"])

    def _get_zero_observation(self) -> np.ndarray:
        """Return an all-zero observation with the full env observation shape."""
        return np.zeros(OBS_DIM, dtype=np.float32)

    def _compute_legitimacy(
        self, features: dict[str, float], is_bot: bool
    ) -> float:
        """Compute a legitimacy score in [0, 1] from raw features.

        Higher score = more legitimate.  Uses a simple weighted heuristic
        based on features that distinguish real users from bots.
        Legitimacy scores for bots will tend to be low, but noise means
        overlap exists.

        Args:
            features: Feature dict from a behavior model.
            is_bot: Ground truth (used to apply a small bias term with noise).

        Returns:
            Float in [0, 1].
        """
        # Weighted combination of signals that indicate legitimacy
        base_score = (
            0.20 * min(features["account_age_days"] / 3650.0, 1.0)
            + 0.15 * features["profile_completeness"]
            + 0.15 * features["follower_ratio"]
            + 0.15 * features["login_time_variance"]
            + 0.15 * (1.0 - features["content_repetition_score"])
            + 0.10 * features["device_fingerprint_uniqueness"]
            + 0.10 * features["ip_diversity_score"]
        )
        raw = base_score + self._rng.normal(0.0, self._noise_level * 0.1)
        return float(np.clip(raw, 0.0, 1.0))
