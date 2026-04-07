"""
env/rewards.py — Multi-objective reward function for NEMESIS-RL.

Reward formula (per step):
    R = α·correctness
      − β·fp_cost
      − γ·collateral_damage
      + δ·speed_bonus
      − ε·escalation_penalty

All coefficients (α, β, γ, δ, ε) are read from the task config dict.
No literal numbers appear in this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from env.spaces import (
    ACTION_ALLOW,
    ACTION_ESCALATE,
    ACTION_REDUCE_REACH,
    ACTION_REMOVE,
    ACTION_WARN,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """Container for the five named sub-reward components.

    Attributes:
        correctness: Positive signal for correctly identifying a threat.
        fp_cost: Penalty for softer false-positive actions on a real user.
        collateral_damage: Penalty for removing legitimate accounts.
        speed_bonus: Bonus for acting early in the spread/detection timeline.
        escalation_penalty: Penalty for overusing the escalate action.
        total: Scalar sum using the configured coefficients.
    """

    correctness: float = 0.0
    fp_cost: float = 0.0
    collateral_damage: float = 0.0
    speed_bonus: float = 0.0
    escalation_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Return sub-rewards as a plain Python dict for info logging."""
        return {
            "correctness": self.correctness,
            "fp_cost": self.fp_cost,
            "collateral_damage": self.collateral_damage,
            "speed_bonus": self.speed_bonus,
            "escalation_penalty": self.escalation_penalty,
            "total": self.total,
        }


# ---------------------------------------------------------------------------
# RewardEngine
# ---------------------------------------------------------------------------

class RewardEngine:
    """Computes the multi-objective step reward from config-driven coefficients.

    Args:
        reward_cfg: Dict with keys alpha, beta, gamma, delta, epsilon, speed_max_hops.
            Typically loaded from configs/default.yaml under the 'reward' key.
    """

    def __init__(self, reward_cfg: dict[str, Any]) -> None:
        """Initialise the reward engine with coefficient config."""
        self._alpha: float = float(reward_cfg["alpha"])
        self._beta: float = float(reward_cfg["beta"])
        self._gamma: float = float(reward_cfg["gamma"])
        self._delta: float = float(reward_cfg["delta"])
        self._epsilon: float = float(reward_cfg["epsilon"])
        self._speed_max_hops: int = int(reward_cfg["speed_max_hops"])
        if self._speed_max_hops <= 0:
            raise ValueError("reward.speed_max_hops must be > 0")

        logger.debug(
            "RewardEngine initialised — α=%.2f β=%.2f γ=%.2f δ=%.2f ε=%.2f "
            "speed_max_hops=%d",
            self._alpha, self._beta, self._gamma,
            self._delta, self._epsilon, self._speed_max_hops,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        *,
        action: int,
        is_bot: bool,
        legitimacy_score: float,
        current_hop: int,
        allowed_actions: list[int],
        escalation_count: int,
        task_name: str | None = None,
    ) -> RewardBreakdown:
        """Compute the step reward and return a fully decomposed breakdown.

        Args:
            action: Integer action ID taken by the agent (0–4).
            is_bot: Ground-truth label — True if the entity is a bot/threat.
            legitimacy_score: Float in [0, 1]; higher means more legitimate.
                Used to scale collateral damage penalty.
            current_hop: How many hops the content/signal has already spread.
                Used to compute the speed bonus (earlier = higher bonus).
            allowed_actions: List of valid action IDs for the current task.
                Actions outside this list receive the invalid-action penalty.
            escalation_count: Number of times escalate has been used this
                episode so far (used to gate escalation penalty).

        Returns:
            RewardBreakdown with each named component and the total scalar.
        """
        bd = RewardBreakdown()

        # ---- correctness ------------------------------------------------
        bd.correctness = self._compute_correctness(action, is_bot)

        # ---- false positive cost ----------------------------------------
        bd.fp_cost = self._compute_fp_cost(action, is_bot, legitimacy_score)

        # ---- collateral damage ------------------------------------------
        bd.collateral_damage = self._compute_collateral_damage(
            action, is_bot, legitimacy_score
        )

        # ---- speed bonus ------------------------------------------------
        if task_name == "task_cib":
            bd.speed_bonus = 0.0
        else:
            bd.speed_bonus = self._compute_speed_bonus(action, is_bot, current_hop)

        # ---- escalation penalty -----------------------------------------
        bd.escalation_penalty = self._compute_escalation_penalty(
            action, escalation_count, allowed_actions
        )

        # ---- total ------------------------------------------------------
        bd.total = (
            self._alpha * bd.correctness
            - self._beta * bd.fp_cost
            - self._gamma * bd.collateral_damage
            + self._delta * bd.speed_bonus
            - self._epsilon * bd.escalation_penalty
        )

        logger.debug("Reward breakdown: %s", bd.to_dict())
        return bd

    # ------------------------------------------------------------------
    # Private sub-reward computations
    # ------------------------------------------------------------------

    def _compute_correctness(self, action: int, is_bot: bool) -> float:
        """Return 1.0 for a true positive removal, 0.0 otherwise.

        A correct detection occurs when the agent removes a confirmed bot.
        Partial credit (0.5) is given for warn/reduce_reach on a real bot.
        """
        if is_bot:
            if action == ACTION_REMOVE:
                return 1.0
            if action == ACTION_REDUCE_REACH:
                return 0.8
            if action == ACTION_WARN:
                return 0.5   # partial — bot slowed but not eliminated
            if action == ACTION_ESCALATE:
                return 0.3   # escalate on a bot is cautious but valid
            # allow on a bot — missed detection
            return 0.0
        # Real user — no positive correctness
        return 0.0

    def _compute_fp_cost(
        self, action: int, is_bot: bool, legitimacy_score: float
    ) -> float:
        """Return false positive cost when a real user is incorrectly actioned.

        Cost is proportional to the entity's legitimacy score. Hard removals are
        handled by collateral_damage so they are not double-counted here.

        Args:
            action: Agent action.
            is_bot: True if entity is a bot (false positive does not apply).
            legitimacy_score: Float [0, 1].

        Returns:
            Non-negative penalty scalar; 0.0 if the entity is a bot.
        """
        if is_bot:
            return 0.0
        if action in (ACTION_WARN, ACTION_REDUCE_REACH):
            return legitimacy_score * 0.3
        if action == ACTION_ESCALATE:
            return legitimacy_score * 0.1
        # allow on a real user — correct
        return 0.0

    def _compute_collateral_damage(
        self, action: int, is_bot: bool, legitimacy_score: float
    ) -> float:
        """Penalty for removing a legitimate (non-bot) entity.

        This is distinct from fp_cost to allow independent tuning of the
        collateral damage coefficient (gamma).

        Args:
            action: Agent action.
            is_bot: Ground truth.
            legitimacy_score: Float [0, 1].

        Returns:
            Non-negative penalty scalar.
        """
        if is_bot:
            return 0.0
        if action == ACTION_REMOVE:
            return max(legitimacy_score, 0.1)
        return 0.0

    def _compute_speed_bonus(
        self, action: int, is_bot: bool, current_hop: int
    ) -> float:
        """Bonus for acting early — decays linearly with hop count.

        Only fires on true positive removals; early == higher reward.

        Args:
            action: Agent action.
            is_bot: Ground truth.
            current_hop: Number of hops already spread (0 = detected at origin).

        Returns:
            Non-negative bonus scalar in [0, 1].
        """
        if not is_bot:
            return 0.0
        if action not in (ACTION_REMOVE, ACTION_REDUCE_REACH):
            return 0.0
        # Linear decay: 1.0 at hop 0, 0.0 at speed_max_hops
        hop_clamped = min(current_hop, self._speed_max_hops)
        bonus = 1.0 - (hop_clamped / self._speed_max_hops)
        return max(bonus, 0.0)

    def _compute_escalation_penalty(
        self,
        action: int,
        escalation_count: int,
        allowed_actions: list[int],
    ) -> float:
        """Penalty for overusing escalate or using an invalid action.

        The penalty grows with escalation_count to discourage the agent from
        treating escalate as a free action.

        Args:
            action: Agent action.
            escalation_count: Times escalate has been used this episode.
            allowed_actions: Valid action IDs for this task.

        Returns:
            Non-negative penalty scalar.
        """
        # Invalid action for this task
        if action not in allowed_actions:
            return 2.0   # hard penalty for invalid action

        if action != ACTION_ESCALATE:
            return 0.0

        # Escalate is valid but overuse accumulates
        return float(escalation_count)
