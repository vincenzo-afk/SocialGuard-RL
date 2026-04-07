"""
baseline.py — Rule-based heuristic agent for NEMESIS-RL.

This deterministic agent uses handcrafted thresholds on the raw observation
features to make moderation decisions.  It serves as the performance floor
that the RL agent must beat.

Task 1 (spam) logic:
    - suspicion_score > high_threshold  → remove
    - suspicion_score > mid_threshold   → warn
    - else                              → allow

Thresholds are loaded from configs/default.yaml under a 'baseline' key if
present, otherwise sensible defaults are used.  No magic literals.

Usage::

    python baseline.py --config configs/task1.yaml --episodes 100
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from env.env import SocialGuardEnv, load_config
from env.spaces import (
    ACTION_ALLOW,
    ACTION_WARN,
    ACTION_REMOVE,
    ACTION_REDUCE_REACH,
    IDX_ACCOUNT_AGE,
    IDX_POSTS_PER_HOUR,
    IDX_FOLLOWER_RATIO,
    IDX_LOGIN_TIME_VARIANCE,
    IDX_CONTENT_REPETITION,
    IDX_PROFILE_COMPLETENESS,
    IDX_DEVICE_FINGERPRINT,
    IDX_IP_DIVERSITY,
    IDX_SPREAD_RATE,
    IDX_SOURCE_CREDIBILITY,
    IDX_HOP_COUNT,
    IDX_TIMESTEP_NORMALIZED,
    IDX_DEGREE_CENTRALITY,
    IDX_CLUSTERING_COEFF,
    IDX_PPH_NORMALIZED,
    OBS_DIM,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default thresholds (used if not in config)
# ---------------------------------------------------------------------------
DEFAULT_HIGH_THRESHOLD: float = 0.70
DEFAULT_MID_THRESHOLD: float = 0.40


class BaselineAgent:
    """Deterministic rule-based agent for Task 1 spam detection.

    Computes a suspicion_score from the raw observation vector using weighted
    feature heuristics, then thresholds that score to pick an action.

    Args:
        high_threshold: suspicion_score above which the agent removes the account.
        mid_threshold: suspicion_score above which the agent warns the account.
    """

    def __init__(
        self,
        high_threshold: float = DEFAULT_HIGH_THRESHOLD,
        mid_threshold: float = DEFAULT_MID_THRESHOLD,
    ) -> None:
        """Initialise the baseline agent with decision thresholds."""
        self._high: float = high_threshold
        self._mid: float = mid_threshold
        logger.info(
            "BaselineAgent thresholds — remove>%.2f  warn>%.2f",
            self._high, self._mid,
        )

    def compute_suspicion_score(self, obs: np.ndarray) -> float:
        """Compute a [0, 1] suspicion score from the Task 1 observation vector.

        Higher score = more likely to be a bot.  Heuristic weights:
        - High posts/hour          → suspicious ↑
        - Low follower ratio       → suspicious ↑
        - Low login time variance  → suspicious ↑
        - High content repetition  → suspicious ↑
        - Low profile completeness → suspicious ↑
        - Low device fingerprint   → suspicious ↑
        - Low IP diversity         → suspicious ↑
        - Low account age          → suspicious ↑

        Args:
            obs: Float32 observation array of shape (OBS_DIM,).

        Returns:
            Suspicion score in [0, 1].
        """
        # Task 1 normalises these into [0, 1] already.
        pph_norm = float(np.clip(obs[IDX_POSTS_PER_HOUR], 0.0, 1.0))
        age_norm = float(np.clip(obs[IDX_ACCOUNT_AGE], 0.0, 1.0))

        score = (
            0.20 * pph_norm
            + 0.15 * (1.0 - obs[IDX_FOLLOWER_RATIO])
            + 0.15 * (1.0 - obs[IDX_LOGIN_TIME_VARIANCE])
            + 0.15 * obs[IDX_CONTENT_REPETITION]
            + 0.15 * (1.0 - obs[IDX_PROFILE_COMPLETENESS])
            + 0.10 * (1.0 - obs[IDX_DEVICE_FINGERPRINT])
            + 0.10 * (1.0 - obs[IDX_IP_DIVERSITY])
        )
        # Newer accounts are slightly more suspicious
        score = 0.85 * score + 0.15 * (1.0 - age_norm)
        return float(np.clip(score, 0.0, 1.0))

    def _infer_task(self, obs: np.ndarray) -> str:
        """Infer active task from sparse padding patterns."""
        if obs.shape[0] >= OBS_DIM and not np.any(np.abs(obs) > 1e-8):
            return "task_spam"
        if np.any(np.abs(obs[8:]) > 1e-8):
            return "task_cib"
        if np.any(np.abs(obs[6:8]) > 1e-8):
            return "task_spam"
        return "task_misinfo"

    def _score_task_misinfo(self, obs: np.ndarray) -> float:
        spread = float(np.clip(obs[IDX_SPREAD_RATE], 0.0, 1.0))
        source_cred = float(np.clip(obs[IDX_SOURCE_CREDIBILITY], 0.0, 1.0))
        hop = float(np.clip(obs[IDX_HOP_COUNT], 0.0, 1.0))
        t = float(np.clip(obs[IDX_TIMESTEP_NORMALIZED], 0.0, 1.0))
        score = (
            0.55 * spread
            + 0.25 * (1.0 - source_cred)
            + 0.10 * (1.0 - hop)
            + 0.10 * (1.0 - t)
        )
        return float(np.clip(score, 0.0, 1.0))

    def _score_task_cib(self, obs: np.ndarray) -> float:
        deg = float(np.clip(obs[IDX_DEGREE_CENTRALITY], 0.0, 1.0))
        clust = float(np.clip(obs[IDX_CLUSTERING_COEFF], 0.0, 1.0))
        pph = float(np.clip(obs[IDX_PPH_NORMALIZED], 0.0, 1.0))
        score = 0.45 * deg + 0.35 * clust + 0.20 * pph
        return float(np.clip(score, 0.0, 1.0))

    def act(self, obs: np.ndarray) -> int:
        """Choose a moderation action based on the computed suspicion score.

        Args:
            obs: Float32 observation array of shape (OBS_DIM,).

        Returns:
            Integer action ID: 0 (allow), 1 (warn), or 3 (remove).
        """
        task = self._infer_task(obs)
        if task == "task_misinfo":
            score = self._score_task_misinfo(obs)
            if score > self._high:
                return ACTION_REMOVE
            if score > self._mid:
                return ACTION_REDUCE_REACH
            return ACTION_ALLOW
        if task == "task_cib":
            score = self._score_task_cib(obs)
            if score > self._high:
                return ACTION_REMOVE
            if score > self._mid:
                return ACTION_WARN
            return ACTION_ALLOW

        score = self.compute_suspicion_score(obs)
        if score > self._high:
            return ACTION_REMOVE
        if score > self._mid:
            return ACTION_WARN
        return ACTION_ALLOW


def run_evaluation(
    agent: BaselineAgent,
    env: SocialGuardEnv,
    n_episodes: int,
) -> dict[str, Any]:
    """Run the baseline agent for n_episodes and collect metrics.

    Args:
        agent: Baseline agent instance.
        env: SocialGuardEnv instance.
        n_episodes: Number of full episodes to run.

    Returns:
        Dict with precision, recall, f1, mean_reward, mean_episode_length.
    """
    tp_total: int = 0
    fp_total: int = 0
    fn_total: int = 0
    rewards: list[float] = []
    lengths: list[int] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward: float = 0.0
        ep_steps: int = 0
        terminated = truncated = False
        ep_seen: set[Any] = set()
        ep_hit: set[Any] = set()

        while not (terminated or truncated):
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1

            gt = int(info["ground_truth"])
            act = int(info["action_taken"])
            entity_id = info.get("entity_id", ep_steps)
            if gt == 1:
                # Count a positive entity once per episode.
                # For Task 2, both REMOVE and REDUCE_REACH are decisive positives.
                ep_seen.add(entity_id)
                if act in (ACTION_REMOVE, ACTION_REDUCE_REACH):
                    ep_hit.add(entity_id)
            elif gt == 0 and act == ACTION_REMOVE:
                fp_total += 1

        tp_total += len(ep_hit)
        fn_total += len(ep_seen - ep_hit)

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        if (ep + 1) % 10 == 0:
            logger.info("Episode %d/%d — reward=%.2f", ep + 1, n_episodes, ep_reward)

    precision = tp_total / (tp_total + fp_total + 1e-9)
    recall = tp_total / (tp_total + fn_total + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    results: dict[str, Any] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_reward": round(float(np.mean(rewards)), 4),
        "mean_episode_length": round(float(np.mean(lengths)), 1),
        "n_episodes": n_episodes,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }
    return results


def main() -> None:
    """CLI entry point for running the baseline agent evaluation."""
    parser = argparse.ArgumentParser(description="NEMESIS-RL Baseline Agent")
    parser.add_argument(
        "--config", default="configs/task1.yaml",
        help="Path to task YAML config",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD,
        help="Suspicion score threshold for REMOVE action",
    )
    parser.add_argument(
        "--mid-threshold", type=float, default=DEFAULT_MID_THRESHOLD,
        help="Suspicion score threshold for WARN action",
    )
    args = parser.parse_args()

    env = SocialGuardEnv(config_path=args.config)
    agent = BaselineAgent(
        high_threshold=args.high_threshold,
        mid_threshold=args.mid_threshold,
    )
    results = run_evaluation(agent, env, args.episodes)
    env.close()

    print("\n=== Baseline Agent Results ===")
    for key, val in results.items():
        print(f"  {key:<25}: {val}")


if __name__ == "__main__":
    main()
