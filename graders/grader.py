"""
graders/grader.py — Evaluation Engine for SocialGuard-RL.

Run evaluation episodes for a given policy and compute metrics:
precision, recall, F1, mean reward, mean length, and time-to-detection.
"""

from __future__ import annotations

import collections
import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from env.env import SocialGuardEnv
from env.spaces import ACTION_REMOVE

logger = logging.getLogger(__name__)


class SB3Policy(Protocol):
    """Protocol for Stable-Baselines3 style policies."""
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray | int, Any]: ...


class BaselinePolicy(Protocol):
    """Protocol for baseline policies."""
    def act(self, obs: np.ndarray) -> int: ...


Policy = SB3Policy | BaselinePolicy


class Grader:
    """Evaluation engine for computing agent performance.

    Runs N episodes and computes standard ML metrics (Precision, Recall, F1)
    and RL-specific metrics (mean reward, time-to-detection).
    """

    def __init__(self, env: SocialGuardEnv, n_episodes: int = 100) -> None:
        """Initialise the grader.

        Args:
            env: The SocialGuard environment instance.
            n_episodes: Number of test episodes to run.
        """
        self.env = env
        self.n_episodes = n_episodes

    def evaluate(
        self,
        agent: Any,
        agent_name: str = "agent",
        base_seed: int = 42,
    ) -> dict[str, Any]:
        """Run evaluation episodes and compute metrics per task.

        Args:
            agent: The policy to evaluate. Needs a .act() or .predict() method.
            agent_name: Name of the agent for reporting.

        Returns:
            Dict containing nested metrics by task name.
        """
        metrics: dict[str, Any] = collections.defaultdict(lambda: {
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "rewards": [], "lengths": [],
            "detection_times": [],
            "collateral": [],
        })

        for ep in range(self.n_episodes):
            obs, info = self.env.reset(seed=int(base_seed) + ep)
            task_name = info.get("task_name", "unknown")
            ep_reward: float = 0.0
            ep_steps: int = 0
            ep_tp_added = False
            bots_seen: set[Any] = set()
            bots_removed: set[Any] = set()
            false_removes: int = 0
             
            terminated = truncated = False
            while not (terminated or truncated):
                if hasattr(agent, "predict"):
                    # stable-baselines3 interface
                    action_arr, _ = agent.predict(obs, deterministic=True)
                    action = int(action_arr.item() if hasattr(action_arr, "item") else action_arr)
                elif hasattr(agent, "act"):
                    # baseline agent interface
                    action = agent.act(obs)
                else:
                    raise ValueError("Agent must have act() or predict() method.")

                obs, reward, terminated, truncated, step_info = self.env.step(action)
                ep_reward += reward
                ep_steps += 1

                gt = step_info["ground_truth"]
                act = step_info["action_taken"]

                entity_id = step_info.get("entity_id")
                if entity_id is None:
                    # Task 2 presents one content entity over multiple steps.
                    entity_id = "misinfo_content" if task_name == "task_misinfo" else step_info.get("episode_step")

                if gt == 1:
                    bots_seen.add(entity_id)
                    if act == ACTION_REMOVE:
                        bots_removed.add(entity_id)
                        if not ep_tp_added:
                            if task_name == "task_misinfo":
                                metrics[task_name]["detection_times"].append(
                                    int(step_info.get("hop_count", step_info["episode_step"]))
                                )
                            else:
                                metrics[task_name]["detection_times"].append(step_info["episode_step"])
                            ep_tp_added = True
                elif act == ACTION_REMOVE:
                    false_removes += 1
                else:
                    metrics[task_name]["tn"] += 1

                if (terminated or truncated) and "collateral_count" in step_info:
                    metrics[task_name]["collateral"].append(step_info["collateral_count"])

            # Episode-level confusion counts (prevents FN double-counting).
            metrics[task_name]["tp"] += len(bots_removed)
            metrics[task_name]["fn"] += len(bots_seen - bots_removed)
            metrics[task_name]["fp"] += false_removes

            metrics[task_name]["rewards"].append(ep_reward)
            metrics[task_name]["lengths"].append(ep_steps)

        return self._compile_results(metrics, agent_name)

    def _compile_results(self, raw_metrics: dict[str, Any], agent_name: str) -> dict[str, Any]:
        """Convert raw counts into standard metrics dict."""
        compiled: dict[str, Any] = {"agent_name": agent_name, "tasks": {}}
        
        for task_name, counts in raw_metrics.items():
            tp = counts["tp"]
            fp = counts["fp"]
            fn = counts["fn"]

            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)

            raw_times = counts["detection_times"]
            if raw_times:
                time_to_detection = float(np.mean(raw_times))
            elif task_name in ("task_misinfo", "task_spam"):
                # No detection: treat as slow detection, not instant.
                time_to_detection = float(np.mean(counts["lengths"])) if counts["lengths"] else 20.0
            else:
                time_to_detection = 0.0
            
            collateral = counts["collateral"]
            mean_collateral = float(np.mean(collateral)) if collateral else 0.0

            compiled["tasks"][task_name] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "mean_reward": round(float(np.mean(counts["rewards"])), 4),
                "mean_episode_length": round(float(np.mean(counts["lengths"])), 1),
                "time_to_detection": round(time_to_detection, 1),
                "mean_collateral": round(mean_collateral, 2),
                "n_episodes": len(counts["rewards"]),
                "tp": tp,
                "fp": fp,
                "fn": fn
            }

        return compiled

    def normalized_score(self, task_name: str, task_metrics: dict[str, float], max_hops: int = 20) -> float:
        """Compute the normalized score [0.0, 1.0] for a task using README formulas.

        Args:
            task_name: One of 'task_spam', 'task_misinfo', 'task_cib'.
            task_metrics: Dict from _compile_results() for a single task.
            max_hops: Max hops for Task 2 speed calculation (default 20).

        Returns:
            Float in [0.0, 1.0].
        """
        f1 = float(task_metrics.get("f1", 0.0))
        mean_reward = float(task_metrics.get("mean_reward", 0.0))
        mean_collateral = float(task_metrics.get("mean_collateral", 0.0))
        time_to_detection = float(
            task_metrics.get(
                "time_to_detection",
                task_metrics.get("mean_episode_length", max_hops),
            )
        )

        if task_name == "task_spam":
            # score = 0.7 × F1 + 0.3 × sigmoid(mean_reward / 50.0)
            import math
            sig = 1.0 / (1.0 + math.exp(-mean_reward / 50.0))
            return float(np.clip(0.7 * f1 + 0.3 * sig, 0.0, 1.0))

        if task_name == "task_misinfo":
            # speed_score = 1.0 − (mean_detection_hop / max_hops)
            # score = 0.6 × F1 + 0.4 × max(0, speed_score)
            speed_score = 1.0 - (time_to_detection / max_hops)
            return float(np.clip(0.6 * f1 + 0.4 * max(0.0, speed_score), 0.0, 1.0))

        if task_name == "task_cib":
            # bots_removed_pct approximated from tp / (tp + fn) = recall
            recall = float(task_metrics.get("recall", 0.0))
            # collateral_rate: mean_collateral relative to a reference of 10 (threshold)
            collateral_rate = min(mean_collateral / 10.0, 1.0)
            collateral_penalty = min(collateral_rate * 2.0, 0.5)
            return float(np.clip(0.5 * recall + 0.5 * f1 - collateral_penalty, 0.0, 1.0))

        # Default: return raw F1
        return float(np.clip(f1, 0.0, 1.0))

    def save_results(self, results: dict[str, Any], filepath: str) -> None:
        """Save grading results to a JSON file."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {filepath}")

def compare_agents(baseline_results: dict[str, Any], model_results: dict[str, Any]) -> dict[str, Any]:
    """Compare an RL model against the baseline and report deltas per task.
    
    Returns:
        Dict highlighting the difference (e.g. +f1, +reward).
    """
    comparison = {}
    for task_name in model_results["tasks"]:
        if task_name in baseline_results["tasks"]:
            b_metrics = baseline_results["tasks"][task_name]
            m_metrics = model_results["tasks"][task_name]
            
            comparison[task_name] = {
                "f1_delta": round(m_metrics["f1"] - b_metrics["f1"], 4),
                "reward_delta": round(m_metrics["mean_reward"] - b_metrics["mean_reward"], 4),
                "time_delta": round(m_metrics["time_to_detection"] - b_metrics["time_to_detection"], 1)
            }
    return comparison
