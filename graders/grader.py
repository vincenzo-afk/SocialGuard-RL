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

    def evaluate(self, agent: Any, agent_name: str = "agent") -> dict[str, Any]:
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
            obs, info = self.env.reset()
            task_name = info.get("task_name", "unknown")
            ep_reward: float = 0.0
            ep_steps: int = 0
            ep_tp_added = False
            
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
                
                if gt == 1:
                    if act == ACTION_REMOVE:
                        metrics[task_name]["tp"] += 1
                        if not ep_tp_added:
                            metrics[task_name]["detection_times"].append(step_info["episode_step"])
                            ep_tp_added = True
                    else:
                        metrics[task_name]["fn"] += 1
                else:
                    if act == ACTION_REMOVE:
                        metrics[task_name]["fp"] += 1
                    else:
                        # Allow on a real user
                        metrics[task_name]["tn"] += 1

                if (terminated or truncated) and "collateral_count" in step_info:
                    metrics[task_name]["collateral"].append(step_info["collateral_count"])

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
            time_to_detection = float(np.mean(raw_times)) if raw_times else 0.0
            
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
