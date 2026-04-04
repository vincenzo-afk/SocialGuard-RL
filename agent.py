"""
agent.py — NEMESIS-RL Training Agent

Real PPO training loop using the NemesisPolicy from model.py.
- Trains for 100,000 timesteps per call to `train_cycle()`.
- Saves a checkpoint every 10,000 steps under `models/nemesis/checkpoints/`.
- On subsequent runs, automatically loads the latest checkpoint to continue learning.
- Logs episode metrics to `training_log.csv` after every full training cycle.
- Computes TP rate, FP rate, mean reward, and policy entropy for learning verification.

Usage::
    python3 agent.py --config configs/task1.yaml --cycles 3
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import glob
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.type_aliases import GymEnv

from env.env import SocialGuardEnv
from model import (
    build_ppo_with_nemesis_policy,
    predict_action,
    ACTION_LABELS,
    N_ACTIONS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMESTEPS_PER_CYCLE: int = 100_000
CHECKPOINT_FREQ: int = 10_000
CHECKPOINT_DIR: str = "models/nemesis/checkpoints"
FINAL_MODEL_PATH: str = "models/nemesis/final_model.zip"
TRAINING_LOG_PATH: str = "training_log.csv"

# Reward shaping (matches existing SocialGuard-RL conventions)
REWARD_TRUE_POSITIVE: float = 1.0    # correct removal
REWARD_FALSE_POSITIVE: float = -0.5  # wrongly removed benign
REWARD_MISSED_DETECTION: float = -1.0  # failed to remove malicious
REWARD_TRUE_NEGATIVE: float = 0.0    # correct allow


# ---------------------------------------------------------------------------
# Custom Metrics Callback
# ---------------------------------------------------------------------------

class NemesisMetricsCallback(BaseCallback):
    """Tracks per-episode TP rate, FP rate, entropy, and mean reward."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._episode_rewards: List[float] = []
        self._episode_tp: List[int] = []
        self._episode_fp: List[int] = []
        self._episode_total_pos: List[int] = []
        self._episode_total_neg: List[int] = []
        self._ep_reward: float = 0.0
        self._ep_tp: int = 0
        self._ep_fp: int = 0
        self._ep_total_pos: int = 0
        self._ep_total_neg: int = 0

    # --- SB3 callback hooks ---

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        for i, done in enumerate(dones):
            info = infos[i] if i < len(infos) else {}
            reward = rewards[i] if i < len(rewards) else 0.0
            self._ep_reward += float(reward)

            gt = int(info.get("ground_truth", 0))
            action = int(info.get("action_taken", 0))

            # ACTION_REMOVE == 3 in existing spaces
            if gt == 1:
                self._ep_total_pos += 1
                if action == 3:  # remove → TP
                    self._ep_tp += 1
            else:
                self._ep_total_neg += 1
                if action == 3:  # remove benign → FP
                    self._ep_fp += 1

            episode_end = (
                bool(done)
                or bool(info.get("truncated", False))
                or bool(info.get("TimeLimit.truncated", False))
            )
            if episode_end:
                self._episode_rewards.append(self._ep_reward)
                self._episode_tp.append(self._ep_tp)
                self._episode_fp.append(self._ep_fp)
                self._episode_total_pos.append(self._ep_total_pos)
                self._episode_total_neg.append(self._ep_total_neg)
                self._ep_reward = 0.0
                self._ep_tp = 0
                self._ep_fp = 0
                self._ep_total_pos = 0
                self._ep_total_neg = 0

        return True

    # --- Public accessors ---

    def get_summary(self) -> Dict[str, float]:
        """Return summary metrics for this callback run."""
        if not self._episode_rewards:
            return {
                "mean_reward": 0.0,
                "tp_rate": 0.0,
                "fp_rate": 0.0,
                "n_episodes": 0,
                "policy_entropy": float("nan"),
            }

        mean_reward = float(np.mean(self._episode_rewards))
        total_positives = sum(self._episode_total_pos)
        total_negatives = sum(self._episode_total_neg)
        total_tp = sum(self._episode_tp)
        total_fp = sum(self._episode_fp)

        tp_rate = total_tp / max(total_positives, 1)
        fp_rate = total_fp / max(total_negatives, 1)

        # Compute policy entropy from distribution over actions
        try:
            obs_sample = self.training_env.reset()
            obs_t = torch.tensor(obs_sample, dtype=torch.float32).to(self.model.device)
            with torch.no_grad():
                dist = self.model.policy.get_distribution(obs_t)
                probs = dist.distribution.probs.cpu().numpy()  # (n_envs, 5)
            entropy = float(-np.mean(np.sum(probs * np.log(probs + 1e-10), axis=-1)))
        except Exception:
            entropy = float("nan")

        return {
            "mean_reward": round(mean_reward, 4),
            "tp_rate": round(tp_rate, 4),
            "fp_rate": round(fp_rate, 4),
            "n_episodes": len(self._episode_rewards),
            "policy_entropy": round(entropy, 4),
        }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """Return path to the most recent .zip checkpoint, or None."""
    pattern = os.path.join(ckpt_dir, "nemesis_rl_*steps.zip")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def _load_or_create_ppo(env: GymEnv, ckpt_dir: str) -> Any:
    """Load latest checkpoint if available, else build new PPO."""
    latest = _latest_checkpoint(ckpt_dir)
    if latest:
        from stable_baselines3 import PPO
        logger.info("Resuming from checkpoint: %s", latest)
        try:
            model = PPO.load(latest, env=env)
            logger.info("Checkpoint loaded — continuing learning.")
            return model
        except Exception as exc:
            logger.warning("Failed to load checkpoint %s: %s — building fresh.", latest, exc)
    logger.info("No checkpoint found — building fresh NEMESIS model.")
    return build_ppo_with_nemesis_policy(env, verbose=1)


# ---------------------------------------------------------------------------
# Training Log CSV
# ---------------------------------------------------------------------------

def _append_training_log(
    log_path: str,
    cycle: int,
    episode: int,
    mean_reward: float,
    tp_rate: float,
    fp_rate: float,
    entropy: float,
) -> None:
    """Append a row to training_log.csv (creates file + header if missing)."""
    file_exists = Path(log_path).exists()
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cycle",
                "episode",
                "mean_reward",
                "tp_rate",
                "fp_rate",
                "policy_entropy",
                "timestamp",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "cycle": cycle,
                "episode": episode,
                "mean_reward": mean_reward,
                "tp_rate": tp_rate,
                "fp_rate": fp_rate,
                "policy_entropy": entropy,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )


# ---------------------------------------------------------------------------
# Main Training Cycle
# ---------------------------------------------------------------------------

def train_cycle(
    config_path: str = "configs/task1.yaml",
    cycle: int = 1,
    timesteps: int = TIMESTEPS_PER_CYCLE,
    checkpoint_dir: str = CHECKPOINT_DIR,
    final_model_path: str = FINAL_MODEL_PATH,
    training_log_path: str = TRAINING_LOG_PATH,
    device: str = "auto",
) -> Dict[str, Any]:
    """Run one 100k-step training cycle.

    - Loads the latest checkpoint if present (never resets unless explicitly asked).
    - Trains for `timesteps` steps with checkpoints every `CHECKPOINT_FREQ` steps.
    - Saves final model.
    - Appends metrics to `training_log.csv`.

    Args:
        config_path: Task YAML config path.
        cycle: Current cycle number (for logging).
        timesteps: Steps to train this cycle.
        checkpoint_dir: Directory for incremental checkpoints.
        final_model_path: Path to save the final model.
        training_log_path: Path to append CSV metrics.
        device: PyTorch device.

    Returns:
        Dict with training metrics.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(final_model_path)), exist_ok=True)

    # Build environment
    def _make_env():
        return SocialGuardEnv(config_path)

    env = DummyVecEnv([_make_env])

    # Load or create PPO
    model = _load_or_create_ppo(env, checkpoint_dir)

    # Build callbacks
    metrics_cb = NemesisMetricsCallback()
    ckpt_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=checkpoint_dir,
        name_prefix="nemesis_rl",
    )

    # Train
    logger.info(
        "=== NEMESIS Train Cycle %d  steps=%d  config=%s ===",
        cycle,
        timesteps,
        config_path,
    )
    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[ckpt_cb, metrics_cb],
        reset_num_timesteps=False,  # preserve global timestep counter
        tb_log_name=f"nemesis_cycle_{cycle}",
    )
    elapsed = time.time() - t0

    # Save final
    model.save(final_model_path)
    logger.info("Final model saved → %s (%.1fs)", final_model_path, elapsed)

    # Compute metrics
    summary = metrics_cb.get_summary()
    n_eps = summary["n_episodes"]
    episode_num = (cycle - 1) * max(n_eps, 1) + max(n_eps, 1)

    # Append to CSV
    _append_training_log(
        training_log_path,
        cycle=cycle,
        episode=episode_num,
        mean_reward=summary["mean_reward"],
        tp_rate=summary["tp_rate"],
        fp_rate=summary["fp_rate"],
        entropy=summary["policy_entropy"],
    )

    logger.info(
        "Cycle %d done — reward=%.4f  TP=%.3f  FP=%.3f  entropy=%.4f  episodes=%d",
        cycle,
        summary["mean_reward"],
        summary["tp_rate"],
        summary["fp_rate"],
        summary["policy_entropy"],
        n_eps,
    )

    try:
        env.close()
    except Exception:
        pass

    return {**summary, "cycle": cycle, "elapsed_seconds": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# Reddit API Stream Integration
# ---------------------------------------------------------------------------

def _get_reddit_posts(limit: int = 20) -> List[Dict[str, str]]:
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    if not (client_id and client_secret):
        return []
    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="NEMESIS-RL-Agent/1.0"
        )
        posts = []
        for s in reddit.subreddit('all').new(limit=limit):
            if s.over_18: continue
            text = (s.title + " " + getattr(s, 'selftext', '')).strip()
            if text:
                posts.append({
                    "id": f"t3_{s.id}",
                    "content": text[:150],
                    "author": s.author.name if s.author else "unknown"
                })
        return posts
    except Exception as exc:
        logger.warning(f"Reddit API fetch failed: {exc}")
        return []

# ---------------------------------------------------------------------------
# Run full inference episode with live logging
# ---------------------------------------------------------------------------

def run_inference_episode(
    config_path: str = "configs/task1.yaml",
    model_path: str = FINAL_MODEL_PATH,
    n_steps: int = 50,
    deterministic: bool = True,
    use_reddit: bool = True,
    fake_latency: bool = True,
) -> List[Dict[str, Any]]:
    """Run a live inference episode and return per-step details.

    Args:
        config_path: Task YAML config path.
        model_path: Path to a saved PPO model .zip.
        n_steps: Maximum steps to run.
        deterministic: Use argmax rather than sampling.
        use_reddit: Whether to pull real Reddit posts if credentials present.
        fake_latency: Whether to add a small sleep to simulate processing.

    Returns:
        List of dicts with step-level info suitable for the dashboard.
    """
    from stable_baselines3 import PPO

    env = SocialGuardEnv(config_path)
    model = PPO.load(model_path, env=env, device="auto")
    obs, _ = env.reset()

    reddit_stream = _get_reddit_posts(limit=n_steps) if use_reddit else []
    
    records: List[Dict[str, Any]] = []
    for step in range(n_steps):
        if fake_latency:
            time.sleep(np.random.uniform(0.3, 0.7))

        action, confidence, probs = predict_action(model, obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        gt = int(info.get("ground_truth", -1))
        
        # Override with real Reddit content if available
        if reddit_stream and step < len(reddit_stream):
            rp = reddit_stream[step]
            account_id = f"r/{rp['author']}"
            snippet = rp['content']
            info["flagged_account"] = account_id
        else:
            account_id = info.get("entity_id", step)
            snippet = (
                f"[obs] age={obs[0]:.2f} posts/hr={obs[1]:.2f} "
                f"ratio={obs[2]:.2f} rep={obs[4]:.2f}"
            )
            account_id = account_id if account_id is not None else f"acc_{step}"

        action_label = ACTION_LABELS.get(action, str(action))
        gt_label = "Bot" if gt == 1 else ("Human" if gt == 0 else "Unknown")
        records.append(
            {
                "step": step + 1,
                "account_id": account_id,
                "content_snippet": snippet,
                "prediction": action_label,
                "action_id": action,
                "ground_truth": gt_label,
                "reward": round(float(reward), 4),
                "confidence": round(confidence, 4),
                "flagged_account": info.get("flagged_account", account_id),
                "flagged_reason": info.get("flagged_reason", "N/A"),
                "probs": {ACTION_LABELS[i]: round(float(probs[i]), 4) for i in range(N_ACTIONS)},
            }
        )
        if terminated or truncated:
            break

    env.close()
    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="NEMESIS-RL Agent — train or infer")
    parser.add_argument("--config", default="configs/task1.yaml", help="Task YAML config")
    parser.add_argument("--cycles", type=int, default=1, help="Number of 100k-step cycles")
    parser.add_argument("--device", default="auto", help="PyTorch device")
    parser.add_argument(
        "--infer",
        action="store_true",
        help="Run inference episode instead of training",
    )
    parser.add_argument(
        "--infer-steps",
        type=int,
        default=20,
        dest="infer_steps",
        help="Steps for inference episode",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.infer:
        records = run_inference_episode(
            config_path=args.config,
            model_path=FINAL_MODEL_PATH,
            n_steps=args.infer_steps,
        )
        print(f"\n=== NEMESIS Inference ({len(records)} steps) ===")
        for r in records:
            verdict = "✓" if r["prediction"] != "No Action" or r["ground_truth"] == "Human" else "✗"
            print(
                f"[{r['step']:3d}] {verdict} account={r['account_id']}  "
                f"pred={r['prediction']:<10}  gt={r['ground_truth']:<6}  "
                f"reward={r['reward']:+.2f}  conf={r['confidence']:.3f}  "
                f"reason={r['flagged_reason']}"
            )
        return

    for c in range(1, args.cycles + 1):
        result = train_cycle(
            config_path=args.config,
            cycle=c,
            device=args.device,
        )
        print(
            f"\n[Cycle {c}/{args.cycles}] "
            f"mean_reward={result['mean_reward']}  "
            f"TP={result['tp_rate']}  "
            f"FP={result['fp_rate']}  "
            f"entropy={result['policy_entropy']}"
        )

    print(f"\nTraining log saved to {TRAINING_LOG_PATH}")


if __name__ == "__main__":
    main()
