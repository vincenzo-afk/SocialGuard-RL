"""
evaluate.py — Evaluate trained RL agents against rule-based baselines.

Usage::

    python evaluate.py --model models/ppo_task1.zip --config configs/task1.yaml --episodes 100
    
Loads the environment via config, initialises a Grader, runs the baseline,
then loads the specified Stable-Baselines3 model, runs it, and compares them.
"""

import argparse
import logging
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, DQN

from env.env import SocialGuardEnv
from baseline import BaselineAgent
from graders.grader import Grader, compare_agents

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> Any:
    """Load SB3 model. Tries PPO first, then DQN based on filename heuristic.

    Warning:
        Stable-Baselines3 model archives are trusted-code inputs. Keep model
        loading restricted to trusted files under the local models directory.
    """
    repo_root = Path(__file__).resolve().parent
    trusted_root = (repo_root / "models").resolve()
    path = (
        (repo_root / model_path).resolve()
        if not Path(model_path).is_absolute()
        else Path(model_path).resolve()
    )
    try:
        path.relative_to(trusted_root)
    except ValueError:
        raise ValueError(
            f"Refusing to load model outside trusted directory: {path}. "
            f"Place models under: {trusted_root}"
        )
    if path.suffix.lower() != ".zip":
        raise ValueError(f"Model must be a .zip file: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Simple heuristic
    if "dqn" in path.name.lower():
        return DQN.load(str(path))
    return PPO.load(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="NEMESIS-RL Evaluation Script")
    parser.add_argument("--model", type=str, required=True, help="Path to trained SB3 model (.zip)")
    parser.add_argument("--config", type=str, default="configs/task1.yaml", help="Path to environment config")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes for evaluation")
    parser.add_argument("--outdir", type=str, default="results", help="Directory to save eval results JSON")
    
    args = parser.parse_args()
    
    logger.info(f"Loading environment from {args.config}...")
    env_baseline = SocialGuardEnv(config_path=args.config)
    
    # 1. Evaluate baseline
    logger.info("Evaluating Rule-Based Baseline...")
    task_name = str(env_baseline._task_cfg.get("name", "task_spam"))
    baseline = BaselineAgent(task_name=task_name)
    grader = Grader(env_baseline, n_episodes=args.episodes)
    
    baseline_stats = grader.evaluate(baseline, agent_name="Baseline")
    
    # 2. Evaluate Trained Model
    logger.info(f"Loading trained model from {args.model}...")
    model = load_model(args.model)
    logger.info("Evaluating Trained RL Model...")
    
    # We must reset grader per eval iteration
    env_model = SocialGuardEnv(config_path=args.config)
    grader = Grader(env_model, n_episodes=args.episodes)
    model_stats = grader.evaluate(model, agent_name=args.model)
    
    # 3. Compare and output
    deltas = compare_agents(baseline_stats, model_stats)
    
    for task_name in deltas:
        logger.info(f"--- Results for {task_name} ---")
        
        b_metrics = baseline_stats["tasks"][task_name]
        m_metrics = model_stats["tasks"][task_name]
        
        logger.info(f"Baseline F1: {b_metrics['f1']:.4f} | Margin Reward: {b_metrics['mean_reward']:.2f}")
        logger.info(f"Model F1:    {m_metrics['f1']:.4f} | Margin Reward: {m_metrics['mean_reward']:.2f}")
        logger.info(f"Delta:       {deltas[task_name]['f1_delta']:+.4f} F1 | {deltas[task_name]['reward_delta']:+.2f} Reward")
        logger.info("-" * 30)

    # Save to JSON
    out_dir = Path(args.outdir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_{Path(args.model).stem}.json"
    grader.save_results(model_stats, str(out_path))
    env_baseline.close()
    env_model.close()

if __name__ == "__main__":
    main()
