from __future__ import annotations

import pytest
import numpy as np

from env.env import SocialGuardEnv
from graders.grader import Grader
from baseline import BaselineAgent

def test_grader_normalized_score_bounds() -> None:
    env = SocialGuardEnv(config_path="configs/task1.yaml")
    grader = Grader(env, n_episodes=2)
    
    # Test task_spam
    score_spam = grader.normalized_score("task_spam", {"f1": 0.5, "mean_reward": 0.0})
    assert 0.0 <= score_spam <= 1.0
    
    # Test task_misinfo
    score_misinfo = grader.normalized_score("task_misinfo", {"f1": 0.5, "mean_episode_length": 5, "max_hops": 20})
    assert 0.0 <= score_misinfo <= 1.0

    # Test task_cib
    # Formula is 0.5*recall + 0.5*f1 - min(collateral_rate*2, 0.5). If 0, then negative clip!
    score_cib = grader.normalized_score("task_cib", {"f1": 0.0, "recall": 0.0, "collateral_damage": 10.0, "real_nodes_count": 50})
    assert 0.0 <= score_cib <= 1.0
    
    env.close()

def test_grader_task_spam_baseline_smoke() -> None:
    env = SocialGuardEnv(config_path="configs/task1.yaml")
    grader = Grader(env, n_episodes=5)  # run 5 episodes to smoke test
    agent = BaselineAgent()
    
    results = grader.evaluate(agent, agent_name="baseline")
    task_metrics = results["tasks"]["task_spam"]
    
    # Assert acceptable F1 (>0 as requested in #73)
    assert task_metrics["f1"] > 0.0
    assert task_metrics["mean_reward"] > 0
    
    env.close()
