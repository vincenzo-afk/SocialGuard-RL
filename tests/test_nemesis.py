"""
tests/test_socialguard.py — Tests for SocialGuard-RL model and agent infrastructure.

Verifies:
  - SocialGuardMlpExtractor forward pass (shape, dtypes)
  - SocialGuardPolicy can be built and run through SB3 PPO
  - predict_action returns valid outputs
  - agent.py imports cleanly and _load_or_create_ppo returns a PPO
  - SocialGuardMetricsCallback accumulates stats correctly
  - training_log CSV row format is valid
"""
from __future__ import annotations

import csv
import os
import tempfile
import pytest
import numpy as np
import torch


# ---------------------------------------------------------------------------
# model.py tests
# ---------------------------------------------------------------------------

class TestSocialGuardNetBackbone:
    def test_forward_shape(self):
        from model import SocialGuardNetBackbone, POLICY_INPUT_DIM
        net = SocialGuardNetBackbone()
        x = torch.randn(8, POLICY_INPUT_DIM)
        out = net(x)
        assert out.shape == (8, 128), f"Expected (8, 128), got {out.shape}"

    def test_no_nan_in_output(self):
        from model import SocialGuardNetBackbone, POLICY_INPUT_DIM
        net = SocialGuardNetBackbone()
        x = torch.randn(4, POLICY_INPUT_DIM)
        out = net(x)
        assert not torch.any(torch.isnan(out)), "NaN in backbone output"


class TestSocialGuardMlpExtractor:
    def test_features_dim(self):
        from model import SocialGuardMlpExtractor
        from gymnasium import spaces
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(68,), dtype=np.float32)
        extractor = SocialGuardMlpExtractor(obs_space, features_dim=128)
        assert extractor.features_dim == 128

    def test_forward_shape(self):
        from model import SocialGuardMlpExtractor
        from gymnasium import spaces
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(68,), dtype=np.float32)
        extractor = SocialGuardMlpExtractor(obs_space, features_dim=128)
        x = torch.randn(4, 68)
        out = extractor(x)
        assert out.shape == (4, 128), f"Expected (4, 128), got {out.shape}"

    def test_output_no_nan(self):
        from model import SocialGuardMlpExtractor
        from gymnasium import spaces
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(68,), dtype=np.float32)
        extractor = SocialGuardMlpExtractor(obs_space, features_dim=128)
        x = torch.zeros(2, 68)
        out = extractor(x)
        assert not torch.any(torch.isnan(out))


class TestBuildPPO:
    def test_build_ppo_with_socialguard_policy(self):
        from model import build_ppo_with_socialguard_policy
        from env.env import SocialGuardEnv
        env = SocialGuardEnv("configs/task1.yaml")
        ppo = build_ppo_with_socialguard_policy(env, verbose=0)
        assert ppo is not None
        assert ppo.policy is not None
        env.close()


class TestPredictAction:
    def test_predict_action_returns_valid_action(self):
        from model import build_ppo_with_socialguard_policy, predict_action, N_ACTIONS
        from env.env import SocialGuardEnv
        env = SocialGuardEnv("configs/task1.yaml")
        ppo = build_ppo_with_socialguard_policy(env, verbose=0)
        obs, _ = env.reset()
        action, confidence, probs = predict_action(ppo, obs, deterministic=True)
        assert 0 <= action < N_ACTIONS, f"Invalid action: {action}"
        assert 0.0 <= confidence <= 1.0, f"Invalid confidence: {confidence}"
        assert len(probs) == N_ACTIONS
        assert abs(probs.sum() - 1.0) < 1e-4, f"Probs don't sum to 1: {probs.sum()}"
        env.close()

    def test_predict_action_stochastic(self):
        from model import build_ppo_with_socialguard_policy, predict_action, N_ACTIONS
        from env.env import SocialGuardEnv
        env = SocialGuardEnv("configs/task1.yaml")
        ppo = build_ppo_with_socialguard_policy(env, verbose=0)
        obs, _ = env.reset()
        action, conf, probs = predict_action(ppo, obs, deterministic=False)
        assert 0 <= action < N_ACTIONS
        env.close()


class TestLlamaAnalysis:
    """Test Llama analysis graceful fallback (no real API call in tests)."""

    def test_fallback_without_token(self):
        from model import analyze_content_with_llama
        orig = os.environ.pop("HF_TOKEN", None)
        try:
            result = analyze_content_with_llama("This is test content")
            assert "risk_score" in result
            assert isinstance(result["risk_score"], float)
            assert 0.0 <= result["risk_score"] <= 1.0
        finally:
            if orig is not None:
                os.environ["HF_TOKEN"] = orig


# ---------------------------------------------------------------------------
# agent.py tests
# ---------------------------------------------------------------------------

class TestAgentImports:
    def test_imports_clean(self):
        import agent
        assert hasattr(agent, "train_cycle")
        assert hasattr(agent, "run_inference_episode")
        assert hasattr(agent, "SocialGuardMetricsCallback")
        assert hasattr(agent, "TIMESTEPS_PER_CYCLE")

    def test_timesteps_per_cycle(self):
        from agent import TIMESTEPS_PER_CYCLE
        assert TIMESTEPS_PER_CYCLE == 100_000


class TestLoadOrCreatePPO:
    def test_creates_fresh_model(self):
        from agent import _load_or_create_ppo
        from stable_baselines3.common.vec_env import DummyVecEnv
        from env.env import SocialGuardEnv
        env = DummyVecEnv([lambda: SocialGuardEnv("configs/task1.yaml")])
        with tempfile.TemporaryDirectory() as tmpdir:
            model = _load_or_create_ppo(env, tmpdir)
            assert model is not None
            assert type(model).__name__ == "PPO"
        env.close()

    def test_loads_checkpoint_when_present(self):
        from agent import _load_or_create_ppo, build_ppo_with_socialguard_policy
        from stable_baselines3.common.vec_env import DummyVecEnv
        from env.env import SocialGuardEnv
        env = DummyVecEnv([lambda: SocialGuardEnv("configs/task1.yaml")])
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a model as checkpoint
            ppo = build_ppo_with_socialguard_policy(DummyVecEnv([lambda: SocialGuardEnv("configs/task1.yaml")]), verbose=0)
            ckpt_path = os.path.join(tmpdir, "socialguard_rl_10steps.zip")
            ppo.save(ckpt_path.replace(".zip", ""))
            # Now load_or_create should load it
            loaded = _load_or_create_ppo(env, tmpdir)
            assert loaded is not None
        env.close()


class TestSocialGuardMetricsCallback:
    def test_summary_empty(self):
        from agent import SocialGuardMetricsCallback
        cb = SocialGuardMetricsCallback()
        summary = cb.get_summary()
        assert summary["n_episodes"] == 0
        assert summary["mean_reward"] == 0.0
        assert summary["tp_rate"] == 0.0
        assert summary["fp_rate"] == 0.0


class TestTrainingLogCSV:
    def test_append_and_read(self):
        from agent import _append_training_log
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "training_log.csv")
            _append_training_log(log_path, cycle=1, episode=10, mean_reward=0.5,
                                  tp_rate=0.7, fp_rate=0.1, entropy=1.2)
            _append_training_log(log_path, cycle=1, episode=20, mean_reward=0.8,
                                  tp_rate=0.8, fp_rate=0.05, entropy=0.9)
            with open(log_path, "r") as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
            assert float(rows[1]["tp_rate"]) > float(rows[0]["tp_rate"]), "TP rate should increase"
            assert float(rows[1]["fp_rate"]) < float(rows[0]["fp_rate"]), "FP rate should decrease"
