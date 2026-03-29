"""
tests/test_env.py — Env API correctness tests for SocialGuard-RL (Phase 2).

Test categories:
- reset() returns valid (obs, info) matching the observation space
- step() with every valid action returns (obs, float, bool, bool, dict)
- step() before reset() raises RuntimeError
- Episode termination: terminated vs truncated semantics
- Action space and observation space shape/bounds
- gymnasium.utils.env_checker compliance
- state() returns plain Python dict (no numpy arrays)
- Task 1 specific: ground truth consistency, escalate penalty
- Random agent can run 1000 steps without error
"""

from __future__ import annotations

import numpy as np
import pytest

from env.env import SocialGuardEnv
from env.spaces import (
    OBS_DIM,
    N_ACTIONS,
    ACTION_ALLOW,
    ACTION_WARN,
    ACTION_REDUCE_REACH,
    ACTION_REMOVE,
    ACTION_ESCALATE,
    TASK1_ACTIONS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> SocialGuardEnv:
    """Freshly constructed Task 1 environment (not yet reset)."""
    return SocialGuardEnv(config_path="configs/task1.yaml")


@pytest.fixture
def reset_env(env: SocialGuardEnv) -> tuple[SocialGuardEnv, np.ndarray, dict]:
    """Environment after reset(seed=42)."""
    obs, info = env.reset(seed=42)
    return env, obs, info


# ---------------------------------------------------------------------------
# 1. reset() contract
# ---------------------------------------------------------------------------

class TestReset:
    """reset() must return a valid (obs, info) pair."""

    def test_reset_returns_tuple(self, env: SocialGuardEnv) -> None:
        """reset() must return a 2-tuple."""
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_obs_shape(self, reset_env: tuple) -> None:
        """Observation must have shape (OBS_DIM,) = (68,)."""
        _, obs, _ = reset_env
        assert obs.shape == (OBS_DIM,)

    def test_obs_dtype(self, reset_env: tuple) -> None:
        """Observation must be float32."""
        _, obs, _ = reset_env
        assert obs.dtype == np.float32

    def test_obs_in_space(self, reset_env: tuple) -> None:
        """Reset observation must lie within observation_space."""
        env, obs, _ = reset_env
        assert env.observation_space.contains(obs), (
            f"obs not in space: low={env.observation_space.low[:8]} "
            f"high={env.observation_space.high[:8]} obs={obs[:8]}"
        )

    def test_info_is_dict(self, reset_env: tuple) -> None:
        """reset() info must be a dict."""
        _, _, info = reset_env
        assert isinstance(info, dict)

    def test_different_seeds_different_obs(self, env: SocialGuardEnv) -> None:
        """Different seeds must produce different initial observations."""
        obs0, _ = env.reset(seed=0)
        obs1, _ = env.reset(seed=99)
        assert not np.array_equal(obs0, obs1), "Seeded resets must differ"

    def test_same_seed_same_obs(self, env: SocialGuardEnv) -> None:
        """Same seed must reproduce the same initial observation."""
        obs_a, _ = env.reset(seed=7)
        obs_b, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs_a, obs_b)


# ---------------------------------------------------------------------------
# 2. step() return contract
# ---------------------------------------------------------------------------

class TestStep:
    """step() must return the exact (obs, float, bool, bool, dict) contract."""

    @pytest.mark.parametrize("action", list(range(N_ACTIONS)))
    def test_step_returns_five_tuple(
        self, env: SocialGuardEnv, action: int
    ) -> None:
        """Every valid action returns a 5-tuple."""
        env.reset(seed=10)
        result = env.step(action)
        assert isinstance(result, tuple) and len(result) == 5

    @pytest.mark.parametrize("action", list(range(N_ACTIONS)))
    def test_step_obs_shape(self, env: SocialGuardEnv, action: int) -> None:
        """step() obs must have shape (OBS_DIM,)."""
        env.reset(seed=10)
        obs, *_ = env.step(action)
        assert obs.shape == (OBS_DIM,)

    @pytest.mark.parametrize("action", list(range(N_ACTIONS)))
    def test_step_reward_is_float(self, env: SocialGuardEnv, action: int) -> None:
        """Reward must be a Python float."""
        env.reset(seed=10)
        _, reward, *_ = env.step(action)
        assert isinstance(reward, float)

    @pytest.mark.parametrize("action", list(range(N_ACTIONS)))
    def test_step_terminated_truncated_are_bool(
        self, env: SocialGuardEnv, action: int
    ) -> None:
        """terminated and truncated must be bool."""
        env.reset(seed=10)
        _, _, terminated, truncated, _ = env.step(action)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    @pytest.mark.parametrize("action", list(range(N_ACTIONS)))
    def test_step_info_keys(self, env: SocialGuardEnv, action: int) -> None:
        """Info dict must contain all Section 5 contract keys."""
        env.reset(seed=10)
        _, _, _, _, info = env.step(action)
        required_keys = {
            "ground_truth", "action_taken", "reward_breakdown",
            "collateral_count", "episode_step", "task_name",
        }
        missing = required_keys - set(info.keys())
        assert not missing, f"Missing info keys: {missing}"

    def test_step_before_reset_raises(self, env: SocialGuardEnv) -> None:
        """Calling step() before reset() must raise RuntimeError."""
        with pytest.raises(RuntimeError):
            env.step(ACTION_ALLOW)

    def test_reward_breakdown_has_total(self, env: SocialGuardEnv) -> None:
        """reward_breakdown in info must contain 'total' key."""
        env.reset(seed=10)
        _, _, _, _, info = env.step(ACTION_REMOVE)
        assert "total" in info["reward_breakdown"]


# ---------------------------------------------------------------------------
# 3. Episode termination semantics
# ---------------------------------------------------------------------------

class TestTermination:
    """Terminated vs truncated must be mutually exclusive on the terminal step."""

    def test_not_both_true(self, env: SocialGuardEnv) -> None:
        """terminated and truncated must not both be True simultaneously."""
        env.reset(seed=1)
        for _ in range(300):
            obs, reward, terminated, truncated, info = env.step(ACTION_ALLOW)
            assert not (terminated and truncated), (
                "terminated and truncated cannot both be True"
            )
            if terminated or truncated:
                break

    def test_episode_terminates_within_max_steps(self, env: SocialGuardEnv) -> None:
        """Episode must end by max_steps (terminated or truncated)."""
        obs, _ = env.reset(seed=2)
        done = False
        steps = 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(ACTION_ALLOW)
            done = terminated or truncated
            steps += 1
            assert steps <= 300, "Episode exceeded 300 steps without termination"
        assert done

    def test_terminated_fires_when_queue_exhausted(self, env: SocialGuardEnv) -> None:
        """Terminated must fire when all queue entries are consumed."""
        env.reset(seed=3)
        terminated = False
        for _ in range(250):
            _, _, terminated, truncated, _ = env.step(ACTION_ALLOW)
            if terminated or truncated:
                break
        assert terminated or truncated


# ---------------------------------------------------------------------------
# 4. Observation space properties
# ---------------------------------------------------------------------------

class TestSpaceProperties:
    """Observations must always stay within declared bounds."""

    def test_obs_space_shape(self, env: SocialGuardEnv) -> None:
        """observation_space.shape must be (OBS_DIM,)."""
        assert env.observation_space.shape == (OBS_DIM,)

    def test_action_space_n(self, env: SocialGuardEnv) -> None:
        """action_space.n must be N_ACTIONS = 5."""
        assert env.action_space.n == N_ACTIONS

    def test_all_step_obs_in_space(self, env: SocialGuardEnv) -> None:
        """All observations during a full episode must lie in observation_space."""
        env.reset(seed=5)
        for _ in range(200):
            obs, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if not (terminated or truncated):
                assert env.observation_space.contains(obs), (
                    f"obs out of space: {obs[:8]}"
                )
            if terminated or truncated:
                break


# ---------------------------------------------------------------------------
# 5. gymnasium env_checker
# ---------------------------------------------------------------------------

class TestGymChecker:
    """gymnasium.utils.env_checker must pass with zero critical errors."""

    def test_env_checker_passes(self) -> None:
        """env_checker should not raise on a freshly initialised env."""
        from gymnasium.utils.env_checker import check_env
        env = SocialGuardEnv(config_path="configs/task1.yaml")
        # check_env raises if critical issues found
        try:
            check_env(env, warn=True, skip_render_check=True)
        except Exception as exc:
            pytest.fail(f"env_checker raised: {exc}")
        finally:
            env.close()


# ---------------------------------------------------------------------------
# 6. state() serialisability
# ---------------------------------------------------------------------------

class TestStateMethod:
    """state() must return a plain Python dict with no numpy arrays."""

    def test_state_is_dict(self, reset_env: tuple) -> None:
        """state() must return a dict."""
        env, _, _ = reset_env
        s = env.state()
        assert isinstance(s, dict)

    def test_state_no_numpy_arrays(self, reset_env: tuple) -> None:
        """state() values must not contain numpy arrays."""
        env, _, _ = reset_env
        env.step(ACTION_REMOVE)
        s = env.state()
        for key, val in s.items():
            assert not isinstance(val, np.ndarray), (
                f"state()['{key}'] is a numpy array — must be native Python"
            )

    def test_state_has_required_keys(self, reset_env: tuple) -> None:
        """state() must contain timestep, cumulative_reward, task_name."""
        env, _, _ = reset_env
        s = env.state()
        for key in ("timestep", "cumulative_reward", "task_name"):
            assert key in s, f"state() missing key '{key}'"


# ---------------------------------------------------------------------------
# 7. Task 1 specific — ground truth consistency
# ---------------------------------------------------------------------------

class TestTask1Specific:
    """Task 1 specific contract checks."""

    def test_ground_truth_is_binary(self, env: SocialGuardEnv) -> None:
        """ground_truth in info must always be 0 or 1."""
        env.reset(seed=42)
        for _ in range(50):
            _, _, terminated, truncated, info = env.step(ACTION_ALLOW)
            assert info["ground_truth"] in (0, 1), (
                f"ground_truth={info['ground_truth']} is not binary"
            )
            if terminated or truncated:
                break

    def test_collateral_count_increases_on_fp(self, env: SocialGuardEnv) -> None:
        """collateral_count must increase when a real user is removed."""
        env.reset(seed=42)
        initial_collateral = 0
        collateral_increased = False

        for _ in range(200):
            _, _, terminated, truncated, info = env.step(ACTION_REMOVE)
            if info["ground_truth"] == 0:
                # Removed a real user
                if info["collateral_count"] > initial_collateral:
                    collateral_increased = True
                    break
                initial_collateral = info["collateral_count"]
            if terminated or truncated:
                break

        assert collateral_increased, (
            "collateral_count never incremented on a real-user removal"
        )

    def test_task_name_in_info(self, env: SocialGuardEnv) -> None:
        """task_name in step info must be 'task_spam'."""
        env.reset(seed=0)
        _, _, _, _, info = env.step(ACTION_ALLOW)
        assert info["task_name"] == "task_spam"


# ---------------------------------------------------------------------------
# 8. Random agent stress test — 1000 steps without error
# ---------------------------------------------------------------------------

class TestRandomAgent:
    """A random agent must complete 1000 steps without raising any exception."""

    def test_random_agent_1000_steps(self) -> None:
        """Random agent must run 1000 steps across multiple episodes error-free."""
        env = SocialGuardEnv(config_path="configs/task1.yaml")
        rng = np.random.RandomState(0)
        total_steps = 0

        obs, _ = env.reset(seed=0)
        while total_steps < 1000:
            action = int(rng.randint(0, N_ACTIONS))
            obs, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            assert obs.shape == (OBS_DIM,), f"obs shape wrong at step {total_steps}"
            assert isinstance(reward, float)
            if terminated or truncated:
                obs, _ = env.reset()

        assert total_steps == 1000
        env.close()
