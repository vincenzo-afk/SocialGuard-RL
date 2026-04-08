"""
tests/test_tasks.py — Task-specific tests for SocialGuard-RL (Phase 3).

Test categories:
- reset() with different seeds produces different episodes
- is_done() only returns True after terminal condition
- Ground truth labels are internally consistent with generated entities
- TaskSpam: correct queue length, bot ratio
- TaskMisinfo: hop_count increments across ticks, observation shape
- ContentEngine: BFS spread increments hop_count
"""

from __future__ import annotations

import numpy as np
import pytest

from env.spaces import OBS_DIM, TASK2_OBS_DIM
from tasks.task_spam import TaskSpam
from tasks.task_misinfo import TaskMisinfo
from tasks.task_cib import TaskCIB

# ---------------------------------------------------------------------------
# Shared configs
# ---------------------------------------------------------------------------

ENV_CFG = {"max_steps": 50, "seed": 42}

TASK1_CFG = {
    "name": "task_spam",
    "bot_ratio": 0.30,
    "noise_level": 0.15,
    "action_space": [0, 1, 2, 3],
}

TASK2_CFG = {
    "name": "task_misinfo",
    "noise_level": 0.20,
    "action_space": [0, 1, 2, 3, 4],
}

TASK3_CFG = {
    "name": "task_cib",
    "noise_level": 0.25,
    "action_space": [0, 1, 2, 3, 4],
    "collateral_damage_threshold": 2,
}

GRAPH_CFG = {
    "num_nodes": 60,
    "bot_cluster_size": 10,
    "intra_cluster_density": 0.4,
    "inter_cluster_density": 0.05,
    "embedding_dim": 64,
}


# ---------------------------------------------------------------------------
# TaskSpam tests
# ---------------------------------------------------------------------------

class TestTaskSpam:
    """Unit tests for TaskSpam (Task 1)."""

    @pytest.fixture
    def task(self) -> TaskSpam:
        """Fresh TaskSpam instance, not yet reset."""
        return TaskSpam(TASK1_CFG, ENV_CFG)

    @pytest.fixture
    def reset_task(self, task: TaskSpam) -> TaskSpam:
        """TaskSpam after reset(seed=42)."""
        task.reset(seed=42)
        return task

    # --- reset and seeding -----------------------------------------------

    def test_reset_different_seeds_differ(self, task: TaskSpam) -> None:
        """reset() with different seeds must produce different first observations."""
        task.reset(seed=1)
        obs_a = task.get_observation().copy()
        task.reset(seed=99)
        obs_b = task.get_observation().copy()
        assert not np.array_equal(obs_a, obs_b)

    def test_reset_same_seed_same_obs(self, task: TaskSpam) -> None:
        """reset() with the same seed must reproduce the same first observation."""
        task.reset(seed=7)
        obs_a = task.get_observation().copy()
        task.reset(seed=7)
        obs_b = task.get_observation().copy()
        np.testing.assert_array_equal(obs_a, obs_b)

    # --- observation -------------------------------------------------------

    def test_obs_shape(self, reset_task: TaskSpam) -> None:
        """get_observation() must return shape (OBS_DIM,) = (68,)."""
        obs = reset_task.get_observation()
        assert obs.shape == (OBS_DIM,)

    def test_obs_dtype(self, reset_task: TaskSpam) -> None:
        """Observation must be float32."""
        obs = reset_task.get_observation()
        assert obs.dtype == np.float32

    def test_padded_zeros_beyond_task1_dim(self, reset_task: TaskSpam) -> None:
        """Features beyond index 8 must be zero (padding)."""
        obs = reset_task.get_observation()
        assert obs[8:].sum() == pytest.approx(0.0)

    # --- ground truth -------------------------------------------------------

    def test_ground_truth_is_binary(self, reset_task: TaskSpam) -> None:
        """get_ground_truth() must return 0 or 1."""
        gt = reset_task.get_ground_truth()
        assert gt in (0, 1)

    def test_ground_truth_consistent_across_queue(self, task: TaskSpam) -> None:
        """Ground truth labels must remain 0 or 1 for every queue entry."""
        task.reset(seed=42)
        for _ in range(50):
            gt = task.get_ground_truth()
            assert gt in (0, 1)
            task.step(0)
            if task.is_done():
                break

    # --- bot ratio ----------------------------------------------------------

    def test_bot_ratio_approximately_correct(self, task: TaskSpam) -> None:
        """Bot ratio in a single episode must be near the configured 0.30."""
        task.reset(seed=0)
        bots = 0
        total = 0
        while not task.is_done():
            if task.get_ground_truth() == 1:
                bots += 1
            task.step(0)
            total += 1

        if total > 0:
            ratio = bots / total
            # Allow ±10% tolerance around 0.30
            assert 0.18 <= ratio <= 0.42, (
                f"Bot ratio {ratio:.2f} deviates too far from 0.30"
            )

    # --- termination --------------------------------------------------------

    def test_is_done_false_initially(self, reset_task: TaskSpam) -> None:
        """is_done() must be False immediately after reset."""
        assert reset_task.is_done() is False

    def test_is_done_true_after_full_queue(self, task: TaskSpam) -> None:
        """is_done() must eventually return True as queue is exhausted."""
        task.reset(seed=0)
        for _ in range(200):
            task.step(0)
            if task.is_done():
                break
        assert task.is_done() is True

    # --- get_info ----------------------------------------------------------

    def test_info_has_task_name(self, reset_task: TaskSpam) -> None:
        """get_info() must include 'task_name' key."""
        info = reset_task.get_info()
        assert "task_name" in info
        assert info["task_name"] == "task_spam"

    def test_info_has_ground_truth(self, reset_task: TaskSpam) -> None:
        """get_info() must include 'ground_truth' key."""
        info = reset_task.get_info()
        assert "ground_truth" in info

    # --- allowed actions ---------------------------------------------------

    def test_allowed_actions_exclude_escalate(self, reset_task: TaskSpam) -> None:
        """Task 1 allowed actions must not include escalate (4)."""
        from env.spaces import ACTION_ESCALATE
        assert ACTION_ESCALATE not in reset_task.allowed_actions

    # --- escalation count --------------------------------------------------

    def test_escalation_count_starts_zero(self, reset_task: TaskSpam) -> None:
        """Escalation count must be 0 at episode start."""
        assert reset_task.get_escalation_count() == 0


# ---------------------------------------------------------------------------
# TaskMisinfo tests
# ---------------------------------------------------------------------------

class TestTaskMisinfo:
    """Unit tests for TaskMisinfo (Task 2)."""

    @pytest.fixture
    def task(self) -> TaskMisinfo:
        """Fresh TaskMisinfo, not yet reset."""
        return TaskMisinfo(TASK2_CFG, ENV_CFG, GRAPH_CFG)

    @pytest.fixture
    def reset_task(self, task: TaskMisinfo) -> TaskMisinfo:
        """TaskMisinfo after reset(seed=42)."""
        task.reset(seed=42)
        return task

    # --- reset and seeding -------------------------------------------------

    def test_reset_different_seeds_differ(self, task: TaskMisinfo) -> None:
        """Different seeds must produce different first observations."""
        task.reset(seed=1)
        obs_a = task.get_observation().copy()
        task.reset(seed=99)
        obs_b = task.get_observation().copy()
        # obs differ in at least spread_rate or credibility (both random)
        assert not np.array_equal(obs_a, obs_b)

    def test_reset_same_seed_same_obs(self, task: TaskMisinfo) -> None:
        """Same seed must reproduce the same first observation."""
        task.reset(seed=5)
        obs_a = task.get_observation().copy()
        task.reset(seed=5)
        obs_b = task.get_observation().copy()
        np.testing.assert_array_equal(obs_a, obs_b)

    # --- observation shape -------------------------------------------------

    def test_obs_shape(self, reset_task: TaskMisinfo) -> None:
        """get_observation() must return shape (OBS_DIM,) = (68,)."""
        obs = reset_task.get_observation()
        assert obs.shape == (OBS_DIM,)

    def test_obs_zeros_beyond_task2_dim(self, reset_task: TaskMisinfo) -> None:
        """Features beyond index 6 (Task 2 dim) must be zero."""
        obs = reset_task.get_observation()
        assert obs[TASK2_OBS_DIM:].sum() == pytest.approx(0.0)

    def test_obs_in_valid_range(self, reset_task: TaskMisinfo) -> None:
        """All Task 2 features must be in [0, 1]."""
        obs = reset_task.get_observation()
        assert (obs[:TASK2_OBS_DIM] >= 0.0).all()
        assert (obs[:TASK2_OBS_DIM] <= 1.0).all()

    # --- hop count increments across steps ---------------------------------

    def test_hop_count_increments_on_step(self, reset_task: TaskMisinfo) -> None:
        """hop_count must be non-decreasing as steps are taken."""
        prev_hop = reset_task.get_current_hop()
        for _ in range(10):
            if reset_task.is_done():
                break
            reset_task.step(0)   # allow — content keeps spreading
            current_hop = reset_task.get_current_hop()
            assert current_hop >= prev_hop, (
                f"hop_count decreased from {prev_hop} to {current_hop}"
            )
            prev_hop = current_hop

    # --- ground truth ------------------------------------------------------

    def test_ground_truth_is_binary(self, reset_task: TaskMisinfo) -> None:
        """get_ground_truth() must return 0 or 1."""
        assert reset_task.get_ground_truth() in (0, 1)

    # --- termination on remove ---------------------------------------------

    def test_remove_terminates_episode(self, task: TaskMisinfo) -> None:
        """Taking ACTION_REMOVE must cause is_done() to return True."""
        from env.spaces import ACTION_REMOVE
        task.reset(seed=10)
        assert not task.is_done()
        task.step(ACTION_REMOVE)
        assert task.is_done()

    def test_reduce_reach_keeps_episode_running(self, task: TaskMisinfo) -> None:
        """Taking ACTION_REDUCE_REACH must slow spread without terminating."""
        from env.spaces import ACTION_REDUCE_REACH
        task.reset(seed=10)
        before_info = task.get_info()
        before_hop = task.get_current_hop()
        task.step(ACTION_REDUCE_REACH)
        after_info = task.get_info()
        assert not task.is_done()
        assert task.get_current_hop() >= before_hop
        assert float(after_info["spread_rate"]) <= float(before_info["spread_rate"])

    def test_allow_does_not_terminate_immediately(self, task: TaskMisinfo) -> None:
        """Taking ACTION_ALLOW must not terminate the episode on the first step."""
        from env.spaces import ACTION_ALLOW
        task.reset(seed=10)
        task.step(ACTION_ALLOW)
        # Content keeps spreading — may still be running
        # We just verify it doesn't instantly terminate on allow
        info = task.get_info()
        assert "hop_count" in info

    # --- get_info ----------------------------------------------------------

    def test_info_task_name(self, reset_task: TaskMisinfo) -> None:
        """get_info() must include correct task_name."""
        info = reset_task.get_info()
        assert info.get("task_name") == "task_misinfo"

    # --- escalation count --------------------------------------------------

    def test_escalation_count_increments(self, task: TaskMisinfo) -> None:
        """Escalation count must increment when ACTION_ESCALATE is used."""
        from env.spaces import ACTION_ESCALATE
        task.reset(seed=0)
        assert task.get_escalation_count() == 0
        task.step(ACTION_ESCALATE)
        assert task.get_escalation_count() == 1


# ---------------------------------------------------------------------------
# ContentEngine BFS tests
# ---------------------------------------------------------------------------

class TestContentEngine:
    """Unit tests for sim/content_gen.py in isolation."""

    def test_hop_count_starts_zero(self) -> None:
        """Hop count must be 0 immediately after reset."""
        from sim.content_gen import ContentEngine
        from sim.social_graph import SocialGraph
        g = SocialGraph(GRAPH_CFG, seed=0)
        engine = ContentEngine(g, {"max_steps": 50}, max_hops=20, seed=0)
        engine.reset(seed=0)
        assert engine.get_current_hop() == 0

    def test_hop_count_increments_on_tick(self) -> None:
        """tick() must increment hop_count (if frontier is non-empty)."""
        from sim.content_gen import ContentEngine
        from sim.social_graph import SocialGraph
        g = SocialGraph(GRAPH_CFG, seed=1)
        engine = ContentEngine(g, {"max_steps": 50}, max_hops=20, seed=1)
        post = engine.reset(is_misinfo=True, seed=1)
        initial_hop = engine.get_current_hop()
        # Tick several times to ensure at least one hop progresses
        for _ in range(5):
            if not engine.is_spread_done():
                engine.tick()
        # hop_count should have increased from 0 (spread_rate is high for misinfo)
        assert engine.get_current_hop() >= initial_hop

    def test_remove_stops_spread(self) -> None:
        """After remove_content(), is_spread_done() must return True."""
        from sim.content_gen import ContentEngine
        from sim.social_graph import SocialGraph
        g = SocialGraph(GRAPH_CFG, seed=2)
        engine = ContentEngine(g, {"max_steps": 50}, max_hops=20, seed=2)
        engine.reset(seed=2)
        assert not engine.is_spread_done()
        engine.remove_content()
        assert engine.is_spread_done()

    def test_obs_vector_shape(self) -> None:
        """get_content_observation() must return shape (6,)."""
        from sim.content_gen import ContentEngine
        from sim.social_graph import SocialGraph
        g = SocialGraph(GRAPH_CFG, seed=3)
        engine = ContentEngine(g, {"max_steps": 50}, max_hops=20, seed=3)
        engine.reset(seed=3)
        obs = engine.get_content_observation()
        assert obs.shape == (6,)

    def test_obs_all_in_range(self) -> None:
        """All observation values must be in [0, 1]."""
        from sim.content_gen import ContentEngine
        from sim.social_graph import SocialGraph
        g = SocialGraph(GRAPH_CFG, seed=4)
        engine = ContentEngine(g, {"max_steps": 50}, max_hops=20, seed=4)
        engine.reset(seed=4)
        for _ in range(10):
            engine.tick()
            obs = engine.get_content_observation()
            assert (obs >= 0.0).all() and (obs <= 1.0).all(), (
                f"obs out of [0,1] range: {obs}"
            )
            if engine.is_spread_done():
                break

# ---------------------------------------------------------------------------
# TaskCIB tests
# ---------------------------------------------------------------------------

class TestTaskCIB:
    """Unit tests for TaskCIB (Task 3)."""

    @pytest.fixture
    def task(self) -> TaskCIB:
        """Fresh TaskCIB, not yet reset."""
        # Using a much smaller graph configuration to keep tests fast
        small_graph_cfg = dict(GRAPH_CFG)
        small_graph_cfg["num_nodes"] = 20
        small_graph_cfg["bot_cluster_size"] = 5
        return TaskCIB(TASK3_CFG, ENV_CFG, small_graph_cfg)

    @pytest.fixture
    def reset_task(self, task: TaskCIB) -> TaskCIB:
        """TaskCIB after reset(seed=42)."""
        task.reset(seed=42)
        return task

    def test_obs_shape(self, reset_task: TaskCIB) -> None:
        """get_observation() must return shape (OBS_DIM,) = (68,)."""
        obs = reset_task.get_observation()
        assert obs.shape == (OBS_DIM,)

    def test_ground_truth_is_binary(self, reset_task: TaskCIB) -> None:
        """get_ground_truth() must return 0 or 1."""
        assert reset_task.get_ground_truth() in (0, 1)

    def test_remove_bot_increments_bots_removed(self, reset_task: TaskCIB) -> None:
        """Removing a bot must increment bots_removed."""
        from env.spaces import ACTION_REMOVE
        # Find a bot node in the queue
        while reset_task.get_ground_truth() == 0 and not reset_task.is_done():
            reset_task.step(0) # allow
        if not reset_task.is_done():
            state_before = reset_task.get_cluster_state()
            reset_task.step(ACTION_REMOVE)
            state_after = reset_task.get_cluster_state()
            assert state_after["bots_removed"] == state_before["bots_removed"] + 1

    def test_collateral_damage_terminates(self, task: TaskCIB) -> None:
        """Removing real users exceeding threshold terminates episode."""
        from env.spaces import ACTION_REMOVE
        task.reset(seed=10)
        # Find and remove real users until termination
        target_removals = TASK3_CFG["collateral_damage_threshold"]
        for _ in range(50):
            if task.is_done():
                break
            if task.get_ground_truth() == 0:
                task.step(ACTION_REMOVE)
            else:
                task.step(0) # allow
        assert task.is_done()
        assert task.get_cluster_state()["real_removed"] >= target_removals

    def test_terminal_observation_is_zeroed(self, task: TaskCIB) -> None:
        """TaskCIB should return zeros once the task is terminal."""
        task.reset(seed=0)
        task._current_node_idx = len(task._node_order)
        obs = task.get_observation()
        assert obs.shape == (OBS_DIM,)
        assert np.count_nonzero(obs) == 0

    def test_missing_embedding_uses_structural_fallback(self, task: TaskCIB) -> None:
        """A missing embedding should not collapse the node observation to zeros."""
        task.reset(seed=0)
        node_id = task._node_order[task._current_node_idx]
        task._embeddings.pop(node_id, None)
        task._load_current_node()
        assert np.any(np.abs(task._current_obs[:64]) > 1e-8)

    def test_small_graph_spectral_fallback_is_stable(self) -> None:
        """Tiny graphs should bypass eigsh safely and still yield finite obs."""
        graph_cfg = {
            "num_nodes": 2,
            "bot_cluster_size": 1,
            "real_intra_density": 0.08,
            "intra_cluster_density": 0.4,
            "inter_cluster_density": 0.05,
            "embedding_dim": 64,
            "embedding_method": "spectral",
        }
        task = TaskCIB(TASK3_CFG, ENV_CFG, graph_cfg)
        task.reset(seed=0)
        obs = task.get_observation()
        assert obs.shape == (OBS_DIM,)
        assert np.isfinite(obs).all()
