"""
tests/test_rewards.py — Unit tests for env/rewards.py (Phase 1).

Test categories:
- Correct bot removal → positive reward
- Incorrect removal of real user → negative reward
- Early detection bonus fires on true positive removal
- Late detection gets lower speed bonus than early detection
- Escalation overuse accumulates penalty
- Invalid action returns maximum penalty
- RewardBreakdown.to_dict() is serialisable
- All coefficients are config-driven (engine respects custom values)
"""

from __future__ import annotations

import pytest

from env.rewards import RewardEngine, RewardBreakdown
from env.spaces import (
    ACTION_ALLOW,
    ACTION_WARN,
    ACTION_REDUCE_REACH,
    ACTION_REMOVE,
    ACTION_ESCALATE,
    N_ACTIONS,
)

# ---------------------------------------------------------------------------
# Shared config fixture — mirrors configs/default.yaml reward block
# ---------------------------------------------------------------------------

DEFAULT_REWARD_CFG = {
    "alpha": 1.0,
    "beta": 0.8,
    "gamma": 1.2,
    "delta": 0.3,
    "epsilon": 0.1,
    "speed_max_hops": 20,
}

ALL_ACTIONS = list(range(N_ACTIONS))   # [0, 1, 2, 3, 4]
TASK1_ACTIONS = [0, 1, 2, 3]          # no escalate in Task 1


@pytest.fixture
def engine() -> RewardEngine:
    """Default reward engine with blueprint coefficients."""
    return RewardEngine(DEFAULT_REWARD_CFG)


# ---------------------------------------------------------------------------
# 1. Correct bot removal → positive total reward
# ---------------------------------------------------------------------------

class TestCorrectBotRemoval:
    """Removing a confirmed bot must yield a positive total reward."""

    def test_remove_bot_positive_reward(self, engine: RewardEngine) -> None:
        """Removing a bot at hop 0 should give clear positive reward."""
        bd = engine.compute(
            action=ACTION_REMOVE,
            is_bot=True,
            legitimacy_score=0.1,
            current_hop=0,
            allowed_actions=ALL_ACTIONS,
            escalation_count=0,
        )
        assert bd.total > 0.0, f"Expected positive reward, got {bd.total}"

    def test_remove_bot_correctness_is_one(self, engine: RewardEngine) -> None:
        """Correctness sub-reward must be 1.0 for a bot removal."""
        bd = engine.compute(
            action=ACTION_REMOVE,
            is_bot=True,
            legitimacy_score=0.1,
            current_hop=0,
            allowed_actions=ALL_ACTIONS,
            escalation_count=0,
        )
        assert bd.correctness == pytest.approx(1.0)

    def test_warn_bot_gives_partial_correctness(self, engine: RewardEngine) -> None:
        """Warning a bot returns 0.5 correctness (partial credit)."""
        bd = engine.compute(
            action=ACTION_WARN,
            is_bot=True,
            legitimacy_score=0.1,
            current_hop=0,
            allowed_actions=ALL_ACTIONS,
            escalation_count=0,
        )
        assert bd.correctness == pytest.approx(0.5)

    def test_allow_bot_zero_correctness(self, engine: RewardEngine) -> None:
        """Allowing a bot (missed detection) → zero correctness."""
        bd = engine.compute(
            action=ACTION_ALLOW,
            is_bot=True,
            legitimacy_score=0.1,
            current_hop=0,
            allowed_actions=ALL_ACTIONS,
            escalation_count=0,
        )
        assert bd.correctness == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 2. Removing a high-legitimacy real user → negative total reward
#    (blueprint pitfall check: legitimacy_score > 0.8 must give negative R)
# ---------------------------------------------------------------------------

class TestFalsePositivePenalty:
    """False positive removals must produce negative reward."""

    def test_remove_high_legitimacy_user_negative_reward(self, engine: RewardEngine) -> None:
        """Removing a real user with legitimacy 0.9 must return negative total."""
        bd = engine.compute(
            action=ACTION_REMOVE,
            is_bot=False,
            legitimacy_score=0.9,
            current_hop=0,
            allowed_actions=ALL_ACTIONS,
            escalation_count=0,
        )
        assert bd.total < 0.0, (
            f"Removing a real user with high legitimacy must be negative, "
            f"got {bd.total}"
        )

    def test_fp_cost_scales_with_legitimacy(self, engine: RewardEngine) -> None:
        """Higher legitimacy → higher fp_cost."""
        bd_low = engine.compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.2,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_high = engine.compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.9,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd_high.fp_cost > bd_low.fp_cost

    def test_allow_real_user_no_fp_cost(self, engine: RewardEngine) -> None:
        """Allowing a real user is correct — fp_cost must be 0."""
        bd = engine.compute(
            action=ACTION_ALLOW, is_bot=False, legitimacy_score=0.9,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd.fp_cost == pytest.approx(0.0)

    def test_collateral_damage_only_on_removal(self, engine: RewardEngine) -> None:
        """collateral_damage fires only when a real user is removed."""
        bd_warn = engine.compute(
            action=ACTION_WARN, is_bot=False, legitimacy_score=0.9,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_remove = engine.compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.9,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd_warn.collateral_damage == pytest.approx(0.0)
        assert bd_remove.collateral_damage > 0.0


# ---------------------------------------------------------------------------
# 3. Speed bonus — early detection > late detection
# ---------------------------------------------------------------------------

class TestSpeedBonus:
    """Speed bonus must degrade with hop count."""

    def test_early_removal_higher_bonus(self, engine: RewardEngine) -> None:
        """Removing a bot at hop 0 must have higher speed_bonus than at hop 10."""
        bd_early = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_late = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=10, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd_early.speed_bonus > bd_late.speed_bonus

    def test_speed_bonus_zero_at_max_hops(self, engine: RewardEngine) -> None:
        """Speed bonus must reach 0 at speed_max_hops."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=20, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd.speed_bonus == pytest.approx(0.0)

    def test_speed_bonus_one_at_hop_zero(self, engine: RewardEngine) -> None:
        """Speed bonus must be 1.0 at hop 0."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd.speed_bonus == pytest.approx(1.0)

    def test_speed_bonus_zero_for_real_user(self, engine: RewardEngine) -> None:
        """Speed bonus must NOT fire on a false positive removal."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.5,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd.speed_bonus == pytest.approx(0.0)

    def test_total_reward_early_greater_than_late(self, engine: RewardEngine) -> None:
        """Total reward for early true positive > late true positive."""
        bd_early = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_late = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=19, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        assert bd_early.total > bd_late.total


# ---------------------------------------------------------------------------
# 4. Escalation overuse
# ---------------------------------------------------------------------------

class TestEscalationPenalty:
    """Escalation penalty must accumulate with escalation_count."""

    def test_first_escalation_small_penalty(self, engine: RewardEngine) -> None:
        """First escalate returns penalty of 0.0 (count=0 at call time)."""
        bd = engine.compute(
            action=ACTION_ESCALATE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        # escalation_penalty = epsilon * escalation_count = 0.1 * 0 = 0.0
        assert bd.escalation_penalty == pytest.approx(0.0)

    def test_escalation_penalty_grows(self, engine: RewardEngine) -> None:
        """Escalation penalty must grow with escalation_count."""
        bd_low = engine.compute(
            action=ACTION_ESCALATE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=1,
        )
        bd_high = engine.compute(
            action=ACTION_ESCALATE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=10,
        )
        assert bd_high.escalation_penalty > bd_low.escalation_penalty

    def test_non_escalate_action_no_penalty(self, engine: RewardEngine) -> None:
        """Non-escalate actions always have zero escalation_penalty."""
        for action in [ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE]:
            bd = engine.compute(
                action=action, is_bot=True, legitimacy_score=0.1,
                current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=99,
            )
            assert bd.escalation_penalty == pytest.approx(0.0), (
                f"Action {action} should have zero escalation_penalty"
            )


# ---------------------------------------------------------------------------
# 5. Invalid action penalty
# ---------------------------------------------------------------------------

class TestInvalidAction:
    """Using an action outside the task's allowed set must be penalised."""

    def test_escalate_in_task1_gets_penalty(self, engine: RewardEngine) -> None:
        """Task 1 forbids escalate — using it must incur the invalid-action penalty."""
        bd = engine.compute(
            action=ACTION_ESCALATE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=TASK1_ACTIONS,  # [0, 1, 2, 3]
            escalation_count=0,
        )
        # Invalid action sets escalation_penalty = 2.0
        assert bd.escalation_penalty == pytest.approx(2.0)

    def test_valid_action_no_invalid_penalty(self, engine: RewardEngine) -> None:
        """A valid action must have no invalid-action penalty."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=TASK1_ACTIONS,
            escalation_count=0,
        )
        assert bd.escalation_penalty == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Config-driven coefficients
# ---------------------------------------------------------------------------

class TestConfigCoefficients:
    """Reward magnitudes must scale with config coefficients."""

    def test_alpha_scales_correctness(self) -> None:
        """Doubling alpha must double the correctness contribution."""
        cfg_base = {**DEFAULT_REWARD_CFG, "alpha": 1.0}
        cfg_double = {**DEFAULT_REWARD_CFG, "alpha": 2.0}

        bd_base = RewardEngine(cfg_base).compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=20, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_double = RewardEngine(cfg_double).compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=20, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        # With hop=20, speed_bonus=0, so only correctness and fp_cost contribute
        # total_double - total_base should equal alpha_diff * correctness
        delta = bd_double.total - bd_base.total
        assert delta == pytest.approx(1.0 * bd_base.correctness, abs=1e-6)

    def test_beta_scales_fp_cost(self) -> None:
        """Higher beta must increase the FP penalty magnitude."""
        cfg_low = {**DEFAULT_REWARD_CFG, "beta": 0.1}
        cfg_high = {**DEFAULT_REWARD_CFG, "beta": 2.0}

        bd_low = RewardEngine(cfg_low).compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.8,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        bd_high = RewardEngine(cfg_high).compute(
            action=ACTION_REMOVE, is_bot=False, legitimacy_score=0.8,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        # Higher beta → more negative total
        assert bd_high.total < bd_low.total


# ---------------------------------------------------------------------------
# 7. RewardBreakdown serialisation
# ---------------------------------------------------------------------------

class TestRewardBreakdownSerialisable:
    """to_dict() must return a plain Python dict with native float values."""

    def test_to_dict_all_keys_present(self, engine: RewardEngine) -> None:
        """All five sub-reward keys plus total must appear in to_dict()."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        d = bd.to_dict()
        expected_keys = {
            "correctness", "fp_cost", "collateral_damage",
            "speed_bonus", "escalation_penalty", "total",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values_are_floats(self, engine: RewardEngine) -> None:
        """All dict values must be native Python floats (serialisable)."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=0, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        for key, val in bd.to_dict().items():
            assert isinstance(val, float), f"Key '{key}' is {type(val)}, expected float"

    def test_breakdown_total_matches_components(self, engine: RewardEngine) -> None:
        """The 'total' in to_dict() must match the manually computed formula."""
        bd = engine.compute(
            action=ACTION_REMOVE, is_bot=True, legitimacy_score=0.1,
            current_hop=5, allowed_actions=ALL_ACTIONS, escalation_count=0,
        )
        expected = (
            DEFAULT_REWARD_CFG["alpha"] * bd.correctness
            - DEFAULT_REWARD_CFG["beta"] * bd.fp_cost
            - DEFAULT_REWARD_CFG["gamma"] * bd.collateral_damage
            + DEFAULT_REWARD_CFG["delta"] * bd.speed_bonus
            - DEFAULT_REWARD_CFG["epsilon"] * bd.escalation_penalty
        )
        assert bd.total == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# 8. Spaces sanity checks (imported here so Phase 1 tests cover spaces.py)
# ---------------------------------------------------------------------------

class TestSpaces:
    """Basic sanity checks on ObservationSpace and ActionSpace."""

    def test_observation_space_shape(self) -> None:
        """Observation space must have shape (68,)."""
        from env.spaces import ObservationSpace, OBS_DIM
        obs = ObservationSpace()
        assert obs.shape == (OBS_DIM,)

    def test_action_space_n(self) -> None:
        """Action space must be Discrete(5)."""
        from env.spaces import ActionSpace, N_ACTIONS
        act = ActionSpace()
        assert act.n == N_ACTIONS

    def test_observation_space_contains_zeros(self) -> None:
        """A zero-padded observation must be within the space."""
        import numpy as np
        from env.spaces import ObservationSpace, OBS_DIM
        obs = ObservationSpace()
        zero_obs = np.zeros(OBS_DIM, dtype=np.float32)
        assert obs.contains(zero_obs)

    def test_pad_observation_correct_length(self) -> None:
        """pad_observation must extend short vectors to OBS_DIM."""
        import numpy as np
        from env.spaces import pad_observation, OBS_DIM, TASK1_OBS_DIM
        short = np.ones(TASK1_OBS_DIM, dtype=np.float32)
        padded = pad_observation(short)
        assert padded.shape == (OBS_DIM,)
        assert padded[TASK1_OBS_DIM:].sum() == pytest.approx(0.0)

    def test_pad_observation_raises_on_overflow(self) -> None:
        """pad_observation must raise ValueError if vector is too long."""
        import numpy as np
        from env.spaces import pad_observation, OBS_DIM
        too_long = np.ones(OBS_DIM + 1, dtype=np.float32)
        with pytest.raises(ValueError):
            pad_observation(too_long)


# ---------------------------------------------------------------------------
# 9. UserBehavior sanity checks (sim/user_behavior.py — Phase 1 scope)
# ---------------------------------------------------------------------------

class TestUserBehavior:
    """Basic sanity checks on HumanBehavior and BotBehavior."""

    def test_human_generates_all_feature_keys(self) -> None:
        """HumanBehavior.generate() must return all 8 feature keys."""
        from sim.user_behavior import HumanBehavior, FEATURE_KEYS
        rng = __import__("numpy").random.RandomState(42)
        human = HumanBehavior(noise_level=0.15, rng=rng)
        features = human.generate()
        assert set(features.keys()) == set(FEATURE_KEYS)

    def test_bot_generates_all_feature_keys(self) -> None:
        """BotBehavior.generate() must return all 8 feature keys."""
        from sim.user_behavior import BotBehavior, FEATURE_KEYS
        rng = __import__("numpy").random.RandomState(42)
        bot = BotBehavior(noise_level=0.15, rng=rng)
        features = bot.generate()
        assert set(features.keys()) == set(FEATURE_KEYS)

    def test_human_is_not_bot(self) -> None:
        """HumanBehavior.is_bot must be False."""
        from sim.user_behavior import HumanBehavior
        human = HumanBehavior(noise_level=0.0)
        assert human.is_bot is False

    def test_bot_is_bot(self) -> None:
        """BotBehavior.is_bot must be True."""
        from sim.user_behavior import BotBehavior
        bot = BotBehavior(noise_level=0.0)
        assert bot.is_bot is True

    def test_human_features_in_valid_range(self) -> None:
        """Human features must be within their declared ranges after 200 samples."""
        import numpy as np
        from sim.user_behavior import HumanBehavior
        rng = np.random.RandomState(0)
        human = HumanBehavior(noise_level=0.15, rng=rng)
        for _ in range(200):
            f = human.generate()
            assert 0.0 <= f["account_age_days"] <= 3650.0
            assert 0.0 <= f["posts_per_hour"] <= 200.0
            for key in ["follower_ratio", "login_time_variance",
                        "content_repetition_score", "profile_completeness",
                        "device_fingerprint_uniqueness", "ip_diversity_score"]:
                assert 0.0 <= f[key] <= 1.0, f"Human feature {key}={f[key]} out of range"

    def test_bot_features_in_valid_range(self) -> None:
        """Bot features must be within their declared ranges after 200 samples."""
        import numpy as np
        from sim.user_behavior import BotBehavior
        rng = np.random.RandomState(0)
        bot = BotBehavior(noise_level=0.15, rng=rng)
        for _ in range(200):
            f = bot.generate()
            assert 0.0 <= f["account_age_days"] <= 3650.0
            assert 0.0 <= f["posts_per_hour"] <= 200.0
            for key in ["follower_ratio", "login_time_variance",
                        "content_repetition_score", "profile_completeness",
                        "device_fingerprint_uniqueness", "ip_diversity_score"]:
                assert 0.0 <= f[key] <= 1.0, f"Bot feature {key}={f[key]} out of range"

    def test_invalid_noise_level_raises(self) -> None:
        """Noise level outside [0, 1] must raise ValueError."""
        from sim.user_behavior import HumanBehavior
        with pytest.raises(ValueError):
            HumanBehavior(noise_level=1.5)

    def test_bots_post_more_than_humans_on_average(self) -> None:
        """Bot posts_per_hour mean must be significantly higher than human mean."""
        import numpy as np
        from sim.user_behavior import HumanBehavior, BotBehavior
        rng_h = np.random.RandomState(1)
        rng_b = np.random.RandomState(2)
        human = HumanBehavior(noise_level=0.0, rng=rng_h)
        bot = BotBehavior(noise_level=0.0, rng=rng_b)
        N = 500
        human_pph = [human.generate()["posts_per_hour"] for _ in range(N)]
        bot_pph = [bot.generate()["posts_per_hour"] for _ in range(N)]
        assert np.mean(bot_pph) > np.mean(human_pph) * 3, (
            "Bots should post significantly more than humans"
        )
