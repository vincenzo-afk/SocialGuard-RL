from __future__ import annotations

from env.env import SocialGuardEnv
from env.spaces import ACTION_ALLOW


def test_apply_overrides_affects_next_reset() -> None:
    env = SocialGuardEnv(config_path="configs/task1.yaml")
    env.apply_overrides({"env": {"max_steps": 3}})
    env.reset(seed=0)

    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = env.step(ACTION_ALLOW)
        steps += 1
        assert steps < 50

    assert terminated or truncated
    assert steps == 3
    env.close()


def test_apply_overrides_does_not_interrupt_active_episode() -> None:
    env = SocialGuardEnv(config_path="configs/task1.yaml")
    env.reset(seed=0)

    _, _, terminated, truncated, info_before = env.step(ACTION_ALLOW)
    assert not (terminated or truncated)
    assert info_before["episode_step"] == 1

    env.apply_overrides({"env": {"max_steps": 3}})
    _, _, terminated, truncated, info_after = env.step(ACTION_ALLOW)

    assert info_after["episode_step"] == 2
    assert not info_after.get("already_done", False)
    assert not (terminated and truncated)
    env.close()
