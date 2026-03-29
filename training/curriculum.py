"""
training/curriculum.py — Simple curriculum schedules for SocialGuard-RL.

Curricula are implemented as a sequence of (timestep_threshold, overrides)
where overrides are merged into `SocialGuardEnv` via `apply_overrides()`.
"""

from __future__ import annotations

from typing import Any


def task_cib_default_schedule(
    *,
    total_timesteps: int,
    final_env_cfg: dict[str, Any],
    final_graph_cfg: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    """Default 3-phase curriculum for Task 3 (CIB).

    Phase goals: start small/easy, then medium, then full-size.
    """
    total_timesteps = int(total_timesteps)
    t0 = 0
    t1 = int(total_timesteps * 0.40)
    t2 = int(total_timesteps * 0.70)

    final_nodes = int(final_graph_cfg.get("num_nodes", 500))
    final_bots = int(final_graph_cfg.get("bot_cluster_size", 80))
    final_steps = int(final_env_cfg.get("max_steps", 500))

    return [
        (
            t0,
            {
                "env": {"max_steps": max(50, min(final_steps, 150))},
                "graph": {
                    "num_nodes": max(50, min(final_nodes, 150)),
                    "bot_cluster_size": max(5, min(final_bots, 20)),
                },
            },
        ),
        (
            t1,
            {
                "env": {"max_steps": max(100, min(final_steps, 300))},
                "graph": {
                    "num_nodes": max(100, min(final_nodes, 300)),
                    "bot_cluster_size": max(10, min(final_bots, 40)),
                },
            },
        ),
        (
            t2,
            {
                "env": {"max_steps": final_steps},
                "graph": {
                    "num_nodes": final_nodes,
                    "bot_cluster_size": final_bots,
                },
            },
        ),
    ]

