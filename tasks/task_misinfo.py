"""
tasks/task_misinfo.py — Task 2: Viral Misinformation Flagging.

Episode structure:
- At reset(), generate a social graph and start a piece of content spreading.
- Each step() advances the spread by one hop (tick) then returns the 6-feature
  content observation vector.
- The timing bonus degrades with each hop the content has already traveled.
- Terminates when: content removed, max hops reached, or max steps reached.

Action space: all 5 actions available (allow, warn, reduce_reach, remove, escalate).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from env.spaces import (
    TASK2_OBS_DIM,
    pad_observation,
    ACTION_REMOVE,
    ACTION_REDUCE_REACH,
)
from sim.social_graph import SocialGraph
from sim.content_gen import ContentEngine
from tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class TaskMisinfo(BaseTask):
    """Task 2 — Viral misinformation flagging.

    The agent observes content spread metrics and must decide when to act.
    Acting early maximises the speed bonus; waiting risks wider propagation.

    Args:
        task_cfg: Task config dict (noise_level, action_space, …).
        env_cfg: Env config dict (max_steps, seed).
        graph_cfg: Graph config dict (num_nodes, bot_cluster_size, …).
    """

    TASK_NAME: str = "task_misinfo"

    def __init__(
        self,
        task_cfg: dict[str, Any],
        env_cfg: dict[str, Any],
        graph_cfg: dict[str, Any],
    ) -> None:
        """Initialise TaskMisinfo with configuration dicts."""
        super().__init__(task_cfg, env_cfg)
        self._graph_cfg: dict[str, Any] = graph_cfg
        self._noise_level: float = float(task_cfg.get("noise_level", 0.20))

        # Episode state — populated in reset()
        self._graph: SocialGraph | None = None
        self._content_engine: ContentEngine | None = None
        self._current_gt: int = 0       # 1 = misinfo, 0 = legit
        self._legitimacy_score: float = 0.5
        self._escalation_count: int = 0
        self._acted: bool = False       # True once agent takes a decisive action

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset: generate a new graph and start a new piece of content.

        Args:
            seed: Optional integer seed for reproducibility.
        """
        self._reset_step_count()
        self._escalation_count = 0
        self._acted = False

        # Build a smaller graph for Task 2 (faster episodes)
        small_cfg = dict(self._graph_cfg)
        small_cfg["num_nodes"] = min(int(self._graph_cfg.get("num_nodes", 100)), 150)
        small_cfg["bot_cluster_size"] = min(
            int(self._graph_cfg.get("bot_cluster_size", 20)), 30
        )

        self._graph = SocialGraph(small_cfg, seed=seed)
        self._content_engine = ContentEngine(
            graph=self._graph,
            content_cfg={"max_steps": self._max_steps},
            max_hops=20,
            seed=seed,
        )

        post = self._content_engine.reset(seed=seed)
        self._current_gt = int(post.is_misinfo)

        # Legitimacy = inverse of spread rate (high-spread content is less legit)
        self._legitimacy_score = float(np.clip(1.0 - post.spread_rate, 0.0, 1.0))

        logger.debug(
            "TaskMisinfo.reset(): misinfo=%s spread=%.2f",
            bool(self._current_gt), post.spread_rate,
        )

    def get_observation(self) -> np.ndarray:
        """Return the padded 6-feature content observation vector.

        Returns:
            Float32 array of shape (OBS_DIM,) — Task 2 features in [0:6],
            zeros in [6:68].
        """
        if self._content_engine is None:
            return pad_observation(np.zeros(TASK2_OBS_DIM, dtype=np.float32))
        raw = self._content_engine.get_content_observation()
        return pad_observation(raw)

    def get_ground_truth(self) -> int:
        """Return 1 if the current content is misinformation, 0 if legitimate."""
        return self._current_gt

    def get_legitimacy_score(self) -> float:
        """Return legitimacy score (inverse of spread_rate) for reward computation."""
        return self._legitimacy_score

    def get_current_hop(self) -> int:
        """Return the current hop count — used for the timing bonus."""
        if self._content_engine is None:
            return 0
        return self._content_engine.get_current_hop()

    def get_escalation_count(self) -> int:
        """Return escalation count for the current episode."""
        return self._escalation_count

    def is_done(self) -> bool:
        """Return True when spread is complete or agent has acted decisively."""
        if self._content_engine is None:
            return True
        return self._content_engine.is_spread_done() or self._acted

    def get_info(self) -> dict[str, Any]:
        """Return task-specific info dict for this step.

        Returns:
            Plain Python dict with task_name, hop_count, ground_truth, etc.
        """
        post_info = (
            self._content_engine.get_post_info()
            if self._content_engine else {}
        )
        return {
            "task_name": self.TASK_NAME,
            "ground_truth": self._current_gt,
            "legitimacy_score": self._legitimacy_score,
            **post_info,
        }

    def step(self, action: int) -> None:
        """Advance the content spread by one tick and apply the agent's action.

        Args:
            action: Integer action ID (0–4).
        """
        if self._content_engine is None:
            return

        from env.spaces import ACTION_ESCALATE
        if action == ACTION_ESCALATE:
            self._escalation_count += 1

        # Decisive actions stop the episode
        if action in (ACTION_REMOVE, ACTION_REDUCE_REACH):
            if action == ACTION_REMOVE:
                self._content_engine.remove_content()
            self._acted = True

        # Always advance spread one tick (even on allow — content keeps moving)
        if not self._content_engine.is_spread_done():
            self._content_engine.tick()

        self._increment_step()
