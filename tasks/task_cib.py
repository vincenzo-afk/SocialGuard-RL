"""
tasks/task_cib.py — Task 3: Coordinated Inauthentic Behaviour (CIB) Network Takedown.

Episode structure:
- At reset(), generate the full 500-node planted-partition graph and compute
  64-dim node2vec embeddings for every node (cached for the episode).
- Each step presents one node's 68-feature observation vector:
    [embedding (64-dim) | degree_centrality | clustering_coeff |
     community_assignment | posts_per_hour_normalized]
- The agent must decide whether to remove the node or take a softer action.
- Tracks: bot nodes removed (true positives) and real nodes removed (collateral).
- Terminates when: all bots removed, collateral damage exceeds threshold, or
  max steps reached.

Blueprint note (Section 12 — pitfall):
  node2vec is slow to train at episode start.
  Fix: precompute embeddings once in reset() and cache them.
  Only recompute when remove_node() is called (or re-use stale embeddings
  with a stale_embeddings flag — we use the re-use approach here for speed).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from env.spaces import (
    TASK3_EMBEDDING_DIM,
    TASK3_OBS_DIM,
    pad_observation,
    ACTION_REMOVE,
    ACTION_ESCALATE,
)
from sim.social_graph import SocialGraph
from tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

_NODE2VEC_CACHE: dict[tuple, dict[int, np.ndarray]] = {}


class TaskCIB(BaseTask):
    """Task 3 — CIB network takedown via graph RL.

    The agent observes a 68-dim vector per node (64-dim node2vec embedding +
    4 graph features) and takes moderation actions to dismantle the hidden
    bot cluster with minimal collateral damage to real users.

    Args:
        task_cfg: Task config dict (noise_level, action_space,
            collateral_damage_threshold, …).
        env_cfg: Env config dict (max_steps, seed).
        graph_cfg: Graph config dict (num_nodes, bot_cluster_size,
            intra_cluster_density, inter_cluster_density, embedding_dim).
    """

    TASK_NAME: str = "task_cib"

    def __init__(
        self,
        task_cfg: dict[str, Any],
        env_cfg: dict[str, Any],
        graph_cfg: dict[str, Any],
    ) -> None:
        """Initialise TaskCIB from configuration dicts."""
        super().__init__(task_cfg, env_cfg)
        self._graph_cfg: dict[str, Any] = graph_cfg
        self._collateral_threshold: int = int(
            task_cfg.get("collateral_damage_threshold", 10)
        )
        self._embedding_dim: int = int(graph_cfg.get("embedding_dim", 64))

        # Episode state – populated in reset()
        self._graph: SocialGraph | None = None
        self._node_order: list[int] = []        # shuffled presentation order
        self._current_node_idx: int = 0
        self._embeddings: dict[int, np.ndarray] = {}  # node_id → 64-dim vector
        self._embeddings_stale: bool = True

        self._bots_removed: int = 0
        self._real_removed: int = 0             # collateral damage count
        self._escalation_count: int = 0

        self._current_obs: np.ndarray = np.zeros(TASK3_OBS_DIM, dtype=np.float32)
        self._current_gt: int = 0
        self._current_legitimacy: float = 0.5
        self._rng: np.random.RandomState = np.random.RandomState()

    # ------------------------------------------------------------------
    # BaseTask interface
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Reset the episode: generate graph, compute embeddings, shuffle order.

        Args:
            seed: Optional integer seed for reproducibility.

        Note:
            node2vec embedding computation runs once here and is cached.
            This is the expensive step (O(nodes × walks)) but happens only
            at episode start, not on every step.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self._reset_step_count()
        self._bots_removed = 0
        self._real_removed = 0
        self._escalation_count = 0

        # Generate graph
        self._graph = SocialGraph(self._graph_cfg, seed=seed)

        # Compute node2vec embeddings (cache across episodes when seed/config repeat)
        cache_key = (
            int(seed) if seed is not None else -1,
            int(self._graph_cfg.get("num_nodes", 0)),
            int(self._graph_cfg.get("bot_cluster_size", 0)),
            float(self._graph_cfg.get("intra_cluster_density", 0.0)),
            float(self._graph_cfg.get("inter_cluster_density", 0.0)),
            int(self._embedding_dim),
        )
        if cache_key in _NODE2VEC_CACHE:
            self._embeddings = _NODE2VEC_CACHE[cache_key]
        else:
            self._embeddings = self._compute_embeddings(seed=seed)
            _NODE2VEC_CACHE[cache_key] = self._embeddings
            # Keep cache bounded (avoid unbounded RAM growth in long runs)
            if len(_NODE2VEC_CACHE) > 2:
                _NODE2VEC_CACHE.pop(next(iter(_NODE2VEC_CACHE)))
        self._embeddings_stale = False

        # Build shuffled presentation order (all nodes)
        all_nodes = list(self._graph.graph.nodes())
        self._rng.shuffle(all_nodes)
        self._node_order = all_nodes
        self._current_node_idx = 0

        # Load first observation
        self._load_current_node()

        logger.info(
            "TaskCIB.reset(): %d nodes, %d bots, embeddings computed.",
            self._graph.num_nodes,
            len(self._graph.get_bot_nodes()),
        )

    def get_observation(self) -> np.ndarray:
        """Return the full 68-dim observation for the current node.

        Returns:
            Float32 array of shape (OBS_DIM,) = (68,).
        """
        return pad_observation(self._current_obs)

    def get_ground_truth(self) -> int:
        """Return 1 if current node is a bot, 0 if real user."""
        return self._current_gt

    def get_legitimacy_score(self) -> float:
        """Return the legitimacy score of the current node."""
        return self._current_legitimacy

    def get_current_hop(self) -> int:
        """Return 0 — hop-based timing not applicable in Task 3."""
        return 0

    def get_escalation_count(self) -> int:
        """Return number of escalate actions taken this episode."""
        return self._escalation_count

    def get_collateral_count(self) -> int:
        """Return number of real users removed this episode."""
        return self._real_removed

    def is_done(self) -> bool:
        """Return True when any terminal condition is met.

        Terminal conditions:
        - All bot nodes have been removed (mission complete)
        - Collateral damage exceeds threshold (too many real users removed)
        - All nodes in the queue have been presented
        """
        if self._graph is None:
            return True
        bots_remaining = len(self._graph.get_bot_nodes())
        if bots_remaining == 0:
            return True
        if self._real_removed >= self._collateral_threshold:
            return True
        if self._current_node_idx >= len(self._node_order):
            return True
        return False

    def get_info(self) -> dict[str, Any]:
        """Return task-specific info dict for the current step.

        Returns:
            Plain Python dict with task state and cluster statistics.
        """
        cluster_state = self.get_cluster_state()
        return {
            "task_name": self.TASK_NAME,
            "ground_truth": self._current_gt,
            "legitimacy_score": self._current_legitimacy,
            "bots_removed": self._bots_removed,
            "real_removed": self._real_removed,
            "collateral_threshold": self._collateral_threshold,
            **cluster_state,
        }

    def step(self, action: int) -> None:
        """Apply the agent's action and advance to the next node.

        If ACTION_REMOVE is taken, the node is removed from the graph.
        Embeddings are NOT recomputed after removal (stale flag approach)
        as described in the blueprint pitfall note.

        Args:
            action: Integer action ID (0–4).
        """
        if self._graph is None:
            return

        current_node = self._node_order[self._current_node_idx]

        if action == ACTION_ESCALATE and ACTION_ESCALATE in self._allowed_actions:
            self._escalation_count += 1

        if action == ACTION_REMOVE:
            # Only remove if the node is still in the graph (not already gone)
            if current_node in self._graph.graph.nodes():
                is_bot = self._graph.get_node_attrs(current_node)["is_bot"]
                self._graph.remove_node(current_node)
                if is_bot:
                    self._bots_removed += 1
                else:
                    self._real_removed += 1
                # Mark embeddings as stale (reuse for speed — blueprint pattern)
                self._embeddings_stale = True

        self._increment_step()
        self._current_node_idx += 1

        # Skip nodes that were removed from the graph (via cascading removal)
        while (
            self._current_node_idx < len(self._node_order)
            and self._node_order[self._current_node_idx]
            not in self._graph.graph.nodes()
        ):
            self._current_node_idx += 1

        if not self.is_done():
            self._load_current_node()

    def get_cluster_state(self) -> dict[str, Any]:
        """Return current bot/real/removed counts per cluster.

        Returns:
            Plain Python dict with bot_remaining, real_remaining,
            bots_removed, real_removed totals.
        """
        if self._graph is None:
            return {}
        return {
            "bots_remaining": len(self._graph.get_bot_nodes()),
            "real_remaining": len(self._graph.get_real_nodes()),
            "bots_removed": self._bots_removed,
            "real_removed": self._real_removed,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_embeddings(
        self, seed: int | None = None
    ) -> dict[int, np.ndarray]:
        """Compute node2vec embeddings for all nodes in the graph.

        Uses the node2vec library with dimensions matching embedding_dim in
        config.  Workers=1 for determinism; walk parameters are fixed for
        stability.  Embeddings are L2-normalised to [-1, 1] per the
        observation space spec.

        Args:
            seed: Optional integer seed for the random walk generator.

        Returns:
            Dict mapping node_id (int) to float32 ndarray of shape
            (embedding_dim,) with values in [-1, 1].
        """
        try:
            from node2vec import Node2Vec
        except ImportError as exc:
            logger.warning(
                "node2vec is not installed; using zero embeddings for TaskCIB. "
                "Install with: pip install node2vec. (%s)",
                exc,
            )
            return {
                int(node): np.zeros(self._embedding_dim, dtype=np.float32)
                for node in self._graph.graph.nodes()
            }

        logger.info("Computing node2vec embeddings (%d nodes)…", self._graph.num_nodes)

        n2v = Node2Vec(
            self._graph.graph,
            dimensions=self._embedding_dim,
            walk_length=10,
            num_walks=20,
            workers=1,
            seed=seed if seed is not None else 42,
            quiet=True,
        )
        model = n2v.fit(
            window=5,
            min_count=1,
            batch_words=4,
            epochs=5,
        )

        embeddings: dict[int, np.ndarray] = {}
        for node in self._graph.graph.nodes():
            vec = model.wv[str(node)].astype(np.float32)
            # L2-normalise to put values in a compact range
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            # Clip to [-1, 1] as mandated by obs space bounds
            embeddings[node] = np.clip(vec[: self._embedding_dim], -1.0, 1.0)

        logger.info("node2vec embeddings computed for %d nodes.", len(embeddings))
        return embeddings

    def _load_current_node(self) -> None:
        """Load the current node's 68-dim observation vector into cache."""
        if self._current_node_idx >= len(self._node_order):
            self._current_obs = np.zeros(TASK3_OBS_DIM, dtype=np.float32)
            return

        node_id = self._node_order[self._current_node_idx]
        if node_id not in self._graph.graph.nodes():
            self._current_obs = np.zeros(TASK3_OBS_DIM, dtype=np.float32)
            return

        # 64-dim embedding (stale embeddings re-used after removal — blueprint)
        if node_id in self._embeddings:
            emb = self._embeddings[node_id]
        else:
            emb = np.zeros(self._embedding_dim, dtype=np.float32)

        # 4 graph-level features
        graph_feats = self._graph.get_graph_features(node_id)
        extra = np.array([
            graph_feats["degree_centrality"],
            graph_feats["clustering_coefficient"],
            graph_feats["community_assignment"],
            graph_feats["posts_per_hour_normalized"],
        ], dtype=np.float32)

        self._current_obs = np.concatenate([emb, extra]).astype(np.float32)

        # Ground truth and legitimacy for reward computation
        attrs = self._graph.get_node_attrs(node_id)
        self._current_gt = int(attrs["is_bot"])
        self._current_legitimacy = float(attrs["legitimacy_score"])
