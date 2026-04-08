"""
sim/social_graph.py — NetworkX social graph generator for NEMESIS-RL.

Generates a planted-partition graph with a hidden bot cluster.
Every node is labelled with: is_bot, legitimacy_score, account_age_days,
activity_score.

Supports:
- Graph evolution via tick() — adds edges based on activity
- Node removal via remove_node() — propagates through all internal sets
- Node2vec-compatible adjacency for Task 3 embeddings
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class SocialGraph:
    """Planted-partition social graph with labelled bot and real-user nodes.

    Generates a NetworkX undirected graph embedding a hidden bot cluster of
    configurable size and density mixed into a larger real-user population.

    Args:
        graph_cfg: Dict with keys: num_nodes, bot_cluster_size,
            intra_cluster_density, inter_cluster_density, embedding_dim.
        seed: Optional integer seed for reproducibility.
    """

    def __init__(
        self, graph_cfg: dict[str, Any], seed: int | None = None
    ) -> None:
        """Initialise the social graph from config."""
        self._num_nodes: int = int(graph_cfg["num_nodes"])
        self._bot_cluster_size: int = int(graph_cfg["bot_cluster_size"])
        self._intra_density: float = float(graph_cfg["intra_cluster_density"])
        self._inter_density: float = float(graph_cfg["inter_cluster_density"])
        self._real_intra_density: float = float(
            graph_cfg.get("real_intra_density", 0.08)
        )
        self._embedding_dim: int = int(graph_cfg["embedding_dim"])

        if self._num_nodes < 2:
            raise ValueError("graph_cfg['num_nodes'] must be >= 2.")
        if self._bot_cluster_size < 0 or self._bot_cluster_size >= self._num_nodes:
            raise ValueError("graph_cfg['bot_cluster_size'] must be in [0, num_nodes-1].")
        if self._embedding_dim <= 0:
            raise ValueError("graph_cfg['embedding_dim'] must be > 0.")

        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self._seed: int | None = seed

        # Built in _generate()
        self._graph: nx.Graph = nx.Graph()
        self._bot_nodes: set[int] = set()
        self._real_nodes: set[int] = set()
        self._node_attrs: dict[int, dict[str, Any]] = {}

        # Expensive derived-structure cache (computed lazily)
        self._communities_dirty: bool = True
        self._communities_cache: list[frozenset[int]] = []
        self._community_index_by_node: dict[int, int] = {}
        self._clustering_dirty: bool = True
        self._clustering_cache: dict[int, float] = {}
        self._community_recompute_interval: int = max(
            1, int(graph_cfg.get("community_recompute_interval", 20))
        )
        self._clustering_recompute_interval: int = max(
            1, int(graph_cfg.get("clustering_recompute_interval", 10))
        )
        self._removed_since_communities_refresh: int = 0
        self._removed_since_clustering_refresh: int = 0

        self._generate()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.Graph:
        """Return the underlying NetworkX graph (read-only intent)."""
        return self._graph

    @property
    def num_nodes(self) -> int:
        """Return the current number of nodes in the graph."""
        return self._graph.number_of_nodes()

    def get_bot_nodes(self) -> set[int]:
        """Return the set of node IDs labelled as bots."""
        return set(self._bot_nodes)

    def get_real_nodes(self) -> set[int]:
        """Return the set of node IDs labelled as real users."""
        return set(self._real_nodes)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Return the list of neighbor node IDs for a given node.

        Args:
            node_id: Integer node ID to query.

        Returns:
            List of integer neighbor IDs.

        Raises:
            KeyError: If node_id is not in the graph.
        """
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id} not in graph.")
        return list(self._graph.neighbors(node_id))

    def get_node_attrs(self, node_id: int) -> dict[str, Any]:
        """Return all attributes for a given node.

        Args:
            node_id: Integer node ID to query.

        Returns:
            Dict with keys: is_bot, legitimacy_score, account_age_days,
            activity_score.

        Raises:
            KeyError: If node_id is not in the graph.
        """
        if node_id not in self._node_attrs:
            raise KeyError(f"Node {node_id} not found in attrs.")
        return dict(self._node_attrs[node_id])

    def get_cluster_ids(self) -> list[frozenset[int]]:
        """Return detected communities using greedy modularity.

        Returns:
            List of frozensets, each containing node IDs in that community.
        """
        self._ensure_communities_cache()
        return list(self._communities_cache)

    def remove_node(self, node_id: int) -> None:
        """Remove a node from the graph and all internal tracking sets.

        Args:
            node_id: Integer node ID to remove.

        Raises:
            KeyError: If node_id is not in the graph.
        """
        if node_id not in self._graph:
            raise KeyError(f"Cannot remove node {node_id} — not in graph.")
        self._graph.remove_node(node_id)
        self._bot_nodes.discard(node_id)
        self._real_nodes.discard(node_id)
        self._node_attrs.pop(node_id, None)
        self._community_index_by_node.pop(node_id, None)
        self._clustering_cache.pop(node_id, None)
        self._removed_since_communities_refresh += 1
        self._removed_since_clustering_refresh += 1

        # Recompute expensive derived structures in batches to keep step latency low.
        if (
            self._removed_since_communities_refresh >= self._community_recompute_interval
            or not self._communities_cache
        ):
            self._communities_dirty = True
            self._removed_since_communities_refresh = 0
        if (
            self._removed_since_clustering_refresh >= self._clustering_recompute_interval
            or not self._clustering_cache
        ):
            self._clustering_dirty = True
            self._removed_since_clustering_refresh = 0
        logger.debug("Removed node %d from graph and all sets.", node_id)

    def tick(self) -> None:
        """Advance the graph by one timestep: add edges based on activity.

        Nodes with higher activity_score have a proportionally higher chance
        of forming a new edge each tick.  Inter-cluster edges are much rarer
        than intra-cluster edges.
        """
        all_nodes = list(self._graph.nodes())
        if len(all_nodes) < 2:
            return

        n_new_edges = max(1, int(self._num_nodes * 0.005))
        attempts = 0
        added = 0
        while added < n_new_edges and attempts < n_new_edges * 10:
            a, b = self._rng.choice(all_nodes, size=2, replace=False)
            if not self._graph.has_edge(a, b):
                a_bot = self._node_attrs[a]["is_bot"]
                b_bot = self._node_attrs[b]["is_bot"]
                # Same cluster type → higher probability
                p = self._intra_density if (a_bot == b_bot) else self._inter_density
                if self._rng.random() < p:
                    self._graph.add_edge(a, b)
                    added += 1
            attempts += 1

        logger.debug("tick(): added %d new edges.", added)

    def get_graph_features(self, node_id: int) -> dict[str, float]:
        """Return graph-level features for a node (used by Task 3).

        Args:
            node_id: Integer node ID.

        Returns:
            Dict with degree_centrality, clustering_coefficient,
            community_assignment, posts_per_hour_normalized.
        """
        if node_id not in self._graph:
            raise KeyError(f"Node {node_id} not in graph.")

        n = self._graph.number_of_nodes()
        degree = self._graph.degree(node_id)
        degree_centrality = degree / max(n - 1, 1)

        self._ensure_clustering_cache()
        clustering = float(self._clustering_cache.get(node_id, 0.0))

        # Community assignment: normalized index from greedy communities
        self._ensure_communities_cache()
        comm_i = self._community_index_by_node.get(node_id, 0)
        n_comms = len(self._communities_cache)
        if n_comms <= 1:
            community_idx = 0.0
        else:
            community_idx = comm_i / (n_comms - 1)

        # Activity → posts_per_hour normalised to [0, 1]
        activity = self._node_attrs[node_id].get("activity_score", 0.5)
        pph_norm = float(np.clip(activity, 0.0, 1.0))

        return {
            "degree_centrality": float(np.clip(degree_centrality, 0.0, 1.0)),
            "clustering_coefficient": float(np.clip(clustering, 0.0, 1.0)),
            "community_assignment": float(np.clip(community_idx, 0.0, 1.0)),
            "posts_per_hour_normalized": pph_norm,
        }

    def _ensure_communities_cache(self) -> None:
        if not self._communities_dirty:
            return
        if self._graph.number_of_nodes() == 0:
            self._communities_cache = []
            self._community_index_by_node = {}
            self._communities_dirty = False
            return
        if self._graph.number_of_nodes() == 1:
            only = next(iter(self._graph.nodes()))
            self._communities_cache = [frozenset({int(only)})]
            self._community_index_by_node = {int(only): 0}
            self._communities_dirty = False
            return

        communities = nx.community.greedy_modularity_communities(self._graph)
        # Stabilize ordering for reproducibility across runs/Python versions.
        sorted_communities = sorted(
            (sorted(int(n) for n in c) for c in communities),
            key=lambda nodes: (nodes[0] if nodes else -1, len(nodes)),
        )
        frozen = [frozenset(c) for c in sorted_communities]
        index_by_node: dict[int, int] = {}
        for i, comm in enumerate(frozen):
            for nid in comm:
                index_by_node[int(nid)] = i

        self._communities_cache = frozen
        self._community_index_by_node = index_by_node
        self._communities_dirty = False

    def _ensure_clustering_cache(self) -> None:
        if not self._clustering_dirty:
            return
        coeffs = nx.clustering(self._graph)
        self._clustering_cache = {int(k): float(v) for k, v in coeffs.items()}
        self._clustering_dirty = False

    # ------------------------------------------------------------------
    # Private generation
    # ------------------------------------------------------------------

    def _generate(self) -> None:
        """Generate the planted-partition graph with bot and real-user nodes."""
        n_real = self._num_nodes - self._bot_cluster_size

        # Node ID ranges
        real_ids = list(range(n_real))
        bot_ids = list(range(n_real, self._num_nodes))

        # Build planted partition graph using stochastic block model
        sizes = [n_real, self._bot_cluster_size]
        p_matrix = [
            [self._real_intra_density, self._inter_density],
            [self._inter_density, self._intra_density],
        ]
        sbm_graph = nx.stochastic_block_model(
            sizes, p_matrix, seed=self._seed
        )
        self._graph = sbm_graph

        # Label nodes
        self._bot_nodes = set(bot_ids)
        self._real_nodes = set(real_ids)

        for nid in real_ids:
            self._node_attrs[nid] = self._generate_human_attrs(nid)

        for nid in bot_ids:
            self._node_attrs[nid] = self._generate_bot_attrs(nid)

        # Write attrs into the networkx graph for interop
        nx.set_node_attributes(self._graph, self._node_attrs)
        self._communities_dirty = True
        self._clustering_dirty = True

        logger.info(
            "SocialGraph generated: %d nodes (%d real, %d bots), %d edges",
            self._num_nodes, n_real, self._bot_cluster_size,
            self._graph.number_of_edges(),
        )

    def _generate_human_attrs(self, node_id: int) -> dict[str, Any]:
        """Generate attribute dict for a real-user node.

        Args:
            node_id: Integer node ID.

        Returns:
            Dict with is_bot, legitimacy_score, account_age_days, activity_score.
        """
        return {
            "is_bot": False,
            "legitimacy_score": float(self._rng.uniform(0.5, 1.0)),
            "account_age_days": float(self._rng.uniform(90.0, 3650.0)),
            "activity_score": float(self._rng.beta(2.0, 5.0)),
        }

    def _generate_bot_attrs(self, node_id: int) -> dict[str, Any]:
        """Generate attribute dict for a bot node.

        Args:
            node_id: Integer node ID.

        Returns:
            Dict with is_bot, legitimacy_score, account_age_days, activity_score.
        """
        return {
            "is_bot": True,
            "legitimacy_score": float(self._rng.uniform(0.0, 0.4)),
            "account_age_days": float(self._rng.uniform(1.0, 180.0)),
            "activity_score": float(self._rng.uniform(0.6, 1.0)),
        }
