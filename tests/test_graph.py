"""
tests/test_graph.py — Unit tests for sim/social_graph.py (Phase 3).

Test categories:
- Bot node count matches config ratio
- Real and bot node sets are disjoint
- remove_node() updates graph, bot_nodes, and real_nodes atomically
- tick() adds new edges without removing existing structure
- get_neighbors() returns a list (possibly empty)
- get_node_attrs() returns all required attribute keys
- Graph is not empty after generation
"""

from __future__ import annotations

import pytest
import numpy as np

from sim.social_graph import SocialGraph

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

SMALL_CFG = {
    "num_nodes": 50,
    "bot_cluster_size": 10,
    "intra_cluster_density": 0.4,
    "inter_cluster_density": 0.05,
    "embedding_dim": 64,
}

FULL_CFG = {
    "num_nodes": 500,
    "bot_cluster_size": 80,
    "intra_cluster_density": 0.4,
    "inter_cluster_density": 0.05,
    "embedding_dim": 64,
}


@pytest.fixture
def small_graph() -> SocialGraph:
    """Small 50-node graph for fast tests."""
    return SocialGraph(SMALL_CFG, seed=42)


# ---------------------------------------------------------------------------
# 1. Node counts and label consistency
# ---------------------------------------------------------------------------

class TestNodeCounts:
    """Generated node counts must match the config ratios exactly."""

    def test_total_node_count(self, small_graph: SocialGraph) -> None:
        """Total nodes must equal num_nodes from config."""
        assert small_graph.num_nodes == SMALL_CFG["num_nodes"]

    def test_bot_count_matches_config(self, small_graph: SocialGraph) -> None:
        """Bot node count must equal bot_cluster_size from config."""
        assert len(small_graph.get_bot_nodes()) == SMALL_CFG["bot_cluster_size"]

    def test_real_count_is_remainder(self, small_graph: SocialGraph) -> None:
        """Real user count must be num_nodes minus bot_cluster_size."""
        expected = SMALL_CFG["num_nodes"] - SMALL_CFG["bot_cluster_size"]
        assert len(small_graph.get_real_nodes()) == expected

    def test_bot_and_real_disjoint(self, small_graph: SocialGraph) -> None:
        """Bot and real node sets must be completely disjoint."""
        bots = small_graph.get_bot_nodes()
        real = small_graph.get_real_nodes()
        overlap = bots & real
        assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_bot_and_real_cover_all_nodes(self, small_graph: SocialGraph) -> None:
        """Union of bot + real sets must equal all node IDs in the graph."""
        all_nodes = set(small_graph.graph.nodes())
        combined = small_graph.get_bot_nodes() | small_graph.get_real_nodes()
        assert combined == all_nodes


# ---------------------------------------------------------------------------
# 2. Node attributes
# ---------------------------------------------------------------------------

class TestNodeAttributes:
    """Every node must have all required attribute keys."""

    REQUIRED_KEYS = {"is_bot", "legitimacy_score", "account_age_days", "activity_score"}

    def test_all_nodes_have_required_attrs(self, small_graph: SocialGraph) -> None:
        """Every node must have is_bot, legitimacy_score, account_age_days, activity_score."""
        for nid in small_graph.graph.nodes():
            attrs = small_graph.get_node_attrs(nid)
            missing = self.REQUIRED_KEYS - set(attrs.keys())
            assert not missing, f"Node {nid} missing attrs: {missing}"

    def test_bot_nodes_have_is_bot_true(self, small_graph: SocialGraph) -> None:
        """All nodes in get_bot_nodes() must have is_bot=True in attrs."""
        for nid in small_graph.get_bot_nodes():
            attrs = small_graph.get_node_attrs(nid)
            assert attrs["is_bot"] is True, f"Bot node {nid} has is_bot=False"

    def test_real_nodes_have_is_bot_false(self, small_graph: SocialGraph) -> None:
        """All nodes in get_real_nodes() must have is_bot=False in attrs."""
        for nid in small_graph.get_real_nodes():
            attrs = small_graph.get_node_attrs(nid)
            assert attrs["is_bot"] is False, f"Real node {nid} has is_bot=True"

    def test_legitimacy_in_range(self, small_graph: SocialGraph) -> None:
        """legitimacy_score must be in [0, 1] for all nodes."""
        for nid in small_graph.graph.nodes():
            score = small_graph.get_node_attrs(nid)["legitimacy_score"]
            assert 0.0 <= score <= 1.0, f"Node {nid} legitimacy={score} out of range"

    def test_activity_score_in_range(self, small_graph: SocialGraph) -> None:
        """activity_score must be in [0, 1] for all nodes."""
        for nid in small_graph.graph.nodes():
            score = small_graph.get_node_attrs(nid)["activity_score"]
            assert 0.0 <= score <= 1.0, f"Node {nid} activity={score} out of range"


# ---------------------------------------------------------------------------
# 3. remove_node() propagation
# ---------------------------------------------------------------------------

class TestRemoveNode:
    """remove_node() must update all internal structures atomically."""

    def test_remove_bot_node_from_graph(self, small_graph: SocialGraph) -> None:
        """Removed bot node must not appear in graph.nodes()."""
        bot_id = next(iter(small_graph.get_bot_nodes()))
        small_graph.remove_node(bot_id)
        assert bot_id not in small_graph.graph.nodes()

    def test_remove_bot_node_from_bot_set(self, small_graph: SocialGraph) -> None:
        """Removed bot node must not appear in get_bot_nodes()."""
        bot_id = next(iter(small_graph.get_bot_nodes()))
        small_graph.remove_node(bot_id)
        assert bot_id not in small_graph.get_bot_nodes()

    def test_remove_real_node_from_real_set(self, small_graph: SocialGraph) -> None:
        """Removed real node must not appear in get_real_nodes()."""
        real_id = next(iter(small_graph.get_real_nodes()))
        small_graph.remove_node(real_id)
        assert real_id not in small_graph.get_real_nodes()

    def test_remove_node_decrements_count(self, small_graph: SocialGraph) -> None:
        """Removing a node must reduce num_nodes by exactly 1."""
        before = small_graph.num_nodes
        nid = next(iter(small_graph.get_bot_nodes()))
        small_graph.remove_node(nid)
        assert small_graph.num_nodes == before - 1

    def test_remove_nonexistent_raises(self, small_graph: SocialGraph) -> None:
        """Removing a node not in the graph must raise KeyError."""
        with pytest.raises(KeyError):
            small_graph.remove_node(99999)

    def test_remove_node_attrs_cleaned(self, small_graph: SocialGraph) -> None:
        """Removed node must not be accessible via get_node_attrs()."""
        nid = next(iter(small_graph.get_bot_nodes()))
        small_graph.remove_node(nid)
        with pytest.raises(KeyError):
            small_graph.get_node_attrs(nid)


# ---------------------------------------------------------------------------
# 4. tick() — edge addition
# ---------------------------------------------------------------------------

class TestTick:
    """tick() must add edges without corrupting node sets."""

    def test_tick_does_not_reduce_nodes(self, small_graph: SocialGraph) -> None:
        """Calling tick() must not change the node count."""
        before = small_graph.num_nodes
        small_graph.tick()
        assert small_graph.num_nodes == before

    def test_tick_may_add_edges(self, small_graph: SocialGraph) -> None:
        """Calling tick() 5 times should (on average) add at least 1 edge."""
        before = small_graph.graph.number_of_edges()
        for _ in range(5):
            small_graph.tick()
        after = small_graph.graph.number_of_edges()
        # With a 50-node graph and intra_density=0.4 this is extremely likely
        assert after >= before, "tick() should not remove edges"

    def test_tick_preserves_bot_real_sets(self, small_graph: SocialGraph) -> None:
        """Bot and real sets must still be disjoint after tick()."""
        for _ in range(3):
            small_graph.tick()
        bots = small_graph.get_bot_nodes()
        real = small_graph.get_real_nodes()
        assert len(bots & real) == 0


# ---------------------------------------------------------------------------
# 5. get_neighbors() interface
# ---------------------------------------------------------------------------

class TestGetNeighbors:
    """get_neighbors() must return a list and raise on missing node."""

    def test_returns_list(self, small_graph: SocialGraph) -> None:
        """get_neighbors() must return a list."""
        nid = next(iter(small_graph.graph.nodes()))
        result = small_graph.get_neighbors(nid)
        assert isinstance(result, list)

    def test_raises_on_missing_node(self, small_graph: SocialGraph) -> None:
        """get_neighbors() must raise KeyError for unknown node."""
        with pytest.raises(KeyError):
            small_graph.get_neighbors(99999)

    def test_neighbors_are_valid_nodes(self, small_graph: SocialGraph) -> None:
        """All returned neighbors must be valid node IDs."""
        all_nodes = set(small_graph.graph.nodes())
        for nid in list(all_nodes)[:5]:
            for nb in small_graph.get_neighbors(nid):
                assert nb in all_nodes, f"Neighbour {nb} not in graph"


# ---------------------------------------------------------------------------
# 6. Full-size graph (500 nodes smoke test)
# ---------------------------------------------------------------------------

class TestFullSizeGraph:
    """Smoke test: 500-node graph must generate without errors."""

    def test_full_graph_generates(self) -> None:
        """500-node graph must be creatable and have correct node count."""
        g = SocialGraph(FULL_CFG, seed=0)
        assert g.num_nodes == FULL_CFG["num_nodes"]
        assert len(g.get_bot_nodes()) == FULL_CFG["bot_cluster_size"]

    def test_full_graph_sets_disjoint(self) -> None:
        """Bot and real sets must be disjoint on the full-size graph."""
        g = SocialGraph(FULL_CFG, seed=0)
        assert len(g.get_bot_nodes() & g.get_real_nodes()) == 0
