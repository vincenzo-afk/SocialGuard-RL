"""
data/synthetic_graph.py — Deterministic SocialGraph fixtures for tests.

This module exists primarily to provide a single, reusable helper for creating
repeatable graphs across tests and scripts.
"""

from __future__ import annotations

from typing import Any

from sim.social_graph import SocialGraph


def generate_fixture(graph_cfg: dict[str, Any], seed: int = 0) -> SocialGraph:
    """Generate a deterministic SocialGraph fixture from config and seed."""
    return SocialGraph(graph_cfg, seed=seed)

