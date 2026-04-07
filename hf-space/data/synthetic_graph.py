"""
data/synthetic_graph.py — Deterministic graph fixtures for tests.

Provides factory functions that generate reproducible, seeded NetworkX graphs
for unit tests and smoke tests.  These are NOT used in production — only in tests/.

All functions accept a `seed` argument (default 42) and always return the same
graph for the same seed, regardless of platform or numpy version.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Small deterministic graphs for fast unit tests
# ---------------------------------------------------------------------------

def make_spam_accounts(
    n_accounts: int = 20,
    bot_ratio: float = 0.30,
    noise_level: float = 0.10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Return a list of synthetic account feature dicts for Task 1 (Spam Detection).

    Each dict mirrors the fields used by TaskSpam, with deterministic values.

    Args:
        n_accounts: Total number of accounts.
        bot_ratio: Fraction that should be bots.
        noise_level: Amount of Gaussian noise added to features.
        seed: RNG seed for reproducibility.

    Returns:
        List of account dicts, each with keys:
            is_bot, account_age_days, posts_per_hour, follower_ratio,
            login_time_variance, content_repetition_score,
            profile_completeness, device_fingerprint_uniqueness, ip_diversity_score.
    """
    rng = np.random.RandomState(seed)
    n_bots = max(1, int(n_accounts * bot_ratio))
    accounts: list[dict[str, Any]] = []

    for i in range(n_accounts):
        is_bot = i < n_bots
        base = {
            "is_bot": int(is_bot),
            "account_age_days": rng.uniform(0, 30 if is_bot else 3650),
            "posts_per_hour": rng.uniform(50, 200) if is_bot else rng.uniform(0, 10),
            "follower_ratio": rng.uniform(0, 0.2) if is_bot else rng.uniform(0.3, 0.9),
            "login_time_variance": rng.uniform(0, 0.2) if is_bot else rng.uniform(0.4, 1.0),
            "content_repetition_score": rng.uniform(0.6, 1.0) if is_bot else rng.uniform(0, 0.3),
            "profile_completeness": rng.uniform(0, 0.3) if is_bot else rng.uniform(0.5, 1.0),
            "device_fingerprint_uniqueness": rng.uniform(0, 0.2) if is_bot else rng.uniform(0.5, 1.0),
            "ip_diversity_score": rng.uniform(0, 0.2) if is_bot else rng.uniform(0.4, 1.0),
        }
        # Add noise to blur the boundary
        for key in list(base.keys()):
            if key != "is_bot":
                base[key] = float(np.clip(base[key] + rng.normal(0, noise_level), 0.0, 1.0 if key != "account_age_days" and key != "posts_per_hour" else None))
        accounts.append(base)

    rng.shuffle(accounts)
    return accounts


def make_small_planted_graph(
    num_nodes: int = 30,
    bot_cluster_size: int = 8,
    intra_density: float = 0.5,
    inter_density: float = 0.05,
    seed: int = 42,
) -> nx.Graph:
    """Return a small planted-partition graph for Task 3 unit tests.

    Args:
        num_nodes: Total number of nodes.
        bot_cluster_size: Size of the bot cluster.
        intra_density: Edge probability within the bot cluster.
        inter_density: Edge probability between clusters.
        seed: RNG seed.

    Returns:
        NetworkX Graph with `is_bot` and `legitimacy_score` node attributes.
    """
    real_size = num_nodes - bot_cluster_size
    sizes = [bot_cluster_size, real_size]
    probs = [
        [intra_density, inter_density],
        [inter_density, 0.1],
    ]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)

    bot_nodes = set(range(bot_cluster_size))
    for node in G.nodes():
        is_bot = node in bot_nodes
        G.nodes[node]["is_bot"] = is_bot
        G.nodes[node]["legitimacy_score"] = 0.1 if is_bot else 0.85
        G.nodes[node]["posts_per_hour"] = 120.0 if is_bot else 2.0

    return G


def make_misinfo_graph(
    num_nodes: int = 50,
    source_node: int = 0,
    seed: int = 42,
) -> nx.Graph:
    """Return a small scale-free graph for Task 2 (Misinformation) tests.

    Args:
        num_nodes: Number of nodes.
        source_node: Node from which misinformation originates.
        seed: RNG seed.

    Returns:
        NetworkX Graph with a designated source node marked.
    """
    G = nx.barabasi_albert_graph(num_nodes, m=2, seed=seed)
    for node in G.nodes():
        G.nodes[node]["is_source"] = node == source_node
        G.nodes[node]["credibility"] = 0.2 if node == source_node else 0.8
    return G


def make_cib_observation_batch(
    n_nodes: int = 10,
    embedding_dim: int = 64,
    bot_ratio: float = 0.4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a batch of 68-dim observation vectors for Task 3 tests.

    Args:
        n_nodes: Number of nodes.
        embedding_dim: Embedding dimensionality (first embedding_dim features).
        bot_ratio: Fraction that are bots.
        seed: RNG seed.

    Returns:
        Tuple of (observations, ground_truths):
            - observations: float32 ndarray of shape (n_nodes, 68)
            - ground_truths: int ndarray of shape (n_nodes,) with values 0/1
    """
    rng = np.random.RandomState(seed)
    obs = rng.uniform(-1, 1, size=(n_nodes, 68)).astype(np.float32)
    # Zero-out padding (indices embedding_dim..67 unused in most tests)
    obs[:, embedding_dim:] = 0.0

    n_bots = max(1, int(n_nodes * bot_ratio))
    gt = np.zeros(n_nodes, dtype=np.int32)
    gt[:n_bots] = 1
    rng.shuffle(gt)
    return obs, gt
