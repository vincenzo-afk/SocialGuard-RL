"""
env/spaces.py — Observation and action space definitions for NEMESIS-RL.

All feature name constants and space dimensions are defined here.
Every other module imports from this file — no magic numbers elsewhere.

Observation vector version: v1.0
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Observation vector version — bump when the shape changes
# ---------------------------------------------------------------------------
OBS_VECTOR_VERSION: str = "1.0"

# ---------------------------------------------------------------------------
# Maximum observation size (must match the largest task — Task 3: 68)
# All tasks pad shorter vectors to this length with zeros.
# ---------------------------------------------------------------------------
OBS_DIM: int = 68

# ---------------------------------------------------------------------------
# Task 1 — Spam Detection (8 features, indices 0–7)
# ---------------------------------------------------------------------------
TASK1_FEATURE_NAMES: list[str] = [
    "account_age_days",           # 0  — 0 to 3650
    "posts_per_hour",             # 1  — 0 to 200
    "follower_ratio",             # 2  — 0 to 1
    "login_time_variance",        # 3  — 0 to 1
    "content_repetition_score",   # 4  — 0 to 1
    "profile_completeness",       # 5  — 0 to 1
    "device_fingerprint_uniqueness",  # 6  — 0 to 1
    "ip_diversity_score",         # 7  — 0 to 1
]
TASK1_OBS_DIM: int = len(TASK1_FEATURE_NAMES)  # 8

# Feature index constants for programmatic access
IDX_ACCOUNT_AGE = 0
IDX_POSTS_PER_HOUR = 1
IDX_FOLLOWER_RATIO = 2
IDX_LOGIN_TIME_VARIANCE = 3
IDX_CONTENT_REPETITION = 4
IDX_PROFILE_COMPLETENESS = 5
IDX_DEVICE_FINGERPRINT = 6
IDX_IP_DIVERSITY = 7

# ---------------------------------------------------------------------------
# Task 2 — Misinformation Flagging (6 features, indices 0–5)
# ---------------------------------------------------------------------------
TASK2_FEATURE_NAMES: list[str] = [
    "spread_rate",           # 0  — 0 to 1
    "fact_check_flag",       # 1  — 0 or 1
    "engagement_ratio",      # 2  — 0 to 1
    "source_credibility",    # 3  — 0 to 1
    "hop_count",             # 4  — 0 to 20 (normalized to 0–1)
    "timestep_normalized",   # 5  — 0 to 1
]
TASK2_OBS_DIM: int = len(TASK2_FEATURE_NAMES)  # 6

IDX_SPREAD_RATE = 0
IDX_FACT_CHECK_FLAG = 1
IDX_ENGAGEMENT_RATIO = 2
IDX_SOURCE_CREDIBILITY = 3
IDX_HOP_COUNT = 4
IDX_TIMESTEP_NORMALIZED = 5

# ---------------------------------------------------------------------------
# Task 3 — CIB Network Takedown (68 features, indices 0–67)
# ---------------------------------------------------------------------------
TASK3_EMBEDDING_DIM: int = 64   # node2vec embedding dimensions
TASK3_FEATURE_NAMES: list[str] = (
    [f"embedding_{i}" for i in range(TASK3_EMBEDDING_DIM)]  # 0–63
    + [
        "degree_centrality",        # 64 — 0 to 1
        "clustering_coefficient",   # 65 — 0 to 1
        "community_assignment",     # 66 — 0 to 1 (normalized)
        "posts_per_hour_normalized",# 67 — 0 to 1
    ]
)
TASK3_OBS_DIM: int = len(TASK3_FEATURE_NAMES)  # 68

IDX_DEGREE_CENTRALITY = 64
IDX_CLUSTERING_COEFF = 65
IDX_COMMUNITY_ASSIGNMENT = 66
IDX_PPH_NORMALIZED = 67

# ---------------------------------------------------------------------------
# Action space — shared across all tasks (task config restricts valid IDs)
# ---------------------------------------------------------------------------
ACTION_ALLOW: int = 0
ACTION_WARN: int = 1
ACTION_REDUCE_REACH: int = 2
ACTION_REMOVE: int = 3
ACTION_ESCALATE: int = 4

ACTION_NAMES: dict[int, str] = {
    ACTION_ALLOW: "allow",
    ACTION_WARN: "warn",
    ACTION_REDUCE_REACH: "reduce_reach",
    ACTION_REMOVE: "remove",
    ACTION_ESCALATE: "escalate",
}
N_ACTIONS: int = 5

# Convenience lists — per-task valid action subsets (loaded into task config)
TASK1_ACTIONS: list[int] = [ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE]
TASK2_ACTIONS: list[int] = [ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE, ACTION_ESCALATE]
TASK3_ACTIONS: list[int] = [ACTION_ALLOW, ACTION_WARN, ACTION_REDUCE_REACH, ACTION_REMOVE, ACTION_ESCALATE]

# ---------------------------------------------------------------------------
# Observation bounds — per feature
# All tasks share the same Box; unused dimensions are clipped to [0, 0].
# ---------------------------------------------------------------------------
_LOW = np.zeros(OBS_DIM, dtype=np.float32)
_HIGH = np.ones(OBS_DIM, dtype=np.float32)

# Task 3 embeddings are in [-1, 1]
_LOW[:TASK3_EMBEDDING_DIM] = -1.0


class ObservationSpace(spaces.Box):
    """Unified observation space for all NEMESIS-RL tasks.

    Shape is always (OBS_DIM,) = (68,).  Tasks with fewer features pad the
    remaining dimensions with zeros.  This guarantees the SB3 policy network
    sees a fixed-size input regardless of the active task.

    Args:
        dtype: NumPy dtype for the observation array.  Defaults to float32.
    """

    def __init__(self, dtype: np.dtype = np.float32) -> None:
        """Initialise the unified observation Box."""
        super().__init__(
            low=_LOW.copy(),
            high=_HIGH.copy(),
            shape=(OBS_DIM,),
            dtype=dtype,
        )


class ActionSpace(spaces.Discrete):
    """Discrete action space with N_ACTIONS = 5.

    The environment validates that the chosen action ID is in the allowed
    subset defined by the active task's config before processing it.
    """

    def __init__(self) -> None:
        """Initialise the Discrete(5) action space."""
        super().__init__(N_ACTIONS)


def pad_observation(features: np.ndarray) -> np.ndarray:
    """Pad a task-specific feature vector to the full OBS_DIM length.

    Args:
        features: 1-D numpy array with length <= OBS_DIM.

    Returns:
        Float32 array of shape (OBS_DIM,) with trailing zeros if needed.

    Raises:
        ValueError: If features is longer than OBS_DIM.
    """
    if len(features) > OBS_DIM:
        raise ValueError(
            f"Feature vector length {len(features)} exceeds OBS_DIM={OBS_DIM}."
        )
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[: len(features)] = features
    return obs
