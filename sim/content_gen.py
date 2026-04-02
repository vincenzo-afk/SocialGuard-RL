"""
sim/content_gen.py — Content generation and BFS spread simulation.

Simulates a single piece of content spreading through the social graph via
breadth-first diffusion.  Each tick() advances spread by one hop layer.

Post contract (get_content_observation returns a 6-feature vector):
    spread_rate, fact_check_flag, engagement_ratio,
    source_credibility, hop_count (normalised), timestep_normalised
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from typing import Any

import numpy as np

from sim.social_graph import SocialGraph

logger = logging.getLogger(__name__)


class Post:
    """Immutable value object representing a single piece of content.

    Args:
        author_id: Node ID of the originating account.
        is_misinfo: True if the post contains misinformation.
        spread_rate: Float [0, 1] — how aggressively the content spreads.
        credibility_score: Float [0, 1] — source credibility signal.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        author_id: int,
        is_misinfo: bool,
        spread_rate: float,
        credibility_score: float,
        seed: int | None = None,
    ) -> None:
        """Initialise a post with generated metadata."""
        self.post_id: str = str(uuid.uuid4())[:8]
        self.author_id: int = author_id
        self.timestamp: int = 0   # set by ContentEngine at creation
        self.is_misinfo: bool = is_misinfo
        self.spread_rate: float = float(np.clip(spread_rate, 0.0, 1.0))
        self.credibility_score: float = float(np.clip(credibility_score, 0.0, 1.0))
        self.hop_count: int = 0
        self.total_reach: int = 1   # starts at 1 (only the author)
        self.fact_check_flag: float = 0.0


class ContentEngine:
    """BFS content spread simulator tied to a social graph.

    At each tick(), the content spreads one hop further through the graph.
    Engagement and reach are tracked and exposed as a feature vector.

    Args:
        graph: SocialGraph instance the content spreads through.
        content_cfg: Dict with spread parameters (currently uses graph edges).
        max_hops: Maximum number of hops before content is considered fully spread.
        seed: Optional integer seed.
    """

    def __init__(
        self,
        graph: SocialGraph,
        content_cfg: dict[str, Any],
        max_hops: int = 20,
        seed: int | None = None,
    ) -> None:
        """Initialise the content engine."""
        self._graph: SocialGraph = graph
        self._max_hops: int = max_hops
        self._rng: np.random.RandomState = np.random.RandomState(seed)
        self._noise_level: float = float(content_cfg.get("noise_level", 0.0))

        # Episode state — set in reset()
        self._post: Post | None = None
        self._visited: set[int] = set()
        self._current_frontier: list[int] = []
        self._next_frontier: list[int] = []
        self._timestep: int = 0
        self._max_timesteps: int = int(content_cfg.get("max_steps", 100))
        self._removed: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        author_id: int | None = None,
        is_misinfo: bool | None = None,
        seed: int | None = None,
    ) -> Post:
        """Reset and create a new piece of content beginning to spread.

        Args:
            author_id: Starting node.  Randomly chosen if None.
            is_misinfo: Whether the post is misinformation.  Random if None.
            seed: Optional RNG seed.

        Returns:
            The newly created Post object.
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        nodes = list(self._graph.graph.nodes())
        if not nodes:
            raise RuntimeError("Graph has no nodes — cannot create content.")

        if author_id is None:
            author_id = int(self._rng.choice(nodes))

        if is_misinfo is None:
            is_misinfo = bool(self._rng.random() < 0.5)

        # Misinfo: high spread rate, low credibility
        if is_misinfo:
            spread_rate = float(self._rng.uniform(0.5, 0.9))
            credibility_score = float(self._rng.uniform(0.0, 0.4))
        else:
            spread_rate = float(self._rng.uniform(0.1, 0.5))
            credibility_score = float(self._rng.uniform(0.5, 1.0))

        if self._noise_level > 0.0:
            spread_rate = float(np.clip(spread_rate + self._rng.normal(0.0, self._noise_level * 0.05), 0.0, 1.0))
            credibility_score = float(
                np.clip(credibility_score + self._rng.normal(0.0, self._noise_level * 0.05), 0.0, 1.0)
            )

        self._post = Post(
            author_id=author_id,
            is_misinfo=is_misinfo,
            spread_rate=spread_rate,
            credibility_score=credibility_score,
        )
        self._post.timestamp = 0
        self._visited = {author_id}
        self._current_frontier = [author_id]
        self._next_frontier = []
        self._timestep = 0
        self._removed = False

        # Fact-check flag: misinfo has a probability of being flagged
        if is_misinfo:
            self._post.fact_check_flag = float(int(self._rng.random() < 0.6))

        logger.debug(
            "ContentEngine.reset(): author=%d, misinfo=%s, spread=%.2f",
            author_id, is_misinfo, spread_rate,
        )
        return self._post

    def tick(self) -> None:
        """Advance content spread by one BFS hop.

        Each neighbour of the current frontier is added to the next frontier
        with probability equal to the post's spread_rate.  Hop count
        increments by one per tick.
        """
        if self._post is None or self._removed:
            return
        if self._post.hop_count >= self._max_hops:
            return

        self._next_frontier = []
        for node in self._current_frontier:
            for neighbour in self._graph.get_neighbors(node):
                if neighbour not in self._visited:
                    local_spread = float(
                        np.clip(
                            self._post.spread_rate + self._rng.normal(0.0, self._noise_level * 0.02),
                            0.0,
                            1.0,
                        )
                    )
                    if self._rng.random() < local_spread:
                        self._visited.add(neighbour)
                        self._next_frontier.append(neighbour)

        if self._next_frontier:
            self._current_frontier = self._next_frontier
            self._post.hop_count += 1
            self._post.total_reach += len(self._next_frontier)

        self._timestep += 1
        logger.debug(
            "ContentEngine.tick(): hop=%d reach=%d frontier=%d",
            self._post.hop_count,
            self._post.total_reach,
            len(self._current_frontier),
        )

    def remove_content(self) -> None:
        """Mark the content as removed — spread stops immediately."""
        self._removed = True
        logger.debug("ContentEngine: content %s removed.", self._post.post_id if self._post else "None")

    def is_spread_done(self) -> bool:
        """Return True when spread has reached max hops or content is removed.

        Returns:
            True if the episode should end.
        """
        if self._removed:
            return True
        if self._post is None:
            return True
        if self._post.hop_count >= self._max_hops:
            return True
        if self._timestep >= self._max_timesteps:
            return True
        # No new nodes reachable
        if not self._current_frontier:
            return True
        return False

    def get_content_observation(self) -> np.ndarray:
        """Return the 6-feature content observation vector (Task 2 spec).

        Returns:
            Float32 numpy array [spread_rate, fact_check_flag,
            engagement_ratio, source_credibility, hop_count_norm,
            timestep_norm].
        """
        if self._post is None:
            return np.zeros(6, dtype=np.float32)

        hop_norm = min(self._post.hop_count / self._max_hops, 1.0)
        ts_norm = min(self._timestep / max(self._max_timesteps, 1), 1.0)

        total_possible = max(self._graph.num_nodes, 1)
        engagement_ratio = min(self._post.total_reach / total_possible, 1.0)

        obs = np.array([
            self._post.spread_rate,
            self._post.fact_check_flag,
            engagement_ratio,
            self._post.credibility_score,
            hop_norm,
            ts_norm,
        ], dtype=np.float32)
        return obs

    def get_current_hop(self) -> int:
        """Return the current hop count of the active post."""
        if self._post is None:
            return 0
        return self._post.hop_count

    def get_post_info(self) -> dict[str, Any]:
        """Return a plain Python dict summary of the active post state."""
        if self._post is None:
            return {}
        return {
            "post_id": self._post.post_id,
            "author_id": self._post.author_id,
            "is_misinfo": self._post.is_misinfo,
            "hop_count": self._post.hop_count,
            "total_reach": self._post.total_reach,
            "spread_rate": float(self._post.spread_rate),
            "credibility_score": float(self._post.credibility_score),
            "fact_check_flag": float(self._post.fact_check_flag),
            "removed": self._removed,
            "timestep": self._timestep,
        }
