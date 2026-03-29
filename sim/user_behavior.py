"""
sim/user_behavior.py — Human and Bot behavior models for SocialGuard-RL.

Both classes produce a structured feature dict whose keys match the Task 1
observation vector defined in env/spaces.py.  The dict is consumed by
tasks/task_spam.py to build the numpy observation array.

Design notes:
- `BaseBehavior` defines the interface; concrete classes override `generate()`.
- Noise is intentional: bots are NOT perfectly separable from humans.
  The `noise_level` config parameter controls the signal overlap.
- All random state is seeded and self-contained — no module-level globals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature dict key constants — match TASK1_FEATURE_NAMES in env/spaces.py
# ---------------------------------------------------------------------------
F_ACCOUNT_AGE = "account_age_days"
F_POSTS_PER_HOUR = "posts_per_hour"
F_FOLLOWER_RATIO = "follower_ratio"
F_LOGIN_TIME_VARIANCE = "login_time_variance"
F_CONTENT_REPETITION = "content_repetition_score"
F_PROFILE_COMPLETENESS = "profile_completeness"
F_DEVICE_FINGERPRINT = "device_fingerprint_uniqueness"
F_IP_DIVERSITY = "ip_diversity_score"

FEATURE_KEYS: list[str] = [
    F_ACCOUNT_AGE,
    F_POSTS_PER_HOUR,
    F_FOLLOWER_RATIO,
    F_LOGIN_TIME_VARIANCE,
    F_CONTENT_REPETITION,
    F_PROFILE_COMPLETENESS,
    F_DEVICE_FINGERPRINT,
    F_IP_DIVERSITY,
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseBehavior(ABC):
    """Abstract base for all behavior models.

    Args:
        noise_level: Float in [0, 1].  Higher values add more Gaussian noise
            to the generated features, making bot/human signals harder to
            distinguish.  Read from task config — never hardcoded.
        rng: Optional numpy RandomState.  If None, a new one is created.
    """

    def __init__(self, noise_level: float, rng: np.random.RandomState | None = None) -> None:
        """Initialise the behavior model with a noise level and RNG."""
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError(f"noise_level must be in [0, 1]; got {noise_level}")
        self._noise: float = noise_level
        self._rng: np.random.RandomState = rng if rng is not None else np.random.RandomState()

    @property
    def is_bot(self) -> bool:
        """Return True if this behavior represents a bot account."""
        raise NotImplementedError

    @abstractmethod
    def generate(self) -> dict[str, float]:
        """Generate one observation feature dict for a single account step.

        Returns:
            Dict mapping each feature key (from FEATURE_KEYS) to a float value
            matching the range contract in SOCIALGUARD_RL_PLAN.md Section 5.
        """

    def _add_noise(self, value: float, scale: float, low: float = 0.0, high: float = 1.0) -> float:
        """Add Gaussian noise to a value and clip to [low, high].

        Args:
            value: Base value before noise.
            scale: Standard deviation of the noise distribution, scaled by
                self._noise so the config noise_level controls everything.
            low: Minimum allowed value post-clip.
            high: Maximum allowed value post-clip.

        Returns:
            Noisy value clipped to [low, high].
        """
        noisy = value + self._rng.normal(0.0, scale * self._noise)
        return float(np.clip(noisy, low, high))

    def _validate_features(self, features: dict[str, float]) -> None:
        """Assert all required feature keys are present (debug aid).

        Args:
            features: Feature dict to validate.

        Raises:
            AssertionError: If any required key is missing.
        """
        missing = [k for k in FEATURE_KEYS if k not in features]
        assert not missing, f"Missing feature keys: {missing}"


# ---------------------------------------------------------------------------
# Human behavior
# ---------------------------------------------------------------------------

class HumanBehavior(BaseBehavior):
    """Simulates a legitimate human user's behavioral fingerprint.

    Human accounts are characterised by:
    - Older account age (right-skewed distribution)
    - Moderate, irregular posting frequency (normal distribution)
    - Balanced follower/following ratio
    - High login time variance (log in at varied hours)
    - Low content repetition (varied posts)
    - High profile completeness
    - High device fingerprint uniqueness (real devices)
    - Moderate IP diversity (home + work + mobile)

    Args:
        noise_level: Config-driven noise to create signal overlap with bots.
        rng: Optional seeded RNG for reproducibility.
    """

    def __init__(self, noise_level: float, rng: np.random.RandomState | None = None) -> None:
        """Initialise a human behavior model."""
        super().__init__(noise_level, rng)

    @property
    def is_bot(self) -> bool:
        """Return False — this is a human behavior model."""
        return False

    def generate(self) -> dict[str, float]:
        """Generate one feature dict representing a real human account.

        Returns:
            Dict with 8 float features matching the Task 1 obs spec.
        """
        # Account age: right-skewed — most real users have been around a while
        # Mean ~730 days (2 years), std ~400, clipped to [0, 3650]
        account_age = float(np.clip(self._rng.normal(730.0, 400.0), 0.0, 3650.0))

        # Posts per hour: moderate, irregular (mean ~3, std ~2, min 0)
        posts_per_hour = self._add_noise(
            self._rng.exponential(3.0), scale=2.0, low=0.0, high=200.0
        )

        # Follower ratio: real users have balanced ratios [0.3, 0.9]
        follower_ratio = self._add_noise(
            self._rng.uniform(0.3, 0.9), scale=0.15
        )

        # Login time variance: humans log in at varied hours → high variance
        login_time_variance = self._add_noise(
            self._rng.uniform(0.5, 1.0), scale=0.15
        )

        # Content repetition: real users rarely repeat content verbatim → low
        content_repetition = self._add_noise(
            self._rng.beta(1.5, 6.0), scale=0.1
        )

        # Profile completeness: real users often fill profiles → high
        profile_completeness = self._add_noise(
            self._rng.uniform(0.6, 1.0), scale=0.15
        )

        # Device fingerprint uniqueness: real devices → high
        device_fingerprint = self._add_noise(
            self._rng.uniform(0.6, 1.0), scale=0.15
        )

        # IP diversity: home + work + mobile → moderate
        ip_diversity = self._add_noise(
            self._rng.uniform(0.3, 0.7), scale=0.12
        )

        features = {
            F_ACCOUNT_AGE: account_age,
            F_POSTS_PER_HOUR: posts_per_hour,
            F_FOLLOWER_RATIO: follower_ratio,
            F_LOGIN_TIME_VARIANCE: login_time_variance,
            F_CONTENT_REPETITION: content_repetition,
            F_PROFILE_COMPLETENESS: profile_completeness,
            F_DEVICE_FINGERPRINT: device_fingerprint,
            F_IP_DIVERSITY: ip_diversity,
        }
        self._validate_features(features)
        logger.debug("HumanBehavior.generate() → %s", features)
        return features


# ---------------------------------------------------------------------------
# Bot behavior
# ---------------------------------------------------------------------------

class BotBehavior(BaseBehavior):
    """Simulates an automated bot account's behavioral fingerprint.

    Bot accounts are characterised by:
    - New account age (created recently for a campaign)
    - Very high posting frequency in tight synchronized windows
    - Low follower ratio (many following, few followers)
    - Low login time variance (scripted, posts at fixed intervals)
    - High content repetition (copy-paste or template-based posts)
    - Low profile completeness (fake accounts are sparse)
    - Low device fingerprint uniqueness (many bots share infrastructure)
    - Low IP diversity (all traffic from one or a few datacenter IPs)

    Intentional signal overlap via noise_level ensures this is not trivially
    solvable — the rule-based baseline should achieve ~65–75% F1, not 95%+.

    Args:
        noise_level: Config-driven noise to create overlap with human signals.
        rng: Optional seeded RNG for reproducibility.
    """

    def __init__(self, noise_level: float, rng: np.random.RandomState | None = None) -> None:
        """Initialise a bot behavior model."""
        super().__init__(noise_level, rng)

    @property
    def is_bot(self) -> bool:
        """Return True — this is a bot behavior model."""
        return True

    def generate(self) -> dict[str, float]:
        """Generate one feature dict representing a bot account.

        Returns:
            Dict with 8 float features matching the Task 1 obs spec.
        """
        # Account age: bots are usually new — exponential with mean ~60 days
        account_age = float(np.clip(self._rng.exponential(60.0), 0.0, 3650.0))

        # Posts per hour: very high, synchronized bursts [20, 200]
        posts_per_hour = self._add_noise(
            self._rng.uniform(20.0, 150.0), scale=10.0, low=0.0, high=200.0
        )

        # Follower ratio: low — bots follow many but are followed by few
        follower_ratio = self._add_noise(
            self._rng.beta(1.5, 8.0), scale=0.1
        )

        # Login time variance: low — scripted posting at fixed intervals
        login_time_variance = self._add_noise(
            self._rng.beta(2.0, 8.0), scale=0.1
        )

        # Content repetition: high — template or copy-paste posts
        content_repetition = self._add_noise(
            self._rng.uniform(0.6, 1.0), scale=0.12
        )

        # Profile completeness: low — fake accounts are sparse
        profile_completeness = self._add_noise(
            self._rng.beta(1.5, 6.0), scale=0.1
        )

        # Device fingerprint uniqueness: low — shared datacenter infra
        device_fingerprint = self._add_noise(
            self._rng.beta(1.5, 8.0), scale=0.1
        )

        # IP diversity: very low — all from few datacenter IPs
        ip_diversity = self._add_noise(
            self._rng.beta(1.0, 8.0), scale=0.08
        )

        features = {
            F_ACCOUNT_AGE: account_age,
            F_POSTS_PER_HOUR: posts_per_hour,
            F_FOLLOWER_RATIO: follower_ratio,
            F_LOGIN_TIME_VARIANCE: login_time_variance,
            F_CONTENT_REPETITION: content_repetition,
            F_PROFILE_COMPLETENESS: profile_completeness,
            F_DEVICE_FINGERPRINT: device_fingerprint,
            F_IP_DIVERSITY: ip_diversity,
        }
        self._validate_features(features)
        logger.debug("BotBehavior.generate() → %s", features)
        return features
