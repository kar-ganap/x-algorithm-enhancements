"""Reward weights for F4 reward modeling.

Defines the RewardWeights dataclass that holds learnable weights
for combining Phoenix's 18 action probabilities into a scalar reward.

Action order (from Phoenix RankingOutput):
    0: favorite
    1: reply
    2: repost
    3: photo_expand
    4: click
    5: profile_click
    6: vqv
    7: share
    8: share_via_dm
    9: share_via_copy_link
    10: dwell
    11: quote
    12: quoted_click
    13: follow_author
    14: not_interested
    15: block_author
    16: mute_author
    17: report
"""

from dataclasses import dataclass

import jax.numpy as jnp

# Action names in Phoenix order
ACTION_NAMES: list[str] = [
    "favorite",
    "reply",
    "repost",
    "photo_expand",
    "click",
    "profile_click",
    "vqv",
    "share",
    "share_via_dm",
    "share_via_copy_link",
    "dwell",
    "quote",
    "quoted_click",
    "follow_author",
    "not_interested",
    "block_author",
    "mute_author",
    "report",
]

NUM_ACTIONS = len(ACTION_NAMES)  # 18

# Indices for key actions (for interpretability checks)
ACTION_INDICES: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}


@dataclass
class RewardWeights:
    """Learnable reward weights for combining action probabilities.

    The reward is computed as: R = w · P(actions)
    where w is this weight vector and P(actions) are Phoenix's
    predicted probabilities for each action type.

    Attributes:
        weights: Array of shape [18] with one weight per action.
    """

    weights: jnp.ndarray  # Shape: [18]

    def __post_init__(self):
        """Validate weights shape."""
        if self.weights.shape != (NUM_ACTIONS,):
            raise ValueError(
                f"Expected weights shape ({NUM_ACTIONS},), got {self.weights.shape}"
            )

    @classmethod
    def default(cls) -> "RewardWeights":
        """Create weights with sensible defaults.

        Positive weights for engagement actions (favorite, repost, follow, etc.)
        Negative weights for negative actions (block, mute, report, not_interested)

        These are hand-tuned starting values, not learned.
        """
        weights = jnp.array([
            +1.0,   # favorite - strong positive signal
            +0.5,   # reply - engagement, but can be negative replies
            +0.8,   # repost - strong endorsement
            +0.1,   # photo_expand - mild interest
            +0.2,   # click - interest
            +0.3,   # profile_click - interest in author
            +0.1,   # vqv - video quality view
            +0.6,   # share - strong endorsement
            +0.4,   # share_via_dm - personal recommendation
            +0.3,   # share_via_copy_link - sharing intent
            +0.1,   # dwell - attention (could be hate-reading)
            +0.7,   # quote - engagement with commentary
            +0.2,   # quoted_click - interest in quoted content
            +1.2,   # follow_author - very strong positive
            -0.5,   # not_interested - explicit negative
            -1.5,   # block_author - strong negative
            -1.0,   # mute_author - negative
            -2.0,   # report - strongest negative signal
        ])
        return cls(weights=weights)

    @classmethod
    def zeros(cls) -> "RewardWeights":
        """Create zero-initialized weights (for training from scratch)."""
        return cls(weights=jnp.zeros(NUM_ACTIONS))

    @classmethod
    def uniform(cls, value: float = 1.0) -> "RewardWeights":
        """Create uniform weights (baseline: all actions equal)."""
        return cls(weights=jnp.ones(NUM_ACTIONS) * value)

    def get_positive_actions(self) -> list[tuple[str, float]]:
        """Get actions with positive weights, sorted by magnitude."""
        pairs = [
            (ACTION_NAMES[i], float(self.weights[i]))
            for i in range(NUM_ACTIONS)
            if self.weights[i] > 0
        ]
        return sorted(pairs, key=lambda x: -x[1])

    def get_negative_actions(self) -> list[tuple[str, float]]:
        """Get actions with negative weights, sorted by magnitude."""
        pairs = [
            (ACTION_NAMES[i], float(self.weights[i]))
            for i in range(NUM_ACTIONS)
            if self.weights[i] < 0
        ]
        return sorted(pairs, key=lambda x: x[1])

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary mapping action names to weights."""
        return {
            ACTION_NAMES[i]: float(self.weights[i])
            for i in range(NUM_ACTIONS)
        }

    def __repr__(self) -> str:
        pos = self.get_positive_actions()[:3]
        neg = self.get_negative_actions()[:3]
        pos_str = ", ".join(f"{n}={v:.2f}" for n, v in pos)
        neg_str = ", ".join(f"{n}={v:.2f}" for n, v in neg)
        return f"RewardWeights(top_pos=[{pos_str}], top_neg=[{neg_str}])"
