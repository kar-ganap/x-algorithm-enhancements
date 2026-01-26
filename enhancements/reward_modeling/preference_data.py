"""Preference data handling for F4 reward learning.

Defines PreferencePair for pairwise preference learning
and utilities for creating preference data from F2's synthetic data.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax.numpy as jnp
import numpy as np


@dataclass
class PreferencePair:
    """A pairwise preference: user prefers candidate A over candidate B.

    Used for Bradley-Terry preference learning where we train:
        P(A preferred to B) = σ(R(A) - R(B))

    Attributes:
        user_idx: Index of the user in the batch
        preferred_idx: Index of the preferred candidate
        rejected_idx: Index of the rejected candidate
        confidence: Optional confidence score (1.0 = certain)
    """

    user_idx: int
    preferred_idx: int
    rejected_idx: int
    confidence: float = 1.0

    def __post_init__(self):
        """Validate that preferred and rejected are different."""
        if self.preferred_idx == self.rejected_idx:
            raise ValueError(
                f"Preferred and rejected must be different, "
                f"got {self.preferred_idx} for both"
            )


@dataclass
class PreferenceBatch:
    """A batch of preference pairs for training.

    Attributes:
        user_indices: Array of user indices [N]
        preferred_indices: Array of preferred candidate indices [N]
        rejected_indices: Array of rejected candidate indices [N]
        confidences: Array of confidence scores [N]
    """

    user_indices: jnp.ndarray
    preferred_indices: jnp.ndarray
    rejected_indices: jnp.ndarray
    confidences: jnp.ndarray

    @classmethod
    def from_pairs(cls, pairs: List[PreferencePair]) -> "PreferenceBatch":
        """Create batch from list of preference pairs."""
        return cls(
            user_indices=jnp.array([p.user_idx for p in pairs]),
            preferred_indices=jnp.array([p.preferred_idx for p in pairs]),
            rejected_indices=jnp.array([p.rejected_idx for p in pairs]),
            confidences=jnp.array([p.confidence for p in pairs]),
        )

    def __len__(self) -> int:
        return len(self.user_indices)


def create_preferences_from_rewards(
    rewards: jnp.ndarray,
    num_pairs_per_user: int = 5,
    min_reward_diff: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> List[PreferencePair]:
    """Create preference pairs from known reward values.

    For each user, samples pairs of candidates where the higher-reward
    candidate is preferred.

    Args:
        rewards: Reward values of shape [batch_size, num_candidates]
        num_pairs_per_user: Number of preference pairs to generate per user
        min_reward_diff: Minimum reward difference to create a pair
        rng: Random number generator

    Returns:
        List of PreferencePair objects
    """
    if rng is None:
        rng = np.random.default_rng()

    batch_size, num_candidates = rewards.shape
    pairs = []

    for user_idx in range(batch_size):
        user_rewards = np.array(rewards[user_idx])

        # Get all pairs with sufficient reward difference
        for _ in range(num_pairs_per_user * 10):  # Try multiple times
            if len(pairs) >= (user_idx + 1) * num_pairs_per_user:
                break

            # Sample two candidates
            i, j = rng.choice(num_candidates, size=2, replace=False)
            diff = user_rewards[i] - user_rewards[j]

            if abs(diff) >= min_reward_diff:
                if diff > 0:
                    pairs.append(PreferencePair(
                        user_idx=user_idx,
                        preferred_idx=int(i),
                        rejected_idx=int(j),
                        confidence=min(1.0, abs(diff)),  # Higher diff = more confident
                    ))
                else:
                    pairs.append(PreferencePair(
                        user_idx=user_idx,
                        preferred_idx=int(j),
                        rejected_idx=int(i),
                        confidence=min(1.0, abs(diff)),
                    ))

    return pairs


def create_preferences_from_ground_truth(
    user_archetypes: List[str],
    candidate_topics: List[str],
    engagement_rules: dict,
    num_pairs: int = 100,
    rng: Optional[np.random.Generator] = None,
) -> List[PreferencePair]:
    """Create preference pairs from F2's ground truth engagement rules.

    Uses the ground truth probabilities to determine which posts
    a user would prefer based on their archetype.

    Args:
        user_archetypes: List of archetype names for each user
        candidate_topics: List of topic names for each candidate
        engagement_rules: F2's ENGAGEMENT_RULES dict
        num_pairs: Number of preference pairs to generate
        rng: Random number generator

    Returns:
        List of PreferencePair objects
    """
    if rng is None:
        rng = np.random.default_rng()

    pairs = []
    num_users = len(user_archetypes)
    num_candidates = len(candidate_topics)

    for _ in range(num_pairs):
        # Sample a user
        user_idx = rng.integers(num_users)
        archetype = user_archetypes[user_idx]

        # Sample two candidates
        i, j = rng.choice(num_candidates, size=2, replace=False)
        topic_i = candidate_topics[i]
        topic_j = candidate_topics[j]

        # Get engagement probabilities from ground truth
        # Higher positive engagement = preferred
        def get_preference_score(arch: str, topic: str) -> float:
            key = (arch, topic)
            if key in engagement_rules:
                probs = engagement_rules[key]
                # Positive: favorite, repost, follow
                # Negative: block, mute, not_interested
                positive = getattr(probs, "favorite", 0) + getattr(probs, "repost", 0)
                negative = getattr(probs, "block_author", 0) + getattr(probs, "not_interested", 0)
                return positive - negative
            return 0.0

        score_i = get_preference_score(archetype, topic_i)
        score_j = get_preference_score(archetype, topic_j)

        if score_i != score_j:
            if score_i > score_j:
                pairs.append(PreferencePair(
                    user_idx=user_idx,
                    preferred_idx=int(i),
                    rejected_idx=int(j),
                    confidence=min(1.0, abs(score_i - score_j)),
                ))
            else:
                pairs.append(PreferencePair(
                    user_idx=user_idx,
                    preferred_idx=int(j),
                    rejected_idx=int(i),
                    confidence=min(1.0, abs(score_i - score_j)),
                ))

    return pairs
