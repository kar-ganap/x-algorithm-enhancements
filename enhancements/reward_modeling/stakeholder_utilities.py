"""Multi-stakeholder utility functions for F4 reward modeling.

Defines utility functions for three stakeholders:
1. User Utility: engagement - discomfort
2. Platform Utility: total engagement + retention proxy
3. Society Utility: diversity - polarization

These utilities allow analyzing the same model through different lenses,
revealing tradeoffs in recommendation system design.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import numpy as np

# Define constants locally to avoid import chain that triggers grok
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

NUM_ACTIONS = len(ACTION_NAMES)
ACTION_INDICES: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}


class Stakeholder(Enum):
    """The three stakeholders in recommendation systems."""
    USER = "user"
    PLATFORM = "platform"
    SOCIETY = "society"


@dataclass
class UtilityWeights:
    """Weights for computing stakeholder utilities from action probabilities."""

    # User utility: engagement they wanted minus discomfort
    user_positive: dict[str, float]  # Actions that increase user utility
    user_negative: dict[str, float]  # Actions that decrease user utility

    # Platform utility: all engagement counts
    platform_weights: dict[str, float]

    @classmethod
    def default(cls) -> "UtilityWeights":
        """Default utility weights based on action semantics."""
        return cls(
            user_positive={
                "favorite": 1.0,      # User liked it
                "repost": 0.8,        # User endorsed it
                "reply": 0.5,         # User engaged (but could be negative)
                "share": 0.9,         # User recommended to others
                "follow_author": 1.2, # User wants more from author
                "quote": 0.6,         # User engaged with commentary
            },
            user_negative={
                "block_author": -2.0,    # Strong negative experience
                "mute_author": -1.5,     # Negative experience
                "report": -2.5,          # Very negative experience
                "not_interested": -1.0,  # Mild negative
            },
            platform_weights={
                # All engagement is valuable to platform
                "favorite": 1.0,
                "reply": 1.2,        # High-value: creates content
                "repost": 1.5,       # High-value: viral spread
                "photo_expand": 0.3,
                "click": 0.5,
                "profile_click": 0.6,
                "vqv": 0.4,
                "share": 1.3,
                "share_via_dm": 1.4,
                "share_via_copy_link": 1.2,
                "dwell": 0.2,        # Time on platform
                "quote": 1.3,
                "quoted_click": 0.4,
                "follow_author": 1.0,
                # Negative actions still count as engagement for platform
                "not_interested": 0.1,
                "block_author": 0.1,
                "mute_author": 0.1,
                "report": 0.2,  # Platform learns from reports
            },
        )


class UserUtilityResult(NamedTuple):
    """Result of computing user utility."""
    total_utility: float
    positive_component: float
    negative_component: float
    action_contributions: dict[str, float]


class PlatformUtilityResult(NamedTuple):
    """Result of computing platform utility."""
    total_utility: float
    engagement_score: float
    retention_proxy: float
    action_contributions: dict[str, float]


class SocietyUtilityResult(NamedTuple):
    """Result of computing society utility."""
    total_utility: float
    diversity_score: float
    polarization_score: float
    cross_exposure_rate: float


class StakeholderAnalysis(NamedTuple):
    """Complete analysis for all stakeholders."""
    user: UserUtilityResult
    platform: PlatformUtilityResult
    society: SocietyUtilityResult


def compute_user_utility(
    action_probs: np.ndarray,
    weights: UtilityWeights | None = None,
) -> UserUtilityResult:
    """Compute user utility from action probabilities.

    User utility = positive engagement - negative signals

    Args:
        action_probs: [num_actions] or [N, num_actions] action probabilities
        weights: Utility weights (default if None)

    Returns:
        UserUtilityResult with breakdown
    """
    if weights is None:
        weights = UtilityWeights.default()

    action_probs = np.atleast_2d(action_probs)
    N = action_probs.shape[0]

    positive = 0.0
    negative = 0.0
    contributions = {}

    # Positive contributions
    for action, weight in weights.user_positive.items():
        idx = ACTION_INDICES[action]
        contrib = float(np.mean(action_probs[:, idx]) * weight)
        contributions[action] = contrib
        positive += contrib

    # Negative contributions
    for action, weight in weights.user_negative.items():
        idx = ACTION_INDICES[action]
        contrib = float(np.mean(action_probs[:, idx]) * weight)
        contributions[action] = contrib
        negative += abs(contrib)  # Track magnitude

    total = positive - negative

    return UserUtilityResult(
        total_utility=total,
        positive_component=positive,
        negative_component=negative,
        action_contributions=contributions,
    )


def compute_platform_utility(
    action_probs: np.ndarray,
    return_rate: float = 0.8,  # Proxy for retention
    weights: UtilityWeights | None = None,
) -> PlatformUtilityResult:
    """Compute platform utility from action probabilities.

    Platform utility = total engagement + retention proxy

    Args:
        action_probs: [num_actions] or [N, num_actions] action probabilities
        return_rate: Estimated probability user returns (retention proxy)
        weights: Utility weights (default if None)

    Returns:
        PlatformUtilityResult with breakdown
    """
    if weights is None:
        weights = UtilityWeights.default()

    action_probs = np.atleast_2d(action_probs)

    engagement = 0.0
    contributions = {}

    for action, weight in weights.platform_weights.items():
        idx = ACTION_INDICES[action]
        contrib = float(np.mean(action_probs[:, idx]) * weight)
        contributions[action] = contrib
        engagement += contrib

    # Retention proxy based on positive vs negative signal ratio
    positive_signals = sum(
        np.mean(action_probs[:, ACTION_INDICES[a]])
        for a in ["favorite", "repost", "follow_author"]
    )
    negative_signals = sum(
        np.mean(action_probs[:, ACTION_INDICES[a]])
        for a in ["block_author", "mute_author", "report"]
    )

    # Retention decreases with negative signals
    retention = return_rate * (1 - 0.5 * negative_signals / (positive_signals + 0.1))
    retention = max(0, min(1, retention))

    total = engagement + retention

    return PlatformUtilityResult(
        total_utility=total,
        engagement_score=engagement,
        retention_proxy=float(retention),
        action_contributions=contributions,
    )


def compute_society_utility(
    recommendations: np.ndarray,
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    num_topics: int = 6,
) -> SocietyUtilityResult:
    """Compute society utility from recommendations.

    Society utility = diversity - polarization

    Args:
        recommendations: [N, K] top-K recommended content indices per user
        user_archetypes: [N] archetype index per user
        content_topics: [M] topic index per content item
        num_topics: Number of content topics

    Returns:
        SocietyUtilityResult with breakdown
    """
    N = len(recommendations)

    # Diversity: How many different topics does each user see?
    topic_coverage = []
    for i in range(N):
        rec_indices = recommendations[i]
        rec_topics = content_topics[rec_indices]
        unique_topics = len(np.unique(rec_topics))
        topic_coverage.append(unique_topics / num_topics)

    diversity = float(np.mean(topic_coverage))

    # Polarization: Do political users only see their side?
    # Archetypes 2=political_L, 3=political_R (from ground truth)
    # Topics 2=politics_L, 3=politics_R
    political_users = np.isin(user_archetypes, [2, 3])
    if political_users.sum() > 0:
        polarization_scores = []
        for i in np.where(political_users)[0]:
            archetype = user_archetypes[i]
            rec_indices = recommendations[i]
            rec_topics = content_topics[rec_indices]

            # Check if they only see their side
            if archetype == 2:  # political_L
                same_side = np.sum(rec_topics == 2)  # politics_L
                other_side = np.sum(rec_topics == 3)  # politics_R
            else:  # political_R
                same_side = np.sum(rec_topics == 3)  # politics_R
                other_side = np.sum(rec_topics == 2)  # politics_L

            total_political = same_side + other_side
            if total_political > 0:
                # Polarization = 1 if only same side, 0 if balanced
                polarization_scores.append(same_side / total_political)

        polarization = float(np.mean(polarization_scores)) if polarization_scores else 0.5
    else:
        polarization = 0.0

    # Cross-exposure: Rate of seeing "other side" content
    cross_exposure = 1 - polarization

    # Society utility: diversity good, polarization bad
    total = diversity - polarization

    return SocietyUtilityResult(
        total_utility=total,
        diversity_score=diversity,
        polarization_score=polarization,
        cross_exposure_rate=cross_exposure,
    )


def compute_society_utility_from_probs(
    action_probs_by_user: dict[int, np.ndarray],
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = 10,
) -> SocietyUtilityResult:
    """Compute society utility from action probability scores.

    Simulates recommendations by taking top-K scoring content per user.

    Args:
        action_probs_by_user: Dict mapping user_idx to [M, num_actions] probs
        user_archetypes: [N] archetype per user
        content_topics: [M] topic per content
        top_k: Number of recommendations per user

    Returns:
        SocietyUtilityResult
    """
    # Compute engagement scores for each user-content pair
    recommendations = []
    user_indices = sorted(action_probs_by_user.keys())

    for user_idx in user_indices:
        probs = action_probs_by_user[user_idx]  # [M, num_actions]
        # Simple engagement score: favorites + reposts
        scores = (
            probs[:, ACTION_INDICES["favorite"]] +
            probs[:, ACTION_INDICES["repost"]] * 0.8
        )
        # Top-K recommendations
        top_indices = np.argsort(scores)[-top_k:][::-1]
        recommendations.append(top_indices)

    recommendations = np.array(recommendations)
    user_archetypes_subset = user_archetypes[user_indices]

    return compute_society_utility(
        recommendations, user_archetypes_subset, content_topics
    )


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    diversity_weight: float
    user_utility: float
    platform_utility: float
    society_utility: float
    is_pareto_optimal: bool = True


def compute_pareto_frontier(
    base_action_probs: np.ndarray,
    diversity_weights: list[float],
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = 10,
) -> list[ParetoPoint]:
    """Compute Pareto frontier by varying diversity weight.

    At each diversity weight, we re-rank content and measure all utilities.

    Args:
        base_action_probs: [N, M, num_actions] - probs for each user-content pair
        diversity_weights: List of weights to try [0, 0.1, ..., 1.0]
        user_archetypes: [N] archetype per user
        content_topics: [M] topic per content
        top_k: Number of recommendations per user

    Returns:
        List of ParetoPoint showing tradeoffs
    """
    N, M, _ = base_action_probs.shape
    num_topics = len(np.unique(content_topics))

    points = []

    for div_weight in diversity_weights:
        # Compute scores with diversity bonus
        recommendations = []

        for user_idx in range(N):
            probs = base_action_probs[user_idx]  # [M, num_actions]

            # Base engagement score
            engagement_scores = (
                probs[:, ACTION_INDICES["favorite"]] +
                probs[:, ACTION_INDICES["repost"]] * 0.8 +
                probs[:, ACTION_INDICES["follow_author"]] * 0.5
            )

            # Diversity bonus: prefer content from underrepresented topics
            if div_weight > 0:
                selected = []
                remaining = list(range(M))
                topic_counts = np.zeros(num_topics)

                for _ in range(top_k):
                    if not remaining:
                        break

                    # Compute diversity-adjusted scores
                    adjusted_scores = []
                    for idx in remaining:
                        topic = content_topics[idx]
                        # Diversity bonus for underrepresented topics
                        diversity_bonus = 1.0 / (topic_counts[topic] + 1)
                        score = (1 - div_weight) * engagement_scores[idx] + div_weight * diversity_bonus
                        adjusted_scores.append((idx, score))

                    # Select best
                    best_idx, _ = max(adjusted_scores, key=lambda x: x[1])
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    topic_counts[content_topics[best_idx]] += 1

                recommendations.append(selected)
            else:
                # Pure engagement ranking
                top_indices = np.argsort(engagement_scores)[-top_k:][::-1]
                recommendations.append(top_indices.tolist())

        recommendations = np.array(recommendations)

        # Compute utilities
        # User utility: average over recommended content
        user_utils = []
        for user_idx in range(N):
            rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
            user_result = compute_user_utility(rec_probs)
            user_utils.append(user_result.total_utility)

        # Platform utility
        platform_utils = []
        for user_idx in range(N):
            rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
            platform_result = compute_platform_utility(rec_probs)
            platform_utils.append(platform_result.total_utility)

        # Society utility
        society_result = compute_society_utility(
            recommendations, user_archetypes, content_topics, num_topics
        )

        points.append(ParetoPoint(
            diversity_weight=div_weight,
            user_utility=float(np.mean(user_utils)),
            platform_utility=float(np.mean(platform_utils)),
            society_utility=society_result.total_utility,
        ))

    # Mark Pareto-optimal points
    # A point is Pareto-optimal if no other point is better in ALL objectives
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                # Check if p2 dominates p1
                if (p2.user_utility >= p1.user_utility and
                    p2.platform_utility >= p1.platform_utility and
                    p2.society_utility >= p1.society_utility and
                    (p2.user_utility > p1.user_utility or
                     p2.platform_utility > p1.platform_utility or
                     p2.society_utility > p1.society_utility)):
                    points[i] = ParetoPoint(
                        diversity_weight=p1.diversity_weight,
                        user_utility=p1.user_utility,
                        platform_utility=p1.platform_utility,
                        society_utility=p1.society_utility,
                        is_pareto_optimal=False,
                    )
                    break

    return points


def analyze_stakeholder_tradeoffs(
    points: list[ParetoPoint],
) -> dict[str, any]:
    """Analyze tradeoffs from Pareto frontier.

    Args:
        points: List of ParetoPoint from compute_pareto_frontier

    Returns:
        Dict with tradeoff analysis
    """
    # Find extreme points
    max_user = max(points, key=lambda p: p.user_utility)
    max_platform = max(points, key=lambda p: p.platform_utility)
    max_society = max(points, key=lambda p: p.society_utility)

    # Find balanced point (highest sum of normalized utilities)
    utilities = np.array([
        [p.user_utility, p.platform_utility, p.society_utility]
        for p in points
    ])

    # Normalize to [0, 1]
    mins = utilities.min(axis=0)
    maxs = utilities.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # Avoid division by zero

    normalized = (utilities - mins) / ranges
    balanced_idx = np.argmax(normalized.sum(axis=1))
    balanced = points[balanced_idx]

    # Compute tradeoff ratios
    # "What does 10% more society utility cost in user utility?"
    pareto_points = [p for p in points if p.is_pareto_optimal]

    return {
        "max_user_utility": {
            "point": max_user,
            "diversity_weight": max_user.diversity_weight,
        },
        "max_platform_utility": {
            "point": max_platform,
            "diversity_weight": max_platform.diversity_weight,
        },
        "max_society_utility": {
            "point": max_society,
            "diversity_weight": max_society.diversity_weight,
        },
        "balanced": {
            "point": balanced,
            "diversity_weight": balanced.diversity_weight,
        },
        "num_pareto_optimal": len(pareto_points),
        "total_points": len(points),
        "user_range": (
            min(p.user_utility for p in points),
            max(p.user_utility for p in points),
        ),
        "platform_range": (
            min(p.platform_utility for p in points),
            max(p.platform_utility for p in points),
        ),
        "society_range": (
            min(p.society_utility for p in points),
            max(p.society_utility for p in points),
        ),
    }
