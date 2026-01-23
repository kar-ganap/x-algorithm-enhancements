"""Behavioral tests for verifying learned preferences.

Tests whether the model predicts higher engagement probabilities
for archetype-topic pairs that should have high engagement based
on the ground truth rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from enhancements.data.ground_truth import (
    UserArchetype,
    ContentTopic,
    get_engagement_probs,
    ENGAGEMENT_RULES,
)
from enhancements.data.synthetic_twitter import SyntheticTwitterDataset
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

from phoenix.recsys_model import RecsysBatch


@dataclass
class BehavioralTestResult:
    """Result from a single behavioral test."""
    archetype: str
    topic: str
    action: str
    expected_prob: float
    predicted_prob: float
    error: float  # |predicted - expected|
    passed: bool


@dataclass
class BehavioralTestResults:
    """Results from all behavioral tests."""
    individual_tests: List[BehavioralTestResult]
    overall_accuracy: float  # % of tests within tolerance
    mean_error: float  # Average |predicted - expected|
    correlation: float  # Correlation between expected and predicted

    def __repr__(self) -> str:
        passed = sum(1 for t in self.individual_tests if t.passed)
        total = len(self.individual_tests)
        return (
            f"BehavioralTestResults(\n"
            f"  tests_passed={passed}/{total} ({100*passed/total:.1f}%)\n"
            f"  mean_error={self.mean_error:.4f}\n"
            f"  correlation={self.correlation:.4f}\n"
            f")"
        )


def predict_engagement_probabilities(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    archetype: UserArchetype,
    topic: ContentTopic,
    num_samples: int = 50,
) -> np.ndarray:
    """Predict engagement probabilities for archetype-topic pair.

    Samples users of the given archetype and posts of the given topic,
    then predicts engagement probabilities.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        archetype: User archetype to test
        topic: Content topic to test
        num_samples: Number of (user, post) pairs to sample

    Returns:
        Mean predicted probabilities [num_actions]
    """
    # Get users of this archetype
    users = dataset.get_users_by_archetype(archetype)
    if not users:
        return np.zeros(18)

    # Get posts of this topic
    posts = dataset.get_posts_by_topic(topic)
    if not posts:
        return np.zeros(18)

    rng = np.random.default_rng(42)
    sampled_users = rng.choice(users, size=min(num_samples, len(users)), replace=True)
    sampled_posts = rng.choice(posts, size=min(num_samples, len(posts)), replace=True)

    all_probs = []

    for user, post in zip(sampled_users, sampled_posts):
        # Create batch for this user-post pair
        batch, _ = adapter.create_batch_for_user(
            user.user_id,
            [post.post_id],
            num_candidates_override=1,
        )

        # Convert to JAX
        batch = RecsysBatch(
            user_hashes=jnp.array(batch.user_hashes),
            history_post_hashes=jnp.array(batch.history_post_hashes),
            history_author_hashes=jnp.array(batch.history_author_hashes),
            history_actions=jnp.array(batch.history_actions),
            history_product_surface=jnp.array(batch.history_product_surface),
            candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
            candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
            candidate_product_surface=jnp.array(batch.candidate_product_surface),
        )

        # Compute embeddings
        embeddings = adapter.compute_embeddings_from_params(
            params["embeddings"], batch
        )

        # Forward pass
        output = runner.rank_candidates(params["model"], batch, embeddings)
        logits = output.scores  # [1, 1, num_actions]

        # Convert logits to probabilities using sigmoid
        probs = 1.0 / (1.0 + np.exp(-np.array(logits[0, 0])))
        all_probs.append(probs[:18])  # First 18 actions (excluding dwell_time)

    return np.mean(all_probs, axis=0)


def run_behavioral_tests(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    tolerance: float = 0.15,
    num_samples: int = 50,
) -> BehavioralTestResults:
    """Run behavioral tests for all archetype-topic pairs.

    Tests that predicted engagement probabilities match the ground truth
    engagement rules.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        tolerance: Maximum allowed error (|predicted - expected|)
        num_samples: Samples per archetype-topic pair

    Returns:
        BehavioralTestResults with all test results
    """
    # Action name mapping (index -> name)
    action_names = [
        "favorite", "reply", "repost", "photo_expand", "click",
        "profile_click", "vqv", "share", "share_via_dm",
        "share_via_copy_link", "dwell", "quote", "quoted_click",
        "follow_author", "not_interested", "block_author",
        "mute_author", "report",
    ]

    results = []
    expected_values = []
    predicted_values = []

    # Test each archetype-topic pair that has explicit rules
    for (archetype, topic_str), action_probs in ENGAGEMENT_RULES.items():
        # Skip wildcard rules for now
        if topic_str == "*":
            continue

        topic = ContentTopic(topic_str)

        # Get predicted probabilities
        predicted = predict_engagement_probabilities(
            adapter, dataset, runner, params,
            archetype, topic, num_samples,
        )

        # Compare to expected
        expected_array = action_probs.to_array()[:18]

        for i, (exp, pred, action) in enumerate(zip(expected_array, predicted, action_names)):
            # Only test actions with non-zero expected probability
            if exp > 0.05:  # Threshold to avoid noise
                error = abs(pred - exp)
                passed = error <= tolerance

                results.append(BehavioralTestResult(
                    archetype=archetype.value,
                    topic=topic_str,
                    action=action,
                    expected_prob=exp,
                    predicted_prob=float(pred),
                    error=error,
                    passed=passed,
                ))

                expected_values.append(exp)
                predicted_values.append(float(pred))

    # Compute overall metrics
    if results:
        overall_accuracy = sum(1 for r in results if r.passed) / len(results)
        mean_error = np.mean([r.error for r in results])

        # Correlation
        if len(expected_values) > 1:
            correlation = float(np.corrcoef(expected_values, predicted_values)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
    else:
        overall_accuracy = 0.0
        mean_error = 1.0
        correlation = 0.0

    return BehavioralTestResults(
        individual_tests=results,
        overall_accuracy=overall_accuracy,
        mean_error=mean_error,
        correlation=correlation,
    )


def test_topic_preference(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    archetype: UserArchetype,
    preferred_topic: ContentTopic,
    other_topic: ContentTopic,
    action: str = "favorite",
    num_samples: int = 50,
) -> Tuple[float, float, bool]:
    """Test that archetype prefers one topic over another.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        archetype: User archetype
        preferred_topic: Topic that should have higher engagement
        other_topic: Topic that should have lower engagement
        action: Action to compare (e.g., "favorite")
        num_samples: Samples per test

    Returns:
        (preferred_prob, other_prob, passed)
    """
    action_idx = [
        "favorite", "reply", "repost", "photo_expand", "click",
        "profile_click", "vqv", "share", "share_via_dm",
        "share_via_copy_link", "dwell", "quote", "quoted_click",
        "follow_author", "not_interested", "block_author",
        "mute_author", "report",
    ].index(action)

    # Get predictions for preferred topic
    preferred_probs = predict_engagement_probabilities(
        adapter, dataset, runner, params,
        archetype, preferred_topic, num_samples,
    )

    # Get predictions for other topic
    other_probs = predict_engagement_probabilities(
        adapter, dataset, runner, params,
        archetype, other_topic, num_samples,
    )

    preferred_prob = float(preferred_probs[action_idx])
    other_prob = float(other_probs[action_idx])
    passed = preferred_prob > other_prob

    return preferred_prob, other_prob, passed
