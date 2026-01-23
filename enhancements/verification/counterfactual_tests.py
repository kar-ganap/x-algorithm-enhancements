"""Counterfactual tests for verifying causal effects.

Tests whether the model responds correctly to interventions:
1. Block effect: Adding a block to history should reduce author ranking
2. Archetype flip: Giving a user a different archetype's history should
   change their predictions to match that archetype
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from enhancements.data.ground_truth import UserArchetype, ContentTopic
from enhancements.data.synthetic_twitter import (
    SyntheticTwitterDataset,
    SyntheticEngagement,
)
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

from phoenix.recsys_model import RecsysBatch


@dataclass
class BlockEffectResult:
    """Result of block effect test."""
    user_id: int
    author_id: int
    baseline_score: float
    after_block_score: float
    score_decreased: bool
    rank_worsened: bool


@dataclass
class ArchetypeFlipResult:
    """Result of archetype flip test."""
    user_id: int
    original_archetype: str
    donor_archetype: str
    original_preferred_topic_score: float
    after_flip_preferred_topic_score: float
    original_donor_topic_score: float
    after_flip_donor_topic_score: float
    predictions_flipped: bool


@dataclass
class CounterfactualTestResults:
    """Results from all counterfactual tests."""
    block_effects: List[BlockEffectResult]
    block_effect_rate: float  # % where block reduced ranking

    archetype_flips: List[ArchetypeFlipResult]
    archetype_flip_rate: float  # % where predictions flipped

    def __repr__(self) -> str:
        return (
            f"CounterfactualTestResults(\n"
            f"  block_effect_rate={self.block_effect_rate:.2%}\n"
            f"  archetype_flip_rate={self.archetype_flip_rate:.2%}\n"
            f")"
        )


def test_block_effect(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    num_tests: int = 50,
) -> Tuple[List[BlockEffectResult], float]:
    """Test that blocking an author reduces their posts' rankings.

    For each test:
    1. Pick a user and an author they haven't blocked
    2. Score the author's posts with baseline history
    3. Add a block action to history
    4. Re-score and verify ranking decreased

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        num_tests: Number of tests to run

    Returns:
        (results, success_rate)
    """
    rng = np.random.default_rng(42)
    results = []

    # Get users with some history
    users_with_history = [
        u for u in dataset.users
        if len(dataset.get_user_engagements(u.user_id)) >= 5
    ]

    if len(users_with_history) < num_tests:
        users_with_history = users_with_history * (num_tests // len(users_with_history) + 1)

    sampled_users = rng.choice(users_with_history, size=num_tests, replace=False)

    for user in sampled_users:
        # Pick a random author
        author = rng.choice(dataset.authors)

        # Get posts by this author
        author_posts = dataset.get_posts_by_author(author.author_id)
        if not author_posts:
            continue

        # Get baseline score for author's posts
        post = rng.choice(author_posts)
        baseline_score = _get_score_for_post(
            adapter, dataset, runner, params,
            user.user_id, post.post_id,
        )

        # Add block to history (by modifying the history temporarily)
        modified_score = _get_score_with_block(
            adapter, dataset, runner, params,
            user.user_id, post.post_id, author.author_id,
        )

        score_decreased = modified_score < baseline_score
        # Rank worsened if score decreased significantly
        rank_worsened = modified_score < baseline_score - 0.1

        results.append(BlockEffectResult(
            user_id=user.user_id,
            author_id=author.author_id,
            baseline_score=baseline_score,
            after_block_score=modified_score,
            score_decreased=score_decreased,
            rank_worsened=rank_worsened,
        ))

    success_rate = sum(1 for r in results if r.score_decreased) / len(results) if results else 0
    return results, success_rate


def _get_score_for_post(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    user_id: int,
    post_id: int,
) -> float:
    """Get predicted score for a single post."""
    batch, _ = adapter.create_batch_for_user(
        user_id, [post_id], num_candidates_override=1
    )

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

    embeddings = adapter.compute_embeddings_from_params(params["embeddings"], batch)
    output = runner.rank_candidates(params["model"], batch, embeddings)

    # Mean score across actions
    return float(jnp.mean(output.scores[0, 0]))


def _get_score_with_block(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    user_id: int,
    post_id: int,
    blocked_author_id: int,
) -> float:
    """Get predicted score with a block action added to history."""
    batch, _ = adapter.create_batch_for_user(
        user_id, [post_id], num_candidates_override=1
    )

    # Modify history to include a block action for this author
    # Add block to the most recent history slot
    history_actions = np.array(batch.history_actions)
    history_author_hashes = np.array(batch.history_author_hashes)

    # Find a slot to add the block (use the last non-zero slot or slot 0)
    history_len = history_actions.shape[1]
    insert_idx = min(history_len - 1, 0)

    # Set block_author action (index 15)
    history_actions[0, insert_idx, :] = 0
    history_actions[0, insert_idx, 15] = 1.0  # block_author_score

    # Set author hash
    for k in range(history_author_hashes.shape[2]):
        history_author_hashes[0, insert_idx, k] = blocked_author_id

    batch = RecsysBatch(
        user_hashes=jnp.array(batch.user_hashes),
        history_post_hashes=jnp.array(batch.history_post_hashes),
        history_author_hashes=jnp.array(history_author_hashes),
        history_actions=jnp.array(history_actions),
        history_product_surface=jnp.array(batch.history_product_surface),
        candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
        candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
        candidate_product_surface=jnp.array(batch.candidate_product_surface),
    )

    embeddings = adapter.compute_embeddings_from_params(params["embeddings"], batch)
    output = runner.rank_candidates(params["model"], batch, embeddings)

    return float(jnp.mean(output.scores[0, 0]))


def test_archetype_flip(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    num_tests: int = 30,
) -> Tuple[List[ArchetypeFlipResult], float]:
    """Test that giving a user a different archetype's history changes predictions.

    For example:
    - Sports fan with tech bro history should predict higher tech engagement
    - Political L with political R history should predict higher R engagement

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        num_tests: Number of tests

    Returns:
        (results, flip_rate)
    """
    rng = np.random.default_rng(42)
    results = []

    # Test pairs: (original archetype, donor archetype, original preferred topic, donor preferred topic)
    test_pairs = [
        (UserArchetype.SPORTS_FAN, UserArchetype.TECH_BRO, ContentTopic.SPORTS, ContentTopic.TECH),
        (UserArchetype.TECH_BRO, UserArchetype.SPORTS_FAN, ContentTopic.TECH, ContentTopic.SPORTS),
        (UserArchetype.POLITICAL_L, UserArchetype.POLITICAL_R, ContentTopic.POLITICS_L, ContentTopic.POLITICS_R),
        (UserArchetype.POLITICAL_R, UserArchetype.POLITICAL_L, ContentTopic.POLITICS_R, ContentTopic.POLITICS_L),
    ]

    tests_per_pair = max(1, num_tests // len(test_pairs))

    for orig_arch, donor_arch, orig_topic, donor_topic in test_pairs:
        orig_users = dataset.get_users_by_archetype(orig_arch)
        donor_users = dataset.get_users_by_archetype(donor_arch)

        if not orig_users or not donor_users:
            continue

        orig_topic_posts = dataset.get_posts_by_topic(orig_topic)
        donor_topic_posts = dataset.get_posts_by_topic(donor_topic)

        if not orig_topic_posts or not donor_topic_posts:
            continue

        for _ in range(tests_per_pair):
            user = rng.choice(orig_users)
            donor_user = rng.choice(donor_users)

            orig_topic_post = rng.choice(orig_topic_posts)
            donor_topic_post = rng.choice(donor_topic_posts)

            # Get original predictions
            orig_on_orig = _get_score_for_post(
                adapter, dataset, runner, params,
                user.user_id, orig_topic_post.post_id,
            )
            orig_on_donor = _get_score_for_post(
                adapter, dataset, runner, params,
                user.user_id, donor_topic_post.post_id,
            )

            # Get predictions with donor's history
            flip_on_orig = _get_score_with_donor_history(
                adapter, dataset, runner, params,
                user.user_id, orig_topic_post.post_id, donor_user.user_id,
            )
            flip_on_donor = _get_score_with_donor_history(
                adapter, dataset, runner, params,
                user.user_id, donor_topic_post.post_id, donor_user.user_id,
            )

            # Check if predictions flipped
            # Originally: orig > donor (user prefers their natural topic)
            # After flip: donor > orig (user now prefers donor's topic)
            originally_preferred_orig = orig_on_orig > orig_on_donor
            after_prefers_donor = flip_on_donor > flip_on_orig
            flipped = originally_preferred_orig and after_prefers_donor

            results.append(ArchetypeFlipResult(
                user_id=user.user_id,
                original_archetype=orig_arch.value,
                donor_archetype=donor_arch.value,
                original_preferred_topic_score=orig_on_orig,
                after_flip_preferred_topic_score=flip_on_orig,
                original_donor_topic_score=orig_on_donor,
                after_flip_donor_topic_score=flip_on_donor,
                predictions_flipped=flipped,
            ))

    flip_rate = sum(1 for r in results if r.predictions_flipped) / len(results) if results else 0
    return results, flip_rate


def _get_score_with_donor_history(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    user_id: int,
    post_id: int,
    donor_user_id: int,
) -> float:
    """Get score using another user's history."""
    # Get donor's batch to copy their history
    donor_batch, _ = adapter.create_batch_for_user(
        donor_user_id, [post_id], num_candidates_override=1
    )

    # Get original user's batch
    batch, _ = adapter.create_batch_for_user(
        user_id, [post_id], num_candidates_override=1
    )

    # Replace history with donor's history
    batch = RecsysBatch(
        user_hashes=jnp.array(batch.user_hashes),  # Keep original user
        history_post_hashes=jnp.array(donor_batch.history_post_hashes),  # Use donor's
        history_author_hashes=jnp.array(donor_batch.history_author_hashes),
        history_actions=jnp.array(donor_batch.history_actions),
        history_product_surface=jnp.array(donor_batch.history_product_surface),
        candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
        candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
        candidate_product_surface=jnp.array(batch.candidate_product_surface),
    )

    embeddings = adapter.compute_embeddings_from_params(params["embeddings"], batch)
    output = runner.rank_candidates(params["model"], batch, embeddings)

    return float(jnp.mean(output.scores[0, 0]))


def run_counterfactual_tests(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    num_block_tests: int = 50,
    num_flip_tests: int = 30,
) -> CounterfactualTestResults:
    """Run all counterfactual tests.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        num_block_tests: Number of block effect tests
        num_flip_tests: Number of archetype flip tests

    Returns:
        CounterfactualTestResults
    """
    block_results, block_rate = test_block_effect(
        adapter, dataset, runner, params, num_block_tests
    )

    flip_results, flip_rate = test_archetype_flip(
        adapter, dataset, runner, params, num_flip_tests
    )

    return CounterfactualTestResults(
        block_effects=block_results,
        block_effect_rate=block_rate,
        archetype_flips=flip_results,
        archetype_flip_rate=flip_rate,
    )
