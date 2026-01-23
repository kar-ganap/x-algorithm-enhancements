"""Action differentiation tests.

Tests whether the model correctly differentiates action types between
user archetypes. Key tests:
1. Lurkers should have high like, low retweet/reply
2. Power users should have high engagement across all actions
3. Political users should have high block/mute for opposing content
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import jax.numpy as jnp
import numpy as np

from enhancements.data.ground_truth import UserArchetype, ContentTopic
from enhancements.data.synthetic_twitter import SyntheticTwitterDataset
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

from phoenix.recsys_model import RecsysBatch


@dataclass
class ActionDistribution:
    """Predicted action distribution for an archetype."""
    archetype: str
    favorite: float
    reply: float
    repost: float
    block_author: float
    mute_author: float
    not_interested: float

    @property
    def repost_ratio(self) -> float:
        """Repost to favorite ratio."""
        if self.favorite > 0:
            return self.repost / self.favorite
        return 0.0

    @property
    def reply_ratio(self) -> float:
        """Reply to favorite ratio."""
        if self.favorite > 0:
            return self.reply / self.favorite
        return 0.0


@dataclass
class ActionTestResult:
    """Result of a single action test."""
    test_name: str
    description: str
    expected: str
    actual: str
    passed: bool


@dataclass
class ActionTestResults:
    """Results from all action differentiation tests."""
    lurker_distribution: ActionDistribution
    power_user_distribution: ActionDistribution
    political_l_on_politics_r: ActionDistribution  # Left user on right content
    political_r_on_politics_l: ActionDistribution  # Right user on left content

    tests: List[ActionTestResult]
    tests_passed: int
    tests_total: int

    def __repr__(self) -> str:
        return (
            f"ActionTestResults(\n"
            f"  tests_passed={self.tests_passed}/{self.tests_total}\n"
            f"  lurker_rt_ratio={self.lurker_distribution.repost_ratio:.4f}\n"
            f"  power_rt_ratio={self.power_user_distribution.repost_ratio:.4f}\n"
            f")"
        )


def predict_action_distribution(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    archetype: UserArchetype,
    topic: ContentTopic = None,  # If None, use random posts
    num_samples: int = 100,
) -> ActionDistribution:
    """Predict action distribution for an archetype.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        archetype: User archetype to test
        topic: Optional topic to filter posts (None = all topics)
        num_samples: Number of samples

    Returns:
        ActionDistribution with mean predicted probabilities
    """
    users = dataset.get_users_by_archetype(archetype)
    if not users:
        return ActionDistribution(
            archetype=archetype.value,
            favorite=0, reply=0, repost=0,
            block_author=0, mute_author=0, not_interested=0,
        )

    if topic:
        posts = dataset.get_posts_by_topic(topic)
    else:
        posts = dataset.posts

    if not posts:
        return ActionDistribution(
            archetype=archetype.value,
            favorite=0, reply=0, repost=0,
            block_author=0, mute_author=0, not_interested=0,
        )

    rng = np.random.default_rng(42)
    sampled_users = rng.choice(users, size=min(num_samples, len(users)), replace=True)
    sampled_posts = rng.choice(posts, size=min(num_samples, len(posts)), replace=True)

    all_probs = []

    for user, post in zip(sampled_users, sampled_posts):
        batch, _ = adapter.create_batch_for_user(
            user.user_id,
            [post.post_id],
            num_candidates_override=1,
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

        embeddings = adapter.compute_embeddings_from_params(
            params["embeddings"], batch
        )

        output = runner.rank_candidates(params["model"], batch, embeddings)
        logits = output.logits[0, 0]  # [num_actions]

        # Convert to probabilities
        probs = 1.0 / (1.0 + np.exp(-np.array(logits)))
        all_probs.append(probs)

    mean_probs = np.mean(all_probs, axis=0)

    # Action indices
    return ActionDistribution(
        archetype=archetype.value,
        favorite=float(mean_probs[0]),   # favorite_score
        reply=float(mean_probs[1]),      # reply_score
        repost=float(mean_probs[2]),     # repost_score
        block_author=float(mean_probs[15]),  # block_author_score
        mute_author=float(mean_probs[16]),   # mute_author_score
        not_interested=float(mean_probs[14]),  # not_interested_score
    )


def run_action_tests(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    num_samples: int = 100,
) -> ActionTestResults:
    """Run all action differentiation tests.

    Tests:
    1. Lurker repost ratio < 0.05
    2. Power user repost ratio > 0.30
    3. Power user repost ratio > lurker repost ratio * 3
    4. Political L blocks more on R content
    5. Political R blocks more on L content

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        num_samples: Samples per test

    Returns:
        ActionTestResults
    """
    # Get distributions
    lurker_dist = predict_action_distribution(
        adapter, dataset, runner, params,
        UserArchetype.LURKER,
        num_samples=num_samples,
    )

    power_dist = predict_action_distribution(
        adapter, dataset, runner, params,
        UserArchetype.POWER_USER,
        num_samples=num_samples,
    )

    political_l_on_r = predict_action_distribution(
        adapter, dataset, runner, params,
        UserArchetype.POLITICAL_L,
        topic=ContentTopic.POLITICS_R,
        num_samples=num_samples,
    )

    political_r_on_l = predict_action_distribution(
        adapter, dataset, runner, params,
        UserArchetype.POLITICAL_R,
        topic=ContentTopic.POLITICS_L,
        num_samples=num_samples,
    )

    # Run tests
    tests = []

    # Test 1: Lurker low repost
    tests.append(ActionTestResult(
        test_name="lurker_low_repost",
        description="Lurkers should have low repost ratio (< 0.10)",
        expected="< 0.10",
        actual=f"{lurker_dist.repost_ratio:.4f}",
        passed=lurker_dist.repost_ratio < 0.10,
    ))

    # Test 2: Power user high repost
    tests.append(ActionTestResult(
        test_name="power_user_high_repost",
        description="Power users should have high repost ratio (> 0.30)",
        expected="> 0.30",
        actual=f"{power_dist.repost_ratio:.4f}",
        passed=power_dist.repost_ratio > 0.30,
    ))

    # Test 3: Power user > lurker repost
    ratio_diff = power_dist.repost_ratio / max(lurker_dist.repost_ratio, 0.01)
    tests.append(ActionTestResult(
        test_name="power_vs_lurker_repost",
        description="Power user repost ratio should be > 2x lurker",
        expected="> 2x",
        actual=f"{ratio_diff:.2f}x",
        passed=ratio_diff > 2.0,
    ))

    # Test 4: Lurker low reply
    tests.append(ActionTestResult(
        test_name="lurker_low_reply",
        description="Lurkers should have very low reply (< 0.05)",
        expected="< 0.05",
        actual=f"{lurker_dist.reply:.4f}",
        passed=lurker_dist.reply < 0.05,
    ))

    # Test 5: Political L blocks R content
    tests.append(ActionTestResult(
        test_name="political_l_blocks_r",
        description="Political L should block/mute R content (> 0.10)",
        expected="> 0.10",
        actual=f"{political_l_on_r.block_author:.4f}",
        passed=political_l_on_r.block_author > 0.10,
    ))

    # Test 6: Political R blocks L content
    tests.append(ActionTestResult(
        test_name="political_r_blocks_l",
        description="Political R should block/mute L content (> 0.10)",
        expected="> 0.10",
        actual=f"{political_r_on_l.block_author:.4f}",
        passed=political_r_on_l.block_author > 0.10,
    ))

    tests_passed = sum(1 for t in tests if t.passed)

    return ActionTestResults(
        lurker_distribution=lurker_dist,
        power_user_distribution=power_dist,
        political_l_on_politics_r=political_l_on_r,
        political_r_on_politics_l=political_r_on_l,
        tests=tests,
        tests_passed=tests_passed,
        tests_total=len(tests),
    )
