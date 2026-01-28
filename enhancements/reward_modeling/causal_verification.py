"""Causal verification framework for F4 reward models.

Implements intervention tests to verify the reward model captures
causal relationships, not just correlations.

Two key tests:
1. Block Intervention Test: Injecting a block should decrease score
2. History Intervention Test: Matching history should increase score

From design doc:
    "Rewards should capture causation, not just correlation"
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import numpy as np

from enhancements.reward_modeling.weights import ACTION_INDICES, NUM_ACTIONS


class InterventionType(Enum):
    """Type of causal intervention."""
    BLOCK = "block"
    MUTE = "mute"
    NOT_INTERESTED = "not_interested"
    FOLLOW = "follow"
    TOPIC_HISTORY = "topic_history"


@dataclass
class CausalTestConfig:
    """Configuration for causal verification tests."""
    # Number of test samples
    num_samples: int = 100

    # For block intervention
    block_strength: float = 0.8  # How strongly to inject block signal

    # For history intervention
    history_strength: float = 0.7  # How much history engagement

    # Thresholds for test passing
    min_effect_size: float = 0.1  # Minimum score change to consider significant
    min_pass_rate: float = 0.9  # 90% of samples must show expected effect

    # Random seed
    random_seed: int = 42


class InterventionResult(NamedTuple):
    """Result of a single intervention test."""
    score_before: float
    score_after: float
    effect_size: float  # score_after - score_before
    expected_direction: str  # "decrease" or "increase"
    passed: bool  # Did effect match expected direction?


class CausalTestResults(NamedTuple):
    """Results of a full causal test suite."""
    test_name: str
    intervention_type: InterventionType
    num_samples: int
    pass_rate: float
    mean_effect_size: float
    std_effect_size: float
    passed: bool  # Overall test passed?
    individual_results: List[InterventionResult]


# Type alias for reward function
RewardFunction = Callable[[np.ndarray, np.ndarray], float]


def create_block_intervention(
    action_probs: np.ndarray,
    block_strength: float = 0.8,
) -> np.ndarray:
    """Create action probabilities with block intervention.

    Simulates what happens when a user blocks an author:
    - High probability of block_author action
    - Reduced positive engagement (favorite, repost, follow)

    Args:
        action_probs: Original [num_actions] action probabilities
        block_strength: How strongly to inject block signal (0-1)

    Returns:
        Modified action probabilities
    """
    modified = action_probs.copy()

    # Increase block probability
    block_idx = ACTION_INDICES["block_author"]
    modified[block_idx] = block_strength

    # Decrease positive engagement probabilities
    positive_actions = ["favorite", "repost", "follow_author", "share"]
    for action in positive_actions:
        idx = ACTION_INDICES[action]
        modified[idx] = modified[idx] * (1 - block_strength)

    # Normalize (optional - depends on model design)
    # modified = modified / modified.sum()

    return modified


def create_mute_intervention(
    action_probs: np.ndarray,
    mute_strength: float = 0.6,
) -> np.ndarray:
    """Create action probabilities with mute intervention.

    Similar to block but weaker signal - user just doesn't want to see content.

    Args:
        action_probs: Original [num_actions] action probabilities
        mute_strength: How strongly to inject mute signal (0-1)

    Returns:
        Modified action probabilities
    """
    modified = action_probs.copy()

    # Increase mute probability
    mute_idx = ACTION_INDICES["mute_author"]
    modified[mute_idx] = mute_strength

    # Slightly decrease engagement
    positive_actions = ["favorite", "repost", "follow_author"]
    for action in positive_actions:
        idx = ACTION_INDICES[action]
        modified[idx] = modified[idx] * (1 - mute_strength * 0.5)

    return modified


def create_not_interested_intervention(
    action_probs: np.ndarray,
    strength: float = 0.7,
) -> np.ndarray:
    """Create action probabilities with 'not interested' intervention.

    Args:
        action_probs: Original [num_actions] action probabilities
        strength: How strongly to inject signal (0-1)

    Returns:
        Modified action probabilities
    """
    modified = action_probs.copy()

    # Increase not_interested probability
    ni_idx = ACTION_INDICES["not_interested"]
    modified[ni_idx] = strength

    return modified


def create_follow_intervention(
    action_probs: np.ndarray,
    follow_strength: float = 0.8,
) -> np.ndarray:
    """Create action probabilities with follow intervention.

    Simulates what happens when a user follows an author:
    - High probability of follow_author action
    - Increased positive engagement

    Args:
        action_probs: Original [num_actions] action probabilities
        follow_strength: How strongly to inject follow signal (0-1)

    Returns:
        Modified action probabilities
    """
    modified = action_probs.copy()

    # Increase follow probability
    follow_idx = ACTION_INDICES["follow_author"]
    modified[follow_idx] = follow_strength

    # Increase positive engagement probabilities
    positive_actions = ["favorite", "repost", "share", "profile_click"]
    for action in positive_actions:
        idx = ACTION_INDICES[action]
        modified[idx] = min(1.0, modified[idx] + follow_strength * 0.3)

    # Decrease negative actions
    modified[ACTION_INDICES["block_author"]] = 0.0
    modified[ACTION_INDICES["mute_author"]] = 0.0
    modified[ACTION_INDICES["not_interested"]] = 0.0

    return modified


def create_history_with_topic(
    base_history: np.ndarray,
    topic_idx: int,
    num_topics: int,
    engagement_strength: float = 0.7,
) -> np.ndarray:
    """Create user history with engagement on a specific topic.

    For rich feature format: [topic × action] = [num_topics × num_actions]

    Args:
        base_history: Original user history features
        topic_idx: Topic index to inject engagement for
        num_topics: Total number of topics (e.g., 6)
        engagement_strength: How much engagement to inject

    Returns:
        Modified user history
    """
    modified = base_history.copy().astype(np.float32)

    # If rich features (topic × action matrix)
    if len(modified) == num_topics * NUM_ACTIONS:
        # Reshape to [num_topics, num_actions]
        history_matrix = modified.reshape(num_topics, NUM_ACTIONS)

        # Inject positive engagement for target topic
        positive_actions = [
            ACTION_INDICES["favorite"],
            ACTION_INDICES["repost"],
            ACTION_INDICES["follow_author"],
        ]
        for action_idx in positive_actions:
            history_matrix[topic_idx, action_idx] = engagement_strength

        modified = history_matrix.flatten()

    # If topic-only features
    elif len(modified) == NUM_ACTIONS:
        # Inject general engagement
        positive_actions = [
            ACTION_INDICES["favorite"],
            ACTION_INDICES["repost"],
            ACTION_INDICES["follow_author"],
        ]
        for action_idx in positive_actions:
            modified[action_idx] = engagement_strength

    return modified


def run_block_intervention_test(
    reward_fn: RewardFunction,
    user_histories: np.ndarray,
    action_probs_batch: np.ndarray,
    config: CausalTestConfig,
) -> CausalTestResults:
    """Run block intervention test.

    Test: After injecting a block signal, score should decrease.

    This verifies the reward model correctly captures that blocking
    is a strong negative signal.

    Args:
        reward_fn: Function (user_history, action_probs) -> reward
        user_histories: [N, num_features] user history features
        action_probs_batch: [N, num_actions] content action probabilities
        config: Test configuration

    Returns:
        Test results
    """
    rng = np.random.default_rng(config.random_seed)
    num_samples = min(config.num_samples, len(user_histories))

    # Sample indices
    indices = rng.choice(len(user_histories), size=num_samples, replace=False)

    results = []
    for idx in indices:
        user_history = user_histories[idx]
        action_probs = action_probs_batch[idx]

        # Score before intervention
        score_before = reward_fn(user_history, action_probs)

        # Apply block intervention
        modified_probs = create_block_intervention(action_probs, config.block_strength)
        score_after = reward_fn(user_history, modified_probs)

        effect_size = score_after - score_before
        passed = effect_size < -config.min_effect_size  # Should decrease

        results.append(InterventionResult(
            score_before=float(score_before),
            score_after=float(score_after),
            effect_size=float(effect_size),
            expected_direction="decrease",
            passed=passed,
        ))

    # Compute statistics
    pass_rate = sum(r.passed for r in results) / len(results)
    effect_sizes = [r.effect_size for r in results]
    mean_effect = np.mean(effect_sizes)
    std_effect = np.std(effect_sizes)

    overall_passed = pass_rate >= config.min_pass_rate

    return CausalTestResults(
        test_name="Block Intervention Test",
        intervention_type=InterventionType.BLOCK,
        num_samples=num_samples,
        pass_rate=pass_rate,
        mean_effect_size=float(mean_effect),
        std_effect_size=float(std_effect),
        passed=overall_passed,
        individual_results=results,
    )


def run_follow_intervention_test(
    reward_fn: RewardFunction,
    user_histories: np.ndarray,
    action_probs_batch: np.ndarray,
    config: CausalTestConfig,
) -> CausalTestResults:
    """Run follow intervention test.

    Test: After injecting a follow signal, score should increase.

    This verifies the reward model correctly captures that following
    is a positive signal.

    Args:
        reward_fn: Function (user_history, action_probs) -> reward
        user_histories: [N, num_features] user history features
        action_probs_batch: [N, num_actions] content action probabilities
        config: Test configuration

    Returns:
        Test results
    """
    rng = np.random.default_rng(config.random_seed + 1)
    num_samples = min(config.num_samples, len(user_histories))

    indices = rng.choice(len(user_histories), size=num_samples, replace=False)

    results = []
    for idx in indices:
        user_history = user_histories[idx]
        action_probs = action_probs_batch[idx]

        score_before = reward_fn(user_history, action_probs)

        modified_probs = create_follow_intervention(action_probs, config.block_strength)
        score_after = reward_fn(user_history, modified_probs)

        effect_size = score_after - score_before
        passed = effect_size > config.min_effect_size  # Should increase

        results.append(InterventionResult(
            score_before=float(score_before),
            score_after=float(score_after),
            effect_size=float(effect_size),
            expected_direction="increase",
            passed=passed,
        ))

    pass_rate = sum(r.passed for r in results) / len(results)
    effect_sizes = [r.effect_size for r in results]

    return CausalTestResults(
        test_name="Follow Intervention Test",
        intervention_type=InterventionType.FOLLOW,
        num_samples=num_samples,
        pass_rate=pass_rate,
        mean_effect_size=float(np.mean(effect_sizes)),
        std_effect_size=float(np.std(effect_sizes)),
        passed=pass_rate >= config.min_pass_rate,
        individual_results=results,
    )


def run_history_intervention_test(
    reward_fn: RewardFunction,
    action_probs_batch: np.ndarray,
    post_topics: np.ndarray,
    num_topics: int,
    base_user_history: np.ndarray,
    config: CausalTestConfig,
) -> CausalTestResults:
    """Run history intervention test.

    Test: A user with matching topic history should score content higher
    than a user with mismatched topic history.

    This verifies the reward model captures that user history causally
    affects content preferences.

    Args:
        reward_fn: Function (user_history, action_probs) -> reward
        action_probs_batch: [N, num_actions] content action probabilities
        post_topics: [N] topic index for each post
        num_topics: Total number of topics
        base_user_history: [num_features] neutral user history template
        config: Test configuration

    Returns:
        Test results
    """
    rng = np.random.default_rng(config.random_seed + 2)
    num_samples = min(config.num_samples, len(action_probs_batch))

    indices = rng.choice(len(action_probs_batch), size=num_samples, replace=False)

    results = []
    for idx in indices:
        action_probs = action_probs_batch[idx]
        post_topic = int(post_topics[idx])

        # Create user with matching history
        matching_history = create_history_with_topic(
            base_user_history.copy(),
            topic_idx=post_topic,
            num_topics=num_topics,
            engagement_strength=config.history_strength,
        )

        # Create user with mismatched history (different topic)
        mismatched_topic = (post_topic + 1) % num_topics
        mismatched_history = create_history_with_topic(
            base_user_history.copy(),
            topic_idx=mismatched_topic,
            num_topics=num_topics,
            engagement_strength=config.history_strength,
        )

        # Score with both histories
        score_match = reward_fn(matching_history, action_probs)
        score_mismatch = reward_fn(mismatched_history, action_probs)

        effect_size = score_match - score_mismatch
        passed = effect_size > config.min_effect_size  # Match should score higher

        results.append(InterventionResult(
            score_before=float(score_mismatch),  # "before" = mismatched
            score_after=float(score_match),  # "after" = matched
            effect_size=float(effect_size),
            expected_direction="increase",
            passed=passed,
        ))

    pass_rate = sum(r.passed for r in results) / len(results)
    effect_sizes = [r.effect_size for r in results]

    return CausalTestResults(
        test_name="History Intervention Test",
        intervention_type=InterventionType.TOPIC_HISTORY,
        num_samples=num_samples,
        pass_rate=pass_rate,
        mean_effect_size=float(np.mean(effect_sizes)),
        std_effect_size=float(np.std(effect_sizes)),
        passed=pass_rate >= config.min_pass_rate,
        individual_results=results,
    )


@dataclass
class CausalVerificationSuite:
    """Complete causal verification test suite."""
    config: CausalTestConfig

    def run_all(
        self,
        reward_fn: RewardFunction,
        user_histories: np.ndarray,
        action_probs_batch: np.ndarray,
        post_topics: Optional[np.ndarray] = None,
        num_topics: int = 6,
        verbose: bool = True,
    ) -> Dict[str, CausalTestResults]:
        """Run all causal verification tests.

        Args:
            reward_fn: Function (user_history, action_probs) -> reward
            user_histories: [N, num_features] user history features
            action_probs_batch: [N, num_actions] content action probabilities
            post_topics: [N] topic index for each post (for history test)
            num_topics: Total number of topics
            verbose: Print progress

        Returns:
            Dict mapping test name to results
        """
        results = {}

        if verbose:
            print("=" * 60)
            print("CAUSAL VERIFICATION SUITE")
            print("=" * 60)

        # Test 1: Block intervention
        if verbose:
            print("\n[1/3] Block Intervention Test...")
        block_results = run_block_intervention_test(
            reward_fn, user_histories, action_probs_batch, self.config
        )
        results["block_intervention"] = block_results
        if verbose:
            status = "PASS" if block_results.passed else "FAIL"
            print(f"  Result: {status}")
            print(f"  Pass rate: {block_results.pass_rate:.1%}")
            print(f"  Mean effect: {block_results.mean_effect_size:.4f}")

        # Test 2: Follow intervention
        if verbose:
            print("\n[2/3] Follow Intervention Test...")
        follow_results = run_follow_intervention_test(
            reward_fn, user_histories, action_probs_batch, self.config
        )
        results["follow_intervention"] = follow_results
        if verbose:
            status = "PASS" if follow_results.passed else "FAIL"
            print(f"  Result: {status}")
            print(f"  Pass rate: {follow_results.pass_rate:.1%}")
            print(f"  Mean effect: {follow_results.mean_effect_size:.4f}")

        # Test 3: History intervention (if topics provided)
        if post_topics is not None:
            if verbose:
                print("\n[3/3] History Intervention Test...")
            # Use float32 for consistency with sklearn models
            base_history = np.zeros(user_histories.shape[1], dtype=np.float32)
            history_results = run_history_intervention_test(
                reward_fn, action_probs_batch, post_topics,
                num_topics, base_history, self.config
            )
            results["history_intervention"] = history_results
            if verbose:
                status = "PASS" if history_results.passed else "FAIL"
                print(f"  Result: {status}")
                print(f"  Pass rate: {history_results.pass_rate:.1%}")
                print(f"  Mean effect: {history_results.mean_effect_size:.4f}")
        else:
            if verbose:
                print("\n[3/3] History Intervention Test... SKIPPED (no topic labels)")

        # Summary
        if verbose:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            all_passed = all(r.passed for r in results.values())
            for name, res in results.items():
                status = "PASS" if res.passed else "FAIL"
                print(f"  {name}: {status} ({res.pass_rate:.1%})")
            print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")

        return results

    def verify_model(
        self,
        reward_fn: RewardFunction,
        user_histories: np.ndarray,
        action_probs_batch: np.ndarray,
        post_topics: Optional[np.ndarray] = None,
        num_topics: int = 6,
    ) -> bool:
        """Quick verification that returns True/False.

        Args:
            reward_fn: Function (user_history, action_probs) -> reward
            user_histories: [N, num_features] user history features
            action_probs_batch: [N, num_actions] content action probabilities
            post_topics: [N] topic index for each post
            num_topics: Total number of topics

        Returns:
            True if all causal tests pass
        """
        results = self.run_all(
            reward_fn, user_histories, action_probs_batch,
            post_topics, num_topics, verbose=False
        )
        return all(r.passed for r in results.values())


def create_reward_fn_from_two_stage(state: Any) -> RewardFunction:
    """Create a reward function from TwoStageState.

    Args:
        state: TwoStageState or TwoStageGMMState

    Returns:
        Reward function compatible with causal tests
    """
    from enhancements.reward_modeling.two_stage import compute_reward

    def reward_fn(user_history: np.ndarray, action_probs: np.ndarray) -> float:
        return compute_reward(state, user_history, action_probs)

    return reward_fn


def create_reward_fn_from_weights(weights: np.ndarray) -> RewardFunction:
    """Create a simple reward function from weight vector.

    R = weights · action_probs

    Args:
        weights: [num_actions] reward weight vector

    Returns:
        Reward function compatible with causal tests
    """
    def reward_fn(_user_history: np.ndarray, action_probs: np.ndarray) -> float:
        # Note: Simple weight-based reward doesn't use user history
        return float(action_probs @ weights)

    return reward_fn
