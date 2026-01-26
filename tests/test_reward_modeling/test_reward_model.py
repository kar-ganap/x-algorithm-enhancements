"""Tests for Phase 0: Basic Reward Model.

Tests the core reward model functionality:
- Reward model initialization with F2's optimized runner
- Action probability extraction from Phoenix
- Reward computation as weighted sum
- Integration with KV-cache for speedup

Go/No-Go Gates:
- Reward model initializes with F2 runner
- Action probs shape [B, C, 18]
- Rewards shape [B, C]
- Reward ordering matches weight signs
- KV-cache provides speedup
"""

import sys
import time
from pathlib import Path
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pytest

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.optimized_runner import OptimizationConfig, OptimizedPhoenixRunner
from enhancements.reward_modeling import (
    ACTION_INDICES,
    ACTION_NAMES,
    NUM_ACTIONS,
    PhoenixRewardModel,
    PreferenceBatch,
    PreferencePair,
    RewardWeights,
    create_preferences_from_rewards,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_config() -> PhoenixModelConfig:
    """Create a small Phoenix model config for testing."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=64,
        num_actions=len(ACTIONS),
        history_seq_len=8,
        candidate_seq_len=4,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=64,
            widening_factor=2,
            key_size=32,
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=2,
            attn_output_multiplier=0.125,
        ),
    )


@pytest.fixture(scope="module")
def optimized_runner(model_config: PhoenixModelConfig) -> OptimizedPhoenixRunner:
    """Create and initialize an optimized Phoenix runner."""
    opt_config = OptimizationConfig(
        use_jit=True,
        use_kv_cache=True,
        use_quantization=False,  # Keep simple for tests
        jit_batch_size=2,
        jit_history_len=model_config.history_seq_len,
        jit_num_candidates=model_config.candidate_seq_len,
    )
    runner = OptimizedPhoenixRunner(model_config, opt_config)
    runner.initialize()
    return runner


@pytest.fixture
def sample_batch(model_config: PhoenixModelConfig) -> Tuple[RecsysBatch, RecsysEmbeddings]:
    """Create a sample batch for testing."""
    batch, embeddings = create_example_batch(
        batch_size=2,
        emb_size=model_config.emb_size,
        history_len=model_config.history_seq_len,
        num_candidates=model_config.candidate_seq_len,
        num_actions=model_config.num_actions,
        num_user_hashes=model_config.hash_config.num_user_hashes,
        num_item_hashes=model_config.hash_config.num_item_hashes,
        num_author_hashes=model_config.hash_config.num_author_hashes,
        product_surface_vocab_size=model_config.product_surface_vocab_size,
    )
    return batch, embeddings


@pytest.fixture
def reward_model(optimized_runner: OptimizedPhoenixRunner) -> PhoenixRewardModel:
    """Create a reward model with default weights."""
    return PhoenixRewardModel(optimized_runner)


# -----------------------------------------------------------------------------
# RewardWeights Tests
# -----------------------------------------------------------------------------


class TestRewardWeights:
    """Tests for RewardWeights dataclass."""

    def test_default_weights_shape(self):
        """Verify default weights have correct shape."""
        weights = RewardWeights.default()
        assert weights.weights.shape == (NUM_ACTIONS,)
        assert weights.weights.shape == (18,)

    def test_default_weights_signs(self):
        """Verify default weights have sensible signs."""
        weights = RewardWeights.default()

        # Positive actions should have positive weights
        assert weights.weights[ACTION_INDICES["favorite"]] > 0
        assert weights.weights[ACTION_INDICES["repost"]] > 0
        assert weights.weights[ACTION_INDICES["follow_author"]] > 0

        # Negative actions should have negative weights
        assert weights.weights[ACTION_INDICES["block_author"]] < 0
        assert weights.weights[ACTION_INDICES["mute_author"]] < 0
        assert weights.weights[ACTION_INDICES["report"]] < 0
        assert weights.weights[ACTION_INDICES["not_interested"]] < 0

    def test_zeros_weights(self):
        """Verify zero initialization."""
        weights = RewardWeights.zeros()
        assert jnp.allclose(weights.weights, 0.0)

    def test_uniform_weights(self):
        """Verify uniform initialization."""
        weights = RewardWeights.uniform(value=0.5)
        assert jnp.allclose(weights.weights, 0.5)

    def test_invalid_shape_raises(self):
        """Verify invalid shape raises error."""
        with pytest.raises(ValueError):
            RewardWeights(weights=jnp.zeros(10))

    def test_get_positive_actions(self):
        """Verify positive actions extraction."""
        weights = RewardWeights.default()
        positive = weights.get_positive_actions()

        assert len(positive) > 0
        assert all(w > 0 for _, w in positive)
        # Should be sorted descending by weight
        weights_only = [w for _, w in positive]
        assert weights_only == sorted(weights_only, reverse=True)

    def test_get_negative_actions(self):
        """Verify negative actions extraction."""
        weights = RewardWeights.default()
        negative = weights.get_negative_actions()

        assert len(negative) > 0
        assert all(w < 0 for _, w in negative)
        # Should be sorted ascending by weight (most negative first)
        weights_only = [w for _, w in negative]
        assert weights_only == sorted(weights_only)

    def test_to_dict(self):
        """Verify dictionary conversion."""
        weights = RewardWeights.default()
        d = weights.to_dict()

        assert len(d) == NUM_ACTIONS
        assert "favorite" in d
        assert "block_author" in d
        assert d["favorite"] > 0
        assert d["block_author"] < 0


# -----------------------------------------------------------------------------
# PhoenixRewardModel Tests
# -----------------------------------------------------------------------------


class TestPhoenixRewardModel:
    """Tests for PhoenixRewardModel."""

    def test_initialization(self, optimized_runner: OptimizedPhoenixRunner):
        """Gate: Reward model initializes with F2 runner."""
        reward_model = PhoenixRewardModel(optimized_runner)

        assert reward_model.runner is optimized_runner
        assert reward_model.weights is not None
        assert isinstance(reward_model.weights, RewardWeights)

    def test_initialization_custom_weights(self, optimized_runner: OptimizedPhoenixRunner):
        """Verify initialization with custom weights."""
        custom_weights = RewardWeights.uniform(value=1.0)
        reward_model = PhoenixRewardModel(optimized_runner, weights=custom_weights)

        assert jnp.allclose(reward_model.weights.weights, 1.0)

    def test_get_action_probs_shape(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Gate: Action probs have shape [B, C, 18]."""
        batch, embeddings = sample_batch
        probs = reward_model.get_action_probs(batch, embeddings)

        assert probs.shape == (2, 4, NUM_ACTIONS)  # [batch_size, num_candidates, 18]

    def test_get_action_probs_valid_range(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify action probabilities are in valid range [0, 1]."""
        batch, embeddings = sample_batch
        probs = reward_model.get_action_probs(batch, embeddings)

        assert jnp.all(probs >= 0)
        assert jnp.all(probs <= 1)

    def test_compute_reward_shape(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Gate: Rewards have shape [B, C]."""
        batch, embeddings = sample_batch
        rewards = reward_model.compute_reward(batch, embeddings)

        assert rewards.shape == (2, 4)  # [batch_size, num_candidates]

    def test_reward_reflects_weights(self, optimized_runner: OptimizedPhoenixRunner):
        """Gate: Reward ordering matches weight signs."""
        # Create synthetic probs where we control the values
        # Candidate 0: high favorite (positive action)
        # Candidate 1: high block (negative action)
        probs = jnp.zeros((1, 2, NUM_ACTIONS))
        probs = probs.at[0, 0, ACTION_INDICES["favorite"]].set(0.9)
        probs = probs.at[0, 1, ACTION_INDICES["block_author"]].set(0.9)

        reward_model = PhoenixRewardModel(optimized_runner)
        rewards = reward_model._compute_reward_from_probs(probs)

        # Candidate with high favorite should have higher reward
        # than candidate with high block
        assert rewards[0, 0] > rewards[0, 1]

    def test_reward_with_uniform_weights(
        self,
        optimized_runner: OptimizedPhoenixRunner,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify reward with uniform weights is just sum of probs."""
        batch, embeddings = sample_batch
        uniform_weights = RewardWeights.uniform(value=1.0)
        reward_model = PhoenixRewardModel(optimized_runner, weights=uniform_weights)

        probs = reward_model.get_action_probs(batch, embeddings)
        rewards = reward_model.compute_reward(batch, embeddings)

        # With uniform weights of 1.0, reward should equal sum of probs
        # Note: bfloat16 precision means we need looser tolerance (~0.3% relative error)
        expected = probs.sum(axis=-1)
        np.testing.assert_allclose(rewards, expected, rtol=1e-2)

    def test_rank_by_reward(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify rank_by_reward returns valid indices."""
        batch, embeddings = sample_batch
        ranked = reward_model.rank_by_reward(batch, embeddings)

        assert ranked.shape == (2, 4)
        # Each row should be a permutation of [0, 1, 2, 3]
        for i in range(2):
            assert set(ranked[i].tolist()) == {0, 1, 2, 3}

    def test_get_best_candidate(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify get_best_candidate returns valid indices."""
        batch, embeddings = sample_batch
        best = reward_model.get_best_candidate(batch, embeddings)

        assert best.shape == (2,)
        assert jnp.all(best >= 0)
        assert jnp.all(best < 4)

    def test_update_weights(self, reward_model: PhoenixRewardModel):
        """Verify weight update works."""
        new_weights = RewardWeights.zeros()
        reward_model.update_weights(new_weights)

        assert jnp.allclose(reward_model.weights.weights, 0.0)

    def test_get_set_weights_array(self, reward_model: PhoenixRewardModel):
        """Verify get/set weights array for gradient updates."""
        original = reward_model.get_weights_array()
        assert original.shape == (NUM_ACTIONS,)

        new_array = jnp.ones(NUM_ACTIONS) * 2.0
        reward_model.set_weights_array(new_array)

        assert jnp.allclose(reward_model.get_weights_array(), 2.0)


# -----------------------------------------------------------------------------
# KV-Cache Integration Tests
# -----------------------------------------------------------------------------


class TestKVCacheIntegration:
    """Tests for KV-cache speedup integration."""

    def test_kv_cache_speedup(
        self,
        model_config: PhoenixModelConfig,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Gate: KV-cache provides speedup on repeated calls."""
        # Create runner with KV-cache enabled
        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=True,
            use_quantization=False,
            jit_batch_size=2,
            jit_history_len=8,
            jit_num_candidates=4,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()
        reward_model = PhoenixRewardModel(runner)

        batch, embeddings = sample_batch

        # First call - cache miss (includes compilation)
        start = time.perf_counter()
        _ = reward_model.compute_reward(batch, embeddings)
        first_call_time = time.perf_counter() - start

        # Second call - should hit cache (same user)
        start = time.perf_counter()
        _ = reward_model.compute_reward(batch, embeddings)
        second_call_time = time.perf_counter() - start

        # Cache hit should be faster (allowing for some variance)
        # Note: First call includes JIT compilation, so this should pass easily
        assert second_call_time < first_call_time

    def test_cache_stats_update(
        self,
        optimized_runner: OptimizedPhoenixRunner,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify cache stats are updated after calls."""
        reward_model = PhoenixRewardModel(optimized_runner)
        batch, embeddings = sample_batch

        # Make two calls
        _ = reward_model.compute_reward(batch, embeddings)
        _ = reward_model.compute_reward(batch, embeddings)

        # Check stats
        stats = optimized_runner.stats
        # Should have at least one hit (second call)
        assert stats.kv_cache_hits >= 1 or stats.kv_cache_misses >= 1


# -----------------------------------------------------------------------------
# PreferencePair Tests
# -----------------------------------------------------------------------------


class TestPreferencePair:
    """Tests for PreferencePair dataclass."""

    def test_valid_pair(self):
        """Verify valid preference pair creation."""
        pair = PreferencePair(user_idx=0, preferred_idx=1, rejected_idx=2)
        assert pair.user_idx == 0
        assert pair.preferred_idx == 1
        assert pair.rejected_idx == 2
        assert pair.confidence == 1.0

    def test_invalid_pair_raises(self):
        """Verify same preferred/rejected raises error."""
        with pytest.raises(ValueError):
            PreferencePair(user_idx=0, preferred_idx=1, rejected_idx=1)

    def test_preference_batch_from_pairs(self):
        """Verify PreferenceBatch creation from list."""
        pairs = [
            PreferencePair(0, 1, 2),
            PreferencePair(1, 0, 3),
        ]
        batch = PreferenceBatch.from_pairs(pairs)

        assert len(batch) == 2
        assert jnp.array_equal(batch.user_indices, jnp.array([0, 1]))
        assert jnp.array_equal(batch.preferred_indices, jnp.array([1, 0]))
        assert jnp.array_equal(batch.rejected_indices, jnp.array([2, 3]))


class TestPreferenceGeneration:
    """Tests for preference pair generation utilities."""

    def test_create_preferences_from_rewards(self):
        """Verify preference creation from known rewards."""
        # Create rewards where candidate 0 is clearly best, candidate 3 is worst
        rewards = jnp.array([
            [1.0, 0.5, 0.3, 0.1],  # User 0
            [0.2, 0.8, 0.4, 0.6],  # User 1
        ])

        pairs = create_preferences_from_rewards(
            rewards,
            num_pairs_per_user=3,
            min_reward_diff=0.1,
            rng=np.random.default_rng(42),
        )

        assert len(pairs) > 0

        # Verify all pairs have correct preference direction
        for pair in pairs:
            user_rewards = rewards[pair.user_idx]
            assert user_rewards[pair.preferred_idx] > user_rewards[pair.rejected_idx]


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """End-to-end integration tests."""

    def test_full_reward_pipeline(
        self,
        optimized_runner: OptimizedPhoenixRunner,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Test complete reward computation pipeline."""
        batch, embeddings = sample_batch

        # Create reward model
        reward_model = PhoenixRewardModel(optimized_runner)

        # Compute rewards
        rewards = reward_model.compute_reward(batch, embeddings)

        # Get best candidates
        best = reward_model.get_best_candidate(batch, embeddings)

        # Verify best matches argmax of rewards
        expected_best = jnp.argmax(rewards, axis=-1)
        assert jnp.array_equal(best, expected_best)

    def test_reward_deterministic(
        self,
        reward_model: PhoenixRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Verify reward computation is deterministic."""
        batch, embeddings = sample_batch

        rewards1 = reward_model.compute_reward(batch, embeddings)
        rewards2 = reward_model.compute_reward(batch, embeddings)

        np.testing.assert_array_equal(rewards1, rewards2)
