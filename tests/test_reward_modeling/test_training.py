"""Tests for F4 Phase 1: Bradley-Terry Training.

Tests the contextual reward model and preference learning:
- ContextualRewardModel with per-archetype weights
- Bradley-Terry loss computation
- Training loop with weight differentiation
- Integration with F2 synthetic data

Go/No-Go Gates:
- Contextual rewards compute correctly
- Different archetypes produce different rewards
- Training loss decreases
- Weights differentiate across archetypes after training
"""

import sys
from pathlib import Path
from typing import Tuple

import jax
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
    NUM_ACTIONS,
    ContextualRewardModel,
    TrainingConfig,
    bradley_terry_loss,
    compute_preference_accuracy,
    contextual_bradley_terry_loss,
    create_synthetic_preference_batch,
    train_contextual_weights,
    train_single_weights,
)
from enhancements.reward_modeling.weights import ACTION_INDICES, RewardWeights


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
        use_quantization=False,
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
def contextual_model(optimized_runner: OptimizedPhoenixRunner) -> ContextualRewardModel:
    """Create a contextual reward model."""
    model = ContextualRewardModel(optimized_runner, num_archetypes=6)
    model.initialize_from_default()
    return model


# -----------------------------------------------------------------------------
# ContextualRewardModel Tests
# -----------------------------------------------------------------------------


class TestContextualRewardModel:
    """Tests for ContextualRewardModel."""

    def test_initialization(self, optimized_runner: OptimizedPhoenixRunner):
        """Verify contextual model initializes correctly."""
        model = ContextualRewardModel(optimized_runner, num_archetypes=6)

        assert model.weights.shape == (6, NUM_ACTIONS)
        assert model.num_archetypes == 6
        assert model.num_actions == NUM_ACTIONS

    def test_initialize_from_default(self, optimized_runner: OptimizedPhoenixRunner):
        """Verify initialization from default weights."""
        model = ContextualRewardModel(optimized_runner, num_archetypes=6)
        model.initialize_from_default()

        # All archetypes should have same initial weights
        default = RewardWeights.default()
        for k in range(6):
            np.testing.assert_array_equal(model.weights[k], default.weights)

    def test_initialize_random(self, optimized_runner: OptimizedPhoenixRunner):
        """Verify random initialization."""
        model = ContextualRewardModel(optimized_runner, num_archetypes=6)
        rng = jax.random.PRNGKey(42)
        model.initialize_random(rng, scale=0.1)

        # Weights should not be all zeros
        assert jnp.abs(model.weights).sum() > 0
        # Weights should be small (scaled)
        assert jnp.abs(model.weights).max() < 1.0

    def test_compute_reward_shape(
        self,
        contextual_model: ContextualRewardModel,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Gate: Contextual rewards have correct shape."""
        batch, embeddings = sample_batch
        archetype_ids = jnp.array([0, 1])  # Two users, different archetypes

        rewards = contextual_model.compute_reward(batch, embeddings, archetype_ids)

        assert rewards.shape == (2, 4)  # [batch_size, num_candidates]

    def test_different_archetypes_different_rewards(
        self,
        optimized_runner: OptimizedPhoenixRunner,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Gate: Different archetypes produce different rewards for same content."""
        batch, embeddings = sample_batch

        # Create model with differentiated weights
        model = ContextualRewardModel(optimized_runner, num_archetypes=6)

        # Set archetype 0 to favor favorites, archetype 1 to favor blocks
        w0 = jnp.zeros(NUM_ACTIONS)
        w0 = w0.at[ACTION_INDICES["favorite"]].set(1.0)
        w1 = jnp.zeros(NUM_ACTIONS)
        w1 = w1.at[ACTION_INDICES["block_author"]].set(1.0)

        model.set_weights_for_archetype(0, w0)
        model.set_weights_for_archetype(1, w1)

        # Same batch, different archetypes
        rewards_arch0 = model.compute_reward(batch, embeddings, jnp.array([0, 0]))
        rewards_arch1 = model.compute_reward(batch, embeddings, jnp.array([1, 1]))

        # Rewards should differ
        assert not jnp.allclose(rewards_arch0, rewards_arch1)

    def test_get_set_weights_for_archetype(
        self,
        contextual_model: ContextualRewardModel,
    ):
        """Verify per-archetype weight get/set."""
        new_weights = jnp.ones(NUM_ACTIONS) * 0.5
        contextual_model.set_weights_for_archetype(2, new_weights)

        retrieved = contextual_model.get_weights_for_archetype(2)
        np.testing.assert_array_equal(retrieved, new_weights)

        # Other archetypes should be unchanged
        w0 = contextual_model.get_weights_for_archetype(0)
        assert not jnp.allclose(w0, new_weights)

    def test_weights_cosine_similarity(
        self,
        optimized_runner: OptimizedPhoenixRunner,
    ):
        """Verify cosine similarity computation."""
        model = ContextualRewardModel(optimized_runner, num_archetypes=3)

        # Set orthogonal weights
        model.weights = jnp.array([
            [1.0, 0.0, 0.0] + [0.0] * (NUM_ACTIONS - 3),
            [0.0, 1.0, 0.0] + [0.0] * (NUM_ACTIONS - 3),
            [0.0, 0.0, 1.0] + [0.0] * (NUM_ACTIONS - 3),
        ])

        sim = model.weights_cosine_similarity()

        # Diagonal should be 1.0
        np.testing.assert_allclose(jnp.diag(sim), 1.0, atol=1e-5)
        # Off-diagonal should be 0.0 for orthogonal vectors
        mask = 1.0 - jnp.eye(3)
        np.testing.assert_allclose(sim * mask, 0.0, atol=1e-5)

    def test_weights_are_differentiated(
        self,
        contextual_model: ContextualRewardModel,
    ):
        """Verify differentiation check works."""
        # Initially all same (from default) - not differentiated
        contextual_model.initialize_from_default()
        assert not contextual_model.weights_are_differentiated(threshold=0.99)

        # Make weights different
        for k in range(contextual_model.num_archetypes):
            contextual_model.weights = contextual_model.weights.at[k, k % NUM_ACTIONS].set(
                float(k)
            )
        assert contextual_model.weights_are_differentiated(threshold=0.99)


# -----------------------------------------------------------------------------
# Bradley-Terry Loss Tests
# -----------------------------------------------------------------------------


class TestBradleyTerryLoss:
    """Tests for Bradley-Terry loss computation."""

    def test_loss_basic(self):
        """Verify basic loss computation."""
        weights = jnp.array([1.0, 0.0, -1.0] + [0.0] * (NUM_ACTIONS - 3))

        # Preferred has high positive action
        probs_pref = jnp.array([[0.9, 0.0, 0.0] + [0.0] * (NUM_ACTIONS - 3)])
        # Rejected has high negative action
        probs_rej = jnp.array([[0.0, 0.0, 0.9] + [0.0] * (NUM_ACTIONS - 3)])

        loss = bradley_terry_loss(weights, probs_pref, probs_rej)

        # R(pref) = 0.9, R(rej) = -0.9, diff = 1.8
        # Loss should be small since preferred is clearly better
        assert loss < 0.5

    def test_loss_with_wrong_order(self):
        """Verify loss is high when preferences are reversed."""
        weights = jnp.array([1.0, 0.0, -1.0] + [0.0] * (NUM_ACTIONS - 3))

        # Now swap: "preferred" has negative action
        probs_pref = jnp.array([[0.0, 0.0, 0.9] + [0.0] * (NUM_ACTIONS - 3)])
        probs_rej = jnp.array([[0.9, 0.0, 0.0] + [0.0] * (NUM_ACTIONS - 3)])

        loss = bradley_terry_loss(weights, probs_pref, probs_rej)

        # Loss should be high since order is wrong
        assert loss > 1.0

    def test_contextual_loss(self):
        """Verify contextual Bradley-Terry loss."""
        num_archetypes = 3
        weights = jnp.zeros((num_archetypes, NUM_ACTIONS))
        weights = weights.at[0, 0].set(1.0)  # Archetype 0 likes action 0
        weights = weights.at[1, 1].set(1.0)  # Archetype 1 likes action 1
        weights = weights.at[2, 2].set(1.0)  # Archetype 2 likes action 2

        # Preferred items have high prob for their archetype's favorite action
        probs_pref = jnp.zeros((3, NUM_ACTIONS))
        probs_pref = probs_pref.at[0, 0].set(0.9)
        probs_pref = probs_pref.at[1, 1].set(0.9)
        probs_pref = probs_pref.at[2, 2].set(0.9)

        probs_rej = jnp.zeros((3, NUM_ACTIONS))
        probs_rej = probs_rej.at[:, 17].set(0.9)  # All rejected have high report

        archetype_ids = jnp.array([0, 1, 2])

        loss = contextual_bradley_terry_loss(
            weights, probs_pref, probs_rej, archetype_ids
        )

        # Loss should be small since each archetype prefers the right action
        assert loss < 0.5


# -----------------------------------------------------------------------------
# Training Tests
# -----------------------------------------------------------------------------


class TestTraining:
    """Tests for training loops."""

    def test_create_synthetic_batch(self):
        """Verify synthetic batch creation."""
        probs_pref, probs_rej, conf = create_synthetic_preference_batch(
            num_samples=10, num_actions=NUM_ACTIONS
        )

        assert probs_pref.shape == (10, NUM_ACTIONS)
        assert probs_rej.shape == (10, NUM_ACTIONS)
        assert conf.shape == (10,)

        # Probs should be in [0, 1]
        assert jnp.all(probs_pref >= 0) and jnp.all(probs_pref <= 1)
        assert jnp.all(probs_rej >= 0) and jnp.all(probs_rej <= 1)

    def test_compute_preference_accuracy(self):
        """Verify accuracy computation."""
        weights = jnp.array([1.0] + [0.0] * (NUM_ACTIONS - 1))

        # Preferred always has higher action 0
        probs_pref = jnp.array([
            [0.9] + [0.0] * (NUM_ACTIONS - 1),
            [0.8] + [0.0] * (NUM_ACTIONS - 1),
        ])
        probs_rej = jnp.array([
            [0.1] + [0.0] * (NUM_ACTIONS - 1),
            [0.2] + [0.0] * (NUM_ACTIONS - 1),
        ])

        acc = compute_preference_accuracy(weights, probs_pref, probs_rej)
        assert acc == 1.0  # Perfect accuracy

    def test_train_single_weights_loss_decreases(self):
        """Gate: Training loss decreases."""
        initial_weights = jnp.zeros(NUM_ACTIONS)

        rng = np.random.default_rng(42)

        def get_batch():
            return create_synthetic_preference_batch(
                num_samples=16,
                num_actions=NUM_ACTIONS,
                rng=rng,
            )

        config = TrainingConfig(
            learning_rate=0.1,
            num_epochs=10,
            batch_size=5,
            log_every=100,  # Suppress logging
        )

        metrics = train_single_weights(
            initial_weights,
            get_batch,
            config,
            verbose=False,
        )

        # Loss should decrease
        assert metrics.loss_history[-1] < metrics.loss_history[0]
        # Final loss should be reasonable
        assert metrics.loss_history[-1] < 1.0

    def test_train_contextual_weights_loss_decreases(self):
        """Gate: Contextual training loss decreases."""
        num_archetypes = 3
        initial_weights = jnp.zeros((num_archetypes, NUM_ACTIONS))

        rng = np.random.default_rng(42)

        def get_batch():
            probs_pref, probs_rej, conf = create_synthetic_preference_batch(
                num_samples=16,
                num_actions=NUM_ACTIONS,
                rng=rng,
            )
            # Random archetype assignments
            arch_ids = jnp.array(rng.integers(0, num_archetypes, size=16))
            return probs_pref, probs_rej, arch_ids, conf

        config = TrainingConfig(
            learning_rate=0.1,
            num_epochs=10,
            batch_size=5,
            log_every=100,
        )

        metrics = train_contextual_weights(
            initial_weights,
            get_batch,
            config,
            verbose=False,
        )

        # Loss should decrease
        assert metrics.loss_history[-1] < metrics.loss_history[0]

    def test_train_contextual_weights_differentiate(self):
        """Gate: Weights differentiate across archetypes after training."""
        num_archetypes = 3
        # Start from default (all same)
        default = RewardWeights.default()
        initial_weights = jnp.tile(
            default.weights[jnp.newaxis, :],
            (num_archetypes, 1),
        )

        rng = np.random.default_rng(42)

        # Create archetype-specific preferences
        def get_batch():
            batch_size = 16
            probs_pref = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
            probs_rej = np.zeros((batch_size, NUM_ACTIONS), dtype=np.float32)
            arch_ids = np.zeros(batch_size, dtype=np.int32)

            for i in range(batch_size):
                arch = i % num_archetypes
                arch_ids[i] = arch
                # Each archetype prefers different action
                probs_pref[i, arch] = 0.9  # Preferred has high prob for archetype's fav
                probs_rej[i, (arch + 1) % NUM_ACTIONS] = 0.9  # Rejected has different

            return (
                jnp.array(probs_pref),
                jnp.array(probs_rej),
                jnp.array(arch_ids),
                jnp.ones(batch_size),
            )

        config = TrainingConfig(
            learning_rate=0.1,
            num_epochs=30,
            batch_size=10,
            log_every=100,
        )

        metrics = train_contextual_weights(
            initial_weights,
            get_batch,
            config,
            verbose=False,
        )

        # Check weights have differentiated
        final_weights = metrics.final_weights

        # Compute pairwise differences
        for i in range(num_archetypes):
            for j in range(i + 1, num_archetypes):
                # Weights should not be identical
                assert not jnp.allclose(
                    final_weights[i], final_weights[j], atol=0.01
                ), f"Archetypes {i} and {j} have same weights"


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining model and training."""

    def test_contextual_model_training_pipeline(
        self,
        optimized_runner: OptimizedPhoenixRunner,
        sample_batch: Tuple[RecsysBatch, RecsysEmbeddings],
    ):
        """Test full training pipeline with contextual model."""
        batch, embeddings = sample_batch

        # Create model
        model = ContextualRewardModel(optimized_runner, num_archetypes=3)
        model.initialize_from_default()

        # Get initial rewards
        archetype_ids = jnp.array([0, 1])
        initial_rewards = model.compute_reward(batch, embeddings, archetype_ids)

        # Train with synthetic data
        rng = np.random.default_rng(42)

        def get_batch():
            probs_pref, probs_rej, conf = create_synthetic_preference_batch(
                num_samples=8, num_actions=NUM_ACTIONS, rng=rng
            )
            arch_ids = jnp.array(rng.integers(0, 3, size=8))
            return probs_pref, probs_rej, arch_ids, conf

        config = TrainingConfig(
            learning_rate=0.05,
            num_epochs=5,
            batch_size=3,
            log_every=100,
        )

        metrics = train_contextual_weights(
            model.weights,
            get_batch,
            config,
            verbose=False,
        )

        # Update model weights
        model.set_weights_array(metrics.final_weights)

        # Compute new rewards
        new_rewards = model.compute_reward(batch, embeddings, archetype_ids)

        # Rewards should have changed after training
        assert not jnp.allclose(initial_rewards, new_rewards)

    def test_training_improves_accuracy(self):
        """Verify training improves preference accuracy."""
        num_archetypes = 3
        rng = np.random.default_rng(42)

        # Create consistent test data
        test_probs_pref, test_probs_rej, _ = create_synthetic_preference_batch(
            num_samples=50, num_actions=NUM_ACTIONS, rng=np.random.default_rng(123)
        )
        test_arch_ids = jnp.array([i % num_archetypes for i in range(50)])

        # Initial random weights
        initial_weights = jax.random.normal(
            jax.random.PRNGKey(42),
            (num_archetypes, NUM_ACTIONS),
        ) * 0.1

        initial_acc = compute_preference_accuracy(
            initial_weights, test_probs_pref, test_probs_rej, test_arch_ids
        )

        # Train
        def get_batch():
            probs_pref, probs_rej, conf = create_synthetic_preference_batch(
                num_samples=16, num_actions=NUM_ACTIONS, rng=rng
            )
            arch_ids = jnp.array(rng.integers(0, num_archetypes, size=16))
            return probs_pref, probs_rej, arch_ids, conf

        config = TrainingConfig(
            learning_rate=0.1,
            num_epochs=20,
            batch_size=10,
            log_every=100,
        )

        metrics = train_contextual_weights(
            initial_weights,
            get_batch,
            config,
            verbose=False,
        )

        final_acc = compute_preference_accuracy(
            metrics.final_weights, test_probs_pref, test_probs_rej, test_arch_ids
        )

        # Accuracy should improve
        assert final_acc >= initial_acc
