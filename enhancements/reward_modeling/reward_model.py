"""Phoenix Reward Model for F4.

Wraps Phoenix model to compute scalar rewards from action probabilities.
Integrates with F2's OptimizedPhoenixRunner for fast inference.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    PhoenixRewardModel                        │
    │                                                              │
    │  ┌──────────────┐      ┌──────────────┐      ┌───────────┐  │
    │  │    Batch     │ ──▶  │   Phoenix    │ ──▶  │  Weights  │  │
    │  │  Embeddings  │      │  (F2 Runner) │      │  [18]     │  │
    │  └──────────────┘      └──────────────┘      └───────────┘  │
    │                              │                     │         │
    │                              ▼                     │         │
    │                       P(actions)                   │         │
    │                       [B, C, 18]                   │         │
    │                              │                     │         │
    │                              ▼                     ▼         │
    │                        R = w · P(actions)                   │
    │                           [B, C]                            │
    └─────────────────────────────────────────────────────────────┘
"""

from typing import Optional, Union

import jax
import jax.numpy as jnp

from phoenix.recsys_model import RecsysBatch, RecsysEmbeddings
from phoenix.runners import RankingOutput, RecsysInferenceRunner

from enhancements.reward_modeling.weights import NUM_ACTIONS, RewardWeights


class PhoenixRewardModel:
    """Reward model that wraps Phoenix for RL training.

    Computes scalar rewards from Phoenix's action probability predictions
    using a weighted sum: R = w · P(actions).

    Integrates with F2's OptimizedPhoenixRunner for KV-cache speedup.

    Example:
        >>> from enhancements.optimization import OptimizedPhoenixRunner
        >>> runner = OptimizedPhoenixRunner(config)
        >>> runner.initialize()
        >>> reward_model = PhoenixRewardModel(runner)
        >>> rewards = reward_model.compute_reward(batch, embeddings)
        >>> # rewards.shape == (batch_size, num_candidates)

    Attributes:
        runner: Phoenix runner (base or optimized)
        weights: Reward weights for combining action probabilities
    """

    def __init__(
        self,
        runner: Union[RecsysInferenceRunner, "OptimizedPhoenixRunner"],
        weights: Optional[RewardWeights] = None,
    ):
        """Initialize the reward model.

        Args:
            runner: Phoenix runner for inference. Can be base RecsysInferenceRunner
                or F2's OptimizedPhoenixRunner (recommended for speed).
            weights: Reward weights. If None, uses sensible defaults.
        """
        self.runner = runner
        self.weights = weights if weights is not None else RewardWeights.default()

    def get_action_probs(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jnp.ndarray:
        """Get action probabilities from Phoenix.

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            Action probabilities of shape [batch_size, num_candidates, 18]
        """
        output: RankingOutput = self.runner.rank(batch, embeddings)

        # RankingOutput.scores contains all action probabilities
        # Shape: [batch_size, num_candidates, num_actions]
        scores = output.scores

        # Ensure we only take the 18 action probabilities (not dwell_time if present)
        if scores.shape[-1] > NUM_ACTIONS:
            scores = scores[..., :NUM_ACTIONS]

        return scores

    def compute_reward(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jnp.ndarray:
        """Compute scalar reward for each candidate.

        Reward is computed as weighted sum of action probabilities:
            R = Σᵢ wᵢ × P(actionᵢ)

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            Rewards of shape [batch_size, num_candidates]
        """
        probs = self.get_action_probs(batch, embeddings)
        return self._compute_reward_from_probs(probs)

    def _compute_reward_from_probs(self, probs: jnp.ndarray) -> jnp.ndarray:
        """Compute reward from action probabilities.

        This is separated out for testing and potential subclass overrides.

        Args:
            probs: Action probabilities of shape [B, C, 18]

        Returns:
            Rewards of shape [B, C]
        """
        # Weighted sum: R[b,c] = Σᵢ w[i] × P[b,c,i]
        # Using einsum for clarity: 'bca,a->bc'
        rewards = jnp.einsum("bca,a->bc", probs, self.weights.weights)
        return rewards

    def rank_by_reward(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jnp.ndarray:
        """Rank candidates by reward (descending).

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            Indices that would sort candidates by reward (highest first)
            Shape: [batch_size, num_candidates]
        """
        rewards = self.compute_reward(batch, embeddings)
        # argsort ascending, then reverse for descending
        return jnp.argsort(-rewards, axis=-1)

    def get_best_candidate(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jnp.ndarray:
        """Get index of highest-reward candidate for each batch item.

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            Indices of best candidates, shape [batch_size]
        """
        rewards = self.compute_reward(batch, embeddings)
        return jnp.argmax(rewards, axis=-1)

    def update_weights(self, new_weights: RewardWeights) -> None:
        """Update the reward weights.

        Args:
            new_weights: New weights to use
        """
        self.weights = new_weights

    def get_weights_array(self) -> jnp.ndarray:
        """Get weights as JAX array (for gradient computation)."""
        return self.weights.weights

    def set_weights_array(self, weights_array: jnp.ndarray) -> None:
        """Set weights from JAX array (after gradient update)."""
        self.weights = RewardWeights(weights=weights_array)


class ContextualRewardModel:
    """Reward model with archetype-specific learned weights.

    Extends PhoenixRewardModel to learn K weight vectors, one per user archetype.
    This enables the model to capture different value systems across user types.

    Architecture:
        ┌─────────────────────────────────────────────────────────────┐
        │                  ContextualRewardModel                       │
        │                                                              │
        │  weights: [K, 18]     K weight vectors (one per archetype)  │
        │                                                              │
        │  For user with archetype k:                                 │
        │      R[b,c] = weights[k] · P[b,c,:]                         │
        └─────────────────────────────────────────────────────────────┘

    Example:
        >>> model = ContextualRewardModel(runner, num_archetypes=6)
        >>> # archetype_ids[i] = archetype index for user i
        >>> rewards = model.compute_reward(batch, embeddings, archetype_ids)
    """

    def __init__(
        self,
        runner: Union[RecsysInferenceRunner, "OptimizedPhoenixRunner"],
        num_archetypes: int = 6,
        num_actions: int = NUM_ACTIONS,
    ):
        """Initialize the contextual reward model.

        Args:
            runner: Phoenix runner for inference
            num_archetypes: Number of user archetypes (K)
            num_actions: Number of actions (default 18)
        """
        self.runner = runner
        self.num_archetypes = num_archetypes
        self.num_actions = num_actions

        # Weight matrix: [K, num_actions] - one weight vector per archetype
        # Initialize with small random values to break symmetry
        self.weights = jnp.zeros((num_archetypes, num_actions))

    def initialize_from_default(self) -> None:
        """Initialize all archetype weights from default single-weight values.

        Useful starting point before training - all archetypes start with
        the same sensible defaults, then differentiate through learning.
        """
        from enhancements.reward_modeling.weights import RewardWeights

        default = RewardWeights.default()
        self.weights = jnp.tile(default.weights[jnp.newaxis, :], (self.num_archetypes, 1))

    def initialize_random(self, rng: jnp.ndarray, scale: float = 0.1) -> None:
        """Initialize weights with small random values.

        Args:
            rng: JAX random key
            scale: Standard deviation of initialization
        """
        self.weights = jax.random.normal(rng, (self.num_archetypes, self.num_actions)) * scale

    def get_action_probs(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> jnp.ndarray:
        """Get action probabilities from Phoenix.

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            Action probabilities of shape [batch_size, num_candidates, num_actions]
        """
        output = self.runner.rank(batch, embeddings)
        scores = output.scores

        if scores.shape[-1] > self.num_actions:
            scores = scores[..., :self.num_actions]

        return scores

    def compute_reward(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
        archetype_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute reward using archetype-specific weights.

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings
            archetype_ids: Archetype index for each user [batch_size]

        Returns:
            Rewards of shape [batch_size, num_candidates]
        """
        probs = self.get_action_probs(batch, embeddings)  # [B, C, num_actions]
        return self._compute_reward_from_probs(probs, archetype_ids)

    def _compute_reward_from_probs(
        self,
        probs: jnp.ndarray,
        archetype_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute reward from action probabilities with archetype-specific weights.

        Args:
            probs: Action probabilities [B, C, num_actions]
            archetype_ids: Archetype index for each user [B]

        Returns:
            Rewards of shape [B, C]
        """
        # Select weights for each user's archetype: [B, num_actions]
        user_weights = self.weights[archetype_ids]

        # Compute reward: R[b,c] = Σᵢ w[archetype[b], i] × P[b,c,i]
        # Using einsum: 'bca,ba->bc'
        rewards = jnp.einsum("bca,ba->bc", probs, user_weights)
        return rewards

    def rank_by_reward(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
        archetype_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Rank candidates by reward (descending).

        Args:
            batch: Input batch
            embeddings: Pre-computed embeddings
            archetype_ids: Archetype index for each user [batch_size]

        Returns:
            Indices that would sort candidates by reward (highest first)
        """
        rewards = self.compute_reward(batch, embeddings, archetype_ids)
        return jnp.argsort(-rewards, axis=-1)

    def get_best_candidate(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
        archetype_ids: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get index of highest-reward candidate for each user.

        Args:
            batch: Input batch
            embeddings: Pre-computed embeddings
            archetype_ids: Archetype index for each user [batch_size]

        Returns:
            Indices of best candidates [batch_size]
        """
        rewards = self.compute_reward(batch, embeddings, archetype_ids)
        return jnp.argmax(rewards, axis=-1)

    def get_weights_for_archetype(self, archetype_id: int) -> jnp.ndarray:
        """Get weight vector for a specific archetype.

        Args:
            archetype_id: Archetype index

        Returns:
            Weight vector [num_actions]
        """
        return self.weights[archetype_id]

    def set_weights_for_archetype(
        self,
        archetype_id: int,
        weights: jnp.ndarray,
    ) -> None:
        """Set weight vector for a specific archetype.

        Args:
            archetype_id: Archetype index
            weights: New weight vector [num_actions]
        """
        self.weights = self.weights.at[archetype_id].set(weights)

    def get_weights_array(self) -> jnp.ndarray:
        """Get all weights as JAX array (for gradient computation)."""
        return self.weights

    def set_weights_array(self, weights: jnp.ndarray) -> None:
        """Set all weights from JAX array (after gradient update)."""
        self.weights = weights

    def weights_cosine_similarity(self) -> jnp.ndarray:
        """Compute pairwise cosine similarity between archetype weights.

        Returns:
            Similarity matrix [K, K]
        """
        # Normalize weights
        norms = jnp.linalg.norm(self.weights, axis=1, keepdims=True)
        normalized = self.weights / (norms + 1e-8)
        # Cosine similarity
        return jnp.dot(normalized, normalized.T)

    def weights_are_differentiated(self, threshold: float = 0.95) -> bool:
        """Check if archetype weights have differentiated (not all identical).

        Args:
            threshold: Maximum allowed average cosine similarity

        Returns:
            True if weights are differentiated
        """
        sim = self.weights_cosine_similarity()
        # Mask diagonal (self-similarity)
        mask = 1.0 - jnp.eye(self.num_archetypes)
        avg_sim = jnp.sum(sim * mask) / jnp.sum(mask)
        return float(avg_sim) < threshold
