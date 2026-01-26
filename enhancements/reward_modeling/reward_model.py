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
