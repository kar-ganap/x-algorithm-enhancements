"""F4: RL Reward Modeling with preference learning on engagement data.

This module provides reward modeling capabilities for Phoenix:
- PhoenixRewardModel: Computes scalar rewards from action probabilities
- RewardWeights: Learnable/configurable weights for reward computation
- PreferencePair: Data structure for pairwise preference learning

Phase 0 (Basic): Fixed-weight reward computation
Phase 1+: Learned weights via preference training

Example:
    >>> from enhancements.optimization import OptimizedPhoenixRunner
    >>> from enhancements.reward_modeling import PhoenixRewardModel, RewardWeights
    >>>
    >>> # Create reward model with F2's optimized runner
    >>> runner = OptimizedPhoenixRunner(config)
    >>> runner.initialize()
    >>> reward_model = PhoenixRewardModel(runner)
    >>>
    >>> # Compute rewards for candidates
    >>> rewards = reward_model.compute_reward(batch, embeddings)
    >>> best_idx = reward_model.get_best_candidate(batch, embeddings)
"""

from enhancements.reward_modeling.preference_data import (
    PreferenceBatch,
    PreferencePair,
    create_preferences_from_ground_truth,
    create_preferences_from_rewards,
)
from enhancements.reward_modeling.reward_model import PhoenixRewardModel
from enhancements.reward_modeling.weights import (
    ACTION_INDICES,
    ACTION_NAMES,
    NUM_ACTIONS,
    RewardWeights,
)

__all__ = [
    # Core classes
    "PhoenixRewardModel",
    "RewardWeights",
    # Preference data
    "PreferencePair",
    "PreferenceBatch",
    "create_preferences_from_rewards",
    "create_preferences_from_ground_truth",
    # Constants
    "ACTION_NAMES",
    "ACTION_INDICES",
    "NUM_ACTIONS",
]
