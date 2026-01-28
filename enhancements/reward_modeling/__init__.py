"""F4: RL Reward Modeling with preference learning on engagement data.

This module provides reward modeling capabilities for Phoenix:
- PhoenixRewardModel: Computes scalar rewards from action probabilities (Phase 0)
- ContextualRewardModel: Per-archetype learned weights (Phase 1)
- RewardWeights: Learnable/configurable weights for reward computation
- PreferencePair: Data structure for pairwise preference learning

Phase 0 (Basic): Fixed-weight reward computation
Phase 1 (Contextual): Per-archetype learned weights via Bradley-Terry
Phase 2+: Pluralistic rewards, causal verification, multi-stakeholder

Example:
    >>> from enhancements.optimization.optimized_runner import OptimizedPhoenixRunner
    >>> from enhancements.reward_modeling import PhoenixRewardModel, RewardWeights
    >>>
    >>> # Phase 0: Basic reward model with fixed weights
    >>> runner = OptimizedPhoenixRunner(config)
    >>> runner.initialize()
    >>> reward_model = PhoenixRewardModel(runner)
    >>> rewards = reward_model.compute_reward(batch, embeddings)
    >>>
    >>> # Phase 1: Contextual model with per-archetype weights
    >>> from enhancements.reward_modeling import ContextualRewardModel
    >>> ctx_model = ContextualRewardModel(runner, num_archetypes=6)
    >>> ctx_model.initialize_from_default()
    >>> rewards = ctx_model.compute_reward(batch, embeddings, archetype_ids)
"""

from enhancements.reward_modeling.preference_data import (
    PreferenceBatch,
    PreferencePair,
    create_preferences_from_ground_truth,
    create_preferences_from_rewards,
)
from enhancements.reward_modeling.reward_model import (
    ContextualRewardModel,
    PhoenixRewardModel,
)
from enhancements.reward_modeling.training import (
    TrainingConfig,
    TrainingMetrics,
    TrainingState,
    bradley_terry_loss,
    compute_preference_accuracy,
    contextual_bradley_terry_loss,
    create_synthetic_preference_batch,
    train_contextual_weights,
    train_single_weights,
)
from enhancements.reward_modeling.weights import (
    ACTION_INDICES,
    ACTION_NAMES,
    NUM_ACTIONS,
    RewardWeights,
)
from enhancements.reward_modeling.pluralistic import (
    PluralConfig,
    PluralMetrics,
    PluralState,
    TrainingApproach,
    compute_mixture_weights,
    compute_pluralistic_reward,
    diversity_loss,
    entropy_loss,
    init_plural_state,
    train_auxiliary,
    train_em,
    train_hybrid,
    train_pluralistic,
)
from enhancements.reward_modeling.structural_recovery import (
    RecoveryMetrics,
    check_recovery_gates,
    compute_ground_truth_weights,
    measure_structural_recovery,
)
from enhancements.reward_modeling.causal_verification import (
    CausalTestConfig,
    CausalTestResults,
    CausalVerificationSuite,
    InterventionResult,
    InterventionType,
    create_reward_fn_from_two_stage,
    create_reward_fn_from_weights,
)

__all__ = [
    # Phase 0: Basic reward model
    "PhoenixRewardModel",
    "RewardWeights",
    # Phase 1: Contextual reward model
    "ContextualRewardModel",
    # Training
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingState",
    "bradley_terry_loss",
    "contextual_bradley_terry_loss",
    "train_single_weights",
    "train_contextual_weights",
    "compute_preference_accuracy",
    "create_synthetic_preference_batch",
    # Preference data
    "PreferencePair",
    "PreferenceBatch",
    "create_preferences_from_rewards",
    "create_preferences_from_ground_truth",
    # Constants
    "ACTION_NAMES",
    "ACTION_INDICES",
    "NUM_ACTIONS",
    # Phase 2: Pluralistic rewards
    "PluralConfig",
    "PluralMetrics",
    "PluralState",
    "TrainingApproach",
    "compute_mixture_weights",
    "compute_pluralistic_reward",
    "diversity_loss",
    "entropy_loss",
    "init_plural_state",
    "train_auxiliary",
    "train_em",
    "train_hybrid",
    "train_pluralistic",
    # Structural recovery
    "RecoveryMetrics",
    "check_recovery_gates",
    "compute_ground_truth_weights",
    "measure_structural_recovery",
    # Phase 3: Causal verification
    "CausalTestConfig",
    "CausalTestResults",
    "CausalVerificationSuite",
    "InterventionResult",
    "InterventionType",
    "create_reward_fn_from_two_stage",
    "create_reward_fn_from_weights",
]
