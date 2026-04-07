"""Alternative loss functions for stakeholder-differentiated reward models.

This module implements several alternative loss functions designed to break
the scale invariance of Bradley-Terry loss and create meaningfully different
stakeholder models.

Loss Functions:
- A1: Margin-Based BT - Forces minimum score gap
- A2: Calibrated BT - Anchors scores to engagement rates
- C1: Constrained BT - Hard constraints per stakeholder
- C2: Post-Hoc Reranking - Serving-time differentiation

The goal is to achieve cosine similarity < 0.95 between stakeholder models
(currently ~0.999 with standard BT).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Define constants locally to avoid import chain
ACTION_NAMES: list[str] = [
    "favorite", "reply", "repost", "photo_expand", "click", "profile_click",
    "vqv", "share", "share_via_dm", "share_via_copy_link", "dwell", "quote",
    "quoted_click", "follow_author", "not_interested", "block_author",
    "mute_author", "report",
]
NUM_ACTIONS = len(ACTION_NAMES)
ACTION_INDICES: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}

# Negative action indices for stakeholder-specific losses
NEGATIVE_ACTIONS = ["block_author", "mute_author", "report", "not_interested"]
NEGATIVE_INDICES = [ACTION_INDICES[a] for a in NEGATIVE_ACTIONS]

POSITIVE_ACTIONS = ["favorite", "repost", "follow_author", "share", "reply"]
POSITIVE_INDICES = [ACTION_INDICES[a] for a in POSITIVE_ACTIONS]


class StakeholderType(Enum):
    """Stakeholder types for differentiated models."""
    USER = "user"
    PLATFORM = "platform"
    SOCIETY = "society"


class LossType(Enum):
    """Available loss function types."""
    BRADLEY_TERRY = "bradley_terry"
    MARGIN_BT = "margin_bt"
    CALIBRATED_BT = "calibrated_bt"
    CONSTRAINED_BT = "constrained_bt"


@dataclass
class LossConfig:
    """Configuration for loss function training."""
    loss_type: LossType
    stakeholder: StakeholderType

    # Training parameters
    learning_rate: float = 0.01
    num_epochs: int = 150
    batch_size: int = 64
    l2_weight: float = 0.001

    # Margin-BT parameters
    margin: float = 0.5

    # Calibrated-BT parameters
    calibration_weight: float = 1.0

    # Constrained-BT parameters
    constraint_weight: float = 10.0
    diversity_target: float = 0.3  # For society
    max_negative_exposure: float = 0.1  # For user

    # Post-hoc reranking
    rerank_alpha: float = 0.5


class TrainedModel(NamedTuple):
    """Result of training with alternative loss."""
    loss_type: LossType
    stakeholder: StakeholderType
    weights: np.ndarray
    loss_history: list[float]
    accuracy: float  # Training accuracy (on training pairs)
    config: LossConfig
    eval_accuracy: float | None = None  # Held-out accuracy (if eval pairs provided)


# =============================================================================
# Standard Bradley-Terry (baseline)
# =============================================================================

def bradley_terry_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
) -> jnp.ndarray:
    """Standard Bradley-Terry loss: -log(σ(r_pref - r_rej))."""
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights
    return jnp.mean(jnp.log(1 + jnp.exp(-(r_pref - r_rej))))


# =============================================================================
# A1: Margin-Based Bradley-Terry
# =============================================================================

def margin_bt_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    margin: float = 0.5,
) -> jnp.ndarray:
    """Margin-based BT: max(0, margin - (r_pref - r_rej)).

    Forces a minimum score gap between preferred and rejected items.
    This breaks scale invariance by requiring absolute score differences.

    Args:
        weights: [num_actions] reward weights
        probs_preferred: [batch, num_actions] preferred content action probs
        probs_rejected: [batch, num_actions] rejected content action probs
        margin: Minimum required score difference

    Returns:
        Scalar loss value
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    # Hinge loss: penalize when gap < margin
    margin_violations = jnp.maximum(0.0, margin - (r_pref - r_rej))

    return jnp.mean(margin_violations)


def margin_bt_loss_smooth(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    margin: float = 0.5,
    temperature: float = 0.1,
) -> jnp.ndarray:
    """Smooth margin loss using softplus for better gradients.

    Uses softplus(margin - diff) instead of max(0, margin - diff)
    for smoother optimization landscape.
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    diff = r_pref - r_rej

    # Softplus approximation: log(1 + exp(x)) ≈ max(0, x) for large x
    smooth_violations = jax.nn.softplus((margin - diff) / temperature) * temperature

    return jnp.mean(smooth_violations)


# =============================================================================
# A2: Calibrated Bradley-Terry
# =============================================================================

def calibrated_bt_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    target_engagement_pref: jnp.ndarray,
    target_engagement_rej: jnp.ndarray,
    calibration_weight: float = 1.0,
) -> jnp.ndarray:
    """Calibrated BT: BT loss + MSE to ground truth engagement.

    Anchors scores to absolute engagement rates, breaking scale invariance.
    The model must learn weights that both:
    1. Rank preferred > rejected (BT loss)
    2. Predict actual engagement rates (calibration loss)

    Args:
        weights: [num_actions] reward weights
        probs_preferred: [batch, num_actions] preferred content action probs
        probs_rejected: [batch, num_actions] rejected content action probs
        target_engagement_pref: [batch] ground truth engagement rate for preferred
        target_engagement_rej: [batch] ground truth engagement rate for rejected
        calibration_weight: Weight for calibration term

    Returns:
        Scalar loss value
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    # Standard Bradley-Terry for ranking
    bt_loss = jnp.mean(jnp.log(1 + jnp.exp(-(r_pref - r_rej))))

    # Calibration: sigmoid(score) should predict engagement rate
    # This anchors scores to meaningful probabilities
    pred_pref = jax.nn.sigmoid(r_pref)
    pred_rej = jax.nn.sigmoid(r_rej)

    mse_pref = jnp.mean((pred_pref - target_engagement_pref) ** 2)
    mse_rej = jnp.mean((pred_rej - target_engagement_rej) ** 2)

    calibration_loss = mse_pref + mse_rej

    return bt_loss + calibration_weight * calibration_loss


# =============================================================================
# C1: Constrained Bradley-Terry
# =============================================================================

def constrained_bt_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    stakeholder: str,
    constraint_weight: float = 10.0,
    diversity_target: float = 0.3,
    max_negative_exposure: float = 0.1,
) -> jnp.ndarray:
    """Constrained BT: BT loss + stakeholder-specific hard constraints.

    Different stakeholders have different constraints:
    - User: Minimize negative signal exposure
    - Society: Maximize weight diversity (avoid all-same weights)
    - Platform: No additional constraints (standard BT)

    Args:
        weights: [num_actions] reward weights
        probs_preferred: [batch, num_actions] preferred content action probs
        probs_rejected: [batch, num_actions] rejected content action probs
        stakeholder: "user", "platform", or "society"
        constraint_weight: Weight for constraint penalty
        diversity_target: Target weight std for society
        max_negative_exposure: Max allowed negative exposure for user

    Returns:
        Scalar loss value
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    # Base Bradley-Terry loss
    bt_loss = jnp.mean(jnp.log(1 + jnp.exp(-(r_pref - r_rej))))

    if stakeholder == "user":
        # User constraint: minimize negative signal influence
        # Penalize if negative action weights are too positive
        neg_weights = weights[jnp.array(NEGATIVE_INDICES)]

        # Negative weights should be negative (penalize positive negative weights)
        neg_exposure = jnp.mean(jnp.maximum(0.0, neg_weights + max_negative_exposure))
        constraint = neg_exposure

    elif stakeholder == "society":
        # Society constraint: encourage diverse weights (not all same)
        # Penalize if weight standard deviation is too low.
        # Uses full weight vector — society's diversity comes from the
        # pos/neg spread, not from variation within positive weights alone.
        weight_std = jnp.std(weights)

        # Want diversity (high std), penalize if below target
        constraint = jnp.maximum(0.0, diversity_target - weight_std)

    else:  # platform
        # Platform: no additional constraints
        constraint = 0.0

    return bt_loss + constraint_weight * constraint


def constrained_bt_loss_user(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    constraint_weight: float = 10.0,
    max_negative_exposure: float = 0.1,
) -> jnp.ndarray:
    """User-specific constrained BT loss."""
    return constrained_bt_loss(
        weights, probs_preferred, probs_rejected,
        stakeholder="user",
        constraint_weight=constraint_weight,
        max_negative_exposure=max_negative_exposure,
    )


def constrained_bt_loss_society(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    constraint_weight: float = 10.0,
    diversity_target: float = 0.3,
) -> jnp.ndarray:
    """Society-specific constrained BT loss."""
    return constrained_bt_loss(
        weights, probs_preferred, probs_rejected,
        stakeholder="society",
        constraint_weight=constraint_weight,
        diversity_target=diversity_target,
    )


# =============================================================================
# C2: Post-Hoc Reranking (No training - evaluation only)
# =============================================================================

class PostHocReranker:
    """Post-hoc reranking for stakeholder differentiation.

    Instead of training different models, uses a single base model
    and applies stakeholder-specific adjustments at serving time.

    This completely sidesteps the BT scale invariance issue since
    the differentiation happens post-training.
    """

    def __init__(self, base_weights: np.ndarray):
        """Initialize with base model weights.

        Args:
            base_weights: [num_actions] weights from any trained model
        """
        self.base_weights = base_weights

    def score(
        self,
        candidates: np.ndarray,
        stakeholder: StakeholderType,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Compute stakeholder-adjusted scores for candidates.

        Args:
            candidates: [num_candidates, num_actions] action probabilities
            stakeholder: Which stakeholder perspective
            alpha: Adjustment strength (0 = base only, 1 = full adjustment)

        Returns:
            [num_candidates] adjusted scores
        """
        # Base scores from shared model
        base_scores = candidates @ self.base_weights

        if stakeholder == StakeholderType.USER:
            # User: penalize content with high negative signals
            neg_probs = candidates[:, NEGATIVE_INDICES]
            negative_penalty = np.sum(neg_probs, axis=1)
            adjusted_scores = base_scores - alpha * negative_penalty

        elif stakeholder == StakeholderType.SOCIETY:
            # Society: boost diverse content, penalize polarizing content
            # Use action diversity as proxy (prefer varied action profiles)
            action_entropy = -np.sum(
                candidates * np.log(candidates + 1e-8), axis=1
            )
            diversity_bonus = action_entropy / np.log(NUM_ACTIONS)  # Normalize
            adjusted_scores = base_scores + alpha * diversity_bonus

        else:  # PLATFORM
            # Platform: use base scores (all engagement is good)
            adjusted_scores = base_scores

        return adjusted_scores

    def rerank(
        self,
        candidates: np.ndarray,
        stakeholder: StakeholderType,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Get ranking of candidates for stakeholder.

        Args:
            candidates: [num_candidates, num_actions] action probabilities
            stakeholder: Which stakeholder perspective
            alpha: Adjustment strength

        Returns:
            [num_candidates] indices sorted by score (highest first)
        """
        scores = self.score(candidates, stakeholder, alpha)
        return np.argsort(-scores)


# =============================================================================
# Generic Training Loop
# =============================================================================

def create_loss_fn(
    config: LossConfig,
    target_engagement_pref: jnp.ndarray | None = None,
    target_engagement_rej: jnp.ndarray | None = None,
) -> Callable:
    """Create a loss function based on config.

    Returns a function with signature:
        loss_fn(weights, probs_pref, probs_rej) -> scalar
    """

    if config.loss_type == LossType.BRADLEY_TERRY:
        def loss_fn(w, p_pref, p_rej):
            return bradley_terry_loss(w, p_pref, p_rej)
        return loss_fn

    elif config.loss_type == LossType.MARGIN_BT:
        def loss_fn(w, p_pref, p_rej):
            return margin_bt_loss_smooth(w, p_pref, p_rej, margin=config.margin)
        return loss_fn

    elif config.loss_type == LossType.CALIBRATED_BT:
        if target_engagement_pref is None or target_engagement_rej is None:
            raise ValueError("Calibrated BT requires target engagement rates")

        # For calibrated BT, targets are passed per-batch via kwargs
        def loss_fn(w, p_pref, p_rej, eng_pref=None, eng_rej=None):
            if eng_pref is None or eng_rej is None:
                # Fallback to full arrays if not provided (shouldn't happen)
                eng_pref = target_engagement_pref
                eng_rej = target_engagement_rej
            return calibrated_bt_loss(
                w, p_pref, p_rej,
                eng_pref, eng_rej,
                calibration_weight=config.calibration_weight,
            )
        return loss_fn

    elif config.loss_type == LossType.CONSTRAINED_BT:
        stakeholder_str = config.stakeholder.value

        def loss_fn(w, p_pref, p_rej):
            return constrained_bt_loss(
                w, p_pref, p_rej,
                stakeholder=stakeholder_str,
                constraint_weight=config.constraint_weight,
                diversity_target=config.diversity_target,
                max_negative_exposure=config.max_negative_exposure,
            )
        return loss_fn

    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


def train_with_loss(
    config: LossConfig,
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    target_engagement_pref: np.ndarray | None = None,
    target_engagement_rej: np.ndarray | None = None,
    initial_weights: np.ndarray | None = None,
    verbose: bool = True,
    eval_probs_preferred: np.ndarray | None = None,
    eval_probs_rejected: np.ndarray | None = None,
) -> TrainedModel:
    """Train a model with the specified loss function.

    Args:
        config: Loss configuration
        probs_preferred: [N, num_actions] preferred content action probs
        probs_rejected: [N, num_actions] rejected content action probs
        target_engagement_pref: [N] ground truth engagement for preferred (calibrated BT)
        target_engagement_rej: [N] ground truth engagement for rejected (calibrated BT)
        initial_weights: Starting weights (default: ones for positive, minus ones for negative)
        verbose: Print progress
        eval_probs_preferred: [N_eval, D] held-out preferred features (optional)
        eval_probs_rejected: [N_eval, D] held-out rejected features (optional)

    Returns:
        TrainedModel with weights and metrics. If eval pairs provided,
        eval_accuracy is populated with held-out accuracy.
    """
    if verbose:
        print(f"Training {config.loss_type.value} for {config.stakeholder.value}")
        print(f"  Epochs: {config.num_epochs}, LR: {config.learning_rate}")

    # Initialize weights
    feature_dim = probs_preferred.shape[1]
    if initial_weights is None:
        if feature_dim == NUM_ACTIONS:
            # Original 18-action initialization
            weights = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for idx in POSITIVE_INDICES:
                weights[idx] = 1.0
            for idx in NEGATIVE_INDICES:
                weights[idx] = -1.0
        else:
            # Generic initialization for arbitrary feature dimension
            weights = np.zeros(feature_dim, dtype=np.float32)
            weights[:] = 0.1  # small positive default
    else:
        weights = initial_weights.copy()

    w = jnp.array(weights)

    # Constrained-BT uses Phoenix-specific positive/negative indices
    if config.loss_type == LossType.CONSTRAINED_BT and feature_dim != NUM_ACTIONS:
        raise ValueError(
            f"Constrained-BT requires {NUM_ACTIONS}-dim features "
            f"(got {feature_dim}). Use standard BT, margin-BT, or "
            f"calibrated-BT for non-Phoenix feature spaces."
        )

    # Create loss function
    loss_fn = create_loss_fn(
        config,
        jnp.array(target_engagement_pref) if target_engagement_pref is not None else None,
        jnp.array(target_engagement_rej) if target_engagement_rej is not None else None,
    )

    # Convert engagement targets to jax arrays if present
    eng_pref_jax = jnp.array(target_engagement_pref) if target_engagement_pref is not None else None
    eng_rej_jax = jnp.array(target_engagement_rej) if target_engagement_rej is not None else None

    # Add L2 regularization (handle variable number of args for calibrated BT)
    def full_loss(weights, p_pref, p_rej, eng_pref=None, eng_rej=None):
        if config.loss_type == LossType.CALIBRATED_BT:
            return loss_fn(weights, p_pref, p_rej, eng_pref, eng_rej) + config.l2_weight * jnp.sum(weights ** 2)
        else:
            return loss_fn(weights, p_pref, p_rej) + config.l2_weight * jnp.sum(weights ** 2)

    # Optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(w)

    # Training loop
    loss_history = []
    rng = np.random.default_rng(42)
    n_samples = len(probs_preferred)

    for epoch in range(config.num_epochs):
        perm = rng.permutation(n_samples)
        epoch_losses = []

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_pref = jnp.array(probs_preferred[idx])
            batch_rej = jnp.array(probs_rejected[idx])

            # Batch engagement targets for calibrated BT
            if config.loss_type == LossType.CALIBRATED_BT and eng_pref_jax is not None:
                batch_eng_pref = eng_pref_jax[idx]
                batch_eng_rej = eng_rej_jax[idx]
                loss, grad = jax.value_and_grad(full_loss)(
                    w, batch_pref, batch_rej, batch_eng_pref, batch_eng_rej
                )
            else:
                loss, grad = jax.value_and_grad(full_loss)(w, batch_pref, batch_rej)

            updates, opt_state = optimizer.update(grad, opt_state, w)
            w = optax.apply_updates(w, updates)

            epoch_losses.append(float(loss))

        loss_history.append(np.mean(epoch_losses))

        if verbose and (epoch + 1) % 30 == 0:
            print(f"    Epoch {epoch + 1}: loss = {loss_history[-1]:.4f}")

    # Compute final accuracy
    final_weights = np.array(w)
    r_pref = probs_preferred @ final_weights
    r_rej = probs_rejected @ final_weights
    accuracy = float(np.mean(r_pref > r_rej))

    # Compute held-out accuracy if eval pairs provided
    eval_accuracy = None
    if eval_probs_preferred is not None and eval_probs_rejected is not None:
        r_eval_pref = eval_probs_preferred @ final_weights
        r_eval_rej = eval_probs_rejected @ final_weights
        eval_accuracy = float(np.mean(r_eval_pref > r_eval_rej))

    if verbose:
        acc_str = f"  Final accuracy: {accuracy:.1%} (train)"
        if eval_accuracy is not None:
            acc_str += f", {eval_accuracy:.1%} (held-out)"
        print(acc_str)

    return TrainedModel(
        loss_type=config.loss_type,
        stakeholder=config.stakeholder,
        weights=final_weights,
        loss_history=loss_history,
        accuracy=accuracy,
        config=config,
        eval_accuracy=eval_accuracy,
    )


# =============================================================================
# Comparison Utilities
# =============================================================================

def compute_weight_similarity(
    models: dict[StakeholderType, TrainedModel]
) -> dict[str, float]:
    """Compute cosine similarity between model weights."""
    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    types = list(models.keys())
    similarities = {}

    for i, t1 in enumerate(types):
        for t2 in types[i+1:]:
            key = f"{t1.value}_{t2.value}"
            similarities[key] = cosine_sim(models[t1].weights, models[t2].weights)

    return similarities


def compute_ranking_correlation(
    models: dict[StakeholderType, TrainedModel],
    content_probs: np.ndarray,
) -> dict[str, float]:
    """Compute Kendall's tau between model rankings."""
    from scipy.stats import kendalltau

    scores = {}
    for stype, model in models.items():
        scores[stype] = content_probs @ model.weights

    correlations = {}
    types = list(models.keys())

    for i, t1 in enumerate(types):
        for t2 in types[i+1:]:
            tau, _ = kendalltau(scores[t1], scores[t2])
            key = f"{t1.value}_{t2.value}"
            correlations[key] = float(tau)

    return correlations
