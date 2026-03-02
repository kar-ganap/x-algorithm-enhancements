"""Bradley-Terry preference learning for F4 Phase 1.

Implements preference-based training for reward weights using the
Bradley-Terry model: P(A preferred to B) = σ(R(A) - R(B))

Integrates with F2's synthetic data for ground-truth preferences.
"""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enhancements.reward_modeling.weights import NUM_ACTIONS


@dataclass
class TrainingConfig:
    """Configuration for reward weight training."""

    learning_rate: float = 0.01
    num_epochs: int = 50
    batch_size: int = 32
    weight_decay: float = 0.0
    margin: float = 0.0  # Optional margin for ranking loss

    # Regularization
    l2_reg: float = 0.0  # L2 regularization on weights
    entropy_reg: float = 0.0  # Encourage diverse weight usage

    # Logging
    log_every: int = 10
    eval_every: int = 10


def bradley_terry_loss(
    weights: jnp.ndarray,
    action_probs_preferred: jnp.ndarray,
    action_probs_rejected: jnp.ndarray,
    confidence: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute Bradley-Terry preference loss.

    Loss = -log(σ(R(preferred) - R(rejected)))

    Where R(x) = weights · action_probs(x)

    Args:
        weights: Reward weights [num_actions] or [K, num_actions] for contextual
        action_probs_preferred: Action probs for preferred items [B, num_actions]
        action_probs_rejected: Action probs for rejected items [B, num_actions]
        confidence: Optional confidence scores for each pair [B]

    Returns:
        Scalar loss value
    """
    # Compute rewards
    if weights.ndim == 1:
        # Single weight vector: R = w · P
        r_preferred = jnp.einsum("ba,a->b", action_probs_preferred, weights)
        r_rejected = jnp.einsum("ba,a->b", action_probs_rejected, weights)
    else:
        # weights is [B, num_actions] - per-sample weights
        r_preferred = jnp.einsum("ba,ba->b", action_probs_preferred, weights)
        r_rejected = jnp.einsum("ba,ba->b", action_probs_rejected, weights)

    # Bradley-Terry loss: -log(σ(r_preferred - r_rejected))
    logits = r_preferred - r_rejected
    loss = -jax.nn.log_sigmoid(logits)

    # Weight by confidence if provided
    if confidence is not None:
        loss = loss * confidence

    return jnp.mean(loss)


def bradley_terry_loss_with_margin(
    weights: jnp.ndarray,
    action_probs_preferred: jnp.ndarray,
    action_probs_rejected: jnp.ndarray,
    margin: float = 0.1,
    confidence: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Bradley-Terry loss with margin for stronger preference signal.

    Loss = max(0, margin - (R(preferred) - R(rejected)))

    Args:
        weights: Reward weights [num_actions]
        action_probs_preferred: Action probs for preferred items [B, num_actions]
        action_probs_rejected: Action probs for rejected items [B, num_actions]
        margin: Minimum reward difference to enforce
        confidence: Optional confidence scores for each pair [B]

    Returns:
        Scalar loss value
    """
    # Compute rewards
    r_preferred = jnp.einsum("ba,a->b", action_probs_preferred, weights)
    r_rejected = jnp.einsum("ba,a->b", action_probs_rejected, weights)

    # Margin loss: max(0, margin - (r_preferred - r_rejected))
    loss = jnp.maximum(0.0, margin - (r_preferred - r_rejected))

    if confidence is not None:
        loss = loss * confidence

    return jnp.mean(loss)


def contextual_bradley_terry_loss(
    weights: jnp.ndarray,
    action_probs_preferred: jnp.ndarray,
    action_probs_rejected: jnp.ndarray,
    archetype_ids: jnp.ndarray,
    confidence: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Bradley-Terry loss with archetype-specific weights.

    Uses per-archetype weight vectors for contextual reward modeling.

    Args:
        weights: Reward weights [K, num_actions] where K is num archetypes
        action_probs_preferred: Action probs for preferred items [B, num_actions]
        action_probs_rejected: Action probs for rejected items [B, num_actions]
        archetype_ids: Archetype index for each sample [B]
        confidence: Optional confidence scores for each pair [B]

    Returns:
        Scalar loss value
    """
    # Select weights for each sample's archetype: [B, num_actions]
    sample_weights = weights[archetype_ids]

    # Compute rewards with per-sample weights
    r_preferred = jnp.einsum("ba,ba->b", action_probs_preferred, sample_weights)
    r_rejected = jnp.einsum("ba,ba->b", action_probs_rejected, sample_weights)

    # Bradley-Terry loss
    logits = r_preferred - r_rejected
    loss = -jax.nn.log_sigmoid(logits)

    if confidence is not None:
        loss = loss * confidence

    return jnp.mean(loss)


@dataclass
class TrainingState:
    """Training state for gradient-based optimization."""

    weights: jnp.ndarray
    opt_state: optax.OptState
    step: int = 0


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    loss_history: list[float]
    accuracy_history: list[float]
    final_weights: jnp.ndarray
    epochs_trained: int


def compute_preference_accuracy(
    weights: jnp.ndarray,
    action_probs_preferred: jnp.ndarray,
    action_probs_rejected: jnp.ndarray,
    archetype_ids: jnp.ndarray | None = None,
) -> float:
    """Compute accuracy of preference predictions.

    Args:
        weights: Reward weights [num_actions] or [K, num_actions]
        action_probs_preferred: [B, num_actions]
        action_probs_rejected: [B, num_actions]
        archetype_ids: Optional archetype indices [B] for contextual weights

    Returns:
        Accuracy as fraction of correctly ordered pairs
    """
    if archetype_ids is not None and weights.ndim == 2:
        # Contextual: select per-archetype weights
        sample_weights = weights[archetype_ids]
        r_preferred = jnp.einsum("ba,ba->b", action_probs_preferred, sample_weights)
        r_rejected = jnp.einsum("ba,ba->b", action_probs_rejected, sample_weights)
    else:
        # Single weight vector
        w = weights if weights.ndim == 1 else weights[0]
        r_preferred = jnp.einsum("ba,a->b", action_probs_preferred, w)
        r_rejected = jnp.einsum("ba,a->b", action_probs_rejected, w)

    correct = (r_preferred > r_rejected).astype(jnp.float32)
    return float(jnp.mean(correct))


def train_single_weights(
    initial_weights: jnp.ndarray,
    get_batch_fn: Callable[[], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]],
    config: TrainingConfig,
    verbose: bool = True,
) -> TrainingMetrics:
    """Train a single reward weight vector.

    Args:
        initial_weights: Starting weights [num_actions]
        get_batch_fn: Function returning (probs_preferred, probs_rejected, confidence)
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        TrainingMetrics with final weights and history
    """
    # Initialize optimizer
    optimizer = optax.adam(config.learning_rate)
    if config.weight_decay > 0:
        optimizer = optax.chain(
            optax.add_decayed_weights(config.weight_decay),
            optimizer,
        )

    opt_state = optimizer.init(initial_weights)
    weights = initial_weights

    loss_history = []
    accuracy_history = []

    def loss_fn(w, probs_pref, probs_rej, conf):
        base_loss = bradley_terry_loss(w, probs_pref, probs_rej, conf)
        # Add L2 regularization
        if config.l2_reg > 0:
            base_loss = base_loss + config.l2_reg * jnp.sum(w ** 2)
        return base_loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for epoch in range(config.num_epochs):
        epoch_losses = []
        epoch_accs = []

        # Multiple batches per epoch
        for _ in range(config.batch_size):
            probs_pref, probs_rej, conf = get_batch_fn()

            loss, grads = grad_fn(weights, probs_pref, probs_rej, conf)
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

            epoch_losses.append(float(loss))

            # Compute accuracy
            acc = compute_preference_accuracy(weights, probs_pref, probs_rej)
            epoch_accs.append(acc)

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        loss_history.append(avg_loss)
        accuracy_history.append(avg_acc)

        if verbose and (epoch + 1) % config.log_every == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return TrainingMetrics(
        loss_history=loss_history,
        accuracy_history=accuracy_history,
        final_weights=weights,
        epochs_trained=config.num_epochs,
    )


def train_contextual_weights(
    initial_weights: jnp.ndarray,
    get_batch_fn: Callable[[], tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray | None]],
    config: TrainingConfig,
    verbose: bool = True,
) -> TrainingMetrics:
    """Train archetype-specific reward weights.

    Args:
        initial_weights: Starting weights [K, num_actions] where K = num archetypes
        get_batch_fn: Function returning (probs_pref, probs_rej, archetype_ids, confidence)
        config: Training configuration
        verbose: Whether to print progress

    Returns:
        TrainingMetrics with final weights and history
    """
    # Initialize optimizer
    optimizer = optax.adam(config.learning_rate)
    if config.weight_decay > 0:
        optimizer = optax.chain(
            optax.add_decayed_weights(config.weight_decay),
            optimizer,
        )

    opt_state = optimizer.init(initial_weights)
    weights = initial_weights

    loss_history = []
    accuracy_history = []

    def loss_fn(w, probs_pref, probs_rej, arch_ids, conf):
        base_loss = contextual_bradley_terry_loss(w, probs_pref, probs_rej, arch_ids, conf)
        # Add L2 regularization
        if config.l2_reg > 0:
            base_loss = base_loss + config.l2_reg * jnp.sum(w ** 2)
        return base_loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for epoch in range(config.num_epochs):
        epoch_losses = []
        epoch_accs = []

        for _ in range(config.batch_size):
            probs_pref, probs_rej, arch_ids, conf = get_batch_fn()

            loss, grads = grad_fn(weights, probs_pref, probs_rej, arch_ids, conf)
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

            epoch_losses.append(float(loss))

            # Compute accuracy
            acc = compute_preference_accuracy(weights, probs_pref, probs_rej, arch_ids)
            epoch_accs.append(acc)

        avg_loss = np.mean(epoch_losses)
        avg_acc = np.mean(epoch_accs)
        loss_history.append(avg_loss)
        accuracy_history.append(avg_acc)

        if verbose and (epoch + 1) % config.log_every == 0:
            print(f"Epoch {epoch + 1}/{config.num_epochs}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")

    return TrainingMetrics(
        loss_history=loss_history,
        accuracy_history=accuracy_history,
        final_weights=weights,
        epochs_trained=config.num_epochs,
    )


def create_synthetic_preference_batch(
    num_samples: int,
    num_actions: int = NUM_ACTIONS,
    rng: np.random.Generator | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create synthetic preference batch for testing.

    Generates random action probabilities where preferred items have
    higher positive action probs and lower negative action probs.

    Args:
        num_samples: Number of preference pairs
        num_actions: Number of actions (default 18)
        rng: Random number generator

    Returns:
        Tuple of (probs_preferred, probs_rejected, confidence)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate base probabilities
    probs_preferred = rng.uniform(0, 1, (num_samples, num_actions)).astype(np.float32)
    probs_rejected = rng.uniform(0, 1, (num_samples, num_actions)).astype(np.float32)

    # Make preferred items better: higher positive actions, lower negative
    # Positive actions: indices 0-13 (favorite, reply, repost, ..., follow_author)
    # Negative actions: indices 14-17 (not_interested, block, mute, report)

    # Boost positive actions for preferred
    probs_preferred[:, :14] *= 1.5
    probs_preferred = np.clip(probs_preferred, 0, 1)

    # Boost negative actions for rejected
    probs_rejected[:, 14:] *= 1.5
    probs_rejected = np.clip(probs_rejected, 0, 1)

    confidence = np.ones(num_samples, dtype=np.float32)

    return jnp.array(probs_preferred), jnp.array(probs_rejected), jnp.array(confidence)
