"""Pluralistic reward model with K latent value systems.

Models rewards as a mixture over K value systems, where each user has a
learned soft assignment to the systems based on their embedding.

R(user, content) = Σ_k π_k(user) · (weights_k · action_probs)

Three training approaches are provided:
1. EM-style: Alternating E-step (update assignments) and M-step (update weights)
2. Auxiliary losses: End-to-end with diversity and entropy regularization
3. Hybrid: EM structure with auxiliary regularization in M-step
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enhancements.reward_modeling.weights import NUM_ACTIONS, RewardWeights


class TrainingApproach(Enum):
    """Training approach for pluralistic model."""
    EM = "em"
    AUXILIARY = "auxiliary"
    HYBRID = "hybrid"


@dataclass
class PluralConfig:
    """Configuration for pluralistic reward model."""
    num_value_systems: int = 6
    num_actions: int = NUM_ACTIONS
    mlp_hidden_dim: int = 64

    # Training hyperparameters
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 64

    # Auxiliary loss weights
    lambda_diversity: float = 0.1  # Penalize similar systems
    lambda_entropy: float = 0.01   # Encourage peaky assignments
    lambda_classification: float = 0.0  # Supervised archetype classification (Fix 1)

    # EM-specific
    em_iterations: int = 10
    m_step_iterations: int = 20


class PluralState(NamedTuple):
    """State for pluralistic reward model."""
    weights: jnp.ndarray          # [K, num_actions] - value system weights
    mlp_weights: jnp.ndarray      # [input_dim, hidden_dim] - first layer
    mlp_bias: jnp.ndarray         # [hidden_dim]
    mlp_output_weights: jnp.ndarray  # [hidden_dim, K] - output layer
    mlp_output_bias: jnp.ndarray  # [K]


class PluralMetrics(NamedTuple):
    """Training metrics for pluralistic model."""
    loss_history: List[float]
    diversity_history: List[float]
    entropy_history: List[float]
    accuracy_history: List[float]
    system_correlations: Optional[jnp.ndarray] = None  # [K, K] pairwise


def init_plural_state(
    key: jax.random.PRNGKey,
    config: PluralConfig,
    input_dim: int,
) -> PluralState:
    """Initialize pluralistic model state."""
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # Initialize value system weights from default reward weights
    default = RewardWeights.default()
    # Add small random perturbation to differentiate systems
    noise = jax.random.normal(k1, (config.num_value_systems, config.num_actions)) * 0.1
    weights = jnp.tile(default.weights[jnp.newaxis, :], (config.num_value_systems, 1)) + noise

    # Initialize MLP (Xavier initialization)
    mlp_weights = jax.random.normal(k2, (input_dim, config.mlp_hidden_dim)) * jnp.sqrt(2.0 / input_dim)
    mlp_bias = jnp.zeros(config.mlp_hidden_dim)
    mlp_output_weights = jax.random.normal(k3, (config.mlp_hidden_dim, config.num_value_systems)) * jnp.sqrt(2.0 / config.mlp_hidden_dim)
    mlp_output_bias = jnp.zeros(config.num_value_systems)

    return PluralState(
        weights=weights,
        mlp_weights=mlp_weights,
        mlp_bias=mlp_bias,
        mlp_output_weights=mlp_output_weights,
        mlp_output_bias=mlp_output_bias,
    )


# =============================================================================
# Core Model Functions
# =============================================================================


def compute_mixture_weights(
    state: PluralState,
    user_embeddings: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mixture weights π(user) over K value systems.

    Args:
        state: Model state with MLP parameters
        user_embeddings: [B, D] user embeddings

    Returns:
        [B, K] mixture weights (sum to 1 per user)
    """
    # Two-layer MLP: input -> hidden -> K
    hidden = jax.nn.relu(user_embeddings @ state.mlp_weights + state.mlp_bias)
    logits = hidden @ state.mlp_output_weights + state.mlp_output_bias
    return jax.nn.softmax(logits, axis=-1)


def compute_system_rewards(
    state: PluralState,
    action_probs: jnp.ndarray,
) -> jnp.ndarray:
    """Compute reward under each value system.

    Args:
        state: Model state with value system weights
        action_probs: [B, C, num_actions] action probabilities

    Returns:
        [B, C, K] rewards under each system
    """
    # einsum: batch, candidate, action × system, action -> batch, candidate, system
    return jnp.einsum('bca,ka->bck', action_probs, state.weights)


def compute_pluralistic_reward(
    state: PluralState,
    action_probs: jnp.ndarray,
    user_embeddings: jnp.ndarray,
) -> jnp.ndarray:
    """Compute pluralistic reward as mixture over value systems.

    Args:
        state: Model state
        action_probs: [B, C, num_actions] action probabilities
        user_embeddings: [B, D] user embeddings

    Returns:
        [B, C] final rewards
    """
    mixture = compute_mixture_weights(state, user_embeddings)  # [B, K]
    system_rewards = compute_system_rewards(state, action_probs)  # [B, C, K]
    # Weighted sum: [B, C, K] × [B, K] -> [B, C]
    return jnp.einsum('bck,bk->bc', system_rewards, mixture)


def get_dominant_system(
    state: PluralState,
    user_embeddings: jnp.ndarray,
) -> jnp.ndarray:
    """Get dominant value system for each user.

    Args:
        state: Model state
        user_embeddings: [B, D] user embeddings

    Returns:
        [B] index of dominant system per user
    """
    mixture = compute_mixture_weights(state, user_embeddings)
    return jnp.argmax(mixture, axis=-1)


# =============================================================================
# Loss Functions
# =============================================================================


def bradley_terry_loss(
    r_preferred: jnp.ndarray,
    r_rejected: jnp.ndarray,
) -> jnp.ndarray:
    """Bradley-Terry preference loss.

    Args:
        r_preferred: [B] rewards for preferred items
        r_rejected: [B] rewards for rejected items

    Returns:
        Scalar loss
    """
    return -jnp.mean(jax.nn.log_sigmoid(r_preferred - r_rejected))


def diversity_loss(weights: jnp.ndarray) -> jnp.ndarray:
    """Penalize similar value systems.

    Computes pairwise cosine similarity and penalizes high off-diagonal values.

    Args:
        weights: [K, num_actions] value system weights

    Returns:
        Scalar loss (lower = more diverse)
    """
    # Normalize to unit vectors
    norms = jnp.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    normed = weights / norms

    # Pairwise cosine similarity
    similarity = normed @ normed.T  # [K, K]

    # Penalize off-diagonal (should be low)
    K = weights.shape[0]
    mask = 1.0 - jnp.eye(K)
    off_diag = similarity * mask

    return jnp.mean(off_diag ** 2)


def entropy_loss(mixture: jnp.ndarray) -> jnp.ndarray:
    """Encourage peaky (low entropy) assignments.

    High entropy = uniform assignment = bad
    Low entropy = clear assignment = good

    Args:
        mixture: [B, K] mixture weights

    Returns:
        Scalar loss (mean entropy, lower = peakier)
    """
    # H(π) = -Σ π_k log π_k
    entropy = -jnp.sum(mixture * jnp.log(mixture + 1e-8), axis=-1)
    return jnp.mean(entropy)


def classification_loss(
    mixture: jnp.ndarray,
    true_archetype_ids: jnp.ndarray,
) -> jnp.ndarray:
    """Supervised classification loss: predict archetype from mixture weights.

    This is Fix 1: directly supervise the MLP to map users to their
    true archetype's value system.

    Args:
        mixture: [B, K] predicted mixture weights (softmax output)
        true_archetype_ids: [B] ground truth archetype indices (0 to K-1)

    Returns:
        Scalar cross-entropy loss
    """
    # Cross-entropy: -log(π[true_archetype])
    log_probs = jnp.log(mixture + 1e-8)  # [B, K]
    # Gather the log prob of the true class for each sample
    batch_size = mixture.shape[0]
    true_log_probs = log_probs[jnp.arange(batch_size), true_archetype_ids]
    return -jnp.mean(true_log_probs)


def compute_preference_accuracy(
    r_preferred: jnp.ndarray,
    r_rejected: jnp.ndarray,
) -> jnp.ndarray:
    """Compute preference prediction accuracy.

    Args:
        r_preferred: [B] rewards for preferred items
        r_rejected: [B] rewards for rejected items

    Returns:
        Accuracy (fraction where preferred > rejected)
    """
    return jnp.mean(r_preferred > r_rejected)


# =============================================================================
# Training Approach 1: EM-Style
# =============================================================================


def e_step_compute_responsibilities(
    weights: jnp.ndarray,
    user_probs_preferred: jnp.ndarray,
    user_probs_rejected: jnp.ndarray,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """E-step: Compute soft assignments based on likelihood under each system.

    For each user, compute how well each value system explains their preferences.

    Args:
        weights: [K, num_actions] value system weights
        user_probs_preferred: [B, num_actions] preferred item action probs
        user_probs_rejected: [B, num_actions] rejected item action probs
        temperature: Softmax temperature (lower = peakier assignments)

    Returns:
        [B, K] responsibilities (soft assignments)
    """
    K = weights.shape[0]
    B = user_probs_preferred.shape[0]

    # Compute reward difference under each system
    # r_pref[b, k] = probs_pref[b] · weights[k]
    r_preferred = jnp.einsum('ba,ka->bk', user_probs_preferred, weights)  # [B, K]
    r_rejected = jnp.einsum('ba,ka->bk', user_probs_rejected, weights)    # [B, K]

    # Log-likelihood under Bradley-Terry: log σ(r_pref - r_rej)
    log_likelihood = jax.nn.log_sigmoid(r_preferred - r_rejected)  # [B, K]

    # Soft assignment via softmax
    responsibilities = jax.nn.softmax(log_likelihood / temperature, axis=-1)

    return responsibilities


def m_step_update_weights(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    responsibilities: jnp.ndarray,
    learning_rate: float = 0.01,
    num_iterations: int = 20,
    lambda_diversity: float = 0.0,
) -> jnp.ndarray:
    """M-step: Update weights given soft assignments.

    Minimize weighted Bradley-Terry loss for each system.

    Args:
        weights: [K, num_actions] current weights
        probs_preferred: [B, num_actions] preferred action probs
        probs_rejected: [B, num_actions] rejected action probs
        responsibilities: [B, K] soft assignments from E-step
        learning_rate: Learning rate for gradient descent
        num_iterations: Number of gradient steps
        lambda_diversity: Diversity regularization (for hybrid approach)

    Returns:
        [K, num_actions] updated weights
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    def loss_fn(w):
        # Reward difference under each system
        r_pref = jnp.einsum('ba,ka->bk', probs_preferred, w)  # [B, K]
        r_rej = jnp.einsum('ba,ka->bk', probs_rejected, w)    # [B, K]

        # Weighted Bradley-Terry loss
        # Weight each sample's contribution by its responsibility to each system
        per_sample_loss = -jax.nn.log_sigmoid(r_pref - r_rej)  # [B, K]
        weighted_loss = jnp.sum(per_sample_loss * responsibilities) / jnp.sum(responsibilities)

        # Optional diversity regularization (for hybrid)
        if lambda_diversity > 0:
            weighted_loss = weighted_loss + lambda_diversity * diversity_loss(w)

        return weighted_loss

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for _ in range(num_iterations):
        loss, grads = grad_fn(weights)
        updates, opt_state = optimizer.update(grads, opt_state, weights)
        weights = optax.apply_updates(weights, updates)

    return weights


def train_mlp_supervised(
    state: PluralState,
    user_embeddings: jnp.ndarray,
    target_responsibilities: jnp.ndarray,
    learning_rate: float = 0.01,
    num_iterations: int = 50,
    archetype_ids: Optional[jnp.ndarray] = None,
    lambda_classification: float = 0.0,
) -> PluralState:
    """Train MLP to predict E-step responsibilities (distillation).

    Optionally adds supervised classification loss when archetype_ids provided.

    Args:
        state: Current model state
        user_embeddings: [B, D] user embeddings
        target_responsibilities: [B, K] target from E-step
        learning_rate: Learning rate
        num_iterations: Number of gradient steps
        archetype_ids: [B] ground truth archetype indices (for supervised loss)
        lambda_classification: Weight for classification loss

    Returns:
        Updated state with trained MLP
    """
    use_classification = lambda_classification > 0 and archetype_ids is not None

    # Create dummy archetype_ids if not provided (won't be used if use_classification=False)
    _archetype_ids = archetype_ids if archetype_ids is not None else jnp.zeros(user_embeddings.shape[0], dtype=jnp.int32)

    # Pack MLP params for optimization
    mlp_params = {
        'w1': state.mlp_weights,
        'b1': state.mlp_bias,
        'w2': state.mlp_output_weights,
        'b2': state.mlp_output_bias,
    }

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(mlp_params)

    def loss_fn(params):
        hidden = jax.nn.relu(user_embeddings @ params['w1'] + params['b1'])
        logits = hidden @ params['w2'] + params['b2']
        predicted = jax.nn.softmax(logits, axis=-1)

        # Distillation loss: match E-step responsibilities
        L_distill = -jnp.mean(jnp.sum(target_responsibilities * jnp.log(predicted + 1e-8), axis=-1))

        # Classification loss: predict true archetype
        if use_classification:
            L_cls = classification_loss(predicted, _archetype_ids)
            return L_distill + lambda_classification * L_cls
        else:
            return L_distill

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    for _ in range(num_iterations):
        _, grads = grad_fn(mlp_params)
        updates, opt_state = optimizer.update(grads, opt_state, mlp_params)
        mlp_params = optax.apply_updates(mlp_params, updates)

    return PluralState(
        weights=state.weights,
        mlp_weights=mlp_params['w1'],
        mlp_bias=mlp_params['b1'],
        mlp_output_weights=mlp_params['w2'],
        mlp_output_bias=mlp_params['b2'],
    )


def train_em(
    initial_state: PluralState,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    config: PluralConfig,
    verbose: bool = True,
) -> Tuple[PluralState, PluralMetrics]:
    """Train pluralistic model using EM-style alternating optimization.

    Args:
        initial_state: Initial model state
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        user_embeddings: [N, D] user embeddings
        config: Training configuration
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    state = initial_state

    loss_history = []
    diversity_history = []
    entropy_history = []
    accuracy_history = []

    for iteration in range(config.em_iterations):
        # === E-step: Compute responsibilities ===
        responsibilities = e_step_compute_responsibilities(
            state.weights,
            probs_preferred,
            probs_rejected,
            temperature=1.0,
        )

        # === M-step: Update weights ===
        state = PluralState(
            weights=m_step_update_weights(
                state.weights,
                probs_preferred,
                probs_rejected,
                responsibilities,
                learning_rate=config.learning_rate,
                num_iterations=config.m_step_iterations,
                lambda_diversity=0.0,  # Pure EM, no diversity regularization
            ),
            mlp_weights=state.mlp_weights,
            mlp_bias=state.mlp_bias,
            mlp_output_weights=state.mlp_output_weights,
            mlp_output_bias=state.mlp_output_bias,
        )

        # === Train MLP to predict responsibilities ===
        state = train_mlp_supervised(
            state,
            user_embeddings,
            responsibilities,
            learning_rate=config.learning_rate,
            num_iterations=50,
        )

        # === Compute metrics ===
        mixture = compute_mixture_weights(state, user_embeddings)
        r_pref = jnp.sum(probs_preferred * state.weights[jnp.argmax(responsibilities, axis=1)], axis=1)
        r_rej = jnp.sum(probs_rejected * state.weights[jnp.argmax(responsibilities, axis=1)], axis=1)

        # Use pluralistic reward for metrics
        r_pref_plural = compute_pluralistic_reward(state, probs_preferred[:, jnp.newaxis, :], user_embeddings)[:, 0]
        r_rej_plural = compute_pluralistic_reward(state, probs_rejected[:, jnp.newaxis, :], user_embeddings)[:, 0]

        loss = float(bradley_terry_loss(r_pref_plural, r_rej_plural))
        div = float(diversity_loss(state.weights))
        ent = float(entropy_loss(mixture))
        acc = float(compute_preference_accuracy(r_pref_plural, r_rej_plural))

        loss_history.append(loss)
        diversity_history.append(div)
        entropy_history.append(ent)
        accuracy_history.append(acc)

        if verbose:
            print(f"  EM iter {iteration+1}/{config.em_iterations}: "
                  f"loss={loss:.4f}, div={div:.4f}, ent={ent:.4f}, acc={acc:.1%}")

    metrics = PluralMetrics(
        loss_history=loss_history,
        diversity_history=diversity_history,
        entropy_history=entropy_history,
        accuracy_history=accuracy_history,
    )

    return state, metrics


# =============================================================================
# Training Approach 2: Auxiliary Losses (End-to-End)
# =============================================================================


def train_auxiliary(
    initial_state: PluralState,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    config: PluralConfig,
    archetype_ids: Optional[jnp.ndarray] = None,
    verbose: bool = True,
) -> Tuple[PluralState, PluralMetrics]:
    """Train pluralistic model end-to-end with auxiliary losses.

    L_total = L_bradley_terry + λ_div · L_diversity + λ_ent · L_entropy + λ_cls · L_classification

    Args:
        initial_state: Initial model state
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        user_embeddings: [N, D] user embeddings
        config: Training configuration
        archetype_ids: [N] ground truth archetype indices (for supervised classification)
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    use_classification = config.lambda_classification > 0 and archetype_ids is not None

    # Pack all parameters
    params = {
        'weights': initial_state.weights,
        'mlp_w1': initial_state.mlp_weights,
        'mlp_b1': initial_state.mlp_bias,
        'mlp_w2': initial_state.mlp_output_weights,
        'mlp_b2': initial_state.mlp_output_bias,
    }

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch_probs_pref, batch_probs_rej, batch_user_emb, batch_arch_ids):
        # Compute mixture weights
        hidden = jax.nn.relu(batch_user_emb @ params['mlp_w1'] + params['mlp_b1'])
        logits = hidden @ params['mlp_w2'] + params['mlp_b2']
        mixture = jax.nn.softmax(logits, axis=-1)  # [B, K]

        # Compute rewards under each system
        r_pref_systems = jnp.einsum('ba,ka->bk', batch_probs_pref, params['weights'])  # [B, K]
        r_rej_systems = jnp.einsum('ba,ka->bk', batch_probs_rej, params['weights'])    # [B, K]

        # Pluralistic rewards
        r_pref = jnp.sum(r_pref_systems * mixture, axis=-1)  # [B]
        r_rej = jnp.sum(r_rej_systems * mixture, axis=-1)    # [B]

        # Bradley-Terry loss
        L_bt = bradley_terry_loss(r_pref, r_rej)

        # Diversity loss
        L_div = diversity_loss(params['weights'])

        # Entropy loss
        L_ent = entropy_loss(mixture)

        # Classification loss (supervised, if enabled)
        L_cls = classification_loss(mixture, batch_arch_ids) if use_classification else 0.0

        L_total = (L_bt
                   + config.lambda_diversity * L_div
                   + config.lambda_entropy * L_ent
                   + config.lambda_classification * L_cls)

        return L_total, (L_bt, L_div, L_ent, L_cls, r_pref, r_rej)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    n_samples = len(probs_preferred)
    loss_history = []
    diversity_history = []
    entropy_history = []
    accuracy_history = []
    classification_history = []

    rng = np.random.default_rng(42)

    # If no archetype_ids provided, create dummy array
    _archetype_ids: jnp.ndarray = archetype_ids if archetype_ids is not None else jnp.zeros(n_samples, dtype=jnp.int32)

    for epoch in range(config.num_epochs):
        # Shuffle
        perm = rng.permutation(n_samples)
        epoch_losses = []
        epoch_divs = []
        epoch_ents = []
        epoch_accs = []
        epoch_cls = []

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_pref = probs_preferred[idx]
            batch_rej = probs_rejected[idx]
            batch_emb = user_embeddings[idx]
            batch_arch = _archetype_ids[idx]

            (_, (L_bt, L_div, L_ent, L_cls, r_pref, r_rej)), grads = grad_fn(
                params, batch_pref, batch_rej, batch_emb, batch_arch
            )

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            epoch_losses.append(float(L_bt))
            epoch_divs.append(float(L_div))
            epoch_ents.append(float(L_ent))
            epoch_accs.append(float(jnp.mean(r_pref > r_rej)))
            if use_classification:
                epoch_cls.append(float(L_cls))

        loss_history.append(np.mean(epoch_losses))
        diversity_history.append(np.mean(epoch_divs))
        entropy_history.append(np.mean(epoch_ents))
        accuracy_history.append(np.mean(epoch_accs))
        if use_classification:
            classification_history.append(np.mean(epoch_cls))

        if verbose and (epoch + 1) % 10 == 0:
            cls_str = f", cls={classification_history[-1]:.4f}" if use_classification else ""
            print(f"  Aux epoch {epoch+1}/{config.num_epochs}: "
                  f"loss={loss_history[-1]:.4f}, div={diversity_history[-1]:.4f}, "
                  f"ent={entropy_history[-1]:.4f}, acc={accuracy_history[-1]:.1%}{cls_str}")

    final_state = PluralState(
        weights=params['weights'],
        mlp_weights=params['mlp_w1'],
        mlp_bias=params['mlp_b1'],
        mlp_output_weights=params['mlp_w2'],
        mlp_output_bias=params['mlp_b2'],
    )

    metrics = PluralMetrics(
        loss_history=loss_history,
        diversity_history=diversity_history,
        entropy_history=entropy_history,
        accuracy_history=accuracy_history,
    )

    return final_state, metrics


# =============================================================================
# Training Approach 3: Hybrid (EM + Auxiliary Regularization)
# =============================================================================


def train_hybrid(
    initial_state: PluralState,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    config: PluralConfig,
    archetype_ids: Optional[jnp.ndarray] = None,
    verbose: bool = True,
) -> Tuple[PluralState, PluralMetrics]:
    """Train pluralistic model using hybrid approach.

    EM structure with auxiliary regularization in M-step:
    - E-step: Compute responsibilities based on likelihood
    - M-step: Update weights with diversity regularization
    - MLP: Trained to predict responsibilities (+ optional classification loss)

    Args:
        initial_state: Initial model state
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        user_embeddings: [N, D] user embeddings
        config: Training configuration
        archetype_ids: [N] ground truth archetype indices (for supervised classification)
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    state = initial_state

    loss_history = []
    diversity_history = []
    entropy_history = []
    accuracy_history = []

    for iteration in range(config.em_iterations):
        # === E-step: Compute responsibilities (same as pure EM) ===
        responsibilities = e_step_compute_responsibilities(
            state.weights,
            probs_preferred,
            probs_rejected,
            temperature=1.0,
        )

        # === M-step: Update weights WITH diversity regularization ===
        state = PluralState(
            weights=m_step_update_weights(
                state.weights,
                probs_preferred,
                probs_rejected,
                responsibilities,
                learning_rate=config.learning_rate,
                num_iterations=config.m_step_iterations,
                lambda_diversity=config.lambda_diversity,  # Key difference from pure EM
            ),
            mlp_weights=state.mlp_weights,
            mlp_bias=state.mlp_bias,
            mlp_output_weights=state.mlp_output_weights,
            mlp_output_bias=state.mlp_output_bias,
        )

        # === Train MLP to predict responsibilities (+ optional classification) ===
        state = train_mlp_supervised(
            state,
            user_embeddings,
            responsibilities,
            learning_rate=config.learning_rate,
            num_iterations=50,
            archetype_ids=archetype_ids,
            lambda_classification=config.lambda_classification,
        )

        # === Compute metrics ===
        mixture = compute_mixture_weights(state, user_embeddings)
        r_pref_plural = compute_pluralistic_reward(state, probs_preferred[:, jnp.newaxis, :], user_embeddings)[:, 0]
        r_rej_plural = compute_pluralistic_reward(state, probs_rejected[:, jnp.newaxis, :], user_embeddings)[:, 0]

        loss = float(bradley_terry_loss(r_pref_plural, r_rej_plural))
        div = float(diversity_loss(state.weights))
        ent = float(entropy_loss(mixture))
        acc = float(compute_preference_accuracy(r_pref_plural, r_rej_plural))

        loss_history.append(loss)
        diversity_history.append(div)
        entropy_history.append(ent)
        accuracy_history.append(acc)

        if verbose:
            print(f"  Hybrid iter {iteration+1}/{config.em_iterations}: "
                  f"loss={loss:.4f}, div={div:.4f}, ent={ent:.4f}, acc={acc:.1%}")

    metrics = PluralMetrics(
        loss_history=loss_history,
        diversity_history=diversity_history,
        entropy_history=entropy_history,
        accuracy_history=accuracy_history,
    )

    return state, metrics


# =============================================================================
# Unified Training Interface
# =============================================================================


def train_pluralistic(
    initial_state: PluralState,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    user_embeddings: jnp.ndarray,
    config: PluralConfig,
    approach: TrainingApproach = TrainingApproach.HYBRID,
    archetype_ids: Optional[jnp.ndarray] = None,
    verbose: bool = True,
) -> Tuple[PluralState, PluralMetrics]:
    """Train pluralistic model using specified approach.

    Args:
        initial_state: Initial model state
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        user_embeddings: [N, D] user embeddings
        config: Training configuration
        approach: Training approach (EM, AUXILIARY, or HYBRID)
        archetype_ids: [N] ground truth archetype indices (for supervised classification)
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    if verbose:
        print(f"Training with {approach.value} approach...")

    if approach == TrainingApproach.EM:
        # EM doesn't use archetype_ids (unsupervised)
        return train_em(initial_state, probs_preferred, probs_rejected,
                       user_embeddings, config, verbose)
    elif approach == TrainingApproach.AUXILIARY:
        return train_auxiliary(initial_state, probs_preferred, probs_rejected,
                              user_embeddings, config, archetype_ids, verbose)
    elif approach == TrainingApproach.HYBRID:
        return train_hybrid(initial_state, probs_preferred, probs_rejected,
                           user_embeddings, config, archetype_ids, verbose)
    else:
        raise ValueError(f"Unknown approach: {approach}")
