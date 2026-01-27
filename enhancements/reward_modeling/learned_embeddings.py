"""Fix 2: Learned user embeddings from interaction history.

Instead of using random embeddings, we learn embeddings from user interaction
patterns. This encodes preference information directly into embeddings,
allowing the pluralistic model to discover meaningful structure.

Key insight: If embeddings encode what users click on, the MLP can learn
meaningful clusters that correspond to actual value systems.
"""

from dataclasses import dataclass
from typing import Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enhancements.reward_modeling.pluralistic import (
    PluralConfig,
    PluralMetrics,
    bradley_terry_loss,
    classification_loss,
    diversity_loss,
    entropy_loss,
)
from enhancements.reward_modeling.weights import NUM_ACTIONS, RewardWeights


@dataclass
class LearnedEmbeddingConfig(PluralConfig):
    """Configuration for pluralistic model with learned embeddings."""
    embedding_dim: int = 32  # Output embedding dimension
    encoder_hidden_dim: int = 64  # Encoder hidden layer
    use_history_encoder: bool = True  # Learn encoder vs use raw features


class LearnedEmbeddingState(NamedTuple):
    """State including user encoder parameters."""
    # Value system weights [K, num_actions]
    weights: jnp.ndarray

    # User encoder: history -> embedding
    encoder_w1: jnp.ndarray  # [num_actions, encoder_hidden]
    encoder_b1: jnp.ndarray  # [encoder_hidden]
    encoder_w2: jnp.ndarray  # [encoder_hidden, embedding_dim]
    encoder_b2: jnp.ndarray  # [embedding_dim]

    # Mixture MLP: embedding -> K
    mlp_w1: jnp.ndarray  # [embedding_dim, mlp_hidden]
    mlp_b1: jnp.ndarray  # [mlp_hidden]
    mlp_w2: jnp.ndarray  # [mlp_hidden, K]
    mlp_b2: jnp.ndarray  # [K]


def init_learned_embedding_state(
    key: jax.random.PRNGKey,
    config: LearnedEmbeddingConfig,
) -> LearnedEmbeddingState:
    """Initialize state with encoder and pluralistic model."""
    keys = jax.random.split(key, 8)

    # Value system weights from default
    default = RewardWeights.default()
    noise = jax.random.normal(keys[0], (config.num_value_systems, config.num_actions)) * 0.1
    weights = jnp.tile(default.weights[jnp.newaxis, :], (config.num_value_systems, 1)) + noise

    # Encoder: num_actions -> embedding_dim
    encoder_w1 = jax.random.normal(keys[1], (config.num_actions, config.encoder_hidden_dim)) * jnp.sqrt(2.0 / config.num_actions)
    encoder_b1 = jnp.zeros(config.encoder_hidden_dim)
    encoder_w2 = jax.random.normal(keys[2], (config.encoder_hidden_dim, config.embedding_dim)) * jnp.sqrt(2.0 / config.encoder_hidden_dim)
    encoder_b2 = jnp.zeros(config.embedding_dim)

    # MLP: embedding_dim -> K
    mlp_w1 = jax.random.normal(keys[3], (config.embedding_dim, config.mlp_hidden_dim)) * jnp.sqrt(2.0 / config.embedding_dim)
    mlp_b1 = jnp.zeros(config.mlp_hidden_dim)
    mlp_w2 = jax.random.normal(keys[4], (config.mlp_hidden_dim, config.num_value_systems)) * jnp.sqrt(2.0 / config.mlp_hidden_dim)
    mlp_b2 = jnp.zeros(config.num_value_systems)

    return LearnedEmbeddingState(
        weights=weights,
        encoder_w1=encoder_w1,
        encoder_b1=encoder_b1,
        encoder_w2=encoder_w2,
        encoder_b2=encoder_b2,
        mlp_w1=mlp_w1,
        mlp_b1=mlp_b1,
        mlp_w2=mlp_w2,
        mlp_b2=mlp_b2,
    )


def encode_user_history(
    params: Dict,
    user_history: jnp.ndarray,
) -> jnp.ndarray:
    """Encode user interaction history into embedding.

    Args:
        params: Encoder parameters (w1, b1, w2, b2)
        user_history: [B, num_actions] aggregated interaction features
            (e.g., average action probabilities across user's history)

    Returns:
        [B, embedding_dim] user embeddings
    """
    hidden = jax.nn.relu(user_history @ params['encoder_w1'] + params['encoder_b1'])
    embedding = hidden @ params['encoder_w2'] + params['encoder_b2']
    # L2 normalize for stable training
    embedding = embedding / (jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)
    return embedding


def compute_mixture_from_history(
    params: Dict,
    user_history: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mixture weights from user history.

    Args:
        params: All model parameters
        user_history: [B, num_actions] user interaction features

    Returns:
        [B, K] mixture weights
    """
    # Encode history -> embedding
    embedding = encode_user_history(params, user_history)

    # MLP: embedding -> mixture
    hidden = jax.nn.relu(embedding @ params['mlp_w1'] + params['mlp_b1'])
    logits = hidden @ params['mlp_w2'] + params['mlp_b2']
    return jax.nn.softmax(logits, axis=-1)


def train_with_learned_embeddings(
    config: LearnedEmbeddingConfig,
    user_histories: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    archetype_ids: Optional[jnp.ndarray] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[LearnedEmbeddingState, PluralMetrics]:
    """Train pluralistic model with learned embeddings.

    The key difference from Fix 1: instead of pre-computed embeddings,
    we jointly learn an encoder from user interaction history.

    Args:
        config: Training configuration
        user_histories: [N, num_actions] aggregated user interaction features
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        archetype_ids: [N] optional ground truth archetypes (for supervised loss)
        seed: Random seed
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    key = jax.random.PRNGKey(seed)
    state = init_learned_embedding_state(key, config)

    use_classification = config.lambda_classification > 0 and archetype_ids is not None

    # Pack parameters for optimization
    params = {
        'weights': state.weights,
        'encoder_w1': state.encoder_w1,
        'encoder_b1': state.encoder_b1,
        'encoder_w2': state.encoder_w2,
        'encoder_b2': state.encoder_b2,
        'mlp_w1': state.mlp_w1,
        'mlp_b1': state.mlp_b1,
        'mlp_w2': state.mlp_w2,
        'mlp_b2': state.mlp_b2,
    }

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)

    # Create dummy archetype_ids if not provided
    _archetype_ids = archetype_ids if archetype_ids is not None else jnp.zeros(len(user_histories), dtype=jnp.int32)

    def loss_fn(params, batch_history, batch_pref, batch_rej, batch_arch):
        # Encode user history -> embedding
        hidden = jax.nn.relu(batch_history @ params['encoder_w1'] + params['encoder_b1'])
        embedding = hidden @ params['encoder_w2'] + params['encoder_b2']
        embedding = embedding / (jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)

        # Embedding -> mixture weights
        mlp_hidden = jax.nn.relu(embedding @ params['mlp_w1'] + params['mlp_b1'])
        logits = mlp_hidden @ params['mlp_w2'] + params['mlp_b2']
        mixture = jax.nn.softmax(logits, axis=-1)  # [B, K]

        # Compute rewards under each system
        r_pref_systems = jnp.einsum('ba,ka->bk', batch_pref, params['weights'])  # [B, K]
        r_rej_systems = jnp.einsum('ba,ka->bk', batch_rej, params['weights'])    # [B, K]

        # Pluralistic rewards
        r_pref = jnp.sum(r_pref_systems * mixture, axis=-1)  # [B]
        r_rej = jnp.sum(r_rej_systems * mixture, axis=-1)    # [B]

        # Bradley-Terry loss
        L_bt = bradley_terry_loss(r_pref, r_rej)

        # Diversity loss
        L_div = diversity_loss(params['weights'])

        # Entropy loss
        L_ent = entropy_loss(mixture)

        # Classification loss (if supervised)
        L_cls = classification_loss(mixture, batch_arch) if use_classification else 0.0

        L_total = (L_bt
                   + config.lambda_diversity * L_div
                   + config.lambda_entropy * L_ent
                   + config.lambda_classification * L_cls)

        return L_total, (L_bt, L_div, L_ent, L_cls, r_pref, r_rej, mixture)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    n_samples = len(user_histories)
    loss_history = []
    diversity_history = []
    entropy_history = []
    accuracy_history = []

    rng = np.random.default_rng(seed)

    for epoch in range(config.num_epochs):
        perm = rng.permutation(n_samples)
        epoch_losses = []
        epoch_divs = []
        epoch_ents = []
        epoch_accs = []
        epoch_cls = []

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_hist = user_histories[idx]
            batch_pref = probs_preferred[idx]
            batch_rej = probs_rejected[idx]
            batch_arch = _archetype_ids[idx]

            (_, (L_bt, L_div, L_ent, L_cls, r_pref, r_rej, _)), grads = grad_fn(
                params, batch_hist, batch_pref, batch_rej, batch_arch
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

        if verbose and (epoch + 1) % 10 == 0:
            cls_str = f", cls={np.mean(epoch_cls):.4f}" if use_classification and epoch_cls else ""
            print(f"  Epoch {epoch+1}/{config.num_epochs}: "
                  f"loss={loss_history[-1]:.4f}, div={diversity_history[-1]:.4f}, "
                  f"ent={entropy_history[-1]:.4f}, acc={accuracy_history[-1]:.1%}{cls_str}")

    final_state = LearnedEmbeddingState(
        weights=params['weights'],
        encoder_w1=params['encoder_w1'],
        encoder_b1=params['encoder_b1'],
        encoder_w2=params['encoder_w2'],
        encoder_b2=params['encoder_b2'],
        mlp_w1=params['mlp_w1'],
        mlp_b1=params['mlp_b1'],
        mlp_w2=params['mlp_w2'],
        mlp_b2=params['mlp_b2'],
    )

    metrics = PluralMetrics(
        loss_history=loss_history,
        diversity_history=diversity_history,
        entropy_history=entropy_history,
        accuracy_history=accuracy_history,
    )

    return final_state, metrics


def compute_embeddings_from_state(
    state: LearnedEmbeddingState,
    user_histories: jnp.ndarray,
) -> jnp.ndarray:
    """Compute embeddings using trained encoder.

    Args:
        state: Trained model state
        user_histories: [N, num_actions] user interaction features

    Returns:
        [N, embedding_dim] embeddings
    """
    hidden = jax.nn.relu(user_histories @ state.encoder_w1 + state.encoder_b1)
    embedding = hidden @ state.encoder_w2 + state.encoder_b2
    embedding = embedding / (jnp.linalg.norm(embedding, axis=-1, keepdims=True) + 1e-8)
    return embedding


def compute_mixture_from_state(
    state: LearnedEmbeddingState,
    user_histories: jnp.ndarray,
) -> jnp.ndarray:
    """Compute mixture weights using trained model.

    Args:
        state: Trained model state
        user_histories: [N, num_actions] user interaction features

    Returns:
        [N, K] mixture weights
    """
    embedding = compute_embeddings_from_state(state, user_histories)
    hidden = jax.nn.relu(embedding @ state.mlp_w1 + state.mlp_b1)
    logits = hidden @ state.mlp_w2 + state.mlp_b2
    return jax.nn.softmax(logits, axis=-1)


def get_dominant_system_from_history(
    state: LearnedEmbeddingState,
    user_histories: jnp.ndarray,
) -> jnp.ndarray:
    """Get dominant value system for each user.

    Args:
        state: Trained model state
        user_histories: [N, num_actions] user interaction features

    Returns:
        [N] dominant system indices
    """
    mixture = compute_mixture_from_state(state, user_histories)
    return jnp.argmax(mixture, axis=-1)
