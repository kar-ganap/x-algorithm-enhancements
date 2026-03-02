"""Per-stakeholder reward model training.

Trains three models optimized for different stakeholders:
1. User-Optimized: engagement - discomfort
2. Platform-Optimized: total engagement
3. Society-Optimized: diversity - polarization

Each model learns different weights, revealing what each stakeholder values.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Define constants locally to avoid import chain that triggers grok
ACTION_NAMES: list[str] = [
    "favorite",
    "reply",
    "repost",
    "photo_expand",
    "click",
    "profile_click",
    "vqv",
    "share",
    "share_via_dm",
    "share_via_copy_link",
    "dwell",
    "quote",
    "quoted_click",
    "follow_author",
    "not_interested",
    "block_author",
    "mute_author",
    "report",
]

NUM_ACTIONS = len(ACTION_NAMES)
ACTION_INDICES: dict[str, int] = {name: i for i, name in enumerate(ACTION_NAMES)}


@dataclass
class RewardWeights:
    """Container for reward model weights."""
    weights: np.ndarray
    name: str = "default"

    @classmethod
    def default(cls) -> "RewardWeights":
        """Default weights: positive=+1, negative=-1, neutral=0."""
        weights = np.zeros(NUM_ACTIONS, dtype=np.float32)

        # Positive signals
        positive = ["favorite", "reply", "repost", "follow_author", "quote", "share"]
        for action in positive:
            weights[ACTION_INDICES[action]] = 1.0

        # Negative signals
        negative = ["block_author", "mute_author", "report", "not_interested"]
        for action in negative:
            weights[ACTION_INDICES[action]] = -1.0

        return cls(weights=weights, name="default")


class StakeholderType(Enum):
    """The three stakeholder types."""
    USER = "user"
    PLATFORM = "platform"
    SOCIETY = "society"


@dataclass
class StakeholderTrainingConfig:
    """Configuration for stakeholder-specific training."""
    stakeholder: StakeholderType

    # Common training params
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 64

    # User-specific: penalty for negative signals
    user_discomfort_weight: float = 2.0

    # Society-specific: diversity and polarization weights
    society_diversity_weight: float = 0.5
    society_polarization_weight: float = 1.0

    # Regularization
    l2_weight: float = 0.001


class StakeholderModelState(NamedTuple):
    """State of a trained stakeholder model."""
    stakeholder: StakeholderType
    weights: np.ndarray  # [num_actions]
    training_loss_history: list[float]
    training_accuracy: float


class StakeholderModelMetrics(NamedTuple):
    """Metrics for a stakeholder model."""
    stakeholder: StakeholderType
    accuracy: float
    user_utility: float
    platform_utility: float
    society_utility: float


# =============================================================================
# Loss Functions
# =============================================================================

def bradley_terry_loss(r_preferred: jnp.ndarray, r_rejected: jnp.ndarray) -> jnp.ndarray:
    """Standard Bradley-Terry loss: -log(σ(r_pref - r_rej))."""
    return jnp.mean(jnp.log(1 + jnp.exp(-(r_preferred - r_rejected))))


def user_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    discomfort_weight: float = 2.0,
) -> jnp.ndarray:
    """User-optimized loss: BT loss + discomfort penalty.

    Users want engagement but hate content that triggers blocks/reports.
    """
    # Compute rewards
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    # Standard BT loss
    bt_loss = bradley_terry_loss(r_pref, r_rej)

    # Discomfort penalty: penalize high negative action predictions
    negative_actions = [
        ACTION_INDICES["block_author"],
        ACTION_INDICES["mute_author"],
        ACTION_INDICES["report"],
        ACTION_INDICES["not_interested"],
    ]

    discomfort = 0.0
    for idx in negative_actions:
        # Penalize recommending content with high negative signal probability
        discomfort += jnp.mean(probs_preferred[:, idx])

    return bt_loss + discomfort_weight * discomfort


def platform_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
) -> jnp.ndarray:
    """Platform-optimized loss: standard BT loss.

    Platform wants all engagement - doesn't penalize negative signals.
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    return bradley_terry_loss(r_pref, r_rej)


def society_loss(
    weights: jnp.ndarray,
    probs_preferred: jnp.ndarray,
    probs_rejected: jnp.ndarray,
    content_topics: jnp.ndarray,
    user_archetypes: jnp.ndarray,
    diversity_weight: float = 0.5,
    polarization_weight: float = 1.0,
) -> jnp.ndarray:
    """Society-optimized loss: BT loss + diversity bonus - polarization penalty.

    Society wants diverse content and reduced echo chambers.
    """
    r_pref = probs_preferred @ weights
    r_rej = probs_rejected @ weights

    # Standard BT loss
    bt_loss = bradley_terry_loss(r_pref, r_rej)

    # Diversity loss: encourage recommending varied topics
    # Penalize when preferred content is same topic as user's archetype
    # (promotes exploration beyond user's comfort zone)
    same_topic = (content_topics == user_archetypes).astype(jnp.float32)
    diversity_penalty = jnp.mean(same_topic)

    # Polarization penalty: for political users, penalize same-side content
    # Archetypes 2=political_L, 3=political_R
    # Topics 2=politics_L, 3=politics_R
    is_political_user = jnp.logical_or(user_archetypes == 2, user_archetypes == 3)

    # Same-side content for political users
    same_side_L = jnp.logical_and(user_archetypes == 2, content_topics == 2)
    same_side_R = jnp.logical_and(user_archetypes == 3, content_topics == 3)
    same_side = jnp.logical_or(same_side_L, same_side_R).astype(jnp.float32)

    polarization_penalty = jnp.sum(same_side) / (jnp.sum(is_political_user.astype(jnp.float32)) + 1e-8)

    return bt_loss + diversity_weight * diversity_penalty + polarization_weight * polarization_penalty


# =============================================================================
# Training Functions
# =============================================================================

def train_user_model(
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: StakeholderTrainingConfig,
    verbose: bool = True,
) -> StakeholderModelState:
    """Train user-optimized reward model."""
    if verbose:
        print("Training USER-optimized model...")
        print(f"  Discomfort weight: {config.user_discomfort_weight}")

    # Initialize weights
    w = jnp.array(RewardWeights.default().weights)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(w)

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

            def loss_fn(weights):
                return user_loss(
                    weights, batch_pref, batch_rej,
                    config.user_discomfort_weight
                ) + config.l2_weight * jnp.sum(weights ** 2)

            loss, grad = jax.value_and_grad(loss_fn)(w)
            updates, opt_state = optimizer.update(grad, opt_state, w)
            w = optax.apply_updates(w, updates)

            epoch_losses.append(float(loss))

        loss_history.append(np.mean(epoch_losses))

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: loss = {loss_history[-1]:.4f}")

    # Compute final accuracy
    r_pref = probs_preferred @ np.array(w)
    r_rej = probs_rejected @ np.array(w)
    accuracy = float(np.mean(r_pref > r_rej))

    if verbose:
        print(f"  Final accuracy: {accuracy:.1%}")

    return StakeholderModelState(
        stakeholder=StakeholderType.USER,
        weights=np.array(w),
        training_loss_history=loss_history,
        training_accuracy=accuracy,
    )


def train_platform_model(
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: StakeholderTrainingConfig,
    verbose: bool = True,
) -> StakeholderModelState:
    """Train platform-optimized reward model."""
    if verbose:
        print("Training PLATFORM-optimized model...")

    w = jnp.array(RewardWeights.default().weights)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(w)

    loss_history = []
    rng = np.random.default_rng(43)
    n_samples = len(probs_preferred)

    for epoch in range(config.num_epochs):
        perm = rng.permutation(n_samples)
        epoch_losses = []

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_pref = jnp.array(probs_preferred[idx])
            batch_rej = jnp.array(probs_rejected[idx])

            def loss_fn(weights):
                return platform_loss(weights, batch_pref, batch_rej) + \
                       config.l2_weight * jnp.sum(weights ** 2)

            loss, grad = jax.value_and_grad(loss_fn)(w)
            updates, opt_state = optimizer.update(grad, opt_state, w)
            w = optax.apply_updates(w, updates)

            epoch_losses.append(float(loss))

        loss_history.append(np.mean(epoch_losses))

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: loss = {loss_history[-1]:.4f}")

    r_pref = probs_preferred @ np.array(w)
    r_rej = probs_rejected @ np.array(w)
    accuracy = float(np.mean(r_pref > r_rej))

    if verbose:
        print(f"  Final accuracy: {accuracy:.1%}")

    return StakeholderModelState(
        stakeholder=StakeholderType.PLATFORM,
        weights=np.array(w),
        training_loss_history=loss_history,
        training_accuracy=accuracy,
    )


def train_society_model(
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    content_topics: np.ndarray,
    user_archetypes: np.ndarray,
    config: StakeholderTrainingConfig,
    verbose: bool = True,
) -> StakeholderModelState:
    """Train society-optimized reward model."""
    if verbose:
        print("Training SOCIETY-optimized model...")
        print(f"  Diversity weight: {config.society_diversity_weight}")
        print(f"  Polarization weight: {config.society_polarization_weight}")

    w = jnp.array(RewardWeights.default().weights)

    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(w)

    loss_history = []
    rng = np.random.default_rng(44)
    n_samples = len(probs_preferred)

    for epoch in range(config.num_epochs):
        perm = rng.permutation(n_samples)
        epoch_losses = []

        for i in range(0, n_samples, config.batch_size):
            idx = perm[i:i + config.batch_size]
            batch_pref = jnp.array(probs_preferred[idx])
            batch_rej = jnp.array(probs_rejected[idx])
            batch_topics = jnp.array(content_topics[idx])
            batch_archetypes = jnp.array(user_archetypes[idx])

            def loss_fn(weights):
                return society_loss(
                    weights, batch_pref, batch_rej,
                    batch_topics, batch_archetypes,
                    config.society_diversity_weight,
                    config.society_polarization_weight
                ) + config.l2_weight * jnp.sum(weights ** 2)

            loss, grad = jax.value_and_grad(loss_fn)(w)
            updates, opt_state = optimizer.update(grad, opt_state, w)
            w = optax.apply_updates(w, updates)

            epoch_losses.append(float(loss))

        loss_history.append(np.mean(epoch_losses))

        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch + 1}: loss = {loss_history[-1]:.4f}")

    r_pref = probs_preferred @ np.array(w)
    r_rej = probs_rejected @ np.array(w)
    accuracy = float(np.mean(r_pref > r_rej))

    if verbose:
        print(f"  Final accuracy: {accuracy:.1%}")

    return StakeholderModelState(
        stakeholder=StakeholderType.SOCIETY,
        weights=np.array(w),
        training_loss_history=loss_history,
        training_accuracy=accuracy,
    )


def train_all_stakeholder_models(
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    content_topics: np.ndarray,
    user_archetypes: np.ndarray,
    config: StakeholderTrainingConfig | None = None,
    verbose: bool = True,
) -> dict[StakeholderType, StakeholderModelState]:
    """Train all three stakeholder models.

    Args:
        probs_preferred: [N, num_actions] preferred content action probs
        probs_rejected: [N, num_actions] rejected content action probs
        content_topics: [N] topic index of preferred content
        user_archetypes: [N] archetype index of users
        config: Training configuration
        verbose: Print progress

    Returns:
        Dict mapping StakeholderType to trained model state
    """
    if config is None:
        config = StakeholderTrainingConfig(stakeholder=StakeholderType.USER)

    if verbose:
        print("=" * 60)
        print("TRAINING PER-STAKEHOLDER MODELS")
        print("=" * 60)

    models = {}

    # Train user model
    if verbose:
        print("\n[1/3] User-Optimized Model")
        print("-" * 40)
    config.stakeholder = StakeholderType.USER
    models[StakeholderType.USER] = train_user_model(
        probs_preferred, probs_rejected, config, verbose
    )

    # Train platform model
    if verbose:
        print("\n[2/3] Platform-Optimized Model")
        print("-" * 40)
    config.stakeholder = StakeholderType.PLATFORM
    models[StakeholderType.PLATFORM] = train_platform_model(
        probs_preferred, probs_rejected, config, verbose
    )

    # Train society model
    if verbose:
        print("\n[3/3] Society-Optimized Model")
        print("-" * 40)
    config.stakeholder = StakeholderType.SOCIETY
    models[StakeholderType.SOCIETY] = train_society_model(
        probs_preferred, probs_rejected, content_topics, user_archetypes, config, verbose
    )

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        for stype, model in models.items():
            print(f"  {stype.value}: accuracy = {model.training_accuracy:.1%}")

    return models


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_weights(
    models: dict[StakeholderType, StakeholderModelState],
) -> dict[str, dict[str, float]]:
    """Compare learned weights across stakeholder models.

    Returns:
        Dict mapping action name to dict of stakeholder weights
    """
    # ACTION_NAMES is defined at module level

    comparison = {}

    for action_name in ACTION_NAMES:
        idx = ACTION_INDICES[action_name]
        comparison[action_name] = {
            stype.value: float(model.weights[idx])
            for stype, model in models.items()
        }

    return comparison


def compute_ranking_correlation(
    models: dict[StakeholderType, StakeholderModelState],
    content_probs: np.ndarray,
) -> dict[tuple[str, str], float]:
    """Compute Kendall's tau correlation between model rankings.

    Args:
        models: Trained stakeholder models
        content_probs: [M, num_actions] action probs for content items

    Returns:
        Dict mapping (model1, model2) to Kendall's tau
    """
    from scipy.stats import kendalltau

    # Compute scores for each model
    scores = {}
    for stype, model in models.items():
        scores[stype] = content_probs @ model.weights

    # Compute pairwise correlations
    correlations = {}
    types = list(models.keys())

    for i, t1 in enumerate(types):
        for t2 in types[i+1:]:
            tau, _ = kendalltau(scores[t1], scores[t2])
            correlations[(t1.value, t2.value)] = float(tau)

    return correlations


def identify_contested_content(
    models: dict[StakeholderType, StakeholderModelState],
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """Identify content items where models disagree most.

    Args:
        models: Trained stakeholder models
        content_probs: [M, num_actions] action probs for content
        content_topics: [M] topic index for content
        top_k: Number of most contested items to return

    Returns:
        List of dicts with contested content info
    """
    M = len(content_probs)

    # Compute ranks for each model
    ranks = {}
    for stype, model in models.items():
        scores = content_probs @ model.weights
        # Convert scores to ranks (higher score = lower rank number)
        ranks[stype] = np.argsort(np.argsort(-scores))

    # Compute rank variance for each content item
    rank_matrix = np.array([ranks[stype] for stype in models.keys()])  # [3, M]
    rank_variance = np.var(rank_matrix, axis=0)

    # Get top-k most contested
    contested_indices = np.argsort(-rank_variance)[:top_k]

    topic_names = ["sports", "tech", "politics_L", "politics_R", "entertainment", "news"]

    contested = []
    for idx in contested_indices:
        contested.append({
            "content_idx": int(idx),
            "topic": topic_names[content_topics[idx]] if content_topics[idx] < len(topic_names) else f"topic_{content_topics[idx]}",
            "rank_variance": float(rank_variance[idx]),
            "ranks": {stype.value: int(ranks[stype][idx]) for stype in models.keys()},
        })

    return contested


def compute_cross_exposure(
    models: dict[StakeholderType, StakeholderModelState],
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    user_archetypes: np.ndarray,
    top_k: int = 10,
) -> dict[str, float]:
    """Compute cross-partisan exposure rate for each model.

    For political users, what % of their top-K includes opposing political content?

    Args:
        models: Trained stakeholder models
        content_probs: [M, num_actions] action probs
        content_topics: [M] topic indices
        user_archetypes: [N] user archetypes
        top_k: Number of recommendations per user

    Returns:
        Dict mapping model name to cross-exposure rate
    """
    cross_exposure = {}

    for stype, model in models.items():
        # Compute content scores
        scores = content_probs @ model.weights

        # Get top-K for each "user" (simulate by archetype preferences)
        total_cross = 0
        total_political = 0

        for archetype in [2, 3]:  # political_L, political_R
            # Get top-K content
            top_indices = np.argsort(-scores)[:top_k]
            top_topics = content_topics[top_indices]

            # Count cross-exposure
            if archetype == 2:  # political_L
                cross = np.sum(top_topics == 3)  # politics_R
            else:  # political_R
                cross = np.sum(top_topics == 2)  # politics_L

            total_cross += cross
            total_political += 1

        cross_exposure[stype.value] = total_cross / (total_political * top_k) if total_political > 0 else 0

    return cross_exposure
