"""Two-stage pluralistic reward model.

Stage 1: Cluster users by interaction patterns (k-means)
Stage 2: Learn per-cluster Bradley-Terry weights

This is the robust production approach - each stage uses mature,
well-understood techniques with no complex end-to-end training dynamics.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from sklearn.cluster import KMeans

from enhancements.reward_modeling.pluralistic import bradley_terry_loss
from enhancements.reward_modeling.weights import NUM_ACTIONS, RewardWeights


@dataclass
class TwoStageConfig:
    """Configuration for two-stage model."""
    num_clusters: int = 6

    # Clustering
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10

    # Per-cluster training (Bradley-Terry)
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 64


class TwoStageState(NamedTuple):
    """State for two-stage model."""
    cluster_centers: np.ndarray  # [K, num_features] - k-means centers
    cluster_weights: np.ndarray  # [K, num_actions] - per-cluster reward weights
    kmeans_model: KMeans  # Fitted k-means model for inference


class TwoStageMetrics(NamedTuple):
    """Metrics for two-stage training."""
    # Clustering metrics
    cluster_sizes: list[int]
    cluster_inertia: float

    # Per-cluster training metrics
    per_cluster_accuracy: dict[int, float]
    overall_accuracy: float

    # Loss history per cluster
    loss_histories: dict[int, list[float]]


def cluster_users(
    user_histories: np.ndarray,
    config: TwoStageConfig,
    verbose: bool = True,
) -> tuple[KMeans, np.ndarray]:
    """Stage 1: Cluster users by interaction history.

    Args:
        user_histories: [N, num_features] user interaction features
        config: Configuration
        verbose: Print progress

    Returns:
        (fitted_kmeans, cluster_assignments)
    """
    if verbose:
        print(f"Clustering {len(user_histories)} users into {config.num_clusters} clusters...")

    kmeans = KMeans(
        n_clusters=config.num_clusters,
        random_state=config.kmeans_random_state,
        n_init=config.kmeans_n_init,
    )

    cluster_ids = kmeans.fit_predict(user_histories)

    if verbose:
        unique, counts = np.unique(cluster_ids, return_counts=True)
        print(f"Cluster sizes: {dict(zip(unique, counts))}")
        print(f"Inertia: {kmeans.inertia_:.2f}")

    return kmeans, cluster_ids


def train_per_cluster_weights(
    cluster_ids: np.ndarray,
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: TwoStageConfig,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[int, list[float]], dict[int, float]]:
    """Stage 2: Train Bradley-Terry weights per cluster.

    Args:
        cluster_ids: [N] cluster assignment per sample
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        config: Configuration
        verbose: Print progress

    Returns:
        (weights [K, num_actions], loss_histories, per_cluster_accuracy)
    """
    K = config.num_clusters
    weights = np.zeros((K, NUM_ACTIONS), dtype=np.float32)
    loss_histories = {}
    accuracies = {}

    # Initialize from default weights
    default = RewardWeights.default()

    for k in range(K):
        mask = cluster_ids == k
        n_samples = mask.sum()

        if n_samples == 0:
            if verbose:
                print(f"  Cluster {k}: empty, using default weights")
            weights[k] = default.weights
            loss_histories[k] = []
            accuracies[k] = 0.0
            continue

        if verbose:
            print(f"  Cluster {k}: training on {n_samples} samples...")

        # Get data for this cluster
        pref_k = probs_preferred[mask]
        rej_k = probs_rejected[mask]

        # Train weights for this cluster using gradient descent
        w_k = default.weights.copy()

        import optax
        optimizer = optax.adam(config.learning_rate)
        opt_state = optimizer.init(w_k)

        history = []
        rng = np.random.default_rng(42 + k)

        for epoch in range(config.num_epochs):
            perm = rng.permutation(n_samples)
            epoch_losses = []
            epoch_correct = 0

            for i in range(0, n_samples, config.batch_size):
                idx = perm[i:i + config.batch_size]
                batch_pref = jnp.array(pref_k[idx])
                batch_rej = jnp.array(rej_k[idx])

                # Compute rewards
                r_pref = batch_pref @ w_k
                r_rej = batch_rej @ w_k

                # Loss and gradient
                def loss_fn(w):
                    r_p = batch_pref @ w
                    r_r = batch_rej @ w
                    return bradley_terry_loss(r_p, r_r)

                import jax
                loss, grad = jax.value_and_grad(loss_fn)(w_k)

                updates, opt_state = optimizer.update(grad, opt_state, w_k)
                w_k = optax.apply_updates(w_k, updates)

                epoch_losses.append(float(loss))
                epoch_correct += int(jnp.sum(r_pref > r_rej))

            history.append(np.mean(epoch_losses))

        # Final accuracy
        r_pref_final = pref_k @ np.array(w_k)
        r_rej_final = rej_k @ np.array(w_k)
        acc = np.mean(r_pref_final > r_rej_final)

        weights[k] = np.array(w_k)
        loss_histories[k] = history
        accuracies[k] = float(acc)

        if verbose:
            print(f"    Final loss: {history[-1]:.4f}, accuracy: {acc:.1%}")

    return weights, loss_histories, accuracies


def train_two_stage(
    user_histories: np.ndarray,
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: TwoStageConfig,
    verbose: bool = True,
) -> tuple[TwoStageState, TwoStageMetrics]:
    """Train two-stage pluralistic model.

    Args:
        user_histories: [N, num_features] user interaction features
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        config: Configuration
        verbose: Print progress

    Returns:
        Trained state and metrics
    """
    if verbose:
        print("=" * 60)
        print("STAGE 1: Clustering users")
        print("=" * 60)

    kmeans, cluster_ids = cluster_users(user_histories, config, verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Training per-cluster weights")
        print("=" * 60)

    weights, loss_histories, per_cluster_acc = train_per_cluster_weights(
        cluster_ids, probs_preferred, probs_rejected, config, verbose
    )

    # Compute overall accuracy
    all_r_pref = np.array([probs_preferred[i] @ weights[cluster_ids[i]]
                          for i in range(len(cluster_ids))])
    all_r_rej = np.array([probs_rejected[i] @ weights[cluster_ids[i]]
                         for i in range(len(cluster_ids))])
    overall_acc = np.mean(all_r_pref > all_r_rej)

    # Cluster sizes
    unique, counts = np.unique(cluster_ids, return_counts=True)
    cluster_sizes = [int(counts[unique == k][0]) if k in unique else 0
                     for k in range(config.num_clusters)]

    state = TwoStageState(
        cluster_centers=kmeans.cluster_centers_,
        cluster_weights=weights,
        kmeans_model=kmeans,
    )

    metrics = TwoStageMetrics(
        cluster_sizes=cluster_sizes,
        cluster_inertia=float(kmeans.inertia_),
        per_cluster_accuracy=per_cluster_acc,
        overall_accuracy=float(overall_acc),
        loss_histories=loss_histories,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Overall accuracy: {overall_acc:.1%}")

    return state, metrics


def predict_cluster(
    state: TwoStageState,
    user_history: np.ndarray,
) -> int:
    """Predict cluster for a user.

    Args:
        state: Trained model state
        user_history: [num_features] or [B, num_features] user features

    Returns:
        Cluster ID(s)
    """
    return state.kmeans_model.predict(user_history.reshape(-1, user_history.shape[-1]))


def compute_reward(
    state: TwoStageState,
    user_history: np.ndarray,
    action_probs: np.ndarray,
) -> np.ndarray:
    """Compute reward for user-content pair.

    Args:
        state: Trained model state
        user_history: [num_features] user features
        action_probs: [num_actions] or [C, num_actions] content action probs

    Returns:
        Reward(s)
    """
    cluster_id = predict_cluster(state, user_history)[0]
    weights = state.cluster_weights[cluster_id]

    if action_probs.ndim == 1:
        return float(action_probs @ weights)
    else:
        return action_probs @ weights
