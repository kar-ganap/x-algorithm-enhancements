"""Two-stage pluralistic reward model with GMM soft clustering.

Stage 1: Cluster users by interaction patterns (GMM - soft membership)
Stage 2: Learn per-cluster Bradley-Terry weights (weighted by membership)

Key difference from k-means version:
- GMM gives soft cluster membership probabilities
- Training uses membership-weighted samples
- Inference blends cluster weights by membership
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from enhancements.reward_modeling.weights import NUM_ACTIONS, RewardWeights


class ClusteringMethod(Enum):
    """Clustering method for Stage 1."""
    KMEANS = "kmeans"  # Hard clustering
    GMM = "gmm"        # Soft clustering


@dataclass
class TwoStageGMMConfig:
    """Configuration for two-stage model with GMM support."""
    num_clusters: int = 6

    # Clustering method
    clustering_method: ClusteringMethod = ClusteringMethod.GMM

    # GMM-specific
    gmm_covariance_type: str = "full"  # "full", "tied", "diag", "spherical"
    gmm_random_state: int = 42
    gmm_n_init: int = 5
    gmm_max_iter: int = 200

    # K-means fallback
    kmeans_random_state: int = 42
    kmeans_n_init: int = 10

    # Per-cluster training (Bradley-Terry)
    learning_rate: float = 0.01
    num_epochs: int = 100
    batch_size: int = 64

    # Soft clustering options
    use_soft_training: bool = True   # Weight samples by membership during training
    use_soft_inference: bool = True  # Blend cluster weights during inference
    membership_threshold: float = 0.1  # Min membership to contribute to cluster


class TwoStageGMMState(NamedTuple):
    """State for two-stage GMM model."""
    cluster_centers: np.ndarray  # [K, num_features] - cluster means
    cluster_weights: np.ndarray  # [K, num_actions] - per-cluster reward weights
    clustering_model: GaussianMixture | KMeans  # Fitted model
    clustering_method: ClusteringMethod


class TwoStageGMMMetrics(NamedTuple):
    """Metrics for two-stage GMM training."""
    cluster_sizes: list[int]  # Hard assignment sizes
    soft_cluster_sizes: list[float]  # Sum of membership probabilities

    per_cluster_accuracy: dict[int, float]
    overall_accuracy: float

    loss_histories: dict[int, list[float]]

    # GMM-specific
    bic: float  # Bayesian Information Criterion (lower is better)
    aic: float  # Akaike Information Criterion (lower is better)


def cluster_users_gmm(
    user_histories: np.ndarray,
    config: TwoStageGMMConfig,
    verbose: bool = True,
) -> tuple[GaussianMixture, np.ndarray, np.ndarray]:
    """Stage 1: Cluster users with GMM (soft membership).

    Args:
        user_histories: [N, num_features] user interaction features
        config: Configuration
        verbose: Print progress

    Returns:
        (fitted_gmm, hard_assignments, soft_memberships [N, K])
    """
    if verbose:
        print(f"GMM clustering {len(user_histories)} users into {config.num_clusters} clusters...")

    gmm = GaussianMixture(
        n_components=config.num_clusters,
        covariance_type=config.gmm_covariance_type,
        random_state=config.gmm_random_state,
        n_init=config.gmm_n_init,
        max_iter=config.gmm_max_iter,
    )

    hard_assignments = gmm.fit_predict(user_histories)
    soft_memberships = gmm.predict_proba(user_histories)  # [N, K]

    if verbose:
        # Hard assignment sizes
        unique, counts = np.unique(hard_assignments, return_counts=True)
        print(f"Hard cluster sizes: {dict(zip(unique, counts))}")

        # Soft cluster sizes (sum of memberships)
        soft_sizes = soft_memberships.sum(axis=0)
        print(f"Soft cluster sizes: {dict(enumerate(soft_sizes.round(1)))}")

        # Model fit metrics
        print(f"BIC: {gmm.bic(user_histories):.2f}, AIC: {gmm.aic(user_histories):.2f}")

        # Average entropy of memberships (higher = more uncertain)
        entropy = -np.sum(soft_memberships * np.log(soft_memberships + 1e-10), axis=1).mean()
        max_entropy = np.log(config.num_clusters)
        print(f"Avg membership entropy: {entropy:.3f} (max: {max_entropy:.3f})")

    return gmm, hard_assignments, soft_memberships


def train_per_cluster_weights_soft(
    soft_memberships: np.ndarray,
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: TwoStageGMMConfig,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[int, list[float]], dict[int, float]]:
    """Stage 2: Train Bradley-Terry weights with soft membership weighting.

    Each sample contributes to cluster k proportionally to its membership π_k.

    Args:
        soft_memberships: [N, K] membership probabilities
        probs_preferred: [N, num_actions] preferred item action probs
        probs_rejected: [N, num_actions] rejected item action probs
        config: Configuration
        verbose: Print progress

    Returns:
        (weights [K, num_actions], loss_histories, per_cluster_accuracy)
    """
    K = config.num_clusters
    N = len(probs_preferred)
    weights = np.zeros((K, NUM_ACTIONS), dtype=np.float32)
    loss_histories: dict[int, list[float]] = {}
    accuracies: dict[int, float] = {}

    default = RewardWeights.default()

    for k in range(K):
        membership_k = soft_memberships[:, k]  # [N]
        effective_samples = membership_k.sum()

        if effective_samples < 1.0:
            if verbose:
                print(f"  Cluster {k}: {effective_samples:.1f} effective samples, using default weights")
            weights[k] = default.weights
            loss_histories[k] = []
            accuracies[k] = 0.0
            continue

        if verbose:
            print(f"  Cluster {k}: {effective_samples:.1f} effective samples...")

        # Train with membership-weighted loss
        w_k = jnp.array(default.weights.copy())

        optimizer = optax.adam(config.learning_rate)
        opt_state = optimizer.init(w_k)

        history = []
        rng = np.random.default_rng(42 + k)

        # Filter to samples with meaningful membership
        significant_mask = membership_k > config.membership_threshold
        n_significant = significant_mask.sum()

        if n_significant < 10:
            # Too few significant samples, use all with weights
            indices = np.arange(N)
            sample_weights = membership_k
        else:
            # Use only significant samples for efficiency
            indices = np.where(significant_mask)[0]
            sample_weights = membership_k[significant_mask]

        # Normalize weights
        sample_weights = sample_weights / sample_weights.sum()

        pref_k = probs_preferred[indices]
        rej_k = probs_rejected[indices]
        n_train = len(indices)

        for epoch in range(config.num_epochs):
            perm = rng.permutation(n_train)
            epoch_losses = []

            for i in range(0, n_train, config.batch_size):
                idx = perm[i:i + config.batch_size]
                batch_pref = jnp.array(pref_k[idx])
                batch_rej = jnp.array(rej_k[idx])
                batch_weights = jnp.array(sample_weights[idx])

                # Weighted Bradley-Terry loss
                def loss_fn(w: jnp.ndarray) -> jnp.ndarray:
                    r_p = batch_pref @ w
                    r_r = batch_rej @ w
                    # Per-sample loss
                    per_sample_loss = jnp.log(1 + jnp.exp(-(r_p - r_r)))
                    # Weighted mean
                    return jnp.sum(per_sample_loss * batch_weights) / (batch_weights.sum() + 1e-8)

                loss, grad = jax.value_and_grad(loss_fn)(w_k)

                updates, opt_state = optimizer.update(grad, opt_state, w_k)
                w_k = optax.apply_updates(w_k, updates)

                epoch_losses.append(float(loss))

            history.append(np.mean(epoch_losses))

        # Compute weighted accuracy
        r_pref_final = pref_k @ np.array(w_k)
        r_rej_final = rej_k @ np.array(w_k)
        correct = (r_pref_final > r_rej_final).astype(float)
        acc = np.sum(correct * sample_weights)  # Weighted accuracy

        weights[k] = np.array(w_k)
        loss_histories[k] = history
        accuracies[k] = float(acc)

        if verbose:
            print(f"    Final loss: {history[-1]:.4f}, weighted accuracy: {acc:.1%}")

    return weights, loss_histories, accuracies


def train_two_stage_gmm(
    user_histories: np.ndarray,
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    config: TwoStageGMMConfig,
    verbose: bool = True,
) -> tuple[TwoStageGMMState, TwoStageGMMMetrics]:
    """Train two-stage pluralistic model with GMM.

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
        print(f"STAGE 1: Clustering users ({config.clustering_method.value})")
        print("=" * 60)

    if config.clustering_method == ClusteringMethod.GMM:
        gmm, hard_ids, soft_memberships = cluster_users_gmm(user_histories, config, verbose)
        model: GaussianMixture | KMeans = gmm
        centers = np.array(gmm.means_)  # Ensure it's ndarray
        bic = gmm.bic(user_histories)
        aic = gmm.aic(user_histories)
    else:
        # Fallback to k-means
        from enhancements.reward_modeling.two_stage import TwoStageConfig, cluster_users
        kmeans_config = TwoStageConfig(
            num_clusters=config.num_clusters,
            kmeans_random_state=config.kmeans_random_state,
            kmeans_n_init=config.kmeans_n_init,
        )
        kmeans, hard_ids = cluster_users(user_histories, kmeans_config, verbose)
        model = kmeans
        centers = kmeans.cluster_centers_
        # Create one-hot soft memberships for k-means
        soft_memberships = np.zeros((len(user_histories), config.num_clusters))
        soft_memberships[np.arange(len(hard_ids)), hard_ids] = 1.0
        bic, aic = 0.0, 0.0

    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Training per-cluster weights")
        print("=" * 60)

    if config.use_soft_training and config.clustering_method == ClusteringMethod.GMM:
        weights, loss_histories, per_cluster_acc = train_per_cluster_weights_soft(
            soft_memberships, probs_preferred, probs_rejected, config, verbose
        )
    else:
        # Hard training (original method)
        from enhancements.reward_modeling.two_stage import TwoStageConfig, train_per_cluster_weights
        hard_config = TwoStageConfig(
            num_clusters=config.num_clusters,
            learning_rate=config.learning_rate,
            num_epochs=config.num_epochs,
            batch_size=config.batch_size,
        )
        weights, loss_histories, per_cluster_acc = train_per_cluster_weights(
            hard_ids, probs_preferred, probs_rejected, hard_config, verbose
        )

    # Compute overall accuracy
    if config.use_soft_inference and config.clustering_method == ClusteringMethod.GMM:
        # Soft inference: blend weights by membership
        blended_weights = soft_memberships @ weights  # [N, num_actions]
        all_r_pref = np.sum(probs_preferred * blended_weights, axis=1)
        all_r_rej = np.sum(probs_rejected * blended_weights, axis=1)
    else:
        # Hard inference
        all_r_pref = np.array([probs_preferred[i] @ weights[hard_ids[i]]
                              for i in range(len(hard_ids))])
        all_r_rej = np.array([probs_rejected[i] @ weights[hard_ids[i]]
                             for i in range(len(hard_ids))])

    overall_acc = np.mean(all_r_pref > all_r_rej)

    # Cluster sizes
    unique, counts = np.unique(hard_ids, return_counts=True)
    cluster_sizes = [int(counts[unique == k][0]) if k in unique else 0
                     for k in range(config.num_clusters)]
    soft_cluster_sizes = soft_memberships.sum(axis=0).tolist()

    state = TwoStageGMMState(
        cluster_centers=centers,
        cluster_weights=weights,
        clustering_model=model,
        clustering_method=config.clustering_method,
    )

    metrics = TwoStageGMMMetrics(
        cluster_sizes=cluster_sizes,
        soft_cluster_sizes=soft_cluster_sizes,
        per_cluster_accuracy=per_cluster_acc,
        overall_accuracy=float(overall_acc),
        loss_histories=loss_histories,
        bic=float(bic),
        aic=float(aic),
    )

    if verbose:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Overall accuracy: {overall_acc:.1%}")
        if config.clustering_method == ClusteringMethod.GMM:
            print(f"Using soft inference: {config.use_soft_inference}")

    return state, metrics


def predict_membership(
    state: TwoStageGMMState,
    user_history: np.ndarray,
) -> np.ndarray:
    """Predict soft cluster membership for a user.

    Args:
        state: Trained model state
        user_history: [num_features] or [B, num_features] user features

    Returns:
        [K] or [B, K] membership probabilities
    """
    user_history = user_history.reshape(-1, user_history.shape[-1])

    if state.clustering_method == ClusteringMethod.GMM:
        # GMM has predict_proba
        gmm_model = state.clustering_model
        if isinstance(gmm_model, GaussianMixture):
            return gmm_model.predict_proba(user_history)
        # Fallback (shouldn't happen)
        raise TypeError("Expected GaussianMixture for GMM clustering method")
    else:
        # K-means: one-hot
        hard_ids = state.clustering_model.predict(user_history)
        K = len(state.cluster_weights)
        memberships = np.zeros((len(hard_ids), K))
        memberships[np.arange(len(hard_ids)), hard_ids] = 1.0
        return memberships


def compute_reward_soft(
    state: TwoStageGMMState,
    user_history: np.ndarray,
    action_probs: np.ndarray,
    use_soft: bool = True,
) -> float | np.ndarray:
    """Compute reward with optional soft membership blending.

    Args:
        state: Trained model state
        user_history: [num_features] user features
        action_probs: [num_actions] or [C, num_actions] content action probs
        use_soft: If True, blend rewards by membership; else use hard assignment

    Returns:
        Reward(s)
    """
    membership = predict_membership(state, user_history)  # [1, K]

    if use_soft and state.clustering_method == ClusteringMethod.GMM:
        # Blend cluster weights by membership
        blended_weights = membership @ state.cluster_weights  # [1, num_actions]
        blended_weights = blended_weights.squeeze(0)  # [num_actions]

        if action_probs.ndim == 1:
            return float(action_probs @ blended_weights)
        else:
            return action_probs @ blended_weights
    else:
        # Hard assignment
        cluster_id = np.argmax(membership, axis=-1)[0]
        weights = state.cluster_weights[cluster_id]

        if action_probs.ndim == 1:
            return float(action_probs @ weights)
        else:
            return action_probs @ weights
