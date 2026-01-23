"""Embedding probes for verifying learned representations.

Tests whether:
1. Users of the same archetype cluster together in embedding space
2. Posts of the same topic cluster together in embedding space

Uses silhouette score to measure cluster quality.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from enhancements.data.ground_truth import UserArchetype, ContentTopic
from enhancements.data.synthetic_twitter import SyntheticTwitterDataset
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter


@dataclass
class EmbeddingProbeResults:
    """Results from embedding probe analysis."""
    # Clustering metrics
    user_silhouette: float  # Silhouette score for user archetype clustering
    topic_silhouette: float  # Silhouette score for post topic clustering

    # Per-archetype metrics
    archetype_cluster_sizes: Dict[str, int]
    archetype_intra_distances: Dict[str, float]  # Avg distance within archetype
    archetype_inter_distances: Dict[str, float]  # Avg distance to other archetypes

    # Per-topic metrics
    topic_cluster_sizes: Dict[str, int]
    topic_intra_distances: Dict[str, float]
    topic_inter_distances: Dict[str, float]

    # Pass/fail
    user_clustering_pass: bool  # silhouette > threshold
    topic_clustering_pass: bool

    def __repr__(self) -> str:
        return (
            f"EmbeddingProbeResults(\n"
            f"  user_silhouette={self.user_silhouette:.4f} ({'PASS' if self.user_clustering_pass else 'FAIL'})\n"
            f"  topic_silhouette={self.topic_silhouette:.4f} ({'PASS' if self.topic_clustering_pass else 'FAIL'})\n"
            f")"
        )


def compute_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute silhouette score for clustering quality.

    Silhouette score ranges from -1 to 1:
    - 1: Perfect clustering
    - 0: Overlapping clusters
    - -1: Wrong clustering

    Args:
        embeddings: [N, D] embeddings
        labels: [N] cluster labels (integers)

    Returns:
        Silhouette score
    """
    n_samples = len(embeddings)
    if n_samples < 2:
        return 0.0

    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters < 2:
        return 0.0

    # Compute pairwise distances
    # For efficiency, we'll use a vectorized approach
    distances = compute_pairwise_distances(embeddings)

    silhouettes = []
    for i in range(n_samples):
        label_i = labels[i]

        # a(i) = mean distance to other points in same cluster
        same_cluster = labels == label_i
        same_cluster[i] = False  # Exclude self
        if np.sum(same_cluster) > 0:
            a_i = np.mean(distances[i, same_cluster])
        else:
            a_i = 0.0

        # b(i) = min mean distance to points in other clusters
        b_i = float('inf')
        for other_label in unique_labels:
            if other_label == label_i:
                continue
            other_cluster = labels == other_label
            if np.sum(other_cluster) > 0:
                mean_dist = np.mean(distances[i, other_cluster])
                b_i = min(b_i, mean_dist)

        if b_i == float('inf'):
            b_i = 0.0

        # Silhouette
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0

        silhouettes.append(s_i)

    return float(np.mean(silhouettes))


def compute_pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances.

    Args:
        embeddings: [N, D] embeddings

    Returns:
        [N, N] distance matrix
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(embeddings ** 2, axis=1, keepdims=True)
    dot_products = embeddings @ embeddings.T
    distances_sq = sq_norms + sq_norms.T - 2 * dot_products
    distances_sq = np.maximum(distances_sq, 0)  # Handle numerical errors
    return np.sqrt(distances_sq)


def compute_cluster_distances(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Compute intra-cluster and inter-cluster distances.

    Args:
        embeddings: [N, D] embeddings
        labels: [N] cluster labels

    Returns:
        (intra_distances, inter_distances) dicts mapping label to avg distance
    """
    unique_labels = np.unique(labels)
    distances = compute_pairwise_distances(embeddings)

    intra = {}
    inter = {}

    for label in unique_labels:
        mask = labels == label
        indices = np.where(mask)[0]

        if len(indices) < 2:
            intra[label] = 0.0
            inter[label] = 0.0
            continue

        # Intra-cluster: mean distance within cluster
        cluster_distances = distances[np.ix_(indices, indices)]
        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices(len(indices), k=1)
        intra_dists = cluster_distances[triu_indices]
        intra[label] = float(np.mean(intra_dists)) if len(intra_dists) > 0 else 0.0

        # Inter-cluster: mean distance to points outside cluster
        other_mask = ~mask
        if np.any(other_mask):
            other_indices = np.where(other_mask)[0]
            inter_dists = distances[np.ix_(indices, other_indices)]
            inter[label] = float(np.mean(inter_dists))
        else:
            inter[label] = 0.0

    return intra, inter


def test_user_archetype_clustering(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    sample_size: int = 500,
    threshold: float = 0.2,
) -> Tuple[float, bool, Dict[str, int], Dict[str, float], Dict[str, float]]:
    """Test if users of same archetype cluster together.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        sample_size: Number of users to sample
        threshold: Minimum silhouette score to pass

    Returns:
        (silhouette, pass_bool, cluster_sizes, intra_distances, inter_distances)
    """
    # Sample users
    rng = np.random.default_rng(42)
    user_ids = rng.choice(
        dataset.all_user_ids,
        size=min(sample_size, len(dataset.all_user_ids)),
        replace=False,
    )

    # Get embeddings and labels
    embeddings = []
    labels = []
    archetype_to_idx = {a: i for i, a in enumerate(UserArchetype)}

    for user_id in user_ids:
        user = dataset.get_user(user_id)
        if user:
            emb = adapter.user_embedding_table[user_id]
            embeddings.append(emb)
            labels.append(archetype_to_idx[user.archetype])

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Compute silhouette
    silhouette = compute_silhouette_score(embeddings, labels)
    pass_test = silhouette >= threshold

    # Compute cluster sizes and distances
    cluster_sizes = {}
    for arch in UserArchetype:
        idx = archetype_to_idx[arch]
        cluster_sizes[arch.value] = int(np.sum(labels == idx))

    intra, inter = compute_cluster_distances(embeddings, labels)
    intra_distances = {UserArchetype(list(UserArchetype)[i]).value: v for i, v in intra.items()}
    inter_distances = {UserArchetype(list(UserArchetype)[i]).value: v for i, v in inter.items()}

    return silhouette, pass_test, cluster_sizes, intra_distances, inter_distances


def test_topic_clustering(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    sample_size: int = 1000,
    threshold: float = 0.2,
) -> Tuple[float, bool, Dict[str, int], Dict[str, float], Dict[str, float]]:
    """Test if posts of same topic cluster together.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        sample_size: Number of posts to sample
        threshold: Minimum silhouette score to pass

    Returns:
        (silhouette, pass_bool, cluster_sizes, intra_distances, inter_distances)
    """
    # Sample posts
    rng = np.random.default_rng(42)
    post_ids = rng.choice(
        dataset.all_post_ids,
        size=min(sample_size, len(dataset.all_post_ids)),
        replace=False,
    )

    # Get embeddings and labels
    embeddings = []
    labels = []
    topic_to_idx = {t: i for i, t in enumerate(ContentTopic)}

    for post_id in post_ids:
        post = dataset.get_post(post_id)
        if post:
            # Get topic embedding
            topic_idx = topic_to_idx[post.topic]
            emb = adapter.topic_projection[topic_idx]
            embeddings.append(emb)
            labels.append(topic_idx)

    embeddings = np.array(embeddings)
    labels = np.array(labels)

    # Compute silhouette
    silhouette = compute_silhouette_score(embeddings, labels)
    pass_test = silhouette >= threshold

    # Compute cluster sizes and distances
    cluster_sizes = {}
    for topic in ContentTopic:
        idx = topic_to_idx[topic]
        cluster_sizes[topic.value] = int(np.sum(labels == idx))

    intra, inter = compute_cluster_distances(embeddings, labels)
    intra_distances = {ContentTopic(list(ContentTopic)[i]).value: v for i, v in intra.items()}
    inter_distances = {ContentTopic(list(ContentTopic)[i]).value: v for i, v in inter.items()}

    return silhouette, pass_test, cluster_sizes, intra_distances, inter_distances


def run_embedding_probes(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    user_sample_size: int = 500,
    post_sample_size: int = 1000,
    silhouette_threshold: float = 0.2,
) -> EmbeddingProbeResults:
    """Run all embedding probe tests.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        user_sample_size: Number of users to sample
        post_sample_size: Number of posts to sample
        silhouette_threshold: Minimum silhouette score to pass

    Returns:
        EmbeddingProbeResults with all metrics
    """
    # User clustering
    (
        user_silhouette,
        user_pass,
        arch_sizes,
        arch_intra,
        arch_inter,
    ) = test_user_archetype_clustering(
        adapter, dataset, user_sample_size, silhouette_threshold
    )

    # Topic clustering
    (
        topic_silhouette,
        topic_pass,
        topic_sizes,
        topic_intra,
        topic_inter,
    ) = test_topic_clustering(
        adapter, dataset, post_sample_size, silhouette_threshold
    )

    return EmbeddingProbeResults(
        user_silhouette=user_silhouette,
        topic_silhouette=topic_silhouette,
        archetype_cluster_sizes=arch_sizes,
        archetype_intra_distances=arch_intra,
        archetype_inter_distances=arch_inter,
        topic_cluster_sizes=topic_sizes,
        topic_intra_distances=topic_intra,
        topic_inter_distances=topic_inter,
        user_clustering_pass=user_pass,
        topic_clustering_pass=topic_pass,
    )
