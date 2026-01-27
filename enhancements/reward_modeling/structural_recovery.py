"""Structural recovery verification for pluralistic reward models.

Measures how well learned value systems recover the ground truth archetypes
from F2's synthetic data. This validates that pluralistic learning discovers
meaningful latent structure rather than arbitrary groupings.

Key metrics:
- Weight correlation: How similar are learned weights to ground truth?
- Assignment accuracy: Are users assigned to their true archetype?
- System interpretability: Do systems have expected positive/negative actions?
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from enhancements.data import ContentTopic, UserArchetype, get_engagement_probs
from enhancements.reward_modeling.pluralistic import (
    PluralState,
    compute_mixture_weights,
    get_dominant_system,
)
from enhancements.reward_modeling.weights import NUM_ACTIONS


@dataclass
class RecoveryMetrics:
    """Metrics for structural recovery evaluation."""

    # Weight-based metrics
    weight_correlations: Dict[int, Tuple[str, float]]  # system_k -> (best_archetype, correlation)
    mean_correlation: float
    correlation_matrix: np.ndarray  # [K, num_archetypes]

    # Assignment-based metrics
    assignment_accuracy: float
    per_archetype_accuracy: Dict[str, float]
    confusion_matrix: np.ndarray  # [num_archetypes, K]

    # Interpretability metrics
    interpretability_score: float
    per_system_interpretability: List[float]

    # System diversity
    system_diversity: float  # Mean pairwise distance between systems


def compute_ground_truth_weights(archetype: UserArchetype) -> np.ndarray:
    """Derive ground truth reward weights from F2's engagement rules.

    Averages action probabilities across all topics for this archetype,
    then converts to reward-like weights (positive actions positive,
    negative actions negative).

    Args:
        archetype: User archetype

    Returns:
        [num_actions] ground truth weights
    """
    weights = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # Average over all topics for this archetype
    for topic in ContentTopic:
        probs = get_engagement_probs(archetype, topic)
        action_array = np.array(probs.to_array(), dtype=np.float32)
        weights = weights + action_array

    weights = weights / len(ContentTopic)

    # Convert probabilities to reward-like weights
    # Positive actions (0-13): keep as positive
    # Negative actions (14-17): negate to make them negative rewards
    weights[14:] = -weights[14:] * 2  # Amplify negative signal

    return weights


def get_all_ground_truth_weights() -> Dict[UserArchetype, np.ndarray]:
    """Get ground truth weights for all archetypes.

    Returns:
        Dictionary mapping archetype -> [num_actions] weights
    """
    return {
        arch: compute_ground_truth_weights(arch)
        for arch in UserArchetype
    }


def compute_correlation_matrix(
    learned_weights: jnp.ndarray,
    gt_weights: Dict[UserArchetype, np.ndarray],
) -> np.ndarray:
    """Compute correlation between learned systems and ground truth archetypes.

    Args:
        learned_weights: [K, num_actions] learned value system weights
        gt_weights: Dictionary mapping archetype -> weights

    Returns:
        [K, num_archetypes] correlation matrix
    """
    K = learned_weights.shape[0]
    archetypes = list(UserArchetype)
    num_archetypes = len(archetypes)

    correlation_matrix = np.zeros((K, num_archetypes), dtype=np.float32)

    for k in range(K):
        learned_k = np.array(learned_weights[k])
        for i, arch in enumerate(archetypes):
            gt = gt_weights[arch]
            # Pearson correlation
            corr = np.corrcoef(learned_k, gt)[0, 1]
            correlation_matrix[k, i] = corr if not np.isnan(corr) else 0.0

    return correlation_matrix


def match_systems_to_archetypes(
    correlation_matrix: np.ndarray,
) -> Dict[int, Tuple[str, float]]:
    """Match learned systems to ground truth archetypes using Hungarian algorithm.

    Args:
        correlation_matrix: [K, num_archetypes] correlations

    Returns:
        Dictionary mapping system_k -> (archetype_name, correlation)
    """
    K, num_archetypes = correlation_matrix.shape
    archetypes = list(UserArchetype)

    # Hungarian algorithm minimizes cost, we want to maximize correlation
    cost_matrix = -correlation_matrix

    # Handle case where K != num_archetypes
    if K <= num_archetypes:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        # More systems than archetypes - some systems won't be matched
        row_ind, col_ind = linear_sum_assignment(cost_matrix[:num_archetypes, :])

    matches = {}
    for k in range(K):
        if k in row_ind:
            idx = np.where(row_ind == k)[0][0]
            arch_idx = col_ind[idx]
            matches[k] = (archetypes[arch_idx].value, correlation_matrix[k, arch_idx])
        else:
            # System not matched - find best archetype anyway
            best_idx = np.argmax(correlation_matrix[k])
            matches[k] = (archetypes[best_idx].value, correlation_matrix[k, best_idx])

    return matches


def compute_assignment_accuracy(
    state: PluralState,
    user_embeddings: jnp.ndarray,
    true_archetype_ids: jnp.ndarray,
    system_to_archetype: Dict[int, Tuple[str, float]],
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Compute how accurately users are assigned to their true archetype's system.

    Args:
        state: Trained pluralistic model state
        user_embeddings: [N, D] user embeddings
        true_archetype_ids: [N] true archetype indices
        system_to_archetype: Mapping from system to archetype

    Returns:
        (overall_accuracy, per_archetype_accuracy, confusion_matrix)
    """
    archetypes = list(UserArchetype)
    num_archetypes = len(archetypes)
    K = state.weights.shape[0]

    # Get predicted dominant system for each user
    predicted_systems = np.array(get_dominant_system(state, user_embeddings))
    true_ids = np.array(true_archetype_ids)

    # Build reverse mapping: archetype -> best system
    archetype_to_system = {}
    for k, (arch_name, corr) in system_to_archetype.items():
        if arch_name not in archetype_to_system or corr > archetype_to_system[arch_name][1]:
            archetype_to_system[arch_name] = (k, corr)

    # Compute accuracy
    correct = 0
    total = 0
    per_arch_correct = {arch.value: 0 for arch in archetypes}
    per_arch_total = {arch.value: 0 for arch in archetypes}
    confusion = np.zeros((num_archetypes, K), dtype=np.int32)

    for i in range(len(true_ids)):
        true_arch = archetypes[true_ids[i]]
        pred_system = predicted_systems[i]

        per_arch_total[true_arch.value] += 1
        confusion[true_ids[i], pred_system] += 1

        # Check if predicted system maps to true archetype
        if true_arch.value in archetype_to_system:
            expected_system = archetype_to_system[true_arch.value][0]
            if pred_system == expected_system:
                correct += 1
                per_arch_correct[true_arch.value] += 1

        total += 1

    overall_acc = correct / total if total > 0 else 0.0
    per_arch_acc = {
        arch: (per_arch_correct[arch] / per_arch_total[arch] if per_arch_total[arch] > 0 else 0.0)
        for arch in per_arch_correct
    }

    return overall_acc, per_arch_acc, confusion


def compute_interpretability_score(
    weights: jnp.ndarray,
    positive_indices: List[int] = list(range(14)),
    negative_indices: List[int] = list(range(14, 18)),
) -> Tuple[float, List[float]]:
    """Compute how interpretable the learned value systems are.

    A system is interpretable if:
    - Positive actions (favorite, repost, etc.) have positive weights
    - Negative actions (block, mute, etc.) have negative weights

    Args:
        weights: [K, num_actions] learned weights
        positive_indices: Indices of positive actions
        negative_indices: Indices of negative actions

    Returns:
        (overall_score, per_system_scores)
    """
    K = weights.shape[0]
    weights = np.array(weights)

    per_system = []
    for k in range(K):
        w = weights[k]

        # Count correct signs
        pos_correct = np.sum(w[positive_indices] > 0)
        neg_correct = np.sum(w[negative_indices] < 0)
        total = len(positive_indices) + len(negative_indices)

        score = (pos_correct + neg_correct) / total
        per_system.append(float(score))

    overall = np.mean(per_system)
    return float(overall), per_system


def compute_system_diversity(weights: jnp.ndarray) -> float:
    """Compute diversity of learned systems (mean pairwise distance).

    Args:
        weights: [K, num_actions] learned weights

    Returns:
        Mean pairwise cosine distance (1 - similarity)
    """
    weights = np.array(weights)
    K = weights.shape[0]

    if K <= 1:
        return 0.0

    # Normalize
    norms = np.linalg.norm(weights, axis=1, keepdims=True) + 1e-8
    normed = weights / norms

    # Pairwise cosine similarity
    similarity = normed @ normed.T

    # Extract upper triangle (excluding diagonal)
    upper_tri = similarity[np.triu_indices(K, k=1)]

    # Distance = 1 - similarity
    distances = 1 - upper_tri
    return float(np.mean(distances))


def measure_structural_recovery(
    state: PluralState,
    user_embeddings: jnp.ndarray,
    true_archetype_ids: jnp.ndarray,
    verbose: bool = True,
) -> RecoveryMetrics:
    """Comprehensive structural recovery evaluation.

    Args:
        state: Trained pluralistic model state
        user_embeddings: [N, D] user embeddings
        true_archetype_ids: [N] true archetype indices (0 to K-1)
        verbose: Print detailed results

    Returns:
        RecoveryMetrics with all evaluation results
    """
    # Get ground truth
    gt_weights = get_all_ground_truth_weights()

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(state.weights, gt_weights)

    # Match systems to archetypes
    matches = match_systems_to_archetypes(corr_matrix)
    mean_corr = np.mean([corr for _, corr in matches.values()])

    # Compute assignment accuracy
    overall_acc, per_arch_acc, confusion = compute_assignment_accuracy(
        state, user_embeddings, true_archetype_ids, matches
    )

    # Compute interpretability
    overall_interp, per_system_interp = compute_interpretability_score(state.weights)

    # Compute diversity
    diversity = compute_system_diversity(state.weights)

    if verbose:
        print("\n" + "=" * 60)
        print("STRUCTURAL RECOVERY RESULTS")
        print("=" * 60)

        print("\n--- System-to-Archetype Matching ---")
        for k, (arch, corr) in sorted(matches.items()):
            print(f"  System {k} -> {arch:15s} (correlation: {corr:.3f})")

        print(f"\n--- Weight Correlation ---")
        print(f"  Mean correlation: {mean_corr:.3f}")
        print(f"  Gate threshold:   0.80")
        print(f"  Status:           {'PASS' if mean_corr > 0.8 else 'FAIL'}")

        print(f"\n--- Assignment Accuracy ---")
        print(f"  Overall: {overall_acc:.1%}")
        for arch, acc in per_arch_acc.items():
            print(f"    {arch:15s}: {acc:.1%}")
        print(f"  Gate threshold: 70%")
        print(f"  Status:         {'PASS' if overall_acc > 0.7 else 'FAIL'}")

        print(f"\n--- Interpretability ---")
        print(f"  Overall: {overall_interp:.1%}")
        for k, score in enumerate(per_system_interp):
            print(f"    System {k}: {score:.1%}")

        print(f"\n--- System Diversity ---")
        print(f"  Mean pairwise distance: {diversity:.3f}")
        print(f"  (0 = identical, 1 = orthogonal)")

        print("\n" + "=" * 60)

    return RecoveryMetrics(
        weight_correlations=matches,
        mean_correlation=mean_corr,
        correlation_matrix=corr_matrix,
        assignment_accuracy=overall_acc,
        per_archetype_accuracy=per_arch_acc,
        confusion_matrix=confusion,
        interpretability_score=overall_interp,
        per_system_interpretability=per_system_interp,
        system_diversity=diversity,
    )


def check_recovery_gates(metrics: RecoveryMetrics) -> Dict[str, bool]:
    """Check if structural recovery passes all gates.

    Args:
        metrics: Recovery metrics

    Returns:
        Dictionary of gate_name -> passed
    """
    return {
        'mean_correlation_gt_0.8': metrics.mean_correlation > 0.8,
        'assignment_accuracy_gt_0.7': metrics.assignment_accuracy > 0.7,
        'interpretability_gt_0.6': metrics.interpretability_score > 0.6,
        'diversity_gt_0.1': metrics.system_diversity > 0.1,
    }
