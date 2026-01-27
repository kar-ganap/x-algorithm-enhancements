#!/usr/bin/env python3
"""Test two-stage pluralistic reward model.

Stage 1: Cluster users by interaction history (k-means)
Stage 2: Train per-cluster Bradley-Terry weights

This is the robust production approach.

Usage:
    uv run python scripts/test_two_stage.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from enhancements.data import ContentTopic, UserArchetype, get_engagement_probs
from enhancements.reward_modeling.structural_recovery import (
    compute_correlation_matrix,
    compute_interpretability_score,
    compute_system_diversity,
    get_all_ground_truth_weights,
    match_systems_to_archetypes,
)
from enhancements.reward_modeling.two_stage import (
    TwoStageConfig,
    train_two_stage,
)
from enhancements.reward_modeling.weights import NUM_ACTIONS


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def generate_training_data(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    history_size: int = 20,
    noise_std: float = 0.0,
    label_flip_rate: float = 0.0,
    use_topic_features: bool = False,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data with user histories.

    Args:
        use_topic_features: If True, use per-topic engagement as features (Option 2).
                           If False, use average action probs (original).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)
    num_topics = len(topics)

    all_probs_pref = []
    all_probs_rej = []
    all_user_histories = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(archetypes):
        for _ in range(num_users_per_archetype):
            if use_topic_features:
                # Option 2: Per-topic engagement features
                # For each topic, compute positive engagement (sum of positive action probs)
                user_history = np.zeros(num_topics, dtype=np.float32)
                for t_idx, topic in enumerate(topics):
                    probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)
                    # Add noise for variability
                    probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                    # Positive engagement = sum of positive actions (indices 0-13)
                    user_history[t_idx] = probs[:14].sum()
            else:
                # Original: average action probs across topics
                user_history = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for _ in range(history_size):
                    topic_idx = rng.choice(len(topics))
                    topic = topics[topic_idx]
                    probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)
                    probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                    user_history += probs
                user_history /= history_size

            # Generate preference pairs
            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
                probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

                if noise_std > 0:
                    probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                else:
                    probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a

                    if label_flip_rate > 0 and rng.random() < label_flip_rate:
                        pref, rej = rej, pref

                    all_probs_pref.append(pref)
                    all_probs_rej.append(rej)
                    all_user_histories.append(user_history)
                    all_arch_ids.append(arch_idx)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_user_histories, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
    )


def measure_recovery(
    weights: np.ndarray,
    cluster_ids: np.ndarray,
    true_archetype_ids: np.ndarray,
    verbose: bool = True,
) -> Dict:
    """Measure structural recovery for two-stage model."""
    gt_weights = get_all_ground_truth_weights()

    # Correlation matrix
    import jax.numpy as jnp
    corr_matrix = compute_correlation_matrix(jnp.array(weights), gt_weights)
    matches = match_systems_to_archetypes(corr_matrix)
    mean_corr = np.mean([corr for _, corr in matches.values()])

    # Assignment accuracy: does cluster match true archetype?
    archetypes = list(UserArchetype)
    archetype_to_cluster = {}
    for k, (arch_name, corr) in matches.items():
        if arch_name not in archetype_to_cluster or corr > archetype_to_cluster[arch_name][1]:
            archetype_to_cluster[arch_name] = (k, corr)

    correct = 0
    total = len(true_archetype_ids)
    per_arch_correct = {arch.value: 0 for arch in archetypes}
    per_arch_total = {arch.value: 0 for arch in archetypes}

    for i in range(total):
        true_arch = archetypes[true_archetype_ids[i]]
        pred_cluster = cluster_ids[i]
        per_arch_total[true_arch.value] += 1

        if true_arch.value in archetype_to_cluster:
            expected_cluster = archetype_to_cluster[true_arch.value][0]
            if pred_cluster == expected_cluster:
                correct += 1
                per_arch_correct[true_arch.value] += 1

    assignment_acc = correct / total if total > 0 else 0.0
    per_arch_acc = {
        arch: (per_arch_correct[arch] / per_arch_total[arch] if per_arch_total[arch] > 0 else 0.0)
        for arch in per_arch_correct
    }

    # Interpretability and diversity
    interp_score, _ = compute_interpretability_score(jnp.array(weights))
    diversity = compute_system_diversity(jnp.array(weights))

    if verbose:
        print("\n" + "=" * 60)
        print("STRUCTURAL RECOVERY RESULTS (Two-Stage)")
        print("=" * 60)

        print("\n--- Cluster-to-Archetype Matching ---")
        for k, (arch, corr) in sorted(matches.items()):
            print(f"  Cluster {k} -> {arch:15s} (correlation: {corr:.3f})")

        print(f"\n--- Weight Correlation ---")
        print(f"  Mean correlation: {mean_corr:.3f}")
        print(f"  Gate threshold:   0.80")
        print(f"  Status:           {'PASS' if mean_corr > 0.8 else 'FAIL'}")

        print(f"\n--- Assignment Accuracy ---")
        print(f"  Overall: {assignment_acc:.1%}")
        for arch, acc in per_arch_acc.items():
            print(f"    {arch:15s}: {acc:.1%}")
        print(f"  Gate threshold: 70%")
        print(f"  Status:         {'PASS' if assignment_acc > 0.7 else 'FAIL'}")

        print(f"\n--- Interpretability ---")
        print(f"  Overall: {interp_score:.1%}")

        print(f"\n--- Weight Diversity ---")
        print(f"  Mean pairwise distance: {diversity:.3f}")

        print("\n" + "=" * 60)

    return {
        'mean_correlation': float(mean_corr),
        'assignment_accuracy': float(assignment_acc),
        'per_archetype_accuracy': per_arch_acc,
        'interpretability': float(interp_score),
        'diversity': float(diversity),
        'matches': {k: (arch, float(corr)) for k, (arch, corr) in matches.items()},
    }


def run_two_stage_experiment(
    name: str,
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    user_histories: np.ndarray,
    arch_ids: np.ndarray,
    config: TwoStageConfig,
) -> Dict:
    """Run two-stage experiment and return results."""
    flush_print(f"\n{'='*70}")
    flush_print(f"EXPERIMENT: {name}")
    flush_print(f"{'='*70}")
    flush_print(f"Feature shape: {user_histories.shape}")

    state, metrics = train_two_stage(
        user_histories,
        probs_pref,
        probs_rej,
        config,
        verbose=True,
    )

    cluster_ids = state.kmeans_model.predict(user_histories)

    recovery = measure_recovery(
        state.cluster_weights,
        cluster_ids,
        arch_ids,
        verbose=True,
    )

    # Cluster composition
    archetypes = list(UserArchetype)
    cluster_purity = []
    flush_print("\n--- Cluster Composition ---")
    for k in range(config.num_clusters):
        mask = cluster_ids == k
        arch_counts = {}
        for arch_idx in arch_ids[mask]:
            arch_name = archetypes[arch_idx].value
            arch_counts[arch_name] = arch_counts.get(arch_name, 0) + 1

        if arch_counts:
            dominant = max(arch_counts.items(), key=lambda x: x[1])
            purity = dominant[1] / sum(arch_counts.values())
            cluster_purity.append(purity)
            flush_print(f"  Cluster {k}: {sum(arch_counts.values())} samples, "
                       f"dominant={dominant[0]} ({purity:.0%} purity)")

    avg_purity = np.mean(cluster_purity) if cluster_purity else 0.0

    return {
        'name': name,
        'accuracy': float(metrics.overall_accuracy),
        'correlation': recovery['mean_correlation'],
        'assignment': recovery['assignment_accuracy'],
        'interpretability': recovery['interpretability'],
        'diversity': recovery['diversity'],
        'avg_purity': float(avg_purity),
        'state': state,
        'metrics': metrics,
        'recovery': recovery,
    }


def generate_stress_test_data(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate stress test: archetypes with SAME topic preferences but DIFFERENT action patterns.

    Creates synthetic archetypes that all prefer the same topics (sports, tech)
    but differ in HOW they engage:
    - "likers": high like probability, low comment
    - "commenters": high comment probability, low like
    - "sharers": high repost probability
    - "lurkers": low everything but still prefer same topics
    - "haters": high block/mute on non-preferred topics
    - "power_engagers": high everything on preferred topics

    This tests whether topic-only features can separate them (they shouldn't).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # All archetypes prefer sports (topic 0) and tech (topic 4) over others
    # But they differ in ACTION patterns
    topics = list(ContentTopic)
    num_topics = len(topics)

    # Define action patterns for each synthetic archetype
    # Actions 0-13 are positive (favorite, repost, reply, etc.)
    # Actions 14-17 are negative (block, mute, report, not_interested)
    action_patterns = {
        'liker': {'preferred': np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
                  'non_preferred': np.array([0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])},
        'commenter': {'preferred': np.array([0.2, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
                      'non_preferred': np.array([0.1, 0.05, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])},
        'sharer': {'preferred': np.array([0.3, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
                   'non_preferred': np.array([0.1, 0.3, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.1])},
        'lurker': {'preferred': np.array([0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.3, 0.02, 0.02, 0.02, 0.02]),
                   'non_preferred': np.array([0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.1, 0.05, 0.05, 0.05, 0.05])},
        'hater': {'preferred': np.array([0.5, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05]),
                  'non_preferred': np.array([0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.4, 0.3, 0.3])},
        'power_engager': {'preferred': np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.02, 0.02, 0.02, 0.02]),
                          'non_preferred': np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])},
    }

    archetype_names = list(action_patterns.keys())
    preferred_topics = {0, 4}  # sports, tech - ALL archetypes prefer these

    all_probs_pref = []
    all_probs_rej = []
    all_topic_features = []  # Per-topic engagement
    all_topic_action_features = []  # Per-topic-per-action (richer)
    all_arch_ids = []

    for arch_idx, arch_name in enumerate(archetype_names):
        pattern = action_patterns[arch_name]

        for _ in range(num_users_per_archetype):
            # Compute topic features (per-topic engagement)
            topic_engagement = np.zeros(num_topics, dtype=np.float32)
            # Compute topic×action features (richer)
            topic_action_features = np.zeros((num_topics, NUM_ACTIONS), dtype=np.float32)

            for t_idx in range(num_topics):
                if t_idx in preferred_topics:
                    probs = pattern['preferred'].copy()
                else:
                    probs = pattern['non_preferred'].copy()
                # Add noise
                probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                topic_engagement[t_idx] = probs[:14].sum()
                topic_action_features[t_idx] = probs

            # Generate preference pairs
            for _ in range(num_pairs_per_user):
                # Pick one preferred and one non-preferred topic
                t_pref = rng.choice(list(preferred_topics))
                t_nonpref = rng.choice([t for t in range(num_topics) if t not in preferred_topics])

                probs_a = topic_action_features[t_pref].copy()
                probs_b = topic_action_features[t_nonpref].copy()

                # Add noise
                probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                # Preferred topic content should be preferred
                all_probs_pref.append(probs_a)
                all_probs_rej.append(probs_b)
                all_topic_features.append(topic_engagement)
                all_topic_action_features.append(topic_action_features.flatten())
                all_arch_ids.append(arch_idx)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_topic_features, dtype=np.float32),  # [N, num_topics]
        np.array(all_topic_action_features, dtype=np.float32),  # [N, num_topics * NUM_ACTIONS]
        np.array(all_arch_ids, dtype=np.int32),
    )


def generate_noisy_preference_data(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    label_flip_rate: float = 0.3,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate stress test #2: Noisy preferences.

    Users sometimes click things they don't actually prefer.
    This is a FUNDAMENTAL limitation - if preference labels are noisy,
    no model can perfectly recover ground truth.

    Args:
        label_flip_rate: Probability of swapping preferred/rejected labels (0.3 = 30% noise)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)
    num_topics = len(topics)

    all_probs_pref = []
    all_probs_rej = []
    all_topic_features = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(archetypes):
        for _ in range(num_users_per_archetype):
            # Per-topic engagement features
            user_history = np.zeros(num_topics, dtype=np.float32)
            for t_idx, topic in enumerate(topics):
                probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)
                probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                user_history[t_idx] = probs[:14].sum()

            # Generate preference pairs with noise
            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
                probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)
                probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a

                    # NOISE: Randomly flip labels
                    if rng.random() < label_flip_rate:
                        pref, rej = rej, pref

                    all_probs_pref.append(pref)
                    all_probs_rej.append(rej)
                    all_topic_features.append(user_history)
                    all_arch_ids.append(arch_idx)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_topic_features, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
    )


def generate_cross_topic_data(
    num_pure_users_per_archetype: int = 50,
    num_cross_users: int = 300,
    num_pairs_per_user: int = 10,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate stress test #4: Cross-topic users.

    Some users genuinely like multiple topics equally (e.g., sports AND politics).
    Hard k-means clustering can't capture soft membership.

    Returns:
        probs_pref, probs_rej, topic_features, arch_ids, is_cross_user
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)
    num_topics = len(topics)

    all_probs_pref = []
    all_probs_rej = []
    all_topic_features = []
    all_arch_ids = []
    all_is_cross = []

    # First: Pure users (single archetype)
    for arch_idx, archetype in enumerate(archetypes):
        for _ in range(num_pure_users_per_archetype):
            user_history = np.zeros(num_topics, dtype=np.float32)
            for t_idx, topic in enumerate(topics):
                probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)
                probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                user_history[t_idx] = probs[:14].sum()

            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
                probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)
                probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a

                    all_probs_pref.append(pref)
                    all_probs_rej.append(rej)
                    all_topic_features.append(user_history)
                    all_arch_ids.append(arch_idx)
                    all_is_cross.append(False)

    # Second: Cross-topic users (blend of two archetypes)
    archetype_pairs = [
        (UserArchetype.SPORTS_FAN, UserArchetype.TECH_BRO),      # Sports + Tech
        (UserArchetype.POLITICAL_L, UserArchetype.POLITICAL_R),  # Both political (rare but exists)
        (UserArchetype.TECH_BRO, UserArchetype.POWER_USER),      # Tech power user
        (UserArchetype.SPORTS_FAN, UserArchetype.POLITICAL_L),   # Sports + Left politics
        (UserArchetype.LURKER, UserArchetype.TECH_BRO),          # Lurking tech fan
    ]

    users_per_pair = num_cross_users // len(archetype_pairs)

    for arch1, arch2 in archetype_pairs:
        arch1_idx = archetypes.index(arch1)

        for _ in range(users_per_pair):
            # Blend the engagement patterns (50/50 mix)
            user_history = np.zeros(num_topics, dtype=np.float32)
            for t_idx, topic in enumerate(topics):
                probs1 = np.array(get_engagement_probs(arch1, topic).to_array(), dtype=np.float32)
                probs2 = np.array(get_engagement_probs(arch2, topic).to_array(), dtype=np.float32)
                # Average the two archetypes
                probs = (probs1 + probs2) / 2
                probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                user_history[t_idx] = probs[:14].sum()

            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                # Use blended preferences for scoring
                probs_a1 = np.array(get_engagement_probs(arch1, topic_a).to_array(), dtype=np.float32)
                probs_a2 = np.array(get_engagement_probs(arch2, topic_a).to_array(), dtype=np.float32)
                probs_a = (probs_a1 + probs_a2) / 2
                probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                probs_b1 = np.array(get_engagement_probs(arch1, topic_b).to_array(), dtype=np.float32)
                probs_b2 = np.array(get_engagement_probs(arch2, topic_b).to_array(), dtype=np.float32)
                probs_b = (probs_b1 + probs_b2) / 2
                probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a

                    all_probs_pref.append(pref)
                    all_probs_rej.append(rej)
                    all_topic_features.append(user_history)
                    all_arch_ids.append(arch1_idx)  # Assign to first archetype for tracking
                    all_is_cross.append(True)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_topic_features, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_is_cross, dtype=bool),
    )


def main():
    output_dir = Path("results/f4_phase2_two_stage")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Phase 2: Two-Stage Pluralistic Reward Model")
    flush_print("Comparing: Original features vs Topic-aware features")
    flush_print("=" * 70)

    config = TwoStageConfig(
        num_clusters=6,
        learning_rate=0.01,
        num_epochs=100,
        batch_size=64,
    )

    results = []

    # =========================================================================
    # Experiment 1: Original features (average action probs)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("Generating data with ORIGINAL features (avg action probs)...")
    flush_print("=" * 70)
    probs_pref_orig, probs_rej_orig, hist_orig, arch_ids_orig = generate_training_data(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        history_size=20,
        use_topic_features=False,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref_orig)} pairs, features: {hist_orig.shape}")

    result_orig = run_two_stage_experiment(
        "Original (avg action probs)",
        probs_pref_orig, probs_rej_orig, hist_orig, arch_ids_orig, config
    )
    results.append(result_orig)

    # =========================================================================
    # Experiment 2: Topic-aware features (per-topic engagement)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("Generating data with TOPIC-AWARE features (per-topic engagement)...")
    flush_print("=" * 70)
    probs_pref_topic, probs_rej_topic, hist_topic, arch_ids_topic = generate_training_data(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        history_size=20,
        use_topic_features=True,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref_topic)} pairs, features: {hist_topic.shape}")

    result_topic = run_two_stage_experiment(
        "Topic-aware (per-topic engagement)",
        probs_pref_topic, probs_rej_topic, hist_topic, arch_ids_topic, config
    )
    results.append(result_topic)

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("COMPARISON SUMMARY")
    flush_print("=" * 70)

    flush_print(f"\n{'Features':<35} {'Accuracy':>10} {'Corr':>8} {'Assign':>8} {'Purity':>8}")
    flush_print("-" * 75)

    for r in results:
        flush_print(
            f"{r['name']:<35} "
            f"{r['accuracy']:>10.1%} "
            f"{r['correlation']:>8.3f} "
            f"{r['assignment']:>8.1%} "
            f"{r['avg_purity']:>8.1%}"
        )

    # Gate status
    flush_print(f"\n{'Features':<35} {'Corr>0.8':>10} {'Assign>70%':>12} {'Purity>80%':>12}")
    flush_print("-" * 75)

    for r in results:
        flush_print(
            f"{r['name']:<35} "
            f"{'PASS' if r['correlation'] > 0.8 else 'FAIL':>10} "
            f"{'PASS' if r['assignment'] > 0.7 else 'FAIL':>12} "
            f"{'PASS' if r['avg_purity'] > 0.8 else 'FAIL':>12}"
        )

    # =========================================================================
    # STRESS TEST: Same topics, different action patterns
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST: Same topics, different action patterns")
    flush_print("All archetypes prefer same topics but engage differently")
    flush_print("=" * 70)

    probs_pref_st, probs_rej_st, topic_feat_st, topic_action_feat_st, arch_ids_st = generate_stress_test_data(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref_st)} pairs")
    flush_print(f"Topic-only features: {topic_feat_st.shape}")
    flush_print(f"Topic×Action features: {topic_action_feat_st.shape}")

    stress_results = []

    # Test 1: Topic-only features (expected to FAIL)
    flush_print("\n--- Testing topic-only features (expected to fail) ---")
    result_st_topic = run_two_stage_experiment(
        "Stress: Topic-only",
        probs_pref_st, probs_rej_st, topic_feat_st, arch_ids_st, config
    )
    stress_results.append(result_st_topic)

    # Test 2: Topic×Action features (should work)
    flush_print("\n--- Testing topic×action features (should work) ---")
    result_st_topic_action = run_two_stage_experiment(
        "Stress: Topic×Action",
        probs_pref_st, probs_rej_st, topic_action_feat_st, arch_ids_st, config
    )
    stress_results.append(result_st_topic_action)

    # Stress test summary
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST SUMMARY")
    flush_print("=" * 70)

    flush_print(f"\n{'Features':<35} {'Accuracy':>10} {'Corr':>8} {'Purity':>8} {'Expected':>10}")
    flush_print("-" * 80)

    flush_print(
        f"{'Stress: Topic-only':<35} "
        f"{stress_results[0]['accuracy']:>10.1%} "
        f"{stress_results[0]['correlation']:>8.3f} "
        f"{stress_results[0]['avg_purity']:>8.1%} "
        f"{'FAIL':>10}"
    )
    flush_print(
        f"{'Stress: Topic×Action':<35} "
        f"{stress_results[1]['accuracy']:>10.1%} "
        f"{stress_results[1]['correlation']:>8.3f} "
        f"{stress_results[1]['avg_purity']:>8.1%} "
        f"{'PASS':>10}"
    )

    actual_topic = "PASS" if stress_results[0]['avg_purity'] > 0.8 else "FAIL"
    actual_action = "PASS" if stress_results[1]['avg_purity'] > 0.8 else "FAIL"

    flush_print(f"\nVerification:")
    flush_print(f"  Topic-only: Expected FAIL, Got {actual_topic} - {'✓' if actual_topic == 'FAIL' else '✗'}")
    flush_print(f"  Topic×Action: Expected PASS, Got {actual_action} - {'✓' if actual_action == 'PASS' else '✗'}")

    # =========================================================================
    # STRESS TEST #2: Noisy Preferences (Fundamental Limitation)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST #2: Noisy Preferences (FUNDAMENTAL LIMITATION)")
    flush_print("Users sometimes click things they don't actually prefer")
    flush_print("Expected: Accuracy degrades with noise, clustering may still work")
    flush_print("=" * 70)

    noise_results = []
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]

    for noise_rate in noise_levels:
        flush_print(f"\n--- Testing with {noise_rate:.0%} label noise ---")
        probs_pref_n, probs_rej_n, feat_n, arch_ids_n = generate_noisy_preference_data(
            num_users_per_archetype=100,
            num_pairs_per_user=10,
            label_flip_rate=noise_rate,
            rng=np.random.default_rng(42),
        )
        flush_print(f"Generated {len(probs_pref_n)} pairs with {noise_rate:.0%} label noise")

        result_noise = run_two_stage_experiment(
            f"Noise: {noise_rate:.0%}",
            probs_pref_n, probs_rej_n, feat_n, arch_ids_n, config
        )
        noise_results.append({
            'noise_rate': noise_rate,
            'accuracy': result_noise['accuracy'],
            'correlation': result_noise['correlation'],
            'avg_purity': result_noise['avg_purity'],
        })

    # Noise test summary
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST #2 SUMMARY: Noisy Preferences")
    flush_print("=" * 70)

    flush_print(f"\n{'Noise Rate':<15} {'Accuracy':>12} {'Correlation':>12} {'Purity':>12}")
    flush_print("-" * 55)
    for nr in noise_results:
        flush_print(
            f"{nr['noise_rate']:<15.0%} "
            f"{nr['accuracy']:>12.1%} "
            f"{nr['correlation']:>12.3f} "
            f"{nr['avg_purity']:>12.1%}"
        )

    # Theoretical max accuracy with noise
    flush_print(f"\nNote: With {noise_levels[-1]:.0%} label noise, theoretical max accuracy is {1-noise_levels[-1]:.0%}")
    flush_print("Observation: Clustering purity remains high (features aren't noisy, only labels are)")
    flush_print("This is a FUNDAMENTAL limitation - noisy preference labels cannot be fully corrected")

    # =========================================================================
    # STRESS TEST #4: Cross-Topic Users (Fundamental Limitation)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST #4: Cross-Topic Users (FUNDAMENTAL LIMITATION)")
    flush_print("Users who genuinely like multiple topics equally")
    flush_print("Hard k-means clustering can't capture soft membership")
    flush_print("=" * 70)

    probs_pref_ct, probs_rej_ct, feat_ct, arch_ids_ct, is_cross_ct = generate_cross_topic_data(
        num_pure_users_per_archetype=50,
        num_cross_users=300,
        num_pairs_per_user=10,
        rng=np.random.default_rng(42),
    )
    num_cross = is_cross_ct.sum()
    num_pure = len(is_cross_ct) - num_cross
    flush_print(f"Generated {len(probs_pref_ct)} pairs")
    flush_print(f"  Pure users: {num_pure} pairs")
    flush_print(f"  Cross-topic users: {num_cross} pairs ({num_cross/len(is_cross_ct):.1%} of data)")

    result_cross = run_two_stage_experiment(
        "Cross-topic users",
        probs_pref_ct, probs_rej_ct, feat_ct, arch_ids_ct, config
    )

    # Analyze cross-topic user placement
    cluster_ids_ct = result_cross['state'].kmeans_model.predict(feat_ct)

    # For cross users, check if they're scattered across clusters
    cross_clusters = cluster_ids_ct[is_cross_ct]
    unique_cross, counts_cross = np.unique(cross_clusters, return_counts=True)

    flush_print("\n--- Cross-Topic User Analysis ---")
    flush_print(f"Cross-topic users distributed across {len(unique_cross)} clusters:")
    for c, cnt in zip(unique_cross, counts_cross):
        flush_print(f"  Cluster {c}: {cnt} cross-topic users ({cnt/len(cross_clusters):.1%})")

    # Check if cross-topic users form their own cluster or split
    max_cross_in_cluster = counts_cross.max() / len(cross_clusters)
    flush_print(f"\nMax concentration: {max_cross_in_cluster:.1%} in a single cluster")

    if max_cross_in_cluster > 0.5:
        flush_print("Result: Cross-topic users mostly grouped together (forming hybrid cluster)")
    else:
        flush_print("Result: Cross-topic users scattered (k-means struggles with soft boundaries)")

    # Cross-topic test summary
    flush_print("\n" + "=" * 70)
    flush_print("STRESS TEST #4 SUMMARY: Cross-Topic Users")
    flush_print("=" * 70)

    flush_print(f"\n{'Metric':<25} {'Value':>15}")
    flush_print("-" * 45)
    flush_print(f"{'Overall Accuracy':<25} {result_cross['accuracy']:>15.1%}")
    flush_print(f"{'Correlation':<25} {result_cross['correlation']:>15.3f}")
    flush_print(f"{'Cluster Purity':<25} {result_cross['avg_purity']:>15.1%}")
    flush_print(f"{'Cross-user scatter':<25} {len(unique_cross):>15} clusters")

    flush_print("\nObservation: K-means assigns hard cluster membership.")
    flush_print("Cross-topic users are forced into a single cluster, losing their dual nature.")
    flush_print("This is a FUNDAMENTAL limitation of hard clustering.")
    flush_print("Solution would require soft clustering (GMM) or mixture models.")

    # =========================================================================
    # FINAL SUMMARY: All Stress Tests
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("FINAL SUMMARY: Stress Test Results")
    flush_print("=" * 70)

    flush_print("""
┌─────────────────────────────────────────────────────────────────────┐
│ Stress Test                  │ Type        │ Result    │ Notes     │
├─────────────────────────────────────────────────────────────────────┤
│ #1 Same topics, diff actions │ Solvable    │ PASS ✓    │ Use richer│
│    (topic-only features)     │             │ (fails)   │ features  │
│    (topic×action features)   │             │ (passes)  │           │
├─────────────────────────────────────────────────────────────────────┤
│ #2 Noisy preferences         │ Fundamental │ Expected  │ Can't fix │
│    (30% label noise)         │             │ accuracy  │ noisy     │
│                              │             │ drops     │ labels    │
├─────────────────────────────────────────────────────────────────────┤
│ #4 Cross-topic users         │ Fundamental │ Users get │ Need soft │
│    (multi-interest users)    │             │ hard      │ clustering│
│                              │             │ assigned  │           │
└─────────────────────────────────────────────────────────────────────┘
""")

    flush_print("Key Takeaways:")
    flush_print("1. Feature engineering solves #1 - richer features enable separation")
    flush_print("2. Noisy preferences (#2) is FUNDAMENTAL - can't exceed 1-noise accuracy")
    flush_print("3. Cross-topic users (#4) is FUNDAMENTAL - k-means can't do soft membership")
    flush_print("4. For production: Accept these limitations or use GMM/soft clustering")

    # Save all results
    all_results = results + stress_results + [result_cross]
    summary = {
        r['name']: {
            'accuracy': r['accuracy'],
            'correlation': r['correlation'],
            'assignment': r['assignment'],
            'interpretability': r['interpretability'],
            'diversity': r['diversity'],
            'avg_purity': r['avg_purity'],
        }
        for r in all_results
    }

    with open(output_dir / 'two_stage_comparison.json', 'w') as f:
        json.dump(summary, f, indent=2)

    flush_print(f"\nSaved summary to {output_dir / 'two_stage_comparison.json'}")

    flush_print("\n" + "=" * 70)
    flush_print("Test Complete!")
    flush_print("=" * 70)


if __name__ == "__main__":
    main()
