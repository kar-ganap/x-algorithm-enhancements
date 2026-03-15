#!/usr/bin/env python3
"""Test Fix 2: Learned embeddings from interaction history.

Instead of using random embeddings, we learn embeddings from user
interaction patterns. This encodes preference information directly
into embeddings.

Key difference from compare_pluralistic_approaches.py:
- User "embedding" = average action probabilities across their history
- Encoder learns to map history -> latent embedding
- Embeddings now encode what users actually click on

Usage:
    uv run python scripts/test_learned_embeddings.py
"""

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from enhancements.data import ContentTopic, UserArchetype, get_engagement_probs
from enhancements.reward_modeling.learned_embeddings import (
    LearnedEmbeddingConfig,
    LearnedEmbeddingState,
    get_dominant_system_from_history,
    train_with_learned_embeddings,
)
from enhancements.reward_modeling.structural_recovery import (
    compute_correlation_matrix,
    compute_interpretability_score,
    compute_system_diversity,
    get_all_ground_truth_weights,
    match_systems_to_archetypes,
)
from enhancements.reward_modeling.weights import NUM_ACTIONS


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def generate_training_data_with_history(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    history_size: int = 20,
    noise_std: float = 0.0,
    label_flip_rate: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data with user interaction histories.

    Key difference: Instead of random embeddings, we compute user "history"
    as the average action probabilities across their past interactions.
    This encodes what the user actually clicks on.

    Args:
        num_users_per_archetype: Users per archetype
        num_pairs_per_user: Preference pairs per user
        history_size: Number of past interactions to aggregate for user history
        noise_std: Noise to add to action probabilities
        label_flip_rate: Rate of label flips
        rng: Random number generator

    Returns:
        (probs_preferred, probs_rejected, user_histories, archetype_ids)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_user_histories = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(archetypes):
        for user_idx in range(num_users_per_archetype):
            # Generate user history: average action probs across random topics
            # This represents "what does this user typically click on?"
            user_history = np.zeros(NUM_ACTIONS, dtype=np.float32)
            for _ in range(history_size):
                topic_idx = rng.choice(len(topics))
                topic = topics[topic_idx]
                probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)
                # Add small noise to simulate variability
                probs = np.clip(probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1).astype(np.float32)
                user_history += probs
            user_history /= history_size

            # Generate preference pairs
            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
                probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

                # Add noise
                if noise_std > 0:
                    probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                else:
                    probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                # Determine preference (based on archetype's typical behavior)
                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a

                    # Label flip
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


def measure_recovery_learned(
    state: LearnedEmbeddingState,
    user_histories: jnp.ndarray,
    true_archetype_ids: jnp.ndarray,
    verbose: bool = True,
) -> dict:
    """Measure structural recovery for learned embedding model.

    Args:
        state: Trained model state
        user_histories: [N, num_actions] user interaction histories
        true_archetype_ids: [N] true archetype indices

    Returns:
        Dictionary with recovery metrics
    """
    # Get ground truth weights
    gt_weights = get_all_ground_truth_weights()

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(state.weights, gt_weights)

    # Match systems to archetypes
    matches = match_systems_to_archetypes(corr_matrix)
    mean_corr = np.mean([corr for _, corr in matches.values()])

    # Get dominant system per user
    predicted_systems = np.array(get_dominant_system_from_history(state, user_histories))
    true_ids = np.array(true_archetype_ids)

    # Build archetype -> best system mapping
    archetypes = list(UserArchetype)
    archetype_to_system = {}
    for k, (arch_name, corr) in matches.items():
        if arch_name not in archetype_to_system or corr > archetype_to_system[arch_name][1]:
            archetype_to_system[arch_name] = (k, corr)

    # Compute assignment accuracy
    correct = 0
    total = len(true_ids)
    per_arch_correct = {arch.value: 0 for arch in archetypes}
    per_arch_total = {arch.value: 0 for arch in archetypes}

    for i in range(len(true_ids)):
        true_arch = archetypes[true_ids[i]]
        pred_system = predicted_systems[i]
        per_arch_total[true_arch.value] += 1

        if true_arch.value in archetype_to_system:
            expected_system = archetype_to_system[true_arch.value][0]
            if pred_system == expected_system:
                correct += 1
                per_arch_correct[true_arch.value] += 1

    assignment_acc = correct / total if total > 0 else 0.0
    per_arch_acc = {
        arch: (per_arch_correct[arch] / per_arch_total[arch] if per_arch_total[arch] > 0 else 0.0)
        for arch in per_arch_correct
    }

    # Interpretability and diversity
    interp_score, per_system_interp = compute_interpretability_score(state.weights)
    diversity = compute_system_diversity(state.weights)

    if verbose:
        print("\n" + "=" * 60)
        print("STRUCTURAL RECOVERY RESULTS (Learned Embeddings)")
        print("=" * 60)

        print("\n--- System-to-Archetype Matching ---")
        for k, (arch, corr) in sorted(matches.items()):
            print(f"  System {k} -> {arch:15s} (correlation: {corr:.3f})")

        print("\n--- Weight Correlation ---")
        print(f"  Mean correlation: {mean_corr:.3f}")
        print("  Gate threshold:   0.80")
        print(f"  Status:           {'PASS' if mean_corr > 0.8 else 'FAIL'}")

        print("\n--- Assignment Accuracy ---")
        print(f"  Overall: {assignment_acc:.1%}")
        for arch, acc in per_arch_acc.items():
            print(f"    {arch:15s}: {acc:.1%}")
        print("  Gate threshold: 70%")
        print(f"  Status:         {'PASS' if assignment_acc > 0.7 else 'FAIL'}")

        print("\n--- Interpretability ---")
        print(f"  Overall: {interp_score:.1%}")

        print("\n--- System Diversity ---")
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


def main():
    output_dir = Path("results/f4_phase2_fix2")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Phase 2 Fix 2: Learned Embeddings from Interaction History")
    flush_print("=" * 70)

    # Generate training data with user histories
    flush_print("\nGenerating training data with user histories...")
    probs_pref, probs_rej, user_histories, arch_ids = generate_training_data_with_history(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        history_size=20,
        noise_std=0.0,
        label_flip_rate=0.0,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref)} preference pairs")
    flush_print(f"User history shape: {user_histories.shape}")
    flush_print(f"Archetypes: {len(set(arch_ids))}")

    # =========================================================================
    # Test 1: Unsupervised (no classification loss)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("TEST 1: UNSUPERVISED LEARNED EMBEDDINGS")
    flush_print("=" * 70)

    config_unsupervised = LearnedEmbeddingConfig(
        num_value_systems=6,
        embedding_dim=32,
        encoder_hidden_dim=64,
        mlp_hidden_dim=64,
        num_epochs=100,
        learning_rate=0.01,
        lambda_diversity=0.1,
        lambda_entropy=0.01,
        lambda_classification=0.0,
        batch_size=64,
    )

    flush_print("\nTraining unsupervised model...")
    state_unsup, metrics_unsup = train_with_learned_embeddings(
        config_unsupervised,
        jnp.array(user_histories),
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        archetype_ids=None,
        seed=42,
        verbose=True,
    )

    recovery_unsup = measure_recovery_learned(
        state_unsup,
        jnp.array(user_histories),
        jnp.array(arch_ids),
        verbose=True,
    )

    # =========================================================================
    # Test 2: Supervised (with classification loss)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("TEST 2: SUPERVISED LEARNED EMBEDDINGS")
    flush_print("=" * 70)

    config_supervised = LearnedEmbeddingConfig(
        num_value_systems=6,
        embedding_dim=32,
        encoder_hidden_dim=64,
        mlp_hidden_dim=64,
        num_epochs=100,
        learning_rate=0.01,
        lambda_diversity=0.1,
        lambda_entropy=0.01,
        lambda_classification=1.0,  # Enable supervision
        batch_size=64,
    )

    flush_print("\nTraining supervised model...")
    state_sup, metrics_sup = train_with_learned_embeddings(
        config_supervised,
        jnp.array(user_histories),
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        archetype_ids=jnp.array(arch_ids),
        seed=42,
        verbose=True,
    )

    recovery_sup = measure_recovery_learned(
        state_sup,
        jnp.array(user_histories),
        jnp.array(arch_ids),
        verbose=True,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("COMPARISON SUMMARY")
    flush_print("=" * 70)

    flush_print("\n--- Final Metrics ---")
    flush_print(f"{'Approach':<25} {'Accuracy':>10} {'Corr':>10} {'Assign':>10} {'Interp':>10} {'Divers':>10}")
    flush_print("-" * 80)

    flush_print(
        f"{'Unsupervised':<25} "
        f"{metrics_unsup.accuracy_history[-1]:>10.1%} "
        f"{recovery_unsup['mean_correlation']:>10.3f} "
        f"{recovery_unsup['assignment_accuracy']:>10.1%} "
        f"{recovery_unsup['interpretability']:>10.1%} "
        f"{recovery_unsup['diversity']:>10.3f}"
    )

    flush_print(
        f"{'Supervised (λ=1.0)':<25} "
        f"{metrics_sup.accuracy_history[-1]:>10.1%} "
        f"{recovery_sup['mean_correlation']:>10.3f} "
        f"{recovery_sup['assignment_accuracy']:>10.1%} "
        f"{recovery_sup['interpretability']:>10.1%} "
        f"{recovery_sup['diversity']:>10.3f}"
    )

    # Gate status
    flush_print("\n--- Gate Status ---")
    flush_print(f"{'Approach':<25} {'Corr>0.8':>12} {'Assign>70%':>12} {'Interp>60%':>12} {'Divers>0.1':>12}")
    flush_print("-" * 80)

    for name, recovery in [("Unsupervised", recovery_unsup), ("Supervised", recovery_sup)]:
        flush_print(
            f"{name:<25} "
            f"{'PASS' if recovery['mean_correlation'] > 0.8 else 'FAIL':>12} "
            f"{'PASS' if recovery['assignment_accuracy'] > 0.7 else 'FAIL':>12} "
            f"{'PASS' if recovery['interpretability'] > 0.6 else 'FAIL':>12} "
            f"{'PASS' if recovery['diversity'] > 0.1 else 'FAIL':>12}"
        )

    # =========================================================================
    # Test 3: Oracle - use one-hot archetype as "history" (upper bound)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("TEST 3: ORACLE - ONE-HOT ARCHETYPE AS INPUT")
    flush_print("(Upper bound: what if we have perfect user clustering?)")
    flush_print("=" * 70)

    # Create one-hot encoding of archetypes
    num_archetypes = 6
    oracle_histories = np.zeros((len(arch_ids), NUM_ACTIONS), dtype=np.float32)
    # Use first 6 dimensions as one-hot archetype, rest as zeros
    for i, aid in enumerate(arch_ids):
        oracle_histories[i, aid] = 1.0

    config_oracle = LearnedEmbeddingConfig(
        num_value_systems=6,
        embedding_dim=32,
        encoder_hidden_dim=64,
        mlp_hidden_dim=64,
        num_epochs=100,
        learning_rate=0.01,
        lambda_diversity=0.1,
        lambda_entropy=0.01,
        lambda_classification=0.0,  # No need - archetype is already input
        batch_size=64,
    )

    flush_print("\nTraining oracle model (archetype as input)...")
    state_oracle, metrics_oracle = train_with_learned_embeddings(
        config_oracle,
        jnp.array(oracle_histories),
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        archetype_ids=None,
        seed=42,
        verbose=True,
    )

    recovery_oracle = measure_recovery_learned(
        state_oracle,
        jnp.array(oracle_histories),
        jnp.array(arch_ids),
        verbose=True,
    )

    # =========================================================================
    # Summary
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("COMPARISON SUMMARY")
    flush_print("=" * 70)

    flush_print("\n--- Final Metrics ---")
    flush_print(f"{'Approach':<30} {'Accuracy':>10} {'Corr':>10} {'Assign':>10} {'Interp':>10} {'Divers':>10}")
    flush_print("-" * 85)

    for name, metrics, recovery in [
        ("Unsupervised (history)", metrics_unsup, recovery_unsup),
        ("Supervised (history)", metrics_sup, recovery_sup),
        ("Oracle (one-hot archetype)", metrics_oracle, recovery_oracle),
    ]:
        flush_print(
            f"{name:<30} "
            f"{metrics.accuracy_history[-1]:>10.1%} "
            f"{recovery['mean_correlation']:>10.3f} "
            f"{recovery['assignment_accuracy']:>10.1%} "
            f"{recovery['interpretability']:>10.1%} "
            f"{recovery['diversity']:>10.3f}"
        )

    # Gate status
    flush_print("\n--- Gate Status ---")
    flush_print(f"{'Approach':<30} {'Corr>0.8':>12} {'Assign>70%':>12} {'Interp>60%':>12} {'Divers>0.1':>12}")
    flush_print("-" * 85)

    for name, recovery in [
        ("Unsupervised (history)", recovery_unsup),
        ("Supervised (history)", recovery_sup),
        ("Oracle (one-hot archetype)", recovery_oracle),
    ]:
        flush_print(
            f"{name:<30} "
            f"{'PASS' if recovery['mean_correlation'] > 0.8 else 'FAIL':>12} "
            f"{'PASS' if recovery['assignment_accuracy'] > 0.7 else 'FAIL':>12} "
            f"{'PASS' if recovery['interpretability'] > 0.6 else 'FAIL':>12} "
            f"{'PASS' if recovery['diversity'] > 0.1 else 'FAIL':>12}"
        )

    # Save results
    summary = {
        'unsupervised': {
            'final_accuracy': float(metrics_unsup.accuracy_history[-1]),
            **recovery_unsup,
        },
        'supervised': {
            'final_accuracy': float(metrics_sup.accuracy_history[-1]),
            **recovery_sup,
        },
        'oracle': {
            'final_accuracy': float(metrics_oracle.accuracy_history[-1]),
            **recovery_oracle,
        },
    }

    with open(output_dir / 'learned_embeddings_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    flush_print(f"\nSaved summary to {output_dir / 'learned_embeddings_summary.json'}")

    flush_print("\n" + "=" * 70)
    flush_print("Test Complete!")
    flush_print("=" * 70)


if __name__ == "__main__":
    main()
