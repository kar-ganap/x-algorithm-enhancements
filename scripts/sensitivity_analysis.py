#!/usr/bin/env python3
"""Sensitivity analysis for Bradley-Terry reward model.

Studies:
1. Effect of noise levels (0%, 5%, 10%, 15%, 20%, 25%, 30%)
2. Effect of held-out archetypes (hold out 1, 2, 3, 4 archetypes)
3. Effect of noisy training data vs clean training data

Usage:
    uv run python scripts/sensitivity_analysis.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from enhancements.data import (
    ContentTopic,
    UserArchetype,
    get_engagement_probs,
)
from enhancements.reward_modeling import (
    NUM_ACTIONS,
    RewardWeights,
    compute_preference_accuracy,
    contextual_bradley_terry_loss,
)


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


# =============================================================================
# Data Generation
# =============================================================================


def generate_training_data(
    archetypes: List[UserArchetype],
    num_pairs_per_archetype: int = 200,
    noise_std: float = 0.0,
    label_flip_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data with optional noise."""
    if rng is None:
        rng = np.random.default_rng(42)

    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(archetypes):
        pairs_created = 0
        max_attempts = num_pairs_per_archetype * 20

        for _ in range(max_attempts):
            if pairs_created >= num_pairs_per_archetype:
                break

            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a, topic_b = topics[topic_indices[0]], topics[topic_indices[1]]

            probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
            probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

            # Add noise to probabilities
            if noise_std > 0:
                probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
            else:
                # Still add tiny noise for topic-independent archetypes
                probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            if abs(score_a - score_b) > 0.03:
                if score_a > score_b:
                    pref, rej = probs_a, probs_b
                else:
                    pref, rej = probs_b, probs_a

                # Flip labels with probability
                if label_flip_rate > 0 and rng.random() < label_flip_rate:
                    pref, rej = rej, pref

                all_probs_pref.append(pref)
                all_probs_rej.append(rej)
                all_arch_ids.append(arch_idx)
                pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
    )


def train_model(
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    arch_ids: np.ndarray,
    num_archetypes: int,
    num_epochs: int = 30,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    rng: Optional[np.random.Generator] = None,
) -> jnp.ndarray:
    """Train Bradley-Terry model."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Convert to JAX
    probs_pref = jnp.array(probs_pref)
    probs_rej = jnp.array(probs_rej)
    arch_ids = jnp.array(arch_ids)

    # Initialize weights
    default = RewardWeights.default()
    weights = jnp.tile(default.weights[jnp.newaxis, :], (num_archetypes, 1))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    def loss_fn(w, p_pref, p_rej, a_ids):
        return contextual_bradley_terry_loss(w, p_pref, p_rej, a_ids, None)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    n_samples = len(probs_pref)

    for _ in range(num_epochs):
        perm = rng.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            _, grads = grad_fn(weights, probs_pref[idx], probs_rej[idx], arch_ids[idx])
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

    return weights


def evaluate_model(
    weights: jnp.ndarray,
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    arch_ids: np.ndarray,
) -> float:
    """Evaluate model accuracy."""
    return float(compute_preference_accuracy(
        weights,
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        jnp.array(arch_ids),
    ))


# =============================================================================
# Sensitivity Studies
# =============================================================================


def study_noise_sensitivity(
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Study effect of noise levels on model performance."""
    flush_print("\n" + "=" * 70)
    flush_print("STUDY 1: Noise Sensitivity")
    flush_print("=" * 70)

    archetypes = list(UserArchetype)
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    label_flip_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    results = {
        "train_noise_test_clean": {},
        "train_clean_test_noise": {},
        "train_noise_test_noise": {},
    }

    # Generate clean test data
    flush_print("\nGenerating clean test data...")
    test_pref, test_rej, test_ids = generate_training_data(
        archetypes, num_pairs_per_archetype=50, noise_std=0.0, label_flip_rate=0.0,
        rng=np.random.default_rng(999)
    )

    # Study 1a: Train with noise, test on clean
    flush_print("\n--- Train with noise, test on clean ---")
    for noise in noise_levels:
        flush_print(f"  Training with noise_std={noise:.2f}...")
        train_pref, train_rej, train_ids = generate_training_data(
            archetypes, num_pairs_per_archetype=100, noise_std=noise, label_flip_rate=0.0,
            rng=np.random.default_rng(42)
        )
        weights = train_model(train_pref, train_rej, train_ids, len(archetypes))
        acc = evaluate_model(weights, test_pref, test_rej, test_ids)
        results["train_noise_test_clean"][f"noise_{noise:.2f}"] = acc
        flush_print(f"    Accuracy: {acc:.1%}")

    # Study 1b: Train clean, test with noise
    flush_print("\n--- Train clean, test with noise ---")
    train_pref, train_rej, train_ids = generate_training_data(
        archetypes, num_pairs_per_archetype=100, noise_std=0.0, label_flip_rate=0.0,
        rng=np.random.default_rng(42)
    )
    weights_clean = train_model(train_pref, train_rej, train_ids, len(archetypes))

    for noise in noise_levels:
        flush_print(f"  Testing with noise_std={noise:.2f}...")
        noisy_test_pref, noisy_test_rej, noisy_test_ids = generate_training_data(
            archetypes, num_pairs_per_archetype=50, noise_std=noise, label_flip_rate=0.0,
            rng=np.random.default_rng(888)
        )
        acc = evaluate_model(weights_clean, noisy_test_pref, noisy_test_rej, noisy_test_ids)
        results["train_clean_test_noise"][f"noise_{noise:.2f}"] = acc
        flush_print(f"    Accuracy: {acc:.1%}")

    # Study 1c: Effect of label flips
    flush_print("\n--- Effect of label flip rate (train clean, test with flips) ---")
    for flip in label_flip_rates:
        flush_print(f"  Testing with flip_rate={flip:.2f}...")
        flip_test_pref, flip_test_rej, flip_test_ids = generate_training_data(
            archetypes, num_pairs_per_archetype=50, noise_std=0.05, label_flip_rate=flip,
            rng=np.random.default_rng(777)
        )
        acc = evaluate_model(weights_clean, flip_test_pref, flip_test_rej, flip_test_ids)
        results["train_noise_test_noise"][f"flip_{flip:.2f}"] = acc
        flush_print(f"    Accuracy: {acc:.1%}")

    return results


def study_held_out_archetypes(
    output_dir: Path,
) -> Dict[str, float]:
    """Study effect of number of held-out archetypes."""
    flush_print("\n" + "=" * 70)
    flush_print("STUDY 2: Held-Out Archetype Generalization")
    flush_print("=" * 70)

    all_archetypes = list(UserArchetype)
    results = {}

    # Test holding out 1, 2, 3, 4 archetypes
    held_out_configs = [
        # (train_archetypes, test_archetypes, name)
        (all_archetypes[:5], [all_archetypes[5]], "hold_1"),  # Hold POWER_USER
        (all_archetypes[:4], all_archetypes[4:], "hold_2"),   # Hold LURKER, POWER_USER
        (all_archetypes[:3], all_archetypes[3:], "hold_3"),   # Hold TECH_BRO, LURKER, POWER_USER
        (all_archetypes[:2], all_archetypes[2:], "hold_4"),   # Hold half
    ]

    for train_archs, test_archs, name in held_out_configs:
        flush_print(f"\n--- {name}: Train on {len(train_archs)}, test on {len(test_archs)} ---")
        flush_print(f"  Train: {[a.value for a in train_archs]}")
        flush_print(f"  Test:  {[a.value for a in test_archs]}")

        # Train on subset
        train_pref, train_rej, train_ids = generate_training_data(
            train_archs, num_pairs_per_archetype=100, noise_std=0.0,
            rng=np.random.default_rng(42)
        )
        weights = train_model(train_pref, train_rej, train_ids, len(train_archs))

        # Test on held-out (using first archetype's weights for all)
        test_pref, test_rej, _ = generate_training_data(
            test_archs, num_pairs_per_archetype=50, noise_std=0.0,
            rng=np.random.default_rng(999)
        )
        # Use weights[0] for all test samples (simulating unknown archetype)
        test_ids = np.zeros(len(test_pref), dtype=np.int32)
        single_weight = weights[0:1]

        acc = evaluate_model(single_weight, test_pref, test_rej, test_ids)
        results[name] = acc
        flush_print(f"  Accuracy: {acc:.1%}")

    # Also test specific archetype transfers
    flush_print("\n--- Specific Archetype Transfers ---")
    transfers = [
        (UserArchetype.SPORTS_FAN, UserArchetype.TECH_BRO, "sports→tech"),
        (UserArchetype.POLITICAL_L, UserArchetype.POLITICAL_R, "polL→polR"),
        (UserArchetype.TECH_BRO, UserArchetype.LURKER, "tech→lurker"),
        (UserArchetype.POWER_USER, UserArchetype.LURKER, "power→lurker"),
    ]

    for src, tgt, name in transfers:
        train_pref, train_rej, train_ids = generate_training_data(
            [src], num_pairs_per_archetype=100, noise_std=0.0,
            rng=np.random.default_rng(42)
        )
        weights = train_model(train_pref, train_rej, train_ids, 1)

        test_pref, test_rej, _ = generate_training_data(
            [tgt], num_pairs_per_archetype=50, noise_std=0.0,
            rng=np.random.default_rng(999)
        )
        test_ids = np.zeros(len(test_pref), dtype=np.int32)

        acc = evaluate_model(weights, test_pref, test_rej, test_ids)
        results[f"transfer_{name}"] = acc
        flush_print(f"  {name}: {acc:.1%}")

    return results


def study_noisy_training(
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Study effect of training with noisy data."""
    flush_print("\n" + "=" * 70)
    flush_print("STUDY 3: Noisy Training Data")
    flush_print("=" * 70)

    archetypes = list(UserArchetype)
    results = {"noisy_train_clean_test": {}, "noisy_train_noisy_test": {}}

    # Generate clean test data
    clean_test_pref, clean_test_rej, clean_test_ids = generate_training_data(
        archetypes, num_pairs_per_archetype=50, noise_std=0.0, label_flip_rate=0.0,
        rng=np.random.default_rng(999)
    )

    noise_levels = [0.0, 0.10, 0.20, 0.30]
    flip_rates = [0.0, 0.10, 0.20]

    flush_print("\n--- Training with various noise/flip combinations ---")
    for noise in noise_levels:
        for flip in flip_rates:
            key = f"noise_{noise:.2f}_flip_{flip:.2f}"
            flush_print(f"  Training with noise={noise:.2f}, flip={flip:.2f}...")

            train_pref, train_rej, train_ids = generate_training_data(
                archetypes, num_pairs_per_archetype=100,
                noise_std=noise, label_flip_rate=flip,
                rng=np.random.default_rng(42)
            )
            weights = train_model(train_pref, train_rej, train_ids, len(archetypes))

            # Test on clean
            acc_clean = evaluate_model(weights, clean_test_pref, clean_test_rej, clean_test_ids)
            results["noisy_train_clean_test"][key] = acc_clean

            # Test on similarly noisy data
            noisy_test_pref, noisy_test_rej, noisy_test_ids = generate_training_data(
                archetypes, num_pairs_per_archetype=50,
                noise_std=noise, label_flip_rate=flip,
                rng=np.random.default_rng(888)
            )
            acc_noisy = evaluate_model(weights, noisy_test_pref, noisy_test_rej, noisy_test_ids)
            results["noisy_train_noisy_test"][key] = acc_noisy

            flush_print(f"    Clean test: {acc_clean:.1%}, Noisy test: {acc_noisy:.1%}")

    return results


# =============================================================================
# Visualization
# =============================================================================


def plot_sensitivity_results(
    noise_results: Dict,
    held_out_results: Dict,
    noisy_train_results: Dict,
    output_dir: Path,
):
    """Create visualization of sensitivity analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Noise sensitivity (train clean, test noisy)
    ax1 = axes[0, 0]
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    accuracies = [noise_results["train_clean_test_noise"].get(f"noise_{n:.2f}", 0) for n in noise_levels]
    ax1.plot(noise_levels, accuracies, 'b-o', linewidth=2, markersize=8)
    ax1.set_xlabel('Test Noise (std)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Effect of Test Noise (Clean Training)')
    ax1.set_ylim([0.4, 1.05])
    ax1.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90%')
    ax1.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='80%')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Label flip sensitivity
    ax2 = axes[0, 1]
    flip_rates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    flip_accs = [noise_results["train_noise_test_noise"].get(f"flip_{f:.2f}", 0) for f in flip_rates]
    ax2.plot(flip_rates, flip_accs, 'r-o', linewidth=2, markersize=8)
    ax2.set_xlabel('Label Flip Rate')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Effect of Label Flips')
    ax2.set_ylim([0.4, 1.05])
    ax2.axhline(y=0.9, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Held-out archetypes
    ax3 = axes[1, 0]
    hold_names = ['hold_1', 'hold_2', 'hold_3', 'hold_4']
    hold_accs = [held_out_results.get(n, 0) for n in hold_names]
    x = [1, 2, 3, 4]
    ax3.bar(x, hold_accs, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Number of Held-Out Archetypes')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Generalization vs Held-Out Count')
    ax3.set_xticks(x)
    ax3.set_ylim([0.4, 1.05])
    ax3.axhline(y=0.9, color='g', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Archetype transfer
    ax4 = axes[1, 1]
    transfers = ['transfer_sports→tech', 'transfer_polL→polR', 'transfer_tech→lurker', 'transfer_power→lurker']
    transfer_labels = ['Sports→Tech', 'PolL→PolR', 'Tech→Lurker', 'Power→Lurker']
    transfer_accs = [held_out_results.get(t, 0) for t in transfers]
    ax4.barh(transfer_labels, transfer_accs, color='coral', alpha=0.7)
    ax4.set_xlabel('Accuracy')
    ax4.set_title('Cross-Archetype Transfer')
    ax4.set_xlim([0.4, 1.05])
    ax4.axvline(x=0.9, color='g', linestyle='--', alpha=0.5)
    ax4.axvline(x=0.8, color='orange', linestyle='--', alpha=0.5)
    ax4.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_analysis.png", dpi=150)
    plt.close()
    flush_print(f"\nSaved plot to {output_dir / 'sensitivity_analysis.png'}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run sensitivity analysis."""
    output_dir = Path("results/f4_phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Phase 1: Sensitivity Analysis")
    flush_print("=" * 70)

    # Run studies
    noise_results = study_noise_sensitivity(output_dir)
    held_out_results = study_held_out_archetypes(output_dir)
    noisy_train_results = study_noisy_training(output_dir)

    # Summary
    flush_print("\n" + "=" * 70)
    flush_print("SUMMARY")
    flush_print("=" * 70)

    flush_print("\n1. Noise Sensitivity (clean training):")
    for key, acc in noise_results["train_clean_test_noise"].items():
        flush_print(f"   {key}: {acc:.1%}")

    flush_print("\n2. Label Flip Sensitivity:")
    for key, acc in noise_results["train_noise_test_noise"].items():
        flush_print(f"   {key}: {acc:.1%}")

    flush_print("\n3. Held-Out Generalization:")
    for key, acc in held_out_results.items():
        flush_print(f"   {key}: {acc:.1%}")

    flush_print("\n4. Noisy Training (tested on clean):")
    for key, acc in noisy_train_results["noisy_train_clean_test"].items():
        flush_print(f"   {key}: {acc:.1%}")

    # Save results
    all_results = {
        "noise_sensitivity": noise_results,
        "held_out_archetypes": held_out_results,
        "noisy_training": noisy_train_results,
    }

    results_path = output_dir / "sensitivity_analysis.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    flush_print(f"\nSaved results to {results_path}")

    # Plot
    plot_sensitivity_results(noise_results, held_out_results, noisy_train_results, output_dir)

    flush_print("\n" + "=" * 70)
    flush_print("Sensitivity Analysis Complete!")
    flush_print("=" * 70)


if __name__ == "__main__":
    main()
