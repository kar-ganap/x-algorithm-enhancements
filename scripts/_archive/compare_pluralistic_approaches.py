#!/usr/bin/env python3
"""Compare three training approaches for pluralistic reward models.

Trains EM, Auxiliary, and Hybrid approaches on F2 synthetic data
and compares:
1. Training dynamics (loss, diversity, entropy curves)
2. Structural recovery (correlation with ground truth archetypes)
3. Noise robustness
4. Final accuracy

Usage:
    uv run python scripts/compare_pluralistic_approaches.py
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from enhancements.data import ContentTopic, UserArchetype, get_engagement_probs
from enhancements.reward_modeling.pluralistic import (
    PluralConfig,
    TrainingApproach,
    init_plural_state,
    train_pluralistic,
)
from enhancements.reward_modeling.structural_recovery import (
    check_recovery_gates,
    measure_structural_recovery,
)
from enhancements.reward_modeling.weights import NUM_ACTIONS


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


# =============================================================================
# Data Generation
# =============================================================================


def generate_training_data(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    embedding_dim: int = 64,
    noise_std: float = 0.0,
    label_flip_rate: float = 0.0,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate training data with user embeddings.

    Args:
        num_users_per_archetype: Users per archetype
        num_pairs_per_user: Preference pairs per user
        embedding_dim: User embedding dimension
        noise_std: Noise to add to action probabilities
        label_flip_rate: Rate of label flips
        rng: Random number generator

    Returns:
        (probs_preferred, probs_rejected, user_embeddings, archetype_ids)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_user_embs = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(archetypes):
        # Create archetype-specific base embedding
        arch_base_emb = rng.normal(0, 1, (embedding_dim,)).astype(np.float32)
        arch_base_emb = arch_base_emb / (np.linalg.norm(arch_base_emb) + 1e-8)

        for user_idx in range(num_users_per_archetype):
            # User embedding = archetype base + small variation
            user_emb = arch_base_emb + rng.normal(0, 0.1, (embedding_dim,)).astype(np.float32)
            user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)

            for _ in range(num_pairs_per_user):
                # Sample two different topics
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                topic_a, topic_b = topics[t1], topics[t2]

                probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
                probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

                # Add noise
                if noise_std > 0:
                    probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
                else:
                    # Small noise for topic-independent archetypes
                    probs_a = np.clip(probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)
                    probs_b = np.clip(probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1).astype(np.float32)

                # Determine preference
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
                    all_user_embs.append(user_emb)
                    all_arch_ids.append(arch_idx)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_user_embs, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
    )


# =============================================================================
# Training and Evaluation
# =============================================================================


def train_and_evaluate(
    approach: TrainingApproach,
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    user_embs: np.ndarray,
    arch_ids: np.ndarray,
    config: PluralConfig,
    seed: int = 42,
    use_supervision: bool = False,
) -> dict:
    """Train model with given approach and evaluate.

    Args:
        approach: Training approach
        probs_pref: Preferred item action probs
        probs_rej: Rejected item action probs
        user_embs: User embeddings
        arch_ids: Ground truth archetype IDs
        config: Training config
        seed: Random seed
        use_supervision: Whether to use supervised classification loss

    Returns dictionary with:
    - state: Final model state
    - metrics: Training metrics
    - recovery: Structural recovery metrics
    - gates: Gate pass/fail status
    """
    supervision_str = " + SUPERVISED" if use_supervision else ""
    flush_print(f"\n{'='*60}")
    flush_print(f"Training: {approach.value.upper()}{supervision_str}")
    flush_print(f"{'='*60}")

    # Initialize
    key = jax.random.PRNGKey(seed)
    embedding_dim = user_embs.shape[1]
    initial_state = init_plural_state(key, config, embedding_dim)

    # Train with optional supervision
    arch_ids_arg = jnp.array(arch_ids) if use_supervision else None

    final_state, metrics = train_pluralistic(
        initial_state,
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        jnp.array(user_embs),
        config,
        approach=approach,
        archetype_ids=arch_ids_arg,
        verbose=True,
    )

    # Evaluate structural recovery
    recovery = measure_structural_recovery(
        final_state,
        jnp.array(user_embs),
        jnp.array(arch_ids),
        verbose=True,
    )

    # Check gates
    gates = check_recovery_gates(recovery)

    return {
        'approach': approach.value,
        'supervised': use_supervision,
        'state': final_state,
        'metrics': metrics,
        'recovery': recovery,
        'gates': gates,
    }


def compare_approaches(
    results: dict[str, dict],
    output_dir: Path,
):
    """Compare results across approaches and create visualizations."""
    flush_print("\n" + "=" * 70)
    flush_print("COMPARISON SUMMARY")
    flush_print("=" * 70)

    # Summary table
    flush_print("\n--- Final Metrics ---")
    flush_print(f"{'Approach':<25} {'Accuracy':>10} {'Corr':>10} {'Assign':>10} {'Interp':>10} {'Divers':>10}")
    flush_print("-" * 80)

    for name, res in results.items():
        metrics = res['metrics']
        recovery = res['recovery']
        flush_print(
            f"{name:<25} "
            f"{metrics.accuracy_history[-1]:>10.1%} "
            f"{recovery.mean_correlation:>10.3f} "
            f"{recovery.assignment_accuracy:>10.1%} "
            f"{recovery.interpretability_score:>10.1%} "
            f"{recovery.system_diversity:>10.3f}"
        )

    # Gate summary
    flush_print("\n--- Gate Status ---")
    flush_print(f"{'Approach':<25} {'Corr>0.8':>12} {'Assign>70%':>12} {'Interp>60%':>12} {'Divers>0.1':>12}")
    flush_print("-" * 80)

    for name, res in results.items():
        gates = res['gates']
        flush_print(
            f"{name:<25} "
            f"{'PASS' if gates['mean_correlation_gt_0.8'] else 'FAIL':>12} "
            f"{'PASS' if gates['assignment_accuracy_gt_0.7'] else 'FAIL':>12} "
            f"{'PASS' if gates['interpretability_gt_0.6'] else 'FAIL':>12} "
            f"{'PASS' if gates['diversity_gt_0.1'] else 'FAIL':>12}"
        )

    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    colors = {
        'em': 'blue',
        'auxiliary': 'orange',
        'hybrid': 'green',
        'auxiliary_supervised': 'red',
        'hybrid_supervised': 'purple',
        'aux_sup_λ1.0': 'pink',
        'aux_sup_λ5.0': 'red',
        'aux_sup_λ10.0': 'darkred',
    }

    # Plot 1: Loss curves
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['metrics'].loss_history, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Iteration/Epoch')
    ax.set_ylabel('Bradley-Terry Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    ax = axes[0, 1]
    for name, res in results.items():
        ax.plot(res['metrics'].accuracy_history, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Iteration/Epoch')
    ax.set_ylabel('Preference Accuracy')
    ax.set_title('Training Accuracy')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.05])

    # Plot 3: Diversity curves
    ax = axes[0, 2]
    for name, res in results.items():
        ax.plot(res['metrics'].diversity_history, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Iteration/Epoch')
    ax.set_ylabel('Diversity Loss')
    ax.set_title('System Diversity (lower = more similar)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Entropy curves
    ax = axes[1, 0]
    for name, res in results.items():
        ax.plot(res['metrics'].entropy_history, label=name, color=colors.get(name, 'gray'))
    ax.set_xlabel('Iteration/Epoch')
    ax.set_ylabel('Entropy')
    ax.set_title('Assignment Entropy (lower = peakier)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 5: Correlation bar chart
    ax = axes[1, 1]
    names = list(results.keys())
    correlations = [res['recovery'].mean_correlation for res in results.values()]
    bar_colors = [colors.get(n, 'gray') for n in names]
    bars = ax.bar(names, correlations, color=bar_colors, alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Gate (0.8)')
    ax.set_ylabel('Mean Correlation')
    ax.set_title('Weight Correlation with Ground Truth')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    for bar, corr in zip(bars, correlations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{corr:.3f}', ha='center', va='bottom', fontsize=8)

    # Plot 6: Assignment accuracy bar chart
    ax = axes[1, 2]
    accuracies = [res['recovery'].assignment_accuracy for res in results.values()]
    bars = ax.bar(names, accuracies, color=bar_colors, alpha=0.7)
    ax.axhline(y=0.7, color='red', linestyle='--', label='Gate (70%)')
    ax.set_ylabel('Assignment Accuracy')
    ax.set_title('User Assignment Accuracy')
    ax.set_ylim([0, 1.1])
    ax.legend()
    ax.tick_params(axis='x', rotation=30)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'approach_comparison.png', dpi=150)
    plt.close()
    flush_print(f"\nSaved comparison plot to {output_dir / 'approach_comparison.png'}")


def test_noise_robustness(
    best_approach: TrainingApproach,
    config: PluralConfig,
    output_dir: Path,
):
    """Test noise robustness of the best approach."""
    flush_print("\n" + "=" * 70)
    flush_print(f"NOISE ROBUSTNESS TEST ({best_approach.value})")
    flush_print("=" * 70)

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    flip_rates = [0.0, 0.05, 0.10, 0.15]

    results = []

    for noise in noise_levels:
        for flip in flip_rates:
            flush_print(f"\n  Testing noise={noise:.2f}, flip={flip:.2f}...")

            # Generate noisy data
            probs_pref, probs_rej, user_embs, arch_ids = generate_training_data(
                num_users_per_archetype=50,
                num_pairs_per_user=5,
                noise_std=noise,
                label_flip_rate=flip,
                rng=np.random.default_rng(42),
            )

            # Train
            key = jax.random.PRNGKey(42)
            initial_state = init_plural_state(key, config, user_embs.shape[1])

            # Use shorter training for speed
            test_config = PluralConfig(
                num_value_systems=config.num_value_systems,
                num_epochs=30,
                em_iterations=5,
                m_step_iterations=10,
                learning_rate=config.learning_rate,
                lambda_diversity=config.lambda_diversity,
                lambda_entropy=config.lambda_entropy,
            )

            final_state, metrics = train_pluralistic(
                initial_state,
                jnp.array(probs_pref),
                jnp.array(probs_rej),
                jnp.array(user_embs),
                test_config,
                approach=best_approach,
                verbose=False,
            )

            # Evaluate
            recovery = measure_structural_recovery(
                final_state,
                jnp.array(user_embs),
                jnp.array(arch_ids),
                verbose=False,
            )

            results.append({
                'noise': float(noise),
                'flip': float(flip),
                'accuracy': float(metrics.accuracy_history[-1]),
                'correlation': float(recovery.mean_correlation),
                'assignment': float(recovery.assignment_accuracy),
            })

            flush_print(f"    Acc={metrics.accuracy_history[-1]:.1%}, "
                       f"Corr={recovery.mean_correlation:.3f}, "
                       f"Assign={recovery.assignment_accuracy:.1%}")

    # Save results
    with open(output_dir / 'noise_robustness.json', 'w') as f:
        json.dump(results, f, indent=2)

    flush_print(f"\nSaved noise results to {output_dir / 'noise_robustness.json'}")


# =============================================================================
# Main
# =============================================================================


def main():
    output_dir = Path("results/f4_phase2")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Phase 2: Pluralistic Reward Model Comparison")
    flush_print("=" * 70)

    # Generate training data first
    flush_print("\nGenerating training data...")
    probs_pref, probs_rej, user_embs, arch_ids = generate_training_data(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        embedding_dim=64,
        noise_std=0.0,
        label_flip_rate=0.0,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref)} preference pairs")
    flush_print(f"User embedding dim: {user_embs.shape[1]}")
    flush_print(f"Archetypes: {len(set(arch_ids))}")

    # =========================================================================
    # Part 1: Test unsupervised approaches (original experiment)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("PART 1: UNSUPERVISED APPROACHES")
    flush_print("=" * 70)

    config_unsupervised = PluralConfig(
        num_value_systems=6,  # Match number of archetypes
        num_epochs=50,        # For auxiliary approach
        em_iterations=10,     # For EM/hybrid approaches
        m_step_iterations=20,
        learning_rate=0.01,
        lambda_diversity=0.1,
        lambda_entropy=0.01,
        lambda_classification=0.0,  # No supervision
        batch_size=64,
    )

    results = {}

    for approach in [TrainingApproach.EM, TrainingApproach.AUXILIARY, TrainingApproach.HYBRID]:
        result = train_and_evaluate(
            approach,
            probs_pref,
            probs_rej,
            user_embs,
            arch_ids,
            config_unsupervised,
            seed=42,
            use_supervision=False,
        )
        results[approach.value] = result

    # =========================================================================
    # Part 2: Test supervised approaches (Fix 1)
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("PART 2: SUPERVISED APPROACHES (Fix 1: Classification Loss)")
    flush_print("=" * 70)

    # Test different classification weights
    for lambda_cls in [1.0, 5.0, 10.0]:
        config_supervised = PluralConfig(
            num_value_systems=6,
            num_epochs=50,
            em_iterations=10,
            m_step_iterations=20,
            learning_rate=0.01,
            lambda_diversity=0.1,
            lambda_entropy=0.01,
            lambda_classification=lambda_cls,
            batch_size=64,
        )

        # Only test auxiliary (best performing) with different lambda values
        result = train_and_evaluate(
            TrainingApproach.AUXILIARY,
            probs_pref,
            probs_rej,
            user_embs,
            arch_ids,
            config_supervised,
            seed=42,
            use_supervision=True,
        )
        results[f"aux_sup_λ{lambda_cls}"] = result

    # Also test hybrid with best lambda
    config_supervised = PluralConfig(
        num_value_systems=6,
        num_epochs=50,
        em_iterations=10,
        m_step_iterations=20,
        learning_rate=0.01,
        lambda_diversity=0.1,
        lambda_entropy=0.01,
        lambda_classification=5.0,
        batch_size=64,
    )

    result = train_and_evaluate(
        TrainingApproach.HYBRID,
        probs_pref,
        probs_rej,
        user_embs,
        arch_ids,
        config_supervised,
        seed=42,
        use_supervision=True,
    )
    results["hybrid_supervised"] = result

    # Compare all approaches
    compare_approaches(results, output_dir)

    # Determine best approach
    best_name = max(
        results.keys(),
        key=lambda k: results[k]['recovery'].mean_correlation
    )
    flush_print(f"\n*** Best approach by correlation: {best_name.upper()} ***")

    # Skip noise robustness test for supervised approaches (complex naming)
    # Use hybrid (best unsupervised) for noise robustness
    test_noise_robustness(
        TrainingApproach.HYBRID,
        config_unsupervised,
        output_dir,
    )

    # Save summary
    summary = {
        name: {
            'final_accuracy': float(res['metrics'].accuracy_history[-1]),
            'mean_correlation': float(res['recovery'].mean_correlation),
            'assignment_accuracy': float(res['recovery'].assignment_accuracy),
            'interpretability': float(res['recovery'].interpretability_score),
            'diversity': float(res['recovery'].system_diversity),
            'gates_passed': int(sum(res['gates'].values())),
            'gates_total': int(len(res['gates'])),
            'supervised': bool(res.get('supervised', False)),
        }
        for name, res in results.items()
    }

    with open(output_dir / 'comparison_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    flush_print(f"\nSaved summary to {output_dir / 'comparison_summary.json'}")

    flush_print("\n" + "=" * 70)
    flush_print("Comparison Complete!")
    flush_print("=" * 70)


if __name__ == "__main__":
    main()
