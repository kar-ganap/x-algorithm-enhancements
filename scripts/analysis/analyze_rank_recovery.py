#!/usr/bin/env python3
"""Rank-order recovery analysis for F4 reward modeling.

Answers: is the two-stage model's 0.554 Pearson correlation a scale artifact
of BT's inherent scale invariance? BT can only recover weight *direction*,
not magnitudes — so Pearson (sensitive to scale) may understate recovery.
Kendall's tau and Spearman rho measure rank agreement and are scale-invariant.

Also computes rank recovery for single-stakeholder BT models (user, platform).

Usage:
    uv run python scripts/analyze_rank_recovery.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

# Add project root + phoenix to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

from enhancements.data import ContentTopic, UserArchetype, get_engagement_probs
from enhancements.reward_modeling.structural_recovery import (
    compute_correlation_matrix,
    compute_rank_correlation_matrix,
    get_all_ground_truth_weights,
    match_systems_to_archetypes,
)
from enhancements.reward_modeling.two_stage import TwoStageConfig, train_two_stage
from enhancements.reward_modeling.weights import ACTION_INDICES, NUM_ACTIONS


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def generate_training_data(
    num_users_per_archetype: int = 100,
    num_pairs_per_user: int = 10,
    use_topic_features: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic training data matching test_two_stage.py config."""
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
                user_history = np.zeros(num_topics, dtype=np.float32)
                for t_idx, topic in enumerate(topics):
                    probs = np.array(
                        get_engagement_probs(archetype, topic).to_array(),
                        dtype=np.float32,
                    )
                    probs = np.clip(
                        probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1
                    ).astype(np.float32)
                    user_history[t_idx] = probs[:14].sum()
            else:
                user_history = np.zeros(NUM_ACTIONS, dtype=np.float32)
                for _ in range(20):
                    topic = topics[rng.choice(len(topics))]
                    probs = np.array(
                        get_engagement_probs(archetype, topic).to_array(),
                        dtype=np.float32,
                    )
                    probs = np.clip(
                        probs + rng.normal(0, 0.05, NUM_ACTIONS), 0, 1
                    ).astype(np.float32)
                    user_history += probs
                user_history /= 20

            for _ in range(num_pairs_per_user):
                t1, t2 = rng.choice(len(topics), size=2, replace=False)
                probs_a = np.array(
                    get_engagement_probs(archetype, topics[t1]).to_array(),
                    dtype=np.float32,
                )
                probs_b = np.array(
                    get_engagement_probs(archetype, topics[t2]).to_array(),
                    dtype=np.float32,
                )
                probs_a = np.clip(
                    probs_a + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1
                ).astype(np.float32)
                probs_b = np.clip(
                    probs_b + rng.normal(0, 0.02, NUM_ACTIONS), 0, 1
                ).astype(np.float32)

                score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
                score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

                if abs(score_a - score_b) > 0.03:
                    if score_a > score_b:
                        pref, rej = probs_a, probs_b
                    else:
                        pref, rej = probs_b, probs_a
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


def build_user_gt_weights() -> np.ndarray:
    """Build 18D ground truth from UtilityWeights for user stakeholder."""
    gt = np.zeros(NUM_ACTIONS, dtype=np.float32)
    # User positive weights
    user_positive = {
        "favorite": 1.0,
        "repost": 0.8,
        "reply": 0.5,
        "share": 0.9,
        "follow_author": 1.2,
        "quote": 0.6,
    }
    for action, weight in user_positive.items():
        gt[ACTION_INDICES[action]] = weight
    # User negative weights
    user_negative = {
        "block_author": -2.0,
        "mute_author": -1.5,
        "report": -2.5,
        "not_interested": -1.0,
    }
    for action, weight in user_negative.items():
        gt[ACTION_INDICES[action]] = weight
    return gt


def build_platform_gt_weights() -> np.ndarray:
    """Build 18D ground truth from UtilityWeights for platform stakeholder."""
    gt = np.zeros(NUM_ACTIONS, dtype=np.float32)
    platform_weights = {
        "favorite": 1.0,
        "reply": 1.2,
        "repost": 1.5,
        "photo_expand": 0.3,
        "click": 0.5,
        "profile_click": 0.6,
        "vqv": 0.4,
        "share": 1.3,
        "share_via_dm": 1.4,
        "share_via_copy_link": 1.2,
        "dwell": 0.2,
        "quote": 1.3,
        "quoted_click": 0.4,
        "follow_author": 1.0,
        "not_interested": 0.1,
        "block_author": 0.1,
        "mute_author": 0.1,
        "report": 0.2,
    }
    for action, weight in platform_weights.items():
        gt[ACTION_INDICES[action]] = weight
    return gt


def analyze_bt_model(
    name: str,
    json_path: Path,
    gt_weights: np.ndarray,
) -> dict:
    """Compute rank recovery for a single BT model."""
    with open(json_path) as f:
        data = json.load(f)

    learned = np.array(data["weights_vector"], dtype=np.float32)

    pearson = float(np.corrcoef(learned, gt_weights)[0, 1])
    tau, tau_p = kendalltau(learned, gt_weights)
    rho, rho_p = spearmanr(learned, gt_weights)

    return {
        "name": name,
        "pearson": pearson,
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "accuracy": data["accuracy"],
    }


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Rank-Order Recovery Analysis")
    flush_print("=" * 70)
    flush_print("\nQuestion: Is the 0.554 mean Pearson a BT scale artifact?")
    flush_print("Method: Compare Pearson vs Kendall's tau vs Spearman rho\n")

    results = {}

    # =========================================================================
    # Part 1: Two-stage model (retrain to extract weights)
    # =========================================================================
    flush_print("--- Part 1: Two-Stage Model ---")
    flush_print("Generating training data (topic-aware features)...")

    probs_pref, probs_rej, histories, arch_ids = generate_training_data(
        num_users_per_archetype=100,
        num_pairs_per_user=10,
        use_topic_features=True,
        rng=np.random.default_rng(42),
    )
    flush_print(f"Generated {len(probs_pref)} pairs, features: {histories.shape}")

    config = TwoStageConfig(
        num_clusters=6,
        learning_rate=0.01,
        num_epochs=100,
        batch_size=64,
    )

    flush_print("Training two-stage model...")
    state, metrics = train_two_stage(
        histories, probs_pref, probs_rej, config, verbose=False
    )
    flush_print(f"Training accuracy: {metrics.overall_accuracy:.4f}")

    # Compute all three correlations
    gt_weights = get_all_ground_truth_weights()
    learned_weights = state.cluster_weights  # [K, 18]

    pearson_matrix = compute_correlation_matrix(learned_weights, gt_weights)
    kendall_matrix, spearman_matrix = compute_rank_correlation_matrix(
        learned_weights, gt_weights
    )

    # Match systems to archetypes using Pearson (same as original)
    matches = match_systems_to_archetypes(pearson_matrix)
    archetypes = list(UserArchetype)

    flush_print(f"\n{'System':<10} {'Archetype':<15} {'Pearson':>8} {'Kendall':>8} {'Spearman':>9}")
    flush_print("-" * 55)

    per_cluster = {}
    for k, (arch_name, pearson_corr) in sorted(matches.items()):
        arch_idx = archetypes.index(UserArchetype(arch_name))
        tau = kendall_matrix[k, arch_idx]
        rho = spearman_matrix[k, arch_idx]
        flush_print(f"  {k:<8} {arch_name:<15} {pearson_corr:>8.3f} {tau:>8.3f} {rho:>9.3f}")
        per_cluster[str(k)] = {
            "archetype": arch_name,
            "pearson": float(pearson_corr),
            "kendall_tau": float(tau),
            "spearman_rho": float(rho),
        }

    mean_pearson = float(np.mean([v["pearson"] for v in per_cluster.values()]))
    mean_kendall = float(np.mean([v["kendall_tau"] for v in per_cluster.values()]))
    mean_spearman = float(np.mean([v["spearman_rho"] for v in per_cluster.values()]))

    flush_print(f"\n  {'Mean':<23} {mean_pearson:>8.3f} {mean_kendall:>8.3f} {mean_spearman:>9.3f}")

    results["two_stage"] = {
        "per_cluster": per_cluster,
        "mean_pearson": mean_pearson,
        "mean_kendall": mean_kendall,
        "mean_spearman": mean_spearman,
        "training_accuracy": float(metrics.overall_accuracy),
    }

    # =========================================================================
    # Part 2: Single-stakeholder BT models
    # =========================================================================
    flush_print("\n--- Part 2: Single-Stakeholder BT Models ---")

    bt_results = []

    # User BT
    user_bt_path = Path("results/loss_experiments/bradley_terry_user.json")
    if user_bt_path.exists():
        user_result = analyze_bt_model("BT (user)", user_bt_path, build_user_gt_weights())
        bt_results.append(user_result)
    else:
        flush_print(f"  Warning: {user_bt_path} not found, skipping")

    # Platform BT
    platform_bt_path = Path("results/loss_experiments/bradley_terry_platform.json")
    if platform_bt_path.exists():
        platform_result = analyze_bt_model("BT (platform)", platform_bt_path, build_platform_gt_weights())
        bt_results.append(platform_result)
    else:
        flush_print(f"  Warning: {platform_bt_path} not found, skipping")

    if bt_results:
        flush_print(f"\n{'Model':<18} {'Pearson':>8} {'Kendall':>8} {'Spearman':>9} {'Accuracy':>9}")
        flush_print("-" * 55)
        for r in bt_results:
            flush_print(
                f"  {r['name']:<16} {r['pearson']:>8.3f} {r['kendall_tau']:>8.3f} "
                f"{r['spearman_rho']:>9.3f} {r['accuracy']:>9.3f}"
            )
        results["bt_models"] = bt_results

    # =========================================================================
    # Summary
    # =========================================================================
    flush_print("\n" + "=" * 70)
    flush_print("SUMMARY")
    flush_print("=" * 70)

    flush_print("\nTwo-stage model (K=6, topic-aware):")
    flush_print(f"  Mean Pearson:   {mean_pearson:.3f}")
    flush_print(f"  Mean Kendall τ: {mean_kendall:.3f}")
    flush_print(f"  Mean Spearman ρ:{mean_spearman:.3f}")

    delta_kendall = mean_kendall - mean_pearson
    delta_spearman = mean_spearman - mean_pearson
    flush_print(f"\n  Δ(Kendall - Pearson):  {delta_kendall:+.3f}")
    flush_print(f"  Δ(Spearman - Pearson): {delta_spearman:+.3f}")

    if delta_spearman > 0.1:
        flush_print(f"\n  Spearman ({mean_spearman:.3f}) > Pearson ({mean_pearson:.3f}):")
        flush_print("  → Nonlinear/scale effects partially suppress Pearson.")
        flush_print("  → Model recovers rank-order better than raw magnitudes.")
    if abs(delta_kendall) < 0.1:
        flush_print(f"\n  Kendall ({mean_kendall:.3f}) ≈ Pearson ({mean_pearson:.3f}):")
        flush_print("  → Pairwise concordance is not much better than linear fit.")
        flush_print("  → The gap is not purely a BT scale artifact.")

    flush_print("=" * 70)

    # Save results
    output_path = output_dir / "f4_rank_recovery.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
