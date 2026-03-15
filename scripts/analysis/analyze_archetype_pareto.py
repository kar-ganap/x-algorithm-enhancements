#!/usr/bin/env python3
"""Per-archetype Pareto frontier analysis for F4 reward modeling.

Answers: does the learned BT scorer benefit all 6 user archetypes, or does
improvement concentrate in some groups? Slices the existing Pareto comparison
by archetype to check for distributional fairness.

Usage:
    uv run python scripts/analyze_archetype_pareto.py
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# Module loading (same pattern as compare_pareto_frontiers.py)
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def load_module_direct(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Create package structure
enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg

reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [os.path.join(project_root, "enhancements/reward_modeling")]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

weights_mod = load_module_direct(
    "enhancements.reward_modeling.weights",
    os.path.join(project_root, "enhancements/reward_modeling/weights.py"),
)
ACTION_INDICES = weights_mod.ACTION_INDICES

load_module_direct(
    "enhancements.reward_modeling.pluralistic",
    os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py"),
)

stakeholder_mod = load_module_direct(
    "enhancements.reward_modeling.stakeholder_utilities",
    os.path.join(project_root, "enhancements/reward_modeling/stakeholder_utilities.py"),
)
compute_user_utility = stakeholder_mod.compute_user_utility
compute_platform_utility = stakeholder_mod.compute_platform_utility

# Reuse data generation and base_action_probs builder
analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analysis/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

pareto_mod = load_module_direct(
    "compare_pareto_frontiers",
    os.path.join(project_root, "scripts/evaluation/compare_pareto_frontiers.py"),
)
build_base_action_probs = pareto_mod.build_base_action_probs


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Archetype mapping matching data generator (analyze_stakeholder_utilities.py:103)
ARCHETYPE_NAMES = {
    0: "sports_fan",
    1: "tech_bro",
    2: "political_L",
    3: "political_R",
    4: "lurker",
    5: "power_user",
}

TOP_K = 10


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def group_utilities_by_archetype(
    user_utilities: np.ndarray,
    platform_utilities: np.ndarray,
    user_archetypes: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Group per-user utilities by archetype, return mean per group."""
    result = {}
    for arch_idx, arch_name in ARCHETYPE_NAMES.items():
        mask = user_archetypes == arch_idx
        count = int(mask.sum())
        if count == 0:
            continue
        result[arch_name] = {
            "user_utility": float(np.mean(user_utilities[mask])),
            "platform_utility": float(np.mean(platform_utilities[mask])),
            "count": count,
        }
    return result


def rerank_content(
    base_action_probs: np.ndarray,
    content_topics: np.ndarray,
    diversity_weight: float,
    engagement_scores_all: np.ndarray,
    top_k: int = 10,
) -> np.ndarray:
    """Rerank content per user using engagement scores + diversity bonus.

    Args:
        base_action_probs: [N, M, 18]
        content_topics: [M]
        diversity_weight: 0.0 to 1.0
        engagement_scores_all: [N, M] per-user engagement scores
        top_k: number of recommendations

    Returns:
        [N, top_k] recommendation indices
    """
    N, M, _ = base_action_probs.shape
    num_topics = len(np.unique(content_topics))
    recommendations = []

    for user_idx in range(N):
        engagement_scores = engagement_scores_all[user_idx]

        if diversity_weight > 0:
            selected = []
            remaining = list(range(M))
            topic_counts = np.zeros(num_topics)

            for _ in range(top_k):
                if not remaining:
                    break
                adjusted_scores = []
                for idx in remaining:
                    topic = content_topics[idx]
                    diversity_bonus = 1.0 / (topic_counts[topic] + 1)
                    score = (1 - diversity_weight) * engagement_scores[idx] + diversity_weight * diversity_bonus
                    adjusted_scores.append((idx, score))

                best_idx, _ = max(adjusted_scores, key=lambda x: x[1])
                selected.append(best_idx)
                remaining.remove(best_idx)
                topic_counts[content_topics[best_idx]] += 1

            recommendations.append(selected)
        else:
            top_indices = np.argsort(engagement_scores)[-top_k:][::-1]
            recommendations.append(top_indices.tolist())

    return np.array(recommendations)


def compute_hardcoded_scores(base_action_probs: np.ndarray) -> np.ndarray:
    """Compute hardcoded engagement scores: favorite + 0.8*repost + 0.5*follow_author.

    Returns: [N, M] scores
    """
    return (
        base_action_probs[:, :, ACTION_INDICES["favorite"]]
        + base_action_probs[:, :, ACTION_INDICES["repost"]] * 0.8
        + base_action_probs[:, :, ACTION_INDICES["follow_author"]] * 0.5
    )


def compute_learned_scores(
    base_action_probs: np.ndarray,
    weight_vector: np.ndarray,
) -> np.ndarray:
    """Compute learned engagement scores: action_probs @ weights.

    Returns: [N, M] scores
    """
    weight_vector = np.asarray(weight_vector, dtype=np.float64)
    return base_action_probs @ weight_vector  # [N, M]


def compute_per_user_utilities(
    base_action_probs: np.ndarray,
    recommendations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-user user and platform utilities.

    Returns: (user_utilities[N], platform_utilities[N])
    """
    N = recommendations.shape[0]
    user_utils = np.zeros(N, dtype=np.float64)
    platform_utils = np.zeros(N, dtype=np.float64)

    for user_idx in range(N):
        rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
        user_utils[user_idx] = compute_user_utility(rec_probs).total_utility
        platform_utils[user_idx] = compute_platform_utility(rec_probs).total_utility

    return user_utils, platform_utils


def compute_archetype_frontier(
    base_action_probs: np.ndarray,
    diversity_weights: list[float],
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    engagement_scores: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """Compute per-archetype utilities at each diversity weight.

    Returns list of dicts with aggregate and per_archetype breakdowns.
    """
    points = []

    for div_weight in diversity_weights:
        recs = rerank_content(
            base_action_probs, content_topics, div_weight, engagement_scores, top_k
        )
        user_utils, platform_utils = compute_per_user_utilities(base_action_probs, recs)
        grouped = group_utilities_by_archetype(user_utils, platform_utils, user_archetypes)

        points.append({
            "diversity_weight": div_weight,
            "aggregate": {
                "user_utility": float(np.mean(user_utils)),
                "platform_utility": float(np.mean(platform_utils)),
            },
            "per_archetype": grouped,
        })

    return points


def compute_archetype_deltas(
    hardcoded: list[dict],
    learned: list[dict],
) -> dict[str, dict[str, float]]:
    """For each archetype, compute improvement stats of learned vs hardcoded."""
    deltas: dict[str, list[float]] = {}

    for h_point, l_point in zip(hardcoded, learned):
        for arch_name in h_point["per_archetype"]:
            if arch_name not in l_point["per_archetype"]:
                continue
            delta = (
                l_point["per_archetype"][arch_name]["user_utility"]
                - h_point["per_archetype"][arch_name]["user_utility"]
            )
            deltas.setdefault(arch_name, []).append(delta)

    result = {}
    for arch_name, delta_list in deltas.items():
        arr = np.array(delta_list)
        result[arch_name] = {
            "mean_delta": float(np.mean(arr)),
            "max_improvement": float(np.max(arr)),
            "max_degradation": float(np.min(arr)),
        }
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    flush_print("=" * 70)
    flush_print("F4 Per-Archetype Value Distribution Analysis")
    flush_print("=" * 70)
    flush_print("\nQuestion: Does the learned BT scorer benefit all archetypes?")

    # 1. Generate data (same config as compare_pareto_frontiers.py)
    flush_print("\n[1/4] Generating synthetic data (seed=42)...")
    data = generate_synthetic_data(num_users=600, num_content=100, num_topics=6, seed=42)
    base_action_probs = build_base_action_probs(data)
    user_archetypes = data["user_archetypes"]
    content_topics = data["content_topics"]
    flush_print(f"  Users: {data['num_users']}, Content: {data['num_content']}")

    diversity_weights = [round(x * 0.1, 1) for x in range(11)]

    # 2. Hardcoded frontier (per-archetype)
    flush_print("\n[2/4] Computing hardcoded frontier (per-archetype)...")
    hardcoded_scores = compute_hardcoded_scores(base_action_probs)
    hardcoded_frontier = compute_archetype_frontier(
        base_action_probs, diversity_weights, user_archetypes, content_topics,
        hardcoded_scores, TOP_K,
    )

    # 3. Learned frontiers
    flush_print("\n[3/4] Computing learned frontiers (per-archetype)...")
    learned_frontiers = {}
    all_deltas = {}

    for stakeholder in ["user", "platform", "society"]:
        bt_path = f"results/loss_experiments/bradley_terry_{stakeholder}.json"
        if not os.path.exists(bt_path):
            flush_print(f"  Warning: {bt_path} not found, skipping")
            continue

        with open(bt_path) as f:
            bt_data = json.load(f)
        wv = np.array(bt_data["weights_vector"], dtype=np.float64)

        learned_scores = compute_learned_scores(base_action_probs, wv)
        frontier = compute_archetype_frontier(
            base_action_probs, diversity_weights, user_archetypes, content_topics,
            learned_scores, TOP_K,
        )
        learned_frontiers[stakeholder] = frontier
        all_deltas[stakeholder] = compute_archetype_deltas(hardcoded_frontier, frontier)
        flush_print(f"  {stakeholder}: done")

    # 4. Print results
    flush_print("\n[4/4] Results")
    flush_print("=" * 70)

    # Print per-archetype table at div_weight=0.1 (typical operating point)
    for stakeholder, frontier in learned_frontiers.items():
        flush_print(f"\n--- {stakeholder.upper()} scorer vs hardcoded (div_weight=0.1) ---")
        h_point = hardcoded_frontier[1]  # div_weight=0.1
        l_point = frontier[1]

        flush_print(f"{'Archetype':<15} {'Hardcoded':>10} {'Learned':>10} {'Delta':>10} {'Δ%':>8}")
        flush_print("-" * 58)

        for arch_name in ARCHETYPE_NAMES.values():
            if arch_name not in h_point["per_archetype"]:
                continue
            h_val = h_point["per_archetype"][arch_name]["user_utility"]
            l_val = l_point["per_archetype"][arch_name]["user_utility"]
            delta = l_val - h_val
            pct = (delta / abs(h_val) * 100) if h_val != 0 else 0.0
            flush_print(f"  {arch_name:<13} {h_val:>10.4f} {l_val:>10.4f} {delta:>+10.4f} {pct:>+7.1f}%")

        h_agg = h_point["aggregate"]["user_utility"]
        l_agg = l_point["aggregate"]["user_utility"]
        delta_agg = l_agg - h_agg
        pct_agg = (delta_agg / abs(h_agg) * 100) if h_agg != 0 else 0.0
        flush_print("-" * 58)
        flush_print(f"  {'AGGREGATE':<13} {h_agg:>10.4f} {l_agg:>10.4f} {delta_agg:>+10.4f} {pct_agg:>+7.1f}%")

    # Summary: which archetypes benefit/lose
    flush_print("\n" + "=" * 70)
    flush_print("SUMMARY (user scorer, mean across all diversity weights)")
    flush_print("=" * 70)

    if "user" in all_deltas:
        user_deltas = all_deltas["user"]
        winners = [a for a, d in user_deltas.items() if d["mean_delta"] > 0.001]
        losers = [a for a, d in user_deltas.items() if d["mean_delta"] < -0.001]
        neutral = [a for a, d in user_deltas.items() if abs(d["mean_delta"]) <= 0.001]

        flush_print(f"\n{'Archetype':<15} {'Mean Δ':>10} {'Best':>10} {'Worst':>10}")
        flush_print("-" * 50)
        for arch_name in ARCHETYPE_NAMES.values():
            if arch_name in user_deltas:
                d = user_deltas[arch_name]
                flush_print(
                    f"  {arch_name:<13} {d['mean_delta']:>+10.4f} "
                    f"{d['max_improvement']:>+10.4f} {d['max_degradation']:>+10.4f}"
                )

        flush_print(f"\n  Winners  ({len(winners)}): {', '.join(winners) if winners else 'none'}")
        flush_print(f"  Losers   ({len(losers)}): {', '.join(losers) if losers else 'none'}")
        flush_print(f"  Neutral  ({len(neutral)}): {', '.join(neutral) if neutral else 'none'}")

        benefits_all = len(losers) == 0
        flush_print(f"\n  Benefits all archetypes: {'YES' if benefits_all else 'NO'}")

    flush_print("=" * 70)

    # Save results
    results = {
        "config": {
            "num_users": 600,
            "num_content": 100,
            "num_topics": 6,
            "top_k": TOP_K,
            "seed": 42,
            "diversity_weights": diversity_weights,
            "archetype_mapping": ARCHETYPE_NAMES,
        },
        "hardcoded_frontier": hardcoded_frontier,
        "learned_frontiers": learned_frontiers,
        "deltas": all_deltas,
    }

    output_path = "results/archetype_pareto_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
