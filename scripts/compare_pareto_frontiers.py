"""Compare trained stakeholder models against the diversity_weight Pareto frontier.

Answers: do trained models (Phase 4, cosine sim 0.478) reach utility tradeoffs
that a simple diversity knob cannot? Also compares old Phase 3 models (cosine sim ~0.999).

Additionally, combines learned scorers with the diversity knob to see if the
combination pushes the frontier outward.
"""

import importlib.util
import json
import os
import sys
import types

import numpy as np

# ---------------------------------------------------------------------------
# Module loading (same pattern as analyze_stakeholder_utilities.py)
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
NUM_ACTIONS = weights_mod.NUM_ACTIONS

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
compute_society_utility = stakeholder_mod.compute_society_utility
compute_pareto_frontier = stakeholder_mod.compute_pareto_frontier

# Reuse data generation from analyze_stakeholder_utilities
analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOP_K = 10


def build_base_action_probs(data: dict) -> np.ndarray:
    """Build [N, M, 18] user-content action probability matrix.

    Same logic as compute_and_analyze_pareto in analyze_stakeholder_utilities.py.
    """
    N = data["num_users"]
    M = data["num_content"]
    base = np.tile(data["content_action_probs"][np.newaxis, :, :], (N, 1, 1))

    for user_idx in range(N):
        archetype = data["user_archetypes"][user_idx]
        for content_idx in range(M):
            topic = data["content_topics"][content_idx]
            if topic == archetype:
                base[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 1.5
                base[user_idx, content_idx, ACTION_INDICES["repost"]] *= 1.3
            if archetype == 2 and topic == 3:
                base[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 0.3
                base[user_idx, content_idx, ACTION_INDICES["block_author"]] *= 2.0
            elif archetype == 3 and topic == 2:
                base[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 0.3
                base[user_idx, content_idx, ACTION_INDICES["block_author"]] *= 2.0

    return np.clip(base, 0, 1)


def evaluate_weight_vector(
    weight_vector: np.ndarray,
    base_action_probs: np.ndarray,
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    num_topics: int,
) -> dict[str, float]:
    """Evaluate a trained weight vector by ranking content per-user and computing utilities."""
    N = base_action_probs.shape[0]
    weight_vector = np.asarray(weight_vector, dtype=np.float64)

    recommendations = []
    for user_idx in range(N):
        user_scores = base_action_probs[user_idx] @ weight_vector  # [M]
        top_k_indices = np.argsort(user_scores)[-TOP_K:][::-1]
        recommendations.append(top_k_indices)

    recommendations = np.array(recommendations)

    # User utility: average over users
    user_utils = []
    for user_idx in range(N):
        rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
        result = compute_user_utility(rec_probs)
        user_utils.append(result.total_utility)

    # Platform utility: average over users
    platform_utils = []
    for user_idx in range(N):
        rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
        result = compute_platform_utility(rec_probs)
        platform_utils.append(result.total_utility)

    # Society utility
    society_result = compute_society_utility(
        recommendations, user_archetypes, content_topics, num_topics
    )

    return {
        "user_utility": float(np.mean(user_utils)),
        "platform_utility": float(np.mean(platform_utils)),
        "society_utility": float(society_result.total_utility),
    }


def compute_learned_frontier(
    weight_vector: np.ndarray,
    base_action_probs: np.ndarray,
    diversity_weights: list[float],
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = 10,
) -> list[dict]:
    """Compute Pareto frontier using learned weights as scorer + diversity knob.

    Same greedy diversity-aware selection as compute_pareto_frontier(), but
    replaces the hardcoded engagement formula with learned weight vector.
    """
    N, M, _ = base_action_probs.shape
    num_topics = len(np.unique(content_topics))
    weight_vector = np.asarray(weight_vector, dtype=np.float64)

    points = []

    for div_weight in diversity_weights:
        recommendations = []

        for user_idx in range(N):
            # Score using LEARNED weights
            engagement_scores = base_action_probs[user_idx] @ weight_vector  # [M]

            if div_weight > 0:
                # Greedy selection with diversity bonus (same logic as stakeholder_utilities.py)
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
                        score = (1 - div_weight) * engagement_scores[idx] + div_weight * diversity_bonus
                        adjusted_scores.append((idx, score))

                    best_idx, _ = max(adjusted_scores, key=lambda x: x[1])
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    topic_counts[content_topics[best_idx]] += 1

                recommendations.append(selected)
            else:
                top_indices = np.argsort(engagement_scores)[-top_k:][::-1]
                recommendations.append(top_indices.tolist())

        recommendations = np.array(recommendations)

        # Compute utilities
        user_utils = []
        platform_utils = []
        for user_idx in range(N):
            rec_probs = base_action_probs[user_idx, recommendations[user_idx]]
            user_utils.append(compute_user_utility(rec_probs).total_utility)
            platform_utils.append(compute_platform_utility(rec_probs).total_utility)

        society_result = compute_society_utility(
            recommendations, user_archetypes, content_topics, num_topics
        )

        points.append({
            "diversity_weight": div_weight,
            "user_utility": float(np.mean(user_utils)),
            "platform_utility": float(np.mean(platform_utils)),
            "society_utility": float(society_result.total_utility),
        })

    return points


def is_dominated(point: dict[str, float], frontier_points: list[dict]) -> tuple[bool, list[int]]:
    """Check if a point is dominated by any frontier point.

    Returns (is_dominated, list_of_dominating_indices).
    """
    dominating = []
    for i, fp in enumerate(frontier_points):
        if (
            fp["user_utility"] >= point["user_utility"]
            and fp["platform_utility"] >= point["platform_utility"]
            and fp["society_utility"] >= point["society_utility"]
            and (
                fp["user_utility"] > point["user_utility"]
                or fp["platform_utility"] > point["platform_utility"]
                or fp["society_utility"] > point["society_utility"]
            )
        ):
            dominating.append(i)
    return len(dominating) > 0, dominating


def classify_vs_frontier(
    point: dict[str, float], frontier_points: list[dict]
) -> str:
    """Classify a point as 'dominated', 'on_frontier', or 'beyond_frontier'."""
    dominated, _ = is_dominated(point, frontier_points)
    if dominated:
        return "dominated"

    # Check if this point dominates any frontier point
    for fp in frontier_points:
        if (
            point["user_utility"] >= fp["user_utility"]
            and point["platform_utility"] >= fp["platform_utility"]
            and point["society_utility"] >= fp["society_utility"]
            and (
                point["user_utility"] > fp["user_utility"]
                or point["platform_utility"] > fp["platform_utility"]
                or point["society_utility"] > fp["society_utility"]
            )
        ):
            return "beyond_frontier"

    return "on_frontier"


def convert_to_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Loading model weights
# ---------------------------------------------------------------------------

def load_loss_experiment(filename: str) -> np.ndarray:
    path = os.path.join(project_root, "results/loss_experiments", filename)
    with open(path) as f:
        data = json.load(f)
    return np.array(data["weights_vector"])


def load_phase3_weights() -> dict[str, np.ndarray]:
    path = os.path.join(project_root, "outputs/stakeholder_comparison_results.json")
    with open(path) as f:
        data = json.load(f)
    return {
        s: np.array(data["model_weights"][s])
        for s in ["user", "platform", "society"]
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

FRONTIER_STYLES = {
    "Hardcoded scorer": {"color": "black", "linestyle": "-", "marker": "o", "markersize": 5},
    "Learned user scorer": {"color": "#e74c3c", "linestyle": "--", "marker": "s", "markersize": 5},
    "Learned platform scorer": {"color": "#3498db", "linestyle": "--", "marker": "D", "markersize": 5},
    "Learned society scorer": {"color": "#2ecc71", "linestyle": "--", "marker": "^", "markersize": 5},
}


def plot_combined_frontiers(
    hardcoded_frontier: list[dict],
    learned_frontiers: dict[str, list[dict]],
) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    dims = [
        ("user_utility", "platform_utility", "User Utility", "Platform Utility"),
        ("user_utility", "society_utility", "User Utility", "Society Utility"),
        ("platform_utility", "society_utility", "Platform Utility", "Society Utility"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        "Learned Scorer + Diversity Knob vs Hardcoded Scorer + Diversity Knob",
        fontsize=13, fontweight="bold", y=1.02,
    )

    # Map learned frontier keys to style keys
    style_map = {
        "user": "Learned user scorer",
        "platform": "Learned platform scorer",
        "society": "Learned society scorer",
    }

    all_curves = [("Hardcoded scorer", hardcoded_frontier)]
    for stakeholder, points in learned_frontiers.items():
        all_curves.append((style_map[stakeholder], points))

    for ax, (xdim, ydim, xlabel, ylabel) in zip(axes, dims):
        for curve_name, points in all_curves:
            style = FRONTIER_STYLES[curve_name]
            fx = [p[xdim] for p in points]
            fy = [p[ydim] for p in points]
            order = np.argsort(fx)
            fx_sorted = [fx[i] for i in order]
            fy_sorted = [fy[i] for i in order]

            ax.plot(
                fx_sorted, fy_sorted,
                color=style["color"], linestyle=style["linestyle"],
                linewidth=1.5, zorder=2,
            )
            ax.scatter(
                fx_sorted, fy_sorted,
                c=style["color"], marker=style["marker"],
                s=style["markersize"] ** 2, zorder=3,
                edgecolors="black" if curve_name == "Hardcoded scorer" else "none",
                linewidths=0.5,
            )

            # Annotate diversity weights on hardcoded frontier only
            if curve_name == "Hardcoded scorer":
                dw_sorted = [points[i]["diversity_weight"] for i in order]
                for x, y, dw in zip(fx_sorted, fy_sorted, dw_sorted):
                    ax.annotate(
                        f"{dw:.1f}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color="0.4",
                    )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Legend
    handles = []
    for name, style in FRONTIER_STYLES.items():
        handles.append(
            mlines.Line2D(
                [], [], color=style["color"], marker=style["marker"],
                markersize=7, linewidth=1.5, linestyle=style["linestyle"],
                markeredgecolor="black" if name == "Hardcoded scorer" else "none",
                markeredgewidth=0.5,
                label=name,
            )
        )
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout()
    plot_path = os.path.join(project_root, "results/pareto_comparison.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PARETO FRONTIER: Learned Scorer + Diversity Knob")
    print("=" * 70)

    # 1. Generate shared evaluation data
    print("\n[1/4] Generating synthetic data (seed=42)...")
    data = generate_synthetic_data(num_users=600, num_content=100, num_topics=6, seed=42)
    base_action_probs = build_base_action_probs(data)
    print(f"  Users: {data['num_users']}, Content: {data['num_content']}, Topics: {data['num_topics']}")

    diversity_weights = [round(x * 0.1, 1) for x in range(11)]

    # 2. Compute hardcoded-scorer frontier (baseline)
    print("\n[2/4] Computing hardcoded-scorer frontier (favorite + 0.8*repost + 0.5*follow)...")
    hardcoded_frontier_raw = compute_pareto_frontier(
        base_action_probs,
        diversity_weights,
        data["user_archetypes"],
        data["content_topics"],
        top_k=TOP_K,
    )
    hardcoded_frontier = [
        {
            "diversity_weight": p.diversity_weight,
            "user_utility": p.user_utility,
            "platform_utility": p.platform_utility,
            "society_utility": p.society_utility,
        }
        for p in hardcoded_frontier_raw
    ]

    # 3. Load BT baseline learned weights and compute learned frontiers
    print("\n[3/4] Loading BT baseline weights & computing learned frontiers...")
    bt_weights = {
        "user": load_loss_experiment("bradley_terry_user.json"),
        "platform": load_loss_experiment("bradley_terry_platform.json"),
        "society": load_loss_experiment("bradley_terry_society.json"),
    }

    learned_frontiers = {}
    for stakeholder, wv in bt_weights.items():
        print(f"  Computing frontier for {stakeholder}-trained scorer...")
        learned_frontiers[stakeholder] = compute_learned_frontier(
            wv, base_action_probs, diversity_weights,
            data["user_archetypes"], data["content_topics"],
            top_k=TOP_K,
        )

    # 4. Print comparison tables
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- Hardcoded Scorer Frontier ---")
    print(f"{'Div Wt':<8} {'User':>10} {'Platform':>10} {'Society':>10}")
    print("-" * 42)
    for fp in hardcoded_frontier:
        print(
            f"{fp['diversity_weight']:<8.1f} "
            f"{fp['user_utility']:>10.4f} "
            f"{fp['platform_utility']:>10.4f} "
            f"{fp['society_utility']:>10.4f}"
        )

    for stakeholder in ["user", "platform", "society"]:
        print(f"\n--- Learned {stakeholder.title()} Scorer + Diversity Knob ---")
        print(f"{'Div Wt':<8} {'User':>10} {'Platform':>10} {'Society':>10}   {'vs hardcoded':>12}")
        print("-" * 58)
        for lp in learned_frontiers[stakeholder]:
            classification = classify_vs_frontier(lp, hardcoded_frontier)
            print(
                f"{lp['diversity_weight']:<8.1f} "
                f"{lp['user_utility']:>10.4f} "
                f"{lp['platform_utility']:>10.4f} "
                f"{lp['society_utility']:>10.4f}   "
                f"{classification:>12}"
            )

    # Dominance summary
    print("\n--- Dominance Summary (learned vs hardcoded frontier) ---")
    for stakeholder in ["user", "platform", "society"]:
        beyond = sum(
            1 for lp in learned_frontiers[stakeholder]
            if classify_vs_frontier(lp, hardcoded_frontier) == "beyond_frontier"
        )
        on = sum(
            1 for lp in learned_frontiers[stakeholder]
            if classify_vs_frontier(lp, hardcoded_frontier) == "on_frontier"
        )
        dominated_count = len(learned_frontiers[stakeholder]) - beyond - on
        print(f"  {stakeholder:>10} scorer: {beyond} beyond, {on} on, {dominated_count} dominated  (of {len(diversity_weights)} points)")

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------
    print("\n[4/4] Generating plot...")
    plot_combined_frontiers(hardcoded_frontier, learned_frontiers)

    # Save results
    output = convert_to_serializable({
        "hardcoded_frontier": hardcoded_frontier,
        "learned_frontiers": learned_frontiers,
        "config": {
            "num_users": data["num_users"],
            "num_content": data["num_content"],
            "num_topics": data["num_topics"],
            "top_k": TOP_K,
            "seed": 42,
            "diversity_weights": diversity_weights,
            "scorer": "BT baseline (bradley_terry_{user,platform,society}.json)",
        },
    })

    output_path = os.path.join(project_root, "results/pareto_comparison.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
