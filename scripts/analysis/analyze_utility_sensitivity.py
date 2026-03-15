#!/usr/bin/env python3
"""Direction 2: Utility function sensitivity analysis.

Phase 1: α-dominance test — does within-group weight shuffling
(preserving pos/neg ratio) affect the Pareto frontier?

Usage:
    uv run python scripts/analyze_utility_sensitivity.py
"""

import importlib.util
import json
import os
import sys
import time
import types
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Module loading
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


enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg
reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [
    os.path.join(project_root, "enhancements/reward_modeling")
]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

load_module_direct(
    "enhancements.reward_modeling.weights",
    os.path.join(project_root, "enhancements/reward_modeling/weights.py"),
)
load_module_direct(
    "enhancements.reward_modeling.pluralistic",
    os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py"),
)
load_module_direct(
    "enhancements.reward_modeling.stakeholder_utilities",
    os.path.join(project_root, "enhancements/reward_modeling/stakeholder_utilities.py"),
)
alt_losses = load_module_direct(
    "enhancements.reward_modeling.alternative_losses",
    os.path.join(project_root, "enhancements/reward_modeling/alternative_losses.py"),
)
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss
POSITIVE_INDICES = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alt_losses.NEGATIVE_INDICES
NUM_ACTIONS = alt_losses.NUM_ACTIONS

loss_exp_mod = load_module_direct(
    "run_loss_experiments",
    os.path.join(project_root, "scripts/experiments/run_loss_experiments.py"),
)
generate_content_pool = loss_exp_mod.generate_content_pool

frontier_mod = load_module_direct(
    "enhancements.reward_modeling.k_stakeholder_frontier",
    os.path.join(project_root, "enhancements/reward_modeling/k_stakeholder_frontier.py"),
)
compute_k_frontier = frontier_mod.compute_k_frontier
compute_scorer_eval_frontier = frontier_mod.compute_scorer_eval_frontier

partial_obs_mod = load_module_direct(
    "analyze_partial_observation",
    os.path.join(project_root, "scripts/analysis/analyze_partial_observation.py"),
)
build_base_action_probs = partial_obs_mod.build_base_action_probs

analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analysis/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

stakeholder_mod = load_module_direct(
    "enhancements.reward_modeling.stakeholder_utilities",
    os.path.join(project_root, "enhancements/reward_modeling/stakeholder_utilities.py"),
)
UtilityWeights = stakeholder_mod.UtilityWeights

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_SEED = 42
N_SEEDS = 5
N_SHUFFLES = 20
ALPHA_PERTURBATION_LEVELS = [0.5, 0.8, 1.2, 1.5, 2.0]
NUM_USERS = 600
NUM_CONTENT = 100
NUM_TOPICS = 6
DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10
N_PAIRS = 2000
NUM_EPOCHS = 50
LEARNING_RATE = 0.01

OUTPUT_PATH = os.path.join(project_root, "results/utility_sensitivity.json")


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Utility weight helpers
# ---------------------------------------------------------------------------


def default_weight_vectors() -> dict[str, np.ndarray]:
    """Extract default UtilityWeights as 18-dim weight vectors."""
    uw = UtilityWeights.default()

    w_user = np.zeros(NUM_ACTIONS)
    for action, weight in uw.user_positive.items():
        w_user[alt_losses.ACTION_INDICES[action]] = weight
    for action, weight in uw.user_negative.items():
        w_user[alt_losses.ACTION_INDICES[action]] = weight

    w_platform = np.zeros(NUM_ACTIONS)
    for action, weight in uw.platform_weights.items():
        w_platform[alt_losses.ACTION_INDICES[action]] = weight

    # Society: pos - 4*neg (structural proxy)
    w_society = np.zeros(NUM_ACTIONS)
    for idx in POSITIVE_INDICES:
        w_society[idx] = 1.0
    for idx in NEGATIVE_INDICES:
        w_society[idx] = -4.0

    return {"user": w_user, "platform": w_platform, "society": w_society}


def shuffle_within_groups(
    w: np.ndarray, rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle weights within positive and negative groups, preserving α."""
    w_new = w.copy()
    pos_vals = w[POSITIVE_INDICES].copy()
    neg_vals = w[NEGATIVE_INDICES].copy()
    rng.shuffle(pos_vals)
    rng.shuffle(neg_vals)
    w_new[POSITIVE_INDICES] = pos_vals
    w_new[NEGATIVE_INDICES] = neg_vals
    return w_new


def scale_alpha(w: np.ndarray, factor: float) -> np.ndarray:
    """Scale the neg/pos ratio (α) by a factor."""
    w_new = w.copy()
    w_new[NEGATIVE_INDICES] *= factor
    return w_new


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def hausdorff_distance(
    frontier_a: list[dict[str, float]],
    frontier_b: list[dict[str, float]],
    dims: list[str],
) -> float:
    """Hausdorff distance between two frontiers in the given dimensions."""
    if not frontier_a or not frontier_b:
        return 0.0

    def point_to_vec(p: dict[str, float]) -> np.ndarray:
        return np.array([p[d] for d in dims])

    vecs_a = np.array([point_to_vec(p) for p in frontier_a])
    vecs_b = np.array([point_to_vec(p) for p in frontier_b])

    # Directed Hausdorff: max over A of min distance to B
    def directed(x: np.ndarray, y: np.ndarray) -> float:
        max_dist = 0.0
        for xi in x:
            dists = np.linalg.norm(y - xi, axis=1)
            max_dist = max(max_dist, float(np.min(dists)))
        return max_dist

    return max(directed(vecs_a, vecs_b), directed(vecs_b, vecs_a))


def misspecification_regret(
    frontier_wrong: list[dict[str, float]],
    frontier_true: list[dict[str, float]],
    dims: list[str],
) -> float:
    """Regret from acting on the wrong frontier.

    For each operating point on frontier_wrong, find the closest point
    on frontier_true (by observed dims) and measure the utility gap.
    """
    if not frontier_wrong or not frontier_true:
        return 0.0

    # Average utility gap across all dimensions
    total_gap = 0.0
    for pw in frontier_wrong:
        # Find matching dw on true frontier
        best_match = min(
            frontier_true,
            key=lambda pt: abs(pt["diversity_weight"] - pw["diversity_weight"]),
        )
        for d in dims:
            total_gap += abs(pw[d] - best_match[d])

    return total_gap / (len(frontier_wrong) * len(dims))


# ---------------------------------------------------------------------------
# α-Dominance Test
# ---------------------------------------------------------------------------


def run_alpha_dominance_test() -> dict[str, Any]:
    """Phase 1: Test whether α (pos/neg ratio) dominates utility sensitivity."""
    flush_print("=" * 60)
    flush_print("DIRECTION 2: α-DOMINANCE TEST")
    flush_print("=" * 60)
    flush_print(f"  Within-group shuffles: {N_SHUFFLES}")
    flush_print(f"  α perturbation levels: {ALPHA_PERTURBATION_LEVELS}")
    flush_print(f"  Seeds: {N_SEEDS}")

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]

    within_distances: list[float] = []
    between_distances: list[float] = []
    within_regrets: list[float] = []
    between_regrets: list[float] = []

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]

        # Baseline frontier
        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        # Within-group shuffles (α-preserving)
        rng = np.random.default_rng(seed + 999)
        for _ in range(N_SHUFFLES):
            shuffled_weights = {
                name: shuffle_within_groups(baseline_weights[name], rng)
                for name in stakeholder_names
            }
            shuffled_frontier = compute_k_frontier(
                base_probs, topics, shuffled_weights,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            hd = hausdorff_distance(
                baseline_frontier, shuffled_frontier, utility_dims,
            )
            mr = misspecification_regret(
                shuffled_frontier, baseline_frontier, utility_dims,
            )
            within_distances.append(hd)
            within_regrets.append(mr)

        # Between-group (α-changing)
        for alpha_factor in ALPHA_PERTURBATION_LEVELS:
            perturbed_weights = {
                name: scale_alpha(baseline_weights[name], alpha_factor)
                for name in stakeholder_names
            }
            perturbed_frontier = compute_k_frontier(
                base_probs, topics, perturbed_weights,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            hd = hausdorff_distance(
                baseline_frontier, perturbed_frontier, utility_dims,
            )
            mr = misspecification_regret(
                perturbed_frontier, baseline_frontier, utility_dims,
            )
            between_distances.append(hd)
            between_regrets.append(mr)

        flush_print(f"    seed {seed_idx + 1}/{N_SEEDS} done")

    # Compute α-dominance ratio
    var_within = float(np.var(within_distances))
    var_between = float(np.var(between_distances))
    dominance_ratio = var_within / var_between if var_between > 1e-10 else 0.0

    mean_within_dist = float(np.mean(within_distances))
    mean_between_dist = float(np.mean(between_distances))
    mean_within_regret = float(np.mean(within_regrets))
    mean_between_regret = float(np.mean(between_regrets))

    alpha_dominates = dominance_ratio < 0.1

    # Summary
    flush_print("\n" + "=" * 60)
    flush_print("RESULTS")
    flush_print("=" * 60)
    flush_print(f"\n  Within-group (α-preserving) shuffle:")
    flush_print(f"    Hausdorff distance: {mean_within_dist:.4f} ± {np.std(within_distances):.4f}")
    flush_print(f"    Misspec regret:     {mean_within_regret:.4f} ± {np.std(within_regrets):.4f}")
    flush_print(f"\n  Between-group (α-changing) perturbation:")
    flush_print(f"    Hausdorff distance: {mean_between_dist:.4f} ± {np.std(between_distances):.4f}")
    flush_print(f"    Misspec regret:     {mean_between_regret:.4f} ± {np.std(between_regrets):.4f}")
    flush_print(f"\n  α-dominance ratio (within_var / between_var): {dominance_ratio:.4f}")
    flush_print(f"  Mean distance ratio (within / between): {mean_within_dist / mean_between_dist:.4f}" if mean_between_dist > 0 else "")
    flush_print(f"  α DOMINATES: {alpha_dominates}")

    if alpha_dominates:
        flush_print(
            "\n  → Individual action weights don't matter. "
            "The pos/neg ratio (α) is sufficient."
        )
        flush_print(
            "  → Structure-without-weights transparency is meaningful: "
            "knowing action types suffices for audit."
        )
    else:
        flush_print(
            "\n  → Individual action weights matter. "
            "α alone is insufficient."
        )
        flush_print(
            "  → Proceed to full parameter sweep (Angle A)."
        )

    return {
        "within_shuffle": {
            "n_samples": len(within_distances),
            "hausdorff_mean": mean_within_dist,
            "hausdorff_std": float(np.std(within_distances)),
            "regret_mean": mean_within_regret,
            "regret_std": float(np.std(within_regrets)),
            "variance": var_within,
        },
        "between_alpha": {
            "n_samples": len(between_distances),
            "alpha_levels": ALPHA_PERTURBATION_LEVELS,
            "hausdorff_mean": mean_between_dist,
            "hausdorff_std": float(np.std(between_distances)),
            "regret_mean": mean_between_regret,
            "regret_std": float(np.std(between_regrets)),
            "variance": var_between,
        },
        "alpha_dominance_ratio": dominance_ratio,
        "alpha_dominates": alpha_dominates,
        "mean_distance_ratio": (
            mean_within_dist / mean_between_dist
            if mean_between_dist > 0 else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Strengthening: magnitude perturbation (#1)
# ---------------------------------------------------------------------------

MAGNITUDE_SIGMAS = [0.1, 0.2, 0.3, 0.5]


def perturb_magnitudes_preserve_alpha(
    w: np.ndarray, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Perturb individual weight magnitudes while preserving α."""
    w_new = w.copy()
    noise = 1 + rng.normal(0, sigma, len(w))
    w_new *= noise
    # Renormalize groups to preserve α
    pos_orig_mean = np.mean(w[POSITIVE_INDICES])
    neg_orig_mean = np.mean(w[NEGATIVE_INDICES])
    pos_new_mean = np.mean(w_new[POSITIVE_INDICES])
    neg_new_mean = np.mean(w_new[NEGATIVE_INDICES])
    if abs(pos_new_mean) > 1e-10:
        w_new[POSITIVE_INDICES] *= pos_orig_mean / pos_new_mean
    if abs(neg_new_mean) > 1e-10:
        w_new[NEGATIVE_INDICES] *= neg_orig_mean / neg_new_mean
    return w_new


def run_magnitude_perturbation_test() -> dict[str, Any]:
    """Criticism #1: magnitude perturbation (not just permutation)."""
    flush_print("\n" + "=" * 60)
    flush_print("STRENGTHENING #1: MAGNITUDE PERTURBATION")
    flush_print("=" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]

    results_by_sigma: dict[float, list[float]] = {s: [] for s in MAGNITUDE_SIGMAS}

    for seed in seeds:
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]

        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        rng = np.random.default_rng(seed + 888)
        for sigma in MAGNITUDE_SIGMAS:
            for _ in range(N_SHUFFLES):
                perturbed = {
                    name: perturb_magnitudes_preserve_alpha(
                        baseline_weights[name], sigma, rng,
                    )
                    for name in stakeholder_names
                }
                frontier = compute_k_frontier(
                    base_probs, topics, perturbed,
                    DIVERSITY_WEIGHTS, TOP_K,
                )
                hd = hausdorff_distance(
                    baseline_frontier, frontier, utility_dims,
                )
                results_by_sigma[sigma].append(hd)

    flush_print("\n  Magnitude perturbation (α-preserving):")
    for sigma in MAGNITUDE_SIGMAS:
        vals = results_by_sigma[sigma]
        flush_print(
            f"    σ={sigma:.1f}: Hausdorff={np.mean(vals):.4f}"
            f" ± {np.std(vals):.4f}"
        )

    return {
        str(sigma): {
            "hausdorff_mean": float(np.mean(vals)),
            "hausdorff_std": float(np.std(vals)),
            "n_samples": len(vals),
        }
        for sigma, vals in results_by_sigma.items()
    }


# ---------------------------------------------------------------------------
# Strengthening: selection-level sensitivity (#2)
# ---------------------------------------------------------------------------


def run_selection_level_test() -> dict[str, Any]:
    """Criticism #2: perturbed weights drive SELECTION, not just evaluation."""
    flush_print("\n" + "=" * 60)
    flush_print("STRENGTHENING #2: SELECTION-LEVEL SENSITIVITY")
    flush_print("=" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]

    # Build a combined scorer from baseline weights (average)
    baseline_scorer = np.mean(
        [baseline_weights[n] for n in stakeholder_names], axis=0,
    )

    within_distances: list[float] = []
    between_distances: list[float] = []

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]

        # Baseline: score with baseline_scorer, eval with baseline_weights
        baseline_frontier = compute_scorer_eval_frontier(
            base_probs, topics, baseline_scorer, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        # Within-shuffle on scorer
        rng = np.random.default_rng(seed + 777)
        for _ in range(N_SHUFFLES):
            shuffled_scorer = shuffle_within_groups(baseline_scorer, rng)
            frontier = compute_scorer_eval_frontier(
                base_probs, topics, shuffled_scorer, baseline_weights,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            hd = hausdorff_distance(
                baseline_frontier, frontier, utility_dims,
            )
            within_distances.append(hd)

        # α-perturbed scorer
        for factor in ALPHA_PERTURBATION_LEVELS:
            perturbed_scorer = scale_alpha(baseline_scorer, factor)
            frontier = compute_scorer_eval_frontier(
                base_probs, topics, perturbed_scorer, baseline_weights,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            hd = hausdorff_distance(
                baseline_frontier, frontier, utility_dims,
            )
            between_distances.append(hd)

        flush_print(f"    seed {seed_idx + 1}/{N_SEEDS} done")

    var_within = float(np.var(within_distances))
    var_between = float(np.var(between_distances))
    ratio = var_within / var_between if var_between > 1e-10 else 0.0
    mean_within = float(np.mean(within_distances))
    mean_between = float(np.mean(between_distances))

    flush_print(f"\n  Selection-level results:")
    flush_print(f"    Within-shuffle scorer: Hausdorff={mean_within:.4f} ± {np.std(within_distances):.4f}")
    flush_print(f"    α-perturbed scorer:    Hausdorff={mean_between:.4f} ± {np.std(between_distances):.4f}")
    flush_print(f"    α-dominance ratio (selection): {ratio:.4f}")
    flush_print(f"    α DOMINATES at selection level: {ratio < 0.1}")

    return {
        "within_shuffle_scorer": {
            "hausdorff_mean": mean_within,
            "hausdorff_std": float(np.std(within_distances)),
            "variance": var_within,
        },
        "alpha_perturbed_scorer": {
            "hausdorff_mean": mean_between,
            "hausdorff_std": float(np.std(between_distances)),
            "variance": var_between,
        },
        "alpha_dominance_ratio_selection": ratio,
        "alpha_dominates_selection": ratio < 0.1,
    }


# ---------------------------------------------------------------------------
# Strengthening: matched magnitude (#3)
# ---------------------------------------------------------------------------


def run_matched_magnitude_test() -> dict[str, Any]:
    """Criticism #3: compare within vs between at MATCHED perturbation size."""
    flush_print("\n" + "=" * 60)
    flush_print("STRENGTHENING #3: MATCHED-MAGNITUDE COMPARISON")
    flush_print("=" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]
    sigma = 0.3  # 30% perturbation budget

    within_distances: list[float] = []
    alpha_distances: list[float] = []
    mixed_distances: list[float] = []

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]

        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        rng = np.random.default_rng(seed + 666)
        for _ in range(N_SHUFFLES):
            # Within-group only (α-preserving, ±30% magnitude)
            within_perturbed = {
                name: perturb_magnitudes_preserve_alpha(
                    baseline_weights[name], sigma, rng,
                )
                for name in stakeholder_names
            }
            f_within = compute_k_frontier(
                base_probs, topics, within_perturbed,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            within_distances.append(
                hausdorff_distance(baseline_frontier, f_within, utility_dims)
            )

            # α only (uniform scale to achieve ~30% total change)
            # Scale neg by (1 + uniform(-σ, σ)) → ~30% α change
            alpha_factor = 1 + rng.uniform(-sigma, sigma)
            alpha_perturbed = {
                name: scale_alpha(baseline_weights[name], alpha_factor)
                for name in stakeholder_names
            }
            f_alpha = compute_k_frontier(
                base_probs, topics, alpha_perturbed,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            alpha_distances.append(
                hausdorff_distance(baseline_frontier, f_alpha, utility_dims)
            )

            # Mixed: 15% within + 15% α
            mixed = {
                name: scale_alpha(
                    perturb_magnitudes_preserve_alpha(
                        baseline_weights[name], sigma / 2, rng,
                    ),
                    1 + rng.uniform(-sigma / 2, sigma / 2),
                )
                for name in stakeholder_names
            }
            f_mixed = compute_k_frontier(
                base_probs, topics, mixed,
                DIVERSITY_WEIGHTS, TOP_K,
            )
            mixed_distances.append(
                hausdorff_distance(baseline_frontier, f_mixed, utility_dims)
            )

        flush_print(f"    seed {seed_idx + 1}/{N_SEEDS} done")

    flush_print(f"\n  Matched-magnitude comparison (σ={sigma}):")
    flush_print(f"    Within-only:  Hausdorff={np.mean(within_distances):.4f} ± {np.std(within_distances):.4f}")
    flush_print(f"    α-only:       Hausdorff={np.mean(alpha_distances):.4f} ± {np.std(alpha_distances):.4f}")
    flush_print(f"    Mixed (50/50): Hausdorff={np.mean(mixed_distances):.4f} ± {np.std(mixed_distances):.4f}")

    ratio = float(np.mean(within_distances)) / float(np.mean(alpha_distances)) if np.mean(alpha_distances) > 0 else 0
    flush_print(f"    Mean ratio (within/α): {ratio:.3f}")

    return {
        "sigma": sigma,
        "within_only": {
            "hausdorff_mean": float(np.mean(within_distances)),
            "hausdorff_std": float(np.std(within_distances)),
        },
        "alpha_only": {
            "hausdorff_mean": float(np.mean(alpha_distances)),
            "hausdorff_std": float(np.std(alpha_distances)),
        },
        "mixed": {
            "hausdorff_mean": float(np.mean(mixed_distances)),
            "hausdorff_std": float(np.std(mixed_distances)),
        },
        "mean_ratio_within_over_alpha": ratio,
    }


# ---------------------------------------------------------------------------
# Angle A: Full Parameter Sweep
# ---------------------------------------------------------------------------

PERTURBATION_FACTORS = [0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

# Parameters to sweep: (stakeholder, action_name, default_value)
SWEEP_PARAMS: list[tuple[str, str, float]] = [
    # User positive
    ("user", "favorite", 1.0),
    ("user", "repost", 0.8),
    ("user", "reply", 0.5),
    ("user", "share", 0.9),
    ("user", "follow_author", 1.2),
    ("user", "quote", 0.6),
    # User negative
    ("user", "block_author", -2.0),
    ("user", "mute_author", -1.5),
    ("user", "report", -2.5),
    ("user", "not_interested", -1.0),
    # Platform (top 4)
    ("platform", "repost", 1.5),
    ("platform", "reply", 1.2),
    ("platform", "share", 1.3),
    ("platform", "block_author", 0.1),
]


def run_parameter_sweep() -> dict[str, Any]:
    """Angle A: per-parameter sensitivity at evaluation level."""
    flush_print("\n" + "=" * 60)
    flush_print("ANGLE A: PARAMETER SWEEP")
    flush_print("=" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]

    per_param_results: dict[str, list[dict[str, Any]]] = {}

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]

        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        for stakeholder, action, default_val in SWEEP_PARAMS:
            param_key = f"{stakeholder}:{action}"
            if param_key not in per_param_results:
                per_param_results[param_key] = []

            action_idx = alt_losses.ACTION_INDICES[action]

            for factor in PERTURBATION_FACTORS:
                if factor == 1.0:
                    per_param_results[param_key].append({
                        "factor": factor,
                        "hausdorff": 0.0,
                        "seed": seed,
                    })
                    continue

                perturbed = {
                    n: w.copy() for n, w in baseline_weights.items()
                }
                perturbed[stakeholder][action_idx] = default_val * factor

                frontier = compute_k_frontier(
                    base_probs, topics, perturbed,
                    DIVERSITY_WEIGHTS, TOP_K,
                )
                hd = hausdorff_distance(
                    baseline_frontier, frontier, utility_dims,
                )
                per_param_results[param_key].append({
                    "factor": factor,
                    "hausdorff": hd,
                    "seed": seed,
                })

        flush_print(f"    seed {seed_idx + 1}/{N_SEEDS} done")

    # Aggregate: per parameter, per factor → mean Hausdorff
    agg: dict[str, dict[str, Any]] = {}
    for param_key, entries in per_param_results.items():
        factor_means: dict[float, float] = {}
        for factor in PERTURBATION_FACTORS:
            vals = [e["hausdorff"] for e in entries if e["factor"] == factor]
            factor_means[factor] = float(np.mean(vals)) if vals else 0.0

        # Sensitivity = mean Hausdorff across all non-baseline factors
        non_baseline = [
            v for f, v in factor_means.items() if f != 1.0
        ]
        sensitivity = float(np.mean(non_baseline)) if non_baseline else 0.0

        # Tolerance radius: largest factor range where Hausdorff < 1.0
        tolerance = 0.0
        for factor in sorted(PERTURBATION_FACTORS):
            if factor >= 1.0:
                continue
            if factor_means.get(factor, 0) < 1.0:
                tolerance = max(tolerance, 1.0 - factor)
        for factor in sorted(PERTURBATION_FACTORS):
            if factor <= 1.0:
                continue
            if factor_means.get(factor, 0) < 1.0:
                tolerance = max(tolerance, factor - 1.0)

        agg[param_key] = {
            "factor_hausdorff": {
                str(f): v for f, v in factor_means.items()
            },
            "sensitivity": sensitivity,
            "tolerance_radius": tolerance,
        }

    # Top-3 ranking
    ranked = sorted(agg.items(), key=lambda x: x[1]["sensitivity"], reverse=True)
    top3 = [k for k, _ in ranked[:3]]

    flush_print("\n  Parameter sensitivity ranking:")
    flush_print(f"  {'Parameter':<25s} {'Sensitivity':>12s} {'Tolerance':>10s}")
    flush_print("  " + "-" * 49)
    for param_key, info in ranked:
        marker = " ←" if param_key in top3 else ""
        flush_print(
            f"  {param_key:<25s} {info['sensitivity']:>12.4f}"
            f" {info['tolerance_radius']:>10.2f}{marker}"
        )

    return {
        "per_parameter": agg,
        "top3_sensitive": top3,
        "ranking": [k for k, _ in ranked],
    }


# ---------------------------------------------------------------------------
# Angle C: Specification vs Data
# ---------------------------------------------------------------------------

SPEC_SIGMAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
DATA_N_VALUES = [25, 50, 100, 200, 500, 1000, 2000]
N_SAMPLES_C = 10  # samples per condition (reduced from 20 for speed)


def generate_preferences_from_weights(
    content_probs: np.ndarray,
    weight_vector: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs using an arbitrary weight vector as utility."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    # Utility = weight_vector · action_probs for each content
    utility = np.array([
        float(np.dot(content_probs[c], weight_vector))
        for c in range(n_content)
    ])

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        noise = rng.normal(0, 0.05)
        if (utility[c1] - utility[c2] + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    return probs_pref, probs_rej


def perturb_all_weights(
    w: np.ndarray, sigma: float, rng: np.random.Generator,
) -> np.ndarray:
    """Perturb all weights by Gaussian noise (NOT preserving α)."""
    noise = 1 + rng.normal(0, sigma, len(w))
    return w * noise


def run_specification_vs_data() -> dict[str, Any]:
    """Angle C: is it cheaper to specify utilities precisely or collect data?"""
    flush_print("\n" + "=" * 60)
    flush_print("ANGLE C: SPECIFICATION VS DATA")
    flush_print("=" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    baseline_weights = default_weight_vectors()
    stakeholder_names = sorted(baseline_weights.keys())
    utility_dims = [f"{n}_utility" for n in stakeholder_names]

    # Use user weights as the "scorer" stakeholder for preference generation
    # (simplification: test on user stakeholder's utility)
    scorer_key = "user"

    # Strategy 1: vary specification precision at fixed data (2000 pairs)
    spec_results: dict[float, list[float]] = {s: [] for s in SPEC_SIGMAS}

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]
        content_probs, _ = generate_content_pool(n_content=500, seed=seed)

        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        rng = np.random.default_rng(seed + 555)

        for sigma in SPEC_SIGMAS:
            for _ in range(N_SAMPLES_C):
                if sigma == 0.0:
                    # Baseline: correct specification, full data
                    spec_results[0.0].append(0.0)
                    continue

                # Perturb weights
                perturbed = {
                    n: perturb_all_weights(w, sigma, rng)
                    for n, w in baseline_weights.items()
                }

                # Generate preferences from perturbed utility
                probs_pref, probs_rej = generate_preferences_from_weights(
                    content_probs, perturbed[scorer_key], N_PAIRS, seed,
                )

                # Train BT
                bt_config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    bt_config, probs_pref, probs_rej, verbose=False,
                )

                # Evaluate frontier using learned weights as evaluator
                learned_w = np.array(model.weights)
                eval_weights = baseline_weights.copy()
                eval_weights[scorer_key] = learned_w

                frontier = compute_k_frontier(
                    base_probs, topics, eval_weights,
                    DIVERSITY_WEIGHTS, TOP_K,
                )
                hd = hausdorff_distance(
                    baseline_frontier, frontier, utility_dims,
                )
                spec_results[sigma].append(hd)

        flush_print(f"    Strategy 1 seed {seed_idx + 1}/{N_SEEDS} done")

    # Strategy 2: vary data at fixed misspecification (σ=0.3)
    data_results: dict[int, list[float]] = {n: [] for n in DATA_N_VALUES}
    fixed_sigma = 0.3

    for seed_idx, seed in enumerate(seeds):
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        topics = data["content_topics"]
        content_probs, _ = generate_content_pool(n_content=500, seed=seed)

        baseline_frontier = compute_k_frontier(
            base_probs, topics, baseline_weights,
            DIVERSITY_WEIGHTS, TOP_K,
        )

        rng = np.random.default_rng(seed + 444)

        for n_pairs in DATA_N_VALUES:
            for _ in range(N_SAMPLES_C):
                perturbed = {
                    n: perturb_all_weights(w, fixed_sigma, rng)
                    for n, w in baseline_weights.items()
                }

                probs_pref, probs_rej = generate_preferences_from_weights(
                    content_probs, perturbed[scorer_key], n_pairs, seed,
                )

                bt_config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    bt_config, probs_pref, probs_rej, verbose=False,
                )

                learned_w = np.array(model.weights)
                eval_weights = baseline_weights.copy()
                eval_weights[scorer_key] = learned_w

                frontier = compute_k_frontier(
                    base_probs, topics, eval_weights,
                    DIVERSITY_WEIGHTS, TOP_K,
                )
                hd = hausdorff_distance(
                    baseline_frontier, frontier, utility_dims,
                )
                data_results[n_pairs].append(hd)

        flush_print(f"    Strategy 2 seed {seed_idx + 1}/{N_SEEDS} done")

    # Print results
    flush_print("\n  Strategy 1 (vary specification, N=2000):")
    for sigma in SPEC_SIGMAS:
        vals = spec_results[sigma]
        flush_print(
            f"    σ={sigma:.2f}: Hausdorff={np.mean(vals):.4f}"
            f" ± {np.std(vals):.4f}"
        )

    flush_print(f"\n  Strategy 2 (vary data, σ={fixed_sigma}):")
    for n in DATA_N_VALUES:
        vals = data_results[n]
        flush_print(
            f"    N={n:>5d}: Hausdorff={np.mean(vals):.4f}"
            f" ± {np.std(vals):.4f}"
        )

    # Crossover: at what σ does Strategy 1 match Strategy 2's best?
    best_data = min(
        np.mean(data_results[n]) for n in DATA_N_VALUES
    )
    crossover_sigma = None
    for sigma in sorted(SPEC_SIGMAS, reverse=True):
        if np.mean(spec_results[sigma]) <= best_data:
            crossover_sigma = sigma
            break

    flush_print(
        f"\n  Crossover: best data (N=2000, σ=0.3) = {best_data:.4f}"
    )
    if crossover_sigma is not None:
        flush_print(
            f"  Specification σ={crossover_sigma} matches this quality"
        )

    return {
        "strategy1_spec": {
            str(s): {
                "hausdorff_mean": float(np.mean(v)),
                "hausdorff_std": float(np.std(v)),
            }
            for s, v in spec_results.items()
        },
        "strategy2_data": {
            str(n): {
                "hausdorff_mean": float(np.mean(v)),
                "hausdorff_std": float(np.std(v)),
            }
            for n, v in data_results.items()
        },
        "fixed_sigma": fixed_sigma,
        "crossover_sigma": crossover_sigma,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()

    # Load cached results
    existing: dict[str, Any] = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            existing = json.load(f)

    # Phase 1: α-dominance (load from cache if exists)
    if "alpha_dominance" not in existing:
        existing["alpha_dominance"] = run_alpha_dominance_test()

    # Phase 2: strengthening tests (load from cache if exist)
    if "magnitude_perturbation" not in existing:
        existing["magnitude_perturbation"] = run_magnitude_perturbation_test()
    if "selection_level" not in existing:
        existing["selection_level"] = run_selection_level_test()
    if "matched_magnitude" not in existing:
        existing["matched_magnitude"] = run_matched_magnitude_test()

    # Phase 3: Angle A — parameter sweep
    existing["parameter_sweep"] = run_parameter_sweep()

    # Phase 4: Angle C — specification vs data
    existing["specification_vs_data"] = run_specification_vs_data()

    wall_time = time.time() - t0
    flush_print(f"\n  Total wall time: {wall_time:.1f}s ({wall_time / 60:.1f}m)")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(existing, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
