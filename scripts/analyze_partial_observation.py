#!/usr/bin/env python3
"""Partial observation analysis: Leave-One-Stakeholder-Out Pareto frontiers.

Direction 3 of the F4 research directions.  Measures Pareto frontier
degradation when a stakeholder is unobservable.

Experiment 1 — LOSO Geometry:
  Project full 3-stakeholder frontier to 2D and measure hidden dim loss.

Experiment 2 — Training-based LOSO:
  Train BT models for 2 observed stakeholders only, compute learned
  frontiers, compare against full 3-stakeholder trained frontier.

Experiment 3 — Aggregation Proxy:
  Given observed weight vectors, find best proxy for hidden stakeholder.
  Oracle linear proxy (ceiling), diversity knob (practical), and
  α-extrapolation (structural).

Usage:
    uv run python scripts/analyze_partial_observation.py
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
# Module loading (same pattern as compare_pareto_frontiers.py)
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
reward_modeling_pkg.__path__ = [
    os.path.join(project_root, "enhancements/reward_modeling")
]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

weights_mod = load_module_direct(
    "enhancements.reward_modeling.weights",
    os.path.join(project_root, "enhancements/reward_modeling/weights.py"),
)
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

# Load alternative_losses for Exp 2
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
ACTION_INDICES = alt_losses.ACTION_INDICES

# Reuse data generation
analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

# Load run_loss_experiments for generate_content_pool
loss_exp_mod = load_module_direct(
    "run_loss_experiments",
    os.path.join(project_root, "scripts/run_loss_experiments.py"),
)
generate_content_pool = loss_exp_mod.generate_content_pool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SEEDS = 5
BASE_SEED = 42
NUM_USERS = 600
NUM_CONTENT = 100
NUM_TOPICS = 6
TOP_K = 10
DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]  # 0.00 to 1.00

# For Exp 2
N_PAIRS = 2000
NUM_EPOCHS = 50
LEARNING_RATE = 0.01

STAKEHOLDERS = ["user", "platform", "society"]
UTILITY_DIMS = ["user_utility", "platform_utility", "society_utility"]

# Stakeholder utility functions (same as run_loss_experiments.py)
STAKEHOLDER_UTILITY = {
    "user": lambda pos, neg: pos - neg,
    "platform": lambda pos, neg: pos - 0.3 * neg,
    "society": lambda pos, neg: pos - 4.0 * neg,
}

STAKEHOLDER_TYPE_MAP = {
    "user": StakeholderType.USER,
    "platform": StakeholderType.PLATFORM,
    "society": StakeholderType.SOCIETY,
}


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Shared helpers (matching test file logic)
# ---------------------------------------------------------------------------

UtilityPoint = dict[str, float]


def extract_pareto_front_2d(
    points: list[UtilityPoint], dim_x: str, dim_y: str
) -> list[UtilityPoint]:
    """Extract 2D Pareto-optimal subset (maximize both dimensions)."""
    if not points:
        return []
    sorted_pts = sorted(points, key=lambda p: p[dim_x], reverse=True)
    pareto: list[UtilityPoint] = []
    max_y = float("-inf")
    for p in sorted_pts:
        if p[dim_y] > max_y:
            pareto.append(p)
            max_y = p[dim_y]
        elif p[dim_y] == max_y and (
            not pareto or p[dim_x] == pareto[-1][dim_x]
        ):
            pareto.append(p)
    return pareto


def is_dominated(
    point: UtilityPoint, candidates: list[UtilityPoint], dims: list[str]
) -> bool:
    """Check if point is strictly dominated by any candidate."""
    for c in candidates:
        all_geq = all(c[d] >= point[d] for d in dims)
        any_gt = any(c[d] > point[d] for d in dims)
        if all_geq and any_gt:
            return True
    return False


def compute_dominated_fraction(
    partial: list[UtilityPoint], full: list[UtilityPoint], dims: list[str]
) -> float:
    if not partial:
        return 0.0
    count = sum(1 for p in partial if is_dominated(p, full, dims))
    return count / len(partial)


def compute_regret(
    partial: list[UtilityPoint], full: list[UtilityPoint], hidden_dim: str
) -> dict[str, float]:
    if not partial or not full:
        return {"max_regret": 0.0, "avg_regret": 0.0, "min_regret": 0.0}
    max_achievable = max(f[hidden_dim] for f in full)
    regrets = [max_achievable - p[hidden_dim] for p in partial]
    return {
        "max_regret": float(max(regrets)),
        "avg_regret": float(np.mean(regrets)),
        "min_regret": float(min(regrets)),
    }


def compute_hypervolume_2d(
    points: list[tuple[float, float]], ref_point: tuple[float, float]
) -> float:
    if not points:
        return 0.0
    sorted_pts = sorted(points, key=lambda p: p[0], reverse=True)
    pareto: list[tuple[float, float]] = []
    max_y = float("-inf")
    for p in sorted_pts:
        if p[1] > max_y:
            pareto.append(p)
            max_y = p[1]
    pareto.sort(key=lambda p: p[0])
    area = 0.0
    prev_x = ref_point[0]
    for x, y in pareto:
        if x > prev_x and y > ref_point[1]:
            area += (x - prev_x) * (y - ref_point[1])
            prev_x = x
    return area


def compute_frontier_distance(
    partial: list[UtilityPoint], full: list[UtilityPoint], dims: list[str]
) -> dict[str, float]:
    if not partial or not full:
        return {"mean": 0.0, "max": 0.0, "median": 0.0}
    distances = []
    for p in partial:
        p_vec = np.array([p[d] for d in dims])
        min_dist = float("inf")
        for f in full:
            f_vec = np.array([f[d] for d in dims])
            dist = float(np.linalg.norm(p_vec - f_vec))
            min_dist = min(min_dist, dist)
        distances.append(min_dist)
    return {
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
    }


# ---------------------------------------------------------------------------
# Data generation helpers
# ---------------------------------------------------------------------------


def build_base_action_probs(data: dict) -> np.ndarray:
    """Build [N, M, 18] user-content action probability matrix."""
    n = data["num_users"]
    m = data["num_content"]
    base = np.tile(data["content_action_probs"][np.newaxis, :, :], (n, 1, 1))
    for user_idx in range(n):
        archetype = data["user_archetypes"][user_idx]
        for content_idx in range(m):
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


def compute_learned_frontier(
    weight_vector: np.ndarray,
    base_action_probs: np.ndarray,
    diversity_weights: list[float],
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = TOP_K,
) -> list[UtilityPoint]:
    """Compute frontier using learned weights as scorer + diversity knob."""
    n, m, _ = base_action_probs.shape
    num_topics = len(np.unique(content_topics))
    weight_vector = np.asarray(weight_vector, dtype=np.float64)

    points: list[UtilityPoint] = []

    for div_weight in diversity_weights:
        recommendations = []
        for user_idx in range(n):
            scores = base_action_probs[user_idx] @ weight_vector
            if div_weight > 0:
                selected: list[int] = []
                remaining = list(range(m))
                topic_counts = np.zeros(num_topics)
                for _ in range(top_k):
                    if not remaining:
                        break
                    adjusted = []
                    for idx in remaining:
                        topic = content_topics[idx]
                        div_bonus = 1.0 / (topic_counts[topic] + 1)
                        s = (1 - div_weight) * scores[idx] + div_weight * div_bonus
                        adjusted.append((idx, s))
                    best_idx = max(adjusted, key=lambda x: x[1])[0]
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    topic_counts[content_topics[best_idx]] += 1
                recommendations.append(selected)
            else:
                top_indices = np.argsort(scores)[-top_k:][::-1]
                recommendations.append(top_indices.tolist())

        recs = np.array(recommendations)
        user_utils = []
        platform_utils = []
        for user_idx in range(n):
            rec_probs = base_action_probs[user_idx, recs[user_idx]]
            user_utils.append(compute_user_utility(rec_probs).total_utility)
            platform_utils.append(compute_platform_utility(rec_probs).total_utility)
        society_result = compute_society_utility(
            recs, user_archetypes, content_topics, num_topics
        )
        points.append({
            "diversity_weight": div_weight,
            "user_utility": float(np.mean(user_utils)),
            "platform_utility": float(np.mean(platform_utils)),
            "society_utility": float(society_result.total_utility),
        })

    return points


# ---------------------------------------------------------------------------
# Exp 1: LOSO Geometry
# ---------------------------------------------------------------------------


def compute_full_frontier(
    base_action_probs: np.ndarray,
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
) -> list[UtilityPoint]:
    """Compute full 3-stakeholder frontier via diversity_weight sweep."""
    raw = compute_pareto_frontier(
        base_action_probs, DIVERSITY_WEIGHTS, user_archetypes,
        content_topics, top_k=TOP_K,
    )
    return [
        {
            "diversity_weight": p.diversity_weight,
            "user_utility": p.user_utility,
            "platform_utility": p.platform_utility,
            "society_utility": p.society_utility,
        }
        for p in raw
    ]


def compute_all_metrics(
    partial: list[UtilityPoint],
    full: list[UtilityPoint],
    hidden_dim: str,
    observed_dims: list[str],
) -> dict[str, Any]:
    """Compute all degradation metrics for one LOSO experiment."""
    dom_frac = compute_dominated_fraction(partial, full, UTILITY_DIMS)
    regret = compute_regret(partial, full, hidden_dim)
    dist = compute_frontier_distance(partial, full, UTILITY_DIMS)

    # Hypervolume in observed 2D space
    obs_pts_partial = [(p[observed_dims[0]], p[observed_dims[1]]) for p in partial]
    obs_pts_full = [(f[observed_dims[0]], f[observed_dims[1]]) for f in full]
    all_pts = obs_pts_partial + obs_pts_full
    ref = (min(x for x, _ in all_pts) - 0.1, min(y for _, y in all_pts) - 0.1)
    hv_partial = compute_hypervolume_2d(obs_pts_partial, ref)
    hv_full = compute_hypervolume_2d(obs_pts_full, ref)
    hv_ratio = hv_partial / hv_full if hv_full > 0 else 1.0

    return {
        "dominated_fraction": dom_frac,
        **regret,
        "hypervolume_ratio_observed_2d": hv_ratio,
        "frontier_distance_mean": dist["mean"],
        "frontier_distance_max": dist["max"],
        "partial_frontier_size": len(partial),
        "full_frontier_size": len(full),
        "hidden_utility_on_partial": [p[hidden_dim] for p in partial],
    }


LOSO_CONFIGS = [
    {
        "hidden": "society",
        "hidden_dim": "society_utility",
        "observed_dims": ["user_utility", "platform_utility"],
    },
    {
        "hidden": "platform",
        "hidden_dim": "platform_utility",
        "observed_dims": ["user_utility", "society_utility"],
    },
    {
        "hidden": "user",
        "hidden_dim": "user_utility",
        "observed_dims": ["platform_utility", "society_utility"],
    },
]


def run_exp1_geometry(seeds: list[int]) -> dict[str, Any]:
    """Exp 1: LOSO Geometry — project frontiers to 2D."""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT 1: LOSO GEOMETRY")
    flush_print("=" * 60)

    results: dict[str, Any] = {}

    for config in LOSO_CONFIGS:
        hidden = config["hidden"]
        hidden_dim = config["hidden_dim"]
        observed_dims = config["observed_dims"]

        flush_print(f"\n  Hidden: {hidden}")
        seed_metrics: list[dict[str, Any]] = []

        for seed in seeds:
            data = generate_synthetic_data(
                num_users=NUM_USERS, num_content=NUM_CONTENT,
                num_topics=NUM_TOPICS, seed=seed,
            )
            base_probs = build_base_action_probs(data)
            full_frontier = compute_full_frontier(
                base_probs, data["user_archetypes"], data["content_topics"],
            )

            # LOSO: Pareto-optimal in observed 2D
            loso_frontier = extract_pareto_front_2d(
                full_frontier, observed_dims[0], observed_dims[1],
            )

            # Full 3D Pareto
            full_pareto = [
                p for p in full_frontier
                if not is_dominated(p, full_frontier, UTILITY_DIMS)
            ]

            metrics = compute_all_metrics(
                loso_frontier, full_pareto, hidden_dim, observed_dims,
            )
            seed_metrics.append(metrics)

        # Aggregate across seeds
        metric_keys = [
            "dominated_fraction", "max_regret", "avg_regret", "min_regret",
            "hypervolume_ratio_observed_2d", "frontier_distance_mean",
        ]
        agg = {}
        for key in metric_keys:
            vals = [m[key] for m in seed_metrics]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))

        flush_print(
            f"    dominated_frac={agg['dominated_fraction_mean']:.3f}±{agg['dominated_fraction_std']:.3f}"
            f"  avg_regret={agg['avg_regret_mean']:.3f}±{agg['avg_regret_std']:.3f}"
            f"  max_regret={agg['max_regret_mean']:.3f}±{agg['max_regret_std']:.3f}"
        )

        results[f"hide_{hidden}"] = {
            "metrics_mean": {k: agg[f"{k}_mean"] for k in metric_keys},
            "metrics_std": {k: agg[f"{k}_std"] for k in metric_keys},
            "per_seed": seed_metrics,
        }

    return results


# ---------------------------------------------------------------------------
# Exp 2: Training-based LOSO
# ---------------------------------------------------------------------------


def generate_stakeholder_preferences(
    content_probs: np.ndarray,
    stakeholder: str,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs for one stakeholder."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)
    utility_fn = STAKEHOLDER_UTILITY[stakeholder]

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    utility = utility_fn(pos_scores, neg_scores)

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]
        noise = rng.normal(0, 0.05)

        if (diff + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    return probs_pref, probs_rej


def check_convergence(loss_history: list[float], window: int = 5, threshold: float = 0.05) -> bool:
    """Check if training loss has plateaued (last `window` epochs < `threshold` relative change)."""
    if len(loss_history) < window:
        return True
    recent = loss_history[-window:]
    if recent[0] == 0:
        return True
    change = abs(recent[-1] - recent[0]) / (abs(recent[0]) + 1e-10)
    return change < threshold


def run_exp2_training(seeds: list[int]) -> dict[str, Any]:
    """Exp 2: Training-based LOSO — train BT on 2 stakeholders."""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT 2: TRAINING-BASED LOSO")
    flush_print("=" * 60)

    results: dict[str, Any] = {}
    convergence_warnings: list[str] = []

    for config in LOSO_CONFIGS:
        hidden = config["hidden"]
        observed = [s for s in STAKEHOLDERS if s != hidden]
        hidden_dim = config["hidden_dim"]
        observed_dims = config["observed_dims"]

        flush_print(f"\n  Hidden: {hidden}, Training: {observed}")
        seed_metrics: list[dict[str, Any]] = []

        for seed_idx, seed in enumerate(seeds):
            # Generate content pool for training
            content_probs, _ = generate_content_pool(
                n_content=500, seed=seed,
            )

            # Generate evaluation data
            data = generate_synthetic_data(
                num_users=NUM_USERS, num_content=NUM_CONTENT,
                num_topics=NUM_TOPICS, seed=seed,
            )
            base_probs = build_base_action_probs(data)

            # Train BT models for observed stakeholders only
            learned_weights: dict[str, np.ndarray] = {}
            for s in observed:
                probs_pref, probs_rej = generate_stakeholder_preferences(
                    content_probs, s, N_PAIRS, seed=seed + hash(s) % 10000,
                )
                bt_config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=STAKEHOLDER_TYPE_MAP[s],
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    bt_config, probs_pref, probs_rej, verbose=False,
                )
                learned_weights[s] = np.array(model.weights)

                # Convergence check
                if not check_convergence(model.loss_history):
                    msg = f"  WARNING: {s} (seed={seed}) may not have converged"
                    flush_print(msg)
                    convergence_warnings.append(msg)

            # Compute learned frontiers for each observed stakeholder
            observed_frontiers: dict[str, list[UtilityPoint]] = {}
            for s in observed:
                observed_frontiers[s] = compute_learned_frontier(
                    learned_weights[s], base_probs, DIVERSITY_WEIGHTS,
                    data["user_archetypes"], data["content_topics"],
                )

            # Full 3-stakeholder frontier (hardcoded scorer baseline)
            full_frontier = compute_full_frontier(
                base_probs, data["user_archetypes"], data["content_topics"],
            )
            full_pareto = [
                p for p in full_frontier
                if not is_dominated(p, full_frontier, UTILITY_DIMS)
            ]

            # Metrics for each observed scorer
            scorer_metrics: dict[str, dict[str, Any]] = {}
            for s in observed:
                loso = extract_pareto_front_2d(
                    observed_frontiers[s], observed_dims[0], observed_dims[1],
                )
                metrics = compute_all_metrics(
                    loso, full_pareto, hidden_dim, observed_dims,
                )
                scorer_metrics[s] = metrics

            # Also: best of the two observed scorers at each diversity_weight
            # (pick the one with better hidden utility)
            combined: list[UtilityPoint] = []
            for dw in DIVERSITY_WEIGHTS:
                pts_at_dw = []
                for s in observed:
                    for p in observed_frontiers[s]:
                        if abs(p["diversity_weight"] - dw) < 0.001:
                            pts_at_dw.append(p)
                if pts_at_dw:
                    # Pick point with best hidden utility (oracle best-of-two)
                    combined.append(max(pts_at_dw, key=lambda p: p[hidden_dim]))
            combined_loso = extract_pareto_front_2d(
                combined, observed_dims[0], observed_dims[1],
            )
            combined_metrics = compute_all_metrics(
                combined_loso, full_pareto, hidden_dim, observed_dims,
            )

            seed_metrics.append({
                "per_scorer": scorer_metrics,
                "combined_best": combined_metrics,
            })

            flush_print(f"    seed {seed_idx + 1}/{len(seeds)} done")

        # Aggregate
        metric_keys = [
            "dominated_fraction", "max_regret", "avg_regret",
            "hypervolume_ratio_observed_2d", "frontier_distance_mean",
        ]
        agg_combined = {}
        for key in metric_keys:
            vals = [m["combined_best"][key] for m in seed_metrics]
            agg_combined[f"{key}_mean"] = float(np.mean(vals))
            agg_combined[f"{key}_std"] = float(np.std(vals))

        flush_print(
            f"    combined: dom_frac={agg_combined['dominated_fraction_mean']:.3f}"
            f"  avg_regret={agg_combined['avg_regret_mean']:.3f}"
            f"  max_regret={agg_combined['max_regret_mean']:.3f}"
        )

        results[f"hide_{hidden}"] = {
            "combined_metrics_mean": {k: agg_combined[f"{k}_mean"] for k in metric_keys},
            "combined_metrics_std": {k: agg_combined[f"{k}_std"] for k in metric_keys},
            "per_seed": seed_metrics,
        }

    results["convergence_warnings"] = convergence_warnings
    return results


# ---------------------------------------------------------------------------
# Exp 3: Optimal Aggregation Proxy
# ---------------------------------------------------------------------------

# Affine calibration from results/alpha_recovery.json (Direction 1)
ALPHA_AFFINE_SLOPE = 1.3207
ALPHA_AFFINE_INTERCEPT = -0.0619


def load_trained_weight_vectors() -> dict[str, np.ndarray]:
    """Load pre-trained BT weight vectors from results/loss_experiments/."""
    vectors: dict[str, np.ndarray] = {}
    for s in STAKEHOLDERS:
        path = os.path.join(
            project_root, f"results/loss_experiments/bradley_terry_{s}.json",
        )
        with open(path) as f:
            data = json.load(f)
        vectors[s] = np.array(data["weights_vector"], dtype=np.float64)
    return vectors


def compute_oracle_linear_proxy(
    w_obs1: np.ndarray,
    w_obs2: np.ndarray,
    w_hidden: np.ndarray,
) -> dict[str, Any]:
    """Find optimal a, b such that a*w_obs1 + b*w_obs2 ≈ w_hidden (least-squares)."""
    X = np.column_stack([w_obs1, w_obs2])
    coeffs, _, _, _ = np.linalg.lstsq(X, w_hidden, rcond=None)
    w_proxy = X @ coeffs
    cos_sim = float(
        np.dot(w_proxy, w_hidden)
        / (np.linalg.norm(w_proxy) * np.linalg.norm(w_hidden) + 1e-12)
    )
    residual_norm = float(np.linalg.norm(w_hidden - w_proxy))
    ss_res = float(np.sum((w_hidden - w_proxy) ** 2))
    ss_tot = float(np.sum((w_hidden - np.mean(w_hidden)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return {
        "coefficients": (float(coeffs[0]), float(coeffs[1])),
        "proxy_weights": w_proxy,
        "cosine_sim": cos_sim,
        "residual_norm": residual_norm,
        "r_squared": r_squared,
    }


def recover_alpha_from_weights(weights: np.ndarray, calibrate: bool = True) -> float:
    """Recover α as -mean(w_neg)/mean(w_pos), optionally with affine calibration."""
    pos_mean = float(np.mean(weights[POSITIVE_INDICES]))
    neg_mean = float(np.mean(weights[NEGATIVE_INDICES]))
    if abs(pos_mean) < 1e-12:
        return 0.0
    raw = -neg_mean / pos_mean
    if calibrate:
        return (raw - ALPHA_AFFINE_INTERCEPT) / ALPHA_AFFINE_SLOPE
    return raw


def synthesize_weights_interpolation(
    w1: np.ndarray, alpha1: float,
    w2: np.ndarray, alpha2: float,
    alpha_target: float,
) -> np.ndarray:
    """Linear interpolation/extrapolation in weight space."""
    if abs(alpha1 - alpha2) < 1e-12:
        return (w1 + w2) / 2.0
    t = (alpha_target - alpha2) / (alpha1 - alpha2)
    return w2 + t * (w1 - w2)


def synthesize_weights_structural(
    observed_weights: dict[str, np.ndarray],
    alpha_target: float,
) -> np.ndarray:
    """Structural synthesis: mean pos/neg/neutral + target α scaling."""
    all_w = np.stack(list(observed_weights.values()))
    mean_w = np.mean(all_w, axis=0)

    pos_mean = float(np.mean(mean_w[POSITIVE_INDICES]))
    all_special = set(POSITIVE_INDICES) | set(NEGATIVE_INDICES)
    neutral_indices = [i for i in range(NUM_ACTIONS) if i not in all_special]

    result = np.zeros(NUM_ACTIONS, dtype=np.float64)
    result[POSITIVE_INDICES] = pos_mean
    result[NEGATIVE_INDICES] = -alpha_target * abs(pos_mean)
    result[neutral_indices] = mean_w[neutral_indices]
    return result


def find_best_hidden_utility(
    frontier: list[UtilityPoint],
    hidden_dim: str,
) -> tuple[float, float]:
    """Find point maximizing hidden utility. Returns (diversity_weight, utility)."""
    if not frontier:
        return (0.0, 0.0)
    best = max(frontier, key=lambda p: p[hidden_dim])
    return (best.get("diversity_weight", 0.0), best[hidden_dim])


def compute_recovery_rate(
    proxy_utility: float,
    no_proxy_utility: float,
    full_utility: float,
) -> float:
    """Recovery rate = (proxy - no_proxy) / (full - no_proxy)."""
    gap = full_utility - no_proxy_utility
    if abs(gap) < 1e-12:
        return 1.0
    return (proxy_utility - no_proxy_utility) / gap


def _evaluate_proxy_on_seed(
    proxy_weights: np.ndarray,
    base_probs: np.ndarray,
    user_archetypes: np.ndarray,
    content_topics: np.ndarray,
    full_pareto: list[UtilityPoint],
    loso_frontier: list[UtilityPoint],
    hidden_dim: str,
    observed_dims: list[str],
) -> dict[str, float]:
    """Evaluate a proxy weight vector on one seed's data.

    Uses proxy to generate frontier, extracts 2D Pareto on observed dims,
    and measures hidden utility / recovery rate.
    """
    proxy_frontier = compute_learned_frontier(
        proxy_weights, base_probs, DIVERSITY_WEIGHTS,
        user_archetypes, content_topics,
    )
    proxy_loso = extract_pareto_front_2d(
        proxy_frontier, observed_dims[0], observed_dims[1],
    )
    # Hidden utility on the proxy's 2D Pareto front
    _, proxy_best_hidden = find_best_hidden_utility(proxy_loso, hidden_dim)

    # Baselines
    full_best_hidden = max(p[hidden_dim] for p in full_pareto)
    _, loso_best_hidden = find_best_hidden_utility(loso_frontier, hidden_dim)

    recovery = compute_recovery_rate(proxy_best_hidden, loso_best_hidden, full_best_hidden)

    regret = compute_regret(proxy_loso, full_pareto, hidden_dim)

    return {
        "proxy_best_hidden": proxy_best_hidden,
        "loso_best_hidden": loso_best_hidden,
        "full_best_hidden": full_best_hidden,
        "recovery_rate": recovery,
        "avg_regret": regret["avg_regret"],
        "max_regret": regret["max_regret"],
    }


def run_exp3_proxy(
    seeds: list[int],
) -> dict[str, Any]:
    """Exp 3: Optimal aggregation proxy for hidden stakeholder."""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT 3: AGGREGATION PROXY")
    flush_print("=" * 60)

    # Load pre-trained weight vectors (for 3a oracle and 3c alpha recovery)
    trained_weights = load_trained_weight_vectors()
    flush_print("  Loaded pre-trained weight vectors")

    results: dict[str, Any] = {}

    for config in LOSO_CONFIGS:
        hidden = config["hidden"]
        hidden_dim = config["hidden_dim"]
        observed_dims = config["observed_dims"]
        observed_stakeholders = [s for s in STAKEHOLDERS if s != hidden]

        flush_print(f"\n  Hidden: {hidden}")

        # --- 3a: Oracle linear proxy (one set of coefficients) ---
        w_obs1 = trained_weights[observed_stakeholders[0]]
        w_obs2 = trained_weights[observed_stakeholders[1]]
        w_hidden = trained_weights[hidden]

        oracle_result = compute_oracle_linear_proxy(w_obs1, w_obs2, w_hidden)
        oracle_proxy = oracle_result["proxy_weights"]
        flush_print(
            f"    3a oracle: cosine={oracle_result['cosine_sim']:.4f}"
            f"  residual={oracle_result['residual_norm']:.4f}"
            f"  R²={oracle_result['r_squared']:.4f}"
            f"  coeffs=({oracle_result['coefficients'][0]:.3f}, {oracle_result['coefficients'][1]:.3f})"
        )

        # --- 3c: Alpha recovery + synthesis ---
        alpha_obs = {}
        for s in observed_stakeholders:
            alpha_obs[s] = recover_alpha_from_weights(trained_weights[s], calibrate=True)
        alpha_obs_raw = {}
        for s in observed_stakeholders:
            alpha_obs_raw[s] = recover_alpha_from_weights(trained_weights[s], calibrate=False)
        flush_print(
            "    3c α recovery (calibrated): "
            + ", ".join(f"{s}={alpha_obs[s]:.3f}" for s in observed_stakeholders)
        )

        # Known stakeholder alphas for ground truth
        true_alphas = {"user": 1.0, "platform": 0.3, "society": 4.0}

        # Synthesis variants
        obs_weights = {s: trained_weights[s] for s in observed_stakeholders}
        alpha_true_hidden = true_alphas[hidden]
        alpha_max_obs_cal = max(alpha_obs.values())
        alpha_blind = 2.0 * alpha_max_obs_cal

        synthesis_configs: dict[str, dict[str, Any]] = {
            "interp_oracle_alpha": {
                "method": "interpolation",
                "alpha_target": alpha_true_hidden,
                "label": f"interp α={alpha_true_hidden:.1f} (oracle)",
            },
            "struct_oracle_alpha": {
                "method": "structural",
                "alpha_target": alpha_true_hidden,
                "label": f"struct α={alpha_true_hidden:.1f} (oracle)",
            },
            "interp_blind_alpha": {
                "method": "interpolation",
                "alpha_target": alpha_blind,
                "label": f"interp α={alpha_blind:.1f} (blind 2×max)",
            },
            "struct_blind_alpha": {
                "method": "structural",
                "alpha_target": alpha_blind,
                "label": f"struct α={alpha_blind:.1f} (blind 2×max)",
            },
        }

        synth_weights: dict[str, np.ndarray] = {}
        for key, sc in synthesis_configs.items():
            if sc["method"] == "interpolation":
                s1, s2 = observed_stakeholders
                synth_weights[key] = synthesize_weights_interpolation(
                    trained_weights[s1], alpha_obs[s1],
                    trained_weights[s2], alpha_obs[s2],
                    sc["alpha_target"],
                )
            else:
                synth_weights[key] = synthesize_weights_structural(
                    obs_weights, sc["alpha_target"],
                )
            # Report cosine with true hidden
            cos = float(
                np.dot(synth_weights[key], w_hidden)
                / (np.linalg.norm(synth_weights[key]) * np.linalg.norm(w_hidden) + 1e-12)
            )
            flush_print(f"    3c {sc['label']}: cosine={cos:.4f}")

        # --- Per-seed evaluation ---
        seed_results: list[dict[str, Any]] = []

        for seed_idx, seed in enumerate(seeds):
            data = generate_synthetic_data(
                num_users=NUM_USERS, num_content=NUM_CONTENT,
                num_topics=NUM_TOPICS, seed=seed,
            )
            base_probs = build_base_action_probs(data)

            # Full frontier + Pareto
            full_frontier = compute_full_frontier(
                base_probs, data["user_archetypes"], data["content_topics"],
            )
            full_pareto = [
                p for p in full_frontier
                if not is_dominated(p, full_frontier, UTILITY_DIMS)
            ]
            # LOSO baseline
            loso_frontier = extract_pareto_front_2d(
                full_frontier, observed_dims[0], observed_dims[1],
            )

            seed_data: dict[str, Any] = {}

            # 3a: Oracle proxy evaluation
            seed_data["oracle_linear"] = _evaluate_proxy_on_seed(
                oracle_proxy, base_probs,
                data["user_archetypes"], data["content_topics"],
                full_pareto, loso_frontier, hidden_dim, observed_dims,
            )

            # 3b: Diversity knob (from full frontier data, no proxy weights)
            full_best_hidden = max(p[hidden_dim] for p in full_pareto)
            _, loso_best_hidden = find_best_hidden_utility(loso_frontier, hidden_dim)
            # Oracle best dw
            dw_oracle, util_oracle = find_best_hidden_utility(full_frontier, hidden_dim)
            # Fixed dw values
            dw_results: dict[str, Any] = {
                "oracle_best_dw": {
                    "diversity_weight": dw_oracle,
                    "hidden_utility": util_oracle,
                    "recovery_rate": compute_recovery_rate(
                        util_oracle, loso_best_hidden, full_best_hidden,
                    ),
                },
            }
            for fixed_dw in [0.3, 0.5, 0.7]:
                # Find the frontier point closest to this dw
                closest = min(
                    full_frontier, key=lambda p: abs(p["diversity_weight"] - fixed_dw),
                )
                dw_results[f"dw_{fixed_dw}"] = {
                    "diversity_weight": closest["diversity_weight"],
                    "hidden_utility": closest[hidden_dim],
                    "recovery_rate": compute_recovery_rate(
                        closest[hidden_dim], loso_best_hidden, full_best_hidden,
                    ),
                }
            seed_data["diversity_knob"] = dw_results

            # 3c: Alpha extrapolation variants
            for key in synthesis_configs:
                seed_data[key] = _evaluate_proxy_on_seed(
                    synth_weights[key], base_probs,
                    data["user_archetypes"], data["content_topics"],
                    full_pareto, loso_frontier, hidden_dim, observed_dims,
                )

            seed_results.append(seed_data)
            flush_print(f"    seed {seed_idx + 1}/{len(seeds)} done")

        # --- Aggregate across seeds ---
        proxy_methods = (
            ["oracle_linear"]
            + list(synthesis_configs.keys())
        )
        aggregated: dict[str, Any] = {}

        for method in proxy_methods:
            recovery_vals = [s[method]["recovery_rate"] for s in seed_results]
            regret_vals = [s[method]["avg_regret"] for s in seed_results]
            aggregated[method] = {
                "recovery_rate_mean": float(np.mean(recovery_vals)),
                "recovery_rate_std": float(np.std(recovery_vals)),
                "avg_regret_mean": float(np.mean(regret_vals)),
                "avg_regret_std": float(np.std(regret_vals)),
            }
            flush_print(
                f"    {method}: recovery={aggregated[method]['recovery_rate_mean']:.3f}"
                f"±{aggregated[method]['recovery_rate_std']:.3f}"
                f"  regret={aggregated[method]['avg_regret_mean']:.3f}"
            )

        # Diversity knob aggregation
        dw_keys = ["oracle_best_dw", "dw_0.3", "dw_0.5", "dw_0.7"]
        dw_agg: dict[str, Any] = {}
        for dk in dw_keys:
            rec_vals = [s["diversity_knob"][dk]["recovery_rate"] for s in seed_results]
            dw_agg[dk] = {
                "recovery_rate_mean": float(np.mean(rec_vals)),
                "recovery_rate_std": float(np.std(rec_vals)),
            }
            flush_print(
                f"    diversity_knob/{dk}: recovery={dw_agg[dk]['recovery_rate_mean']:.3f}"
                f"±{dw_agg[dk]['recovery_rate_std']:.3f}"
            )
        aggregated["diversity_knob"] = dw_agg

        results[f"hide_{hidden}"] = {
            "oracle_proxy_info": {
                "coefficients": oracle_result["coefficients"],
                "cosine_sim": oracle_result["cosine_sim"],
                "residual_norm": oracle_result["residual_norm"],
                "r_squared": oracle_result["r_squared"],
            },
            "alpha_recovery": {
                s: {"calibrated": alpha_obs[s], "raw": alpha_obs_raw[s]}
                for s in observed_stakeholders
            },
            "synthesis_cosines": {
                key: float(
                    np.dot(synth_weights[key], w_hidden)
                    / (np.linalg.norm(synth_weights[key]) * np.linalg.norm(w_hidden) + 1e-12)
                )
                for key in synthesis_configs
            },
            "aggregated": aggregated,
            "per_seed": seed_results,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


OUTPUT_PATH = os.path.join(project_root, "results/partial_observation.json")


def load_cached_results() -> dict[str, Any]:
    """Load previously saved results from JSON, or return empty dict."""
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return {}


def save_results(output: dict[str, Any]) -> None:
    """Save results to JSON."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to: {OUTPUT_PATH}")


def parse_experiments(args: list[str]) -> set[int]:
    """Parse --exp flag to determine which experiments to run.

    Usage: --exp 3       (run only Exp 3)
           --exp 1,3     (run Exp 1 and 3)
           (no flag)     (run all)
    """
    for i, arg in enumerate(args):
        if arg == "--exp" and i + 1 < len(args):
            return {int(x) for x in args[i + 1].split(",")}
    return {1, 2, 3}


def main() -> None:
    t0 = time.time()
    requested = parse_experiments(sys.argv[1:])

    flush_print("=" * 60)
    flush_print("DIRECTION 3: PARTIAL OBSERVATION ANALYSIS")
    flush_print("=" * 60)
    flush_print(f"Seeds: {N_SEEDS}, Users: {NUM_USERS}, Content: {NUM_CONTENT}")
    flush_print(f"Diversity weights: {len(DIVERSITY_WEIGHTS)} points (step 0.05)")
    flush_print(f"Experiments: {sorted(requested)}")

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    cached = load_cached_results()

    # Exp 1
    if 1 in requested:
        exp1_results = run_exp1_geometry(seeds)
    elif "exp1_loso_geometry" in cached:
        exp1_results = cached["exp1_loso_geometry"]
        flush_print("\n  Exp 1: loaded from cache")
    else:
        flush_print("\n  WARNING: Exp 1 not cached and not requested, skipping")
        exp1_results = {}

    # Exp 2
    if 2 in requested:
        exp2_results = run_exp2_training(seeds)
    elif "exp2_training_loso" in cached:
        exp2_results = cached["exp2_training_loso"]
        flush_print("  Exp 2: loaded from cache")
    else:
        flush_print("  WARNING: Exp 2 not cached and not requested, skipping")
        exp2_results = {}

    # Exp 3
    if 3 in requested:
        exp3_results = run_exp3_proxy(seeds)
    elif "exp3_aggregation_proxy" in cached:
        exp3_results = cached["exp3_aggregation_proxy"]
        flush_print("  Exp 3: loaded from cache")
    else:
        exp3_results = {}

    # Summary
    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)

    if exp1_results:
        flush_print("\nExp 1 — LOSO Geometry:")
        flush_print(f"  {'Hidden':<10} {'Dom.Frac':>10} {'Avg Regret':>12} {'Max Regret':>12} {'HV Ratio':>10}")
        flush_print("  " + "-" * 56)
        for config in LOSO_CONFIGS:
            hidden = config["hidden"]
            m = exp1_results[f"hide_{hidden}"]["metrics_mean"]
            flush_print(
                f"  {hidden:<10} "
                f"{m['dominated_fraction']:>10.3f} "
                f"{m['avg_regret']:>12.4f} "
                f"{m['max_regret']:>12.4f} "
                f"{m['hypervolume_ratio_observed_2d']:>10.4f}"
            )

    if exp2_results:
        flush_print("\nExp 2 — Training-based LOSO (combined best-of-two scorer):")
        flush_print(f"  {'Hidden':<10} {'Dom.Frac':>10} {'Avg Regret':>12} {'Max Regret':>12}")
        flush_print("  " + "-" * 46)
        for config in LOSO_CONFIGS:
            hidden = config["hidden"]
            m = exp2_results[f"hide_{hidden}"]["combined_metrics_mean"]
            flush_print(
                f"  {hidden:<10} "
                f"{m['dominated_fraction']:>10.3f} "
                f"{m['avg_regret']:>12.4f} "
                f"{m['max_regret']:>12.4f}"
            )

    if exp3_results:
        flush_print("\nExp 3 — Aggregation Proxy (recovery rate):")
        hdr = f"  {'Hidden':<10} {'Oracle':>10} {'IntOrc':>10} {'StrOrc':>10} {'DW0.5':>10} {'IntBld':>10}"
        flush_print(hdr)
        flush_print("  " + "-" * 62)
        for config in LOSO_CONFIGS:
            hidden = config["hidden"]
            agg = exp3_results[f"hide_{hidden}"]["aggregated"]
            dw_agg = agg.get("diversity_knob", {})
            flush_print(
                f"  {hidden:<10} "
                f"{agg['oracle_linear']['recovery_rate_mean']:>10.3f} "
                f"{agg['interp_oracle_alpha']['recovery_rate_mean']:>10.3f} "
                f"{agg['struct_oracle_alpha']['recovery_rate_mean']:>10.3f} "
                f"{dw_agg.get('dw_0.5', {}).get('recovery_rate_mean', 0):>10.3f} "
                f"{agg['interp_blind_alpha']['recovery_rate_mean']:>10.3f}"
            )

    if exp1_results:
        exp1_ranking = sorted(
            LOSO_CONFIGS,
            key=lambda c: exp1_results[f"hide_{c['hidden']}"]["metrics_mean"]["avg_regret"],
            reverse=True,
        )
        flush_print(
            f"\n  Degradation ranking (Exp 1, by avg regret): "
            f"{' > '.join(c['hidden'] for c in exp1_ranking)}"
        )

    wall_time = time.time() - t0
    flush_print(f"\n  Wall time: {wall_time:.1f}s ({wall_time / 60:.1f}m)")

    # Save results (merge with cached)
    output = cached.copy()
    output["config"] = {
        "n_seeds": N_SEEDS,
        "num_users": NUM_USERS,
        "num_content": NUM_CONTENT,
        "num_topics": NUM_TOPICS,
        "top_k": TOP_K,
        "diversity_weights": DIVERSITY_WEIGHTS,
        "seeds": seeds,
        "n_pairs_exp2": N_PAIRS,
        "num_epochs_exp2": NUM_EPOCHS,
    }
    if exp1_results:
        output["exp1_loso_geometry"] = exp1_results
    if exp2_results:
        output["exp2_training_loso"] = exp2_results
    if exp3_results:
        output["exp3_aggregation_proxy"] = exp3_results

    save_results(output)


if __name__ == "__main__":
    main()
