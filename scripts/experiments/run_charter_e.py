#!/usr/bin/env python3
"""Charter E: Scaling with K stakeholders.

Tests how LOSO degradation changes as the number of stakeholders grows.
Two angles: (1) redundancy — do more observed stakeholders reduce regret?
(2) dimensionality — does the Pareto frontier geometry help or hurt?

Usage:
    uv run python scripts/run_charter_e.py
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
    os.path.join(
        project_root, "enhancements/reward_modeling/stakeholder_utilities.py"
    ),
)
load_module_direct(
    "enhancements.reward_modeling.alternative_losses",
    os.path.join(
        project_root, "enhancements/reward_modeling/alternative_losses.py"
    ),
)

factor_mod = load_module_direct(
    "enhancements.reward_modeling.factor_stakeholders",
    os.path.join(
        project_root, "enhancements/reward_modeling/factor_stakeholders.py"
    ),
)
generate_stakeholder_weights = factor_mod.generate_stakeholder_weights
compute_effective_rank = factor_mod.compute_effective_rank
mean_pairwise_cosine = factor_mod.mean_pairwise_cosine

frontier_mod = load_module_direct(
    "enhancements.reward_modeling.k_stakeholder_frontier",
    os.path.join(
        project_root, "enhancements/reward_modeling/k_stakeholder_frontier.py"
    ),
)
compute_k_frontier = frontier_mod.compute_k_frontier
extract_pareto_front_nd = frontier_mod.extract_pareto_front_nd
compute_regret_on_dim = frontier_mod.compute_regret_on_dim

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


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_VALUES = [3, 5, 7, 10]
N_CONFIGS = 10
N_SEEDS = 5
BASE_SEED = 42
DEFAULT_CONCENTRATION = 2.0
CONCENTRATION_SWEEP = [0.5, 2.0, 5.0]  # secondary sweep at K=5

NUM_USERS = 600
NUM_CONTENT = 100
NUM_TOPICS = 6
DIVERSITY_WEIGHTS = [round(x * 0.01, 2) for x in range(101)]  # 0.00 to 1.00, step 0.01
TOP_K = 10

OUTPUT_PATH = os.path.join(project_root, "results/charter_e_scaling.json")


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
# Core LOSO computation for K stakeholders
# ---------------------------------------------------------------------------


def compute_loso_for_config(
    base_action_probs: np.ndarray,
    content_topics: np.ndarray,
    stakeholder_weights: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Run LOSO for all hidden stakeholders in a K-stakeholder config.

    Returns per-hidden regret + aggregate metrics.
    """
    names = sorted(stakeholder_weights.keys())
    k = len(names)
    utility_dims = [f"{n}_utility" for n in names]

    # Compute K-dim frontier (once)
    frontier = compute_k_frontier(
        base_action_probs, content_topics,
        stakeholder_weights, DIVERSITY_WEIGHTS, TOP_K,
    )

    # Full K-dim Pareto
    full_pareto = extract_pareto_front_nd(frontier, utility_dims)

    per_hidden: dict[str, dict[str, float]] = {}
    for hidden_name in names:
        hidden_dim = f"{hidden_name}_utility"
        observed_dims = [d for d in utility_dims if d != hidden_dim]

        # (K-1)-dim Pareto
        loso_pareto = extract_pareto_front_nd(frontier, observed_dims)

        regret = compute_regret_on_dim(loso_pareto, frontier, hidden_dim)
        per_hidden[hidden_name] = regret

    # Aggregate
    regrets = [per_hidden[n]["avg_regret"] for n in names]
    return {
        "per_hidden": per_hidden,
        "avg_regret": float(np.mean(regrets)),
        "max_regret": float(np.max(regrets)),
        "min_regret": float(np.min(regrets)),
        "full_pareto_size": len(full_pareto),
        "frontier_size": len(frontier),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def _checkpoint(data: dict[str, Any], label: str) -> None:
    """Save partial results after each major step."""
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Load existing checkpoint if any
    existing: dict[str, Any] = {}
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    existing.update(data)
    existing["_partial"] = True
    with open(OUTPUT_PATH, "w") as f:
        json.dump(existing, f, indent=2, cls=NumpyEncoder)
    flush_print(f"  [checkpoint] {label}")


def _checkpoint_k_sweep(results_by_k: dict[int, list[dict[str, Any]]]) -> None:
    _checkpoint(
        {"k_sweep": {str(k): v for k, v in results_by_k.items()}},
        f"K sweep: {len(results_by_k)} K-values saved",
    )


def _checkpoint_conc_sweep(
    results_by_conc: dict[float, list[dict[str, Any]]],
) -> None:
    _checkpoint(
        {"concentration_sweep": {str(c): v for c, v in results_by_conc.items()}},
        f"Concentration sweep: {len(results_by_conc)} values saved",
    )


def run_k_sweep() -> dict[str, Any]:
    """Run the K-stakeholder scaling experiment."""
    flush_print("=" * 60)
    flush_print("CHARTER E: SCALING WITH K STAKEHOLDERS")
    flush_print("=" * 60)
    flush_print(f"  K values: {K_VALUES}")
    flush_print(f"  Configs per K: {N_CONFIGS}, Seeds: {N_SEEDS}")

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    results_by_k: dict[int, list[dict[str, Any]]] = {}

    for k in K_VALUES:
        flush_print(f"\n  K={k}:")
        config_results: list[dict[str, Any]] = []

        for cfg_idx in range(N_CONFIGS):
            cfg_seed = BASE_SEED + cfg_idx * 7
            weights = generate_stakeholder_weights(
                k, concentration=DEFAULT_CONCENTRATION, seed=cfg_seed,
            )
            eff_rank = compute_effective_rank(weights)
            mean_cos = mean_pairwise_cosine(weights)

            seed_results: list[dict[str, Any]] = []
            for seed in seeds:
                data = generate_synthetic_data(
                    num_users=NUM_USERS, num_content=NUM_CONTENT,
                    num_topics=NUM_TOPICS, seed=seed,
                )
                base_probs = build_base_action_probs(data)
                topics = data["content_topics"]

                loso = compute_loso_for_config(
                    base_probs, topics, weights,
                )
                seed_results.append(loso)

            # Aggregate across seeds
            avg_regrets = [s["avg_regret"] for s in seed_results]
            max_regrets = [s["max_regret"] for s in seed_results]
            pareto_fracs = [
                s["full_pareto_size"] / s["frontier_size"]
                for s in seed_results
            ]

            config_results.append({
                "config_idx": cfg_idx,
                "effective_rank": eff_rank,
                "mean_pairwise_cosine": mean_cos,
                "avg_regret_mean": float(np.mean(avg_regrets)),
                "avg_regret_std": float(np.std(avg_regrets)),
                "max_regret_mean": float(np.mean(max_regrets)),
                "pareto_fraction_mean": float(np.mean(pareto_fracs)),
            })

        results_by_k[k] = config_results

        # Print K summary
        all_avg = [c["avg_regret_mean"] for c in config_results]
        all_pareto = [c["pareto_fraction_mean"] for c in config_results]
        flush_print(
            f"    avg_regret: {np.mean(all_avg):.4f} ± {np.std(all_avg):.4f}"
            f"  pareto_frac: {np.mean(all_pareto):.3f}"
        )

        # Checkpoint after each K
        _checkpoint_k_sweep(results_by_k)

    return results_by_k


def run_concentration_sweep() -> dict[str, Any]:
    """Secondary: vary Dirichlet concentration at K=5."""
    flush_print("\n" + "-" * 60)
    flush_print("CONCENTRATION SWEEP (K=5)")
    flush_print("-" * 60)

    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]
    results_by_conc: dict[float, list[dict[str, Any]]] = {}

    for conc in CONCENTRATION_SWEEP:
        flush_print(f"\n  concentration={conc}:")
        config_results: list[dict[str, Any]] = []

        for cfg_idx in range(N_CONFIGS):
            cfg_seed = BASE_SEED + cfg_idx * 7
            weights = generate_stakeholder_weights(
                5, concentration=conc, seed=cfg_seed,
            )
            mean_cos = mean_pairwise_cosine(weights)

            seed_results: list[dict[str, Any]] = []
            for seed in seeds:
                data = generate_synthetic_data(
                    num_users=NUM_USERS, num_content=NUM_CONTENT,
                    num_topics=NUM_TOPICS, seed=seed,
                )
                base_probs = build_base_action_probs(data)
                topics = data["content_topics"]

                loso = compute_loso_for_config(
                    base_probs, topics, weights,
                )
                seed_results.append(loso)

            avg_regrets = [s["avg_regret"] for s in seed_results]
            config_results.append({
                "config_idx": cfg_idx,
                "mean_pairwise_cosine": mean_cos,
                "avg_regret_mean": float(np.mean(avg_regrets)),
                "avg_regret_std": float(np.std(avg_regrets)),
            })

        results_by_conc[conc] = config_results

        all_avg = [c["avg_regret_mean"] for c in config_results]
        all_cos = [c["mean_pairwise_cosine"] for c in config_results]
        flush_print(
            f"    avg_regret: {np.mean(all_avg):.4f} ± {np.std(all_avg):.4f}"
            f"  mean_cos: {np.mean(all_cos):.3f}"
        )

        _checkpoint_conc_sweep(results_by_conc)

    return results_by_conc


def main() -> None:
    t0 = time.time()

    k_results = run_k_sweep()
    conc_results = run_concentration_sweep()

    # Summary
    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)

    flush_print(
        f"\n  {'K':>3} {'avg_regret':>12} {'max_regret':>12} "
        f"{'pareto_frac':>12} {'eff_rank':>10}"
    )
    flush_print("  " + "-" * 50)

    baseline_regret = None
    for k in K_VALUES:
        configs = k_results[k]
        avg_r = np.mean([c["avg_regret_mean"] for c in configs])
        max_r = np.mean([c["max_regret_mean"] for c in configs])
        pf = np.mean([c["pareto_fraction_mean"] for c in configs])
        er = np.mean([c["effective_rank"] for c in configs])

        if baseline_regret is None:
            baseline_regret = avg_r
        ratio = avg_r / baseline_regret if baseline_regret > 0 else 0

        flush_print(
            f"  {k:>3} {avg_r:>12.4f} {max_r:>12.4f} "
            f"{pf:>12.3f} {er:>10.1f}"
            f"  (ratio={ratio:.3f})"
        )

    flush_print("\n  Concentration sweep (K=5):")
    flush_print(f"  {'conc':>6} {'avg_regret':>12} {'mean_cos':>10}")
    flush_print("  " + "-" * 30)
    for conc in CONCENTRATION_SWEEP:
        configs = conc_results[conc]
        avg_r = np.mean([c["avg_regret_mean"] for c in configs])
        avg_cos = np.mean([c["mean_pairwise_cosine"] for c in configs])
        flush_print(f"  {conc:>6.1f} {avg_r:>12.4f} {avg_cos:>10.3f}")

    wall_time = time.time() - t0
    flush_print(f"\n  Wall time: {wall_time:.1f}s ({wall_time / 60:.1f}m)")

    # Save
    output = {
        "config": {
            "k_values": K_VALUES,
            "n_configs": N_CONFIGS,
            "n_seeds": N_SEEDS,
            "default_concentration": DEFAULT_CONCENTRATION,
            "concentration_sweep": CONCENTRATION_SWEEP,
        },
        "k_sweep": {str(k): v for k, v in k_results.items()},
        "concentration_sweep": {
            str(c): v for c, v in conc_results.items()
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
