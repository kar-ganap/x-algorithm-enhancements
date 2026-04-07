"""LOSO and data-budget experiments on MovieLens-100K.

Exp 1: Geometric LOSO regret per hidden stakeholder (5 seeds)
Exp 2: Data budget sweep — train diversity BT on N pairs, measure recovery
Exp 3: Comparison with synthetic results

Usage:
    uv run python scripts/experiments/run_movielens_loso.py --all
    uv run python scripts/experiments/run_movielens_loso.py --exp loso
    uv run python scripts/experiments/run_movielens_loso.py --exp budget
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ml = _load("movielens", ROOT / "enhancements" / "data" / "movielens.py")
_st = _load("movielens_stakeholders", ROOT / "enhancements" / "data" / "movielens_stakeholders.py")
_al = _load("alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py")
_kf = _load("k_stakeholder_frontier", ROOT / "enhancements" / "reward_modeling" / "k_stakeholder_frontier.py")

MovieLensDataset = _ml.MovieLensDataset
build_stakeholder_configs = _st.build_stakeholder_configs
generate_movielens_content_pool = _st.generate_movielens_content_pool
generate_movielens_preferences = _st.generate_movielens_preferences
split_preferences = _st.split_preferences

compute_k_frontier = _kf.compute_k_frontier
compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier
compute_regret_on_dim = _kf.compute_regret_on_dim
extract_pareto_front_nd = _kf.extract_pareto_front_nd

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10
SEEDS = [42, 142, 242, 342, 442]
N_PAIRS_SWEEP = [0, 25, 50, 100, 200, 500, 1000, 2000]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Exp 1: Geometric LOSO
# ---------------------------------------------------------------------------

def run_loso(base_probs, genres, configs, scorer):
    """Geometric LOSO: compute full frontier, project to 2D, measure regret."""
    print("\n" + "=" * 60)
    print("Exp 1: Geometric LOSO Regret")
    print("=" * 60)

    stakeholder_names = ["user", "platform", "diversity"]
    utility_keys = {s: f"{s}_utility" for s in stakeholder_names}

    # Compute full 3-stakeholder frontier (seed-independent for geometric LOSO
    # since content pool is fixed)
    full_frontier = compute_k_frontier(
        base_probs, genres, configs, DIVERSITY_WEIGHTS,
        top_k=TOP_K, scorer_weights=scorer,
    )
    print(f"  Full frontier: {len(full_frontier)} points")

    results = {}
    for hidden in stakeholder_names:
        hidden_key = utility_keys[hidden]
        observed = [s for s in stakeholder_names if s != hidden]
        observed_keys = [utility_keys[s] for s in observed]

        # Extract 2D Pareto front from observed dimensions
        pareto_2d = extract_pareto_front_nd(full_frontier, observed_keys)

        # Measure regret on hidden dimension
        regret = compute_regret_on_dim(pareto_2d, full_frontier, hidden_key)

        status = "✓" if regret["avg_regret"] >= 0 else "✗"
        print(f"  {status} Hide {hidden}: avg_regret={regret['avg_regret']:.3f}, "
              f"max_regret={regret['max_regret']:.3f}, "
              f"2D Pareto points={len(pareto_2d)}/{len(full_frontier)}")

        results[f"hide_{hidden}"] = {
            "avg_regret": round(regret["avg_regret"], 4),
            "max_regret": round(regret["max_regret"], 4),
            "min_regret": round(regret["min_regret"], 4),
            "pareto_2d_size": len(pareto_2d),
        }

    # Regret ranking
    regret_ranking = sorted(results.items(), key=lambda x: x[1]["avg_regret"], reverse=True)
    print(f"\n  Regret ranking: {' > '.join(k for k, _ in regret_ranking)}")

    return results


# ---------------------------------------------------------------------------
# Exp 2: Data Budget
# ---------------------------------------------------------------------------

def run_data_budget(base_probs, genres, pool, configs, scorer, seeds):
    """Sweep N diversity preference pairs, measure regret recovery.

    Uses compute_scorer_eval_frontier: content selection with a learned
    scorer, utility evaluation with true stakeholder weights. This avoids
    scale mismatch between learned and true weights in the utility space.
    """
    print("\n" + "=" * 60)
    print(f"Exp 2: Data Budget ({len(seeds)} seeds × {len(N_PAIRS_SWEEP)} N values)")
    print("=" * 60)

    # Full frontier (gold standard) — uses true diversity weights as scorer
    full_frontier = compute_k_frontier(
        base_probs, genres, configs, DIVERSITY_WEIGHTS,
        top_k=TOP_K, scorer_weights=scorer,
    )

    # LOSO baseline: select using mean(user, platform) as scorer, no diversity info
    loso_scorer = (configs["user"] + configs["platform"]) / 2.0
    loso_frontier = compute_scorer_eval_frontier(
        base_probs, genres, loso_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
    )
    loso_regret = compute_regret_on_dim(loso_frontier, full_frontier, "diversity_utility")
    print(f"  LOSO baseline regret: {loso_regret['avg_regret']:.3f}")

    sweep_results = []
    for n_pairs in N_PAIRS_SWEEP:
        per_seed_regrets = []
        per_seed_spearman = []

        for seed in seeds:
            if n_pairs == 0:
                # No diversity data — use LOSO scorer
                frontier = compute_scorer_eval_frontier(
                    base_probs, genres, loso_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
                )
            else:
                # Train diversity BT on N pairs
                pref, rej = generate_movielens_preferences(
                    pool, configs["diversity"], n_pairs=n_pairs, seed=seed,
                )
                tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=seed)
                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.SOCIETY,
                    learning_rate=0.01,
                    num_epochs=50,
                    batch_size=64,
                )
                model = train_with_loss(config, tp, tr, verbose=False,
                                        eval_probs_preferred=ep, eval_probs_rejected=er)

                # Use learned diversity weights as content scorer
                # Combine with user+platform for a balanced scorer
                learned_scorer = (configs["user"] + configs["platform"] + model.weights) / 3.0
                frontier = compute_scorer_eval_frontier(
                    base_probs, genres, learned_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
                )

                # Track weight recovery
                rho, _ = spearmanr(model.weights, configs["diversity"])
                per_seed_spearman.append(float(rho))

            regret = compute_regret_on_dim(frontier, full_frontier, "diversity_utility")
            per_seed_regrets.append(regret["avg_regret"])

        mean_regret = float(np.mean(per_seed_regrets))
        std_regret = float(np.std(per_seed_regrets))
        mean_spearman = float(np.mean(per_seed_spearman)) if per_seed_spearman else None

        status = "✓" if mean_regret < loso_regret["avg_regret"] or n_pairs == 0 else "~"
        spearman_str = f", ρ={mean_spearman:.3f}" if mean_spearman is not None else ""
        print(f"  {status} N={n_pairs:>4}: regret={mean_regret:.3f} ± {std_regret:.3f}{spearman_str}")

        sweep_results.append({
            "n_pairs": n_pairs,
            "avg_regret_mean": round(mean_regret, 4),
            "avg_regret_std": round(std_regret, 4),
            "per_seed_regret": [round(r, 4) for r in per_seed_regrets],
            "weight_spearman_mean": round(mean_spearman, 4) if mean_spearman else None,
        })

    # Recovery at N=25
    loso_val = sweep_results[0]["avg_regret_mean"]  # N=0
    n25_val = next((s["avg_regret_mean"] for s in sweep_results if s["n_pairs"] == 25), None)
    recovery_25 = None
    if n25_val is not None and loso_val > 0:
        recovery_25 = round(1.0 - n25_val / loso_val, 4)

    return {
        "loso_baseline_regret": round(loso_regret["avg_regret"], 4),
        "sweep": sweep_results,
        "recovery_at_25": recovery_25,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MovieLens LOSO + data budget")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--exp", type=str, choices=["loso", "budget"])
    args = parser.parse_args()

    if not args.all and not args.exp:
        args.all = True

    print("=" * 60)
    print("MovieLens LOSO + Data Budget Experiments")
    print("=" * 60)

    data_dir = ROOT / "data" / "ml-100k"
    if not data_dir.exists():
        print(f"ERROR: MovieLens data not found at {data_dir}")
        sys.exit(1)

    t0 = time.time()
    dataset = MovieLensDataset(str(data_dir))
    configs = build_stakeholder_configs(dataset)
    pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
    base_probs = pool[np.newaxis, :, :]  # [1, M, 19]
    scorer = configs["platform"]  # Platform weights as engagement scorer
    print(f"Loaded: {pool.shape[0]} movies, {pool.shape[1]} genres")

    results = {
        "config": {
            "n_movies": int(pool.shape[0]),
            "feature_dim": int(pool.shape[1]),
            "top_k": TOP_K,
            "diversity_weights": DIVERSITY_WEIGHTS,
            "n_seeds": len(SEEDS),
            "scorer": "platform_weights",
        }
    }

    if args.all or args.exp == "loso":
        results["loso_regret"] = run_loso(base_probs, genres, configs, scorer)

    if args.all or args.exp == "budget":
        results["data_budget"] = run_data_budget(base_probs, genres, pool, configs, scorer, SEEDS)

    # Comparison with synthetic
    synthetic_path = ROOT / "results" / "partial_observation.json"
    if synthetic_path.exists():
        with open(synthetic_path) as f:
            synthetic = json.load(f)
        syn_loso = synthetic.get("exp1_loso_geometry", {})
        results["comparison_with_synthetic"] = {
            "synthetic_loso": {
                "hide_society": round(syn_loso.get("hide_society", {}).get("metrics_mean", {}).get("avg_regret", 0), 4),
                "hide_platform": round(syn_loso.get("hide_platform", {}).get("metrics_mean", {}).get("avg_regret", 0), 4),
                "hide_user": round(syn_loso.get("hide_user", {}).get("metrics_mean", {}).get("avg_regret", 0), 4),
            },
        }
        if "loso_regret" in results:
            results["comparison_with_synthetic"]["movielens_loso"] = {
                k: v["avg_regret"] for k, v in results["loso_regret"].items()
            }

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    print(f"\n{'=' * 60}")
    print(f"Complete in {elapsed:.0f}s")
    print("=" * 60)

    out_path = ROOT / "results" / "movielens_loso.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
