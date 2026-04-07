"""Goodhart effect experiment on MovieLens-100K.

Strategy 1: Better specification (reduce σ) at fixed N=2000
Strategy 2: More data (increase N) at fixed σ=0.3

Shows that more data amplifies misspecified utility weights — a Goodhart
effect where the BT model more faithfully learns the wrong objective.

Usage:
    uv run python scripts/experiments/run_movielens_goodhart.py

Output:
    results/movielens_goodhart.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

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

compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10

# Experiment parameters
SIGMA_SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
N_SWEEP = [25, 50, 100, 200, 500, 1000, 2000]
FIXED_SIGMA = 0.3
FIXED_N = 2000
N_SEEDS = 5
N_NOISE_SAMPLES = 5
STAKEHOLDER = "user"


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def perturb_weights(w: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative Gaussian perturbation: w * (1 + N(0, σ))."""
    return w * (1 + rng.normal(0, sigma, len(w)))


def hausdorff_distance(
    frontier_a: list[dict], frontier_b: list[dict], dims: list[str],
) -> float:
    """Hausdorff distance between two frontiers in utility space."""
    if not frontier_a or not frontier_b:
        return 0.0
    vals_a = np.array([[p[d] for d in dims] for p in frontier_a])
    vals_b = np.array([[p[d] for d in dims] for p in frontier_b])
    d_a_to_b = max(
        min(float(np.linalg.norm(a - b)) for b in vals_b) for a in vals_a
    )
    d_b_to_a = max(
        min(float(np.linalg.norm(a - b)) for a in vals_a) for b in vals_b
    )
    return max(d_a_to_b, d_b_to_a)


def run_single_condition(
    pool: np.ndarray,
    genres: np.ndarray,
    base_probs: np.ndarray,
    true_configs: dict[str, np.ndarray],
    true_weights: np.ndarray,
    baseline_frontier: list[dict],
    utility_dims: list[str],
    sigma: float,
    n_pairs: int,
    seed: int,
) -> float:
    """Run one condition: perturb, generate misspecified prefs, train, measure."""
    rng = np.random.default_rng(seed)

    # Perturb the target stakeholder's weights
    perturbed = perturb_weights(true_weights, sigma, rng)

    # Generate preference pairs using MISSPECIFIED weights
    pref, rej = generate_movielens_preferences(pool, perturbed, n_pairs, seed)
    tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)

    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        learning_rate=0.01, num_epochs=50, batch_size=64,
    )
    model = train_with_loss(config, tp, tr, verbose=False,
                            eval_probs_preferred=ep, eval_probs_rejected=er)

    # Compute frontier using learned (misspecified) weights as scorer
    learned_frontier = compute_scorer_eval_frontier(
        base_probs, genres, model.weights, true_configs,
        DIVERSITY_WEIGHTS, top_k=TOP_K,
    )

    return hausdorff_distance(learned_frontier, baseline_frontier, utility_dims)


def main():
    print("=" * 60)
    print("Phase 5: Goodhart Effect on MovieLens")
    print("=" * 60)

    data_dir = ROOT / "data" / "ml-100k"
    if not data_dir.exists():
        print(f"ERROR: MovieLens data not found at {data_dir}")
        sys.exit(1)

    t0 = time.time()
    dataset = MovieLensDataset(str(data_dir))
    configs = build_stakeholder_configs(dataset)
    pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
    base_probs = pool[np.newaxis, :, :]
    true_weights = configs[STAKEHOLDER]

    stakeholder_names = sorted(configs.keys())
    utility_dims = [f"{s}_utility" for s in stakeholder_names]

    print(f"Loaded: {pool.shape[0]} movies, {pool.shape[1]} genres")
    print(f"Target stakeholder: {STAKEHOLDER}")

    # Compute baseline frontier using true weights as scorer
    baseline_scorer = (configs["user"] + configs["platform"] + configs["diversity"]) / 3.0
    baseline_frontier = compute_scorer_eval_frontier(
        base_probs, genres, baseline_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
    )
    print(f"Baseline frontier: {len(baseline_frontier)} points")

    results = {
        "config": {
            "n_seeds": N_SEEDS,
            "n_noise_samples": N_NOISE_SAMPLES,
            "fixed_sigma": FIXED_SIGMA,
            "fixed_n": FIXED_N,
            "sigma_sweep": SIGMA_SWEEP,
            "n_sweep": N_SWEEP,
            "stakeholder": STAKEHOLDER,
            "feature_dim": int(pool.shape[1]),
        }
    }

    # --- Strategy 1: Better specification (reduce σ, fixed N=2000) ---
    print(f"\nStrategy 1: Better specification (fixed N={FIXED_N})")
    print("-" * 40)
    strategy1 = {}
    for sigma in SIGMA_SWEEP:
        hausdorffs = []
        for seed_idx in range(N_SEEDS):
            for noise_idx in range(N_NOISE_SAMPLES):
                seed = 42 + seed_idx * 1000 + noise_idx * 100
                hd = run_single_condition(
                    pool, genres, base_probs, configs, true_weights,
                    baseline_frontier, utility_dims, sigma, FIXED_N, seed,
                )
                hausdorffs.append(hd)

        mean_hd = float(np.mean(hausdorffs))
        std_hd = float(np.std(hausdorffs))
        print(f"  σ={sigma:.2f}: Hausdorff = {mean_hd:.3f} ± {std_hd:.3f}")
        strategy1[str(sigma)] = {
            "hausdorff_mean": round(mean_hd, 4),
            "hausdorff_std": round(std_hd, 4),
            "n_samples": len(hausdorffs),
        }

    results["strategy1_better_spec"] = strategy1

    # --- Strategy 2: More data (increase N, fixed σ=0.3) ---
    print(f"\nStrategy 2: More data (fixed σ={FIXED_SIGMA})")
    print("-" * 40)
    strategy2 = {}
    for n_pairs in N_SWEEP:
        hausdorffs = []
        for seed_idx in range(N_SEEDS):
            for noise_idx in range(N_NOISE_SAMPLES):
                seed = 42 + seed_idx * 1000 + noise_idx * 100
                hd = run_single_condition(
                    pool, genres, base_probs, configs, true_weights,
                    baseline_frontier, utility_dims, FIXED_SIGMA, n_pairs, seed,
                )
                hausdorffs.append(hd)

        mean_hd = float(np.mean(hausdorffs))
        std_hd = float(np.std(hausdorffs))
        print(f"  N={n_pairs:>4}: Hausdorff = {mean_hd:.3f} ± {std_hd:.3f}")
        strategy2[str(n_pairs)] = {
            "hausdorff_mean": round(mean_hd, 4),
            "hausdorff_std": round(std_hd, 4),
            "n_samples": len(hausdorffs),
        }

    results["strategy2_more_data"] = strategy2

    # --- Goodhart detection ---
    n_values = N_SWEEP
    hd_values = [strategy2[str(n)]["hausdorff_mean"] for n in n_values]

    # Find minimum
    min_idx = int(np.argmin(hd_values))
    min_n = n_values[min_idx]
    min_hd = hd_values[min_idx]

    # Check if any later N has higher Hausdorff
    goodhart_detected = any(hd_values[i] > min_hd * 1.1 for i in range(min_idx + 1, len(hd_values)))

    # Find peak after minimum
    if min_idx < len(hd_values) - 1:
        post_min = hd_values[min_idx + 1:]
        peak_idx = min_idx + 1 + int(np.argmax(post_min))
        peak_n = n_values[peak_idx]
        peak_hd = hd_values[peak_idx]
    else:
        peak_n = min_n
        peak_hd = min_hd

    results["goodhart_detected"] = goodhart_detected
    results["goodhart_minimum_n"] = min_n
    results["goodhart_minimum_hausdorff"] = round(min_hd, 4)
    results["goodhart_peak_n"] = peak_n
    results["goodhart_peak_hausdorff"] = round(peak_hd, 4)

    # Strategy 1 monotonicity check
    s1_values = [strategy1[str(s)]["hausdorff_mean"] for s in SIGMA_SWEEP]
    s1_monotonic = all(s1_values[i] >= s1_values[i + 1] - 0.1 for i in range(len(s1_values) - 1))
    results["strategy1_monotonic"] = s1_monotonic

    # Comparison with synthetic
    results["comparison_with_synthetic"] = {
        "synthetic_minimum_n": 100,
        "synthetic_peak_n": 500,
        "synthetic_minimum_hausdorff": 2.27,
        "synthetic_peak_hausdorff": 6.82,
        "movielens_minimum_n": min_n,
        "movielens_peak_n": peak_n,
        "movielens_minimum_hausdorff": round(min_hd, 4),
        "movielens_peak_hausdorff": round(peak_hd, 4),
    }

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    # Summary
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Strategy 1 monotonic: {'✓' if s1_monotonic else '✗'}")
    print(f"  Goodhart detected: {'✓' if goodhart_detected else '✗'}")
    print(f"  Minimum at N={min_n} (Hausdorff={min_hd:.3f})")
    print(f"  Peak at N={peak_n} (Hausdorff={peak_hd:.3f})")
    print(f"  Complete in {elapsed:.0f}s")
    print("=" * 60)

    out_path = ROOT / "results" / "movielens_goodhart.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
