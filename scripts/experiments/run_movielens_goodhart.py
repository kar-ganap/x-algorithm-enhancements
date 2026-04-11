"""Goodhart effect experiment (registry-driven, dataset-agnostic).

Tests whether more training data with misspecified utility weights
degrades true stakeholder utilities and topic-level entropy.

Metrics (replacing Hausdorff from Phase 5/5b):
  - True per-stakeholder utility (evaluated with TRUE weights)
  - Topic entropy of selected set (nonlinear, set-level, scale-invariant)

Strategy 1: Better specification (reduce σ) at fixed N=2000
Strategy 2: More data (increase N) at fixed σ=0.3

Usage:
    uv run python scripts/experiments/run_movielens_goodhart.py
    uv run python scripts/experiments/run_movielens_goodhart.py --dataset mind-small
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _dataset_registry import REGISTRY, load_dataset  # noqa: E402


def _load(name: str, path: Path):
    """Load reward-modeling modules via importlib (registry handles dataset modules)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_al = _load("alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py")
_kf = _load("k_stakeholder_frontier", ROOT / "enhancements" / "reward_modeling" / "k_stakeholder_frontier.py")

compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10

SIGMA_SWEEP = [0.5, 0.3, 0.2, 0.1, 0.05, 0.0]
N_SWEEP = [25, 50, 100, 200, 500, 1000, 2000]
FIXED_SIGMA = 0.3
FIXED_N = 2000
N_SEEDS = 5
N_NOISE_SAMPLES = 5


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
    return w * (1 + rng.normal(0, sigma, len(w)))


def compute_genre_entropy(selected_indices: np.ndarray, content_features: np.ndarray) -> float:
    """Shannon entropy of genre distribution in selected set."""
    genres = content_features[selected_indices]
    genre_counts = np.sum(genres > 0, axis=0).astype(float)
    total = np.sum(genre_counts)
    if total == 0:
        return 0.0
    probs = genre_counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def select_top_k(
    content_features: np.ndarray,
    content_genres: np.ndarray,
    scorer_weights: np.ndarray,
    diversity_weight: float,
    top_k: int,
) -> np.ndarray:
    """Greedy top-K selection with diversity bonus (same logic as compute_k_frontier)."""
    n_content = len(content_features)
    num_topics = int(np.max(content_genres)) + 1
    engagement_scores = content_features @ scorer_weights

    if diversity_weight > 0:
        selected = []
        remaining = list(range(n_content))
        topic_counts = np.zeros(num_topics)
        for _ in range(top_k):
            if not remaining:
                break
            best_idx = max(
                remaining,
                key=lambda idx: (
                    (1 - diversity_weight) * engagement_scores[idx]
                    + diversity_weight / (topic_counts[content_genres[idx]] + 1)
                ),
            )
            selected.append(best_idx)
            remaining.remove(best_idx)
            topic_counts[content_genres[best_idx]] += 1
        return np.array(selected)
    else:
        return np.argsort(engagement_scores)[-top_k:][::-1]


def run_single_condition(
    ds,
    pool: np.ndarray,
    genres: np.ndarray,
    base_probs: np.ndarray,
    true_configs: dict[str, np.ndarray],
    true_weights: np.ndarray,
    sigma: float,
    n_pairs: int,
    seed: int,
) -> dict:
    """Run one condition: perturb, train, evaluate true utilities + entropy."""
    rng = np.random.default_rng(seed)
    perturbed = perturb_weights(true_weights, sigma, rng)

    pref, rej = ds.generate_preferences(perturbed, n_pairs, seed)
    tp, tr, ep, er = ds.stakeholders_mod.split_preferences(pref, rej, 0.2, seed)

    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        learning_rate=0.01, num_epochs=50, batch_size=64,
    )
    model = train_with_loss(config, tp, tr, verbose=False,
                            eval_probs_preferred=ep, eval_probs_rejected=er)

    # Use learned weights as SCORER, true weights as EVALUATOR
    frontier = compute_scorer_eval_frontier(
        base_probs, genres, model.weights, true_configs,
        DIVERSITY_WEIGHTS, top_k=TOP_K,
    )

    # Extract mean utilities across all diversity weights.
    # NOTE: byte-equivalence requires preserving the legacy
    # ``sorted(configs.keys())`` order on MovieLens, where the alphabetical
    # order ("diversity", "platform", "user") differs from the canonical
    # primary_stakeholder_order ("user", "platform", "diversity"). The
    # output JSON key set is identical either way; only iteration ordering
    # of the ``mean_utilities`` dict construction differs, which jq -S
    # normalizes during diffing.
    stakeholder_names = sorted(true_configs.keys())
    mean_utilities = {}
    for name in stakeholder_names:
        key = f"{name}_utility"
        mean_utilities[key] = float(np.mean([p[key] for p in frontier]))

    # Genre entropy at δ=0.0 (pure engagement) and δ=0.5 (balanced)
    selected_d0 = select_top_k(pool, genres, model.weights, 0.0, TOP_K)
    selected_d05 = select_top_k(pool, genres, model.weights, 0.5, TOP_K)
    entropy_d0 = compute_genre_entropy(selected_d0, pool)
    entropy_d05 = compute_genre_entropy(selected_d05, pool)

    # Also count unique genres
    n_genres_d0 = len(np.unique(np.argmax(pool[selected_d0], axis=1)))
    n_genres_d05 = len(np.unique(np.argmax(pool[selected_d05], axis=1)))

    return {
        **mean_utilities,
        "entropy_delta0": entropy_d0,
        "entropy_delta05": entropy_d05,
        "n_unique_genres_d0": int(n_genres_d0),
        "n_unique_genres_d05": int(n_genres_d05),
        "eval_accuracy": float(model.eval_accuracy) if model.eval_accuracy else None,
    }


def aggregate(results_list: list[dict]) -> dict:
    """Aggregate a list of per-condition result dicts into mean ± std."""
    keys = results_list[0].keys()
    agg = {}
    for key in keys:
        vals = [r[key] for r in results_list if r[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    return agg


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Goodhart effect (registry-driven)")
    parser.add_argument("--data", default=None,
                        help="(deprecated) data directory; use --dataset")
    parser.add_argument("--dataset", default=None, choices=list(REGISTRY.keys()),
                        help="Dataset name from registry")
    args = parser.parse_args()

    if args.dataset:
        dataset_name = args.dataset
    elif args.data:
        warnings.warn(
            "--data is deprecated; use --dataset {ml-100k,ml-1m,mind-small,amazon-kindle}",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset_name = Path(args.data).name
    else:
        dataset_name = "ml-100k"

    print("=" * 60)
    print(f"Goodhart with Set-Level Metrics ({dataset_name})")
    print("=" * 60)

    t0 = time.time()
    ds = load_dataset(dataset_name)
    configs = ds.configs
    pool, genres = ds.pool, ds.topics
    base_probs = pool[np.newaxis, :, :]
    target_stakeholder = ds.spec.primary_stakeholder_order[0]
    true_weights = configs[target_stakeholder]
    diversity_name = ds.spec.diversity_stakeholder
    diversity_util_key = f"{diversity_name}_utility_mean"

    print(f"Loaded: {pool.shape[0]} items, {pool.shape[1]} features")
    print(f"Target stakeholder: {target_stakeholder}")

    # Baseline: true weights as scorer
    baseline = run_single_condition(
        ds, pool, genres, base_probs, configs, true_weights,
        sigma=0.0, n_pairs=FIXED_N, seed=42,
    )
    print(f"\nBaseline (σ=0, N=2000):")
    if diversity_util_key in baseline:
        print(f"  {diversity_name}_utility: {baseline[diversity_util_key]:.4f}")
    else:
        print(f"  {diversity_name}_utility: N/A")

    # Actually compute baseline separately as reference
    baseline_selected_d0 = select_top_k(pool, genres, true_weights, 0.0, TOP_K)
    baseline_entropy_d0 = compute_genre_entropy(baseline_selected_d0, pool)
    baseline_n_genres_d0 = len(np.unique(np.argmax(pool[baseline_selected_d0], axis=1)))
    print(f"  True scorer entropy (δ=0): {baseline_entropy_d0:.4f}, genres: {baseline_n_genres_d0}")

    results = {
        "config": {
            "n_seeds": N_SEEDS,
            "n_noise_samples": N_NOISE_SAMPLES,
            "fixed_sigma": FIXED_SIGMA,
            "fixed_n": FIXED_N,
            "sigma_sweep": SIGMA_SWEEP,
            "n_sweep": N_SWEEP,
            "stakeholder": target_stakeholder,
            "feature_dim": int(pool.shape[1]),
            "top_k": TOP_K,
            # Preserve the 2-stakeholder + 2-metric layout from the
            # original MovieLens-only script for regression byte-equivalence.
            "metrics": [
                f"{target_stakeholder}_utility",
                f"{diversity_name}_utility",
                "genre_entropy",
                "n_unique_genres",
            ],
        },
        "baseline": {
            "entropy_delta0": baseline_entropy_d0,
            "n_unique_genres_d0": baseline_n_genres_d0,
        },
    }

    # --- Strategy 1: Better specification ---
    print(f"\nStrategy 1: Better specification (fixed N={FIXED_N})")
    print("-" * 50)
    strategy1 = {}
    target_util_key = f"{target_stakeholder}_utility_mean"
    for sigma in SIGMA_SWEEP:
        condition_results = []
        for seed_idx in range(N_SEEDS):
            for noise_idx in range(N_NOISE_SAMPLES):
                seed = 42 + seed_idx * 1000 + noise_idx * 100
                r = run_single_condition(
                    ds, pool, genres, base_probs, configs, true_weights,
                    sigma, FIXED_N, seed,
                )
                condition_results.append(r)

        agg = aggregate(condition_results)
        print(f"  σ={sigma:.2f}: div_util={agg.get(diversity_util_key, 'N/A'):.4f}, "
              f"entropy_d0={agg.get('entropy_delta0_mean', 'N/A'):.4f}, "
              f"target_util={agg.get(target_util_key, 'N/A'):.4f}")
        strategy1[str(sigma)] = agg

    results["strategy1_better_spec"] = strategy1

    # --- Strategy 2: More data ---
    print(f"\nStrategy 2: More data (fixed σ={FIXED_SIGMA})")
    print("-" * 50)
    strategy2 = {}
    for n_pairs in N_SWEEP:
        condition_results = []
        for seed_idx in range(N_SEEDS):
            for noise_idx in range(N_NOISE_SAMPLES):
                seed = 42 + seed_idx * 1000 + noise_idx * 100
                r = run_single_condition(
                    ds, pool, genres, base_probs, configs, true_weights,
                    FIXED_SIGMA, n_pairs, seed,
                )
                condition_results.append(r)

        agg = aggregate(condition_results)
        print(f"  N={n_pairs:>4}: div_util={agg.get(diversity_util_key, 'N/A'):.4f}, "
              f"entropy_d0={agg.get('entropy_delta0_mean', 'N/A'):.4f}, "
              f"target_util={agg.get(target_util_key, 'N/A'):.4f}, "
              f"genres_d0={agg.get('n_unique_genres_d0_mean', 'N/A'):.1f}")
        strategy2[str(n_pairs)] = agg

    results["strategy2_more_data"] = strategy2

    # --- Goodhart detection ---
    n_values = N_SWEEP
    # Check if target utility increases with N (positive control)
    user_utils = [strategy2[str(n)].get(target_util_key, 0) for n in n_values]
    user_increasing = user_utils[-1] > user_utils[0]

    # Check if diversity utility decreases with N (Goodhart signal)
    div_utils = [strategy2[str(n)].get(diversity_util_key, 0) for n in n_values]
    div_min_idx = int(np.argmin(div_utils))
    # Goodhart: diversity utility has a maximum, then decreases
    div_max_idx = int(np.argmax(div_utils))
    div_decreasing_after_max = any(div_utils[i] < div_utils[div_max_idx] * 0.95
                                    for i in range(div_max_idx + 1, len(div_utils)))

    # Check if entropy decreases with N (set-level narrowing)
    entropies = [strategy2[str(n)].get("entropy_delta0_mean", 0) for n in n_values]
    entropy_max_idx = int(np.argmax(entropies))
    entropy_decreasing = any(entropies[i] < entropies[entropy_max_idx] * 0.95
                              for i in range(entropy_max_idx + 1, len(entropies)))

    results["goodhart_detected"] = {
        "user_increasing": bool(user_increasing),
        "diversity_decreasing_after_peak": bool(div_decreasing_after_max),
        "entropy_decreasing_after_peak": bool(entropy_decreasing),
        "diversity_peak_n": n_values[div_max_idx],
        "entropy_peak_n": n_values[entropy_max_idx],
    }

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    # Summary
    print(f"\n{'=' * 60}")
    print("Goodhart Detection:")
    print(f"  User utility increasing with N: {'✓' if user_increasing else '✗'}")
    print(f"  Diversity utility decreasing after peak: {'✓' if div_decreasing_after_max else '✗'} (peak at N={n_values[div_max_idx]})")
    print(f"  Genre entropy decreasing after peak: {'✓' if entropy_decreasing else '✗'} (peak at N={n_values[entropy_max_idx]})")
    print(f"\n  Complete in {elapsed:.0f}s")
    print("=" * 60)

    out_path = ROOT / "results" / f"{dataset_name}_goodhart.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
