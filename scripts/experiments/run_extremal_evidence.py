"""Evidence that the Goodhart effect is extremal, not regressional.

Two experiments:
1. σ-invariance: Direction condition holds across σ ∈ {0.1, 0.3, 0.5}.
   Under regressional Goodhart, direction would depend on noise level.
   Under extremal, direction depends only on cos(target, hidden).

2. Selection concentration: As N increases, top-K selection concentrates
   on fewer unique items/genres — the hallmark of extremal selection pressure.

Usage:
    uv run python scripts/experiments/run_extremal_evidence.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ml = _load("movielens", ROOT / "enhancements" / "data" / "movielens.py")
_st = _load("movielens_stakeholders", ROOT / "enhancements" / "data" / "movielens_stakeholders.py")
_al = _load("alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py")

MovieLensDataset = _ml.MovieLensDataset
build_stakeholder_configs = _st.build_stakeholder_configs
generate_movielens_content_pool = _st.generate_movielens_content_pool
generate_movielens_preferences = _st.generate_movielens_preferences
split_preferences = _st.split_preferences
NUM_GENRES = _st.NUM_GENRES

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

SEEDS = [42, 142, 242, 342, 442]
TOP_K = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def perturb_weights(w, sigma, rng):
    return w * (1 + rng.normal(0, sigma, len(w)))


def select_top_k(pool, scorer_weights, k=10):
    """Pure score-based top-K selection (δ=0). Returns indices."""
    scores = pool @ scorer_weights
    return np.argsort(scores)[-k:][::-1]


def train_and_select(pool, genres, target_weights, sigma, n_pairs, seed):
    """Train BT model and return learned weights + top-K selection."""
    rng = np.random.default_rng(seed)
    perturbed = perturb_weights(target_weights, sigma, rng)
    pref, rej = generate_movielens_preferences(pool, perturbed, n_pairs, seed)
    tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)
    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        learning_rate=0.01, num_epochs=50, batch_size=64,
    )
    model = train_with_loss(config, tp, tr, verbose=False,
                            eval_probs_preferred=ep, eval_probs_rejected=er)
    selected = select_top_k(pool, model.weights, TOP_K)
    return model.weights, selected


# ═══════════════════════════════════════════════════════════════
# Experiment 1: σ-Invariance
# ═══════════════════════════════════════════════════════════════

def run_sigma_invariance(datasets):
    """Test direction condition across σ ∈ {0.1, 0.3, 0.5}.

    Uses Method A (target rotation) for 12 points per σ per dataset.
    """
    print("\n" + "=" * 60)
    print("Experiment 1: σ-Invariance of Direction Condition")
    print("=" * 60)

    sigma_values = [0.1, 0.3, 0.5]
    all_results = []

    for ds_name, dataset, pool, genres, configs in datasets:
        print(f"\n  Dataset: {ds_name}")
        base_names = ["user", "platform", "diversity"]

        for sigma in sigma_values:
            print(f"    σ = {sigma}")
            for target_name in base_names:
                for hidden_name in base_names:
                    if target_name == hidden_name:
                        continue

                    cos_val = cosine_sim(configs[target_name], configs[hidden_name])
                    utilities = {25: [], 2000: []}

                    for n_pairs in [25, 2000]:
                        for seed in SEEDS:
                            rng = np.random.default_rng(seed)
                            perturbed = perturb_weights(configs[target_name], sigma, rng)
                            pref, rej = generate_movielens_preferences(pool, perturbed, n_pairs, seed)
                            tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)
                            config = LossConfig(
                                loss_type=LossType.BRADLEY_TERRY,
                                stakeholder=StakeholderType.USER,
                                learning_rate=0.01, num_epochs=50, batch_size=64,
                            )
                            model = train_with_loss(config, tp, tr, verbose=False,
                                                    eval_probs_preferred=ep, eval_probs_rejected=er)
                            # Evaluate hidden utility on top-K selected content
                            selected = select_top_k(pool, model.weights, TOP_K)
                            hidden_util = float(np.sum(pool[selected] @ configs[hidden_name]))
                            utilities[n_pairs].append(hidden_util)

                    mean_25 = float(np.mean(utilities[25]))
                    mean_2000 = float(np.mean(utilities[2000]))
                    improving = mean_2000 > mean_25
                    predicted_improve = cos_val > 0
                    match = (predicted_improve == improving)
                    change_pct = (mean_2000 - mean_25) / abs(mean_25) * 100 if abs(mean_25) > 1e-6 else 0.0

                    status = "✓" if match else "✗"
                    print(f"      {status} target={target_name:>10}, hidden={hidden_name:>10}: "
                          f"cos={cos_val:+.3f}, change={change_pct:+.1f}%")

                    all_results.append({
                        "dataset": ds_name,
                        "sigma": sigma,
                        "target": target_name,
                        "hidden": hidden_name,
                        "cosine": round(cos_val, 4),
                        "utility_n25": round(mean_25, 4),
                        "utility_n2000": round(mean_2000, 4),
                        "change_pct": round(change_pct, 1),
                        "improving": improving,
                        "match": match,
                    })

    # Summarize: does direction hold at every σ?
    print(f"\n  {'─' * 50}")
    print("  σ-Invariance Summary:")
    for sigma in sigma_values:
        pts = [r for r in all_results if r["sigma"] == sigma]
        matches = sum(1 for r in pts if r["match"])
        strong = [r for r in pts if abs(r["cosine"]) > 0.2]
        strong_matches = sum(1 for r in strong if r["match"])
        print(f"    σ={sigma}: {matches}/{len(pts)} total, "
              f"{strong_matches}/{len(strong)} for |cos|>0.2")

    return all_results


# ═══════════════════════════════════════════════════════════════
# Experiment 2: Selection Concentration
# ═══════════════════════════════════════════════════════════════

def run_selection_concentration(datasets):
    """Measure how top-K selection concentrates as N increases.

    At higher N, BT converges more precisely → selection concentrates
    on fewer items that score high on the target dimension.
    """
    print("\n" + "=" * 60)
    print("Experiment 2: Selection Concentration")
    print("=" * 60)

    sigma = 0.3
    n_values = [25, 50, 100, 200, 500, 2000]
    all_results = []

    for ds_name, dataset, pool, genres, configs in datasets:
        print(f"\n  Dataset: {ds_name} ({len(pool)} movies)")
        target_name = "user"
        target_w = configs[target_name]

        for n_pairs in n_values:
            seed_data = []
            for seed in SEEDS:
                weights, selected = train_and_select(pool, genres, target_w, sigma, n_pairs, seed)

                # Metrics on selected content
                unique_genres = len(np.unique(genres[selected]))
                target_util = float(np.sum(pool[selected] @ target_w))
                div_util = float(np.sum(pool[selected] @ configs["diversity"]))

                # Score concentration: std of scores across pool
                all_scores = pool @ weights
                selected_scores = all_scores[selected]
                score_percentile = float(np.mean(
                    [np.searchsorted(np.sort(all_scores), s) / len(all_scores) * 100
                     for s in selected_scores]
                ))

                seed_data.append({
                    "unique_genres": unique_genres,
                    "target_utility": target_util,
                    "diversity_utility": div_util,
                    "mean_score_percentile": score_percentile,
                    "selected_indices": selected.tolist(),
                })

            # Aggregate across seeds
            mean_genres = float(np.mean([s["unique_genres"] for s in seed_data]))
            mean_target = float(np.mean([s["target_utility"] for s in seed_data]))
            mean_div = float(np.mean([s["diversity_utility"] for s in seed_data]))
            mean_percentile = float(np.mean([s["mean_score_percentile"] for s in seed_data]))

            # Overlap: how many items appear in top-K across ALL seeds?
            all_selected = [set(s["selected_indices"]) for s in seed_data]
            intersection_size = len(set.intersection(*all_selected))
            union_size = len(set.union(*all_selected))

            print(f"    N={n_pairs:>5}: genres={mean_genres:.1f}, "
                  f"target_U={mean_target:+.2f}, div_U={mean_div:+.2f}, "
                  f"percentile={mean_percentile:.1f}%, "
                  f"overlap={intersection_size}/{TOP_K}")

            all_results.append({
                "dataset": ds_name,
                "n_pairs": n_pairs,
                "mean_unique_genres": round(mean_genres, 2),
                "mean_target_utility": round(mean_target, 4),
                "mean_diversity_utility": round(mean_div, 4),
                "mean_score_percentile": round(mean_percentile, 1),
                "cross_seed_intersection": intersection_size,
                "cross_seed_union": union_size,
                "pool_size": len(pool),
            })

    return all_results


# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("Extremal Goodhart Evidence")
    print("=" * 60)

    t0 = time.time()

    # Load datasets
    datasets = []
    for ds_name, ds_path in [("ml-100k", "data/ml-100k"), ("ml-1m", "data/ml-1m")]:
        ds_dir = ROOT / ds_path
        if not ds_dir.exists():
            print(f"  Skipping {ds_name} (data not found)")
            continue
        dataset = MovieLensDataset(str(ds_dir))
        configs = build_stakeholder_configs(dataset)
        pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
        datasets.append((ds_name, dataset, pool, genres, configs))

    sigma_results = run_sigma_invariance(datasets)
    concentration_results = run_selection_concentration(datasets)

    elapsed = time.time() - t0

    results = {
        "sigma_invariance": {
            "config": {
                "sigma_values": [0.1, 0.3, 0.5],
                "n_values": [25, 2000],
                "n_seeds": len(SEEDS),
                "method": "A (target rotation)",
            },
            "points": sigma_results,
            "summary": {},
        },
        "selection_concentration": {
            "config": {
                "sigma": 0.3,
                "n_values": [25, 50, 100, 200, 500, 2000],
                "n_seeds": len(SEEDS),
                "top_k": TOP_K,
                "target": "user",
            },
            "points": concentration_results,
        },
        "total_time_seconds": round(elapsed, 1),
    }

    # Build σ-invariance summary
    for sigma in [0.1, 0.3, 0.5]:
        pts = [r for r in sigma_results if r["sigma"] == sigma]
        strong = [r for r in pts if abs(r["cosine"]) > 0.2]
        results["sigma_invariance"]["summary"][f"sigma_{sigma}"] = {
            "total": len(pts),
            "matches": sum(1 for r in pts if r["match"]),
            "strong_total": len(strong),
            "strong_matches": sum(1 for r in strong if r["match"]),
        }

    # Direction consistency: for each (dataset, target, hidden), is the
    # direction the same across all 3 σ values?
    direction_consistent = 0
    direction_total = 0
    keys = set((r["dataset"], r["target"], r["hidden"]) for r in sigma_results)
    for key in keys:
        directions = [r["improving"] for r in sigma_results
                      if (r["dataset"], r["target"], r["hidden"]) == key]
        direction_total += 1
        if len(set(directions)) == 1:
            direction_consistent += 1
    results["sigma_invariance"]["summary"]["direction_consistency"] = {
        "consistent": direction_consistent,
        "total": direction_total,
        "fraction": round(direction_consistent / direction_total, 3) if direction_total > 0 else 0,
    }

    out_path = ROOT / "results" / "extremal_evidence.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
