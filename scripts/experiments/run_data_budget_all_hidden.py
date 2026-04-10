"""Data budget sweep for ALL hidden stakeholders.

The existing data budget experiment (run_movielens_loso.py) only tests
diversity-hidden. This generalizes to all 3 hidden stakeholders so we can
claim "N pairs recover X% of harm" across hidden stakeholders, not just
the anti-correlated one.

For each hidden stakeholder h:
  1. Build LOSO scorer from the other 2 (mean of their weights)
  2. Compute LOSO baseline regret on h
  3. Sweep N ∈ {0, 25, 50, 100, 200, 500}: train BT on h's preferences,
     use learned weights + observed stakeholders as new scorer, measure
     regret on h

Usage:
    uv run python scripts/experiments/run_data_budget_all_hidden.py
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
_kf = _load("k_stakeholder_frontier", ROOT / "enhancements" / "reward_modeling" / "k_stakeholder_frontier.py")

MovieLensDataset = _ml.MovieLensDataset
build_stakeholder_configs = _st.build_stakeholder_configs
generate_movielens_content_pool = _st.generate_movielens_content_pool
generate_movielens_preferences = _st.generate_movielens_preferences
split_preferences = _st.split_preferences

compute_k_frontier = _kf.compute_k_frontier
compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier
compute_regret_on_dim = _kf.compute_regret_on_dim

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10
SEEDS = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942,
         1042, 1142, 1242, 1342, 1442, 1542, 1642, 1742, 1842, 1942]
N_PAIRS_SWEEP = [0, 25, 50, 100, 200, 500]
STAKEHOLDER_NAMES = ["user", "platform", "diversity"]


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def run_data_budget_for_hidden(
    base_probs, genres, pool, configs, full_scorer, hidden_name, seeds
):
    """Sweep N preference pairs for one hidden stakeholder."""
    print(f"\n  Hidden: {hidden_name}")
    hidden_key = f"{hidden_name}_utility"
    observed = [s for s in STAKEHOLDER_NAMES if s != hidden_name]

    # Gold standard: full frontier using full scorer
    full_frontier = compute_k_frontier(
        base_probs, genres, configs, DIVERSITY_WEIGHTS,
        top_k=TOP_K, scorer_weights=full_scorer,
    )

    # LOSO scorer: mean of observed stakeholders
    loso_scorer = sum(configs[s] for s in observed) / len(observed)
    loso_frontier = compute_scorer_eval_frontier(
        base_probs, genres, loso_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
    )
    loso_regret = compute_regret_on_dim(loso_frontier, full_frontier, hidden_key)
    print(f"    LOSO baseline regret: {loso_regret['avg_regret']:.3f}")

    sweep_results = []
    for n_pairs in N_PAIRS_SWEEP:
        per_seed_regrets = []

        for seed in seeds:
            if n_pairs == 0:
                # No hidden data — use LOSO scorer
                frontier = compute_scorer_eval_frontier(
                    base_probs, genres, loso_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
                )
            else:
                # Train BT on hidden stakeholder's preferences
                pref, rej = generate_movielens_preferences(
                    pool, configs[hidden_name], n_pairs=n_pairs, seed=seed,
                )
                tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=seed)
                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,  # label only, not semantic
                    learning_rate=0.01,
                    num_epochs=50,
                    batch_size=64,
                )
                model = train_with_loss(config, tp, tr, verbose=False,
                                        eval_probs_preferred=ep, eval_probs_rejected=er)

                # Composite scorer: observed stakeholders + learned hidden
                learned_scorer = (sum(configs[s] for s in observed) + model.weights) / 3.0
                frontier = compute_scorer_eval_frontier(
                    base_probs, genres, learned_scorer, configs, DIVERSITY_WEIGHTS, top_k=TOP_K,
                )

            regret = compute_regret_on_dim(frontier, full_frontier, hidden_key)
            per_seed_regrets.append(regret["avg_regret"])

        mean_regret = float(np.mean(per_seed_regrets))
        std_regret = float(np.std(per_seed_regrets))

        status = "✓" if mean_regret < loso_regret["avg_regret"] or n_pairs == 0 else "~"
        print(f"    {status} N={n_pairs:>4}: regret={mean_regret:.3f} ± {std_regret:.3f}")

        sweep_results.append({
            "n_pairs": n_pairs,
            "avg_regret_mean": round(mean_regret, 4),
            "avg_regret_std": round(std_regret, 4),
            "per_seed_regret": [round(r, 4) for r in per_seed_regrets],
        })

    # Recovery at N=25
    loso_val = sweep_results[0]["avg_regret_mean"]
    n25_val = next((s["avg_regret_mean"] for s in sweep_results if s["n_pairs"] == 25), None)
    recovery_25 = None
    recovery_25_ci = None
    if n25_val is not None and loso_val > 0:
        recovery_25 = 1.0 - n25_val / loso_val
        # CI from per-seed values
        n25_entry = next(s for s in sweep_results if s["n_pairs"] == 25)
        per_seed = np.array(n25_entry["per_seed_regret"])
        per_seed_recovery = 1.0 - per_seed / loso_val
        # 95% CI via normal approximation across 20 seeds
        mean_r = float(np.mean(per_seed_recovery))
        std_r = float(np.std(per_seed_recovery, ddof=1))
        se = std_r / np.sqrt(len(per_seed_recovery))
        recovery_25_ci = [round(mean_r - 1.96 * se, 4),
                          round(mean_r + 1.96 * se, 4)]

    return {
        "hidden": hidden_name,
        "loso_baseline_regret": round(loso_regret["avg_regret"], 4),
        "sweep": sweep_results,
        "recovery_at_25": round(recovery_25, 4) if recovery_25 else None,
        "recovery_at_25_ci": recovery_25_ci,
    }


def main():
    print("=" * 60)
    print("Data Budget Sweep: All Hidden Stakeholders")
    print("=" * 60)

    t0 = time.time()
    all_results = {}

    for ds_name, ds_path in [("ml-100k", "data/ml-100k"), ("ml-1m", "data/ml-1m")]:
        ds_dir = ROOT / ds_path
        if not ds_dir.exists():
            print(f"\nSkipping {ds_name} (data not found)")
            continue

        print(f"\n{'─' * 60}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 60}")

        dataset = MovieLensDataset(str(ds_dir))
        configs = build_stakeholder_configs(dataset)
        pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
        base_probs = pool[np.newaxis, :, :]

        # Full scorer is mean of all 3 stakeholders (for gold-standard frontier)
        full_scorer = sum(configs[s] for s in STAKEHOLDER_NAMES) / 3.0

        ds_results = {}
        for hidden in STAKEHOLDER_NAMES:
            result = run_data_budget_for_hidden(
                base_probs, genres, pool, configs, full_scorer, hidden, SEEDS
            )
            ds_results[f"hide_{hidden}"] = result

        all_results[ds_name] = ds_results

    elapsed = time.time() - t0

    # Summary across all (dataset × hidden) combinations
    print(f"\n{'=' * 60}")
    print("Summary: Recovery at N=25")
    print(f"{'=' * 60}")
    all_recoveries = []
    for ds_name, ds_results in all_results.items():
        for hide_key, res in ds_results.items():
            rec = res["recovery_at_25"]
            ci = res["recovery_at_25_ci"]
            if rec is not None:
                print(f"  {ds_name} {hide_key}: {rec*100:.1f}% "
                      f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
                all_recoveries.append(rec)

    if all_recoveries:
        mean_recovery = float(np.mean(all_recoveries))
        min_recovery = float(np.min(all_recoveries))
        max_recovery = float(np.max(all_recoveries))
        print(f"\n  Across all {len(all_recoveries)} (dataset × hidden) combos:")
        print(f"    Mean: {mean_recovery*100:.1f}%")
        print(f"    Range: [{min_recovery*100:.1f}%, {max_recovery*100:.1f}%]")

    results = {
        "config": {
            "n_seeds": len(SEEDS),
            "n_pairs_sweep": N_PAIRS_SWEEP,
            "stakeholders": STAKEHOLDER_NAMES,
            "top_k": TOP_K,
        },
        "datasets": all_results,
        "summary": {
            "mean_recovery_at_25": round(float(np.mean(all_recoveries)), 4) if all_recoveries else None,
            "min_recovery_at_25": round(float(np.min(all_recoveries)), 4) if all_recoveries else None,
            "max_recovery_at_25": round(float(np.max(all_recoveries)), 4) if all_recoveries else None,
            "n_conditions": len(all_recoveries),
        },
        "total_time_seconds": round(elapsed, 1),
    }

    out_path = ROOT / "results" / "data_budget_all_hidden.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
