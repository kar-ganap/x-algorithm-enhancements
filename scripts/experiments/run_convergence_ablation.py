"""Convergence ablation: does BT non-convergence cause direction-condition failures?

Sweeps label noise σ ∈ {0.05, 0.15, 0.30, 0.60, 1.00, 1.50} on MIND-small.
At each noise level, measures:
  1. Within-loss cosine per stakeholder (BT convergence diagnostic)
  2. Trained cross-stakeholder cosines
  3. Direction condition match rate (N=25 vs N=2000, Method A)

The target finding: a monotone correlation between within-loss cosine and
direction-condition match rate, supporting the claim that BT convergence
is a necessary precondition for the direction condition.

Usage:
    uv run python scripts/experiments/run_convergence_ablation.py --dataset mind-small
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _dataset_registry import REGISTRY, load_dataset  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_al = _load("alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py")
LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

SEEDS = [42, 142, 242, 342, 442]
SIGMA_P = 0.3       # target perturbation (for direction condition)
N_LOW = 25
N_HIGH = 2000
N_TRAIN = 2000      # for within-loss measurement
NOISE_LEVELS = [0.05, 0.15, 0.30, 0.60, 1.00, 1.50]


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


def train_bt(ds, features, weights, n_pairs, seed, noise_std):
    """Train a single BT model with specified label noise."""
    pref_fn = getattr(ds.stakeholders_mod, ds.spec.preferences_fn)
    pref, rej = pref_fn(features, weights, n_pairs, seed, noise_std=noise_std)
    tp, tr, ep, er = ds.stakeholders_mod.split_preferences(pref, rej, 0.2, seed)
    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        learning_rate=0.01, num_epochs=50, batch_size=64,
    )
    model = train_with_loss(config, tp, tr, verbose=False,
                            eval_probs_preferred=ep, eval_probs_rejected=er)
    return model.weights


def measure_within_loss(ds, features, configs, stakeholder_names, noise_std):
    """Train 3 losses × K stakeholders × 5 seeds. Return within-loss cos per stakeholder."""
    loss_configs = {
        "bradley_terry": {"loss_type": LossType.BRADLEY_TERRY},
        "margin_bt": {"loss_type": LossType.MARGIN_BT, "margin": 0.5},
        "calibrated_bt": {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 1.0},
    }

    # Train all models
    models = {}  # (stakeholder, loss, seed) -> weights
    for s_name in stakeholder_names:
        for l_name, l_kwargs in loss_configs.items():
            for seed in SEEDS:
                pref_fn = getattr(ds.stakeholders_mod, ds.spec.preferences_fn)
                pref, rej = pref_fn(features, configs[s_name], N_TRAIN, seed,
                                    noise_std=noise_std)
                tp, tr, ep, er = ds.stakeholders_mod.split_preferences(pref, rej, 0.2, seed)

                eng_kw = {}
                if l_kwargs.get("loss_type") == LossType.CALIBRATED_BT:
                    eng_kw["target_engagement_pref"] = np.clip(tp @ configs[s_name], 0, 1)
                    eng_kw["target_engagement_rej"] = np.clip(tr @ configs[s_name], 0, 1)

                config = LossConfig(
                    stakeholder=StakeholderType.USER,
                    learning_rate=0.01, num_epochs=50, batch_size=64,
                    **l_kwargs,
                )
                model = train_with_loss(config, tp, tr, verbose=False,
                                        eval_probs_preferred=ep, eval_probs_rejected=er,
                                        **eng_kw)
                models[(s_name, l_name, seed)] = model.weights

    # Compute within-loss cosine per stakeholder
    within_loss = {}
    for s_name in stakeholder_names:
        per_seed_sims = []
        for seed in SEEDS:
            seed_weights = [models[(s_name, l, seed)] for l in loss_configs]
            sims = []
            for i, j in combinations(range(len(seed_weights)), 2):
                sims.append(cosine_sim(seed_weights[i], seed_weights[j]))
            per_seed_sims.append(float(np.mean(sims)))
        within_loss[s_name] = {
            "mean": round(float(np.mean(per_seed_sims)), 4),
            "std": round(float(np.std(per_seed_sims)), 4),
        }

    # Compute trained cross-stakeholder cosines (BT only, averaged over seeds)
    cross_cos = {}
    for s_a, s_b in combinations(stakeholder_names, 2):
        per_seed = []
        for seed in SEEDS:
            per_seed.append(cosine_sim(
                models[(s_a, "bradley_terry", seed)],
                models[(s_b, "bradley_terry", seed)],
            ))
        cross_cos[f"{s_a}-{s_b}"] = {
            "mean": round(float(np.mean(per_seed)), 4),
            "std": round(float(np.std(per_seed)), 4),
        }

    return within_loss, cross_cos


def measure_direction_condition(ds, pool, configs, stakeholder_names, noise_std):
    """Run Method A direction condition at specified noise level."""
    features = pool
    pairs = []
    for target_name in stakeholder_names:
        for hidden_name in stakeholder_names:
            if target_name == hidden_name:
                continue

            # Train at N_LOW and N_HIGH, measure hidden utility change
            utilities = {N_LOW: [], N_HIGH: []}
            trained_weights_high = []
            for n_pairs in [N_LOW, N_HIGH]:
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    perturbed = perturb_weights(configs[target_name], SIGMA_P, rng)

                    # Train BT with specified noise level
                    w = train_bt(ds, features, perturbed, n_pairs, seed, noise_std)

                    # Evaluate hidden utility on top-K selected content
                    scores = features @ w
                    top_k = np.argsort(scores)[-10:][::-1]
                    hidden_util = float(np.sum(features[top_k] @ configs[hidden_name]))
                    utilities[n_pairs].append(hidden_util)

                    if n_pairs == N_HIGH:
                        trained_weights_high.append(w)

            mean_low = float(np.mean(utilities[N_LOW]))
            mean_high = float(np.mean(utilities[N_HIGH]))
            improving = mean_high > mean_low
            change_pct = (mean_high - mean_low) / abs(mean_low) * 100 if abs(mean_low) > 1e-6 else 0.0

            # Trained cosine: average N_HIGH weights for target, raw for hidden
            avg_trained = np.mean(trained_weights_high, axis=0)
            trained_cos = cosine_sim(avg_trained, configs[hidden_name])
            raw_cos = cosine_sim(configs[target_name], configs[hidden_name])

            raw_match = (raw_cos > 0) == improving
            trained_match = (trained_cos > 0) == improving

            pairs.append({
                "target": target_name,
                "hidden": hidden_name,
                "raw_cos": round(raw_cos, 4),
                "trained_cos": round(trained_cos, 4),
                "change_pct": round(change_pct, 1),
                "improving": improving,
                "raw_match": raw_match,
                "trained_match": trained_match,
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Convergence ablation")
    parser.add_argument("--dataset", default="mind-small", choices=list(REGISTRY.keys()))
    args = parser.parse_args()

    dataset_name = args.dataset
    print("=" * 70)
    print(f"Convergence Ablation: {dataset_name}")
    print(f"Noise levels: {NOISE_LEVELS}")
    print("=" * 70)

    t0 = time.time()
    ds = load_dataset(dataset_name)
    pool, genres = ds.pool, ds.topics
    configs = ds.configs
    stakeholder_names = list(ds.spec.primary_stakeholder_order)

    results = {
        "config": {
            "dataset": dataset_name,
            "noise_levels": NOISE_LEVELS,
            "n_seeds": len(SEEDS),
            "n_low": N_LOW,
            "n_high": N_HIGH,
            "n_train_within_loss": N_TRAIN,
            "sigma_p": SIGMA_P,
            "stakeholders": stakeholder_names,
        },
        "sweeps": [],
    }

    for sigma_noise in NOISE_LEVELS:
        print(f"\n{'='*60}")
        print(f"σ_noise = {sigma_noise}")
        print(f"{'='*60}")

        # 1. Within-loss + cross-stakeholder cosines
        print("  Measuring within-loss convergence...")
        within_loss, cross_cos = measure_within_loss(
            ds, pool, configs, stakeholder_names, sigma_noise,
        )
        for s, v in within_loss.items():
            status = "✓" if v["mean"] > 0.85 else "✗"
            print(f"    {status} {s:<20} within-loss={v['mean']:.3f} ± {v['std']:.3f}")

        # 2. Direction condition
        print("  Measuring direction condition...")
        dc_pairs = measure_direction_condition(
            ds, pool, configs, stakeholder_names, sigma_noise,
        )

        # Summarize match rates
        strong_raw = [p for p in dc_pairs if abs(p["raw_cos"]) > 0.2]
        strong_trn = [p for p in dc_pairs if abs(p["trained_cos"]) > 0.2]
        raw_match = sum(p["raw_match"] for p in dc_pairs)
        trn_match = sum(p["trained_match"] for p in dc_pairs)
        raw_strong_match = sum(p["raw_match"] for p in strong_raw)
        trn_strong_match = sum(p["trained_match"] for p in strong_trn)

        print(f"  Direction condition match rates:")
        print(f"    Raw cos:     {raw_match}/{len(dc_pairs)} overall, "
              f"{raw_strong_match}/{len(strong_raw)} for |cos|>0.2")
        print(f"    Trained cos: {trn_match}/{len(dc_pairs)} overall, "
              f"{trn_strong_match}/{len(strong_trn)} for |cos|>0.2")

        # Per-stakeholder match rate (as hidden)
        per_stk_match = {}
        for s in stakeholder_names:
            s_pairs = [p for p in dc_pairs if p["hidden"] == s]
            s_strong = [p for p in s_pairs if abs(p["trained_cos"]) > 0.2]
            s_match = sum(p["trained_match"] for p in s_strong)
            wl = within_loss[s]["mean"]
            per_stk_match[s] = {
                "within_loss": wl,
                "n_strong_pairs": len(s_strong),
                "n_strong_match": s_match,
                "match_rate": round(s_match / len(s_strong), 3) if s_strong else None,
            }
            rate_str = f"{s_match}/{len(s_strong)}" if s_strong else "N/A"
            print(f"      {s:<20} within-loss={wl:.3f} match(|trn|>0.2)={rate_str}")

        sweep_entry = {
            "sigma_noise": sigma_noise,
            "within_loss": within_loss,
            "cross_stakeholder_cos": cross_cos,
            "direction_condition": {
                "pairs": dc_pairs,
                "summary": {
                    "n_total": len(dc_pairs),
                    "raw_match": raw_match,
                    "raw_strong_total": len(strong_raw),
                    "raw_strong_match": raw_strong_match,
                    "trained_match": trn_match,
                    "trained_strong_total": len(strong_trn),
                    "trained_strong_match": trn_strong_match,
                },
                "per_stakeholder_as_hidden": per_stk_match,
            },
        }
        results["sweeps"].append(sweep_entry)

    elapsed = time.time() - t0

    # Final summary: scatter data for the correlation plot
    print("\n" + "=" * 70)
    print("SUMMARY: within-loss cos vs direction-condition match rate")
    print("=" * 70)
    print(f"{'σ_noise':>8} {'stakeholder':<20} {'within_loss':>12} {'match_rate':>12} {'n_pairs':>8}")
    print("-" * 65)

    scatter_points = []
    for sweep in results["sweeps"]:
        sigma = sweep["sigma_noise"]
        for s, info in sweep["direction_condition"]["per_stakeholder_as_hidden"].items():
            wl = info["within_loss"]
            rate = info["match_rate"]
            n = info["n_strong_pairs"]
            rate_str = f"{rate:.3f}" if rate is not None else "N/A"
            print(f"{sigma:>8.2f} {s:<20} {wl:>12.3f} {rate_str:>12} {n:>8}")
            if rate is not None:
                scatter_points.append({"sigma": sigma, "stakeholder": s,
                                       "within_loss": wl, "match_rate": rate, "n": n})

    results["scatter_points"] = scatter_points
    results["total_time_seconds"] = round(elapsed, 1)

    # Correlation
    if len(scatter_points) > 3:
        wl_vals = [p["within_loss"] for p in scatter_points]
        mr_vals = [p["match_rate"] for p in scatter_points]
        from scipy.stats import spearmanr
        rho, pval = spearmanr(wl_vals, mr_vals)
        print(f"\nSpearman ρ(within-loss, match-rate): {rho:.3f} (p={pval:.4f})")
        results["correlation"] = {"spearman_rho": round(rho, 4), "p_value": round(pval, 6)}

    out_path = ROOT / "results" / f"{dataset_name}_convergence_ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
