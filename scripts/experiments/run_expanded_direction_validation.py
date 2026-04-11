"""Expanded direction condition validation (registry-driven).

Method A: target rotation over the dataset's primary stakeholders
Method B: named interpretable stakeholders (MovieLens-only —
          gated by hasattr(stakeholders_mod, "build_named_stakeholder_configs"))

Total points per dataset depend on the primary stakeholder count
(K base stakeholders → K(K-1) Method A points; +5×K named pairs
on MovieLens for Method B).

Note (Phase B refactor): the script now takes one --dataset at a time
(was previously hardcoded to loop over [ml-100k, ml-1m]). Output
filename is dataset-prefixed.

Usage:
    uv run python scripts/experiments/run_expanded_direction_validation.py --dataset ml-100k
"""

from __future__ import annotations

import argparse
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


def _load(name, path):
    """Load reward-modeling modules via importlib."""
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
SEEDS = [42, 142, 242, 342, 442]
SIGMA = 0.3


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


# Named stakeholder definitions moved to enhancements/data/movielens_stakeholders.py
# in Phase B Commit 1 (build_named_stakeholder_configs). MIND/Amazon do not
# implement this function — Method B is feature-flagged via hasattr below.


def run_goodhart_pair(ds, pool, genres, base_probs, target_weights, eval_weights,
                      target_name, hidden_name, hidden_weights, seeds):
    """Run abbreviated Goodhart: N=25 and N=2000, return utility change."""
    cos_val = cosine_sim(target_weights, hidden_weights)

    utilities = {25: [], 2000: []}
    for n_pairs in [25, 2000]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perturbed = perturb_weights(target_weights, SIGMA, rng)
            pref, rej = ds.generate_preferences(perturbed, n_pairs, seed)
            tp, tr, ep, er = ds.stakeholders_mod.split_preferences(pref, rej, 0.2, seed)
            config = LossConfig(
                loss_type=LossType.BRADLEY_TERRY,
                stakeholder=StakeholderType.USER,
                learning_rate=0.01, num_epochs=50, batch_size=64,
            )
            model = train_with_loss(config, tp, tr, verbose=False,
                                    eval_probs_preferred=ep, eval_probs_rejected=er)
            frontier = compute_scorer_eval_frontier(
                base_probs, genres, model.weights, eval_weights,
                DIVERSITY_WEIGHTS, top_k=TOP_K,
            )
            hidden_key = f"{hidden_name}_utility"
            hidden_util = float(np.mean([p[hidden_key] for p in frontier]))
            utilities[n_pairs].append(hidden_util)

    mean_25 = float(np.mean(utilities[25]))
    mean_2000 = float(np.mean(utilities[2000]))
    change_pct = (mean_2000 - mean_25) / abs(mean_25) * 100 if abs(mean_25) > 1e-6 else 0.0
    improving = mean_2000 > mean_25

    return {
        "target": target_name,
        "hidden": hidden_name,
        "cosine": round(cos_val, 4),
        "utility_n25": round(mean_25, 4),
        "utility_n2000": round(mean_2000, 4),
        "change_pct": round(change_pct, 1),
        "improving": improving,
        "prediction": "IMPROVE" if cos_val > 0 else "DEGRADE",
        "match": (cos_val > 0 and improving) or (cos_val < 0 and not improving),
    }


def main():
    parser = argparse.ArgumentParser(description="Expanded direction validation (registry-driven)")
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
        # Backward compat: when invoked without args, default to ml-100k.
        dataset_name = "ml-100k"

    print("=" * 65)
    print(f"Expanded Direction Condition Validation ({dataset_name})")
    print("=" * 65)

    t0 = time.time()
    all_points = []

    print(f"\n{'─' * 50}")
    print(f"Dataset: {dataset_name}")
    print(f"{'─' * 50}")

    ds = load_dataset(dataset_name)
    base_configs = ds.configs
    pool, genres = ds.pool, ds.topics
    base_probs = pool[np.newaxis, :, :]

    # Method B (named stakeholders) is feature-flagged: only MovieLens
    # implements build_named_stakeholder_configs (see Phase B Commit 1).
    has_named = hasattr(ds.stakeholders_mod, "build_named_stakeholder_configs")
    named_configs: dict = {}
    if has_named:
        named_configs = ds.stakeholders_mod.build_named_stakeholder_configs(ds.dataset)

    # Build all stakeholder weights (base + named)
    all_weights = dict(base_configs)
    for name, w in named_configs.items():
        all_weights[name] = w

    # Method A: target rotation over base stakeholders
    print("\n  Method A: Different optimization targets")
    base_names = list(ds.spec.primary_stakeholder_order)
    for target_name in base_names:
        for hidden_name in base_names:
            if target_name == hidden_name:
                continue
            eval_w = {n: all_weights[n] for n in base_names}
            result = run_goodhart_pair(
                ds, pool, genres, base_probs,
                all_weights[target_name], eval_w,
                target_name, hidden_name, all_weights[hidden_name],
                SEEDS,
            )
            result["dataset"] = dataset_name
            result["method"] = "A"
            status = "✓" if result["match"] else "✗"
            print(f"    {status} target={target_name:>10}, hidden={hidden_name:>10}: "
                  f"cos={result['cosine']:+.3f}, change={result['change_pct']:+.1f}%")
            all_points.append(result)

    # Method B: Named stakeholders as hidden, each target (MovieLens only)
    if has_named:
        print("\n  Method B: Named interpretable stakeholders")
        for named in named_configs:
            for target_name in base_names:
                eval_w = {n: all_weights[n] for n in base_names}
                eval_w[named] = all_weights[named]
                result = run_goodhart_pair(
                    ds, pool, genres, base_probs,
                    all_weights[target_name], eval_w,
                    target_name, named, all_weights[named],
                    SEEDS,
                )
                result["dataset"] = dataset_name
                result["method"] = "B"
                status = "✓" if result["match"] else "✗"
                print(f"    {status} target={target_name:>10}, hidden={named:>12}: "
                      f"cos={result['cosine']:+.3f}, change={result['change_pct']:+.1f}%")
                all_points.append(result)
    else:
        print(f"\n  Method B: skipped (not supported for {dataset_name})")

    elapsed = time.time() - t0

    # Summary
    total = len(all_points)
    matches = sum(1 for p in all_points if p["match"])
    violations = total - matches

    print(f"\n{'=' * 65}")
    print(f"Summary: {matches}/{total} match, {violations} violations")
    cos_values = [p["cosine"] for p in all_points]
    print(f"Cosine range: [{min(cos_values):+.3f}, {max(cos_values):+.3f}]")
    print(f"Points with cos < -0.1: {sum(1 for c in cos_values if c < -0.1)}")
    print(f"Points with |cos| < 0.2: {sum(1 for c in cos_values if abs(c) < 0.2)}")
    print(f"Points with cos > 0.5: {sum(1 for c in cos_values if c > 0.5)}")
    print(f"Complete in {elapsed:.0f}s")
    print("=" * 65)

    if violations > 0:
        print("\nViolations:")
        for p in all_points:
            if not p["match"]:
                print(f"  {p['dataset']} target={p['target']}, hidden={p['hidden']}: "
                      f"cos={p['cosine']:+.3f}, predicted={p['prediction']}, "
                      f"change={p['change_pct']:+.1f}%")

    results = {
        "config": {
            "sigma": SIGMA,
            "n_seeds": len(SEEDS),
            "n_values": [25, 2000],
            "named_stakeholders": list(named_configs.keys()),
        },
        "points": all_points,
        "summary": {
            "total": total,
            "matches": matches,
            "violations": violations,
            "cosine_range": [round(min(cos_values), 4), round(max(cos_values), 4)],
        },
        "total_time_seconds": round(elapsed, 1),
    }

    out_path = ROOT / "results" / f"{dataset_name}_expanded_direction_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
