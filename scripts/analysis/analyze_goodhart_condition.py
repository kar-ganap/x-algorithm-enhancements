"""Validate the multi-stakeholder Goodhart condition.

Proposition: Hidden stakeholder utility degrades with N when
cos(w_target, w_hidden) < 0, and improves when cos > 0.

Reads existing Goodhart and foundation results — no new experiments.

Usage:
    uv run python scripts/analysis/analyze_goodhart_condition.py --data data/ml-100k
    uv run python scripts/analysis/analyze_goodhart_condition.py --data data/ml-1m
    uv run python scripts/analysis/analyze_goodhart_condition.py --all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def analyze_dataset(dataset_name: str) -> dict:
    """Validate Goodhart condition on one dataset."""

    # Resolve filenames (ml-100k results may be named 'movielens_*')
    goodhart_path = ROOT / "results" / f"{dataset_name}_goodhart.json"
    if not goodhart_path.exists():
        goodhart_path = ROOT / "results" / "movielens_goodhart.json"
    foundation_path = ROOT / "results" / f"{dataset_name}_foundation.json"
    if not foundation_path.exists():
        foundation_path = ROOT / "results" / "movielens_foundation.json"

    if not goodhart_path.exists() or not foundation_path.exists():
        print(f"  Missing data for {dataset_name}")
        return {}

    goodhart = json.load(open(goodhart_path))
    foundation = json.load(open(foundation_path))

    cos_sims = foundation["cosine_similarity"]
    strategy2 = goodhart["strategy2_more_data"]
    target = goodhart["config"]["stakeholder"]  # "user"

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"Target stakeholder: {target}")
    print(f"{'=' * 60}")

    # Extract all three utility curves
    n_values = sorted(strategy2.keys(), key=lambda x: int(x))
    stakeholders = ["user", "platform", "diversity"]
    utility_curves = {}
    for s in stakeholders:
        key = f"{s}_utility_mean"
        curve = [(int(n), strategy2[n][key]) for n in n_values if key in strategy2[n]]
        utility_curves[s] = curve

    # Print utility curves
    print(f"\n  {'N':>5} | {'user':>10} | {'platform':>10} | {'diversity':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for i, n_key in enumerate(n_values):
        n = int(n_key)
        vals = {s: strategy2[n_key].get(f"{s}_utility_mean", 0) for s in stakeholders}
        print(f"  {n:>5} | {vals['user']:>10.4f} | {vals['platform']:>10.4f} | {vals['diversity']:>10.4f}")

    # For each hidden stakeholder, compute:
    # - cosine with target
    # - peak utility and N* (sweet spot)
    # - degradation from peak to N=2000
    # - monotonic increase/decrease after peak

    print(f"\n  Cosine similarities with target ({target}):")
    results_per_hidden = {}
    for hidden in stakeholders:
        if hidden == target:
            continue

        # Get cosine similarity
        cos_key = f"{target}-{hidden}"
        cos_val = cos_sims.get(cos_key, cos_sims.get(f"{hidden}-{target}", None))
        if cos_val is None:
            continue

        curve = utility_curves[hidden]
        if not curve:
            continue

        utilities = [u for _, u in curve]
        n_vals = [n for n, _ in curve]

        peak_idx = int(np.argmax(utilities))
        peak_n = n_vals[peak_idx]
        peak_util = utilities[peak_idx]
        final_util = utilities[-1]

        # Degradation: how much worse is final vs peak?
        if abs(peak_util) > 1e-6:
            degradation = (peak_util - final_util) / abs(peak_util)
        else:
            degradation = 0.0

        # Is the curve monotonically degrading after peak?
        post_peak = utilities[peak_idx:]
        monotonic_degrading = all(post_peak[i] >= post_peak[i+1] - 0.01
                                  for i in range(len(post_peak) - 1))

        # Overall trend: first vs last
        improving = utilities[-1] > utilities[0]

        prediction = "DEGRADE" if cos_val < 0 else "IMPROVE"
        actual = "degrading" if not improving else "improving"
        match = (cos_val < 0 and not improving) or (cos_val >= 0 and improving)

        status = "✓" if match else "✗"
        print(f"  {status} {hidden}: cos={cos_val:+.3f}, predict={prediction}, actual={actual}")
        print(f"    Peak at N={peak_n} ({peak_util:.4f}), final={final_util:.4f}, degradation={degradation:+.1%}")

        results_per_hidden[hidden] = {
            "cosine_with_target": round(cos_val, 4),
            "prediction": prediction,
            "actual": actual,
            "match": match,
            "peak_n": peak_n,
            "peak_utility": round(peak_util, 4),
            "final_utility": round(final_util, 4),
            "degradation_from_peak": round(degradation, 4),
            "monotonic_after_peak": monotonic_degrading,
            "overall_improving": improving,
        }

    # Also check target stakeholder (should always improve)
    target_curve = utility_curves[target]
    target_improving = target_curve[-1][1] > target_curve[0][1]
    target_change = (target_curve[-1][1] - target_curve[0][1]) / abs(target_curve[0][1])
    print(f"\n  Target ({target}): {'✓ improving' if target_improving else '✗ NOT improving'} ({target_change:+.1%})")

    # Summary
    all_match = all(r["match"] for r in results_per_hidden.values())
    print(f"\n  Direction condition: {'✓ ALL MATCH' if all_match else '✗ MISMATCH'}")

    return {
        "dataset": dataset_name,
        "target": target,
        "cosine_similarities": {s: results_per_hidden[s]["cosine_with_target"]
                                 for s in results_per_hidden},
        "per_hidden_stakeholder": results_per_hidden,
        "target_improving": target_improving,
        "target_change": round(target_change, 4),
        "direction_condition_holds": all_match,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate Goodhart condition")
    parser.add_argument("--data", default="data/ml-100k", help="MovieLens data directory")
    parser.add_argument("--all", action="store_true", help="Run on all available datasets")
    args = parser.parse_args()

    if args.all:
        datasets = ["ml-100k", "ml-1m"]
    else:
        datasets = [Path(args.data).name]

    print("=" * 60)
    print("Multi-Stakeholder Goodhart Condition Validation")
    print("=" * 60)

    all_results = {}
    all_points = []  # (cos, degradation) for cross-dataset analysis

    for ds in datasets:
        result = analyze_dataset(ds)
        if result:
            all_results[ds] = result
            for hidden, data in result.get("per_hidden_stakeholder", {}).items():
                all_points.append({
                    "dataset": ds,
                    "hidden": hidden,
                    "cosine": data["cosine_with_target"],
                    "degradation": data["degradation_from_peak"],
                    "improving": data["overall_improving"],
                })

    # Cross-dataset analysis
    if len(all_points) >= 3:
        print(f"\n{'=' * 60}")
        print("Cross-Dataset Analysis")
        print(f"{'=' * 60}")

        print(f"\n  {'Dataset':>8} | {'Hidden':>10} | {'cos':>6} | {'Degrad':>8} | {'Direction':>10}")
        print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}")
        for p in sorted(all_points, key=lambda x: x["cosine"]):
            direction = "degrades" if not p["improving"] else "improves"
            print(f"  {p['dataset']:>8} | {p['hidden']:>10} | {p['cosine']:>+6.3f} | {p['degradation']:>+7.1%} | {direction:>10}")

        # Correlation between cos and degradation
        cosines = [p["cosine"] for p in all_points]
        degradations = [p["degradation"] for p in all_points]
        if len(cosines) >= 3:
            from scipy.stats import spearmanr
            rho, pval = spearmanr(cosines, degradations)
            print(f"\n  Spearman(cos, degradation) = {rho:.3f} (p={pval:.3f})")
            print(f"  {'✓' if rho > 0.5 else '✗'} More anti-correlated → more degradation: rho={rho:.3f}")

        # Direction condition: all cos < 0 → degrading, all cos > 0 → improving
        direction_correct = all(
            (p["cosine"] < 0 and not p["improving"]) or
            (p["cosine"] >= 0 and p["improving"])
            for p in all_points
        )
        print(f"\n  Direction condition across all datasets: {'✓ HOLDS' if direction_correct else '✗ VIOLATED'}")

        all_results["cross_dataset"] = {
            "points": all_points,
            "spearman_rho": round(float(rho), 4) if len(cosines) >= 3 else None,
            "spearman_pval": round(float(pval), 4) if len(cosines) >= 3 else None,
            "direction_condition_holds": direction_correct,
        }

    # Save
    out_path = ROOT / "results" / "goodhart_condition_validation.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
