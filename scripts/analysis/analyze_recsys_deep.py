"""Deep analysis for RecSys paper: composition, degradation, bootstrap CIs.

A. Composition explains scalarization — project scalarized weights onto per-stakeholder span
B. LOSO degradation prediction — cosine-based ranking vs regret ranking
C. Bootstrap CIs on data budget recovery

Usage:
    uv run python scripts/analysis/analyze_recsys_deep.py --all --data data/ml-1m
    uv run python scripts/analysis/analyze_recsys_deep.py --analysis composition --data data/ml-100k
    uv run python scripts/analysis/analyze_recsys_deep.py --analysis bootstrap --data data/ml-1m
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def project_onto_span(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vector onto column span of basis via least squares."""
    coeffs, _, _, _ = np.linalg.lstsq(basis, vector, rcond=None)
    return basis @ coeffs


def bootstrap_ci(
    values: list[float], n_resamples: int = 10000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap CI. Returns (mean, ci_low, ci_high)."""
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return (
        float(np.mean(arr)),
        float(np.percentile(means, 100 * alpha)),
        float(np.percentile(means, 100 * (1 - alpha))),
    )


# ---------------------------------------------------------------------------
# A: Composition explains scalarization
# ---------------------------------------------------------------------------

def analyze_composition(dataset_name: str) -> dict:
    """Project each scalarized weight vector onto the per-stakeholder span."""
    print("\n" + "=" * 60)
    print("A: Composition Explains Scalarization")
    print("=" * 60)

    scalar_path = ROOT / "results" / f"{dataset_name}_scalarization_baseline.json"
    if not scalar_path.exists():
        # Try old filename
        scalar_path = ROOT / "results" / "scalarization_baseline.json"
    if not scalar_path.exists():
        print(f"  ERROR: No scalarization results at {scalar_path}")
        return {}

    with open(scalar_path) as f:
        data = json.load(f)

    # Find the dataset subtree. Prefer exact dataset_name match
    # (e.g. "mind-small"), fall back to legacy "movielens" key
    # (ml-100k output uses this for backward compat with analysis scripts).
    if dataset_name in data:
        ds_data = data[dataset_name]
    elif "movielens" in data:
        ds_data = data["movielens"]
    else:
        ds_data = data

    if "weight_vectors" not in ds_data:
        print("  ERROR: Weight vectors not saved in scalarization results.")
        print("  Re-run: uv run python scripts/experiments/run_scalarization_baseline.py --data data/{dataset}")
        return {"error": "weight_vectors not in results — rerun scalarization script"}

    wv = ds_data["weight_vectors"]
    per_stk = wv["per_stakeholder"]
    scalarized = wv["scalarized"]

    # Build basis matrix [D, K] from per-stakeholder vectors
    stakeholder_names = sorted(per_stk.keys())
    basis = np.column_stack([np.array(per_stk[s]) for s in stakeholder_names])
    D, K = basis.shape
    print(f"  Per-stakeholder basis: {D}-dim, {K} vectors")
    print(f"  Scalarized models: {len(scalarized)}")

    projection_cosines = []
    residual_norms = []
    results_per_model = []

    for entry in scalarized:
        w_s = np.array(entry["weights"])
        proj = project_onto_span(w_s, basis)
        cos = cosine_sim(w_s, proj)
        residual = w_s - proj
        residual_norm = float(np.linalg.norm(residual))

        projection_cosines.append(cos)
        residual_norms.append(residual_norm)
        results_per_model.append({
            "mixing": entry["mixing"],
            "projection_cosine": round(cos, 4),
            "residual_norm": round(residual_norm, 4),
        })

    mean_cos = float(np.mean(projection_cosines))
    min_cos = float(np.min(projection_cosines))
    mean_residual = float(np.mean(residual_norms))

    status = "✓" if mean_cos > 0.90 else "✗"
    print(f"  {status} Mean projection cosine: {mean_cos:.4f}")
    print(f"  Min projection cosine:  {min_cos:.4f}")
    print(f"  Mean residual norm:     {mean_residual:.4f}")
    print(f"  Models with cos > 0.95: {sum(1 for c in projection_cosines if c > 0.95)}/{len(projection_cosines)}")

    return {
        "mean_projection_cosine": round(mean_cos, 4),
        "min_projection_cosine": round(min_cos, 4),
        "std_projection_cosine": round(float(np.std(projection_cosines)), 4),
        "mean_residual_norm": round(mean_residual, 4),
        "n_above_095": sum(1 for c in projection_cosines if c > 0.95),
        "n_total": len(projection_cosines),
        "per_model": results_per_model,
    }


# ---------------------------------------------------------------------------
# B: LOSO degradation prediction
# ---------------------------------------------------------------------------

def analyze_degradation(dataset_name: str) -> dict:
    """Check if cosine-based proxy predicts LOSO regret ranking."""
    print("\n" + "=" * 60)
    print("B: LOSO Degradation Prediction")
    print("=" * 60)

    # Load LOSO regret
    loso_path = ROOT / "results" / f"{dataset_name}_loso.json"
    if not loso_path.exists():
        loso_path = ROOT / "results" / "movielens_loso.json"
    if not loso_path.exists():
        print(f"  ERROR: No LOSO results found")
        return {}

    with open(loso_path) as f:
        loso = json.load(f)

    # Load foundation (stakeholder weights)
    found_path = ROOT / "results" / f"{dataset_name}_foundation.json"
    if not found_path.exists():
        found_path = ROOT / "results" / "movielens_foundation.json"
    if not found_path.exists():
        print(f"  ERROR: No foundation results found")
        return {}

    with open(found_path) as f:
        foundation = json.load(f)

    # Extract regret per hidden stakeholder
    regret_ranking = {}
    for key, val in loso.get("loso_regret", {}).items():
        stakeholder = key.replace("hide_", "")
        regret_ranking[stakeholder] = val["avg_regret"]

    # Extract trained weight cosine similarities
    cos_sims = foundation.get("cosine_similarity", {})

    print(f"  Regret ranking:")
    for s, r in sorted(regret_ranking.items(), key=lambda x: x[1], reverse=True):
        print(f"    {s}: {r:.3f}")

    # Proxy: for each hidden stakeholder, compute mean cosine with the other two
    # Lower cosine with observed pair → harder to predict → higher regret
    stakeholders = sorted(regret_ranking.keys())
    proxy_scores = {}
    for hidden in stakeholders:
        observed = [s for s in stakeholders if s != hidden]
        pair_key = f"{observed[0]}-{observed[1]}"
        # Find the cosine between the two observed stakeholders
        # And the cosines between hidden and each observed
        cos_hidden_obs = []
        for obs in observed:
            key1 = f"{hidden}-{obs}"
            key2 = f"{obs}-{hidden}"
            cos_val = cos_sims.get(key1, cos_sims.get(key2, None))
            if cos_val is not None:
                cos_hidden_obs.append(cos_val)
        proxy_scores[hidden] = float(np.mean(cos_hidden_obs)) if cos_hidden_obs else 0.0

    print(f"\n  Proxy (mean cos with observed):")
    for s, p in sorted(proxy_scores.items(), key=lambda x: x[1]):
        print(f"    {s}: {p:.3f}")

    # Check ranking: lowest proxy → highest regret
    regret_order = sorted(regret_ranking, key=lambda s: regret_ranking[s], reverse=True)
    proxy_order = sorted(proxy_scores, key=lambda s: proxy_scores[s])  # lowest cos → highest regret

    ranking_matches = regret_order == proxy_order
    print(f"\n  Regret order: {' > '.join(regret_order)}")
    print(f"  Proxy order:  {' > '.join(proxy_order)}")
    print(f"  {'✓' if ranking_matches else '✗'} Rankings match: {ranking_matches}")

    return {
        "regret_ranking": regret_ranking,
        "proxy_scores": proxy_scores,
        "regret_order": regret_order,
        "proxy_order": proxy_order,
        "ranking_matches": ranking_matches,
    }


# ---------------------------------------------------------------------------
# C: Bootstrap CIs on data budget
# ---------------------------------------------------------------------------

def analyze_bootstrap(dataset_name: str) -> dict:
    """Compute bootstrap CIs for data budget recovery."""
    print("\n" + "=" * 60)
    print("C: Bootstrap CIs on Data Budget")
    print("=" * 60)

    loso_path = ROOT / "results" / f"{dataset_name}_loso.json"
    if not loso_path.exists():
        loso_path = ROOT / "results" / "movielens_loso.json"
    if not loso_path.exists():
        print(f"  ERROR: No LOSO results found")
        return {}

    with open(loso_path) as f:
        loso = json.load(f)

    budget = loso.get("data_budget", {})
    sweep = budget.get("sweep", [])
    baseline_regret = budget.get("loso_baseline_regret", None)

    if not sweep:
        print("  ERROR: No data budget sweep in LOSO results")
        return {}

    print(f"  Baseline regret (N=0): {baseline_regret:.3f}")
    print(f"  {'N':>5} | {'mean':>8} | {'95% CI':>16} | {'recovery':>10} | {'recovery CI':>16}")
    print(f"  {'-'*5}-+-{'-'*8}-+-{'-'*16}-+-{'-'*10}-+-{'-'*16}")

    results_per_n = []
    for entry in sweep:
        n = entry["n_pairs"]
        per_seed = entry.get("per_seed_regret", [])
        if not per_seed:
            continue

        mean, lo, hi = bootstrap_ci(per_seed)

        # Recovery CI
        if baseline_regret and baseline_regret > 0:
            recovery_vals = [(baseline_regret - r) / baseline_regret for r in per_seed]
            rec_mean, rec_lo, rec_hi = bootstrap_ci(recovery_vals)
        else:
            rec_mean = rec_lo = rec_hi = 0.0

        print(f"  {n:>5} | {mean:>8.3f} | [{lo:>6.3f}, {hi:>6.3f}] | {rec_mean:>9.1%} | [{rec_lo:>6.1%}, {rec_hi:>6.1%}]")

        results_per_n.append({
            "n_pairs": n,
            "regret_mean": round(mean, 4),
            "regret_ci_lo": round(lo, 4),
            "regret_ci_hi": round(hi, 4),
            "recovery_mean": round(rec_mean, 4),
            "recovery_ci_lo": round(rec_lo, 4),
            "recovery_ci_hi": round(rec_hi, 4),
            "n_seeds": len(per_seed),
        })

    # Is N=25 recovery significantly > 0?
    n25 = next((r for r in results_per_n if r["n_pairs"] == 25), None)
    if n25:
        sig = n25["recovery_ci_lo"] > 0
        print(f"\n  N=25 recovery: {n25['recovery_mean']:.1%} [{n25['recovery_ci_lo']:.1%}, {n25['recovery_ci_hi']:.1%}]")
        print(f"  {'✓' if sig else '✗'} Significantly > 0: {sig}")

    return {
        "baseline_regret": baseline_regret,
        "per_n": results_per_n,
        "n25_significant": n25["recovery_ci_lo"] > 0 if n25 else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RecSys deep analysis")
    parser.add_argument("--data", default=None, help="(deprecated) data directory; use --dataset")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name from registry (ml-100k, ml-1m, mind-small, amazon-kindle)")
    parser.add_argument("--analysis", type=str, choices=["composition", "degradation", "bootstrap"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # Resolve dataset name: prefer --dataset, fall back to --data, default to ml-100k.
    if args.dataset:
        dataset_name = args.dataset
    elif args.data:
        import warnings
        warnings.warn(
            "--data is deprecated; use --dataset {ml-100k,ml-1m,mind-small,amazon-kindle}",
            DeprecationWarning,
            stacklevel=2,
        )
        dataset_name = Path(args.data).name
    else:
        dataset_name = "ml-100k"

    if not args.all and not args.analysis:
        args.all = True
    analyses = set()
    if args.all:
        analyses = {"composition", "degradation", "bootstrap"}
    elif args.analysis:
        analyses = {args.analysis}

    print("=" * 60)
    print(f"RecSys Deep Analysis ({dataset_name})")
    print("=" * 60)

    results = {"dataset": dataset_name}

    if "composition" in analyses:
        results["composition"] = analyze_composition(dataset_name)
    if "degradation" in analyses:
        results["degradation"] = analyze_degradation(dataset_name)
    if "bootstrap" in analyses:
        results["bootstrap"] = analyze_bootstrap(dataset_name)

    out_path = ROOT / "results" / f"{dataset_name}_deep_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
