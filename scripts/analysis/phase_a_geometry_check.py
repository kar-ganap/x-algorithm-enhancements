"""Phase A dry run: verify stakeholder cosine geometry on MIND and Amazon.

This script runs NO BT training. It only:
  1. Loads each dataset
  2. Builds stakeholder weight vectors
  3. Computes pairwise cosine matrix
  4. Computes label disagreement rate across 10K random content pairs
  5. Computes per-stakeholder label balance
  6. Checks go/no-go criteria G1-G6 from the tier 2 expansion plan

Runtime: ~60 seconds per dataset. Total <2 minutes.

Usage:
    uv run python scripts/analysis/phase_a_geometry_check.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _load(name, path):
    """Load a module via importlib, registering in sys.modules first."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / n) if n > 1e-12 else 0.0


def pairwise_cosine_matrix(weights: dict[str, np.ndarray]) -> dict[tuple[str, str], float]:
    """Return all ordered pairs of stakeholder names → cosine similarity."""
    names = list(weights.keys())
    pairs = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j > i:
                pairs[(ni, nj)] = cosine_sim(weights[ni], weights[nj])
    return pairs


def label_disagreement_rate(
    content_features: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42,
) -> float:
    """Fraction of random content pairs where two stakeholders disagree."""
    rng = np.random.default_rng(seed)
    n_content = len(content_features)
    utility_a = content_features @ weights_a
    utility_b = content_features @ weights_b

    disagreements = 0
    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        pref_a = utility_a[c1] > utility_a[c2]
        pref_b = utility_b[c1] > utility_b[c2]
        if pref_a != pref_b:
            disagreements += 1
    return disagreements / n_pairs


def stakeholder_label_balance(
    content_features: np.ndarray,
    weights: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Compute fraction of preferences where item-1 > item-2 for a stakeholder.

    Ideal is ~0.5 (balanced). Degenerate at 0 or 1 means the utility
    function is near-constant (G5 catches this).
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_features)
    utility = content_features @ weights

    n_pref_first = 0
    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        if utility[c1] > utility[c2]:
            n_pref_first += 1
    frac_first = n_pref_first / n_pairs
    return {"frac_first": frac_first, "imbalance": abs(frac_first - 0.5)}


def evaluate_criteria(
    cosines: dict[tuple[str, str], float],
    label_balances: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """Evaluate the 6 go/no-go criteria from the plan."""
    cos_values = list(cosines.values())

    # G1: at least one pair cos < -0.1
    n_negative = sum(1 for c in cos_values if c < -0.1)
    g1 = {
        "pass": n_negative >= 1,
        "detail": f"{n_negative} pairs with cos < -0.1",
    }

    # G2: cosine range spans >= 0.5
    cos_range = max(cos_values) - min(cos_values) if cos_values else 0.0
    g2 = {
        "pass": cos_range >= 0.5,
        "detail": f"range = {cos_range:.3f} (max {max(cos_values):+.3f}, min {min(cos_values):+.3f})",
    }

    # G3: at least one pair |cos| < 0.2
    n_transition = sum(1 for c in cos_values if abs(c) < 0.2)
    g3 = {
        "pass": n_transition >= 1,
        "detail": f"{n_transition} pairs in transition zone |cos| < 0.2",
    }

    # G4: at least 2 pairs cos > 0.5  (INFORMATIONAL — positive controls can come from MovieLens)
    n_strong_positive = sum(1 for c in cos_values if c > 0.5)
    g4 = {
        "pass": n_strong_positive >= 2,
        "detail": f"{n_strong_positive} pairs with cos > 0.5",
        "note": "Informational only — positive controls are provided by MovieLens datasets",
    }

    # G5: no stakeholder has label imbalance > 90/10 (i.e., imbalance > 0.4)
    max_imb = max(b["imbalance"] for b in label_balances.values())
    worst = max(label_balances.items(), key=lambda x: x[1]["imbalance"])
    g5 = {
        "pass": max_imb <= 0.4,
        "detail": f"worst: {worst[0]} at frac_first={worst[1]['frac_first']:.3f}",
    }

    # G6: no stakeholder pair has cos > 0.95 (i.e., redundant pair)
    n_redundant = sum(1 for c in cos_values if c > 0.95)
    max_cos = max(cos_values) if cos_values else 0.0
    g6 = {
        "pass": n_redundant == 0,
        "detail": f"max cos = {max_cos:+.3f}",
    }

    return {"G1": g1, "G2": g2, "G3": g3, "G4": g4, "G5": g5, "G6": g6}


def run_dataset(
    dataset_name: str,
    dataset_module,
    stakeholder_module,
    data_path: Path,
    content_pool_fn_name: str,
) -> dict:
    """Run geometry check on a single dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    if not data_path.exists():
        print(f"  SKIP: data not found at {data_path}")
        return {"dataset": dataset_name, "skipped": True, "reason": "data not found"}

    t0 = time.time()
    # Each dataset's class has a slightly different constructor; use duck-typed lookup.
    DatasetClass = getattr(dataset_module, "MINDDataset", None) or getattr(dataset_module, "AmazonDataset")
    dataset = DatasetClass(str(data_path))
    print(f"  Loaded {dataset.num_items:,} items in {time.time() - t0:.1f}s")
    print(f"  Feature dim: {dataset.num_features}")

    # Build content pool
    content_pool_fn = getattr(stakeholder_module, content_pool_fn_name)
    pool, _topics = content_pool_fn(dataset, seed=42)
    print(f"  Content pool: {len(pool):,} items")

    # Build stakeholders
    configs = stakeholder_module.build_stakeholder_configs(dataset)
    names = list(configs.keys())
    print(f"  Stakeholders ({len(names)}): {names}")

    # Pairwise cosines
    cosines = pairwise_cosine_matrix(configs)
    print("\n  --- Cosine matrix ---")
    for (a, b), c in sorted(cosines.items(), key=lambda x: x[1]):
        print(f"    {a:>18} vs {b:>18}: {c:+.3f}")

    # Label balance per stakeholder (how skewed are the implied preferences?)
    print("\n  --- Label balance (frac_first, closer to 0.5 is better) ---")
    label_balances = {}
    for name in names:
        lb = stakeholder_label_balance(pool, configs[name], n_pairs=2000, seed=42)
        label_balances[name] = lb
        print(f"    {name:>18}: frac_first = {lb['frac_first']:.3f}, imbalance = {lb['imbalance']:.3f}")

    # Label disagreement per pair (useful complement to cosine)
    print("\n  --- Pairwise label disagreement rate (fraction of 10k pairs) ---")
    disagreements = {}
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if j > i:
                dr = label_disagreement_rate(pool, configs[ni], configs[nj], n_pairs=2000, seed=42)
                disagreements[f"{ni}||{nj}"] = dr
                print(f"    {ni:>18} vs {nj:>18}: {dr:.3f}")

    # Evaluate criteria
    criteria = evaluate_criteria(cosines, label_balances)
    print("\n  --- Go/No-Go Criteria ---")
    for gid, g in criteria.items():
        mark = "PASS" if g["pass"] else "FAIL"
        print(f"    [{mark}] {gid}: {g['detail']}")
        if "note" in g:
            print(f"           note: {g['note']}")

    all_blocking_pass = all(
        g["pass"] for gid, g in criteria.items() if gid != "G4"  # G4 is informational
    )
    g4_pass = criteria["G4"]["pass"]

    if all_blocking_pass and g4_pass:
        verdict = "PASS"
    elif all_blocking_pass and not g4_pass:
        verdict = "PASS (G4 fails — informational only, positive controls from MovieLens)"
    else:
        verdict = "FAIL"

    print(f"\n  VERDICT: {verdict}")

    return {
        "dataset": dataset_name,
        "num_items": dataset.num_items,
        "num_features": int(dataset.num_features),
        "stakeholders": names,
        "cosines": {f"{a}||{b}": c for (a, b), c in cosines.items()},
        "label_balances": label_balances,
        "disagreements": disagreements,
        "criteria": criteria,
        "verdict": verdict,
    }


def main():
    t_start = time.time()

    # Load MIND modules
    _mind_mod = _load("mind_mod", ROOT / "enhancements" / "data" / "mind.py")
    _mind_st = _load("mind_st_mod", ROOT / "enhancements" / "data" / "mind_stakeholders.py")

    # Load Amazon modules
    _amazon_mod = _load("amazon_mod", ROOT / "enhancements" / "data" / "amazon.py")
    _amazon_st = _load("amazon_st_mod", ROOT / "enhancements" / "data" / "amazon_stakeholders.py")

    print("Phase A: Geometry Verification")
    print("=" * 60)
    print("This script runs NO BT training. It only checks cosine")
    print("geometry for stakeholder definitions on MIND and Amazon.")

    results = {}
    results["mind"] = run_dataset(
        "MIND-small",
        _mind_mod,
        _mind_st,
        ROOT / "data" / "mind-small",
        content_pool_fn_name="generate_mind_content_pool",
    )
    results["amazon"] = run_dataset(
        "Amazon Kindle",
        _amazon_mod,
        _amazon_st,
        ROOT / "data" / "amazon-kindle",
        content_pool_fn_name="generate_amazon_content_pool",
    )

    elapsed = time.time() - t_start

    # Final summary
    print(f"\n{'=' * 60}")
    print("Phase A Summary")
    print(f"{'=' * 60}")
    for key, r in results.items():
        if r.get("skipped"):
            print(f"  {key}: SKIPPED ({r['reason']})")
        else:
            print(f"  {key}: {r['verdict']}")
    print(f"\nTotal time: {elapsed:.1f}s")

    # Write JSON
    out_path = ROOT / "results" / "phase_a_geometry.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "results": results,
                "total_time_seconds": round(elapsed, 1),
            },
            f,
            indent=2,
            cls=NumpyEncoder,
        )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
