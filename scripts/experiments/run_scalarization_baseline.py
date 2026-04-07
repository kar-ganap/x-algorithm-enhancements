"""Scalarization baseline comparison.

Compares per-stakeholder BT training + frontier composition against
single-model scalarization (mixed preference training) on both
synthetic and MovieLens data.

Per-stakeholder: 3 models × 21 diversity weights → 21 frontier points
Scalarization: 21 models × 21 diversity weights → 441 points → Pareto front
(Generous to scalarization: 21× more frontier evaluations)

Usage:
    uv run python scripts/experiments/run_scalarization_baseline.py --all
    uv run python scripts/experiments/run_scalarization_baseline.py --dataset movielens
    uv run python scripts/experiments/run_scalarization_baseline.py --dataset synthetic
"""

from __future__ import annotations

import argparse
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
extract_pareto_front_nd = _kf.extract_pareto_front_nd
is_dominated_nd = _kf.is_dominated_nd

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss
NUM_ACTIONS = _al.NUM_ACTIONS
ACTION_INDICES = _al.ACTION_INDICES
POSITIVE_INDICES = _al.POSITIVE_INDICES
NEGATIVE_INDICES = _al.NEGATIVE_INDICES

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10
SEED = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def simplex_grid(k: int = 3, resolution: int = 5) -> list[tuple[float, ...]]:
    """Generate uniform grid on k-simplex."""
    if k == 1:
        return [(1.0,)]
    if resolution == 0:
        return [(0.0,) * (k - 1) + (1.0,)]
    points = []
    for i in range(resolution + 1):
        w_first = i / resolution
        remaining = resolution - i
        for sub in simplex_grid(k - 1, remaining):
            scaled = tuple(round(s * (1 - w_first), 6) for s in sub)
            points.append((round(w_first, 6), *scaled))
    return points


def generate_scalarized_preferences(
    stakeholder_prefs: dict[str, tuple[np.ndarray, np.ndarray]],
    mixing: dict[str, float],
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mixed preference pairs by sampling from stakeholders."""
    rng = np.random.default_rng(seed)
    names = sorted(mixing.keys())
    probs = np.array([mixing[n] for n in names])

    first_key = names[0]
    feature_dim = stakeholder_prefs[first_key][0].shape[1]
    pref = np.zeros((n_pairs, feature_dim), dtype=np.float32)
    rej = np.zeros((n_pairs, feature_dim), dtype=np.float32)

    for i in range(n_pairs):
        chosen = rng.choice(len(names), p=probs)
        s_name = names[chosen]
        s_pref, s_rej = stakeholder_prefs[s_name]
        idx = rng.integers(0, len(s_pref))
        pref[i] = s_pref[idx]
        rej[i] = s_rej[idx]

    return pref, rej


def compute_hypervolume_3d(
    frontier: list[dict[str, float]],
    dims: list[str],
    ref_point: tuple[float, float, float],
) -> float:
    """Approximate 3D hypervolume using Monte Carlo sampling.

    For exact computation we'd need a proper HV algorithm, but for
    comparison purposes MC is sufficient.
    """
    if not frontier:
        return 0.0

    values = np.array([[p[d] for d in dims] for p in frontier])
    ref = np.array(ref_point)

    # Only count points that dominate the reference
    above_ref = values[np.all(values > ref, axis=1)]
    if len(above_ref) == 0:
        return 0.0

    # Simple decomposition: sum of dominated boxes
    # Sort by first dimension, use inclusion-exclusion
    # For robustness, use the product-of-ranges approximation
    maxs = np.max(above_ref, axis=0)
    mins = ref
    ranges = maxs - mins

    # MC sampling
    rng = np.random.default_rng(42)
    n_samples = 50000
    samples = rng.uniform(mins, maxs, size=(n_samples, 3))

    dominated_count = 0
    for sample in samples:
        for point in above_ref:
            if np.all(sample <= point):
                dominated_count += 1
                break

    volume_box = float(np.prod(ranges))
    return volume_box * dominated_count / n_samples


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
# Synthetic data generation (inlined from verify_held_out.py)
# ---------------------------------------------------------------------------

STAKEHOLDER_UTILITY_SYNTHETIC = {
    "user": lambda pos, neg: pos - neg,
    "platform": lambda pos, neg: pos - 0.3 * neg,
    "society": lambda pos, neg: pos - 4.0 * neg,
}


def generate_content_pool_synthetic(n_content: int, seed: int):
    rng = np.random.default_rng(seed)
    content_probs = np.zeros((n_content, NUM_ACTIONS), dtype=np.float32)
    content_topics = rng.integers(0, 6, size=n_content)
    for i in range(n_content):
        topic = content_topics[i]
        base_probs = rng.uniform(0.05, 0.3, NUM_ACTIONS)
        if topic == 0:
            base_probs[ACTION_INDICES["favorite"]] *= 1.5
            base_probs[ACTION_INDICES["repost"]] *= 1.3
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.3
        elif topic == 1:
            base_probs[ACTION_INDICES["favorite"]] *= 1.2
            base_probs[ACTION_INDICES["reply"]] *= 1.4
        elif topic in [2, 3]:
            base_probs[ACTION_INDICES["favorite"]] *= 1.3
            base_probs[ACTION_INDICES["repost"]] *= 1.8
            base_probs[ACTION_INDICES["reply"]] *= 2.0
            base_probs[ACTION_INDICES["block_author"]] *= 2.5
            base_probs[ACTION_INDICES["report"]] *= 2.0
        elif topic == 4:
            base_probs[ACTION_INDICES["favorite"]] *= 1.8
            base_probs[ACTION_INDICES["share"]] *= 1.5
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.2
        elif topic == 5:
            base_probs[ACTION_INDICES["follow_author"]] *= 1.3
            base_probs[ACTION_INDICES["reply"]] *= 1.2
        content_probs[i] = np.clip(base_probs, 0, 1)
    return content_probs, content_topics


def generate_synthetic_preferences(content_probs, stakeholder, n_pairs, seed):
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)
    utility_fn = STAKEHOLDER_UTILITY_SYNTHETIC[stakeholder]
    pos_scores = np.array([np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)])
    neg_scores = np.array([np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)])
    utility = utility_fn(pos_scores, neg_scores)
    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]
        noise = rng.normal(0, 0.05)
        if (diff + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]
    return probs_pref, probs_rej


# ---------------------------------------------------------------------------
# Core experiment
# ---------------------------------------------------------------------------

def run_dataset(
    dataset_name: str,
    base_probs: np.ndarray,
    genres: np.ndarray,
    stakeholder_weights: dict[str, np.ndarray],
    stakeholder_prefs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Run scalarization comparison on one dataset."""
    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 60}")

    stakeholder_names = sorted(stakeholder_weights.keys())
    utility_dims = [f"{s}_utility" for s in stakeholder_names]
    grid = simplex_grid(k=len(stakeholder_names), resolution=5)
    print(f"  Simplex grid: {len(grid)} mixing points")

    # --- Per-stakeholder frontier ---
    print("\n  Per-stakeholder frontier:")
    # Train 3 BT models
    per_stk_weights = {}
    for s_name in stakeholder_names:
        pref, rej = stakeholder_prefs[s_name]
        tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=SEED)
        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=StakeholderType.USER,
            learning_rate=0.01, num_epochs=50, batch_size=64,
        )
        model = train_with_loss(config, tp, tr, verbose=False,
                                eval_probs_preferred=ep, eval_probs_rejected=er)
        per_stk_weights[s_name] = model.weights
        print(f"    {s_name}: eval_acc={model.eval_accuracy:.1%}")

    # Composite scorer = mean of 3 learned weight vectors
    composite_scorer = np.mean(list(per_stk_weights.values()), axis=0)
    per_stk_frontier = compute_scorer_eval_frontier(
        base_probs, genres, composite_scorer, stakeholder_weights,
        DIVERSITY_WEIGHTS, top_k=TOP_K,
    )
    per_stk_pareto = extract_pareto_front_nd(per_stk_frontier, utility_dims)
    print(f"    Frontier: {len(per_stk_frontier)} points, {len(per_stk_pareto)} Pareto-optimal")

    # --- Scalarized frontier ---
    print(f"\n  Scalarized frontier ({len(grid)} models × {len(DIVERSITY_WEIGHTS)} δ):")
    all_scalarized_points = []
    for gi, mixing_tuple in enumerate(grid):
        mixing = {s: w for s, w in zip(stakeholder_names, mixing_tuple)}

        # Skip degenerate points (all weight on one stakeholder = same as per-stakeholder)
        if max(mixing.values()) > 0.99:
            continue

        pref, rej = generate_scalarized_preferences(stakeholder_prefs, mixing, 2000, seed=SEED + gi)
        tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=SEED + gi)
        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=StakeholderType.USER,
            learning_rate=0.01, num_epochs=50, batch_size=64,
        )
        model = train_with_loss(config, tp, tr, verbose=False,
                                eval_probs_preferred=ep, eval_probs_rejected=er)

        # Sweep diversity weights with this model's learned weights as scorer
        frontier = compute_scorer_eval_frontier(
            base_probs, genres, model.weights, stakeholder_weights,
            DIVERSITY_WEIGHTS, top_k=TOP_K,
        )
        all_scalarized_points.extend(frontier)

        if (gi + 1) % 5 == 0:
            print(f"    {gi + 1}/{len(grid)} mixing points processed")

    scalar_pareto = extract_pareto_front_nd(all_scalarized_points, utility_dims)
    print(f"    Total points: {len(all_scalarized_points)}, Pareto-optimal: {len(scalar_pareto)}")

    # --- Comparison ---
    print("\n  Comparison:")

    # Reference point for hypervolume: minimum across both frontiers
    all_points = per_stk_frontier + all_scalarized_points
    ref_point = tuple(
        min(p[d] for p in all_points) - 0.01 for d in utility_dims
    )

    hv_per_stk = compute_hypervolume_3d(per_stk_pareto, utility_dims, ref_point)
    hv_scalar = compute_hypervolume_3d(scalar_pareto, utility_dims, ref_point)
    hv_ratio = hv_per_stk / hv_scalar if hv_scalar > 0 else float("inf")

    # Dominated fraction: % of scalarized Pareto points dominated by per-stakeholder
    dominated = sum(
        1 for sp in scalar_pareto
        if is_dominated_nd(sp, per_stk_pareto, utility_dims)
    )
    dominated_frac = dominated / len(scalar_pareto) if scalar_pareto else 0

    # Max per-dimension
    max_per_stk = {d: max(p[d] for p in per_stk_pareto) for d in utility_dims}
    max_scalar = {d: max(p[d] for p in scalar_pareto) for d in utility_dims}

    print(f"    Per-stakeholder HV: {hv_per_stk:.3f}")
    print(f"    Scalarized HV:     {hv_scalar:.3f}")
    print(f"    HV ratio:          {hv_ratio:.3f}")
    print(f"    Dominated frac:    {dominated_frac:.1%}")
    print(f"    Max per-dim (per-stk): {', '.join(f'{d}={v:.2f}' for d, v in max_per_stk.items())}")
    print(f"    Max per-dim (scalar):  {', '.join(f'{d}={v:.2f}' for d, v in max_scalar.items())}")

    return {
        "per_stakeholder_frontier": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}
            for p in per_stk_pareto
        ],
        "scalarized_frontier": [
            {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}
            for p in scalar_pareto
        ],
        "comparison": {
            "per_stakeholder_hypervolume": round(hv_per_stk, 4),
            "scalarized_hypervolume": round(hv_scalar, 4),
            "hypervolume_ratio": round(hv_ratio, 4),
            "dominated_fraction": round(dominated_frac, 4),
            "n_per_stakeholder_pareto": len(per_stk_pareto),
            "n_scalarized_pareto": len(scalar_pareto),
            "n_scalarized_total": len(all_scalarized_points),
            "max_per_stakeholder": {k: round(v, 4) for k, v in max_per_stk.items()},
            "max_scalarized": {k: round(v, 4) for k, v in max_scalar.items()},
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scalarization baseline comparison")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, choices=["synthetic", "movielens"])
    args = parser.parse_args()

    if not args.all and not args.dataset:
        args.all = True

    datasets_to_run = set()
    if args.all:
        datasets_to_run = {"synthetic", "movielens"}
    elif args.dataset:
        datasets_to_run = {args.dataset}

    t0 = time.time()
    results = {"config": {
        "simplex_resolution": 5,
        "n_mixing_points": 21,
        "n_pairs_per_model": 2000,
        "diversity_weights": DIVERSITY_WEIGHTS,
        "top_k": TOP_K,
    }}

    if "synthetic" in datasets_to_run:
        content_probs, content_topics = generate_content_pool_synthetic(500, SEED)
        base_probs = content_probs[np.newaxis, :, :]
        # Synthetic stakeholder weights (true utility direction)
        synth_weights = {
            "user": np.zeros(NUM_ACTIONS, dtype=np.float32),
            "platform": np.zeros(NUM_ACTIONS, dtype=np.float32),
            "society": np.zeros(NUM_ACTIONS, dtype=np.float32),
        }
        for idx in POSITIVE_INDICES:
            synth_weights["user"][idx] = 1.0
            synth_weights["platform"][idx] = 1.0
            synth_weights["society"][idx] = 1.0
        for idx in NEGATIVE_INDICES:
            synth_weights["user"][idx] = -1.0
            synth_weights["platform"][idx] = -0.3
            synth_weights["society"][idx] = -4.0

        synth_prefs = {}
        for s in ["user", "platform", "society"]:
            synth_prefs[s] = generate_synthetic_preferences(content_probs, s, 2000, SEED)

        results["synthetic"] = run_dataset(
            "synthetic", base_probs, content_topics, synth_weights, synth_prefs,
        )

    if "movielens" in datasets_to_run:
        data_dir = ROOT / "data" / "ml-100k"
        if not data_dir.exists():
            print(f"ERROR: MovieLens data not found at {data_dir}")
            sys.exit(1)
        dataset = MovieLensDataset(str(data_dir))
        configs = build_stakeholder_configs(dataset)
        pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=SEED)
        base_probs = pool[np.newaxis, :, :]

        ml_prefs = {}
        for s in ["user", "platform", "diversity"]:
            ml_prefs[s] = generate_movielens_preferences(pool, configs[s], 2000, SEED)

        results["movielens"] = run_dataset(
            "movielens", base_probs, genres, configs, ml_prefs,
        )

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    print(f"\n{'=' * 60}")
    print(f"Complete in {elapsed:.0f}s")
    print("=" * 60)

    out_path = ROOT / "results" / "scalarization_baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
