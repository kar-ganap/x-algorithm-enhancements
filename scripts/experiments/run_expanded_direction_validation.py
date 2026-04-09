"""Expanded direction condition validation.

Method A: 3 targets × 2 hidden × 2 MovieLens datasets = 12 points
Method B: 5 named stakeholders × 3 targets × 2 datasets = 30 points
Total: ~42 data points spanning cos ≈ -0.9 to +0.9

Usage:
    uv run python scripts/experiments/run_expanded_direction_validation.py
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
NUM_GENRES = _st.NUM_GENRES

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


# ── Named stakeholder definitions ──

def compute_creator_weights(dataset):
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    mx = np.max(genre_counts)
    return (genre_counts / mx).astype(np.float32) if mx > 0 else genre_counts.astype(np.float32)


def compute_advertiser_weights(dataset):
    genre_ratings = np.zeros(NUM_GENRES, dtype=np.float64)
    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is not None:
            genre_ratings += movie.genres
    mx = np.max(genre_ratings)
    return (genre_ratings / mx).astype(np.float32) if mx > 0 else genre_ratings.astype(np.float32)


def compute_niche_weights(dataset):
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    active = genre_counts > 0
    median_count = np.median(genre_counts[active]) if np.any(active) else 1.0
    weights = np.zeros(NUM_GENRES, dtype=np.float32)
    weights[active & (genre_counts < median_count)] = 1.0
    weights[active & (genre_counts >= median_count)] = -1.0
    mx = np.max(np.abs(weights))
    return weights / mx if mx > 0 else weights


def compute_mainstream_weights(dataset):
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    top3 = np.argsort(genre_counts)[-3:]
    weights = np.zeros(NUM_GENRES, dtype=np.float32)
    weights[top3] = 1.0
    return weights


def compute_moderator_weights(dataset):
    genre_low = np.zeros(NUM_GENRES, dtype=np.float64)
    genre_total = np.zeros(NUM_GENRES, dtype=np.float64)
    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_total[g] += 1
                if rating.rating <= 2:
                    genre_low[g] += 1
    active = genre_total > 0
    frac = np.zeros(NUM_GENRES, dtype=np.float32)
    frac[active] = (genre_low[active] / genre_total[active]).astype(np.float32)
    weights = -frac
    mx = np.max(np.abs(weights))
    return weights / mx if mx > 0 else weights


NAMED_FNS = {
    "creator": compute_creator_weights,
    "advertiser": compute_advertiser_weights,
    "niche": compute_niche_weights,
    "mainstream": compute_mainstream_weights,
    "moderator": compute_moderator_weights,
}


def run_goodhart_pair(pool, genres, base_probs, target_weights, eval_weights,
                      target_name, hidden_name, hidden_weights, seeds):
    """Run abbreviated Goodhart: N=25 and N=2000, return utility change."""
    cos_val = cosine_sim(target_weights, hidden_weights)

    utilities = {25: [], 2000: []}
    for n_pairs in [25, 2000]:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            perturbed = perturb_weights(target_weights, SIGMA, rng)
            pref, rej = generate_movielens_preferences(pool, perturbed, n_pairs, seed)
            tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)
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
    print("=" * 65)
    print("Phase 14: Expanded Direction Condition Validation")
    print("=" * 65)

    t0 = time.time()
    all_points = []

    for ds_name, ds_path in [("ml-100k", "data/ml-100k"), ("ml-1m", "data/ml-1m")]:
        ds_dir = ROOT / ds_path
        if not ds_dir.exists():
            print(f"  Skipping {ds_name} (data not found)")
            continue

        print(f"\n{'─' * 50}")
        print(f"Dataset: {ds_name}")
        print(f"{'─' * 50}")

        dataset = MovieLensDataset(str(ds_dir))
        base_configs = build_stakeholder_configs(dataset)
        pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
        base_probs = pool[np.newaxis, :, :]

        # Build all stakeholder weights (base + named)
        all_weights = dict(base_configs)
        for name, fn in NAMED_FNS.items():
            all_weights[name] = fn(dataset)

        # Method A: 3 targets × remaining hidden
        print("\n  Method A: Different optimization targets")
        base_names = ["user", "platform", "diversity"]
        for target_name in base_names:
            for hidden_name in base_names:
                if target_name == hidden_name:
                    continue
                eval_w = {n: all_weights[n] for n in base_names}
                result = run_goodhart_pair(
                    pool, genres, base_probs,
                    all_weights[target_name], eval_w,
                    target_name, hidden_name, all_weights[hidden_name],
                    SEEDS,
                )
                result["dataset"] = ds_name
                result["method"] = "A"
                status = "✓" if result["match"] else "✗"
                print(f"    {status} target={target_name:>10}, hidden={hidden_name:>10}: "
                      f"cos={result['cosine']:+.3f}, change={result['change_pct']:+.1f}%")
                all_points.append(result)

        # Method B: Named stakeholders as hidden, each target
        print("\n  Method B: Named interpretable stakeholders")
        for named in NAMED_FNS:
            for target_name in base_names:
                eval_w = {n: all_weights[n] for n in base_names}
                eval_w[named] = all_weights[named]
                result = run_goodhart_pair(
                    pool, genres, base_probs,
                    all_weights[target_name], eval_w,
                    target_name, named, all_weights[named],
                    SEEDS,
                )
                result["dataset"] = ds_name
                result["method"] = "B"
                status = "✓" if result["match"] else "✗"
                print(f"    {status} target={target_name:>10}, hidden={named:>12}: "
                      f"cos={result['cosine']:+.3f}, change={result['change_pct']:+.1f}%")
                all_points.append(result)

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
            "named_stakeholders": list(NAMED_FNS.keys()),
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

    out_path = ROOT / "results" / "expanded_direction_validation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
