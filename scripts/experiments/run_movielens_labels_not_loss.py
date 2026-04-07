"""Labels-Not-Loss experiment on MovieLens-100K.

Replicates the paper's central claim on real data and extends with
four MovieLens-specific tests (temporal, downstream, user-groups, genre correlation).

Groups:
  A: Core labels-not-loss replication (3 losses × 3 stakeholders × 5 seeds)
  B: Hyperparameter robustness
  C: Temporal generalization (train on early, eval on late)
  D: Downstream rating prediction (BT weights predict actual ratings?)
  E: User-group preference heterogeneity (per-genre-group convergence)
  F: Genre correlation matrix (informational)

Usage:
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --all
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --group A
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --group C

Output:
    results/movielens_labels_not_loss.json
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
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
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
generate_movielens_content_pool_temporal = _st.generate_movielens_content_pool_temporal
generate_movielens_preferences = _st.generate_movielens_preferences
split_preferences = _st.split_preferences
compute_user_genre_weights_for_group = _st.compute_user_genre_weights_for_group
get_user_genre_groups = _st.get_user_genre_groups
NUM_GENRES = _st.NUM_GENRES

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


STAKEHOLDER_MAP = {
    "user": StakeholderType.USER,
    "platform": StakeholderType.PLATFORM,
    "diversity": StakeholderType.SOCIETY,
}

LOSS_CONFIGS = {
    "bradley_terry": {"loss_type": LossType.BRADLEY_TERRY},
    "margin_bt": {"loss_type": LossType.MARGIN_BT, "margin": 0.5},
    "calibrated_bt": {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 1.0},
}

SEEDS = [42, 142, 242, 342, 442]


def _train(features, stakeholder_weights, stakeholder_name, loss_name, loss_kwargs,
           n_pairs=2500, seed=42, epochs=50):
    """Train a single model and return results dict."""
    pref, rej = generate_movielens_preferences(features, stakeholder_weights, n_pairs, seed)
    tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)

    eng_kw = {}
    if loss_kwargs.get("loss_type") == LossType.CALIBRATED_BT:
        eng_kw["target_engagement_pref"] = np.clip(tp @ stakeholder_weights, 0, 1)
        eng_kw["target_engagement_rej"] = np.clip(tr @ stakeholder_weights, 0, 1)

    config = LossConfig(
        stakeholder=STAKEHOLDER_MAP[stakeholder_name],
        learning_rate=0.01,
        num_epochs=epochs,
        batch_size=64,
        **loss_kwargs,
    )
    model = train_with_loss(config, tp, tr, verbose=False,
                            eval_probs_preferred=ep, eval_probs_rejected=er,
                            **eng_kw)
    return {
        "stakeholder": stakeholder_name,
        "loss": loss_name,
        "seed": seed,
        "train_accuracy": round(float(model.accuracy), 4),
        "eval_accuracy": round(float(model.eval_accuracy), 4) if model.eval_accuracy else None,
        "weights": model.weights.tolist(),
    }


# ---------------------------------------------------------------------------
# Group A: Core replication
# ---------------------------------------------------------------------------

def run_group_a(pool, configs):
    print("\n" + "=" * 60)
    print("Group A: Core Labels-Not-Loss Replication")
    print(f"  3 losses × 3 stakeholders × {len(SEEDS)} seeds = {3*3*len(SEEDS)} runs")
    print("=" * 60)

    features, _ = pool
    models = []
    for seed in SEEDS:
        for s_name in ["user", "platform", "diversity"]:
            for l_name, l_kwargs in LOSS_CONFIGS.items():
                m = _train(features, configs[s_name], s_name, l_name, l_kwargs, seed=seed)
                models.append(m)
                print(f"  {s_name}/{l_name}/seed={seed}: "
                      f"train={m['train_accuracy']:.1%}, eval={m['eval_accuracy']:.1%}")

    # Within-loss similarity: for each stakeholder+seed, cosine sim across losses
    within_loss = {}
    for s_name in ["user", "platform", "diversity"]:
        per_seed_sims = []
        for seed in SEEDS:
            seed_models = [m for m in models if m["stakeholder"] == s_name and m["seed"] == seed]
            sims = []
            for i, j in combinations(range(len(seed_models)), 2):
                sims.append(cosine_sim(
                    np.array(seed_models[i]["weights"]),
                    np.array(seed_models[j]["weights"]),
                ))
            per_seed_sims.append(float(np.mean(sims)))
        within_loss[s_name] = {
            "mean": round(float(np.mean(per_seed_sims)), 4),
            "std": round(float(np.std(per_seed_sims)), 4),
            "min": round(float(np.min(per_seed_sims)), 4),
            "per_seed": [round(s, 4) for s in per_seed_sims],
        }
        status = "✓" if within_loss[s_name]["mean"] > 0.85 else "✗"
        print(f"\n  {status} Within-loss ({s_name}): "
              f"{within_loss[s_name]['mean']:.3f} ± {within_loss[s_name]['std']:.3f}")

    # Across-stakeholder similarity: for each loss+seed, cosine sim across stakeholders
    across_stakeholder = {}
    pairs = [("user", "platform"), ("user", "diversity"), ("platform", "diversity")]
    for l_name in LOSS_CONFIGS:
        across_stakeholder[l_name] = {}
        for s_a, s_b in pairs:
            per_seed_sims = []
            for seed in SEEDS:
                m_a = [m for m in models if m["stakeholder"] == s_a and m["loss"] == l_name and m["seed"] == seed][0]
                m_b = [m for m in models if m["stakeholder"] == s_b and m["loss"] == l_name and m["seed"] == seed][0]
                per_seed_sims.append(cosine_sim(np.array(m_a["weights"]), np.array(m_b["weights"])))
            across_stakeholder[l_name][f"{s_a}-{s_b}"] = {
                "mean": round(float(np.mean(per_seed_sims)), 4),
                "std": round(float(np.std(per_seed_sims)), 4),
                "per_seed": [round(s, 4) for s in per_seed_sims],
            }

    print(f"\n  Across-stakeholder (BT):")
    for pair, vals in across_stakeholder["bradley_terry"].items():
        print(f"    {pair}: {vals['mean']:.3f} ± {vals['std']:.3f}")

    # Accuracy summary
    eval_accs = [m["eval_accuracy"] for m in models if m["eval_accuracy"] is not None]
    acc_summary = {
        "train_mean": round(float(np.mean([m["train_accuracy"] for m in models])), 4),
        "eval_mean": round(float(np.mean(eval_accs)), 4),
        "eval_min": round(float(np.min(eval_accs)), 4),
    }

    return {
        "within_loss_similarity": within_loss,
        "across_stakeholder_similarity": across_stakeholder,
        "accuracy_summary": acc_summary,
        "n_models": len(models),
    }


# ---------------------------------------------------------------------------
# Group B: Hyperparameter robustness
# ---------------------------------------------------------------------------

def run_group_b(pool, configs):
    print("\n" + "=" * 60)
    print("Group B: Hyperparameter Robustness")
    print("=" * 60)

    features, _ = pool
    hp_configs = {
        "margin_bt": [
            ("margin_0.1", {"loss_type": LossType.MARGIN_BT, "margin": 0.1}),
            ("margin_0.5", {"loss_type": LossType.MARGIN_BT, "margin": 0.5}),
            ("margin_1.0", {"loss_type": LossType.MARGIN_BT, "margin": 1.0}),
        ],
        "calibrated_bt": [
            ("cal_0.1", {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 0.1}),
            ("cal_1.0", {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 1.0}),
            ("cal_5.0", {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 5.0}),
        ],
    }

    results = {}
    for loss_family, variants in hp_configs.items():
        results[loss_family] = {}
        for s_name in ["user", "platform", "diversity"]:
            models = []
            for hp_name, hp_kwargs in variants:
                m = _train(features, configs[s_name], s_name, hp_name, hp_kwargs, seed=42)
                models.append(m)

            # Pairwise cosine similarity across hyperparams
            sims = []
            for i, j in combinations(range(len(models)), 2):
                sims.append(cosine_sim(
                    np.array(models[i]["weights"]),
                    np.array(models[j]["weights"]),
                ))
            mean_sim = float(np.mean(sims))
            status = "✓" if mean_sim > 0.90 else "✗"
            print(f"  {status} {loss_family}/{s_name}: cos={mean_sim:.3f}")
            results[loss_family][s_name] = {
                "pairwise_cos": [round(s, 4) for s in sims],
                "mean_cos": round(mean_sim, 4),
            }

    return results


# ---------------------------------------------------------------------------
# Group C: Temporal generalization
# ---------------------------------------------------------------------------

def run_group_c(dataset, configs):
    print("\n" + "=" * 60)
    print("Group C: Temporal Generalization")
    print("=" * 60)

    timestamps = [r.timestamp for r in dataset.train_ratings]
    median_ts = int(np.median(timestamps))
    print(f"  Median timestamp: {median_ts}")

    early_pool, early_genres = generate_movielens_content_pool_temporal(
        dataset, before_timestamp=median_ts, min_ratings=3, seed=42
    )
    late_pool, late_genres = generate_movielens_content_pool_temporal(
        dataset, before_timestamp=999999999, min_ratings=3, seed=42
    )
    # Late-only: movies that are NOT in early pool
    # Use full pool for late eval (it includes movies that gained ratings after median)
    print(f"  Early pool: {early_pool.shape[0]} movies, Full pool: {late_pool.shape[0]} movies")

    results = {
        "cutoff_timestamp": median_ts,
        "n_early_movies": int(early_pool.shape[0]),
        "n_full_movies": int(late_pool.shape[0]),
    }

    # Build stakeholder weights from early data only
    from copy import deepcopy
    early_dataset = deepcopy(dataset)
    early_dataset.train_ratings = [r for r in dataset.train_ratings if r.timestamp < median_ts]
    early_configs = build_stakeholder_configs(early_dataset)

    within_loss_temporal = {}
    temporal_eval = {}

    for s_name in ["user", "platform", "diversity"]:
        # Train on early preferences
        models = {}
        for l_name, l_kwargs in LOSS_CONFIGS.items():
            m = _train(early_pool, early_configs[s_name], s_name, l_name, l_kwargs, seed=42)
            models[l_name] = m

        # Within-loss convergence on early data
        weights_list = [np.array(models[l]["weights"]) for l in LOSS_CONFIGS]
        sims = [cosine_sim(weights_list[i], weights_list[j])
                for i, j in combinations(range(len(weights_list)), 2)]
        mean_sim = float(np.mean(sims))
        within_loss_temporal[s_name] = round(mean_sim, 4)

        # Temporal eval: score late-period movies with early-trained weights
        bt_weights = np.array(models["bradley_terry"]["weights"])
        late_pref, late_rej = generate_movielens_preferences(
            late_pool, early_configs[s_name], n_pairs=500, seed=99
        )
        r_pref = late_pref @ bt_weights
        r_rej = late_rej @ bt_weights
        temporal_acc = float(np.mean(r_pref > r_rej))
        temporal_eval[s_name] = round(temporal_acc, 4)

        status_c = "✓" if mean_sim > 0.85 else "✗"
        status_t = "✓" if temporal_acc > 0.75 else "✗"
        print(f"  {status_c} {s_name} within-loss: {mean_sim:.3f}  "
              f"{status_t} temporal eval: {temporal_acc:.1%}")

    results["within_loss_similarity_temporal"] = within_loss_temporal
    results["temporal_eval_accuracy"] = temporal_eval
    return results


# ---------------------------------------------------------------------------
# Group D: Downstream rating prediction
# ---------------------------------------------------------------------------

def run_group_d(dataset, pool, configs):
    print("\n" + "=" * 60)
    print("Group D: Downstream Rating Prediction")
    print("=" * 60)

    features, _ = pool
    results = {}

    # Train BT for each stakeholder
    for s_name in ["user", "platform", "diversity"]:
        m = _train(features, configs[s_name], s_name, "bradley_terry",
                   {"loss_type": LossType.BRADLEY_TERRY}, seed=42)
        w = np.array(m["weights"])

        # Score test movies
        scores = []
        ratings = []
        for rating in dataset.test_ratings:
            movie = dataset.movies.get(rating.movie_id)
            if movie is None:
                continue
            scores.append(float(w @ movie.genres.astype(np.float32)))
            ratings.append(rating.rating)

        rho, p = spearmanr(scores, ratings)
        results[f"{s_name}_bt_spearman"] = round(float(rho), 4)
        results[f"{s_name}_bt_pvalue"] = round(float(p), 6)
        print(f"  {s_name} BT weights: Spearman={rho:.4f} (p={p:.2e})")

    # Random baseline
    rng = np.random.default_rng(42)
    random_w = rng.normal(0, 1, NUM_GENRES).astype(np.float32)
    scores_rand = []
    ratings_rand = []
    for rating in dataset.test_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        scores_rand.append(float(random_w @ movie.genres.astype(np.float32)))
        ratings_rand.append(rating.rating)
    rho_rand, _ = spearmanr(scores_rand, ratings_rand)
    results["random_spearman"] = round(float(rho_rand), 4)
    print(f"  Random weights: Spearman={rho_rand:.4f}")

    # Popularity baseline (mean rating per movie)
    movie_means: dict[int, float] = {}
    movie_counts: dict[int, int] = {}
    for r in dataset.train_ratings:
        movie_means[r.movie_id] = movie_means.get(r.movie_id, 0) + r.rating
        movie_counts[r.movie_id] = movie_counts.get(r.movie_id, 0) + 1
    scores_pop = []
    ratings_pop = []
    for rating in dataset.test_ratings:
        if rating.movie_id in movie_means:
            scores_pop.append(movie_means[rating.movie_id] / movie_counts[rating.movie_id])
            ratings_pop.append(rating.rating)
    rho_pop, _ = spearmanr(scores_pop, ratings_pop)
    results["popularity_spearman"] = round(float(rho_pop), 4)
    print(f"  Popularity baseline: Spearman={rho_pop:.4f}")

    return results


# ---------------------------------------------------------------------------
# Group E: User-group heterogeneity
# ---------------------------------------------------------------------------

def run_group_e(dataset, pool):
    print("\n" + "=" * 60)
    print("Group E: User-Group Preference Heterogeneity")
    print("=" * 60)

    features, _ = pool
    groups = get_user_genre_groups(dataset, min_group_size=50)
    print(f"  Found {len(groups)} groups: {', '.join(f'{k}({len(v)})' for k, v in groups.items())}")

    results = {}
    for group_name, user_ids in groups.items():
        # Group-specific user weights
        group_user_w = compute_user_genre_weights_for_group(dataset, user_ids)

        # Train BT and margin-BT on same group preferences
        pref, rej = generate_movielens_preferences(features, group_user_w, n_pairs=2000, seed=42)
        tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=42)

        config_bt = LossConfig(loss_type=LossType.BRADLEY_TERRY, stakeholder=StakeholderType.USER,
                               learning_rate=0.01, num_epochs=50, batch_size=64)
        config_margin = LossConfig(loss_type=LossType.MARGIN_BT, stakeholder=StakeholderType.USER,
                                   learning_rate=0.01, num_epochs=50, batch_size=64, margin=0.5)

        m_bt = train_with_loss(config_bt, tp, tr, verbose=False,
                               eval_probs_preferred=ep, eval_probs_rejected=er)
        m_margin = train_with_loss(config_margin, tp, tr, verbose=False,
                                   eval_probs_preferred=ep, eval_probs_rejected=er)

        sim = cosine_sim(m_bt.weights, m_margin.weights)
        status = "✓" if sim > 0.80 else "✗"
        print(f"  {status} {group_name} ({len(user_ids)} users): "
              f"BT-margin cos={sim:.3f}, eval acc={m_bt.eval_accuracy:.1%}")

        results[group_name] = {
            "n_users": len(user_ids),
            "within_loss_cos": round(sim, 4),
            "bt_eval_accuracy": round(float(m_bt.eval_accuracy or 0), 4),
            "margin_eval_accuracy": round(float(m_margin.eval_accuracy or 0), 4),
        }

    return results


# ---------------------------------------------------------------------------
# Group F: Genre correlation
# ---------------------------------------------------------------------------

def run_group_f(pool, dataset):
    print("\n" + "=" * 60)
    print("Group F: Genre Correlation Matrix")
    print("=" * 60)

    features, _ = pool
    # Correlation between genre columns
    corr = np.corrcoef(features.T)
    genre_names = dataset.genres if dataset.genres else [f"g{i}" for i in range(NUM_GENRES)]

    # Top correlated pairs (excluding self-correlation)
    pairs = []
    for i in range(NUM_GENRES):
        for j in range(i + 1, NUM_GENRES):
            if not np.isnan(corr[i, j]):
                pairs.append((genre_names[i], genre_names[j], float(corr[i, j])))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print("  Top 5 correlated genre pairs:")
    for g1, g2, r in pairs[:5]:
        print(f"    {g1}-{g2}: {r:.3f}")

    # Condition number (measure of multicollinearity)
    # Use only active genres (columns with variance > 0)
    active = features[:, features.std(axis=0) > 0]
    cond = float(np.linalg.cond(active.T @ active)) if active.shape[1] > 0 else float("inf")
    print(f"  Condition number: {cond:.1f}")

    return {
        "top_correlated_pairs": [(g1, g2, round(r, 4)) for g1, g2, r in pairs[:10]],
        "condition_number": round(cond, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def main():
    parser = argparse.ArgumentParser(description="MovieLens labels-not-loss experiment")
    parser.add_argument("--all", action="store_true", help="Run all groups")
    parser.add_argument("--group", type=str, help="Run specific group (A-F)")
    args = parser.parse_args()

    if not args.all and not args.group:
        args.all = True

    groups_to_run = set()
    if args.all:
        groups_to_run = {"A", "B", "C", "D", "E", "F"}
    elif args.group:
        groups_to_run = {args.group.upper()}

    print("=" * 60)
    print("MovieLens Labels-Not-Loss Experiment")
    print(f"Groups: {', '.join(sorted(groups_to_run))}")
    print("=" * 60)

    data_dir = ROOT / "data" / "ml-100k"
    if not data_dir.exists():
        print(f"ERROR: MovieLens data not found at {data_dir}")
        sys.exit(1)

    t0 = time.time()
    dataset = MovieLensDataset(str(data_dir))
    configs = build_stakeholder_configs(dataset)
    pool = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
    print(f"Loaded: {len(dataset.users)} users, {pool[0].shape[0]} movies")

    results = {
        "config": {
            "n_seeds": len(SEEDS),
            "n_pairs": 2500,
            "eval_fraction": 0.2,
            "loss_types": list(LOSS_CONFIGS.keys()),
            "stakeholders": ["user", "platform", "diversity"],
            "feature_dim": NUM_GENRES,
        }
    }

    # Load synthetic comparison values
    comparison_path = ROOT / "results" / "loss_experiments" / "comparison.json"
    synthetic_comparison = {}
    if comparison_path.exists():
        with open(comparison_path) as f:
            synthetic_comparison = json.load(f)

    if "A" in groups_to_run:
        results["group_a_core"] = run_group_a(pool, configs)
    if "B" in groups_to_run:
        results["group_b_hyperparameter"] = run_group_b(pool, configs)
    if "C" in groups_to_run:
        results["group_c_temporal"] = run_group_c(dataset, configs)
    if "D" in groups_to_run:
        results["group_d_downstream"] = run_group_d(dataset, pool, configs)
    if "E" in groups_to_run:
        results["group_e_user_groups"] = run_group_e(dataset, pool)
    if "F" in groups_to_run:
        results["group_f_genre_correlation"] = run_group_f(pool, dataset)

    # Comparison with synthetic
    if "A" in groups_to_run and synthetic_comparison:
        syn_bt = synthetic_comparison.get("bradley", {}).get("cosine_similarities", {})
        ml_bt = results["group_a_core"]["across_stakeholder_similarity"].get("bradley_terry", {})
        results["comparison_with_synthetic"] = {
            "synthetic_across": {k: round(v, 4) for k, v in syn_bt.items()},
            "movielens_across": {k: round(v["mean"], 4) for k, v in ml_bt.items()},
            "synthetic_within_loss_min": 0.92,
            "movielens_within_loss_min": min(
                v["min"] for v in results["group_a_core"]["within_loss_similarity"].values()
            ),
        }

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    # Summary
    print("\n" + "=" * 60)
    print(f"Complete in {elapsed:.0f}s")
    if "A" in groups_to_run:
        a = results["group_a_core"]
        print(f"\nWithin-loss similarity (mean):")
        for s, v in a["within_loss_similarity"].items():
            print(f"  {s}: {v['mean']:.3f}")
        print(f"Across-stakeholder (BT):")
        for p, v in a["across_stakeholder_similarity"]["bradley_terry"].items():
            print(f"  {p}: {v['mean']:.3f}")
    print("=" * 60)

    out_path = ROOT / "results" / "movielens_labels_not_loss.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
