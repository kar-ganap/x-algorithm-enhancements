"""Labels-Not-Loss experiment (registry-driven).

Replicates the paper's central claim on real data and extends with
MovieLens-specific tests (temporal, downstream, user-groups, genre correlation).

Groups:
  A: Core labels-not-loss replication — DATASET-AGNOSTIC
  B: Hyperparameter robustness — DATASET-AGNOSTIC
  C: Temporal generalization — MovieLens only
  D: Downstream rating prediction — MovieLens only (Spearman over 1-5 scale)
  E: User-group heterogeneity — MovieLens only (genre-group clustering)
  F: Genre correlation matrix — MovieLens only

Note (Phase B refactor): Groups C/D/E/F are MovieLens-specific. Running
the script on MIND or Amazon will only execute Groups A and B by default;
attempting to run C/D/E/F on a non-MovieLens dataset returns an error.

Usage:
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --all
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --group A
    uv run python scripts/experiments/run_movielens_labels_not_loss.py --dataset mind-small --groups A,B

Output:
    results/{dataset}_labels_not_loss.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from _dataset_registry import REGISTRY, load_dataset  # noqa: E402


def _load(name: str, path: Path):
    """Load reward-modeling modules via importlib."""
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# Stakeholder name → StakeholderType enum mapping. The enum is only used
# as a label inside LossConfig (no semantic dispatch — confirmed by the
# comment in run_data_budget_all_hidden.py:118). Returns USER as a fallback
# for non-MovieLens stakeholder names.
def _stakeholder_type(name: str) -> "StakeholderType":
    return {
        "user": StakeholderType.USER,
        "platform": StakeholderType.PLATFORM,
        "diversity": StakeholderType.SOCIETY,
    }.get(name, StakeholderType.USER)

LOSS_CONFIGS = {
    "bradley_terry": {"loss_type": LossType.BRADLEY_TERRY},
    "margin_bt": {"loss_type": LossType.MARGIN_BT, "margin": 0.5},
    "calibrated_bt": {"loss_type": LossType.CALIBRATED_BT, "calibration_weight": 1.0},
}

SEEDS = [42, 142, 242, 342, 442]


def _train(ds, features, stakeholder_weights, stakeholder_name, loss_name, loss_kwargs,
           n_pairs=2500, seed=42, epochs=50):
    """Train a single model and return results dict.

    Note: calls the dataset's preferences function directly with the
    explicit ``features`` parameter rather than ``ds.generate_preferences``
    (which uses ``ds.pool``). Group C of this script trains on a temporal
    subset of the pool, so we cannot rely on ``ds.pool`` here.
    """
    pref_fn = getattr(ds.stakeholders_mod, ds.spec.preferences_fn)
    pref, rej = pref_fn(features, stakeholder_weights, n_pairs, seed)
    tp, tr, ep, er = ds.stakeholders_mod.split_preferences(pref, rej, 0.2, seed)

    eng_kw = {}
    if loss_kwargs.get("loss_type") == LossType.CALIBRATED_BT:
        eng_kw["target_engagement_pref"] = np.clip(tp @ stakeholder_weights, 0, 1)
        eng_kw["target_engagement_rej"] = np.clip(tr @ stakeholder_weights, 0, 1)

    config = LossConfig(
        stakeholder=_stakeholder_type(stakeholder_name),
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

def run_group_a(ds, pool, configs):
    stakeholder_names = list(ds.spec.primary_stakeholder_order)
    print("\n" + "=" * 60)
    print("Group A: Core Labels-Not-Loss Replication")
    print(f"  3 losses × {len(stakeholder_names)} stakeholders × {len(SEEDS)} seeds "
          f"= {3*len(stakeholder_names)*len(SEEDS)} runs")
    print("=" * 60)

    features, _ = pool
    models = []
    for seed in SEEDS:
        for s_name in stakeholder_names:
            for l_name, l_kwargs in LOSS_CONFIGS.items():
                m = _train(ds, features, configs[s_name], s_name, l_name, l_kwargs, seed=seed)
                models.append(m)
                print(f"  {s_name}/{l_name}/seed={seed}: "
                      f"train={m['train_accuracy']:.1%}, eval={m['eval_accuracy']:.1%}")

    # Within-loss similarity: for each stakeholder+seed, cosine sim across losses
    within_loss = {}
    for s_name in stakeholder_names:
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
    # Generate all pairs from canonical order (preserves byte-equivalence on
    # MovieLens because the order is ("user", "platform", "diversity") and
    # combinations(2) yields exactly the same 3 pairs in the same order).
    pairs = list(combinations(stakeholder_names, 2))
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

def run_group_b(ds, pool, configs):
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

    stakeholder_names = list(ds.spec.primary_stakeholder_order)
    results = {}
    for loss_family, variants in hp_configs.items():
        results[loss_family] = {}
        for s_name in stakeholder_names:
            models = []
            for hp_name, hp_kwargs in variants:
                m = _train(ds, features, configs[s_name], s_name, hp_name, hp_kwargs, seed=42)
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

def run_group_c(ds, dataset, configs):
    """MovieLens-only: temporal generalization."""
    print("\n" + "=" * 60)
    print("Group C: Temporal Generalization")
    print("=" * 60)

    timestamps = [r.timestamp for r in dataset.train_ratings]
    median_ts = int(np.median(timestamps))
    print(f"  Median timestamp: {median_ts}")

    generate_movielens_content_pool_temporal = ds.stakeholders_mod.generate_movielens_content_pool_temporal
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
    build_stakeholder_configs = ds.stakeholders_mod.build_stakeholder_configs
    generate_movielens_preferences = ds.stakeholders_mod.generate_movielens_preferences
    early_dataset = deepcopy(dataset)
    early_dataset.train_ratings = [r for r in dataset.train_ratings if r.timestamp < median_ts]
    early_configs = build_stakeholder_configs(early_dataset)

    within_loss_temporal = {}
    temporal_eval = {}

    stakeholder_names = list(ds.spec.primary_stakeholder_order)
    for s_name in stakeholder_names:
        # Train on early preferences
        models = {}
        for l_name, l_kwargs in LOSS_CONFIGS.items():
            m = _train(ds, early_pool, early_configs[s_name], s_name, l_name, l_kwargs, seed=42)
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

def run_group_d(ds, dataset, pool, configs):
    """MovieLens-only: downstream rating prediction (Spearman over 1-5 scale)."""
    print("\n" + "=" * 60)
    print("Group D: Downstream Rating Prediction")
    print("=" * 60)

    NUM_GENRES_LOCAL = 19  # MovieLens-only group; safe to hardcode
    features, _ = pool
    results = {}

    # Train BT for each stakeholder
    for s_name in list(ds.spec.primary_stakeholder_order):
        m = _train(ds, features, configs[s_name], s_name, "bradley_terry",
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
    random_w = rng.normal(0, 1, NUM_GENRES_LOCAL).astype(np.float32)
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

def run_group_e(ds, dataset, pool):
    """MovieLens-only: user grouping by top-rated genre."""
    print("\n" + "=" * 60)
    print("Group E: User-Group Preference Heterogeneity")
    print("=" * 60)

    get_user_genre_groups = ds.stakeholders_mod.get_user_genre_groups
    compute_user_genre_weights_for_group = ds.stakeholders_mod.compute_user_genre_weights_for_group
    generate_movielens_preferences = ds.stakeholders_mod.generate_movielens_preferences
    split_preferences = ds.stakeholders_mod.split_preferences

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
    """MovieLens-only: genre correlation matrix."""
    print("\n" + "=" * 60)
    print("Group F: Genre Correlation Matrix")
    print("=" * 60)

    NUM_GENRES_LOCAL = 19  # MovieLens-only group
    features, _ = pool
    # Correlation between genre columns
    corr = np.corrcoef(features.T)
    genre_names = dataset.genres if dataset.genres else [f"g{i}" for i in range(NUM_GENRES_LOCAL)]

    # Top correlated pairs (excluding self-correlation)
    pairs = []
    for i in range(NUM_GENRES_LOCAL):
        for j in range(i + 1, NUM_GENRES_LOCAL):
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
    parser = argparse.ArgumentParser(description="Labels-not-loss experiment (registry-driven)")
    parser.add_argument("--data", default=None,
                        help="(deprecated) data directory; use --dataset")
    parser.add_argument("--dataset", default=None, choices=list(REGISTRY.keys()),
                        help="Dataset name from registry")
    parser.add_argument("--all", action="store_true", help="Run all groups (default for MovieLens)")
    parser.add_argument("--group", type=str, help="Run specific group (A-F, single)")
    parser.add_argument("--groups", type=str, default=None,
                        help="Comma-separated groups to run (e.g. 'A,B,C'). "
                             "Default: 'A,B,C,D,E,F' for MovieLens, 'A,B' for other datasets.")
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
        dataset_name = "ml-100k"

    is_movielens = dataset_name.startswith("ml-")

    # Determine groups to run
    groups_to_run: set[str] = set()
    if args.groups:
        groups_to_run = set(args.groups.upper().split(","))
    elif args.group:
        groups_to_run = {args.group.upper()}
    elif args.all:
        groups_to_run = {"A", "B", "C", "D", "E", "F"} if is_movielens else {"A", "B"}
    else:
        # Default: all on MovieLens, A+B on others
        groups_to_run = {"A", "B", "C", "D", "E", "F"} if is_movielens else {"A", "B"}

    # Hard guard: Groups C/D/E/F are MovieLens-only
    if not is_movielens and (groups_to_run & {"C", "D", "E", "F"}):
        forbidden = sorted(groups_to_run & {"C", "D", "E", "F"})
        print(f"ERROR: Groups {forbidden} are MovieLens-specific and not supported on {dataset_name}.")
        print(f"       Use --groups A,B (default for non-MovieLens datasets).")
        sys.exit(2)

    print("=" * 60)
    print(f"Labels-Not-Loss Experiment ({dataset_name})")
    print(f"Groups: {', '.join(sorted(groups_to_run))}")
    print("=" * 60)

    t0 = time.time()
    ds = load_dataset(dataset_name)
    dataset = ds.dataset
    configs = ds.configs
    pool = (ds.pool, ds.topics)
    print(f"Loaded: {len(dataset.users)} users, {pool[0].shape[0]} items")

    results = {
        "config": {
            "n_seeds": len(SEEDS),
            "n_pairs": 2500,
            "eval_fraction": 0.2,
            "loss_types": list(LOSS_CONFIGS.keys()),
            "stakeholders": list(ds.spec.primary_stakeholder_order),
            "feature_dim": ds.spec.feature_dim,
        }
    }

    # Load synthetic comparison values (MovieLens only)
    comparison_path = ROOT / "results" / "loss_experiments" / "comparison.json"
    synthetic_comparison = {}
    if comparison_path.exists():
        with open(comparison_path) as f:
            synthetic_comparison = json.load(f)

    if "A" in groups_to_run:
        results["group_a_core"] = run_group_a(ds, pool, configs)
    if "B" in groups_to_run:
        results["group_b_hyperparameter"] = run_group_b(ds, pool, configs)
    if "C" in groups_to_run:
        results["group_c_temporal"] = run_group_c(ds, dataset, configs)
    if "D" in groups_to_run:
        results["group_d_downstream"] = run_group_d(ds, dataset, pool, configs)
    if "E" in groups_to_run:
        results["group_e_user_groups"] = run_group_e(ds, dataset, pool)
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

    out_path = ROOT / "results" / f"{dataset_name}_labels_not_loss.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
