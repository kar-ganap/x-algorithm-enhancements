"""Verify MovieLens multi-stakeholder foundation (Phase 1).

Produces inspectable output for Phase 1 success criteria:
1. Stakeholder genre weights (shape, values, interpretation)
2. Content pool statistics
3. Pairwise label disagreement rates
4. BT training convergence for each stakeholder (D=19)
5. Cosine similarity matrix of trained weight vectors

Usage:
    uv run python scripts/experiments/verify_movielens_foundation.py

Output:
    results/movielens_foundation.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module loading (avoid __init__.py import chains that pull in Phoenix/grok)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


movielens = _load("movielens", ROOT / "enhancements" / "data" / "movielens.py")
stakeholders = _load(
    "movielens_stakeholders", ROOT / "enhancements" / "data" / "movielens_stakeholders.py"
)
alt_losses = _load(
    "alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py"
)

MovieLensDataset = movielens.MovieLensDataset
build_stakeholder_configs = stakeholders.build_stakeholder_configs
generate_movielens_content_pool = stakeholders.generate_movielens_content_pool
generate_movielens_preferences = stakeholders.generate_movielens_preferences
compute_label_disagreement = stakeholders.compute_label_disagreement
NUM_GENRES = stakeholders.NUM_GENRES

LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 70)
    print("Phase 1: MovieLens Multi-Stakeholder Foundation Verification")
    print("=" * 70)

    data_dir = ROOT / "data" / "ml-100k"
    if not data_dir.exists():
        print(f"ERROR: MovieLens data not found at {data_dir}")
        sys.exit(1)

    results: dict = {}

    # ── Step 1: Load data and build stakeholder configs ──────────────────
    print("\n1. Loading MovieLens-100K...")
    ds = MovieLensDataset(str(data_dir))
    print(f"   Users: {len(ds.users)}, Movies: {len(ds.movies)}, "
          f"Train ratings: {len(ds.train_ratings)}")

    configs = build_stakeholder_configs(ds)
    genre_names = ds.genres if ds.genres else [f"genre_{i}" for i in range(NUM_GENRES)]

    results["dataset"] = {
        "num_users": len(ds.users),
        "num_movies": len(ds.movies),
        "num_train_ratings": len(ds.train_ratings),
        "num_genres": NUM_GENRES,
    }

    print("\n2. Stakeholder genre weights:")
    results["stakeholder_weights"] = {}
    for name in ["user", "platform", "diversity"]:
        w = configs[name]
        top_3 = sorted(range(NUM_GENRES), key=lambda i: w[i], reverse=True)[:3]
        bot_3 = sorted(range(NUM_GENRES), key=lambda i: w[i])[:3]
        print(f"\n   {name}:")
        print(f"     Shape: {w.shape}, dtype: {w.dtype}")
        print(f"     Range: [{w.min():.3f}, {w.max():.3f}], mean: {w.mean():.3f}")
        print(f"     Top 3: {', '.join(f'{genre_names[i]}={w[i]:.3f}' for i in top_3)}")
        print(f"     Bot 3: {', '.join(f'{genre_names[i]}={w[i]:.3f}' for i in bot_3)}")
        results["stakeholder_weights"][name] = {
            "values": w.tolist(),
            "min": float(w.min()),
            "max": float(w.max()),
            "mean": float(w.mean()),
            "top_3_genres": [genre_names[i] for i in top_3],
            "bot_3_genres": [genre_names[i] for i in bot_3],
        }

    # ── Step 2: Content pool ────────────────────────────────────────────
    print("\n3. Content pool:")
    pool, genres = generate_movielens_content_pool(ds, min_ratings=5, seed=42)
    print(f"   Shape: {pool.shape} (movies × genres)")
    print(f"   Genre distribution: {np.bincount(genres, minlength=NUM_GENRES).tolist()}")
    print(f"   Feature range: [{pool.min():.3f}, {pool.max():.3f}]")
    print(f"   Mean features per movie: {(pool > 0).sum(axis=1).mean():.1f} genres")

    results["content_pool"] = {
        "num_movies": int(pool.shape[0]),
        "feature_dim": int(pool.shape[1]),
        "genre_distribution": np.bincount(genres, minlength=NUM_GENRES).tolist(),
        "feature_range": [float(pool.min()), float(pool.max())],
        "mean_genres_per_movie": float((pool > 0).sum(axis=1).mean()),
    }

    # ── Step 3: Label disagreement ──────────────────────────────────────
    print("\n4. Pairwise label disagreement (10,000 pairs):")
    pairs = [("user", "platform"), ("user", "diversity"), ("platform", "diversity")]
    results["disagreement"] = {}
    for name_a, name_b in pairs:
        d = compute_label_disagreement(
            pool, configs[name_a], configs[name_b], n_pairs=10000, seed=42
        )
        status = "✓" if d > 0.05 else "✗"
        print(f"   {status} {name_a}-{name_b}: {d:.1%}")
        results["disagreement"][f"{name_a}-{name_b}"] = round(d, 4)

    # ── Step 4: BT training convergence ─────────────────────────────────
    print("\n5. BT training convergence (D=19, 2500 pairs, 80/20 split, 50 epochs):")
    stakeholder_map = {
        "user": StakeholderType.USER,
        "platform": StakeholderType.PLATFORM,
        "diversity": StakeholderType.SOCIETY,
    }
    trained_weights = {}
    results["bt_training"] = {}

    # Import split utility
    split_preferences = stakeholders.split_preferences

    for name, st_type in stakeholder_map.items():
        pref, rej = generate_movielens_preferences(
            pool, configs[name], n_pairs=2500, seed=42
        )
        train_pref, train_rej, eval_pref, eval_rej = split_preferences(
            pref, rej, eval_fraction=0.2, seed=42
        )
        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=st_type,
            learning_rate=0.01,
            num_epochs=50,
            batch_size=64,
        )
        t0 = time.time()
        model = train_with_loss(
            config, train_pref, train_rej, verbose=False,
            eval_probs_preferred=eval_pref, eval_probs_rejected=eval_rej,
        )
        elapsed = time.time() - t0

        trained_weights[name] = model.weights
        converged = model.loss_history[-1] < model.loss_history[0]
        eval_acc = model.eval_accuracy or 0.0
        status = "✓" if eval_acc > 0.80 and converged else "✗"

        print(f"   {status} {name}: train={model.accuracy:.1%}, "
              f"held-out={eval_acc:.1%}, "
              f"loss {model.loss_history[0]:.4f}→{model.loss_history[-1]:.4f}, "
              f"{elapsed:.1f}s")

        results["bt_training"][name] = {
            "train_accuracy": round(model.accuracy, 4),
            "eval_accuracy": round(eval_acc, 4),
            "gap": round(abs(model.accuracy - eval_acc), 4),
            "loss_start": round(model.loss_history[0], 4),
            "loss_end": round(model.loss_history[-1], 4),
            "converged": converged,
            "weights": model.weights.tolist(),
            "training_time_s": round(elapsed, 2),
        }

    # ── Step 5: Cosine similarity matrix ────────────────────────────────
    print("\n6. Trained weight vector cosine similarity:")
    results["cosine_similarity"] = {}
    for name_a, name_b in pairs:
        sim = cosine_sim(trained_weights[name_a], trained_weights[name_b])
        print(f"   {name_a}-{name_b}: {sim:.3f}")
        results["cosine_similarity"][f"{name_a}-{name_b}"] = round(sim, 4)

    min_sim = min(results["cosine_similarity"].values())
    print(f"\n   Min similarity: {min_sim:.3f} {'✓ < 0.9' if min_sim < 0.9 else '✗ ≥ 0.9'}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Phase 1 Success Criteria:")
    all_pass = True

    checks = [
        ("BT converges for all stakeholders (held-out acc > 80%)",
         all(r["converged"] and r["eval_accuracy"] > 0.80 for r in results["bt_training"].values())),
        ("All disagreement rates > 5%",
         all(d > 0.05 for d in results["disagreement"].values())),
        ("Diversity disagreement > 40%",
         results["disagreement"].get("user-diversity", 0) > 0.40),
        ("Trained weight vectors differentiated (min cos < 0.9)",
         min_sim < 0.9),
    ]

    for desc, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {desc}")
        if not passed:
            all_pass = False

    print("=" * 70)

    # ── Save results ────────────────────────────────────────────────────
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super().default(obj)

    out_path = ROOT / "results" / "movielens_foundation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=_Encoder)
    print(f"\nResults saved to {out_path}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
