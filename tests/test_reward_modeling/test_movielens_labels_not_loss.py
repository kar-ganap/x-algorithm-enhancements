"""Smoke tests for MovieLens labels-not-loss experiments.

Lightweight tests that verify the experiment infrastructure works.
Full experiment runs via scripts/experiments/run_movielens_labels_not_loss.py.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ml = _load("movielens", _root / "enhancements" / "data" / "movielens.py")
_st = _load("movielens_stakeholders", _root / "enhancements" / "data" / "movielens_stakeholders.py")
_al = _load("alternative_losses", _root / "enhancements" / "reward_modeling" / "alternative_losses.py")

MovieLensDataset = _ml.MovieLensDataset
build_stakeholder_configs = _st.build_stakeholder_configs
generate_movielens_content_pool = _st.generate_movielens_content_pool
generate_movielens_content_pool_temporal = _st.generate_movielens_content_pool_temporal
generate_movielens_preferences = _st.generate_movielens_preferences
split_preferences = _st.split_preferences
get_user_genre_groups = _st.get_user_genre_groups
NUM_GENRES = _st.NUM_GENRES

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DATA_DIR = _root / "data" / "ml-100k"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@pytest.fixture(scope="module")
def dataset() -> MovieLensDataset:
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available")
    return MovieLensDataset(str(DATA_DIR))


@pytest.fixture(scope="module")
def configs(dataset):
    return build_stakeholder_configs(dataset)


@pytest.fixture(scope="module")
def pool(dataset):
    return generate_movielens_content_pool(dataset, min_ratings=5, seed=42)


def _train_bt(features, weights, loss_type=LossType.BRADLEY_TERRY, **kwargs):
    """Helper: generate prefs, split, train, return model."""
    pref, rej = generate_movielens_preferences(features, weights, n_pairs=2500, seed=42)
    tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=42)

    engagement_kw = {}
    if loss_type == LossType.CALIBRATED_BT:
        engagement_kw["target_engagement_pref"] = np.clip(tp @ weights, 0, 1)
        engagement_kw["target_engagement_rej"] = np.clip(tr @ weights, 0, 1)

    config = LossConfig(
        loss_type=loss_type,
        stakeholder=StakeholderType.USER,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=64,
        **kwargs,
    )
    return train_with_loss(config, tp, tr, verbose=False,
                           eval_probs_preferred=ep, eval_probs_rejected=er,
                           **engagement_kw)


class TestLabelsNotLoss:

    def test_same_labels_different_loss(self, pool, configs):
        """BT and margin-BT on same labels should produce similar weights."""
        features, _ = pool
        w_user = configs["user"]
        model_bt = _train_bt(features, w_user, LossType.BRADLEY_TERRY)
        model_margin = _train_bt(features, w_user, LossType.MARGIN_BT, margin=0.5)
        sim = cosine_sim(model_bt.weights, model_margin.weights)
        assert sim > 0.85, f"Within-loss cos {sim:.3f} too low"

    def test_different_labels_same_loss(self, pool, configs):
        """User and diversity BT should produce different weights."""
        features, _ = pool
        model_user = _train_bt(features, configs["user"])
        model_div = _train_bt(features, configs["diversity"])
        sim = cosine_sim(model_user.weights, model_div.weights)
        assert sim < 0.5, f"Across-stakeholder cos {sim:.3f} too high"

    def test_calibrated_bt_converges(self, pool, configs):
        """Calibrated-BT should converge on D=19 with engagement targets."""
        features, _ = pool
        model = _train_bt(features, configs["user"], LossType.CALIBRATED_BT,
                          calibration_weight=1.0)
        assert model.eval_accuracy is not None
        assert model.eval_accuracy > 0.70, f"Calibrated-BT eval acc {model.eval_accuracy:.1%} too low"


class TestTemporalSplit:

    def test_temporal_split_produces_data(self, dataset):
        """Temporal split should produce non-empty early and late pools."""
        timestamps = [r.timestamp for r in dataset.train_ratings]
        median_ts = int(np.median(timestamps))
        early_pool, early_genres = generate_movielens_content_pool_temporal(
            dataset, before_timestamp=median_ts, min_ratings=3, seed=42
        )
        assert early_pool.shape[0] > 100, f"Only {early_pool.shape[0]} early movies"
        assert early_pool.shape[1] == NUM_GENRES


class TestDownstreamPrediction:

    def test_bt_weights_predict_directionally(self, dataset, pool, configs):
        """BT-learned weights should have non-negative correlation with ratings.

        Genre-level weights are aggregate preference signals, not individual
        rating predictors. We expect weak but positive correlation for user
        weights, and potentially negative for diversity weights (by design).
        The key validation is that user > diversity (directional correctness).
        """
        from scipy.stats import spearmanr

        features, _ = pool

        results = {}
        for name in ["user", "diversity"]:
            model = _train_bt(features, configs[name])
            scores = []
            ratings = []
            for rating in dataset.test_ratings[:1000]:
                movie = dataset.movies.get(rating.movie_id)
                if movie is None:
                    continue
                genre_vec = movie.genres.astype(np.float32)
                scores.append(float(model.weights @ genre_vec))
                ratings.append(rating.rating)
            if len(scores) < 50:
                pytest.skip("Too few test ratings")
            rho, _ = spearmanr(scores, ratings)
            results[name] = rho

        # User weights should correlate more positively with ratings than
        # diversity weights (which penalize popular well-rated genres)
        assert results["user"] > results["diversity"], (
            f"User Spearman {results['user']:.3f} not greater than "
            f"diversity {results['diversity']:.3f}"
        )


class TestUserGroups:

    def test_user_groups_exist(self, dataset):
        """Should find at least 3 genre-based user groups with 50+ members."""
        groups = get_user_genre_groups(dataset, min_group_size=50)
        assert len(groups) >= 3, f"Only {len(groups)} groups found"
        for name, uids in groups.items():
            assert len(uids) >= 50, f"Group {name} has only {len(uids)} users"
