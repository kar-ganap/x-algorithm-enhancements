"""Tests for expanded direction condition validation.

Tests Method A (different targets) and Method B (named stakeholders).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).parent.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ml = _load("movielens", _root / "enhancements" / "data" / "movielens.py")
_st = _load("movielens_stakeholders", _root / "enhancements" / "data" / "movielens_stakeholders.py")

MovieLensDataset = _ml.MovieLensDataset
NUM_GENRES = _st.NUM_GENRES
DATA_DIR = _root / "data" / "ml-100k"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ── Named stakeholder definitions ──

def compute_creator_weights(dataset):
    """Creator: weights ∝ number of movies per genre (rewards supply)."""
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    max_val = np.max(genre_counts)
    return (genre_counts / max_val).astype(np.float32) if max_val > 0 else genre_counts.astype(np.float32)


def compute_advertiser_weights(dataset):
    """Advertiser: weights ∝ total ratings per genre (rewards eyeballs)."""
    genre_ratings = np.zeros(NUM_GENRES, dtype=np.float64)
    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is not None:
            genre_ratings += movie.genres
    max_val = np.max(genre_ratings)
    return (genre_ratings / max_val).astype(np.float32) if max_val > 0 else genre_ratings.astype(np.float32)


def compute_niche_weights(dataset):
    """Niche enthusiast: upweights rare genres, downweights popular."""
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    active = genre_counts > 0
    median_count = np.median(genre_counts[active]) if np.any(active) else 1.0
    weights = np.zeros(NUM_GENRES, dtype=np.float32)
    weights[active & (genre_counts < median_count)] = 1.0
    weights[active & (genre_counts >= median_count)] = -1.0
    max_abs = np.max(np.abs(weights))
    return weights / max_abs if max_abs > 0 else weights


def compute_mainstream_weights(dataset):
    """Mainstream: upweights top-3 most popular genres."""
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)
    for movie in dataset.movies.values():
        genre_counts += movie.genres
    top3 = np.argsort(genre_counts)[-3:]
    weights = np.zeros(NUM_GENRES, dtype=np.float32)
    weights[top3] = 1.0
    return weights


def compute_moderator_weights(dataset):
    """Content moderator: downweights genres with low ratings."""
    genre_low_rating_frac = np.zeros(NUM_GENRES, dtype=np.float64)
    genre_total = np.zeros(NUM_GENRES, dtype=np.float64)
    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_total[g] += 1
                if rating.rating <= 2:
                    genre_low_rating_frac[g] += 1
    active = genre_total > 0
    frac = np.zeros(NUM_GENRES, dtype=np.float32)
    frac[active] = (genre_low_rating_frac[active] / genre_total[active]).astype(np.float32)
    # Moderator penalizes high-low-rating genres
    weights = -frac
    max_abs = np.max(np.abs(weights))
    return weights / max_abs if max_abs > 0 else weights


NAMED_STAKEHOLDER_FNS = {
    "creator": compute_creator_weights,
    "advertiser": compute_advertiser_weights,
    "niche": compute_niche_weights,
    "mainstream": compute_mainstream_weights,
    "moderator": compute_moderator_weights,
}


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available")
    return MovieLensDataset(str(DATA_DIR))


class TestNamedStakeholders:

    def test_all_produce_valid_weights(self, dataset):
        for name, fn in NAMED_STAKEHOLDER_FNS.items():
            w = fn(dataset)
            assert w.shape == (NUM_GENRES,), f"{name} wrong shape"
            assert np.any(w != 0), f"{name} is all zeros"

    def test_cosine_range(self, dataset):
        """Named stakeholders should span a range of cosines with user."""
        user_w = _st.compute_user_genre_weights(dataset)
        cosines = {}
        for name, fn in NAMED_STAKEHOLDER_FNS.items():
            w = fn(dataset)
            cosines[name] = cosine_sim(user_w, w)
        # Should have some variety
        cos_values = list(cosines.values())
        assert max(cos_values) - min(cos_values) > 0.3, (
            f"Cosine range too narrow: {cosines}"
        )

    def test_niche_opposes_mainstream(self, dataset):
        """Niche and mainstream should be anti-correlated."""
        niche_w = compute_niche_weights(dataset)
        mainstream_w = compute_mainstream_weights(dataset)
        cos = cosine_sim(niche_w, mainstream_w)
        assert cos < 0, f"Niche-mainstream cos {cos:.3f} should be negative"


class TestMethodA:

    def test_different_targets_give_different_cosines(self, dataset):
        """Platform-target and diversity-target should produce different cos values."""
        configs = _st.build_stakeholder_configs(dataset)
        # cos(platform, diversity) ≠ cos(user, diversity)
        cos_user_div = cosine_sim(configs["user"], configs["diversity"])
        cos_plat_div = cosine_sim(configs["platform"], configs["diversity"])
        assert abs(cos_user_div - cos_plat_div) > 0.05
