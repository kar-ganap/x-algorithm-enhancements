"""Tests for MovieLens multi-stakeholder pipeline.

Tests stakeholder utility definitions, preference pair generation,
BT training convergence on genre features, and stakeholder differentiation.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, path: Path):
    """Load a module directly to avoid __init__.py import chains."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_root = Path(__file__).parent.parent.parent
_movielens_mod = _load_module("movielens", _root / "enhancements" / "data" / "movielens.py")
_stakeholders_mod = _load_module(
    "movielens_stakeholders", _root / "enhancements" / "data" / "movielens_stakeholders.py"
)

MovieLensDataset = _movielens_mod.MovieLensDataset
NUM_GENRES = _stakeholders_mod.NUM_GENRES
build_stakeholder_configs = _stakeholders_mod.build_stakeholder_configs
compute_diversity_genre_weights = _stakeholders_mod.compute_diversity_genre_weights
compute_label_disagreement = _stakeholders_mod.compute_label_disagreement
compute_platform_genre_weights = _stakeholders_mod.compute_platform_genre_weights
compute_user_genre_weights = _stakeholders_mod.compute_user_genre_weights
generate_movielens_content_pool = _stakeholders_mod.generate_movielens_content_pool
generate_movielens_preferences = _stakeholders_mod.generate_movielens_preferences

DATA_DIR = Path("data/ml-100k")


@pytest.fixture(scope="module")
def dataset() -> MovieLensDataset:
    """Load MovieLens dataset once for all tests."""
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available at data/ml-100k")
    return MovieLensDataset(str(DATA_DIR))


@pytest.fixture(scope="module")
def stakeholder_weights(dataset: MovieLensDataset) -> dict[str, np.ndarray]:
    """Build stakeholder configs once for all tests."""
    return build_stakeholder_configs(dataset)


@pytest.fixture(scope="module")
def content_pool(dataset: MovieLensDataset) -> tuple[np.ndarray, np.ndarray]:
    """Build content pool once for all tests."""
    return generate_movielens_content_pool(dataset, min_ratings=5, seed=42)


class TestGenreWeights:
    """Test stakeholder genre weight computation."""

    def test_genre_weights_shape(self, stakeholder_weights: dict[str, np.ndarray]) -> None:
        for name, weights in stakeholder_weights.items():
            assert weights.shape == (NUM_GENRES,), f"{name} weights wrong shape"
            assert weights.dtype == np.float32, f"{name} weights wrong dtype"

    def test_user_weights_centered(self, dataset: MovieLensDataset) -> None:
        """User weights should reflect rating patterns: some positive, some negative."""
        weights = compute_user_genre_weights(dataset)
        assert np.any(weights > 0), "User weights should have positive values"
        assert np.any(weights < 0), "User weights should have negative values"
        assert np.max(np.abs(weights)) == pytest.approx(1.0, abs=0.01)

    def test_platform_weights_nonnegative(self, dataset: MovieLensDataset) -> None:
        """Platform weights should be non-negative (popularity * avg_rating)."""
        weights = compute_platform_genre_weights(dataset)
        assert np.all(weights >= 0), "Platform weights should be non-negative"
        assert np.max(weights) == pytest.approx(1.0, abs=0.01)

    def test_diversity_weights_zero_sum(self, dataset: MovieLensDataset) -> None:
        """Diversity weights should approximately sum to zero."""
        weights = compute_diversity_genre_weights(dataset)
        assert np.any(weights > 0), "Diversity should upweight rare genres"
        assert np.any(weights < 0), "Diversity should downweight popular genres"
        # Not exactly zero due to normalization, but close
        assert abs(np.sum(weights)) < 1.0


class TestContentPool:
    """Test content pool generation."""

    def test_content_pool_shape(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        features, genres = content_pool
        assert features.ndim == 2
        assert features.shape[1] == NUM_GENRES
        assert genres.shape == (features.shape[0],)
        # Should have a reasonable number of movies (MovieLens-100K has ~1600+)
        assert features.shape[0] > 500

    def test_content_features_nonnegative(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        """Features are genre_vec * (avg_rating / 5), so should be in [0, 1]."""
        features, _ = content_pool
        assert np.all(features >= 0)
        assert np.all(features <= 1.0 + 1e-6)

    def test_content_genres_valid(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        _, genres = content_pool
        assert np.all(genres >= 0)
        assert np.all(genres < NUM_GENRES)


class TestPreferenceGeneration:
    """Test preference pair generation."""

    def test_preference_shape(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        features, _ = content_pool
        pref, rej = generate_movielens_preferences(
            features, stakeholder_weights["user"], n_pairs=100, seed=42
        )
        assert pref.shape == (100, NUM_GENRES)
        assert rej.shape == (100, NUM_GENRES)

    def test_preference_deterministic(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """Same seed should produce same preferences."""
        features, _ = content_pool
        pref1, rej1 = generate_movielens_preferences(
            features, stakeholder_weights["user"], n_pairs=50, seed=123
        )
        pref2, rej2 = generate_movielens_preferences(
            features, stakeholder_weights["user"], n_pairs=50, seed=123
        )
        np.testing.assert_array_equal(pref1, pref2)
        np.testing.assert_array_equal(rej1, rej2)


class TestStakeholderDisagreement:
    """Test that stakeholders actually disagree on preferences."""

    def test_stakeholder_disagreement(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """Stakeholder pairs should disagree enough for meaningful differentiation.

        User-platform agreement is naturally high (~92%) since both prefer
        well-rated content. The key tension is with diversity (>50% disagreement).
        """
        features, _ = content_pool
        pairs = [("user", "platform"), ("user", "diversity"), ("platform", "diversity")]
        for name_a, name_b in pairs:
            disagreement = compute_label_disagreement(
                features,
                stakeholder_weights[name_a],
                stakeholder_weights[name_b],
                n_pairs=10000,
                seed=42,
            )
            assert disagreement > 0.05, (
                f"{name_a}-{name_b} disagreement {disagreement:.1%} is below 5% threshold"
            )
        # Diversity pairs should have much higher disagreement
        div_user = compute_label_disagreement(
            features, stakeholder_weights["user"], stakeholder_weights["diversity"],
            n_pairs=10000, seed=42,
        )
        assert div_user > 0.40, f"user-diversity disagreement {div_user:.1%} too low"


class TestBTConvergence:
    """Test that BT training converges on D=19 genre features."""

    def test_bt_convergence_d19(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """BT training should converge on genre features with held-out accuracy > 80%."""
        _alt_losses = _load_module(
            "alternative_losses",
            _root / "enhancements" / "reward_modeling" / "alternative_losses.py",
        )
        LossConfig = _alt_losses.LossConfig
        LossType = _alt_losses.LossType
        StakeholderType = _alt_losses.StakeholderType
        train_with_loss = _alt_losses.train_with_loss

        split_preferences = _stakeholders_mod.split_preferences

        features, _ = content_pool
        pref, rej = generate_movielens_preferences(
            features, stakeholder_weights["user"], n_pairs=2500, seed=42
        )
        train_pref, train_rej, eval_pref, eval_rej = split_preferences(
            pref, rej, eval_fraction=0.2, seed=42
        )

        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=StakeholderType.USER,
            learning_rate=0.01,
            num_epochs=50,
            batch_size=64,
        )
        model = train_with_loss(
            config, train_pref, train_rej, verbose=False,
            eval_probs_preferred=eval_pref, eval_probs_rejected=eval_rej,
        )

        assert model.eval_accuracy is not None, "Held-out accuracy not computed"
        assert model.eval_accuracy > 0.80, f"Held-out accuracy {model.eval_accuracy:.1%} below 80%"
        assert model.weights.shape == (NUM_GENRES,)
        assert model.loss_history[-1] < model.loss_history[0]
        # Gap between train and held-out should be small
        gap = abs(model.accuracy - model.eval_accuracy)
        assert gap < 0.05, f"Train/held-out gap {gap:.1%} too large"

    def test_stakeholder_weight_vectors_differ(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """Trained weight vectors for different stakeholders should differ."""
        _alt_losses = _load_module(
            "alternative_losses",
            _root / "enhancements" / "reward_modeling" / "alternative_losses.py",
        )
        LossConfig = _alt_losses.LossConfig
        LossType = _alt_losses.LossType
        StakeholderType = _alt_losses.StakeholderType
        train_with_loss = _alt_losses.train_with_loss

        features, _ = content_pool
        trained = {}
        stakeholder_map = {
            "user": StakeholderType.USER,
            "platform": StakeholderType.PLATFORM,
            "diversity": StakeholderType.SOCIETY,
        }

        for name, st in stakeholder_map.items():
            pref, rej = generate_movielens_preferences(
                features, stakeholder_weights[name], n_pairs=2000, seed=42
            )
            config = LossConfig(
                loss_type=LossType.BRADLEY_TERRY,
                stakeholder=st,
                learning_rate=0.01,
                num_epochs=50,
                batch_size=64,
            )
            model = train_with_loss(config, pref, rej, verbose=False)
            trained[name] = model.weights

        # At least one pair should have cosine similarity < 0.9
        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        pairs = [("user", "platform"), ("user", "diversity"), ("platform", "diversity")]
        min_sim = min(cosine_sim(trained[a], trained[b]) for a, b in pairs)
        assert min_sim < 0.9, (
            f"Minimum cosine similarity {min_sim:.3f} — stakeholders not differentiated enough"
        )
