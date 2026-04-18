"""Tests for MIND-small multi-stakeholder pipeline.

Tests stakeholder utility definitions, preference pair generation,
BT training convergence on 35-dim news features, and stakeholder differentiation.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_module(name: str, path: Path):
    """Load a module directly to avoid __init__.py import chains."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_root = Path(__file__).parent.parent.parent
_mind_mod = _load_module("mind", _root / "enhancements" / "data" / "mind.py")
_stakeholders_mod = _load_module(
    "mind_stakeholders", _root / "enhancements" / "data" / "mind_stakeholders.py"
)

MINDDataset = _mind_mod.MINDDataset
NUM_FEATURES = _stakeholders_mod.NUM_FEATURES
build_stakeholder_configs = _stakeholders_mod.build_stakeholder_configs
compute_label_disagreement = _stakeholders_mod.compute_label_disagreement
generate_mind_content_pool = _stakeholders_mod.generate_mind_content_pool
generate_mind_preferences = _stakeholders_mod.generate_mind_preferences

DATA_DIR = Path("data/mind-small")
STAKEHOLDER_NAMES = ["reader", "publisher", "advertiser", "journalist", "civic_diversity"]


@pytest.fixture(scope="module")
def dataset() -> MINDDataset:
    if not DATA_DIR.exists():
        pytest.skip("MIND data not available at data/mind-small")
    return MINDDataset(str(DATA_DIR))


@pytest.fixture(scope="module")
def stakeholder_weights(dataset: MINDDataset) -> dict[str, np.ndarray]:
    return build_stakeholder_configs(dataset)


@pytest.fixture(scope="module")
def content_pool(dataset: MINDDataset) -> tuple[np.ndarray, np.ndarray]:
    return generate_mind_content_pool(dataset, min_impressions=5, seed=42)


class TestWeightProperties:
    """Test stakeholder weight computation."""

    def test_all_stakeholders_present(self, stakeholder_weights: dict[str, np.ndarray]) -> None:
        assert set(stakeholder_weights.keys()) == set(STAKEHOLDER_NAMES)

    def test_weight_shape(self, stakeholder_weights: dict[str, np.ndarray]) -> None:
        for name, weights in stakeholder_weights.items():
            assert weights.shape == (NUM_FEATURES,), f"{name} weights wrong shape"

    def test_weights_nonzero(self, stakeholder_weights: dict[str, np.ndarray]) -> None:
        for name, weights in stakeholder_weights.items():
            assert np.any(weights != 0), f"{name} weights are all zero"

    def test_weights_normalized(self, stakeholder_weights: dict[str, np.ndarray]) -> None:
        for name, weights in stakeholder_weights.items():
            assert np.max(np.abs(weights)) == pytest.approx(1.0, abs=0.01), (
                f"{name} weights not unit-max-normalized: max={np.max(np.abs(weights))}"
            )


class TestContentPool:
    """Test content pool generation."""

    def test_content_pool_shape(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        features, topics = content_pool
        assert features.ndim == 2
        assert features.shape[1] == NUM_FEATURES
        assert topics.shape == (features.shape[0],)
        assert features.shape[0] > 1000, f"Pool too small: {features.shape[0]}"

    def test_content_features_range(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        features, _ = content_pool
        assert np.all(features >= 0)
        assert np.all(features <= 1.0 + 1e-6)

    def test_topics_valid(self, content_pool: tuple[np.ndarray, np.ndarray]) -> None:
        _, topics = content_pool
        assert np.all(topics >= 0)
        assert np.all(topics < 17), "Topics should be in [0, 16] for 17 MIND categories"


class TestPreferenceGeneration:
    """Test preference pair generation."""

    def test_preference_shape(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        features, _ = content_pool
        pref, rej = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=100, seed=42
        )
        assert pref.shape == (100, NUM_FEATURES)
        assert rej.shape == (100, NUM_FEATURES)

    def test_preference_deterministic(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        features, _ = content_pool
        pref1, rej1 = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=50, seed=123
        )
        pref2, rej2 = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=50, seed=123
        )
        np.testing.assert_array_equal(pref1, pref2)
        np.testing.assert_array_equal(rej1, rej2)

    def test_different_seeds_differ(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        features, _ = content_pool
        pref1, _ = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=50, seed=1
        )
        pref2, _ = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=50, seed=2
        )
        assert not np.array_equal(pref1, pref2)


class TestStakeholderDisagreement:
    """Test that stakeholders actually disagree on preferences."""

    def test_minimum_disagreement(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        features, _ = content_pool
        high_disagreement_count = 0
        for i, a in enumerate(STAKEHOLDER_NAMES):
            for b in STAKEHOLDER_NAMES[i + 1:]:
                disagreement = compute_label_disagreement(
                    features, stakeholder_weights[a], stakeholder_weights[b],
                    n_pairs=5000, seed=42,
                )
                if disagreement > 0.05:
                    high_disagreement_count += 1
        assert high_disagreement_count >= 3, (
            f"Only {high_disagreement_count} pairs have >5% disagreement"
        )

    def test_publisher_civic_diversity_high_disagreement(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """Publisher and civic_diversity are strongly opposed (cos ≈ -0.84)."""
        features, _ = content_pool
        disagreement = compute_label_disagreement(
            features, stakeholder_weights["publisher"],
            stakeholder_weights["civic_diversity"],
            n_pairs=10000, seed=42,
        )
        assert disagreement > 0.40, (
            f"publisher-civic_diversity disagreement {disagreement:.1%} too low"
        )


class TestBTConvergence:
    """Test that BT training converges on D=35 MIND features."""

    def test_bt_convergence(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        _alt_losses = _load_module(
            "alternative_losses",
            _root / "enhancements" / "reward_modeling" / "alternative_losses.py",
        )
        split_preferences = _stakeholders_mod.split_preferences

        features, _ = content_pool
        pref, rej = generate_mind_preferences(
            features, stakeholder_weights["reader"], n_pairs=2000, seed=42
        )
        train_pref, train_rej, eval_pref, eval_rej = split_preferences(
            pref, rej, eval_fraction=0.2, seed=42
        )

        config = _alt_losses.LossConfig(
            loss_type=_alt_losses.LossType.BRADLEY_TERRY,
            stakeholder=_alt_losses.StakeholderType.USER,
            learning_rate=0.01, num_epochs=50, batch_size=64,
        )
        model = _alt_losses.train_with_loss(
            config, train_pref, train_rej, verbose=False,
            eval_probs_preferred=eval_pref, eval_probs_rejected=eval_rej,
        )

        assert model.eval_accuracy is not None
        assert model.eval_accuracy > 0.65, (
            f"Held-out accuracy {model.eval_accuracy:.1%} below 65%"
        )
        assert model.weights.shape == (NUM_FEATURES,)

    def test_stakeholder_differentiation(
        self,
        content_pool: tuple[np.ndarray, np.ndarray],
        stakeholder_weights: dict[str, np.ndarray],
    ) -> None:
        """Trained weight vectors for different stakeholders should differ."""
        _alt_losses = _load_module(
            "alternative_losses",
            _root / "enhancements" / "reward_modeling" / "alternative_losses.py",
        )

        features, _ = content_pool
        trained = {}
        for name in ["reader", "publisher", "civic_diversity"]:
            pref, rej = generate_mind_preferences(
                features, stakeholder_weights[name], n_pairs=2000, seed=42
            )
            config = _alt_losses.LossConfig(
                loss_type=_alt_losses.LossType.BRADLEY_TERRY,
                stakeholder=_alt_losses.StakeholderType.USER,
                learning_rate=0.01, num_epochs=50, batch_size=64,
            )
            model = _alt_losses.train_with_loss(config, pref, rej, verbose=False)
            trained[name] = model.weights

        def cosine_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

        min_sim = min(
            cosine_sim(trained[a], trained[b])
            for a in trained for b in trained if a < b
        )
        assert min_sim < 0.95, (
            f"Min cosine similarity {min_sim:.3f} — stakeholders not differentiated"
        )
