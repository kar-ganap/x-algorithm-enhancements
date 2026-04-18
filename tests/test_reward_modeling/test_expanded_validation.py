"""Tests for expanded direction condition validation.

Tests Method A (different targets) and Method B (named stakeholders).

Note (Phase B refactor): the 5 named-stakeholder helper functions used
to be duplicated in this file. They now live in
``enhancements/data/movielens_stakeholders.py`` as
``build_named_stakeholder_configs``; this test imports from there.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_root = Path(__file__).parent.parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ml = _load("movielens", _root / "enhancements" / "data" / "movielens.py")
_st = _load("movielens_stakeholders", _root / "enhancements" / "data" / "movielens_stakeholders.py")

MovieLensDataset = _ml.MovieLensDataset
NUM_GENRES = _st.NUM_GENRES
DATA_DIR = _root / "data" / "ml-100k"


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available")
    return MovieLensDataset(str(DATA_DIR))


class TestNamedStakeholders:

    def test_all_produce_valid_weights(self, dataset):
        configs = _st.build_named_stakeholder_configs(dataset)
        assert set(configs.keys()) == {"creator", "advertiser", "niche", "mainstream", "moderator"}
        for name, w in configs.items():
            assert w.shape == (NUM_GENRES,), f"{name} wrong shape"
            assert np.any(w != 0), f"{name} is all zeros"

    def test_cosine_range(self, dataset):
        """Named stakeholders should span a range of cosines with user."""
        user_w = _st.compute_user_genre_weights(dataset)
        configs = _st.build_named_stakeholder_configs(dataset)
        cos_values = [cosine_sim(user_w, w) for w in configs.values()]
        assert max(cos_values) - min(cos_values) > 0.3, (
            f"Cosine range too narrow: {cos_values}"
        )

    def test_niche_opposes_mainstream(self, dataset):
        """Niche and mainstream should be anti-correlated."""
        configs = _st.build_named_stakeholder_configs(dataset)
        cos = cosine_sim(configs["niche"], configs["mainstream"])
        assert cos < 0, f"Niche-mainstream cos {cos:.3f} should be negative"


class TestMethodA:

    def test_different_targets_give_different_cosines(self, dataset):
        """Platform-target and diversity-target should produce different cos values."""
        configs = _st.build_stakeholder_configs(dataset)
        # cos(platform, diversity) ≠ cos(user, diversity)
        cos_user_div = cosine_sim(configs["user"], configs["diversity"])
        cos_plat_div = cosine_sim(configs["platform"], configs["diversity"])
        assert abs(cos_user_div - cos_plat_div) > 0.05


class TestMethodAMultiDataset:
    """Test that Method A direction condition structure is valid across datasets.

    Uses the dataset registry to load each available dataset and verify
    that stakeholder configs produce well-formed cosine matrices. Does NOT
    run BT training — just checks shapes and geometry.
    """

    @staticmethod
    def _load_registry():
        sys.path.insert(0, str(_root / "scripts"))
        reg = _load("_dataset_registry", _root / "scripts" / "_dataset_registry.py")
        return reg

    @pytest.mark.parametrize("dataset_name", ["ml-100k", "mind-small", "amazon-kindle"])
    def test_stakeholder_cosine_matrix(self, dataset_name):
        reg = self._load_registry()
        data_dir = _root / "data" / dataset_name
        if not data_dir.exists():
            pytest.skip(f"Data not available at {data_dir}")

        ds = reg.load_dataset(dataset_name)
        configs = ds.configs
        names = list(ds.spec.primary_stakeholder_order)
        K = len(names)

        # All stakeholders have correct feature dimension
        for name in names:
            assert configs[name].shape == (ds.spec.feature_dim,), (
                f"{dataset_name}/{name} wrong shape"
            )

        # Cosine matrix is K×K, symmetric, diagonal = 1
        cos_matrix = np.zeros((K, K))
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                cos_matrix[i, j] = cosine_sim(configs[a], configs[b])

        np.testing.assert_allclose(np.diag(cos_matrix), 1.0, atol=1e-6)
        np.testing.assert_allclose(cos_matrix, cos_matrix.T, atol=1e-6)

        # At least one pair should have cos < 0 (otherwise no Goodhart)
        off_diag = cos_matrix[np.triu_indices(K, k=1)]
        assert np.any(off_diag < 0), (
            f"{dataset_name}: no negative cosine pairs — direction condition is trivial"
        )
