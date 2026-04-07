"""Smoke tests for MovieLens Goodhart experiment.

Verifies weight perturbation, Hausdorff distance, and basic
Goodhart dynamics. Full experiment runs via
scripts/experiments/run_movielens_goodhart.py.
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
_kf = _load("k_stakeholder_frontier", _root / "enhancements" / "reward_modeling" / "k_stakeholder_frontier.py")

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

DATA_DIR = _root / "data" / "ml-100k"
DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]


def perturb_weights(w: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative Gaussian perturbation: w * (1 + N(0, σ))."""
    return w * (1 + rng.normal(0, sigma, len(w)))


def hausdorff_distance(
    frontier_a: list[dict], frontier_b: list[dict], dims: list[str],
) -> float:
    """Hausdorff distance between two frontiers in utility space."""
    if not frontier_a or not frontier_b:
        return 0.0
    vals_a = np.array([[p[d] for d in dims] for p in frontier_a])
    vals_b = np.array([[p[d] for d in dims] for p in frontier_b])
    # max over A of min distance to B
    d_a_to_b = max(
        min(np.linalg.norm(a - b) for b in vals_b) for a in vals_a
    )
    d_b_to_a = max(
        min(np.linalg.norm(a - b) for a in vals_a) for b in vals_b
    )
    return max(d_a_to_b, d_b_to_a)


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available")
    return MovieLensDataset(str(DATA_DIR))


@pytest.fixture(scope="module")
def setup(dataset):
    configs = build_stakeholder_configs(dataset)
    pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
    base_probs = pool[np.newaxis, :, :]
    return {"configs": configs, "pool": pool, "genres": genres, "base_probs": base_probs}


class TestPerturbWeights:

    def test_shape(self, setup):
        w = setup["configs"]["user"]
        rng = np.random.default_rng(42)
        perturbed = perturb_weights(w, 0.3, rng)
        assert perturbed.shape == w.shape

    def test_different_from_true(self, setup):
        w = setup["configs"]["user"]
        rng = np.random.default_rng(42)
        perturbed = perturb_weights(w, 0.3, rng)
        assert not np.allclose(perturbed, w)

    def test_zero_sigma_unchanged(self, setup):
        w = setup["configs"]["user"]
        rng = np.random.default_rng(42)
        perturbed = perturb_weights(w, 0.0, rng)
        np.testing.assert_array_equal(perturbed, w)


class TestHausdorffDistance:

    def test_zero_for_identical(self):
        frontier = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        assert hausdorff_distance(frontier, frontier, ["a", "b"]) == 0.0

    def test_positive_for_different(self):
        f1 = [{"a": 0.0, "b": 0.0}]
        f2 = [{"a": 1.0, "b": 1.0}]
        d = hausdorff_distance(f1, f2, ["a", "b"])
        assert d > 0


class TestGoodhartDynamics:

    def test_misspecified_training_runs(self, setup):
        """BT should converge even with misspecified labels."""
        pool = setup["pool"]
        configs = setup["configs"]
        rng = np.random.default_rng(42)

        perturbed = perturb_weights(configs["user"], 0.3, rng)
        pref, rej = generate_movielens_preferences(pool, perturbed, 500, seed=42)
        tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=42)

        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=StakeholderType.USER,
            learning_rate=0.01, num_epochs=50, batch_size=64,
        )
        model = train_with_loss(config, tp, tr, verbose=False,
                                eval_probs_preferred=ep, eval_probs_rejected=er)
        # Should converge — it's learning the misspecified signal
        assert model.eval_accuracy > 0.60
