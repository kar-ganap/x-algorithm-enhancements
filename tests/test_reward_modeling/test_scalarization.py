"""Smoke tests for scalarization baseline.

Verifies simplex grid generation, mixed preference training,
and basic frontier comparison. Full experiment runs via
scripts/experiments/run_scalarization_baseline.py.
"""

import importlib.util
from itertools import combinations
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

compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier
extract_pareto_front_nd = _kf.extract_pareto_front_nd

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss

DATA_DIR = _root / "data" / "ml-100k"
DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]


def simplex_grid(k: int = 3, resolution: int = 5) -> list[tuple[float, ...]]:
    """Generate uniform grid on k-simplex with given resolution.

    For k=3, resolution=5: 21 points (vertices, edges, interior).
    Each point is a tuple summing to 1.0 with non-negative entries.
    """
    if k == 1:
        return [(1.0,)]
    if resolution == 0:
        return [(0.0,) * (k - 1) + (1.0,)]
    points = []
    for i in range(resolution + 1):
        w_first = i / resolution
        remaining = resolution - i
        for sub in simplex_grid(k - 1, remaining):
            # Scale sub-weights by (1 - w_first)
            scaled = tuple(round(s * (1 - w_first), 6) for s in sub)
            points.append((round(w_first, 6), *scaled))
    return points


def generate_scalarized_preferences(
    stakeholder_prefs: dict[str, tuple[np.ndarray, np.ndarray]],
    mixing: dict[str, float],
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate mixed preference pairs by sampling from stakeholders."""
    rng = np.random.default_rng(seed)
    names = sorted(mixing.keys())
    probs = np.array([mixing[n] for n in names])

    first_key = names[0]
    feature_dim = stakeholder_prefs[first_key][0].shape[1]
    pref = np.zeros((n_pairs, feature_dim), dtype=np.float32)
    rej = np.zeros((n_pairs, feature_dim), dtype=np.float32)

    for i in range(n_pairs):
        chosen = rng.choice(len(names), p=probs)
        s_name = names[chosen]
        s_pref, s_rej = stakeholder_prefs[s_name]
        idx = rng.integers(0, len(s_pref))
        pref[i] = s_pref[idx]
        rej[i] = s_rej[idx]

    return pref, rej


@pytest.fixture(scope="module")
def dataset():
    if not DATA_DIR.exists():
        pytest.skip("MovieLens data not available")
    return MovieLensDataset(str(DATA_DIR))


@pytest.fixture(scope="module")
def setup(dataset):
    configs = build_stakeholder_configs(dataset)
    pool, genres = generate_movielens_content_pool(dataset, min_ratings=5, seed=42)
    # Pre-generate stakeholder preferences
    stakeholder_prefs = {}
    for name in ["user", "platform", "diversity"]:
        p, r = generate_movielens_preferences(pool, configs[name], n_pairs=2000, seed=42)
        stakeholder_prefs[name] = (p, r)
    return {
        "configs": configs,
        "pool": pool,
        "genres": genres,
        "stakeholder_prefs": stakeholder_prefs,
    }


class TestSimplexGrid:

    def test_sums_to_one(self):
        grid = simplex_grid(k=3, resolution=5)
        for point in grid:
            assert abs(sum(point) - 1.0) < 1e-6, f"Point {point} doesn't sum to 1"

    def test_correct_count(self):
        # C(n+k-1, k-1) = C(5+2, 2) = 21 for k=3, n=5
        grid = simplex_grid(k=3, resolution=5)
        assert len(grid) == 21

    def test_includes_vertices(self):
        grid = simplex_grid(k=3, resolution=5)
        assert (1.0, 0.0, 0.0) in grid
        assert (0.0, 1.0, 0.0) in grid
        assert (0.0, 0.0, 1.0) in grid

    def test_all_nonnegative(self):
        grid = simplex_grid(k=3, resolution=5)
        for point in grid:
            assert all(w >= 0 for w in point), f"Negative weight in {point}"


class TestScalarizedPreferences:

    def test_shape(self, setup):
        prefs = setup["stakeholder_prefs"]
        mixing = {"user": 0.4, "platform": 0.4, "diversity": 0.2}
        pref, rej = generate_scalarized_preferences(prefs, mixing, 500, seed=42)
        assert pref.shape == (500, 19)
        assert rej.shape == (500, 19)

    def test_deterministic(self, setup):
        prefs = setup["stakeholder_prefs"]
        mixing = {"user": 0.5, "platform": 0.3, "diversity": 0.2}
        p1, r1 = generate_scalarized_preferences(prefs, mixing, 100, seed=42)
        p2, r2 = generate_scalarized_preferences(prefs, mixing, 100, seed=42)
        np.testing.assert_array_equal(p1, p2)


class TestScalarizedTraining:

    def test_converges(self, setup):
        """BT on mixed preferences should converge."""
        prefs = setup["stakeholder_prefs"]
        mixing = {"user": 0.33, "platform": 0.34, "diversity": 0.33}
        pref, rej = generate_scalarized_preferences(prefs, mixing, 2000, seed=42)
        tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=42)
        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=StakeholderType.USER,
            learning_rate=0.01, num_epochs=50, batch_size=64,
        )
        model = train_with_loss(config, tp, tr, verbose=False,
                                eval_probs_preferred=ep, eval_probs_rejected=er)
        assert model.eval_accuracy > 0.65, f"Scalarized eval acc {model.eval_accuracy:.1%} too low"


class TestFrontierComparison:

    def test_per_stakeholder_produces_frontier(self, setup):
        """Per-stakeholder approach should produce a valid frontier."""
        configs = setup["configs"]
        pool = setup["pool"]
        genres = setup["genres"]
        base_probs = pool[np.newaxis, :, :]
        scorer = (configs["user"] + configs["platform"] + configs["diversity"]) / 3.0

        frontier = compute_scorer_eval_frontier(
            base_probs, genres, scorer, configs, DIVERSITY_WEIGHTS, top_k=10,
        )
        assert len(frontier) == 21
        pareto = extract_pareto_front_nd(
            frontier, ["user_utility", "platform_utility", "diversity_utility"]
        )
        assert len(pareto) >= 1, "No Pareto-optimal points"

    def test_composition_sweep_more_pareto_than_naive(self, setup):
        """Composition sweep should find more Pareto points than single mean scorer."""
        configs = setup["configs"]
        pool = setup["pool"]
        genres = setup["genres"]
        base_probs = pool[np.newaxis, :, :]
        dims = ["user_utility", "platform_utility", "diversity_utility"]

        # Train 3 BT models
        per_stk_weights = {}
        for name in ["user", "platform", "diversity"]:
            pref, rej = setup["stakeholder_prefs"][name]
            tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed=42)
            cfg = LossConfig(loss_type=LossType.BRADLEY_TERRY, stakeholder=StakeholderType.USER,
                             learning_rate=0.01, num_epochs=50, batch_size=64)
            model = train_with_loss(cfg, tp, tr, verbose=False,
                                    eval_probs_preferred=ep, eval_probs_rejected=er)
            per_stk_weights[name] = model.weights

        # Naive: single mean scorer
        mean_scorer = np.mean(list(per_stk_weights.values()), axis=0)
        naive_frontier = compute_scorer_eval_frontier(
            base_probs, genres, mean_scorer, configs, DIVERSITY_WEIGHTS, top_k=10,
        )
        naive_pareto = extract_pareto_front_nd(naive_frontier, dims)

        # Composition sweep: 3 compositions × 21 δ
        grid = simplex_grid(k=3, resolution=3)  # small grid for speed
        comp_points = []
        for mixing_tuple in grid:
            names = sorted(per_stk_weights.keys())
            mixing = {s: w for s, w in zip(names, mixing_tuple)}
            if max(mixing.values()) > 0.99:
                continue
            composed = sum(mixing[s] * per_stk_weights[s] for s in names)
            frontier = compute_scorer_eval_frontier(
                base_probs, genres, composed, configs, DIVERSITY_WEIGHTS, top_k=10,
            )
            comp_points.extend(frontier)

        comp_pareto = extract_pareto_front_nd(comp_points, dims)

        assert len(comp_pareto) >= len(naive_pareto), (
            f"Composition ({len(comp_pareto)}) should find >= Pareto points than naive ({len(naive_pareto)})"
        )
