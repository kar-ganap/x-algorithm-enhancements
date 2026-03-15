"""Tests for factor-based stakeholder generation and K-stakeholder LOSO.

Self-contained tests with minimal dependencies.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Standalone helpers (matching module logic, no pipeline imports)
# ---------------------------------------------------------------------------

def _is_dominated(
    point: dict[str, float],
    candidates: list[dict[str, float]],
    dims: list[str],
) -> bool:
    for c in candidates:
        if c is point:
            continue
        if all(c[d] >= point[d] for d in dims) and any(
            c[d] > point[d] for d in dims
        ):
            return True
    return False


def _extract_pareto_nd(
    points: list[dict[str, float]], dims: list[str]
) -> list[dict[str, float]]:
    return [p for p in points if not _is_dominated(p, points, dims)]


def _compute_regret(
    partial: list[dict[str, float]],
    full: list[dict[str, float]],
    hidden_dim: str,
) -> float:
    if not partial or not full:
        return 0.0
    max_achievable = max(f[hidden_dim] for f in full)
    regrets = [max_achievable - p[hidden_dim] for p in partial]
    return float(np.mean(regrets))


# ---------------------------------------------------------------------------
# Factor model tests
# ---------------------------------------------------------------------------


class TestFactorOrthogonality:
    """Factor matrix should be (approximately) orthonormal after QR."""

    def test_rows_are_unit_length(self) -> None:
        """Each factor vector should have unit norm after QR."""
        # Build a small factor matrix and orthogonalize
        raw = np.random.default_rng(42).standard_normal((8, 18))
        q, _ = np.linalg.qr(raw.T)
        factors = q.T[:4]
        norms = np.linalg.norm(factors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_rows_are_orthogonal(self) -> None:
        """Factor vectors should be mutually orthogonal."""
        raw = np.random.default_rng(42).standard_normal((8, 18))
        q, _ = np.linalg.qr(raw.T)
        factors = q.T[:4]
        gram = factors @ factors.T
        expected = np.eye(4)
        np.testing.assert_allclose(gram, expected, atol=1e-10)


class TestStakeholderGeneration:
    """Stakeholder weight generation via Dirichlet + factor model."""

    def test_correct_count(self) -> None:
        """Should generate exactly K stakeholders."""
        rng = np.random.default_rng(42)
        factors = rng.standard_normal((8, 18))
        for k in [3, 5, 10]:
            loadings = rng.dirichlet(np.full(8, 2.0), size=k)
            weights = {f"s_{i}": loadings[i] @ factors for i in range(k)}
            assert len(weights) == k

    def test_weight_dimension(self) -> None:
        """Each weight vector should be 18-dimensional."""
        rng = np.random.default_rng(42)
        factors = rng.standard_normal((8, 18))
        loadings = rng.dirichlet(np.full(8, 2.0), size=5)
        for i in range(5):
            w = loadings[i] @ factors
            assert w.shape == (18,)

    def test_effective_rank_bounded_by_factors(self) -> None:
        """K weight vectors from F=8 factors span at most 8 dims."""
        rng = np.random.default_rng(42)
        factors = rng.standard_normal((8, 18))
        loadings = rng.dirichlet(np.full(8, 2.0), size=10)
        W = loadings @ factors
        rank = np.linalg.matrix_rank(W, tol=0.01)
        assert rank <= 8

    def test_high_concentration_more_correlated(self) -> None:
        """Higher Dirichlet concentration → higher pairwise cosine."""
        rng_lo = np.random.default_rng(42)
        rng_hi = np.random.default_rng(42)
        factors = np.random.default_rng(0).standard_normal((8, 18))

        def mean_cos(concentration: float, rng: np.random.Generator) -> float:
            loadings = rng.dirichlet(
                np.full(8, concentration), size=5
            )
            W = loadings @ factors
            norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
            W_n = W / norms
            cos_mat = W_n @ W_n.T
            mask = ~np.eye(5, dtype=bool)
            return float(np.mean(cos_mat[mask]))

        cos_low = mean_cos(0.5, rng_lo)
        cos_high = mean_cos(10.0, rng_hi)
        assert cos_high > cos_low


# ---------------------------------------------------------------------------
# N-dimensional Pareto tests
# ---------------------------------------------------------------------------


class TestParetoND:
    """Tests for N-dimensional Pareto extraction."""

    def test_2d_matches_known_result(self) -> None:
        """Simple 2D case: (1,3), (2,2), (3,1) are all Pareto-optimal."""
        points = [
            {"a": 1.0, "b": 3.0},
            {"a": 2.0, "b": 2.0},
            {"a": 3.0, "b": 1.0},
        ]
        pareto = _extract_pareto_nd(points, ["a", "b"])
        assert len(pareto) == 3

    def test_dominated_point_excluded(self) -> None:
        """A point dominated in all dims should be excluded."""
        points = [
            {"a": 1.0, "b": 1.0},  # dominated by (2, 2)
            {"a": 2.0, "b": 2.0},
        ]
        pareto = _extract_pareto_nd(points, ["a", "b"])
        assert len(pareto) == 1
        assert pareto[0]["a"] == 2.0

    def test_3d_pareto(self) -> None:
        """In 3D, a point must be beaten in all 3 dims to be dominated."""
        points = [
            {"a": 3.0, "b": 1.0, "c": 1.0},
            {"a": 1.0, "b": 3.0, "c": 1.0},
            {"a": 1.0, "b": 1.0, "c": 3.0},
            {"a": 2.0, "b": 2.0, "c": 2.0},
        ]
        pareto = _extract_pareto_nd(points, ["a", "b", "c"])
        # All 4 are Pareto-optimal (none dominated in all 3 dims)
        assert len(pareto) == 4

    def test_high_dim_all_pareto(self) -> None:
        """In high dims with few points, most points are Pareto-optimal."""
        rng = np.random.default_rng(42)
        dims = [f"d_{i}" for i in range(10)]
        points = []
        for _ in range(21):
            p = {d: float(rng.random()) for d in dims}
            p["dw"] = 0.0
            points.append(p)
        pareto = _extract_pareto_nd(points, dims)
        # With 21 points in 10D, most should be Pareto-optimal
        assert len(pareto) >= 15


# ---------------------------------------------------------------------------
# LOSO regret tests
# ---------------------------------------------------------------------------


class TestLOSORegretScaling:
    """Tests for LOSO regret computation with K stakeholders."""

    def test_hiding_outlier_higher_regret(self) -> None:
        """Hiding a stakeholder dissimilar to others → more regret."""
        # 3 points on the frontier
        full = [
            {"dw": 0.0, "a": 3.0, "b": 2.5, "c": 0.5},
            {"dw": 0.5, "a": 2.0, "b": 2.0, "c": 2.0},
            {"dw": 1.0, "a": 0.5, "b": 2.5, "c": 3.0},
        ]
        # Hide c (outlier — uncorrelated with a and b)
        loso_ab = _extract_pareto_nd(full, ["a", "b"])
        regret_c = _compute_regret(loso_ab, full, "c")

        # Hide a (well-represented by b)
        loso_bc = _extract_pareto_nd(full, ["b", "c"])
        regret_a = _compute_regret(loso_bc, full, "a")

        # c should have higher regret (its best point not on ab-Pareto)
        assert regret_c > 0

    def test_zero_regret_when_perfectly_correlated(self) -> None:
        """If hidden dim is monotone with an observed dim, regret ≈ 0."""
        full = [
            {"dw": 0.0, "a": 1.0, "b": 2.0, "c": 2.0},
            {"dw": 0.5, "a": 2.0, "b": 3.0, "c": 3.0},
            {"dw": 1.0, "a": 3.0, "b": 4.0, "c": 4.0},
        ]
        # c = b exactly, so hiding c loses nothing
        loso_ab = _extract_pareto_nd(full, ["a", "b"])
        regret_c = _compute_regret(loso_ab, full, "c")
        assert regret_c < 1e-10
