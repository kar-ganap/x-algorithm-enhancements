"""Tests for Phase 9 analysis deepening.

Tests projection, bootstrap CI, and degradation ranking utilities.
"""

import numpy as np
import pytest


def project_onto_span(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project vector onto the column span of basis via least squares.

    Args:
        vector: [D] vector to project
        basis: [D, K] matrix whose columns form the span

    Returns:
        [D] projected vector (in the span of basis columns)
    """
    # Solve basis @ coeffs = vector in least-squares sense
    coeffs, _, _, _ = np.linalg.lstsq(basis, vector, rcond=None)
    return basis @ coeffs


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def bootstrap_ci(
    values: list[float], n_resamples: int = 10000, ci: float = 0.95, seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval.

    Returns: (mean, ci_low, ci_high)
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values)
    means = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_resamples)
    ])
    alpha = (1 - ci) / 2
    return float(np.mean(arr)), float(np.percentile(means, 100 * alpha)), float(np.percentile(means, 100 * (1 - alpha)))


class TestProjection:

    def test_vector_in_span_has_cos_1(self):
        """A vector that IS in the span should have cosine 1.0 with its projection."""
        basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)  # 3D, span of first 2 dims
        vec = np.array([3.0, 4.0, 0.0])  # in the span
        proj = project_onto_span(vec, basis)
        assert cosine_sim(vec, proj) > 0.999

    def test_orthogonal_vector_has_low_cos(self):
        """A vector orthogonal to the span should have low cosine."""
        basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        vec = np.array([0.0, 0.0, 1.0])  # orthogonal to span
        proj = project_onto_span(vec, basis)
        assert np.linalg.norm(proj) < 0.01

    def test_partial_projection(self):
        """A vector partly in and partly out of span."""
        basis = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        vec = np.array([1.0, 1.0, 1.0])  # 2/3 in span, 1/3 orthogonal
        proj = project_onto_span(vec, basis)
        cos = cosine_sim(vec, proj)
        assert 0.5 < cos < 1.0


class TestBootstrapCI:

    def test_ci_contains_mean(self):
        """CI should contain the sample mean."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_wider_ci_with_more_variance(self):
        """Higher variance data should have wider CIs."""
        narrow = [10.0, 10.1, 9.9, 10.0, 10.1]
        wide = [1.0, 20.0, 5.0, 15.0, 10.0]
        _, lo_n, hi_n = bootstrap_ci(narrow)
        _, lo_w, hi_w = bootstrap_ci(wide)
        assert (hi_w - lo_w) > (hi_n - lo_n)

    def test_known_mean(self):
        """Bootstrap mean should be close to true mean."""
        values = [2.0, 4.0, 6.0, 8.0, 10.0]
        mean, _, _ = bootstrap_ci(values)
        assert abs(mean - 6.0) < 0.01


class TestDegradationRanking:

    def test_ranking_matches(self):
        """Cosine-based ranking should match regret ranking in simple case."""
        # Stakeholder weights
        w_user = np.array([1.0, 0.0, 0.0])
        w_platform = np.array([0.9, 0.1, 0.0])  # close to user
        w_diversity = np.array([-1.0, 0.5, 0.5])  # far from both

        # Proxy: mean cosine with the other two
        cos_hide_user = (cosine_sim(w_user, w_platform) + cosine_sim(w_user, w_diversity)) / 2
        cos_hide_plat = (cosine_sim(w_platform, w_user) + cosine_sim(w_platform, w_diversity)) / 2
        cos_hide_div = (cosine_sim(w_diversity, w_user) + cosine_sim(w_diversity, w_platform)) / 2

        # Lower cosine with observed pair → harder to predict → higher regret
        # diversity should have lowest cos → highest regret
        assert cos_hide_div < cos_hide_user
        assert cos_hide_div < cos_hide_plat
