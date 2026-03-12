"""Tests for α-recovery analysis.

Self-contained tests with no pipeline imports.
Verifies: α recovery from weight vectors via neg/pos ratio,
scale invariance, edge cases, and regression-based recovery.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Standalone helpers (matching the script's logic, no imports from scripts/)
# ---------------------------------------------------------------------------


def compute_alpha_ratio(
    weights: np.ndarray,
    pos_indices: list[int],
    neg_indices: list[int],
) -> float:
    """Recover α as -mean(w_neg) / mean(w_pos)."""
    pos_mean = np.mean(weights[pos_indices])
    neg_mean = np.mean(weights[neg_indices])
    if abs(pos_mean) < 1e-10:
        return float("inf")
    return float(-neg_mean / pos_mean)


def compute_alpha_sum_ratio(
    weights: np.ndarray,
    pos_indices: list[int],
    neg_indices: list[int],
) -> float:
    """Recover α as -sum(w_neg) / sum(w_pos)."""
    pos_sum = np.sum(weights[pos_indices])
    neg_sum = np.sum(weights[neg_indices])
    if abs(pos_sum) < 1e-10:
        return float("inf")
    return float(-neg_sum / pos_sum)


def compute_alpha_regression(
    weights: np.ndarray,
    pos_indices: list[int],
    neg_indices: list[int],
) -> float:
    """Recover α via regression: w_i = β_pos if positive, β_neg if negative.

    α_eff = -β_neg / β_pos.
    """
    n = len(pos_indices) + len(neg_indices)
    x_mat = np.zeros((n, 2))  # [is_positive, is_negative]
    y = np.zeros(n)

    for i, idx in enumerate(pos_indices):
        x_mat[i, 0] = 1.0
        y[i] = weights[idx]
    for i, idx in enumerate(neg_indices):
        x_mat[len(pos_indices) + i, 1] = 1.0
        y[len(pos_indices) + i] = weights[idx]

    beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    if abs(beta[0]) < 1e-10:
        return float("inf")
    return float(-beta[1] / beta[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Simple indices for testing: 3 positive, 2 negative
POS = [0, 1, 2]
NEG = [3, 4]


class TestAlphaRatio:
    """Tests for the ratio-based α recovery method."""

    def test_known_alpha_2(self) -> None:
        """Weights [1,1,1,-2,-2] should recover α=2."""
        w = np.array([1.0, 1.0, 1.0, -2.0, -2.0])
        assert abs(compute_alpha_ratio(w, POS, NEG) - 2.0) < 1e-10

    def test_known_alpha_0_5(self) -> None:
        """Weights [2,2,2,-1,-1] should recover α=0.5."""
        w = np.array([2.0, 2.0, 2.0, -1.0, -1.0])
        assert abs(compute_alpha_ratio(w, POS, NEG) - 0.5) < 1e-10

    def test_scale_invariance(self) -> None:
        """Scaling all weights by a constant should not change α_recovered."""
        w = np.array([1.0, 1.0, 1.0, -3.0, -3.0])
        alpha_base = compute_alpha_ratio(w, POS, NEG)
        for scale in [0.1, 2.0, 10.0, 100.0]:
            alpha_scaled = compute_alpha_ratio(w * scale, POS, NEG)
            assert abs(alpha_scaled - alpha_base) < 1e-10

    def test_alpha_zero_all_positive(self) -> None:
        """All positive weights → α ≈ 0 (negative actions have zero weight)."""
        w = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        assert abs(compute_alpha_ratio(w, POS, NEG)) < 1e-10

    def test_monotonicity(self) -> None:
        """Higher true α → higher recovered α."""
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
        recovered = []
        for alpha in alphas:
            w = np.array([1.0, 1.0, 1.0, -alpha, -alpha])
            recovered.append(compute_alpha_ratio(w, POS, NEG))
        for i in range(len(recovered) - 1):
            assert recovered[i] < recovered[i + 1]


class TestAlphaSumRatio:
    """Tests for sum-ratio α recovery."""

    def test_known_alpha(self) -> None:
        """Sum ratio should handle unequal positive weights."""
        # 3 positive actions with different weights, 2 negative
        w = np.array([1.0, 2.0, 3.0, -4.0, -2.0])
        # sum_pos = 6, sum_neg = -6, α = 6/6 = 1.0
        assert abs(compute_alpha_sum_ratio(w, POS, NEG) - 1.0) < 1e-10

    def test_scale_invariance(self) -> None:
        """Sum ratio should also be scale-invariant."""
        w = np.array([1.0, 2.0, 3.0, -4.0, -2.0])
        alpha_base = compute_alpha_sum_ratio(w, POS, NEG)
        alpha_scaled = compute_alpha_sum_ratio(w * 7.3, POS, NEG)
        assert abs(alpha_scaled - alpha_base) < 1e-10

    def test_different_n_pos_neg(self) -> None:
        """Sum ratio accounts for different numbers of positive vs negative."""
        # With 5 positive and 4 negative (like real 18D case)
        pos5 = [0, 1, 2, 3, 4]
        neg4 = [5, 6, 7, 8]
        w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -2.0, -2.0, -2.0])
        # sum_pos = 5, sum_neg = -8, α = 8/5 = 1.6
        assert abs(compute_alpha_sum_ratio(w, pos5, neg4) - 1.6) < 1e-10


class TestAlphaRegression:
    """Tests for regression-based α recovery."""

    def test_uniform_weights(self) -> None:
        """Uniform positive and negative weights: regression = ratio."""
        w = np.array([1.0, 1.0, 1.0, -3.0, -3.0])
        alpha_ratio = compute_alpha_ratio(w, POS, NEG)
        alpha_reg = compute_alpha_regression(w, POS, NEG)
        assert abs(alpha_ratio - alpha_reg) < 1e-10

    def test_nonuniform_weights(self) -> None:
        """Regression averages over non-uniform weights within groups."""
        w = np.array([1.0, 2.0, 3.0, -4.0, -2.0])
        alpha_reg = compute_alpha_regression(w, POS, NEG)
        # β_pos = mean(1,2,3) = 2, β_neg = mean(-4,-2) = -3, α = 3/2 = 1.5
        assert abs(alpha_reg - 1.5) < 1e-10

    def test_scale_invariance(self) -> None:
        """Regression should be scale-invariant."""
        w = np.array([1.0, 2.0, 3.0, -4.0, -2.0])
        alpha_base = compute_alpha_regression(w, POS, NEG)
        alpha_scaled = compute_alpha_regression(w * 5.0, POS, NEG)
        assert abs(alpha_scaled - alpha_base) < 1e-10


class TestEdgeCases:
    """Edge case tests."""

    def test_near_zero_positive_returns_inf(self) -> None:
        """If positive weights are ~0, α should be inf."""
        w = np.array([0.0, 0.0, 0.0, -1.0, -1.0])
        assert compute_alpha_ratio(w, POS, NEG) == float("inf")

    def test_negative_alpha_positive_neg_weights(self) -> None:
        """If negative action weights are positive, α < 0."""
        w = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
        alpha = compute_alpha_ratio(w, POS, NEG)
        assert alpha < 0

    def test_ratio_and_regression_agree_on_uniform(self) -> None:
        """Ratio and regression should agree for uniform weights within groups."""
        w = np.array([2.0, 2.0, 2.0, -6.0, -6.0])
        r = compute_alpha_ratio(w, POS, NEG)
        g = compute_alpha_regression(w, POS, NEG)
        assert abs(r - g) < 1e-10
        assert abs(r - 3.0) < 1e-10

    def test_sum_ratio_has_count_factor(self) -> None:
        """Sum ratio = α * (n_neg/n_pos) due to different group sizes."""
        w = np.array([2.0, 2.0, 2.0, -6.0, -6.0])
        r = compute_alpha_ratio(w, POS, NEG)  # true α = 3.0
        s = compute_alpha_sum_ratio(w, POS, NEG)
        # sum_ratio = α * n_neg/n_pos = 3.0 * 2/3 = 2.0
        assert abs(s - r * len(NEG) / len(POS)) < 1e-10
