"""Tests for rank-order recovery metrics.

Verifies that rank-order correlations (Kendall's tau, Spearman rho)
correctly identify scale-invariant weight recovery — the key property
that makes them appropriate for BT-learned weights where Pearson fails.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from enhancements.data import UserArchetype
from enhancements.reward_modeling.structural_recovery import (
    compute_correlation_matrix,
    compute_rank_correlation_matrix,
)


def _full_gt(target_arch: UserArchetype, weights: np.ndarray) -> dict[UserArchetype, np.ndarray]:
    """Create a full ground-truth dict with one real vector and zeros for the rest."""
    rng = np.random.default_rng(99)
    n = len(weights)
    return {
        arch: (weights.copy() if arch == target_arch else rng.standard_normal(n).astype(np.float32))
        for arch in UserArchetype
    }


FIRST_ARCH = list(UserArchetype)[0]


class TestComputeRankCorrelationMatrix:
    """Tests for compute_rank_correlation_matrix()."""

    def test_perfect_agreement(self) -> None:
        """Identical weights should produce tau=1.0, rho=1.0."""
        w = np.array([3.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        learned = np.array([w])  # [1, 5]
        gt = _full_gt(FIRST_ARCH, w)

        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        assert kendall.shape == (1, len(UserArchetype))
        assert spearman.shape == (1, len(UserArchetype))
        assert kendall[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert spearman[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_scale_invariance(self) -> None:
        """Scale-transformed weights: Pearson = 1.0 (linear), but test rank too.

        This is THE key test. BT can only recover weight direction, not magnitude.
        Multiplying by a positive constant preserves rank order.
        """
        w = np.array([3.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        w_scaled = w * 5.0  # Same rank order, different scale

        learned = np.array([w_scaled])
        gt = _full_gt(FIRST_ARCH, w)

        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        # Rank correlations should be perfect despite scale difference
        assert kendall[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert spearman[0, 0] == pytest.approx(1.0, abs=1e-6)

    def test_nonlinear_monotonic_distortion(self) -> None:
        """Nonlinear monotonic transform preserves rank but breaks Pearson.

        This simulates what BT does: learned weights are a monotonic but
        non-linear function of the true weights.
        """
        w = np.array([3.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        # Cube: preserves ordering but distorts magnitudes
        w_distorted = np.sign(w) * np.abs(w) ** 3

        learned = np.array([w_distorted])
        gt = _full_gt(FIRST_ARCH, w)

        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        # Rank order is preserved by any monotonic transform
        assert kendall[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert spearman[0, 0] == pytest.approx(1.0, abs=1e-6)

        # But Pearson drops because the relationship is non-linear
        pearson = compute_correlation_matrix(learned, gt)
        assert pearson[0, 0] < 0.99  # Notably less than 1.0

    def test_reversed_weights(self) -> None:
        """Reversed weights should produce tau=-1.0, rho=-1.0."""
        w = np.array([3.0, 1.0, 2.0, -1.0, 0.5], dtype=np.float32)
        w_reversed = -w

        learned = np.array([w_reversed])
        gt = _full_gt(FIRST_ARCH, w)

        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        assert kendall[0, 0] == pytest.approx(-1.0, abs=1e-6)
        assert spearman[0, 0] == pytest.approx(-1.0, abs=1e-6)

    def test_output_shape_matches_pearson(self) -> None:
        """Rank correlation matrices should have same shape as Pearson."""
        rng = np.random.default_rng(42)
        K = 3
        n_actions = 18
        learned = rng.standard_normal((K, n_actions)).astype(np.float32)
        gt = {arch: rng.standard_normal(n_actions).astype(np.float32) for arch in UserArchetype}

        pearson = compute_correlation_matrix(learned, gt)
        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        assert kendall.shape == pearson.shape
        assert spearman.shape == pearson.shape

    def test_uncorrelated_weights(self) -> None:
        """Random weights should have correlations near zero."""
        rng = np.random.default_rng(42)
        w_random = rng.standard_normal(100).astype(np.float32)
        learned = rng.standard_normal((1, 100)).astype(np.float32)
        gt = _full_gt(FIRST_ARCH, w_random)

        kendall, spearman = compute_rank_correlation_matrix(learned, gt)

        # With 100 random elements, correlations should be near 0
        assert abs(kendall[0, 0]) < 0.3
        assert abs(spearman[0, 0]) < 0.3
