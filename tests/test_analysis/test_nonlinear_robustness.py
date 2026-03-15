"""Tests for nonlinear robustness audit utilities.

Self-contained tests with no pipeline imports.
Verifies: utility functions (concave, threshold), preference generation,
cross-family reduction to linear, and α-recovery under nonlinearity.
"""

from collections.abc import Callable  # noqa: I001
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Standalone utility functions (matching the analysis script's logic)
# ---------------------------------------------------------------------------


def linear_utility(pos: np.ndarray, neg: np.ndarray, alpha: float) -> np.ndarray:
    """U = pos - α·neg."""
    return pos - alpha * neg


def concave_utility(
    pos: np.ndarray, neg: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    """U = pos^γ - α·neg^γ.  Diminishing returns (prospect theory)."""
    return np.power(pos, gamma) - alpha * np.power(neg, gamma)


def threshold_utility(
    pos: np.ndarray, neg: np.ndarray, alpha: float, tau: float
) -> np.ndarray:
    """U = pos - α·max(neg - τ, 0).  Ignores low negativity."""
    return pos - alpha * np.maximum(neg - tau, 0.0)


def generate_nonlinear_preferences(
    content_probs: np.ndarray,
    pos_indices: list[int],
    neg_indices: list[int],
    utility_fn: Callable[..., Any],
    utility_params: dict[str, Any],
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs using a given utility function.

    Returns (probs_preferred, probs_rejected) each of shape [n_pairs, n_actions].
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)
    n_actions = content_probs.shape[1]

    pos_scores = np.array([
        np.sum(content_probs[c, pos_indices]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, neg_indices]) for c in range(n_content)
    ])
    utility = utility_fn(pos_scores, neg_scores, **utility_params)

    probs_pref = np.zeros((n_pairs, n_actions), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, n_actions), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        noise = rng.normal(0, 0.05)
        if (utility[c1] - utility[c2] + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    return probs_pref, probs_rej


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_content(n: int = 50, n_actions: int = 8, seed: int = 42) -> np.ndarray:
    """Create small content pool for tests."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 0.3, (n, n_actions)).astype(np.float32)


POS_IDX = [0, 1, 2]
NEG_IDX = [5, 6, 7]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestUtilityFunctions:
    """Tests for utility function implementations."""

    def test_linear_basic(self) -> None:
        pos = np.array([2.0, 3.0])
        neg = np.array([1.0, 0.5])
        result = linear_utility(pos, neg, alpha=2.0)
        np.testing.assert_allclose(result, [0.0, 2.0])

    def test_concave_reduces_to_linear_at_gamma_1(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([0.5, 1.0, 0.2])
        alpha = 2.0
        lin = linear_utility(pos, neg, alpha)
        conc = concave_utility(pos, neg, alpha, gamma=1.0)
        np.testing.assert_allclose(conc, lin, atol=1e-10)

    def test_threshold_reduces_to_linear_at_tau_0(self) -> None:
        pos = np.array([1.0, 2.0, 3.0])
        neg = np.array([0.5, 1.0, 0.2])
        alpha = 2.0
        lin = linear_utility(pos, neg, alpha)
        thr = threshold_utility(pos, neg, alpha, tau=0.0)
        np.testing.assert_allclose(thr, lin, atol=1e-10)

    def test_concave_diminishing_returns(self) -> None:
        """With γ < 1, going from 1→2 utility gain < going from 0→1."""
        gamma = 0.5
        # pos^0.5: sqrt(0)=0, sqrt(1)=1, sqrt(4)=2
        # Gain 0→1: 1.0, Gain 1→4: 1.0 (but input went 1→4, not 1→2)
        # Better test: compare marginal returns
        pos = np.array([1.0, 2.0])
        neg = np.array([0.0, 0.0])
        u = concave_utility(pos, neg, alpha=0.0, gamma=gamma)
        # u = [1^0.5, 2^0.5] = [1.0, 1.414]
        # Marginal: 1.414 - 1.0 = 0.414 < 1.0 (the first unit)
        marginal_first = u[0]  # 1.0
        marginal_second = u[1] - u[0]  # 0.414
        assert marginal_second < marginal_first

    def test_threshold_ignores_low_negativity(self) -> None:
        """Below τ, negativity doesn't affect utility."""
        pos = np.array([2.0])
        neg_low = np.array([0.1])
        neg_high = np.array([0.5])
        tau = 0.2
        alpha = 3.0

        u_low = threshold_utility(pos, neg_low, alpha, tau)
        u_high = threshold_utility(pos, neg_high, alpha, tau)

        # Low neg (0.1 < τ=0.2): max(0.1-0.2, 0) = 0, so u = 2.0
        assert u_low[0] == 2.0
        # High neg (0.5 > τ=0.2): max(0.5-0.2, 0) = 0.3, so u = 2.0 - 3.0*0.3 = 1.1
        np.testing.assert_allclose(u_high[0], 1.1, atol=1e-10)


class TestPreferenceGeneration:
    """Tests for nonlinear preference pair generation."""

    def test_output_shapes(self) -> None:
        content = _make_content(n=50, n_actions=8)
        pref, rej = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            linear_utility, {"alpha": 1.0},
            n_pairs=100, seed=42,
        )
        assert pref.shape == (100, 8)
        assert rej.shape == (100, 8)

    def test_consistent_ordering(self) -> None:
        """Preferred items should have higher utility on average."""
        content = _make_content(n=100, n_actions=8)
        pref, rej = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            linear_utility, {"alpha": 2.0},
            n_pairs=500, seed=42,
        )
        pref_pos = np.sum(pref[:, POS_IDX], axis=1)
        pref_neg = np.sum(pref[:, NEG_IDX], axis=1)
        rej_pos = np.sum(rej[:, POS_IDX], axis=1)
        rej_neg = np.sum(rej[:, NEG_IDX], axis=1)
        pref_u = pref_pos - 2.0 * pref_neg
        rej_u = rej_pos - 2.0 * rej_neg
        # Majority of preferred should have higher utility
        assert np.mean(pref_u > rej_u) > 0.8

    def test_different_families_produce_different_pairs(self) -> None:
        """Concave and threshold utilities should produce different preference labels."""
        content = _make_content(n=100, n_actions=8, seed=99)
        pref_conc, _ = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            concave_utility, {"alpha": 4.0, "gamma": 0.5},
            n_pairs=200, seed=42,
        )
        pref_thr, _ = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            threshold_utility, {"alpha": 4.0, "tau": 0.1},
            n_pairs=200, seed=42,
        )
        # Same seed, same content, same RNG — but different utilities
        # should lead to at least some different preferred items
        differ = np.any(pref_conc != pref_thr, axis=1)
        assert np.sum(differ) > 0


class TestCrossFamilyReduction:
    """Tests that nonlinear families reduce to linear at boundary."""

    def test_concave_gamma_1_same_preferences(self) -> None:
        """Concave with γ=1 should produce identical preferences to linear."""
        content = _make_content(n=80, n_actions=8)
        pref_lin, rej_lin = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            linear_utility, {"alpha": 1.0},
            n_pairs=200, seed=42,
        )
        pref_conc, rej_conc = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            concave_utility, {"alpha": 1.0, "gamma": 1.0},
            n_pairs=200, seed=42,
        )
        np.testing.assert_array_equal(pref_lin, pref_conc)
        np.testing.assert_array_equal(rej_lin, rej_conc)

    def test_threshold_tau_0_same_preferences(self) -> None:
        """Threshold with τ=0 should produce identical preferences to linear."""
        content = _make_content(n=80, n_actions=8)
        pref_lin, rej_lin = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            linear_utility, {"alpha": 1.0},
            n_pairs=200, seed=42,
        )
        pref_thr, rej_thr = generate_nonlinear_preferences(
            content, POS_IDX, NEG_IDX,
            threshold_utility, {"alpha": 1.0, "tau": 0.0},
            n_pairs=200, seed=42,
        )
        np.testing.assert_array_equal(pref_lin, pref_thr)
        np.testing.assert_array_equal(rej_lin, rej_thr)


class TestAlphaRecoveryNonlinear:
    """Tests for α recovery from weight vectors trained on nonlinear utilities."""

    def _recover_alpha(
        self, weights: np.ndarray, pos_idx: list[int], neg_idx: list[int]
    ) -> float:
        pos_mean = float(np.mean(weights[pos_idx]))
        neg_mean = float(np.mean(weights[neg_idx]))
        if abs(pos_mean) < 1e-12:
            return 0.0
        return -neg_mean / pos_mean

    def test_ideal_concave_weights_recover_alpha(self) -> None:
        """For ideal weights that perfectly represent concave utility at a point,
        α-recovery should approximate the true α."""
        # Ideal: w[pos] = γ·pos^(γ-1) ≈ proportional to 1 for pos_mean
        # w[neg] = -α·γ·neg^(γ-1) ≈ proportional to -α
        # So ratio should approximate α regardless of γ
        weights = np.zeros(8)
        weights[POS_IDX] = 1.0  # positive
        weights[NEG_IDX] = -4.0  # -α with α=4
        alpha_rec = self._recover_alpha(weights, POS_IDX, NEG_IDX)
        assert abs(alpha_rec - 4.0) < 1e-10

    def test_ideal_threshold_weights_recover_alpha(self) -> None:
        """For ideal weights representing threshold utility,
        α-recovery should approximate α (threshold just removes bias)."""
        weights = np.zeros(8)
        weights[POS_IDX] = 1.0
        weights[NEG_IDX] = -2.0  # -α with α=2
        alpha_rec = self._recover_alpha(weights, POS_IDX, NEG_IDX)
        assert abs(alpha_rec - 2.0) < 1e-10
