"""Tests for α-recovery stress test helpers.

Self-contained tests with no pipeline imports.
Verifies: label flipping, BT-probabilistic preferences, correlated content
generation, and stress sweep building blocks.
"""

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Standalone helpers (matching the stress script's logic, no imports)
# ---------------------------------------------------------------------------


def flip_labels(
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    p_flip: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly swap preferred/rejected with probability p_flip."""
    if p_flip <= 0:
        return probs_pref.copy(), probs_rej.copy()
    rng = np.random.default_rng(seed)
    flip_mask = rng.random(len(probs_pref)) < p_flip
    out_pref = probs_pref.copy()
    out_rej = probs_rej.copy()
    out_pref[flip_mask] = probs_rej[flip_mask]
    out_rej[flip_mask] = probs_pref[flip_mask]
    return out_pref, out_rej


def bt_preference_probability(utility_diff: float, beta: float) -> float:
    """P(prefer item 1) = σ(β · Δu).  β=∞ → hard label."""
    if beta >= 1e6:
        return 1.0 if utility_diff > 0 else (0.0 if utility_diff < 0 else 0.5)
    z = beta * utility_diff
    # Clip to avoid overflow
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + np.exp(-z))


def generate_correlated_content(
    n: int,
    rho: float,
    n_actions: int = 18,
    pos_indices: list[int] | None = None,
    neg_indices: list[int] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate content with controlled correlation between pos and neg scores.

    Uses a bivariate normal copula: generate correlated (z_pos, z_neg),
    then map to uniform(0.05, 0.3) for individual action probabilities,
    scaling positive actions by z_pos and negative by z_neg.
    """
    if pos_indices is None:
        pos_indices = [0, 1, 2]
    if neg_indices is None:
        neg_indices = [3, 4]

    rng = np.random.default_rng(seed)

    # Generate correlated latent factors for pos and neg engagement
    mean = [0.0, 0.0]
    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal(mean, cov, size=n)

    # Map to [0.5, 2.0] multiplier range via CDF
    from scipy.stats import norm

    pos_mult = 0.5 + 1.5 * norm.cdf(z[:, 0])  # [0.5, 2.0]
    neg_mult = 0.5 + 1.5 * norm.cdf(z[:, 1])  # [0.5, 2.0]

    content = np.zeros((n, n_actions), dtype=np.float32)
    for i in range(n):
        base = rng.uniform(0.05, 0.3, n_actions).astype(np.float32)
        for idx in pos_indices:
            base[idx] *= pos_mult[i]
        for idx in neg_indices:
            base[idx] *= neg_mult[i]
        content[i] = np.clip(base, 0, 1)

    return content


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# Simple indices for testing
POS = [0, 1, 2]
NEG = [3, 4]


class TestFlipLabels:
    """Tests for label flipping."""

    def test_p_zero_no_change(self) -> None:
        """p_flip=0 should leave data unchanged."""
        pref = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rej = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
        out_pref, out_rej = flip_labels(pref, rej, 0.0)
        np.testing.assert_array_equal(out_pref, pref)
        np.testing.assert_array_equal(out_rej, rej)

    def test_p_one_flips_all(self) -> None:
        """p_flip=1 should swap all pairs."""
        pref = np.array([[1.0, 2.0], [3.0, 4.0]])
        rej = np.array([[5.0, 6.0], [7.0, 8.0]])
        out_pref, out_rej = flip_labels(pref, rej, 1.0)
        np.testing.assert_array_equal(out_pref, rej)
        np.testing.assert_array_equal(out_rej, pref)

    def test_partial_flip_rate(self) -> None:
        """p_flip=0.5 should flip roughly half the pairs."""
        n = 10000
        pref = np.arange(n).reshape(-1, 1).astype(float)
        rej = (np.arange(n) + n).reshape(-1, 1).astype(float)
        out_pref, _ = flip_labels(pref, rej, 0.5, seed=123)
        flip_count = np.sum(out_pref[:, 0] >= n)
        assert 4500 < flip_count < 5500  # ~50% ± tolerance

    def test_does_not_modify_input(self) -> None:
        """Flipping should not modify original arrays."""
        pref = np.array([[1.0], [2.0]])
        rej = np.array([[3.0], [4.0]])
        pref_orig = pref.copy()
        rej_orig = rej.copy()
        flip_labels(pref, rej, 0.5)
        np.testing.assert_array_equal(pref, pref_orig)
        np.testing.assert_array_equal(rej, rej_orig)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should give same flips."""
        pref = np.random.default_rng(0).random((100, 5))
        rej = np.random.default_rng(1).random((100, 5))
        out1_pref, out1_rej = flip_labels(pref, rej, 0.3, seed=42)
        out2_pref, out2_rej = flip_labels(pref, rej, 0.3, seed=42)
        np.testing.assert_array_equal(out1_pref, out2_pref)
        np.testing.assert_array_equal(out1_rej, out2_rej)


class TestBTPreferenceProb:
    """Tests for BT-probabilistic preference generation."""

    def test_hard_label_positive(self) -> None:
        """Large β with positive diff → P ≈ 1.0."""
        assert bt_preference_probability(1.0, 1e7) == 1.0

    def test_hard_label_negative(self) -> None:
        """Large β with negative diff → P ≈ 0.0."""
        assert bt_preference_probability(-1.0, 1e7) == 0.0

    def test_hard_label_tie(self) -> None:
        """Large β with zero diff → P = 0.5."""
        assert bt_preference_probability(0.0, 1e7) == 0.5

    def test_beta_zero_gives_half(self) -> None:
        """β → 0 should give P ≈ 0.5 for any diff."""
        assert abs(bt_preference_probability(5.0, 0.001) - 0.5) < 0.01
        assert abs(bt_preference_probability(-5.0, 0.001) - 0.5) < 0.01

    def test_symmetry(self) -> None:
        """P(+Δ) + P(-Δ) = 1 for any β."""
        for beta in [0.1, 1.0, 5.0]:
            for diff in [0.5, 1.0, 3.0]:
                p_pos = bt_preference_probability(diff, beta)
                p_neg = bt_preference_probability(-diff, beta)
                assert abs(p_pos + p_neg - 1.0) < 1e-10

    def test_monotone_in_diff(self) -> None:
        """Larger utility diff → higher P(prefer)."""
        beta = 1.0
        probs = [bt_preference_probability(d, beta) for d in [0.1, 0.5, 1.0, 2.0]]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_monotone_in_beta(self) -> None:
        """Higher β sharpens: P moves away from 0.5."""
        diff = 1.0
        probs = [bt_preference_probability(diff, b) for b in [0.1, 0.5, 1.0, 5.0]]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_moderate_beta(self) -> None:
        """β=1, diff=0 → P = 0.5 exactly."""
        assert abs(bt_preference_probability(0.0, 1.0) - 0.5) < 1e-10


class TestCorrelatedContent:
    """Tests for correlated content generation."""

    def test_zero_correlation(self) -> None:
        """ρ=0 should produce near-zero correlation."""
        content = generate_correlated_content(
            5000, rho=0.0, pos_indices=POS, neg_indices=NEG
        )
        pos_scores = np.mean(content[:, POS], axis=1)
        neg_scores = np.mean(content[:, NEG], axis=1)
        r = np.corrcoef(pos_scores, neg_scores)[0, 1]
        assert abs(r) < 0.1

    def test_high_correlation(self) -> None:
        """ρ=0.8 should produce correlation near 0.8."""
        content = generate_correlated_content(
            5000, rho=0.8, pos_indices=POS, neg_indices=NEG
        )
        pos_scores = np.mean(content[:, POS], axis=1)
        neg_scores = np.mean(content[:, NEG], axis=1)
        r = np.corrcoef(pos_scores, neg_scores)[0, 1]
        assert 0.3 < r < 0.95  # copula + clipping compresses correlation

    def test_moderate_correlation(self) -> None:
        """ρ=0.5 should produce positive correlation."""
        content = generate_correlated_content(
            5000, rho=0.5, pos_indices=POS, neg_indices=NEG
        )
        pos_scores = np.mean(content[:, POS], axis=1)
        neg_scores = np.mean(content[:, NEG], axis=1)
        r = np.corrcoef(pos_scores, neg_scores)[0, 1]
        assert 0.2 < r < 0.8

    def test_correct_shape(self) -> None:
        """Output should be (n, n_actions)."""
        content = generate_correlated_content(100, rho=0.0)
        assert content.shape == (100, 18)

    def test_values_in_range(self) -> None:
        """All probabilities should be in [0, 1]."""
        content = generate_correlated_content(1000, rho=0.8)
        assert np.all(content >= 0)
        assert np.all(content <= 1)

    def test_deterministic_with_seed(self) -> None:
        """Same seed should give same content."""
        c1 = generate_correlated_content(100, rho=0.5, seed=42)
        c2 = generate_correlated_content(100, rho=0.5, seed=42)
        np.testing.assert_array_equal(c1, c2)

    def test_monotone_correlation(self) -> None:
        """Higher ρ → higher empirical correlation."""
        correlations = []
        for rho in [0.0, 0.3, 0.6, 0.9]:
            content = generate_correlated_content(
                5000, rho=rho, pos_indices=POS, neg_indices=NEG
            )
            pos_scores = np.mean(content[:, POS], axis=1)
            neg_scores = np.mean(content[:, NEG], axis=1)
            r = np.corrcoef(pos_scores, neg_scores)[0, 1]
            correlations.append(r)
        for i in range(len(correlations) - 1):
            assert correlations[i] < correlations[i + 1]


class TestAlphaRatioRecovery:
    """Sanity tests for ratio recovery under clean conditions."""

    def test_perfect_weights(self) -> None:
        """Known weights should give exact α."""
        w = np.array([1.0, 1.0, 1.0, -3.0, -3.0])
        assert abs(compute_alpha_ratio(w, POS, NEG) - 3.0) < 1e-10

    def test_scale_invariant(self) -> None:
        """Scaling weights shouldn't change recovered α."""
        w = np.array([1.0, 1.0, 1.0, -2.0, -2.0])
        base = compute_alpha_ratio(w, POS, NEG)
        scaled = compute_alpha_ratio(w * 100, POS, NEG)
        assert abs(base - scaled) < 1e-10


class TestStressSweepIntegration:
    """Integration tests for stress sweep building blocks."""

    def test_flip_then_recover(self) -> None:
        """Flipping labels with p=0 should not affect ideal recovery."""
        # Simulate: preferred has positive weights, rejected has negative
        n = 100
        pref = np.zeros((n, 5))
        rej = np.zeros((n, 5))
        # Preferred items: high positive, low negative
        pref[:, :3] = 0.8  # pos indices
        pref[:, 3:] = 0.1  # neg indices
        rej[:, :3] = 0.2
        rej[:, 3:] = 0.6
        out_pref, out_rej = flip_labels(pref, rej, 0.0)
        # Verify structure preserved
        assert np.mean(out_pref[:, :3]) > np.mean(out_rej[:, :3])

    def test_bt_prob_samples_correctly(self) -> None:
        """BT-probabilistic sampling should match expected flip rate."""
        # For β=1, diff=1: P(prefer) = σ(1) ≈ 0.731
        expected = 1.0 / (1.0 + np.exp(-1.0))
        rng = np.random.default_rng(42)
        n = 10000
        preferred_count = sum(
            1 for _ in range(n) if rng.random() < bt_preference_probability(1.0, 1.0)
        )
        observed_rate = preferred_count / n
        assert abs(observed_rate - expected) < 0.02

    def test_spearman_degrades_with_noise(self) -> None:
        """Adding noise to α_recovered should degrade Spearman."""
        alphas_true = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0])
        # Perfect recovery
        alphas_perfect = alphas_true * 1.3  # affine transform, still perfect
        rho_perfect = float(spearmanr(alphas_true, alphas_perfect).statistic)
        assert rho_perfect == 1.0

        # Noisy recovery
        rng = np.random.default_rng(42)
        alphas_noisy = alphas_true * 1.3 + rng.normal(0, 3.0, len(alphas_true))
        rho_noisy = float(spearmanr(alphas_true, alphas_noisy).statistic)
        assert rho_noisy < rho_perfect
