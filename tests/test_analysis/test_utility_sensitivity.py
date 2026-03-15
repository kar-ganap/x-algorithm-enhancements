"""Tests for Direction 2: utility function sensitivity analysis."""

import numpy as np


class TestHausdorffDistance:
    """Tests for Hausdorff distance metric."""

    def test_zero_for_identical(self) -> None:
        dims = ["a", "b"]
        frontier = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        # Inline hausdorff
        vecs = np.array([[p[d] for d in dims] for p in frontier])
        max_dist = 0.0
        for xi in vecs:
            dists = np.linalg.norm(vecs - xi, axis=1)
            max_dist = max(max_dist, float(np.min(dists)))
        assert max_dist == 0.0

    def test_positive_for_shifted(self) -> None:
        v1 = np.array([[0.0, 0.0]])
        v2 = np.array([[1.0, 0.0]])
        d12 = float(np.min(np.linalg.norm(v2 - v1[0], axis=1)))
        d21 = float(np.min(np.linalg.norm(v1 - v2[0], axis=1)))
        hd = max(d12, d21)
        assert abs(hd - 1.0) < 1e-10


class TestAlphaShuffle:
    """Tests for α-preserving within-group shuffle."""

    def test_alpha_preserved(self) -> None:
        """Shuffling within groups preserves the pos/neg ratio."""
        rng = np.random.default_rng(42)
        pos_idx = [0, 1, 2, 3, 4]
        neg_idx = [14, 15, 16, 17]

        w = np.zeros(18)
        w[pos_idx] = [1.0, 0.8, 1.2, 0.9, 0.5]
        w[neg_idx] = [-2.0, -1.5, -2.5, -1.0]

        alpha_before = -np.mean(w[neg_idx]) / np.mean(w[pos_idx])

        # Shuffle within groups
        w_new = w.copy()
        pos_vals = w[pos_idx].copy()
        neg_vals = w[neg_idx].copy()
        rng.shuffle(pos_vals)
        rng.shuffle(neg_vals)
        w_new[pos_idx] = pos_vals
        w_new[neg_idx] = neg_vals

        alpha_after = -np.mean(w_new[neg_idx]) / np.mean(w_new[pos_idx])
        assert abs(alpha_before - alpha_after) < 1e-10

    def test_magnitude_perturbation_preserves_alpha(self) -> None:
        """Gaussian perturbation with renormalization preserves α."""
        pos_idx = [0, 1, 2, 3, 4]
        neg_idx = [14, 15, 16, 17]

        w = np.zeros(18)
        w[pos_idx] = [1.0, 0.8, 1.2, 0.9, 0.5]
        w[neg_idx] = [-2.0, -1.5, -2.5, -1.0]
        alpha_orig = -np.mean(w[neg_idx]) / np.mean(w[pos_idx])

        rng = np.random.default_rng(42)
        w_new = w.copy()
        noise = 1 + rng.normal(0, 0.3, 18)
        w_new *= noise
        # Renormalize
        w_new[pos_idx] *= np.mean(w[pos_idx]) / np.mean(w_new[pos_idx])
        w_new[neg_idx] *= np.mean(w[neg_idx]) / np.mean(w_new[neg_idx])
        alpha_new = -np.mean(w_new[neg_idx]) / np.mean(w_new[pos_idx])

        assert abs(alpha_orig - alpha_new) < 1e-10

    def test_alpha_changed_by_scale(self) -> None:
        """Scaling negative weights changes α."""
        pos_idx = [0, 1, 2]
        neg_idx = [14, 15, 16]

        w = np.zeros(18)
        w[pos_idx] = [1.0, 0.8, 1.2]
        w[neg_idx] = [-2.0, -1.5, -2.5]

        alpha_before = -np.mean(w[neg_idx]) / np.mean(w[pos_idx])

        w_scaled = w.copy()
        w_scaled[neg_idx] *= 2.0
        alpha_after = -np.mean(w_scaled[neg_idx]) / np.mean(w_scaled[pos_idx])

        assert abs(alpha_after - 2.0 * alpha_before) < 1e-10
