"""Tests for partial observation (Direction 3) helpers.

Self-contained tests with no pipeline imports.
Verifies: 2D Pareto extraction, dominance checks, regret computation,
hypervolume indicator, frontier distance, LOSO frontier construction,
and Exp 3 proxy helpers (oracle linear proxy, α-extrapolation,
recovery rate, diversity knob proxy).
"""

from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Standalone helpers (matching the analysis script's logic, no imports)
# ---------------------------------------------------------------------------

UtilityPoint = dict[str, float]


def extract_pareto_front_2d(
    points: list[UtilityPoint],
    dim_x: str,
    dim_y: str,
) -> list[UtilityPoint]:
    """Extract 2D Pareto-optimal subset (maximize both dimensions).

    Returns the full dicts (all keys preserved) but only points that are
    Pareto-optimal when projected to (dim_x, dim_y).
    """
    if not points:
        return []

    # Sort by dim_x descending
    sorted_pts = sorted(points, key=lambda p: p[dim_x], reverse=True)

    pareto: list[UtilityPoint] = []
    max_y = float("-inf")

    for p in sorted_pts:
        if p[dim_y] > max_y:
            pareto.append(p)
            max_y = p[dim_y]
        elif p[dim_y] == max_y and (not pareto or p[dim_x] == pareto[-1][dim_x]):
            # Tie on both dims — include
            pareto.append(p)

    return pareto


def is_dominated(
    point: UtilityPoint,
    candidates: list[UtilityPoint],
    dims: list[str],
) -> bool:
    """Check if point is strictly dominated by any candidate in given dimensions."""
    for c in candidates:
        all_geq = all(c[d] >= point[d] for d in dims)
        any_gt = any(c[d] > point[d] for d in dims)
        if all_geq and any_gt:
            return True
    return False


def compute_dominated_fraction(
    partial_frontier: list[UtilityPoint],
    full_frontier: list[UtilityPoint],
    dims: list[str],
) -> float:
    """Fraction of partial frontier points dominated by full frontier in dims."""
    if not partial_frontier:
        return 0.0
    count = sum(1 for p in partial_frontier if is_dominated(p, full_frontier, dims))
    return count / len(partial_frontier)


def compute_regret(
    partial_frontier: list[UtilityPoint],
    full_frontier: list[UtilityPoint],
    hidden_dim: str,
) -> dict[str, float]:
    """Compute regret for hidden dimension: how much worse is partial vs full.

    Returns max, average, and min regret (all >= 0 by construction).
    """
    if not partial_frontier or not full_frontier:
        return {"max_regret": 0.0, "avg_regret": 0.0, "min_regret": 0.0}

    max_achievable = max(f[hidden_dim] for f in full_frontier)
    regrets = [max_achievable - p[hidden_dim] for p in partial_frontier]

    return {
        "max_regret": float(max(regrets)),
        "avg_regret": float(np.mean(regrets)),
        "min_regret": float(min(regrets)),
    }


def compute_hypervolume_2d(
    points: list[tuple[float, float]],
    ref_point: tuple[float, float],
) -> float:
    """Compute 2D hypervolume indicator (area dominated by Pareto front).

    Points are (x, y) pairs; ref_point is the anti-ideal (lower-left corner).
    Only Pareto-optimal points contribute. Both dimensions are maximized.
    """
    if not points:
        return 0.0

    # Filter to Pareto-optimal points
    sorted_pts = sorted(points, key=lambda p: p[0], reverse=True)
    pareto: list[tuple[float, float]] = []
    max_y = float("-inf")
    for p in sorted_pts:
        if p[1] > max_y:
            pareto.append(p)
            max_y = p[1]

    # Sort Pareto points by x ascending for area sweep
    pareto.sort(key=lambda p: p[0])

    area = 0.0
    prev_x = ref_point[0]
    for x, y in pareto:
        if x > prev_x and y > ref_point[1]:
            area += (x - prev_x) * (y - ref_point[1])
            prev_x = x

    return area


def compute_frontier_distance(
    partial_frontier: list[UtilityPoint],
    full_frontier: list[UtilityPoint],
    dims: list[str],
) -> dict[str, float]:
    """Point-to-nearest-point Euclidean distance from partial to full frontier."""
    if not partial_frontier or not full_frontier:
        return {"mean": 0.0, "max": 0.0, "median": 0.0}

    distances = []
    for p in partial_frontier:
        p_vec = np.array([p[d] for d in dims])
        min_dist = float("inf")
        for f in full_frontier:
            f_vec = np.array([f[d] for d in dims])
            dist = float(np.linalg.norm(p_vec - f_vec))
            min_dist = min(min_dist, dist)
        distances.append(min_dist)

    return {
        "mean": float(np.mean(distances)),
        "max": float(np.max(distances)),
        "median": float(np.median(distances)),
    }


def compute_loso_frontier(
    all_points: list[UtilityPoint],
    observed_dims: list[str],
) -> list[UtilityPoint]:
    """Extract the LOSO frontier: Pareto-optimal in observed dimensions only.

    Returns full dicts (all utility values) for points that are Pareto-optimal
    when projected to the observed dimensions.
    """
    if len(observed_dims) != 2:
        raise ValueError("LOSO requires exactly 2 observed dimensions")
    return extract_pareto_front_2d(all_points, observed_dims[0], observed_dims[1])


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestParetoOptimality2D:
    """Tests for 2D Pareto extraction."""

    def test_single_point_is_pareto(self) -> None:
        points = [{"x": 1.0, "y": 2.0}]
        result = extract_pareto_front_2d(points, "x", "y")
        assert len(result) == 1
        assert result[0]["x"] == 1.0

    def test_dominated_point_excluded(self) -> None:
        points = [
            {"x": 3.0, "y": 3.0},  # dominates the other
            {"x": 1.0, "y": 1.0},  # dominated
        ]
        result = extract_pareto_front_2d(points, "x", "y")
        assert len(result) == 1
        assert result[0]["x"] == 3.0

    def test_non_dominated_coexist(self) -> None:
        points = [
            {"x": 3.0, "y": 1.0},
            {"x": 1.0, "y": 3.0},
        ]
        result = extract_pareto_front_2d(points, "x", "y")
        assert len(result) == 2

    def test_three_point_frontier(self) -> None:
        points = [
            {"x": 5.0, "y": 1.0},
            {"x": 3.0, "y": 3.0},
            {"x": 1.0, "y": 5.0},
            {"x": 2.0, "y": 2.0},  # dominated by (3, 3)
        ]
        result = extract_pareto_front_2d(points, "x", "y")
        assert len(result) == 3
        xs = sorted(r["x"] for r in result)
        assert xs == [1.0, 3.0, 5.0]

    def test_empty_input(self) -> None:
        assert extract_pareto_front_2d([], "x", "y") == []

    def test_preserves_extra_keys(self) -> None:
        points = [{"x": 1.0, "y": 2.0, "z": 99.0}]
        result = extract_pareto_front_2d(points, "x", "y")
        assert result[0]["z"] == 99.0


class TestDominatedFraction:
    """Tests for dominated fraction metric."""

    def test_identical_frontiers_zero(self) -> None:
        frontier = [{"a": 1.0, "b": 2.0}, {"a": 2.0, "b": 1.0}]
        dims = ["a", "b"]
        assert compute_dominated_fraction(frontier, frontier, dims) == 0.0

    def test_fully_dominated(self) -> None:
        partial = [{"a": 1.0, "b": 1.0}]
        full = [{"a": 2.0, "b": 2.0}]
        dims = ["a", "b"]
        assert compute_dominated_fraction(partial, full, dims) == 1.0

    def test_partial_domination(self) -> None:
        partial = [
            {"a": 1.0, "b": 3.0},  # not dominated
            {"a": 1.0, "b": 1.0},  # dominated by (2, 2)
        ]
        full = [{"a": 2.0, "b": 2.0}]
        dims = ["a", "b"]
        assert compute_dominated_fraction(partial, full, dims) == 0.5

    def test_empty_partial(self) -> None:
        assert compute_dominated_fraction([], [{"a": 1.0}], ["a"]) == 0.0

    def test_three_dimensions(self) -> None:
        partial = [{"a": 1.0, "b": 1.0, "c": 1.0}]
        full = [{"a": 2.0, "b": 2.0, "c": 2.0}]
        assert compute_dominated_fraction(partial, full, ["a", "b", "c"]) == 1.0


class TestRegret:
    """Tests for regret computation."""

    def test_zero_regret_when_identical(self) -> None:
        frontier = [{"h": 5.0}, {"h": 3.0}]
        result = compute_regret(frontier, frontier, "h")
        # Best achievable = 5.0, regrets = [0, 2]
        assert result["min_regret"] == 0.0

    def test_positive_regret_when_worse(self) -> None:
        partial = [{"h": 1.0}]
        full = [{"h": 5.0}]
        result = compute_regret(partial, full, "h")
        assert result["max_regret"] == 4.0
        assert result["avg_regret"] == 4.0

    def test_regret_non_negative(self) -> None:
        partial = [{"h": 3.0}, {"h": 1.0}]
        full = [{"h": 4.0}]
        result = compute_regret(partial, full, "h")
        assert result["min_regret"] >= 0.0
        assert result["avg_regret"] >= 0.0
        assert result["max_regret"] >= 0.0

    def test_multiple_full_points(self) -> None:
        partial = [{"h": 2.0}]
        full = [{"h": 3.0}, {"h": 5.0}, {"h": 4.0}]
        result = compute_regret(partial, full, "h")
        assert result["max_regret"] == 3.0  # 5.0 - 2.0

    def test_empty_returns_zero(self) -> None:
        result = compute_regret([], [{"h": 1.0}], "h")
        assert result["max_regret"] == 0.0


class TestHypervolume2D:
    """Tests for 2D hypervolume computation."""

    def test_single_point(self) -> None:
        hv = compute_hypervolume_2d([(3.0, 4.0)], (0.0, 0.0))
        assert hv == 12.0  # 3 * 4

    def test_two_points_on_frontier(self) -> None:
        # Two non-dominated points: (4,1) and (1,3), ref (0,0)
        # Area: (1-0)*(3-0) + (4-1)*(1-0) = 3 + 3 = 6
        hv = compute_hypervolume_2d([(4.0, 1.0), (1.0, 3.0)], (0.0, 0.0))
        assert abs(hv - 6.0) < 1e-10

    def test_dominated_point_ignored(self) -> None:
        # (3,3) dominates (2,2), so (2,2) should not affect hypervolume
        hv_with = compute_hypervolume_2d([(3.0, 3.0), (2.0, 2.0)], (0.0, 0.0))
        hv_without = compute_hypervolume_2d([(3.0, 3.0)], (0.0, 0.0))
        assert abs(hv_with - hv_without) < 1e-10

    def test_hypervolume_increases_with_better_frontier(self) -> None:
        hv1 = compute_hypervolume_2d([(2.0, 2.0)], (0.0, 0.0))
        hv2 = compute_hypervolume_2d([(3.0, 3.0)], (0.0, 0.0))
        assert hv2 > hv1

    def test_empty_points(self) -> None:
        assert compute_hypervolume_2d([], (0.0, 0.0)) == 0.0

    def test_reference_point_matters(self) -> None:
        hv1 = compute_hypervolume_2d([(5.0, 5.0)], (0.0, 0.0))
        hv2 = compute_hypervolume_2d([(5.0, 5.0)], (2.0, 2.0))
        assert hv1 > hv2


class TestFrontierDistance:
    """Tests for frontier distance metric."""

    def test_identical_frontiers_zero(self) -> None:
        frontier = [{"a": 1.0, "b": 2.0}]
        result = compute_frontier_distance(frontier, frontier, ["a", "b"])
        assert result["mean"] == 0.0
        assert result["max"] == 0.0

    def test_shifted_positive_distance(self) -> None:
        partial = [{"a": 0.0, "b": 0.0}]
        full = [{"a": 3.0, "b": 4.0}]
        result = compute_frontier_distance(partial, full, ["a", "b"])
        assert abs(result["mean"] - 5.0) < 1e-10  # 3-4-5 triangle

    def test_nearest_point_used(self) -> None:
        partial = [{"a": 0.0, "b": 0.0}]
        full = [
            {"a": 10.0, "b": 10.0},  # far
            {"a": 1.0, "b": 0.0},  # near (distance = 1)
        ]
        result = compute_frontier_distance(partial, full, ["a", "b"])
        assert abs(result["mean"] - 1.0) < 1e-10

    def test_empty_returns_zero(self) -> None:
        result = compute_frontier_distance([], [{"a": 1.0}], ["a"])
        assert result["mean"] == 0.0


class TestLOSOFrontier:
    """Tests for leave-one-stakeholder-out frontier construction."""

    def test_2d_pareto_includes_more_points(self) -> None:
        """2D Pareto may include points that are dominated in 3D."""
        points = [
            {"u": 5.0, "p": 1.0, "s": 1.0},  # 3D-Pareto
            {"u": 1.0, "p": 5.0, "s": 1.0},  # 3D-Pareto
            {"u": 3.0, "p": 3.0, "s": 0.5},  # 2D-Pareto in (u,p), dominated in 3D
        ]
        loso = compute_loso_frontier(points, ["u", "p"])
        # In (u,p) space: (5,1), (3,3), (1,5) are all Pareto-optimal
        assert len(loso) == 3

    def test_preserves_all_utility_values(self) -> None:
        points = [{"u": 1.0, "p": 2.0, "s": 3.0}]
        loso = compute_loso_frontier(points, ["u", "p"])
        assert loso[0]["s"] == 3.0

    def test_full_3d_pareto_subset_of_2d(self) -> None:
        """Every 3D Pareto point should also be 2D Pareto-optimal."""
        points = [
            {"u": 5.0, "p": 1.0, "s": 3.0},
            {"u": 1.0, "p": 5.0, "s": 2.0},
            {"u": 3.0, "p": 3.0, "s": 5.0},
        ]
        # All three are 3D-Pareto (none dominates another in all 3 dims)
        loso_up = compute_loso_frontier(points, ["u", "p"])
        loso_u_values = {p["u"] for p in loso_up}
        # All original u values should be present
        for p in points:
            assert p["u"] in loso_u_values

    def test_requires_two_observed_dims(self) -> None:
        try:
            compute_loso_frontier([{"a": 1.0}], ["a"])
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestPartialObservationIntegration:
    """Integration tests combining LOSO + metrics."""

    def _make_frontier(self) -> list[UtilityPoint]:
        """Create a realistic-ish frontier with known properties."""
        # Simulate diversity_weight sweep: more diversity → better society,
        # worse platform, user peaks in the middle
        points = []
        for i in range(11):
            dw = i / 10.0
            points.append({
                "diversity_weight": dw,
                "user_utility": 1.0 + 0.15 * (1 - abs(dw - 0.2) * 3),
                "platform_utility": 3.5 - 1.0 * dw,
                "society_utility": -0.5 + 1.0 * dw,
            })
        return points

    def test_full_loso_pipeline(self) -> None:
        """End-to-end: generate frontier, do LOSO, verify metrics are valid."""
        points = self._make_frontier()

        dims_all = ["user_utility", "platform_utility", "society_utility"]

        # Full 3D Pareto
        full_pareto = [
            p for p in points if not is_dominated(p, points, dims_all)
        ]
        assert len(full_pareto) >= 1

        # LOSO: hide society
        loso = compute_loso_frontier(points, ["user_utility", "platform_utility"])
        assert len(loso) >= 1

        # Metrics
        dom_frac = compute_dominated_fraction(loso, full_pareto, dims_all)
        assert 0.0 <= dom_frac <= 1.0

        regret = compute_regret(loso, full_pareto, "society_utility")
        assert regret["max_regret"] >= 0.0
        assert regret["avg_regret"] >= regret["min_regret"]

        dist = compute_frontier_distance(loso, full_pareto, dims_all)
        assert dist["mean"] >= 0.0

    def test_hiding_anticorrelated_gives_more_regret(self) -> None:
        """When hidden dim is anti-correlated with observed, regret is higher."""
        # Society anti-correlated with platform; user varies independently
        points = [
            {"u": 4.0, "p": 5.0, "s": -1.0},
            {"u": 3.0, "p": 3.0, "s": 1.0},
            {"u": 2.0, "p": 1.0, "s": 3.0},
        ]

        # Hide society (anti-correlated with platform)
        # LOSO (u,p) picks (4,5) as best → s = -1.0, max regret = 3 - (-1) = 4
        loso_hide_s = compute_loso_frontier(points, ["u", "p"])
        regret_s = compute_regret(loso_hide_s, points, "s")

        # Hide user (positively correlated with platform)
        # LOSO (p,s) picks all 3 → worst u is 2.0, max regret = 4 - 2 = 2
        loso_hide_u = compute_loso_frontier(points, ["p", "s"])
        regret_u = compute_regret(loso_hide_u, points, "u")

        # Hiding the anti-correlated dimension produces more regret
        assert regret_s["max_regret"] > regret_u["max_regret"]


# ---------------------------------------------------------------------------
# Exp 3: Standalone helpers for proxy experiments
# ---------------------------------------------------------------------------


def compute_oracle_linear_proxy(
    w_obs1: np.ndarray,
    w_obs2: np.ndarray,
    w_hidden: np.ndarray,
) -> dict[str, Any]:
    """Find optimal a, b such that a*w_obs1 + b*w_obs2 ≈ w_hidden (least-squares)."""
    X = np.column_stack([w_obs1, w_obs2])
    coeffs, _, _, _ = np.linalg.lstsq(X, w_hidden, rcond=None)
    w_proxy = X @ coeffs
    cos_sim = float(
        np.dot(w_proxy, w_hidden)
        / (np.linalg.norm(w_proxy) * np.linalg.norm(w_hidden) + 1e-12)
    )
    residual_norm = float(np.linalg.norm(w_hidden - w_proxy))
    return {
        "coefficients": (float(coeffs[0]), float(coeffs[1])),
        "proxy_weights": w_proxy,
        "cosine_sim": cos_sim,
        "residual_norm": residual_norm,
    }


def recover_alpha_from_weights(
    weights: np.ndarray,
    pos_indices: list[int],
    neg_indices: list[int],
) -> float:
    """Recover α as -mean(w_neg) / mean(w_pos)."""
    pos_mean = float(np.mean(weights[pos_indices]))
    neg_mean = float(np.mean(weights[neg_indices]))
    if abs(pos_mean) < 1e-12:
        return 0.0
    return -neg_mean / pos_mean


def synthesize_weights_interpolation(
    w1: np.ndarray,
    alpha1: float,
    w2: np.ndarray,
    alpha2: float,
    alpha_target: float,
) -> np.ndarray:
    """Linear interpolation/extrapolation in weight space."""
    if abs(alpha1 - alpha2) < 1e-12:
        return (w1 + w2) / 2.0
    t = (alpha_target - alpha2) / (alpha1 - alpha2)
    return w2 + t * (w1 - w2)


def synthesize_weights_structural(
    observed_weights: dict[str, np.ndarray],
    alpha_target: float,
    pos_indices: list[int],
    neg_indices: list[int],
) -> np.ndarray:
    """Structural synthesis: mean pos/neg/neutral patterns + target α scaling."""
    all_w = np.stack(list(observed_weights.values()))
    mean_w = np.mean(all_w, axis=0)

    pos_mean = float(np.mean(mean_w[pos_indices]))
    all_indices = set(pos_indices) | set(neg_indices)
    neutral_indices = [i for i in range(len(mean_w)) if i not in all_indices]

    result = np.zeros_like(mean_w)
    result[pos_indices] = pos_mean
    result[neg_indices] = -alpha_target * abs(pos_mean)
    result[neutral_indices] = mean_w[neutral_indices]
    return result


def compute_recovery_rate(
    proxy_utility: float,
    no_proxy_utility: float,
    full_utility: float,
) -> float:
    """Recovery rate = (proxy - no_proxy) / (full - no_proxy)."""
    gap = full_utility - no_proxy_utility
    if abs(gap) < 1e-12:
        return 1.0  # No gap to close
    return (proxy_utility - no_proxy_utility) / gap


def find_best_hidden_utility(
    frontier: list[UtilityPoint],
    hidden_dim: str,
) -> tuple[float, float]:
    """Find the point maximizing hidden utility. Returns (diversity_weight, utility)."""
    if not frontier:
        return (0.0, 0.0)
    best = max(frontier, key=lambda p: p[hidden_dim])
    return (best.get("diversity_weight", 0.0), best[hidden_dim])


# ---------------------------------------------------------------------------
# Exp 3: Test classes
# ---------------------------------------------------------------------------


class TestOracleLinearProxy:
    """Tests for Exp 3a: oracle linear proxy (least-squares)."""

    def test_perfect_recovery_when_in_span(self) -> None:
        """If w_hidden = 0.5*w1 + 0.5*w2, proxy should be exact."""
        w1 = np.array([1.0, 0.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0, 0.0])
        w_hidden = 0.5 * w1 + 0.5 * w2
        result = compute_oracle_linear_proxy(w1, w2, w_hidden)
        assert result["residual_norm"] < 1e-8
        assert result["cosine_sim"] > 0.999

    def test_residual_positive_when_orthogonal_component(self) -> None:
        """If w_hidden has component outside span, residual > 0."""
        w1 = np.array([1.0, 0.0, 0.0])
        w2 = np.array([0.0, 1.0, 0.0])
        w_hidden = np.array([0.5, 0.5, 1.0])  # z component not in span
        result = compute_oracle_linear_proxy(w1, w2, w_hidden)
        assert result["residual_norm"] > 0.5

    def test_cosine_at_least_as_good_as_best_single(self) -> None:
        """Proxy cosine sim >= max of individual cosine sims."""
        w1 = np.array([3.0, 2.0, -1.0, 0.5])
        w2 = np.array([1.0, -1.0, 2.0, 0.5])
        w_hidden = np.array([2.0, 1.0, 0.5, 1.0])
        result = compute_oracle_linear_proxy(w1, w2, w_hidden)
        cos1 = np.dot(w1, w_hidden) / (np.linalg.norm(w1) * np.linalg.norm(w_hidden))
        cos2 = np.dot(w2, w_hidden) / (np.linalg.norm(w2) * np.linalg.norm(w_hidden))
        assert result["cosine_sim"] >= max(cos1, cos2) - 1e-8

    def test_works_with_18_dim_vectors(self) -> None:
        """Real-sized weight vectors."""
        rng = np.random.default_rng(42)
        w1 = rng.normal(0, 1, 18)
        w2 = rng.normal(0, 1, 18)
        w_hidden = rng.normal(0, 1, 18)
        result = compute_oracle_linear_proxy(w1, w2, w_hidden)
        assert -1.0 <= result["cosine_sim"] <= 1.0
        assert result["residual_norm"] >= 0.0
        a, b = result["coefficients"]
        proxy = a * w1 + b * w2
        np.testing.assert_allclose(proxy, result["proxy_weights"], atol=1e-8)


class TestAlphaExtrapolation:
    """Tests for Exp 3c: α-based weight synthesis."""

    def test_interpolation_recovers_endpoints(self) -> None:
        """At alpha=alpha1, returns w1; at alpha=alpha2, returns w2."""
        w1 = np.array([1.0, 2.0, 3.0])
        w2 = np.array([4.0, 5.0, 6.0])
        result_at_1 = synthesize_weights_interpolation(w1, 1.0, w2, 0.3, 1.0)
        result_at_2 = synthesize_weights_interpolation(w1, 1.0, w2, 0.3, 0.3)
        np.testing.assert_allclose(result_at_1, w1, atol=1e-10)
        np.testing.assert_allclose(result_at_2, w2, atol=1e-10)

    def test_extrapolation_beyond_range(self) -> None:
        """At alpha > max, extrapolates linearly."""
        w1 = np.array([2.0, -1.0])
        w2 = np.array([4.0, -0.3])
        # alpha_target=4.0, alpha1=1.0, alpha2=0.3
        # t = (4.0 - 0.3) / (1.0 - 0.3) = 3.7 / 0.7 ≈ 5.286
        result = synthesize_weights_interpolation(w1, 1.0, w2, 0.3, 4.0)
        t = (4.0 - 0.3) / (1.0 - 0.3)
        expected = w2 + t * (w1 - w2)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_structural_correct_sign_pattern(self) -> None:
        """Positive indices positive, negative indices negative (for alpha > 0)."""
        pos_idx = [0, 1]
        neg_idx = [2, 3]
        observed = {
            "a": np.array([2.0, 3.0, -1.0, -2.0, 0.1]),
            "b": np.array([4.0, 3.0, -0.5, -1.0, -0.1]),
        }
        result = synthesize_weights_structural(observed, 4.0, pos_idx, neg_idx)
        assert all(result[i] > 0 for i in pos_idx)
        assert all(result[i] < 0 for i in neg_idx)

    def test_structural_alpha_scaling(self) -> None:
        """Doubling alpha doubles the magnitude of negative weights."""
        pos_idx = [0, 1]
        neg_idx = [2, 3]
        observed = {"a": np.array([2.0, 3.0, -1.0, -2.0, 0.0])}
        r1 = synthesize_weights_structural(observed, 2.0, pos_idx, neg_idx)
        r2 = synthesize_weights_structural(observed, 4.0, pos_idx, neg_idx)
        np.testing.assert_allclose(r2[neg_idx], 2.0 * r1[neg_idx], atol=1e-10)

    def test_interpolation_vs_structural_agree_for_ideal(self) -> None:
        """When weights perfectly follow U=pos-α·neg, both approaches should
        produce vectors with the correct α ratio for the target."""
        pos_idx = [0, 1]
        neg_idx = [2, 3]
        # Ideal weights: w[pos] = 1.0, w[neg] = -α
        w1 = np.array([1.0, 1.0, -1.0, -1.0])  # α=1.0
        w2 = np.array([1.0, 1.0, -0.3, -0.3])   # α=0.3

        alpha_target = 4.0
        interp = synthesize_weights_interpolation(w1, 1.0, w2, 0.3, alpha_target)
        struct = synthesize_weights_structural(
            {"a": w1, "b": w2}, alpha_target, pos_idx, neg_idx,
        )
        # Both should have neg ≈ -4.0 * pos_mean
        interp_ratio = -np.mean(interp[neg_idx]) / np.mean(interp[pos_idx])
        struct_ratio = -np.mean(struct[neg_idx]) / np.mean(struct[pos_idx])
        assert abs(struct_ratio - alpha_target) < 1e-10
        assert abs(interp_ratio - alpha_target) < 1e-10


class TestRecoveryRate:
    """Tests for Exp 3d: recovery rate metric."""

    def test_full_recovery_returns_1(self) -> None:
        assert compute_recovery_rate(10.0, 5.0, 10.0) == 1.0

    def test_zero_recovery_returns_0(self) -> None:
        assert compute_recovery_rate(5.0, 5.0, 10.0) == 0.0

    def test_partial_recovery_midpoint(self) -> None:
        assert abs(compute_recovery_rate(7.5, 5.0, 10.0) - 0.5) < 1e-10

    def test_over_recovery_greater_than_1(self) -> None:
        """Proxy does better than full 3D optimum (shouldn't happen, but handle)."""
        rate = compute_recovery_rate(12.0, 5.0, 10.0)
        assert rate > 1.0

    def test_no_gap_returns_1(self) -> None:
        """When full == no_proxy (no gap), recovery is 1.0 by convention."""
        assert compute_recovery_rate(5.0, 5.0, 5.0) == 1.0


class TestDiversityKnobProxy:
    """Tests for Exp 3b: diversity knob as society proxy."""

    def test_finds_point_with_max_hidden_utility(self) -> None:
        frontier: list[UtilityPoint] = [
            {"diversity_weight": 0.0, "society_utility": -1.0},
            {"diversity_weight": 0.5, "society_utility": 0.5},
            {"diversity_weight": 1.0, "society_utility": 2.0},
        ]
        dw, util = find_best_hidden_utility(frontier, "society_utility")
        assert dw == 1.0
        assert util == 2.0

    def test_works_with_single_point(self) -> None:
        frontier: list[UtilityPoint] = [
            {"diversity_weight": 0.3, "society_utility": 1.5},
        ]
        dw, util = find_best_hidden_utility(frontier, "society_utility")
        assert dw == 0.3
        assert util == 1.5

    def test_returns_correct_diversity_weight(self) -> None:
        """When max hidden utility is not at extreme dw."""
        frontier: list[UtilityPoint] = [
            {"diversity_weight": 0.0, "society_utility": 0.0},
            {"diversity_weight": 0.4, "society_utility": 3.0},
            {"diversity_weight": 0.8, "society_utility": 2.0},
        ]
        dw, util = find_best_hidden_utility(frontier, "society_utility")
        assert dw == 0.4
        assert util == 3.0
