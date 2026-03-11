"""Tests for per-archetype Pareto frontier analysis.

Verifies that slicing utilities by archetype preserves aggregate consistency
and correctly partitions users.
"""

import numpy as np
import pytest

# Archetype mapping matching the data generator
ARCHETYPE_NAMES = {
    0: "sports_fan",
    1: "tech_bro",
    2: "political_L",
    3: "political_R",
    4: "lurker",
    5: "power_user",
}


def group_utilities_by_archetype(
    user_utilities: np.ndarray,
    platform_utilities: np.ndarray,
    user_archetypes: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Import-free reference impl for testing. Real impl lives in the script."""
    result = {}
    for arch_idx, arch_name in ARCHETYPE_NAMES.items():
        mask = user_archetypes == arch_idx
        count = int(mask.sum())
        if count == 0:
            continue
        result[arch_name] = {
            "user_utility": float(np.mean(user_utilities[mask])),
            "platform_utility": float(np.mean(platform_utilities[mask])),
            "count": count,
        }
    return result


class TestGroupUtilitiesByArchetype:
    """Tests for the archetype grouping/slicing logic."""

    def test_weighted_mean_equals_aggregate(self) -> None:
        """Weighted mean of per-archetype utilities must equal np.mean(all)."""
        rng = np.random.default_rng(42)
        N = 600
        user_utils = rng.standard_normal(N).astype(np.float32) + 1.0
        platform_utils = rng.standard_normal(N).astype(np.float32) + 3.0
        archetypes = np.array([i % 6 for i in range(N)], dtype=np.int32)

        grouped = group_utilities_by_archetype(user_utils, platform_utils, archetypes)

        # Weighted reconstruction must match aggregate
        total = sum(g["count"] for g in grouped.values())
        weighted_user = sum(
            g["user_utility"] * g["count"] for g in grouped.values()
        ) / total
        weighted_platform = sum(
            g["platform_utility"] * g["count"] for g in grouped.values()
        ) / total

        assert weighted_user == pytest.approx(float(np.mean(user_utils)), abs=1e-6)
        assert weighted_platform == pytest.approx(float(np.mean(platform_utils)), abs=1e-6)

    def test_partition_is_exhaustive(self) -> None:
        """Every user appears in exactly one group, all 6 archetypes present."""
        N = 120  # 20 per archetype
        user_utils = np.ones(N, dtype=np.float32)
        platform_utils = np.ones(N, dtype=np.float32)
        archetypes = np.array([i % 6 for i in range(N)], dtype=np.int32)

        grouped = group_utilities_by_archetype(user_utils, platform_utils, archetypes)

        assert len(grouped) == 6
        total_count = sum(g["count"] for g in grouped.values())
        assert total_count == N
        for g in grouped.values():
            assert g["count"] == 20

    def test_empty_archetype_handled(self) -> None:
        """If an archetype has 0 users, it should be excluded, not crash."""
        # Only archetypes 0 and 1
        user_utils = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        platform_utils = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        archetypes = np.array([0, 0, 1, 1], dtype=np.int32)

        grouped = group_utilities_by_archetype(user_utils, platform_utils, archetypes)

        assert len(grouped) == 2
        assert "sports_fan" in grouped
        assert "tech_bro" in grouped
        assert grouped["sports_fan"]["user_utility"] == pytest.approx(1.5)
        assert grouped["tech_bro"]["user_utility"] == pytest.approx(3.5)

    def test_different_archetypes_get_different_means(self) -> None:
        """Archetypes with different utility distributions produce different means."""
        # 3 archetypes with distinct utility levels
        user_utils = np.array([1.0, 1.0, 5.0, 5.0, 3.0, 3.0], dtype=np.float32)
        platform_utils = np.ones(6, dtype=np.float32)
        archetypes = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)

        grouped = group_utilities_by_archetype(user_utils, platform_utils, archetypes)

        assert grouped["sports_fan"]["user_utility"] == pytest.approx(1.0)
        assert grouped["tech_bro"]["user_utility"] == pytest.approx(5.0)
        assert grouped["political_L"]["user_utility"] == pytest.approx(3.0)
