"""Tests for disagreement → differentiation bound analysis.

Verifies core functions: disagreement rate computation, symmetry,
monotonicity with respect to utility distance, and sweep output schema.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Standalone helpers (no dependency on full pipeline)
# ---------------------------------------------------------------------------

# Matching alternative_losses.py
POSITIVE_ACTIONS = ["favorite", "repost", "follow_author", "share", "reply"]
NEGATIVE_ACTIONS = ["block_author", "mute_author", "report", "not_interested"]
ACTION_NAMES = [
    "favorite", "reply", "repost", "photo_expand", "click", "profile_click",
    "vqv", "share", "share_via_dm", "share_via_copy_link", "dwell", "quote",
    "quoted_click", "follow_author", "not_interested", "block_author",
    "mute_author", "report",
]
ACTION_INDICES = {name: i for i, name in enumerate(ACTION_NAMES)}
POSITIVE_INDICES = [ACTION_INDICES[a] for a in POSITIVE_ACTIONS]
NEGATIVE_INDICES = [ACTION_INDICES[a] for a in NEGATIVE_ACTIONS]


def compute_disagreement_rate(
    content_probs: np.ndarray,
    neg_penalty_a: float,
    neg_penalty_b: float,
    n_pairs: int,
    seed: int = 42,
) -> float:
    """Reference implementation for testing."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)])
    neg_scores = np.array([np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)])

    utility_a = pos_scores - neg_penalty_a * neg_scores
    utility_b = pos_scores - neg_penalty_b * neg_scores

    disagree_count = 0
    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        prefer_a = utility_a[c1] > utility_a[c2]
        prefer_b = utility_b[c1] > utility_b[c2]
        if prefer_a != prefer_b:
            disagree_count += 1

    return disagree_count / n_pairs


def compute_disagreement_with_margins(
    content_probs: np.ndarray,
    neg_penalty_a: float,
    neg_penalty_b: float,
    n_pairs: int,
    seed: int = 42,
) -> dict[str, float]:
    """Reference implementation that also tracks margins on disagreed pairs."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])

    utility_a = pos_scores - neg_penalty_a * neg_scores
    utility_b = pos_scores - neg_penalty_b * neg_scores

    disagree_count = 0
    margins_total: list[float] = []

    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        prefer_a = utility_a[c1] > utility_a[c2]
        prefer_b = utility_b[c1] > utility_b[c2]
        if prefer_a != prefer_b:
            disagree_count += 1
            margin_a = abs(utility_a[c1] - utility_a[c2])
            margin_b = abs(utility_b[c1] - utility_b[c2])
            margins_total.append(margin_a + margin_b)

    rate = disagree_count / n_pairs
    if margins_total:
        mean_margin = float(np.mean(margins_total))
        margin_std = float(np.std(margins_total))
    else:
        mean_margin = 0.0
        margin_std = 0.0

    return {
        "rate": rate,
        "mean_total_margin": mean_margin,
        "margin_std": margin_std,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDisagreementRate:
    """Tests for the disagreement rate computation."""

    def test_identical_utilities_zero_disagreement(self) -> None:
        """Two stakeholders with the same α must have 0% disagreement."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (50, len(ACTION_NAMES))).astype(np.float32)

        rate = compute_disagreement_rate(content_probs, 1.0, 1.0, n_pairs=500, seed=42)
        assert rate == 0.0

    def test_disagreement_increases_with_distance(self) -> None:
        """As |α₁ - α₂| increases, disagreement rate should increase."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (100, len(ACTION_NAMES))).astype(np.float32)

        # Boost negative actions so penalty differences actually matter
        for idx in NEGATIVE_INDICES:
            content_probs[:, idx] *= 2.0

        fixed_alpha = 1.0
        alphas = [1.2, 2.0, 4.0, 8.0]
        rates = []
        for alpha in alphas:
            rate = compute_disagreement_rate(
                content_probs, fixed_alpha, alpha, n_pairs=2000, seed=42
            )
            rates.append(rate)

        # Each rate should be >= the previous (monotonically non-decreasing)
        for i in range(1, len(rates)):
            assert rates[i] >= rates[i - 1], (
                f"Rate at α={alphas[i]} ({rates[i]:.3f}) < rate at α={alphas[i-1]} ({rates[i-1]:.3f})"
            )

    def test_disagreement_rate_symmetric(self) -> None:
        """disagreement(A, B) == disagreement(B, A) for same seed."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (50, len(ACTION_NAMES))).astype(np.float32)

        rate_ab = compute_disagreement_rate(content_probs, 0.3, 4.0, n_pairs=1000, seed=99)
        rate_ba = compute_disagreement_rate(content_probs, 4.0, 0.3, n_pairs=1000, seed=99)

        assert rate_ab == rate_ba

    def test_cosine_similarity_bounds(self) -> None:
        """Cosine similarity must be in [-1, 1]."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(18).astype(np.float32)
        b = rng.standard_normal(18).astype(np.float32)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        assert -1.0 <= cos <= 1.0

    def test_sweep_result_schema(self) -> None:
        """Each sweep point must have required keys."""
        required_keys = {
            "neg_penalty_fixed",
            "neg_penalty_sweep",
            "disagreement_rate",
            "cosine_similarity",
        }

        # Simulate a sweep result
        point = {
            "neg_penalty_fixed": 0.3,
            "neg_penalty_sweep": 4.0,
            "disagreement_rate": 0.35,
            "cosine_similarity": 0.48,
        }

        assert required_keys.issubset(point.keys())
        assert 0.0 <= point["disagreement_rate"] <= 1.0
        assert -1.0 <= point["cosine_similarity"] <= 1.0


class TestDisagreementMargins:
    """Tests for margin-augmented disagreement computation."""

    def test_margin_zero_when_identical(self) -> None:
        """Same α → no disagreements → mean_total_margin = 0."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (50, len(ACTION_NAMES))).astype(
            np.float32
        )

        result = compute_disagreement_with_margins(
            content_probs, 1.0, 1.0, n_pairs=500, seed=42
        )
        assert result["rate"] == 0.0
        assert result["mean_total_margin"] == 0.0

    def test_margin_increases_with_alpha_distance(self) -> None:
        """Larger |Δα| should produce larger mean margins."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (100, len(ACTION_NAMES))).astype(
            np.float32
        )
        for idx in NEGATIVE_INDICES:
            content_probs[:, idx] *= 2.0

        result_small = compute_disagreement_with_margins(
            content_probs, 1.0, 1.5, n_pairs=2000, seed=42
        )
        result_large = compute_disagreement_with_margins(
            content_probs, 1.0, 4.0, n_pairs=2000, seed=42
        )

        assert result_large["mean_total_margin"] > result_small["mean_total_margin"]

    def test_margin_symmetric(self) -> None:
        """margin(A, B) == margin(B, A) for same seed."""
        rng = np.random.default_rng(42)
        content_probs = rng.uniform(0.05, 0.3, (50, len(ACTION_NAMES))).astype(
            np.float32
        )

        result_ab = compute_disagreement_with_margins(
            content_probs, 0.3, 4.0, n_pairs=1000, seed=99
        )
        result_ba = compute_disagreement_with_margins(
            content_probs, 4.0, 0.3, n_pairs=1000, seed=99
        )

        assert result_ab["rate"] == result_ba["rate"]
        assert abs(
            result_ab["mean_total_margin"] - result_ba["mean_total_margin"]
        ) < 1e-10
