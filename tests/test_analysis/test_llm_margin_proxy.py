"""Tests for LLM confidence as margin proxy experiment (v2).

Self-contained tests with no API calls and no pipeline imports.
Verifies: content score computation, prompt construction, response parsing,
disagreement computation, regression fitting, and go/no-go evaluation.

v2: natural language stakeholder descriptions + engagement/negativity scores.
The LLM applies its own implicit utility function — no formula provided.
"""

import json
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Standalone helpers (matching the script's logic, no imports from scripts/)
# ---------------------------------------------------------------------------

POSITIVE_ACTIONS = ["favorite", "repost", "follow_author", "share", "reply"]
NEGATIVE_ACTIONS = ["block_author", "mute_author", "report", "not_interested"]


def compute_scaled_scores(
    content_probs: np.ndarray,
    positive_indices: list[int],
    negative_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute engagement and negativity scores scaled to 0-100."""
    n_content = len(content_probs)
    pos_raw = np.array([
        np.sum(content_probs[c, positive_indices]) for c in range(n_content)
    ])
    neg_raw = np.array([
        np.sum(content_probs[c, negative_indices]) for c in range(n_content)
    ])
    pos_max = pos_raw.max() if pos_raw.max() > 0 else 1.0
    neg_max = neg_raw.max() if neg_raw.max() > 0 else 1.0
    engagement = (pos_raw / pos_max * 100).astype(int)
    negativity = (neg_raw / neg_max * 100).astype(int)
    return engagement, negativity


def stakeholder_description(alpha: float) -> str:
    """Map alpha to a natural language stakeholder perspective."""
    if alpha <= 0.3:
        return (
            "You are an engagement-focused platform operator. "
            "User activity is what matters most to you. "
            "Some negative reactions are normal and acceptable — "
            "they barely factor into your decisions."
        )
    if alpha <= 0.8:
        return (
            "You are a platform manager who values engagement but keeps "
            "an eye on negative reactions. Both matter, but engagement "
            "is somewhat more important to you."
        )
    if alpha <= 1.5:
        return (
            "You balance engagement with user safety equally. "
            "Negative reactions concern you just as much as "
            "positive engagement."
        )
    if alpha <= 3.0:
        return (
            "You prioritize a healthy platform. Negative reactions "
            "weigh heavily in your decisions — more than raw "
            "engagement numbers. You'd rather show less engaging "
            "content than risk harmful reactions."
        )
    return (
        "You are a safety-first content moderator. Reducing harmful "
        "content is your top priority. High negativity is a dealbreaker "
        "regardless of how much engagement a post gets. "
        "You strongly prefer low-negativity content."
    )


def build_batch_prompt(
    engagement: np.ndarray,
    negativity: np.ndarray,
    pair_indices: list[tuple[int, int]],
    alpha: float,
    rng: np.random.Generator,
) -> tuple[str, str, list[bool]]:
    """Build system and user prompts for a batch of content pairs."""
    perspective = stakeholder_description(alpha)

    system = (
        "You evaluate content pairs for a social media recommendation system.\n"
        "Each post has two scores (0-100):\n"
        "  - engagement: how much users interact positively "
        "(likes, shares, replies, follows)\n"
        "  - negativity: how much users react negatively "
        "(blocks, mutes, reports, not-interested)\n\n"
        f"Your perspective: {perspective}\n\n"
        "For each pair, pick the post YOU would rather recommend "
        "given your perspective.\n"
        "Then rate your confidence: 0.50 = coin flip, 1.00 = certain.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"pairs": [{"pair_id": 1, "preferred": "A", "confidence": 0.85},'
        " ...]}"
    )

    user_parts = [f"Evaluate these {len(pair_indices)} content pairs:\n"]
    swap_flags = []
    for i, (c1, c2) in enumerate(pair_indices, 1):
        swap = bool(rng.integers(2))
        swap_flags.append(swap)
        if swap:
            a_idx, b_idx = c2, c1
        else:
            a_idx, b_idx = c1, c2

        user_parts.append(
            f"Pair {i}: "
            f"Post A: engagement={engagement[a_idx]} negativity={negativity[a_idx]} | "
            f"Post B: engagement={engagement[b_idx]} negativity={negativity[b_idx]}"
        )

    return system, "\n".join(user_parts), swap_flags


def parse_llm_response(
    response_text: str, expected_count: int = 0  # noqa: ARG001
) -> list[dict[str, object]]:
    """Parse and validate LLM JSON response."""
    data = json.loads(response_text)
    results = []
    for entry in data["pairs"]:
        conf = float(entry["confidence"])
        conf = max(0.50, min(1.00, conf))  # clamp
        preferred = entry["preferred"]
        if preferred not in ("A", "B"):
            raise ValueError(f"Invalid preferred: {preferred}")
        results.append(
            {
                "pair_id": int(entry["pair_id"]),
                "preferred": preferred,
                "confidence": conf,
            }
        )
    return results


def compute_llm_disagreement_and_confidence(
    responses_a: list[dict[str, object]],
    responses_b: list[dict[str, object]],
) -> dict[str, float]:
    """Compute disagreement rate and mean confidence on disagreed pairs."""
    assert len(responses_a) == len(responses_b)
    n = len(responses_a)
    if n == 0:
        return {"rate": 0.0, "mean_confidence_sum": 0.0, "n_disagreed": 0}

    disagree_count = 0
    confidence_sums: list[float] = []

    for ra, rb in zip(responses_a, responses_b):
        if ra["preferred"] != rb["preferred"]:
            disagree_count += 1
            conf_a = ra["confidence"]
            conf_b = rb["confidence"]
            assert isinstance(conf_a, (int, float))
            assert isinstance(conf_b, (int, float))
            conf_sum = float(conf_a) + float(conf_b)
            confidence_sums.append(conf_sum)

    rate = disagree_count / n
    mean_conf = float(np.mean(confidence_sums)) if confidence_sums else 0.0

    return {
        "rate": rate,
        "mean_confidence_sum": mean_conf,
        "n_disagreed": disagree_count,
    }


def fit_linear_r2(x: np.ndarray, y: np.ndarray) -> float:
    """Fit y = a + b*x and return R²."""
    a_mat = np.column_stack([np.ones_like(x), x])
    result, _, _, _ = np.linalg.lstsq(a_mat, y, rcond=None)
    y_pred = a_mat @ result
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_two_var_r2(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray
) -> float:
    """Fit y = a + b1*x1 + b2*x2 and return R²."""
    a_mat = np.column_stack([np.ones_like(x1), x1, x2])
    result, _, _, _ = np.linalg.lstsq(a_mat, y, rcond=None)
    y_pred = a_mat @ result
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def evaluate_go_no_go(
    llm_two_var_spearman: float,
    confidence_margin_spearman: float,
    spearman_gap: float,
    disagreement_correlation: float,
) -> dict[str, Any]:
    """Evaluate go/no-go criteria (v2: Spearman-based, no formula_agreement)."""
    criteria: dict[str, Any] = {
        "llm_two_var_spearman": {
            "value": llm_two_var_spearman,
            "threshold": 0.85,
            "type": "MUST",
            "pass": llm_two_var_spearman >= 0.85,
        },
        "confidence_margin_spearman": {
            "value": confidence_margin_spearman,
            "threshold": 0.70,
            "type": "SHOULD",
            "pass": confidence_margin_spearman >= 0.70,
        },
        "spearman_gap": {
            "value": spearman_gap,
            "threshold": 0.10,
            "type": "NICE",
            "pass": spearman_gap <= 0.10,
        },
        "disagreement_correlation": {
            "value": disagreement_correlation,
            "threshold": 0.90,
            "type": "MUST",
            "pass": disagreement_correlation >= 0.90,
        },
    }
    must_pass = all(
        c["pass"] for c in criteria.values() if c["type"] == "MUST"
    )
    should_pass = all(
        c["pass"] for c in criteria.values() if c["type"] == "SHOULD"
    )
    if must_pass and should_pass:
        overall = "GO"
    elif must_pass:
        overall = "CONDITIONAL_GO"
    else:
        overall = "NO_GO"
    return {"criteria": criteria, "overall": overall}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContentScoring:
    """Tests for engagement/negativity score computation."""

    def test_scores_in_0_100_range(self) -> None:
        """All scores should be in [0, 100]."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(9), size=50)
        pos_idx = list(range(5))
        neg_idx = list(range(5, 9))
        eng, neg = compute_scaled_scores(probs, pos_idx, neg_idx)
        assert eng.min() >= 0 and eng.max() <= 100
        assert neg.min() >= 0 and neg.max() <= 100

    def test_max_score_is_100(self) -> None:
        """The content with highest pos/neg sum should have score 100."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(9), size=50)
        pos_idx = list(range(5))
        neg_idx = list(range(5, 9))
        eng, neg = compute_scaled_scores(probs, pos_idx, neg_idx)
        assert eng.max() == 100
        assert neg.max() == 100

    def test_scores_are_integers(self) -> None:
        """Scores should be integer-typed (for clean LLM prompts)."""
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(9), size=10)
        pos_idx = list(range(5))
        neg_idx = list(range(5, 9))
        eng, neg = compute_scaled_scores(probs, pos_idx, neg_idx)
        assert eng.dtype in (np.int64, np.int32, np.intp)
        assert neg.dtype in (np.int64, np.int32, np.intp)

    def test_relative_ordering_preserved(self) -> None:
        """Higher raw positive sum → higher engagement score."""
        # Content 0: high positive, low negative
        # Content 1: low positive, high negative
        probs = np.array([
            [0.3, 0.2, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.16],
            [0.05, 0.05, 0.02, 0.02, 0.01, 0.3, 0.2, 0.15, 0.20],
        ])
        pos_idx = list(range(5))
        neg_idx = list(range(5, 9))
        eng, neg = compute_scaled_scores(probs, pos_idx, neg_idx)
        assert eng[0] > eng[1], "Content 0 should have higher engagement"
        assert neg[0] < neg[1], "Content 0 should have lower negativity"


class TestStakeholderDescriptions:
    """Tests for natural language stakeholder perspective mapping."""

    def test_low_alpha_engagement_focused(self) -> None:
        """Alpha ≤ 0.3 → engagement-focused description."""
        desc = stakeholder_description(0.1)
        assert "engagement" in desc.lower()
        assert "barely" in desc.lower() or "normal" in desc.lower()

    def test_high_alpha_safety_focused(self) -> None:
        """Alpha > 3.0 → safety-first description."""
        desc = stakeholder_description(5.0)
        assert "safety" in desc.lower()
        assert "dealbreaker" in desc.lower()

    def test_all_alphas_return_nonempty(self) -> None:
        """Every alpha value should produce a non-empty description."""
        for alpha in [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            desc = stakeholder_description(alpha)
            assert len(desc) > 20, f"Empty description for α={alpha}"

    def test_different_alphas_different_descriptions(self) -> None:
        """Different alpha ranges should produce different descriptions."""
        descs = {stakeholder_description(a) for a in [0.1, 0.5, 1.0, 2.0, 5.0]}
        assert len(descs) == 5, "Expected 5 distinct stakeholder descriptions"


class TestPromptConstruction:
    """Tests for v2 batch prompt construction."""

    def test_prompt_contains_perspective_not_formula(self) -> None:
        """System prompt should have stakeholder perspective, not α value."""
        eng = np.array([72, 65, 80])
        neg = np.array([15, 45, 30])
        rng = np.random.default_rng(42)
        system, _, _ = build_batch_prompt(eng, neg, [(0, 1)], alpha=4.0, rng=rng)
        # Should NOT contain the formula or alpha
        assert "4.0" not in system
        assert "alpha" not in system.lower()
        # Should contain perspective keywords
        assert "safety" in system.lower() or "moderator" in system.lower()

    def test_prompt_contains_engagement_negativity(self) -> None:
        """User prompt should show engagement=X negativity=Y format."""
        eng = np.array([72, 65])
        neg = np.array([15, 45])
        rng = np.random.default_rng(42)
        _, user, _ = build_batch_prompt(eng, neg, [(0, 1)], alpha=1.0, rng=rng)
        assert "engagement=" in user
        assert "negativity=" in user

    def test_prompt_contains_n_pairs(self) -> None:
        """A batch of 5 pairs produces exactly 5 'Pair N:' sections."""
        eng = np.array([50, 60, 70, 80, 90, 40])
        neg = np.array([10, 20, 30, 40, 50, 60])
        pairs = [(0, 1), (2, 3), (4, 5), (0, 3), (1, 4)]
        rng = np.random.default_rng(42)
        _, user, swaps = build_batch_prompt(eng, neg, pairs, alpha=1.0, rng=rng)
        for i in range(1, 6):
            assert f"Pair {i}:" in user
        assert len(swaps) == 5

    def test_system_prompt_has_json_schema(self) -> None:
        """System prompt must instruct JSON response format."""
        eng = np.array([50, 60])
        neg = np.array([10, 20])
        rng = np.random.default_rng(42)
        system, _, _ = build_batch_prompt(eng, neg, [(0, 1)], alpha=1.0, rng=rng)
        assert "JSON" in system
        assert "confidence" in system

    def test_swap_flags_affect_ab_ordering(self) -> None:
        """When swap=True, Post A shows content from c2, not c1."""
        eng = np.array([10, 99])
        neg = np.array([5, 50])
        # Use a fixed rng that produces swap=True for first pair
        # We'll check both orderings exist across multiple seeds
        found_swap = False
        found_no_swap = False
        for seed in range(20):
            rng = np.random.default_rng(seed)
            _, user, swaps = build_batch_prompt(
                eng, neg, [(0, 1)], alpha=1.0, rng=rng
            )
            if swaps[0]:
                found_swap = True
                assert "Post A: engagement=99" in user
            else:
                found_no_swap = True
                assert "Post A: engagement=10" in user
        assert found_swap and found_no_swap, "Expected both swap and no-swap"


class TestResponseParsing:
    """Tests for LLM response parsing and validation."""

    def test_valid_json_parsed(self) -> None:
        """A well-formed JSON response returns correct values."""
        response = json.dumps(
            {"pairs": [{"pair_id": 1, "preferred": "A", "confidence": 0.85}]}
        )
        results = parse_llm_response(response, expected_count=1)
        assert len(results) == 1
        assert results[0]["preferred"] == "A"
        assert results[0]["confidence"] == 0.85

    def test_confidence_clamped_low(self) -> None:
        """Confidence below 0.50 is clamped to 0.50."""
        response = json.dumps(
            {"pairs": [{"pair_id": 1, "preferred": "A", "confidence": 0.30}]}
        )
        results = parse_llm_response(response, expected_count=1)
        assert results[0]["confidence"] == 0.50

    def test_confidence_clamped_high(self) -> None:
        """Confidence above 1.00 is clamped to 1.00."""
        response = json.dumps(
            {"pairs": [{"pair_id": 1, "preferred": "B", "confidence": 1.20}]}
        )
        results = parse_llm_response(response, expected_count=1)
        assert results[0]["confidence"] == 1.00

    def test_invalid_preferred_raises(self) -> None:
        """'preferred' must be 'A' or 'B'."""
        response = json.dumps(
            {"pairs": [{"pair_id": 1, "preferred": "C", "confidence": 0.80}]}
        )
        try:
            parse_llm_response(response, expected_count=1)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestDisagreementComputation:
    """Tests for LLM disagreement rate and confidence computation."""

    def test_identical_preferences_zero_disagreement(self) -> None:
        """If both stakeholders prefer the same item in all pairs, d=0."""
        responses = [
            {"pair_id": i, "preferred": "A", "confidence": 0.80}
            for i in range(10)
        ]
        result = compute_llm_disagreement_and_confidence(responses, responses)
        assert result["rate"] == 0.0
        assert result["n_disagreed"] == 0

    def test_all_disagreed_rate_one(self) -> None:
        """If they disagree on every pair, d=1.0."""
        resp_a = [
            {"pair_id": i, "preferred": "A", "confidence": 0.80}
            for i in range(5)
        ]
        resp_b = [
            {"pair_id": i, "preferred": "B", "confidence": 0.70}
            for i in range(5)
        ]
        result = compute_llm_disagreement_and_confidence(resp_a, resp_b)
        assert result["rate"] == 1.0
        assert result["n_disagreed"] == 5

    def test_confidence_computed_on_disagreed_only(self) -> None:
        """Mean confidence should only include disagreed pairs."""
        resp_a = [
            {"pair_id": 0, "preferred": "A", "confidence": 0.90},
            {"pair_id": 1, "preferred": "A", "confidence": 0.90},
            {"pair_id": 2, "preferred": "A", "confidence": 0.90},
            {"pair_id": 3, "preferred": "A", "confidence": 0.60},
            {"pair_id": 4, "preferred": "A", "confidence": 0.60},
        ]
        resp_b = [
            {"pair_id": 0, "preferred": "A", "confidence": 0.90},
            {"pair_id": 1, "preferred": "A", "confidence": 0.90},
            {"pair_id": 2, "preferred": "A", "confidence": 0.90},
            {"pair_id": 3, "preferred": "B", "confidence": 0.70},
            {"pair_id": 4, "preferred": "B", "confidence": 0.70},
        ]
        result = compute_llm_disagreement_and_confidence(resp_a, resp_b)
        assert result["rate"] == 0.4
        assert result["n_disagreed"] == 2
        assert abs(result["mean_confidence_sum"] - 1.30) < 1e-10

    def test_uses_sum_not_mean_for_confidence(self) -> None:
        """Total confidence = conf_a + conf_b (parallels margin_a + margin_b)."""
        resp_a = [{"pair_id": 0, "preferred": "A", "confidence": 0.80}]
        resp_b = [{"pair_id": 0, "preferred": "B", "confidence": 0.90}]
        result = compute_llm_disagreement_and_confidence(resp_a, resp_b)
        assert abs(result["mean_confidence_sum"] - 1.70) < 1e-10

    def test_no_disagreed_returns_zero(self) -> None:
        """If d=0, mean confidence on disagreed pairs should be 0.0."""
        resp = [{"pair_id": 0, "preferred": "A", "confidence": 0.95}]
        result = compute_llm_disagreement_and_confidence(resp, resp)
        assert result["mean_confidence_sum"] == 0.0


class TestRegressionModels:
    """Tests for the regression fitting (synthetic data, no LLM)."""

    def test_perfect_linear_r2_one(self) -> None:
        """If cos = 1.0 - 2.0*d exactly, R² should be ~1.0."""
        d = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        y = 1.0 - 2.0 * d
        r2 = fit_linear_r2(d, y)
        assert abs(r2 - 1.0) < 1e-10

    def test_two_var_r2_geq_single(self) -> None:
        """2-variable model R² must be >= single-variable R²."""
        rng = np.random.default_rng(42)
        x1 = rng.uniform(0, 1, 20)
        x2 = rng.uniform(0, 1, 20)
        y = 1.0 - 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.05, 20)
        r2_single = fit_linear_r2(x1, y)
        r2_two = fit_two_var_r2(x1, x2, y)
        assert r2_two >= r2_single - 1e-10


class TestGoNoGo:
    """Tests for the go/no-go criteria evaluation (v2: Spearman-based)."""

    def test_all_pass_is_go(self) -> None:
        """All criteria passing should yield GO."""
        result = evaluate_go_no_go(
            llm_two_var_spearman=0.92,
            confidence_margin_spearman=0.80,
            spearman_gap=0.05,
            disagreement_correlation=0.95,
        )
        assert result["overall"] == "GO"

    def test_must_fail_is_no_go(self) -> None:
        """A MUST criterion failing should yield NO_GO."""
        result = evaluate_go_no_go(
            llm_two_var_spearman=0.50,  # MUST fails
            confidence_margin_spearman=0.80,
            spearman_gap=0.05,
            disagreement_correlation=0.95,
        )
        assert result["overall"] == "NO_GO"

    def test_should_fail_is_conditional(self) -> None:
        """MUST passing but SHOULD failing should yield CONDITIONAL_GO."""
        result = evaluate_go_no_go(
            llm_two_var_spearman=0.90,
            confidence_margin_spearman=0.50,  # SHOULD fails
            spearman_gap=0.05,
            disagreement_correlation=0.95,
        )
        assert result["overall"] == "CONDITIONAL_GO"

    def test_uses_spearman_not_r2(self) -> None:
        """v2: primary criterion is Spearman, not R²."""
        result = evaluate_go_no_go(
            llm_two_var_spearman=0.90,
            confidence_margin_spearman=0.80,
            spearman_gap=0.05,
            disagreement_correlation=0.95,
        )
        assert "llm_two_var_spearman" in result["criteria"]
        assert "llm_two_var_r2" not in result["criteria"]
        assert "formula_agreement" not in result["criteria"]
