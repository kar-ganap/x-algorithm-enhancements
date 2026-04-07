"""Generalized K-stakeholder frontier computation and N-dimensional Pareto.

Computes the Pareto frontier for arbitrary numbers of stakeholders,
using the same engagement scorer + diversity bonus mechanism as
stakeholder_utilities.compute_pareto_frontier but evaluating K
weight-vector-based utilities instead of the hardcoded 3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Lazy import to avoid __init__.py chain pulling in Phoenix/grok.
# ACTION_INDICES only needed for legacy Phoenix scorer (when scorer_weights is None).
ACTION_INDICES: dict[str, int] | None = None


def _get_action_indices() -> dict[str, int]:
    global ACTION_INDICES
    if ACTION_INDICES is None:
        from enhancements.reward_modeling.weights import ACTION_INDICES as _AI
        ACTION_INDICES = _AI
    return ACTION_INDICES

UtilityPoint = dict[str, float]


def compute_k_frontier(
    base_action_probs: np.ndarray,
    content_topics: np.ndarray,
    stakeholder_weights: dict[str, np.ndarray],
    diversity_weights: list[float],
    top_k: int = 10,
    scorer_weights: np.ndarray | None = None,
) -> list[UtilityPoint]:
    """Compute frontier with K stakeholder utilities.

    Uses engagement scorer + diversity bonus for content selection,
    then evaluates each stakeholder's utility as w_k @ action_probs.

    Args:
        base_action_probs: [N_users, N_content, D] tensor.
        content_topics: [N_content] topic assignments.
        stakeholder_weights: Map of name -> D-dim weight vector.
        diversity_weights: List of dw values to sweep (e.g., 0.0 to 1.0).
        top_k: Number of recommendations per user.
        scorer_weights: [D] custom engagement scorer. If None, uses the
            hardcoded Phoenix formula (fav + 0.8*repost + 0.5*follow).

    Returns:
        List of UtilityPoint dicts with "diversity_weight" and one
        "{name}_utility" key per stakeholder.
    """
    n_users, n_content, _ = base_action_probs.shape
    num_topics = int(np.max(content_topics)) + 1
    stakeholder_names = sorted(stakeholder_weights.keys())
    points: list[UtilityPoint] = []

    for dw in diversity_weights:
        recommendations = []

        for user_idx in range(n_users):
            probs = base_action_probs[user_idx]
            if scorer_weights is not None:
                engagement_scores = probs @ scorer_weights
            else:
                # Legacy Phoenix formula
                ai = _get_action_indices()
                engagement_scores = (
                    probs[:, ai["favorite"]]
                    + probs[:, ai["repost"]] * 0.8
                    + probs[:, ai["follow_author"]] * 0.5
                )

            if dw > 0:
                selected: list[int] = []
                remaining = list(range(n_content))
                topic_counts = np.zeros(num_topics)
                for _ in range(top_k):
                    if not remaining:
                        break
                    adjusted = []
                    for idx in remaining:
                        topic = content_topics[idx]
                        div_bonus = 1.0 / (topic_counts[topic] + 1)
                        score = (
                            (1 - dw) * engagement_scores[idx]
                            + dw * div_bonus
                        )
                        adjusted.append((idx, score))
                    best_idx = max(adjusted, key=lambda x: x[1])[0]
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    topic_counts[content_topics[best_idx]] += 1
                recommendations.append(selected)
            else:
                top_indices = np.argsort(engagement_scores)[-top_k:][::-1]
                recommendations.append(top_indices.tolist())

        recs = np.array(recommendations)

        # Evaluate all K stakeholder utilities
        point: UtilityPoint = {"diversity_weight": dw}
        for name in stakeholder_names:
            w = stakeholder_weights[name]
            utils = []
            for user_idx in range(n_users):
                rec_probs = base_action_probs[user_idx, recs[user_idx]]
                utils.append(float(np.sum(rec_probs @ w)))
            point[f"{name}_utility"] = float(np.mean(utils))

        points.append(point)

    return points


def compute_scorer_eval_frontier(
    base_action_probs: np.ndarray,
    content_topics: np.ndarray,
    scorer_weights: np.ndarray,
    eval_weights: dict[str, np.ndarray],
    diversity_weights: list[float],
    top_k: int = 10,
) -> list[UtilityPoint]:
    """Frontier with separate scorer (selection) and eval (measurement) weights.

    Scores content with scorer_weights for top-K selection,
    then evaluates with eval_weights for utility measurement.
    """
    n_users, n_content, _ = base_action_probs.shape
    num_topics = int(np.max(content_topics)) + 1
    stakeholder_names = sorted(eval_weights.keys())
    scorer = np.asarray(scorer_weights, dtype=np.float64)
    points: list[UtilityPoint] = []

    for dw in diversity_weights:
        recommendations = []

        for user_idx in range(n_users):
            scores = base_action_probs[user_idx] @ scorer

            if dw > 0:
                selected: list[int] = []
                remaining = list(range(n_content))
                topic_counts = np.zeros(num_topics)
                for _ in range(top_k):
                    if not remaining:
                        break
                    adjusted = []
                    for idx in remaining:
                        topic = content_topics[idx]
                        div_bonus = 1.0 / (topic_counts[topic] + 1)
                        s = (1 - dw) * scores[idx] + dw * div_bonus
                        adjusted.append((idx, s))
                    best_idx = max(adjusted, key=lambda x: x[1])[0]
                    selected.append(best_idx)
                    remaining.remove(best_idx)
                    topic_counts[content_topics[best_idx]] += 1
                recommendations.append(selected)
            else:
                top_indices = np.argsort(scores)[-top_k:][::-1]
                recommendations.append(top_indices.tolist())

        recs = np.array(recommendations)

        point: UtilityPoint = {"diversity_weight": dw}
        for name in stakeholder_names:
            w = eval_weights[name]
            utils = []
            for user_idx in range(n_users):
                rec_probs = base_action_probs[user_idx, recs[user_idx]]
                utils.append(float(np.sum(rec_probs @ w)))
            point[f"{name}_utility"] = float(np.mean(utils))

        points.append(point)

    return points


def is_dominated_nd(
    point: UtilityPoint,
    candidates: list[UtilityPoint],
    dims: list[str],
) -> bool:
    """Check if point is strictly dominated by any candidate in N dims."""
    for c in candidates:
        if c is point:
            continue
        all_geq = all(c[d] >= point[d] for d in dims)
        any_gt = any(c[d] > point[d] for d in dims)
        if all_geq and any_gt:
            return True
    return False


def extract_pareto_front_nd(
    points: list[UtilityPoint],
    dims: list[str],
) -> list[UtilityPoint]:
    """Extract N-dimensional Pareto-optimal subset (maximize all dims)."""
    return [p for p in points if not is_dominated_nd(p, points, dims)]


def compute_regret_on_dim(
    partial: list[UtilityPoint],
    full: list[UtilityPoint],
    hidden_dim: str,
) -> dict[str, float]:
    """Compute regret on hidden dimension.

    Same logic as compute_regret in analyze_partial_observation.py.
    """
    if not partial or not full:
        return {"max_regret": 0.0, "avg_regret": 0.0, "min_regret": 0.0}
    max_achievable = max(f[hidden_dim] for f in full)
    regrets = [max_achievable - p[hidden_dim] for p in partial]
    return {
        "max_regret": float(max(regrets)),
        "avg_regret": float(np.mean(regrets)),
        "min_regret": float(min(regrets)),
    }
