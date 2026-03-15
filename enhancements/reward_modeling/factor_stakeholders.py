"""Factor-based stakeholder generation for K-stakeholder scaling experiments.

Generates K stakeholders as linear combinations of F=4 interpretable
latent factors over the 18-action space. Factor loadings sampled from
Dirichlet distributions with controllable concentration (correlation).

Usage:
    weights = generate_stakeholder_weights(K=5, concentration=2.0, seed=42)
    # weights["s_0"], weights["s_1"], ... are 18-dim weight vectors
"""

from __future__ import annotations

import numpy as np

from enhancements.reward_modeling.weights import ACTION_INDICES, NUM_ACTIONS

# ---------------------------------------------------------------------------
# Raw factor definitions (before orthogonalization)
# ---------------------------------------------------------------------------

_POSITIVE_ACTIONS = ["favorite", "repost", "follow_author", "share", "reply"]
_NEGATIVE_ACTIONS = ["block_author", "mute_author", "report", "not_interested"]


def _build_raw_factors() -> np.ndarray:
    """Build 8 raw factor vectors (8 x 18), then orthogonalize via QR.

    8 factors cover all 18 actions with interpretable groupings:
    0-3: Original factors (engagement, safety, virality, retention)
    4-7: New factors (passive consumption, link sharing, discovery, quality)
    """
    factors = np.zeros((8, NUM_ACTIONS))

    # Factor 0: Engagement — direct positive interactions
    for a in _POSITIVE_ACTIONS:
        factors[0, ACTION_INDICES[a]] = 1.0
    for a in ["click", "profile_click", "dwell"]:
        factors[0, ACTION_INDICES[a]] = 0.3
    for a in _NEGATIVE_ACTIONS:
        factors[0, ACTION_INDICES[a]] = -0.2

    # Factor 1: Safety — penalizes harmful content signals
    factors[1, ACTION_INDICES["block_author"]] = -1.5
    factors[1, ACTION_INDICES["mute_author"]] = -1.0
    factors[1, ACTION_INDICES["report"]] = -2.0
    factors[1, ACTION_INDICES["not_interested"]] = -0.5
    for a in _POSITIVE_ACTIONS:
        factors[1, ACTION_INDICES[a]] = 0.3

    # Factor 2: Virality — amplification and sharing
    factors[2, ACTION_INDICES["repost"]] = 1.0
    factors[2, ACTION_INDICES["share"]] = 1.0
    factors[2, ACTION_INDICES["share_via_dm"]] = 0.8
    factors[2, ACTION_INDICES["share_via_copy_link"]] = 0.7
    factors[2, ACTION_INDICES["quote"]] = 0.9
    factors[2, ACTION_INDICES["favorite"]] = 0.1
    factors[2, ACTION_INDICES["follow_author"]] = 0.2
    for a in _NEGATIVE_ACTIONS:
        factors[2, ACTION_INDICES[a]] = -0.3

    # Factor 3: Retention — relationship-building signals
    factors[3, ACTION_INDICES["follow_author"]] = 1.5
    factors[3, ACTION_INDICES["reply"]] = 1.0
    factors[3, ACTION_INDICES["profile_click"]] = 0.8
    factors[3, ACTION_INDICES["favorite"]] = 0.3
    factors[3, ACTION_INDICES["dwell"]] = 0.4
    factors[3, ACTION_INDICES["block_author"]] = -1.0
    factors[3, ACTION_INDICES["mute_author"]] = -1.0
    factors[3, ACTION_INDICES["report"]] = -0.5
    factors[3, ACTION_INDICES["not_interested"]] = -0.3

    # Factor 4: Passive consumption — time-based engagement
    factors[4, ACTION_INDICES["dwell"]] = 1.5
    factors[4, ACTION_INDICES["vqv"]] = 1.2  # video quality view
    factors[4, ACTION_INDICES["click"]] = 0.8
    factors[4, ACTION_INDICES["photo_expand"]] = 1.0
    factors[4, ACTION_INDICES["not_interested"]] = -0.4

    # Factor 5: Link sharing — off-platform distribution
    factors[5, ACTION_INDICES["share_via_copy_link"]] = 1.5
    factors[5, ACTION_INDICES["share_via_dm"]] = 1.2
    factors[5, ACTION_INDICES["share"]] = 1.0
    factors[5, ACTION_INDICES["quote"]] = 0.3
    factors[5, ACTION_INDICES["report"]] = -0.5

    # Factor 6: Discovery — exploring new content/creators
    factors[6, ACTION_INDICES["profile_click"]] = 1.5
    factors[6, ACTION_INDICES["quoted_click"]] = 1.2
    factors[6, ACTION_INDICES["follow_author"]] = 0.8
    factors[6, ACTION_INDICES["click"]] = 0.6
    factors[6, ACTION_INDICES["photo_expand"]] = 0.4
    factors[6, ACTION_INDICES["mute_author"]] = -0.8

    # Factor 7: Content quality — signals that content merits attention
    factors[7, ACTION_INDICES["photo_expand"]] = 1.2
    factors[7, ACTION_INDICES["vqv"]] = 1.0
    factors[7, ACTION_INDICES["dwell"]] = 0.8
    factors[7, ACTION_INDICES["reply"]] = 0.6
    factors[7, ACTION_INDICES["favorite"]] = 0.4
    factors[7, ACTION_INDICES["report"]] = -1.0
    factors[7, ACTION_INDICES["not_interested"]] = -0.8

    return factors


def _orthogonalize(factors: np.ndarray) -> np.ndarray:
    """QR-orthogonalize factor matrix (F x 18) -> (F x 18)."""
    q, _ = np.linalg.qr(factors.T)  # (18 x F)
    return q.T[:factors.shape[0]]  # (F x 18)


# Module-level constant: 8 x 18 orthogonalized factor basis
FACTOR_MATRIX: np.ndarray = _orthogonalize(_build_raw_factors())
N_FACTORS: int = FACTOR_MATRIX.shape[0]


# ---------------------------------------------------------------------------
# Stakeholder generation
# ---------------------------------------------------------------------------


def generate_stakeholder_weights(
    k: int,
    concentration: float | list[float] = 2.0,
    seed: int = 42,
    sign_flip_prob: float = 0.2,
) -> dict[str, np.ndarray]:
    """Generate K stakeholder weight vectors from the factor model.

    Args:
        k: Number of stakeholders to generate.
        concentration: Dirichlet concentration parameter. Scalar (uniform)
            or list of length N_FACTORS. Higher = more correlated stakeholders.
        sign_flip_prob: Probability of flipping the sign of each factor
            loading (enables anti-virality, risk-tolerant stakeholders).
        seed: Random seed.

    Returns:
        Dict mapping "s_0", "s_1", ... to 18-dim weight vectors.
    """
    rng = np.random.default_rng(seed)

    if isinstance(concentration, (int, float)):
        alpha = np.full(N_FACTORS, float(concentration))
    else:
        alpha = np.array(concentration, dtype=float)

    weights: dict[str, np.ndarray] = {}
    for i in range(k):
        # Sample factor loadings from Dirichlet
        loadings = rng.dirichlet(alpha)

        # Optional sign flips
        if sign_flip_prob > 0:
            flips = rng.random(N_FACTORS) < sign_flip_prob
            loadings[flips] *= -1

        # Construct weight vector
        w = loadings @ FACTOR_MATRIX
        weights[f"s_{i}"] = w

    return weights


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def compute_effective_rank(
    weights: dict[str, np.ndarray], tol: float = 0.01,
) -> int:
    """Effective rank of the K x 18 weight matrix."""
    W = np.array(list(weights.values()))
    return int(np.linalg.matrix_rank(W, tol=tol))


def compute_pairwise_cosine_matrix(
    weights: dict[str, np.ndarray],
) -> np.ndarray:
    """K x K pairwise cosine similarity matrix."""
    names = sorted(weights.keys())
    k = len(names)
    W = np.array([weights[n] for n in names])
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-12
    W_norm = W / norms
    return W_norm @ W_norm.T


def mean_pairwise_cosine(weights: dict[str, np.ndarray]) -> float:
    """Average off-diagonal cosine similarity."""
    cos_mat = compute_pairwise_cosine_matrix(weights)
    k = cos_mat.shape[0]
    mask = ~np.eye(k, dtype=bool)
    return float(np.mean(cos_mat[mask]))
