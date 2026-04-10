"""MIND multi-stakeholder preference generation.

Defines five domain-native stakeholder utility functions over 35-dim feature
vectors (17 top categories + 14 subcategories + 4 quality signals), and
generates preference pairs from MIND click data.

Stakeholders (domain-native to news recommendation):
- reader:         CTR-weighted per category (what users actually click)
- publisher:      Category volume (rewards high-supply categories)
- advertiser:     Positive on non-controversial, negative on newscrime/newspolitics
- journalist:     Serious news focus (newsworld/newspolitics/financenews)
- civic_diversity: Zero-sum anti-concentration (control stakeholder)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load MINDDataset via importlib to avoid any Phoenix import chain.
_mind_path = Path(__file__).parent / "mind.py"
_spec = importlib.util.spec_from_file_location("_mind", _mind_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MINDDataset = _mod.MINDDataset
NUM_FEATURES = _mod.NUM_FEATURES
NUM_TOP_CATEGORIES = _mod.NUM_TOP_CATEGORIES
NUM_TOP_SUBCATEGORIES = _mod.NUM_TOP_SUBCATEGORIES

# Categories and subcategories that each stakeholder has opinions about.
# Used by `advertiser` and `journalist` for their opinionated weighting.
_ADVERTISER_SAFE_CATEGORIES = {
    "foodanddrink", "autos", "travel", "lifestyle", "music",
    "entertainment", "movies", "tv",
}
_ADVERTISER_RISKY_CATEGORIES = {"news"}  # contains politics, crime subcategories
_ADVERTISER_RISKY_SUBCATEGORIES = {"newscrime", "newspolitics"}

_JOURNALIST_SERIOUS_SUBCATEGORIES = {
    "newsworld", "newspolitics", "newsus",
    "newsscienceandtechnology", "financenews",
}
_JOURNALIST_FRIVOLOUS_CATEGORIES = {
    "kids", "video", "tv", "music", "entertainment",
}


def compute_reader_weights(dataset: MINDDataset) -> np.ndarray:
    """Reader: per-category CTR from impression logs.

    For each category, compute clicks / impressions and assign that as
    the weight. Normalized to max = 1. Quality signals get a small
    positive weight (readers prefer longer, entity-rich articles).

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_clicks = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    cat_impressions = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for rating in dataset.train_ratings:
        item = dataset.items.get(rating.item_id)
        if item is None:
            continue
        idx = cat_to_idx.get(item.category)
        if idx is None:
            continue
        cat_impressions[idx] += 1
        if rating.rating > 0:
            cat_clicks[idx] += 1

    # Smoothed CTR to avoid dividing by small numbers
    ctr = (cat_clicks + 1.0) / (cat_impressions + 10.0)
    max_ctr = float(np.max(ctr)) if np.max(ctr) > 0 else 1.0
    weights[:NUM_TOP_CATEGORIES] = (ctr / max_ctr).astype(np.float32)

    # Quality signals: small positive — readers prefer longer articles with entities.
    qstart = NUM_TOP_CATEGORIES + NUM_TOP_SUBCATEGORIES
    weights[qstart + 0] = 0.2  # title length
    weights[qstart + 1] = 0.2  # abstract length
    weights[qstart + 2] = 0.3  # entity count
    return weights


def compute_publisher_weights(dataset: MINDDataset) -> np.ndarray:
    """Publisher: category volume (articles published per category).

    Rewards high-volume categories. Normalized to max = 1.

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_counts = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for item in dataset.items.values():
        idx = cat_to_idx.get(item.category)
        if idx is not None:
            cat_counts[idx] += 1

    max_c = float(np.max(cat_counts)) if np.max(cat_counts) > 0 else 1.0
    weights[:NUM_TOP_CATEGORIES] = (cat_counts / max_c).astype(np.float32)
    return weights


def compute_advertiser_weights(dataset: MINDDataset) -> np.ndarray:
    """Advertiser: positive on non-controversial, negative on crime/politics.

    Positive on lifestyle/travel/food/autos/entertainment categories.
    Negative on newscrime/newspolitics subcategories and the 'news' category
    (which mostly contains crime/politics content).

    Intended to produce cos < 0 with `reader` (because readers click news
    heavily but advertisers avoid it) and cos < 0 with `journalist`.

    Returns:
        [NUM_FEATURES] float32 in [-1, 1].
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    for i, cat in enumerate(dataset.categories):
        if cat in _ADVERTISER_SAFE_CATEGORIES:
            weights[i] = 1.0
        elif cat in _ADVERTISER_RISKY_CATEGORIES:
            weights[i] = -1.0
    for j, sub in enumerate(dataset.subcategories):
        if sub in _ADVERTISER_RISKY_SUBCATEGORIES:
            weights[NUM_TOP_CATEGORIES + j] = -1.0
    return weights


def compute_journalist_weights(dataset: MINDDataset) -> np.ndarray:
    """Journalist: positive on serious news, negative on frivolous.

    Positive on newsworld/newspolitics/newsus/financenews/newsscienceandtechnology
    (top-14 subcats that are 'serious news').
    Negative on kids/video/tv/music/entertainment categories.

    Intended to produce cos < 0 with `advertiser` and roughly orthogonal
    to `reader`.

    Quality signals: positive (serious journalism values longer articles
    with more entities — real reporting, not clickbait).

    Returns:
        [NUM_FEATURES] float32 in [-1, 1].
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    for i, cat in enumerate(dataset.categories):
        if cat in _JOURNALIST_FRIVOLOUS_CATEGORIES:
            weights[i] = -0.5
    for j, sub in enumerate(dataset.subcategories):
        if sub in _JOURNALIST_SERIOUS_SUBCATEGORIES:
            weights[NUM_TOP_CATEGORIES + j] = 1.0

    qstart = NUM_TOP_CATEGORIES + NUM_TOP_SUBCATEGORIES
    weights[qstart + 0] = 0.5  # title length (longer = more serious)
    weights[qstart + 1] = 0.5  # abstract length
    weights[qstart + 2] = 0.5  # entities
    return weights


def compute_civic_diversity_weights(dataset: MINDDataset) -> np.ndarray:
    """Civic diversity: zero-sum anti-concentration across categories.

    weight[cat] = -fraction(cat) + 1/num_active_cats.
    Popular categories get negative weight, rare get positive. Weights
    sum to approximately 0 (within the top-17 category block).

    Same structural role as MovieLens `diversity` stakeholder — provides
    a cross-dataset control.

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_counts = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for item in dataset.items.values():
        idx = cat_to_idx.get(item.category)
        if idx is not None:
            cat_counts[idx] += 1

    total = float(np.sum(cat_counts))
    num_active = int(np.sum(cat_counts > 0))
    if total == 0 or num_active == 0:
        return weights

    fractions = cat_counts / total
    uniform = 1.0 / num_active
    w_cats = np.where(cat_counts > 0, -fractions + uniform, 0.0)

    max_abs = float(np.max(np.abs(w_cats))) if np.max(np.abs(w_cats)) > 0 else 1.0
    weights[:NUM_TOP_CATEGORIES] = (w_cats / max_abs).astype(np.float32)
    return weights


def build_stakeholder_configs(dataset: MINDDataset) -> dict[str, np.ndarray]:
    """Build all five stakeholder weight vectors for MIND.

    Returns:
        Dict with keys: reader, publisher, advertiser, journalist, civic_diversity.
    """
    return {
        "reader": compute_reader_weights(dataset),
        "publisher": compute_publisher_weights(dataset),
        "advertiser": compute_advertiser_weights(dataset),
        "journalist": compute_journalist_weights(dataset),
        "civic_diversity": compute_civic_diversity_weights(dataset),
    }


def generate_mind_content_pool(
    dataset: MINDDataset,
    min_impressions: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build content pool with feature vectors from MIND articles.

    Filters to articles that received at least `min_impressions` impressions
    (which is our MIND equivalent of MovieLens's "min_ratings" filter).
    Content feature = article.features (the 35-dim vector), unmodified.

    Args:
        dataset: Loaded MIND dataset.
        min_impressions: Minimum number of impressions for an article to be included.
        seed: Random seed (for consistent ordering).

    Returns:
        content_features: [M, NUM_FEATURES] feature vectors.
        content_topics: [M] primary category index per article (for diversity selection).
    """
    impression_counts: dict[int, int] = {}
    for rating in dataset.train_ratings:
        impression_counts[rating.item_id] = impression_counts.get(rating.item_id, 0) + 1

    eligible_ids = sorted(
        item_id for item_id, count in impression_counts.items()
        if count >= min_impressions and item_id in dataset.items
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_ids)

    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}
    features = np.zeros((len(eligible_ids), NUM_FEATURES), dtype=np.float32)
    topics = np.zeros(len(eligible_ids), dtype=np.int32)

    for i, item_id in enumerate(eligible_ids):
        item = dataset.items[item_id]
        features[i] = item.features
        topics[i] = cat_to_idx.get(item.category, 0)

    return features, topics


def generate_mind_preferences(
    content_features: np.ndarray,
    stakeholder_weights: np.ndarray,
    n_pairs: int,
    seed: int,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs for one stakeholder from content features.

    For each pair: sample two articles, score with stakeholder weights,
    add noise, label higher-score article as preferred. Same algorithm as
    the MovieLens version — the only difference is content_features shape.

    Args:
        content_features: [M, D] feature vectors per article.
        stakeholder_weights: [D] stakeholder utility weights.
        n_pairs: Number of preference pairs to generate.
        seed: Random seed.
        noise_std: Standard deviation of Gaussian noise on utility scores.

    Returns:
        (probs_preferred [n, D], probs_rejected [n, D])
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_features)
    feature_dim = content_features.shape[1]

    utility = content_features @ stakeholder_weights

    probs_pref = np.zeros((n_pairs, feature_dim), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, feature_dim), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]
        noise = rng.normal(0, noise_std)

        if (diff + noise) > 0:
            probs_pref[i] = content_features[c1]
            probs_rej[i] = content_features[c2]
        else:
            probs_pref[i] = content_features[c2]
            probs_rej[i] = content_features[c1]

    return probs_pref, probs_rej


def split_preferences(
    probs_pref: np.ndarray,
    probs_rej: np.ndarray,
    eval_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split preference pairs into train and eval sets.

    Identical to the MovieLens version — kept here for drop-in compatibility.
    """
    rng = np.random.default_rng(seed)
    n = len(probs_pref)
    n_eval = int(n * eval_fraction)

    indices = rng.permutation(n)
    eval_idx = indices[:n_eval]
    train_idx = indices[n_eval:]

    return (
        probs_pref[train_idx],
        probs_rej[train_idx],
        probs_pref[eval_idx],
        probs_rej[eval_idx],
    )


def compute_label_disagreement(
    content_features: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    n_pairs: int = 10000,
    seed: int = 42,
) -> float:
    """Compute label disagreement rate between two stakeholders."""
    rng = np.random.default_rng(seed)
    n_content = len(content_features)

    utility_a = content_features @ weights_a
    utility_b = content_features @ weights_b

    disagreements = 0
    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        pref_a = utility_a[c1] > utility_a[c2]
        pref_b = utility_b[c1] > utility_b[c2]
        if pref_a != pref_b:
            disagreements += 1

    return disagreements / n_pairs
