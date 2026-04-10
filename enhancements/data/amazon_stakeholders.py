"""Amazon Kindle multi-stakeholder preference generation.

Defines five domain-native stakeholder utility functions over 32-dim feature
vectors (20 categories + 4 price + 3 volume + 3 sentiment + 2 length), and
generates preference pairs from Amazon Kindle 5-core review data.

Stakeholders (domain-native to e-commerce/book recommendation):
- reader:          Star-rating-weighted per category (what readers actually like)
- publisher:       Category × rating-count (rewards high-volume categories)
- indie_author:    Negative on top-3 populated categories, positive on niche
- premium_seller:  Positive on high-price × high-rating, negative on free tier
- diversity:       Zero-sum anti-concentration (cross-dataset control)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Load AmazonDataset via importlib to avoid any Phoenix import chain.
# Must register in sys.modules BEFORE exec_module so that dataclass
# type introspection (which reads cls.__module__) works with PEP-604
# union types (e.g. `float | None`) under Python 3.11.
_amazon_path = Path(__file__).parent / "amazon.py"
_spec = importlib.util.spec_from_file_location("_amazon", _amazon_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_amazon"] = _mod
_spec.loader.exec_module(_mod)
AmazonDataset = _mod.AmazonDataset
NUM_FEATURES = _mod.NUM_FEATURES
NUM_TOP_CATEGORIES = _mod.NUM_TOP_CATEGORIES
NUM_PRICE_BUCKETS = _mod.NUM_PRICE_BUCKETS
NUM_VOLUME_BUCKETS = _mod.NUM_VOLUME_BUCKETS
NUM_SENTIMENT_BUCKETS = _mod.NUM_SENTIMENT_BUCKETS
NUM_LENGTH_INDICATORS = _mod.NUM_LENGTH_INDICATORS

# Feature slot constants (must match amazon.py layout)
_CAT_START = 0
_PRICE_START = NUM_TOP_CATEGORIES
_VOL_START = _PRICE_START + NUM_PRICE_BUCKETS
_SENT_START = _VOL_START + NUM_VOLUME_BUCKETS
_LEN_START = _SENT_START + NUM_SENTIMENT_BUCKETS


def compute_reader_weights(dataset: AmazonDataset) -> np.ndarray:
    """Reader: per-category mean star rating from review data.

    For each category, compute mean rating (centered at 3.0) and assign
    that as the weight. Also reward high sentiment buckets and medium
    price tiers (readers prefer decent-quality mid-priced books).

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_rating_sums = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    cat_rating_counts = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)

    for rating in dataset.train_ratings:
        item = dataset.items.get(rating.item_id)
        if item is None:
            continue
        idx = cat_to_idx.get(item.main_category)
        if idx is None:
            continue
        cat_rating_sums[idx] += rating.rating - 3.0
        cat_rating_counts[idx] += 1

    mask = cat_rating_counts > 0
    w_cats = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    w_cats[mask] = cat_rating_sums[mask] / cat_rating_counts[mask]

    max_abs = float(np.max(np.abs(w_cats))) if np.max(np.abs(w_cats)) > 0 else 1.0
    weights[_CAT_START:_CAT_START + NUM_TOP_CATEGORIES] = (w_cats / max_abs).astype(np.float32)

    # Sentiment: reward high-rated buckets strongly (readers want 4-5 star books)
    weights[_SENT_START + 0] = -0.5  # 1-3 star bucket: negative
    weights[_SENT_START + 1] = 0.2   # 3-4 star bucket: mild positive
    weights[_SENT_START + 2] = 0.5   # 4-5 star bucket: strong positive

    # Price: mild preference for mid-range (not free, not premium)
    weights[_PRICE_START + 1] = 0.2  # under_5
    weights[_PRICE_START + 2] = 0.3  # mid_5_15
    return weights


def compute_publisher_weights(dataset: AmazonDataset) -> np.ndarray:
    """Publisher: category × review-count (rewards high-volume categories).

    Weight = (sum of rating_number per category) normalized. Rewards
    categories where publishers can sell volume. Also mildly rewards
    popular-volume bucket.

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_volume = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for item in dataset.items.values():
        idx = cat_to_idx.get(item.main_category)
        if idx is None:
            continue
        vol = item.rating_number or 0
        cat_volume[idx] += vol

    max_v = float(np.max(cat_volume)) if np.max(cat_volume) > 0 else 1.0
    weights[_CAT_START:_CAT_START + NUM_TOP_CATEGORIES] = (cat_volume / max_v).astype(np.float32)

    # Reward high-volume and mid-volume buckets
    weights[_VOL_START + 0] = -0.2  # niche bucket: negative
    weights[_VOL_START + 1] = 0.3   # mid bucket
    weights[_VOL_START + 2] = 0.6   # popular bucket: strong positive
    return weights


def compute_indie_author_weights(dataset: AmazonDataset) -> np.ndarray:
    """Indie author: negative on top-3 populated categories, positive on niche.

    Intended to produce cos < 0 with `publisher` (indie authors want the
    opposite of what big publishers want: niche categories with less
    competition).

    Returns:
        [NUM_FEATURES] float32 in [-1, 1].
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    # Count items per category (matches publisher's "populated" definition)
    cat_counts = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for item in dataset.items.values():
        idx = cat_to_idx.get(item.main_category)
        if idx is not None:
            cat_counts[idx] += 1

    # Negative on top-3 populated, positive on bottom half
    sorted_idx = np.argsort(cat_counts)[::-1]  # descending
    top3 = sorted_idx[:3]
    bottom_half = sorted_idx[NUM_TOP_CATEGORIES // 2:]

    w_cats = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float32)
    w_cats[top3] = -1.0
    for idx in bottom_half:
        if cat_counts[idx] > 0:
            w_cats[idx] = 0.5
    weights[_CAT_START:_CAT_START + NUM_TOP_CATEGORIES] = w_cats

    # Reward niche volume bucket, penalize popular
    weights[_VOL_START + 0] = 0.8   # niche: strong positive
    weights[_VOL_START + 1] = 0.2   # mid: mild positive
    weights[_VOL_START + 2] = -0.8  # popular: strong negative
    return weights


def compute_premium_seller_weights(dataset: AmazonDataset) -> np.ndarray:
    """Premium seller: positive on high-price × high-rating, negative on free.

    Tests whether price can anchor a stakeholder orthogonal to content
    categories. Should be roughly orthogonal to reader/publisher/indie
    (which operate mostly on categories) but partially aligned with reader
    on sentiment.

    Returns:
        [NUM_FEATURES] float32 in [-1, 1].
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Price: strong positive on over-$15, negative on free
    weights[_PRICE_START + 0] = -1.0  # free: strong negative
    weights[_PRICE_START + 1] = -0.3  # under_5: mild negative
    weights[_PRICE_START + 2] = 0.3   # mid_5_15: mild positive
    weights[_PRICE_START + 3] = 1.0   # over_15: strong positive

    # Sentiment: reward high-rating (premium sellers want their premium books loved)
    weights[_SENT_START + 0] = -0.8  # 1-3 stars
    weights[_SENT_START + 1] = 0.0   # 3-4 stars
    weights[_SENT_START + 2] = 0.8   # 4-5 stars

    # Length: prefer longer books (more "premium")
    weights[_LEN_START + 0] = -0.2  # short
    weights[_LEN_START + 1] = 0.4   # long
    return weights


def compute_diversity_weights(dataset: AmazonDataset) -> np.ndarray:
    """Diversity: zero-sum anti-concentration across categories.

    weight[cat] = -fraction(cat) + 1/num_active_cats.
    Popular categories get negative weight, rare get positive. Provides
    cross-dataset control (same structure as MovieLens diversity).

    Returns:
        [NUM_FEATURES] float32.
    """
    weights = np.zeros(NUM_FEATURES, dtype=np.float32)
    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}

    cat_counts = np.zeros(NUM_TOP_CATEGORIES, dtype=np.float64)
    for item in dataset.items.values():
        idx = cat_to_idx.get(item.main_category)
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
    weights[_CAT_START:_CAT_START + NUM_TOP_CATEGORIES] = (w_cats / max_abs).astype(np.float32)
    return weights


def build_stakeholder_configs(dataset: AmazonDataset) -> dict[str, np.ndarray]:
    """Build all five stakeholder weight vectors for Amazon Kindle.

    Returns:
        Dict with keys: reader, publisher, indie_author, premium_seller, diversity.
    """
    return {
        "reader": compute_reader_weights(dataset),
        "publisher": compute_publisher_weights(dataset),
        "indie_author": compute_indie_author_weights(dataset),
        "premium_seller": compute_premium_seller_weights(dataset),
        "diversity": compute_diversity_weights(dataset),
    }


def generate_amazon_content_pool(
    dataset: AmazonDataset,
    min_ratings: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build content pool with feature vectors from Amazon books.

    Args:
        dataset: Loaded Amazon dataset.
        min_ratings: Minimum number of ratings for a book to be included
            (in the subset, which was already 5-core filtered).
        seed: Random seed (for consistent ordering).

    Returns:
        content_features: [M, NUM_FEATURES] feature vectors.
        content_topics: [M] primary category index per book.
    """
    item_rating_counts: dict[int, int] = {}
    for rating in dataset.train_ratings:
        item_rating_counts[rating.item_id] = item_rating_counts.get(rating.item_id, 0) + 1

    eligible_ids = sorted(
        item_id for item_id, count in item_rating_counts.items()
        if count >= min_ratings and item_id in dataset.items
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_ids)

    cat_to_idx = {c: i for i, c in enumerate(dataset.categories)}
    features = np.zeros((len(eligible_ids), NUM_FEATURES), dtype=np.float32)
    topics = np.zeros(len(eligible_ids), dtype=np.int32)

    for i, item_id in enumerate(eligible_ids):
        item = dataset.items[item_id]
        features[i] = item.features
        topics[i] = cat_to_idx.get(item.main_category, 0)

    return features, topics


def generate_amazon_preferences(
    content_features: np.ndarray,
    stakeholder_weights: np.ndarray,
    n_pairs: int,
    seed: int,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs for one stakeholder from content features.

    Same algorithm as the MovieLens/MIND versions.

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
    """Split preference pairs into train and eval sets."""
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
