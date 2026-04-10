"""Amazon Kindle Store dataset loader (McAuley Lab 2023 release).

Duck-typed to the MovieLensDataset interface used by experiment scripts:
- attributes: train_ratings, val_ratings, test_ratings, items, users
- Rating dataclass with (user_id, item_id, rating, timestamp)
- Items (Book) carry feature vectors for stakeholder utility computation

For Amazon, "rating" is a 1-5 star score. `parent_asin` is the join key
between reviews and metadata (in the 2023 release, not `asin`).

Feature vector (32-dim):
    - 20 top-level Kindle genres (binary one-hot)
    - 4 price-tier buckets (free/<$5/$5-15/>$15)
    - 3 review-volume buckets (niche/mid/popular)
    - 3 review-sentiment buckets (1-3/3-4/4-5 stars)
    - 2 book-length indicators (<200 pages, >=200 pages)

Loads from a pre-built subset created by scripts/data/download_amazon.py:
    data/amazon-kindle/reviews_subset.jsonl
    data/amazon-kindle/meta_subset.jsonl

Usage:
    dataset = AmazonDataset("data/amazon-kindle")
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Feature dimensionality
NUM_TOP_CATEGORIES = 20
NUM_PRICE_BUCKETS = 4
NUM_VOLUME_BUCKETS = 3
NUM_SENTIMENT_BUCKETS = 3
NUM_LENGTH_INDICATORS = 2
NUM_FEATURES = (
    NUM_TOP_CATEGORIES
    + NUM_PRICE_BUCKETS
    + NUM_VOLUME_BUCKETS
    + NUM_SENTIMENT_BUCKETS
    + NUM_LENGTH_INDICATORS
)  # 32

# Feature-space slot boundaries (inclusive-start / exclusive-end)
_CAT_SLICE = slice(0, NUM_TOP_CATEGORIES)
_PRICE_SLICE = slice(NUM_TOP_CATEGORIES, NUM_TOP_CATEGORIES + NUM_PRICE_BUCKETS)
_VOL_SLICE = slice(_PRICE_SLICE.stop, _PRICE_SLICE.stop + NUM_VOLUME_BUCKETS)
_SENT_SLICE = slice(_VOL_SLICE.stop, _VOL_SLICE.stop + NUM_SENTIMENT_BUCKETS)
_LEN_SLICE = slice(_SENT_SLICE.stop, _SENT_SLICE.stop + NUM_LENGTH_INDICATORS)

# Price-tier thresholds in USD
_PRICE_BUCKETS = [
    (0.0, 0.001, "free"),        # exactly 0 = free
    (0.001, 5.0, "under_5"),
    (5.0, 15.0, "mid_5_15"),
    (15.0, float("inf"), "over_15"),
]

# Volume buckets — decided after observing rating_number distribution at
# data load time. Exposed as constants here for transparency.
_VOL_NICHE_MAX = 50
_VOL_MID_MAX = 500


@dataclass
class Rating:
    """A single user-item review."""
    user_id: int
    item_id: int
    rating: float       # 1.0-5.0
    timestamp: int


@dataclass
class Book:
    """Amazon Kindle book with derived feature vector."""
    item_id: int            # integer surrogate
    parent_asin: str        # native join key
    title: str
    main_category: str
    price: float | None     # USD, or None if unknown
    average_rating: float | None
    rating_number: int | None
    num_pages: int | None
    features: np.ndarray    # [NUM_FEATURES] float32

    @property
    def feature_indices(self) -> list[int]:
        return list(np.where(self.features > 0)[0])


@dataclass
class User:
    """Amazon user (minimal)."""
    user_id: int            # integer surrogate
    user_uid: str           # native 'reviewerID' string


@dataclass
class AmazonDataset:
    """Amazon Kindle Store 2023 subset loader.

    Duck-types the MovieLensDataset interface: experiment scripts that
    read train_ratings, val_ratings, test_ratings, items will work.
    """
    data_dir: Path
    items: dict[int, Book] = field(default_factory=dict)
    users: dict[int, User] = field(default_factory=dict)
    train_ratings: list[Rating] = field(default_factory=list)
    val_ratings: list[Rating] = field(default_factory=list)
    test_ratings: list[Rating] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # top-20 Kindle genres

    _user_history: dict[int, list[Rating]] = field(default_factory=dict)
    _asin_to_item_id: dict[str, int] = field(default_factory=dict)
    _user_uid_to_user_id: dict[str, int] = field(default_factory=dict)

    def __init__(self, data_dir: str | Path, val_ratio: float = 0.1, test_ratio: float = 0.1):
        """Load Amazon Kindle subset.

        Args:
            data_dir: Path to data/amazon-kindle/ containing
                reviews_subset.jsonl and meta_subset.jsonl
            val_ratio: Fraction of ratings to hold out for validation (by timestamp).
            test_ratio: Fraction of ratings to hold out for test (by timestamp).
        """
        self.data_dir = Path(data_dir)
        self.items = {}
        self.users = {}
        self.train_ratings = []
        self.val_ratings = []
        self.test_ratings = []
        self.categories = []
        self._user_history = {}
        self._asin_to_item_id = {}
        self._user_uid_to_user_id = {}

        reviews_path = self.data_dir / "reviews_subset.jsonl"
        meta_path = self.data_dir / "meta_subset.jsonl"
        if not reviews_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Amazon subset files not found at {self.data_dir}. "
                f"Run scripts/data/download_amazon.py first."
            )

        # Pass 1: scan metadata to build global category set (top-20) and
        # rating_number distribution (for volume buckets).
        self._scan_metadata(meta_path)

        # Pass 2: load metadata and build Book objects with feature vectors.
        self._load_metadata(meta_path)

        # Pass 3: load ratings and split by timestamp.
        self._load_ratings(reviews_path, val_ratio, test_ratio)

        # Build user history cache
        for rating in self.train_ratings:
            self._user_history.setdefault(rating.user_id, []).append(rating)

    # ── loading helpers ────────────────────────────────────────────

    def _scan_metadata(self, meta_path: Path) -> None:
        """Pass 1: scan metadata for top-20 categories."""
        category_counts: dict[str, int] = {}
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Amazon 2023 'categories' is a list of category-hierarchy lists
                # (or sometimes a flat list). Use the top-level (first element)
                # as the primary category label.
                cats = m.get("categories", [])
                top_cat = _extract_top_category(cats) or m.get("main_category", "")
                if top_cat:
                    category_counts[top_cat] = category_counts.get(top_cat, 0) + 1

        self.categories = [
            c for c, _ in sorted(category_counts.items(), key=lambda x: -x[1])[:NUM_TOP_CATEGORIES]
        ]

    def _load_metadata(self, meta_path: Path) -> None:
        """Pass 2: load book metadata and build feature vectors."""
        cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        next_id = 0

        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parent_asin = m.get("parent_asin")
                if not parent_asin or parent_asin in self._asin_to_item_id:
                    continue

                # Parse fields
                cats = m.get("categories", [])
                top_cat = _extract_top_category(cats) or m.get("main_category", "")
                price = _parse_price(m.get("price"))
                avg_rating = _to_float(m.get("average_rating"))
                rating_number = _to_int(m.get("rating_number"))
                num_pages = _extract_num_pages(m.get("details", {}))

                # Build features
                features = np.zeros(NUM_FEATURES, dtype=np.float32)
                # Category one-hot
                if top_cat in cat_to_idx:
                    features[cat_to_idx[top_cat]] = 1.0
                # Price bucket
                pi = _price_bucket_index(price)
                if pi is not None:
                    features[_PRICE_SLICE.start + pi] = 1.0
                # Volume bucket
                vi = _volume_bucket_index(rating_number)
                if vi is not None:
                    features[_VOL_SLICE.start + vi] = 1.0
                # Sentiment bucket
                si = _sentiment_bucket_index(avg_rating)
                if si is not None:
                    features[_SENT_SLICE.start + si] = 1.0
                # Length indicator
                if num_pages is not None:
                    if num_pages < 200:
                        features[_LEN_SLICE.start + 0] = 1.0
                    else:
                        features[_LEN_SLICE.start + 1] = 1.0

                self._asin_to_item_id[parent_asin] = next_id
                self.items[next_id] = Book(
                    item_id=next_id,
                    parent_asin=parent_asin,
                    title=m.get("title", ""),
                    main_category=top_cat,
                    price=price,
                    average_rating=avg_rating,
                    rating_number=rating_number,
                    num_pages=num_pages,
                    features=features,
                )
                next_id += 1

    def _load_ratings(self, reviews_path: Path, val_ratio: float, test_ratio: float) -> None:
        """Pass 3: load ratings and split chronologically."""
        all_ratings: list[Rating] = []
        with open(reviews_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue

                parent_asin = r.get("parent_asin")
                user_uid = r.get("user_id")
                rating_value = _to_float(r.get("rating"))
                timestamp = _to_int(r.get("timestamp")) or 0

                if parent_asin is None or user_uid is None or rating_value is None:
                    continue
                item_id = self._asin_to_item_id.get(parent_asin)
                if item_id is None:
                    continue

                if user_uid not in self._user_uid_to_user_id:
                    new_uid = len(self._user_uid_to_user_id)
                    self._user_uid_to_user_id[user_uid] = new_uid
                    self.users[new_uid] = User(user_id=new_uid, user_uid=user_uid)
                user_id = self._user_uid_to_user_id[user_uid]

                all_ratings.append(Rating(
                    user_id=user_id,
                    item_id=item_id,
                    rating=rating_value,
                    timestamp=timestamp,
                ))

        # Chronological split: oldest for train, middle for val, newest for test.
        all_ratings.sort(key=lambda r: r.timestamp)
        n_total = len(all_ratings)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val - n_test

        self.train_ratings = all_ratings[:n_train]
        self.val_ratings = all_ratings[n_train:n_train + n_val]
        self.test_ratings = all_ratings[n_train + n_val:]

    # ── public API (duck-typed to MovieLensDataset) ───────────────

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_items(self) -> int:
        return len(self.items)

    @property
    def num_features(self) -> int:
        return NUM_FEATURES

    @property
    def all_item_ids(self) -> list[int]:
        return sorted(self.items.keys())

    @property
    def all_user_ids(self) -> list[int]:
        return sorted(self.users.keys())

    def get_item(self, item_id: int) -> Book | None:
        return self.items.get(item_id)

    def get_user(self, user_id: int) -> User | None:
        return self.users.get(user_id)

    def get_user_history(self, user_id: int) -> list[Rating]:
        return self._user_history.get(user_id, [])

    def get_item_feature_vector(self, item_id: int) -> np.ndarray:
        item = self.items.get(item_id)
        if item is None:
            return np.zeros(NUM_FEATURES, dtype=np.float32)
        return item.features


# ── parsing helpers ──────────────────────────────────────────────

def _extract_top_category(cats) -> str:
    """Extract the meaningful genre label from the 2023 categories field.

    For Kindle books the list has a fixed prefix of generic wrappers:
        ['Kindle Store', 'Kindle eBooks', <GENRE>, <SUBGENRE>?, ...]
    We want the first entry AFTER those wrappers, which is the actual
    genre ('Mystery, Thriller & Suspense', 'Romance', 'Literature & Fiction', ...).

    Falls back to the first non-wrapper string, or empty string if none.
    """
    _WRAPPERS = {"Kindle Store", "Kindle eBooks"}

    def _first_non_wrapper(items) -> str:
        for entry in items:
            if isinstance(entry, str) and entry and entry not in _WRAPPERS:
                return entry
            if isinstance(entry, list) and entry:
                inner = _first_non_wrapper(entry)
                if inner:
                    return inner
        return ""

    if isinstance(cats, str) and cats:
        return cats if cats not in _WRAPPERS else ""
    if isinstance(cats, list):
        return _first_non_wrapper(cats)
    return ""


def _parse_price(price) -> float | None:
    """Parse Amazon price field ('$9.99', '9.99', '', None, etc.)."""
    if price is None:
        return None
    if isinstance(price, (int, float)):
        return float(price)
    if isinstance(price, str):
        s = price.strip().lstrip("$").replace(",", "")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _to_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return None


def _extract_num_pages(details) -> int | None:
    """Extract page count from the details dict.

    Amazon's 'details' dict has varied keys; look for common spellings.
    """
    if not isinstance(details, dict):
        return None
    for key in ("Print length", "Print Length", "print_length", "Pages", "pages"):
        if key in details:
            val = details[key]
            if isinstance(val, (int, float)):
                return int(val)
            if isinstance(val, str):
                m = re.search(r"(\d+)", val)
                if m:
                    return int(m.group(1))
    return None


def _price_bucket_index(price: float | None) -> int | None:
    if price is None:
        return None
    for i, (lo, hi, _name) in enumerate(_PRICE_BUCKETS):
        if lo <= price < hi:
            return i
    return None


def _volume_bucket_index(rating_number: int | None) -> int | None:
    if rating_number is None:
        return None
    if rating_number < _VOL_NICHE_MAX:
        return 0
    if rating_number < _VOL_MID_MAX:
        return 1
    return 2


def _sentiment_bucket_index(avg_rating: float | None) -> int | None:
    if avg_rating is None:
        return None
    if avg_rating < 3.0:
        return 0
    if avg_rating < 4.0:
        return 1
    return 2
