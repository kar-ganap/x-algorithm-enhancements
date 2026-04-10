"""MIND (Microsoft News Dataset) loader.

Duck-typed to the MovieLensDataset interface used by experiment scripts:
- attributes: train_ratings, val_ratings, test_ratings, items, users, categories
- Rating dataclass with (user_id, item_id, rating, timestamp)
- Items (NewsArticle) carry feature vectors for stakeholder utility computation

For MIND, "rating" is interpreted as CLICK (1) or IMPRESSION-WITHOUT-CLICK (0),
derived from behaviors.tsv. Timestamps come from the Time column.

Feature vector (35-dim):
    - 17 top-level categories (binary one-hot over observed top-17 categories)
    - 14 top subcategories (binary one-hot)
    - 4 quality signals (title length, abstract length, entity count, subcategory-is-top)

Usage:
    dataset = MINDDataset("data/mind-small")
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# Fixed dimensionality chosen in Phase A planning.
NUM_TOP_CATEGORIES = 17
NUM_TOP_SUBCATEGORIES = 14
NUM_QUALITY_SIGNALS = 4  # title_len, abstract_len, entity_count, (reserved)
NUM_FEATURES = NUM_TOP_CATEGORIES + NUM_TOP_SUBCATEGORIES + NUM_QUALITY_SIGNALS  # 35


@dataclass
class Rating:
    """A single user-item interaction.

    rating: 1.0 if the user clicked, 0.0 if the article was impressed but not clicked.
    """
    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass
class NewsArticle:
    """MIND news article with derived feature vector."""
    item_id: int        # integer surrogate; MIND's native NewsID is a string
    news_id: str        # original "NxxxxxxX" id
    category: str
    subcategory: str
    title: str
    abstract: str
    n_title_entities: int
    n_abstract_entities: int
    features: np.ndarray  # [NUM_FEATURES] float32

    @property
    def feature_indices(self) -> list[int]:
        return list(np.where(self.features > 0)[0])


@dataclass
class User:
    """MIND user (minimal)."""
    user_id: int         # integer surrogate
    user_uid: str        # original "UxxxxxX" id


@dataclass
class MINDDataset:
    """Microsoft News Dataset (MIND) loader.

    Duck-types the MovieLensDataset interface: experiment scripts that
    read `train_ratings`, `val_ratings`, `test_ratings`, `items` will work.
    """
    data_dir: Path
    items: dict[int, NewsArticle] = field(default_factory=dict)
    users: dict[int, User] = field(default_factory=dict)
    train_ratings: list[Rating] = field(default_factory=list)
    val_ratings: list[Rating] = field(default_factory=list)
    test_ratings: list[Rating] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)       # top-17
    subcategories: list[str] = field(default_factory=list)    # top-14

    _user_history: dict[int, list[Rating]] = field(default_factory=dict)
    _news_id_to_item_id: dict[str, int] = field(default_factory=dict)
    _user_uid_to_user_id: dict[str, int] = field(default_factory=dict)

    def __init__(self, data_dir: str | Path, val_ratio: float = 0.1):
        """Load MIND-small (train + dev splits).

        Args:
            data_dir: Path to data/mind-small/ containing train/ and dev/ subdirs.
            val_ratio: Fraction of train impressions to hold out for validation
                (ignored if dev/ split is present — we use dev as val).
        """
        self.data_dir = Path(data_dir)
        self.items = {}
        self.users = {}
        self.train_ratings = []
        self.val_ratings = []
        self.test_ratings = []
        self.categories = []
        self.subcategories = []
        self._user_history = {}
        self._news_id_to_item_id = {}
        self._user_uid_to_user_id = {}

        train_dir = self.data_dir / "train"
        dev_dir = self.data_dir / "dev"
        if not train_dir.exists():
            raise FileNotFoundError(f"MIND train directory not found: {train_dir}")

        # Pass 1: load all news articles from train (and dev if present),
        # determine top categories/subcategories, build feature vectors.
        self._load_news(train_dir, dev_dir if dev_dir.exists() else None)

        # Pass 2: load behaviors (impressions) and build ratings.
        self._load_behaviors(train_dir, split="train")
        if dev_dir.exists():
            self._load_behaviors(dev_dir, split="val")

        # Precompute user history cache.
        for rating in self.train_ratings:
            self._user_history.setdefault(rating.user_id, []).append(rating)

    # ── loading helpers ────────────────────────────────────────────

    def _load_news(self, train_dir: Path, dev_dir: Path | None) -> None:
        """Load news.tsv and compute global feature space + articles."""
        # Column order: NewsID, Category, SubCategory, Title, Abstract, URL, TitleEntities, AbstractEntities
        sources = [train_dir / "news.tsv"]
        if dev_dir is not None:
            sources.append(dev_dir / "news.tsv")

        category_counts: dict[str, int] = {}
        subcategory_counts: dict[str, int] = {}
        raw_articles: dict[str, dict] = {}

        for src in sources:
            if not src.exists():
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 8:
                        continue
                    news_id, category, subcategory, title, abstract, _url, title_ents, abstract_ents = parts[:8]
                    if news_id in raw_articles:
                        continue
                    raw_articles[news_id] = {
                        "news_id": news_id,
                        "category": category,
                        "subcategory": subcategory,
                        "title": title,
                        "abstract": abstract,
                        "title_ents": title_ents,
                        "abstract_ents": abstract_ents,
                    }
                    category_counts[category] = category_counts.get(category, 0) + 1
                    subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1

        # Choose top-17 categories and top-14 subcategories.
        self.categories = [
            c for c, _ in sorted(category_counts.items(), key=lambda x: -x[1])[:NUM_TOP_CATEGORIES]
        ]
        self.subcategories = [
            s for s, _ in sorted(subcategory_counts.items(), key=lambda x: -x[1])[:NUM_TOP_SUBCATEGORIES]
        ]
        cat_to_idx = {c: i for i, c in enumerate(self.categories)}
        sub_to_idx = {s: i for i, s in enumerate(self.subcategories)}

        # Length normalization: use 95th percentile of title/abstract word counts.
        title_lens = np.array([len(a["title"].split()) for a in raw_articles.values()], dtype=np.float32)
        abstract_lens = np.array([len(a["abstract"].split()) for a in raw_articles.values()], dtype=np.float32)
        title_p95 = float(np.percentile(title_lens, 95)) if len(title_lens) else 1.0
        abstract_p95 = float(np.percentile(abstract_lens, 95)) if len(abstract_lens) else 1.0
        if title_p95 <= 0:
            title_p95 = 1.0
        if abstract_p95 <= 0:
            abstract_p95 = 1.0

        # Build NewsArticle objects with 35-dim feature vectors.
        for idx, (news_id, a) in enumerate(sorted(raw_articles.items())):
            features = np.zeros(NUM_FEATURES, dtype=np.float32)
            # Top-level category one-hot
            if a["category"] in cat_to_idx:
                features[cat_to_idx[a["category"]]] = 1.0
            # Subcategory one-hot
            if a["subcategory"] in sub_to_idx:
                features[NUM_TOP_CATEGORIES + sub_to_idx[a["subcategory"]]] = 1.0

            # Quality signals (normalized to [0, 1])
            qstart = NUM_TOP_CATEGORIES + NUM_TOP_SUBCATEGORIES
            title_len_norm = min(1.0, len(a["title"].split()) / title_p95)
            abstract_len_norm = min(1.0, len(a["abstract"].split()) / abstract_p95)
            n_title_ents = a["title_ents"].count("{") if a["title_ents"] else 0
            n_abstract_ents = a["abstract_ents"].count("{") if a["abstract_ents"] else 0
            entity_count_norm = min(1.0, (n_title_ents + n_abstract_ents) / 10.0)
            # 4th slot reserved for "subcategory-in-top" flag (redundant but keeps
            # dim = 35 as planned in Phase A spec).
            features[qstart + 0] = title_len_norm
            features[qstart + 1] = abstract_len_norm
            features[qstart + 2] = entity_count_norm
            features[qstart + 3] = 1.0 if a["subcategory"] in sub_to_idx else 0.0

            self.items[idx] = NewsArticle(
                item_id=idx,
                news_id=news_id,
                category=a["category"],
                subcategory=a["subcategory"],
                title=a["title"],
                abstract=a["abstract"],
                n_title_entities=n_title_ents,
                n_abstract_entities=n_abstract_ents,
                features=features,
            )
            self._news_id_to_item_id[news_id] = idx

    def _load_behaviors(self, split_dir: Path, split: str) -> None:
        """Load behaviors.tsv and emit Rating records.

        Column order: ImpressionID, UserID, Time, History, Impressions.
        Impressions is a space-separated list of "NewsID-0" or "NewsID-1".
        We emit one Rating per impression (click=1.0, no-click=0.0).
        """
        path = split_dir / "behaviors.tsv"
        if not path.exists():
            return

        target_list = self.train_ratings if split == "train" else self.val_ratings

        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                _impression_id, user_uid, time_str, _history, impressions = parts[:5]
                # Parse time: "11/13/2019 8:36:57 AM" format.
                timestamp = _parse_mind_time(time_str)
                # Assign integer user id.
                if user_uid not in self._user_uid_to_user_id:
                    new_id = len(self._user_uid_to_user_id)
                    self._user_uid_to_user_id[user_uid] = new_id
                    self.users[new_id] = User(user_id=new_id, user_uid=user_uid)
                user_id = self._user_uid_to_user_id[user_uid]

                for token in impressions.split():
                    if "-" not in token:
                        continue
                    news_id, label = token.rsplit("-", 1)
                    item_id = self._news_id_to_item_id.get(news_id)
                    if item_id is None:
                        continue
                    rating = 1.0 if label == "1" else 0.0
                    target_list.append(Rating(
                        user_id=user_id,
                        item_id=item_id,
                        rating=rating,
                        timestamp=timestamp,
                    ))

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

    def get_item(self, item_id: int) -> NewsArticle | None:
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


def _parse_mind_time(time_str: str) -> int:
    """Parse MIND timestamp 'M/D/YYYY H:MM:SS AM/PM' to unix seconds.
    On failure, returns 0."""
    try:
        from datetime import datetime
        dt = datetime.strptime(time_str.strip(), "%m/%d/%Y %I:%M:%S %p")
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return 0
