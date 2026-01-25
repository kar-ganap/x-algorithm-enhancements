"""Synthetic Twitter-like data generator.

Generates structured data with planted patterns for verifying that
Phoenix can learn expected behaviors. Not actual tweet text - just
structured engagement data.

Usage:
    generator = SyntheticTwitterGenerator(seed=42)
    dataset = generator.generate(
        num_users=1000,
        num_posts=50000,
        num_engagements=200000,
    )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from enhancements.data.ground_truth import (
    UserArchetype,
    ContentTopic,
    ARCHETYPE_DISTRIBUTION,
    TOPIC_DISTRIBUTION,
    AUTHORS_PER_TOPIC,
    ActionProbabilities,
    SyntheticUser,
    SyntheticPost,
    SyntheticEngagement,
    SyntheticAuthor,
    get_engagement_probs,
)


@dataclass
class SyntheticTwitterDataset:
    """Container for synthetic Twitter-like data.

    All data is structured (not text) with known ground truth patterns.
    """
    users: List[SyntheticUser]
    authors: List[SyntheticAuthor]
    posts: List[SyntheticPost]
    engagements: List[SyntheticEngagement]

    # Index lookups for efficient access
    _users_by_id: Dict[int, SyntheticUser] = field(default_factory=dict)
    _users_by_archetype: Dict[UserArchetype, List[SyntheticUser]] = field(default_factory=dict)
    _posts_by_id: Dict[int, SyntheticPost] = field(default_factory=dict)
    _posts_by_topic: Dict[ContentTopic, List[SyntheticPost]] = field(default_factory=dict)
    _posts_by_author: Dict[int, List[SyntheticPost]] = field(default_factory=dict)
    _authors_by_id: Dict[int, SyntheticAuthor] = field(default_factory=dict)
    _authors_by_topic: Dict[ContentTopic, List[SyntheticAuthor]] = field(default_factory=dict)
    _engagements_by_user: Dict[int, List[SyntheticEngagement]] = field(default_factory=dict)

    def __post_init__(self):
        """Build index lookups."""
        self._build_indices()

    def _build_indices(self):
        """Build all index lookups for fast access."""
        # Users
        self._users_by_id = {u.user_id: u for u in self.users}
        self._users_by_archetype = {a: [] for a in UserArchetype}
        for u in self.users:
            self._users_by_archetype[u.archetype].append(u)

        # Authors
        self._authors_by_id = {a.author_id: a for a in self.authors}
        self._authors_by_topic = {t: [] for t in ContentTopic}
        for a in self.authors:
            self._authors_by_topic[a.primary_topic].append(a)

        # Posts
        self._posts_by_id = {p.post_id: p for p in self.posts}
        self._posts_by_topic = {t: [] for t in ContentTopic}
        self._posts_by_author = {}
        for p in self.posts:
            self._posts_by_topic[p.topic].append(p)
            if p.author_id not in self._posts_by_author:
                self._posts_by_author[p.author_id] = []
            self._posts_by_author[p.author_id].append(p)

        # Engagements
        self._engagements_by_user = {}
        for e in self.engagements:
            if e.user_id not in self._engagements_by_user:
                self._engagements_by_user[e.user_id] = []
            self._engagements_by_user[e.user_id].append(e)

    def get_user(self, user_id: int) -> Optional[SyntheticUser]:
        """Get user by ID."""
        return self._users_by_id.get(user_id)

    def get_users_by_archetype(self, archetype: UserArchetype) -> List[SyntheticUser]:
        """Get all users of a given archetype."""
        return self._users_by_archetype.get(archetype, [])

    def get_post(self, post_id: int) -> Optional[SyntheticPost]:
        """Get post by ID."""
        return self._posts_by_id.get(post_id)

    def get_posts_by_topic(self, topic: ContentTopic) -> List[SyntheticPost]:
        """Get all posts of a given topic."""
        return self._posts_by_topic.get(topic, [])

    def get_posts_by_author(self, author_id: int) -> List[SyntheticPost]:
        """Get all posts by a given author."""
        return self._posts_by_author.get(author_id, [])

    def get_author(self, author_id: int) -> Optional[SyntheticAuthor]:
        """Get author by ID."""
        return self._authors_by_id.get(author_id)

    def get_authors_by_topic(self, topic: ContentTopic) -> List[SyntheticAuthor]:
        """Get all authors for a given topic."""
        return self._authors_by_topic.get(topic, [])

    def get_user_engagements(self, user_id: int) -> List[SyntheticEngagement]:
        """Get all engagements for a user."""
        return self._engagements_by_user.get(user_id, [])

    def get_user_history(self, user_id: int) -> List[SyntheticEngagement]:
        """Get user's engagement history sorted by timestamp."""
        engagements = self.get_user_engagements(user_id)
        return sorted(engagements, key=lambda e: e.timestamp)

    @property
    def num_users(self) -> int:
        return len(self.users)

    @property
    def num_posts(self) -> int:
        return len(self.posts)

    @property
    def num_authors(self) -> int:
        return len(self.authors)

    @property
    def num_engagements(self) -> int:
        return len(self.engagements)

    @property
    def all_user_ids(self) -> List[int]:
        return list(self._users_by_id.keys())

    @property
    def all_post_ids(self) -> List[int]:
        return list(self._posts_by_id.keys())

    @property
    def all_author_ids(self) -> List[int]:
        return list(self._authors_by_id.keys())

    def __repr__(self) -> str:
        archetype_counts = {a.value: len(self._users_by_archetype[a])
                           for a in UserArchetype}
        topic_counts = {t.value: len(self._posts_by_topic[t])
                       for t in ContentTopic}
        return (
            f"SyntheticTwitterDataset(\n"
            f"  users={self.num_users},\n"
            f"  posts={self.num_posts},\n"
            f"  authors={self.num_authors},\n"
            f"  engagements={self.num_engagements},\n"
            f"  archetype_distribution={archetype_counts},\n"
            f"  topic_distribution={topic_counts},\n"
            f")"
        )


class SyntheticTwitterGenerator:
    """Generator for synthetic Twitter-like data with planted patterns."""

    def __init__(self, seed: int = 42):
        """Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        num_users: int = 1000,
        num_posts: int = 50000,
        num_engagements: int = 200000,
    ) -> SyntheticTwitterDataset:
        """Generate a synthetic dataset.

        Args:
            num_users: Number of users to generate
            num_posts: Number of posts to generate
            num_engagements: Number of engagement events to generate

        Returns:
            SyntheticTwitterDataset with all data
        """
        # Step 1: Generate users with archetypes
        users = self._generate_users(num_users)

        # Step 2: Generate authors (content creators)
        authors = self._generate_authors()

        # Step 3: Generate posts by authors
        posts = self._generate_posts(num_posts, authors)

        # Step 4: Generate engagements based on ground truth rules
        engagements = self._generate_engagements(num_engagements, users, posts)

        return SyntheticTwitterDataset(
            users=users,
            authors=authors,
            posts=posts,
            engagements=engagements,
        )

    def _generate_users(self, num_users: int) -> List[SyntheticUser]:
        """Generate users with archetype distribution."""
        users = []
        archetypes = list(ARCHETYPE_DISTRIBUTION.keys())
        probs = list(ARCHETYPE_DISTRIBUTION.values())

        for user_id in range(1, num_users + 1):
            # Use index to avoid numpy type issues
            arch_idx = self.rng.choice(len(archetypes), p=probs)
            archetype = archetypes[arch_idx]
            users.append(SyntheticUser(user_id=user_id, archetype=archetype))

        return users

    def _generate_authors(self) -> List[SyntheticAuthor]:
        """Generate authors for each topic."""
        authors = []
        author_id = 1

        for topic in ContentTopic:
            num_authors = AUTHORS_PER_TOPIC[topic]
            for _ in range(num_authors):
                authors.append(SyntheticAuthor(
                    author_id=author_id,
                    primary_topic=topic,
                ))
                author_id += 1

        return authors

    def _generate_posts(
        self,
        num_posts: int,
        authors: List[SyntheticAuthor],
    ) -> List[SyntheticPost]:
        """Generate posts with topic distribution."""
        posts = []

        # Group authors by topic for sampling
        authors_by_topic = {t: [] for t in ContentTopic}
        for author in authors:
            authors_by_topic[author.primary_topic].append(author)

        topics = list(TOPIC_DISTRIBUTION.keys())
        probs = list(TOPIC_DISTRIBUTION.values())

        for post_id in range(1, num_posts + 1):
            # Sample topic according to distribution (use index to avoid numpy type issues)
            topic_idx = self.rng.choice(len(topics), p=probs)
            topic = topics[topic_idx]

            # Sample author from that topic
            topic_authors = authors_by_topic[topic]
            author_idx = self.rng.integers(0, len(topic_authors))
            author = topic_authors[author_idx]

            posts.append(SyntheticPost(
                post_id=post_id,
                author_id=author.author_id,
                topic=topic,
            ))

        return posts

    def _generate_engagements(
        self,
        num_engagements: int,
        users: List[SyntheticUser],
        posts: List[SyntheticPost],
    ) -> List[SyntheticEngagement]:
        """Generate engagement events based on ground truth rules.

        Uses the engagement probability rules to determine which actions
        a user takes on a post.
        """
        engagements = []

        # Create index for posts by topic for efficient sampling
        posts_by_topic = {t: [] for t in ContentTopic}
        for p in posts:
            posts_by_topic[p.topic].append(p)

        topic_weights = np.array([TOPIC_DISTRIBUTION[t] for t in ContentTopic])
        topic_weights = topic_weights / topic_weights.sum()
        topics = list(ContentTopic)

        # Distribute engagements across users
        engagements_per_user = num_engagements // len(users)
        extra = num_engagements % len(users)

        timestamp = 0
        for user in users:
            # Determine number of engagements for this user
            n = engagements_per_user + (1 if user.user_id <= extra else 0)

            for _ in range(n):
                # Sample a topic weighted by distribution
                topic_idx = self.rng.choice(len(topics), p=topic_weights)
                topic = topics[topic_idx]

                # Sample a post from that topic
                topic_posts = posts_by_topic[topic]
                post_idx = self.rng.integers(0, len(topic_posts))
                post = topic_posts[post_idx]

                # Get engagement probabilities for (archetype, topic)
                probs = get_engagement_probs(user.archetype, topic)

                # Sample actions based on probabilities
                actions = self._sample_actions(probs)

                # Only create engagement if at least one action was taken
                if any(v > 0 for v in actions.values()):
                    engagements.append(SyntheticEngagement(
                        user_id=user.user_id,
                        post_id=post.post_id,
                        actions=actions,
                        timestamp=timestamp,
                    ))
                    timestamp += 1

        return engagements

    def _sample_actions(self, probs: ActionProbabilities) -> Dict[str, float]:
        """Sample actions based on probabilities.

        Each action is independently sampled according to its probability.

        Args:
            probs: ActionProbabilities for this (archetype, topic) pair

        Returns:
            Dict mapping action name to 1.0 (taken) or 0.0 (not taken)
        """
        actions = {}
        prob_dict = probs.to_dict()

        for action_name, prob in prob_dict.items():
            if self.rng.random() < prob:
                actions[action_name] = 1.0
            else:
                actions[action_name] = 0.0

        return actions


def create_train_val_test_split(
    dataset: SyntheticTwitterDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[SyntheticEngagement], List[SyntheticEngagement], List[SyntheticEngagement]]:
    """Split engagements into train/val/test sets.

    Uses chronological split: earliest engagements for train,
    middle for val, latest for test.

    Args:
        dataset: The synthetic dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed (unused for chronological split)

    Returns:
        Tuple of (train_engagements, val_engagements, test_engagements)
    """
    # Sort by timestamp
    sorted_engagements = sorted(dataset.engagements, key=lambda e: e.timestamp)

    n = len(sorted_engagements)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = sorted_engagements[:train_end]
    val = sorted_engagements[train_end:val_end]
    test = sorted_engagements[val_end:]

    return train, val, test


def load_or_generate_dataset(
    data_dir: str = "data/synthetic_twitter",
    num_users: int = 1000,
    num_posts: int = 50000,
    num_engagements: int = 200000,
    seed: int = 42,
    force_regenerate: bool = False,
) -> SyntheticTwitterDataset:
    """Load cached dataset or generate new one.

    Args:
        data_dir: Directory for caching
        num_users: Number of users if generating
        num_posts: Number of posts if generating
        num_engagements: Number of engagements if generating
        seed: Random seed
        force_regenerate: If True, always regenerate

    Returns:
        SyntheticTwitterDataset
    """
    import os
    import pickle

    cache_path = os.path.join(data_dir, "dataset.pkl")

    if not force_regenerate and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Generate new dataset
    generator = SyntheticTwitterGenerator(seed=seed)
    dataset = generator.generate(
        num_users=num_users,
        num_posts=num_posts,
        num_engagements=num_engagements,
    )

    # Cache it
    os.makedirs(data_dir, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset
