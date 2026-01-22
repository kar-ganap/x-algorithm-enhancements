"""MovieLens 100K dataset loader.

Provides classes for loading and processing the MovieLens 100K dataset
for training recommendation models.

Usage:
    dataset = MovieLensDataset("data/ml-100k")
    train_ratings = dataset.train_ratings
    movie = dataset.get_movie(movie_id=1)
    user_history = dataset.get_user_history(user_id=1)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Rating:
    """A single user-movie rating."""
    user_id: int
    movie_id: int
    rating: int  # 1-5
    timestamp: int


@dataclass
class Movie:
    """Movie metadata."""
    movie_id: int
    title: str
    release_date: str
    genres: np.ndarray  # Binary vector of 19 genres

    @property
    def genre_indices(self) -> List[int]:
        """Get list of genre indices for this movie."""
        return list(np.where(self.genres > 0)[0])


@dataclass
class User:
    """User metadata."""
    user_id: int
    age: int
    gender: str
    occupation: str
    zip_code: str


@dataclass
class MovieLensDataset:
    """MovieLens 100K dataset.

    Attributes:
        data_dir: Path to ml-100k directory
        movies: Dict mapping movie_id to Movie
        users: Dict mapping user_id to User
        train_ratings: List of training ratings
        val_ratings: List of validation ratings
        test_ratings: List of test ratings
        genres: List of genre names
    """
    data_dir: Path
    movies: Dict[int, Movie] = field(default_factory=dict)
    users: Dict[int, User] = field(default_factory=dict)
    train_ratings: List[Rating] = field(default_factory=list)
    val_ratings: List[Rating] = field(default_factory=list)
    test_ratings: List[Rating] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)

    # User history cache
    _user_history: Dict[int, List[Rating]] = field(default_factory=dict)

    def __init__(self, data_dir: str | Path, val_ratio: float = 0.1):
        """Load MovieLens 100K dataset.

        Args:
            data_dir: Path to ml-100k directory
            val_ratio: Fraction of training data to use for validation
        """
        self.data_dir = Path(data_dir)
        self.movies = {}
        self.users = {}
        self.train_ratings = []
        self.val_ratings = []
        self.test_ratings = []
        self.genres = []
        self._user_history = {}

        self._load_genres()
        self._load_movies()
        self._load_users()
        self._load_ratings(val_ratio)
        self._build_user_history()

    def _load_genres(self) -> None:
        """Load genre names."""
        genre_file = self.data_dir / "u.genre"
        self.genres = []
        with open(genre_file, encoding="latin-1") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("|")
                    if len(parts) >= 2:
                        self.genres.append(parts[0])

    def _load_movies(self) -> None:
        """Load movie metadata."""
        movie_file = self.data_dir / "u.item"
        with open(movie_file, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 24:  # movie_id, title, date, video_date, url, 19 genres
                    movie_id = int(parts[0])
                    title = parts[1]
                    release_date = parts[2]
                    # Genres are in positions 5-23 (19 genres)
                    genres = np.array([int(g) for g in parts[5:24]], dtype=np.float32)

                    self.movies[movie_id] = Movie(
                        movie_id=movie_id,
                        title=title,
                        release_date=release_date,
                        genres=genres,
                    )

    def _load_users(self) -> None:
        """Load user metadata."""
        user_file = self.data_dir / "u.user"
        with open(user_file, encoding="latin-1") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 5:
                    user_id = int(parts[0])
                    self.users[user_id] = User(
                        user_id=user_id,
                        age=int(parts[1]),
                        gender=parts[2],
                        occupation=parts[3],
                        zip_code=parts[4],
                    )

    def _load_ratings(self, val_ratio: float) -> None:
        """Load ratings with train/val/test splits.

        Uses the predefined ua.base/ua.test split, then further splits
        training into train/val.
        """
        # Load training data (ua.base)
        train_file = self.data_dir / "ua.base"
        all_train = []
        with open(train_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    all_train.append(Rating(
                        user_id=int(parts[0]),
                        movie_id=int(parts[1]),
                        rating=int(parts[2]),
                        timestamp=int(parts[3]),
                    ))

        # Split train into train/val
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(all_train))
        val_size = int(len(all_train) * val_ratio)

        val_indices = set(indices[:val_size])
        for i, rating in enumerate(all_train):
            if i in val_indices:
                self.val_ratings.append(rating)
            else:
                self.train_ratings.append(rating)

        # Load test data (ua.test)
        test_file = self.data_dir / "ua.test"
        with open(test_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    self.test_ratings.append(Rating(
                        user_id=int(parts[0]),
                        movie_id=int(parts[1]),
                        rating=int(parts[2]),
                        timestamp=int(parts[3]),
                    ))

    def _build_user_history(self) -> None:
        """Build user history from training ratings."""
        self._user_history = {}
        for rating in self.train_ratings:
            if rating.user_id not in self._user_history:
                self._user_history[rating.user_id] = []
            self._user_history[rating.user_id].append(rating)

        # Sort by timestamp
        for user_id in self._user_history:
            self._user_history[user_id].sort(key=lambda r: r.timestamp)

    def get_movie(self, movie_id: int) -> Optional[Movie]:
        """Get movie by ID."""
        return self.movies.get(movie_id)

    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)

    def get_user_history(self, user_id: int) -> List[Rating]:
        """Get user's rating history sorted by timestamp."""
        return self._user_history.get(user_id, [])

    def get_movie_genre_vector(self, movie_id: int) -> np.ndarray:
        """Get binary genre vector for a movie."""
        movie = self.movies.get(movie_id)
        if movie is None:
            return np.zeros(19, dtype=np.float32)
        return movie.genres

    @property
    def num_users(self) -> int:
        """Number of unique users."""
        return len(self.users)

    @property
    def num_movies(self) -> int:
        """Number of unique movies."""
        return len(self.movies)

    @property
    def num_genres(self) -> int:
        """Number of genres."""
        return len(self.genres)

    @property
    def all_movie_ids(self) -> List[int]:
        """List of all movie IDs."""
        return list(self.movies.keys())

    @property
    def all_user_ids(self) -> List[int]:
        """List of all user IDs."""
        return list(self.users.keys())

    def get_unrated_movies(self, user_id: int) -> List[int]:
        """Get movies not rated by user (for negative sampling)."""
        rated = {r.movie_id for r in self.get_user_history(user_id)}
        return [m for m in self.all_movie_ids if m not in rated]

    def sample_negative_movies(
        self,
        user_id: int,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """Sample n movies not rated by user."""
        if rng is None:
            rng = np.random.default_rng()

        unrated = self.get_unrated_movies(user_id)
        if len(unrated) <= n:
            return unrated

        indices = rng.choice(len(unrated), size=n, replace=False)
        return [unrated[i] for i in indices]

    def __repr__(self) -> str:
        return (
            f"MovieLensDataset(\n"
            f"  data_dir={self.data_dir},\n"
            f"  num_users={self.num_users},\n"
            f"  num_movies={self.num_movies},\n"
            f"  num_genres={self.num_genres},\n"
            f"  train_ratings={len(self.train_ratings)},\n"
            f"  val_ratings={len(self.val_ratings)},\n"
            f"  test_ratings={len(self.test_ratings)},\n"
            f")"
        )


def load_movielens(data_dir: str = "data/ml-100k") -> MovieLensDataset:
    """Convenience function to load MovieLens dataset.

    Args:
        data_dir: Path to ml-100k directory

    Returns:
        MovieLensDataset instance
    """
    return MovieLensDataset(data_dir)
