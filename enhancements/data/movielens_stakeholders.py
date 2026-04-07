"""MovieLens multi-stakeholder preference generation.

Defines three stakeholder utility functions over 19-dim genre vectors
and generates preference pairs from real MovieLens ratings.

Stakeholders:
- User: prefers genres they historically rate highly
- Platform: prefers popular, high-engagement genres
- Diversity: penalizes genre concentration, rewards breadth
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Import MovieLensDataset directly to avoid __init__.py import chain
# that triggers Phoenix/grok dependencies
_movielens_path = Path(__file__).parent / "movielens.py"
_spec = importlib.util.spec_from_file_location("_movielens", _movielens_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MovieLensDataset = _mod.MovieLensDataset


NUM_GENRES = 19


def compute_user_genre_weights(dataset: MovieLensDataset) -> np.ndarray:
    """Compute user stakeholder weights from aggregate rating history.

    For each genre, computes the mean rating (centered at 3.0) across all
    users who rated movies in that genre. Genres with higher-than-average
    ratings get positive weight; lower get negative.

    Returns:
        [19] array normalized to [-1, 1].
    """
    genre_rating_sums = np.zeros(NUM_GENRES, dtype=np.float64)
    genre_rating_counts = np.zeros(NUM_GENRES, dtype=np.float64)

    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_rating_sums[g] += rating.rating - 3.0
                genre_rating_counts[g] += 1

    mask = genre_rating_counts > 0
    weights = np.zeros(NUM_GENRES, dtype=np.float64)
    weights[mask] = genre_rating_sums[mask] / genre_rating_counts[mask]

    # Normalize to [-1, 1]
    max_abs = np.max(np.abs(weights))
    if max_abs > 0:
        weights /= max_abs

    return weights.astype(np.float32)


def compute_platform_genre_weights(dataset: MovieLensDataset) -> np.ndarray:
    """Compute platform stakeholder weights from popularity and engagement.

    Platform utility = avg_rating(genre) * log2(num_ratings(genre)).
    Popular genres with high ratings get the most weight.

    Returns:
        [19] array normalized to [0, 1].
    """
    genre_rating_sums = np.zeros(NUM_GENRES, dtype=np.float64)
    genre_rating_counts = np.zeros(NUM_GENRES, dtype=np.float64)

    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_rating_sums[g] += rating.rating
                genre_rating_counts[g] += 1

    mask = genre_rating_counts > 0
    avg_rating = np.zeros(NUM_GENRES, dtype=np.float64)
    avg_rating[mask] = genre_rating_sums[mask] / genre_rating_counts[mask]

    log_count = np.zeros(NUM_GENRES, dtype=np.float64)
    log_count[mask] = np.log2(genre_rating_counts[mask] + 1)

    weights = avg_rating * log_count

    # Normalize to [0, 1]
    max_val = np.max(weights)
    if max_val > 0:
        weights /= max_val

    return weights.astype(np.float32)


def compute_diversity_genre_weights(dataset: MovieLensDataset) -> np.ndarray:
    """Compute diversity stakeholder weights that penalize genre concentration.

    Weight = -fraction_of_ratings_in_genre + (1 / num_active_genres).
    Popular genres get negative weight; rare genres get positive weight.
    Weights sum to approximately 0 (zero-sum between genres).

    Returns:
        [19] array.
    """
    genre_counts = np.zeros(NUM_GENRES, dtype=np.float64)

    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_counts[g] += 1

    total = np.sum(genre_counts)
    num_active = np.sum(genre_counts > 0)

    if total == 0 or num_active == 0:
        return np.zeros(NUM_GENRES, dtype=np.float32)

    fractions = genre_counts / total
    uniform = 1.0 / num_active

    weights = np.where(genre_counts > 0, -fractions + uniform, 0.0)

    # Normalize so max absolute value is 1
    max_abs = np.max(np.abs(weights))
    if max_abs > 0:
        weights /= max_abs

    return weights.astype(np.float32)


def build_stakeholder_configs(
    dataset: MovieLensDataset,
) -> dict[str, np.ndarray]:
    """Build all three stakeholder genre weight vectors.

    Returns:
        Dict with keys "user", "platform", "diversity", each mapping
        to a [19] weight vector.
    """
    return {
        "user": compute_user_genre_weights(dataset),
        "platform": compute_platform_genre_weights(dataset),
        "diversity": compute_diversity_genre_weights(dataset),
    }


def generate_movielens_content_pool(
    dataset: MovieLensDataset,
    min_ratings: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build content pool with genre features from MovieLens movies.

    Each movie's feature vector = genre_binary_vec * (avg_rating / 5.0).
    This weights genre membership by how well-rated the movie is overall.

    Args:
        dataset: Loaded MovieLens dataset.
        min_ratings: Minimum number of ratings for a movie to be included.
        seed: Random seed (for consistent ordering).

    Returns:
        content_features: [M, 19] genre feature vectors.
        content_genres: [M] primary genre index per movie (for diversity).
    """
    # Compute per-movie average rating
    movie_rating_sums: dict[int, float] = {}
    movie_rating_counts: dict[int, int] = {}
    for rating in dataset.train_ratings:
        mid = rating.movie_id
        movie_rating_sums[mid] = movie_rating_sums.get(mid, 0.0) + rating.rating
        movie_rating_counts[mid] = movie_rating_counts.get(mid, 0) + 1

    # Filter to movies with enough ratings
    eligible_ids = sorted(
        mid for mid, count in movie_rating_counts.items()
        if count >= min_ratings and mid in dataset.movies
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_ids)

    features = np.zeros((len(eligible_ids), NUM_GENRES), dtype=np.float32)
    genres = np.zeros(len(eligible_ids), dtype=np.int32)

    for i, mid in enumerate(eligible_ids):
        movie = dataset.movies[mid]
        avg_rating = movie_rating_sums[mid] / movie_rating_counts[mid]
        features[i] = movie.genres * (avg_rating / 5.0)
        # Primary genre = first genre with value 1
        genre_indices = np.where(movie.genres > 0)[0]
        genres[i] = genre_indices[0] if len(genre_indices) > 0 else 0

    return features, genres


def generate_movielens_preferences(
    content_features: np.ndarray,
    stakeholder_weights: np.ndarray,
    n_pairs: int,
    seed: int,
    noise_std: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs for one stakeholder from MovieLens data.

    For each pair: sample two movies, score with stakeholder weights,
    add noise, label higher-score movie as preferred.

    Args:
        content_features: [M, D] feature vectors per movie.
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

    # Score all content
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

    Args:
        probs_pref: [N, D] preferred item features.
        probs_rej: [N, D] rejected item features.
        eval_fraction: Fraction of pairs to hold out for evaluation.
        seed: Random seed for reproducible splits.

    Returns:
        (train_pref, train_rej, eval_pref, eval_rej)
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
    """Compute label disagreement rate between two stakeholders.

    Samples random pairs and checks how often the two stakeholders
    disagree on which item is preferred.

    Returns:
        Fraction of pairs where stakeholders disagree (0.0 to 1.0).
    """
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


def generate_movielens_content_pool_temporal(
    dataset: MovieLensDataset,
    before_timestamp: int,
    min_ratings: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Build content pool from movies rated before a timestamp cutoff.

    Only includes movies where all counted ratings are before the cutoff.
    Uses the same genre-feature representation as generate_movielens_content_pool.

    Args:
        dataset: Loaded MovieLens dataset.
        before_timestamp: Unix timestamp cutoff (exclusive).
        min_ratings: Minimum ratings before cutoff for inclusion.
        seed: Random seed for ordering.

    Returns:
        content_features: [M, 19] genre feature vectors.
        content_genres: [M] primary genre index per movie.
    """
    movie_rating_sums: dict[int, float] = {}
    movie_rating_counts: dict[int, int] = {}
    for rating in dataset.train_ratings:
        if rating.timestamp >= before_timestamp:
            continue
        mid = rating.movie_id
        movie_rating_sums[mid] = movie_rating_sums.get(mid, 0.0) + rating.rating
        movie_rating_counts[mid] = movie_rating_counts.get(mid, 0) + 1

    eligible_ids = sorted(
        mid for mid, count in movie_rating_counts.items()
        if count >= min_ratings and mid in dataset.movies
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(eligible_ids)

    features = np.zeros((len(eligible_ids), NUM_GENRES), dtype=np.float32)
    genres = np.zeros(len(eligible_ids), dtype=np.int32)

    for i, mid in enumerate(eligible_ids):
        movie = dataset.movies[mid]
        avg_rating = movie_rating_sums[mid] / movie_rating_counts[mid]
        features[i] = movie.genres * (avg_rating / 5.0)
        genre_indices = np.where(movie.genres > 0)[0]
        genres[i] = genre_indices[0] if len(genre_indices) > 0 else 0

    return features, genres


def compute_user_genre_weights_for_group(
    dataset: MovieLensDataset,
    user_ids: list[int] | set[int],
) -> np.ndarray:
    """Compute user genre weights for a specific subset of users.

    Same logic as compute_user_genre_weights but restricted to ratings
    from the given user IDs.

    Returns:
        [19] array normalized to [-1, 1].
    """
    user_set = set(user_ids)
    genre_rating_sums = np.zeros(NUM_GENRES, dtype=np.float64)
    genre_rating_counts = np.zeros(NUM_GENRES, dtype=np.float64)

    for rating in dataset.train_ratings:
        if rating.user_id not in user_set:
            continue
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                genre_rating_sums[g] += rating.rating - 3.0
                genre_rating_counts[g] += 1

    mask = genre_rating_counts > 0
    weights = np.zeros(NUM_GENRES, dtype=np.float64)
    weights[mask] = genre_rating_sums[mask] / genre_rating_counts[mask]

    max_abs = np.max(np.abs(weights))
    if max_abs > 0:
        weights /= max_abs

    return weights.astype(np.float32)


def get_user_genre_groups(
    dataset: MovieLensDataset,
    min_group_size: int = 50,
) -> dict[str, list[int]]:
    """Group users by their top-rated genre.

    For each user, finds the genre with highest mean rating.
    Returns groups with at least min_group_size members.

    Returns:
        Dict mapping genre_name -> list of user_ids.
    """
    genre_names = dataset.genres if dataset.genres else [f"genre_{i}" for i in range(NUM_GENRES)]

    user_genre_ratings: dict[int, list[list[int]]] = {}
    for rating in dataset.train_ratings:
        movie = dataset.movies.get(rating.movie_id)
        if movie is None:
            continue
        if rating.user_id not in user_genre_ratings:
            user_genre_ratings[rating.user_id] = [[] for _ in range(NUM_GENRES)]
        for g in range(NUM_GENRES):
            if movie.genres[g] > 0:
                user_genre_ratings[rating.user_id][g].append(rating.rating)

    groups: dict[str, list[int]] = {}
    for user_id, genre_lists in user_genre_ratings.items():
        means = [np.mean(gl) if gl else 0.0 for gl in genre_lists]
        top_genre = int(np.argmax(means))
        name = genre_names[top_genre] if top_genre < len(genre_names) else f"genre_{top_genre}"
        groups.setdefault(name, []).append(user_id)

    return {name: uids for name, uids in groups.items() if len(uids) >= min_group_size}
