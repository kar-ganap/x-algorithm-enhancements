"""Adapter to convert MovieLens data to Phoenix format.

Maps MovieLens users and movies to Phoenix's RecsysBatch and RecsysEmbeddings.

Usage:
    from enhancements.data.movielens_adapter import MovieLensPhoenixAdapter

    adapter = MovieLensPhoenixAdapter(dataset, model_config)
    batch, embeddings, labels = adapter.get_training_batch(batch_size=32)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
import numpy as np

from phoenix.recsys_model import (
    PhoenixModelConfig,
    RecsysBatch,
    RecsysEmbeddings,
)

from enhancements.data.movielens import MovieLensDataset, Rating


# Type alias for embedding parameters
EmbeddingParams = Dict[str, jnp.ndarray]


@dataclass
class MovieLensPhoenixAdapter:
    """Adapter to convert MovieLens data to Phoenix format.

    Creates RecsysBatch and RecsysEmbeddings from MovieLens users and movies.

    Mapping:
    - user_hashes: [user_id] (single hash)
    - post_hashes: [movie_id] (single hash)
    - author_hashes: [0] (dummy - movies don't have authors)
    - history_actions: rating converted to one-hot over action indices
    - embeddings: Generated from genre vectors via learned projection
    """

    dataset: MovieLensDataset
    model_config: PhoenixModelConfig
    emb_size: int = 128
    history_len: int = 32
    num_candidates: int = 8
    rng: np.random.Generator = None

    # Learned embedding projections (set during training)
    genre_projection: Optional[np.ndarray] = None  # [19, emb_size]
    user_embedding_table: Optional[np.ndarray] = None  # [num_users+1, emb_size]

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(42)

        self.emb_size = self.model_config.emb_size
        self.history_len = self.model_config.history_seq_len
        self.num_candidates = self.model_config.candidate_seq_len

        # Initialize random projections (will be learned during training)
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embedding projections."""
        # Genre to embedding projection
        # 19 genres -> emb_size
        self.genre_projection = self.rng.normal(
            0, 0.1, size=(19, self.emb_size)
        ).astype(np.float32)

        # User embedding table (user_id 1-943 maps to index 1-943)
        self.user_embedding_table = self.rng.normal(
            0, 0.1, size=(self.dataset.num_users + 1, self.emb_size)
        ).astype(np.float32)

        # Precompute genre vectors for all movies (for efficient lookup)
        self._movie_genre_vectors = np.zeros((self.dataset.num_movies + 1, 19), dtype=np.float32)
        for movie_id in range(1, self.dataset.num_movies + 1):
            self._movie_genre_vectors[movie_id] = self.dataset.get_movie_genre_vector(movie_id)

    def get_embedding_params(self) -> EmbeddingParams:
        """Get embedding parameters as a dict for optimization.

        Returns:
            Dict with 'genre_projection' and 'user_embeddings' arrays
        """
        return {
            "genre_projection": jnp.array(self.genre_projection),
            "user_embeddings": jnp.array(self.user_embedding_table),
        }

    def set_embedding_params(self, params: EmbeddingParams):
        """Update internal embedding tables from optimized parameters.

        Args:
            params: Dict with 'genre_projection' and 'user_embeddings' arrays
        """
        self.genre_projection = np.array(params["genre_projection"])
        self.user_embedding_table = np.array(params["user_embeddings"])

    def compute_embeddings_from_params(
        self,
        emb_params: EmbeddingParams,
        batch: RecsysBatch,
    ) -> RecsysEmbeddings:
        """Compute embeddings using learnable parameters (for training with gradients).

        This method uses JAX operations so gradients can flow through.

        Args:
            emb_params: Embedding parameters dict
            batch: The batch containing user/movie IDs

        Returns:
            RecsysEmbeddings computed from the learnable parameters
        """
        genre_proj = emb_params["genre_projection"]  # [19, emb_size]
        user_embs = emb_params["user_embeddings"]    # [num_users+1, emb_size]

        batch_size = batch.user_hashes.shape[0]
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes

        # User embeddings: look up from user_embeddings table
        # user_hashes: [B, num_user_hashes] -> user_embeddings: [B, num_user_hashes, emb_size]
        user_ids = batch.user_hashes[:, 0]  # [B] - take first hash as user ID
        user_embeddings = user_embs[user_ids]  # [B, emb_size]
        user_embeddings = jnp.tile(
            user_embeddings[:, jnp.newaxis, :],
            (1, num_user_hashes, 1)
        )  # [B, num_user_hashes, emb_size]

        # Movie genre vectors (precomputed, not learnable)
        # Add small epsilon to prevent zero vectors causing gradient issues
        movie_genres = jnp.array(self._movie_genre_vectors) + 1e-6  # [num_movies+1, 19]

        # History post embeddings: genre_vec @ genre_projection
        # history_post_hashes: [B, history_len, num_item_hashes]
        history_movie_ids = batch.history_post_hashes[:, :, 0]  # [B, history_len]
        # Clip indices to valid range (0 means padding, use index 1 as fallback)
        history_movie_ids = jnp.clip(history_movie_ids, 1, self.dataset.num_movies)
        history_genres = movie_genres[history_movie_ids]  # [B, history_len, 19]
        history_post_embs = jnp.einsum('blg,ge->ble', history_genres, genre_proj)  # [B, history_len, emb_size]
        # L2 normalize with safe division (no normalization to avoid gradient issues)
        # history_norms = jnp.linalg.norm(history_post_embs, axis=-1, keepdims=True)
        # history_norms = jnp.maximum(history_norms, 1e-6)
        # history_post_embs = history_post_embs / history_norms
        # Tile for num_item_hashes
        history_post_embeddings = jnp.tile(
            history_post_embs[:, :, jnp.newaxis, :],
            (1, 1, num_item_hashes, 1)
        )  # [B, history_len, num_item_hashes, emb_size]

        # Candidate post embeddings
        candidate_movie_ids = batch.candidate_post_hashes[:, :, 0]  # [B, num_candidates]
        # Clip indices to valid range
        candidate_movie_ids = jnp.clip(candidate_movie_ids, 1, self.dataset.num_movies)
        candidate_genres = movie_genres[candidate_movie_ids]  # [B, num_candidates, 19]
        candidate_post_embs = jnp.einsum('bcg,ge->bce', candidate_genres, genre_proj)  # [B, num_candidates, emb_size]
        # L2 normalize with safe division (no normalization to avoid gradient issues)
        # candidate_norms = jnp.linalg.norm(candidate_post_embs, axis=-1, keepdims=True)
        # candidate_norms = jnp.maximum(candidate_norms, 1e-6)
        # candidate_post_embs = candidate_post_embs / candidate_norms
        # Tile for num_item_hashes
        candidate_post_embeddings = jnp.tile(
            candidate_post_embs[:, :, jnp.newaxis, :],
            (1, 1, num_item_hashes, 1)
        )  # [B, num_candidates, num_item_hashes, emb_size]

        # Author embeddings (dummy - small constant for stability)
        history_author_embeddings = jnp.full(
            (batch_size, self.history_len, num_author_hashes, self.emb_size),
            0.01,
            dtype=jnp.float32
        )
        candidate_author_embeddings = jnp.full(
            (batch_size, self.num_candidates, num_author_hashes, self.emb_size),
            0.01,
            dtype=jnp.float32
        )

        return RecsysEmbeddings(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embeddings,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            candidate_author_embeddings=candidate_author_embeddings,
        )

    def get_movie_embedding(self, movie_id: int) -> np.ndarray:
        """Get embedding for a movie from its genre vector.

        Args:
            movie_id: Movie ID

        Returns:
            Embedding vector [emb_size]
        """
        genre_vec = self.dataset.get_movie_genre_vector(movie_id)
        # Project genres to embedding space
        emb = genre_vec @ self.genre_projection
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get embedding for a user.

        Args:
            user_id: User ID

        Returns:
            Embedding vector [emb_size]
        """
        return self.user_embedding_table[user_id].astype(np.float32)

    def rating_to_actions(self, rating: int) -> np.ndarray:
        """Convert rating (1-5) to action vector.

        Maps ratings to different "actions" in the 19-action space:
        - Rating 5: Strong positive actions (like, love)
        - Rating 4: Moderate positive (engage)
        - Rating 3: Neutral (view)
        - Rating 1-2: Negative (skip) - represented as zeros

        Returns:
            Action vector [num_actions] with engagement signals
        """
        num_actions = self.model_config.num_actions
        actions = np.zeros(num_actions, dtype=np.float32)

        if rating >= 5:
            # Strong positive: multiple action types
            actions[0] = 1.0  # "like"
            actions[1] = 0.8  # "engage"
            actions[2] = 0.6  # "view"
        elif rating >= 4:
            # Moderate positive
            actions[1] = 1.0  # "engage"
            actions[2] = 0.7  # "view"
        elif rating >= 3:
            # Neutral
            actions[2] = 1.0  # "view"
        # Rating 1-2: all zeros (negative signal)

        return actions

    def create_batch_for_user(
        self,
        user_id: int,
        candidate_movie_ids: List[int],
        history_limit: Optional[int] = None,
    ) -> Tuple[RecsysBatch, RecsysEmbeddings]:
        """Create Phoenix batch for a single user.

        Args:
            user_id: User ID
            candidate_movie_ids: Movie IDs to rank
            history_limit: Max history items to use (None = use all up to history_len)

        Returns:
            Tuple of (RecsysBatch, RecsysEmbeddings)
        """
        history = self.dataset.get_user_history(user_id)

        if history_limit is not None:
            history = history[:history_limit]

        # Truncate or pad history to history_len
        if len(history) > self.history_len:
            history = history[-self.history_len:]  # Take most recent

        # Truncate or pad candidates to num_candidates
        candidates = candidate_movie_ids[:self.num_candidates]

        # Build batch arrays
        batch_size = 1
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes
        num_actions = self.model_config.num_actions

        # User hashes (just the user_id repeated)
        user_hashes = np.full((batch_size, num_user_hashes), user_id, dtype=np.int32)

        # History post hashes (movie_ids)
        history_post_hashes = np.zeros(
            (batch_size, self.history_len, num_item_hashes), dtype=np.int32
        )
        for i, rating in enumerate(history):
            history_post_hashes[0, i, :] = rating.movie_id

        # History author hashes (dummy - use 1 as placeholder)
        history_author_hashes = np.ones(
            (batch_size, self.history_len, num_author_hashes), dtype=np.int32
        )

        # History actions (from ratings)
        history_actions = np.zeros(
            (batch_size, self.history_len, num_actions), dtype=np.float32
        )
        for i, rating in enumerate(history):
            history_actions[0, i, :] = self.rating_to_actions(rating.rating)

        # History product surface (constant 0)
        history_product_surface = np.zeros(
            (batch_size, self.history_len), dtype=np.int32
        )

        # Candidate post hashes
        candidate_post_hashes = np.zeros(
            (batch_size, self.num_candidates, num_item_hashes), dtype=np.int32
        )
        for i, movie_id in enumerate(candidates):
            candidate_post_hashes[0, i, :] = movie_id

        # Candidate author hashes (dummy)
        candidate_author_hashes = np.ones(
            (batch_size, self.num_candidates, num_author_hashes), dtype=np.int32
        )

        # Candidate product surface (constant 0)
        candidate_product_surface = np.zeros(
            (batch_size, self.num_candidates), dtype=np.int32
        )

        batch = RecsysBatch(
            user_hashes=user_hashes,
            history_post_hashes=history_post_hashes,
            history_author_hashes=history_author_hashes,
            history_actions=history_actions,
            history_product_surface=history_product_surface,
            candidate_post_hashes=candidate_post_hashes,
            candidate_author_hashes=candidate_author_hashes,
            candidate_product_surface=candidate_product_surface,
        )

        # Build embeddings
        embeddings = self._build_embeddings(
            user_id, history, candidates, batch_size
        )

        return batch, embeddings

    def _build_embeddings(
        self,
        user_id: int,
        history: List[Rating],
        candidates: List[int],
        batch_size: int,
    ) -> RecsysEmbeddings:
        """Build embedding arrays from user and movie data."""
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes

        # User embeddings [B, num_user_hashes, emb_size]
        user_emb = self.get_user_embedding(user_id)
        user_embeddings = np.tile(
            user_emb[np.newaxis, np.newaxis, :],
            (batch_size, num_user_hashes, 1)
        )

        # History post embeddings [B, history_len, num_item_hashes, emb_size]
        history_post_embeddings = np.zeros(
            (batch_size, self.history_len, num_item_hashes, self.emb_size),
            dtype=np.float32
        )
        for i, rating in enumerate(history):
            emb = self.get_movie_embedding(rating.movie_id)
            history_post_embeddings[0, i, :, :] = emb

        # Candidate post embeddings [B, num_candidates, num_item_hashes, emb_size]
        candidate_post_embeddings = np.zeros(
            (batch_size, self.num_candidates, num_item_hashes, self.emb_size),
            dtype=np.float32
        )
        for i, movie_id in enumerate(candidates):
            emb = self.get_movie_embedding(movie_id)
            candidate_post_embeddings[0, i, :, :] = emb

        # Author embeddings (dummy - small random for stability)
        history_author_embeddings = np.full(
            (batch_size, self.history_len, num_author_hashes, self.emb_size),
            0.01,
            dtype=np.float32
        )
        candidate_author_embeddings = np.full(
            (batch_size, self.num_candidates, num_author_hashes, self.emb_size),
            0.01,
            dtype=np.float32
        )

        return RecsysEmbeddings(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embeddings,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_embeddings=history_author_embeddings,
            candidate_author_embeddings=candidate_author_embeddings,
        )

    def get_training_example(
        self,
        rating: Rating,
        num_negatives: int = 4,
    ) -> Tuple[RecsysBatch, RecsysEmbeddings, np.ndarray]:
        """Create training example from a rating.

        Creates a batch with:
        - The rated movie as a positive candidate
        - num_negatives random unrated movies as negative candidates

        Args:
            rating: The positive rating
            num_negatives: Number of negative samples

        Returns:
            Tuple of (batch, embeddings, labels)
            labels: [num_candidates] with 1 for positive, 0 for negative
        """
        user_id = rating.user_id
        positive_movie = rating.movie_id

        # Sample negatives
        negatives = self.dataset.sample_negative_movies(
            user_id, num_negatives, self.rng
        )

        # Combine positive + negatives as candidates
        candidates = [positive_movie] + negatives

        # Shuffle to avoid position bias (positive always first)
        order = self.rng.permutation(len(candidates))
        candidates = [candidates[i] for i in order]

        # Labels: 1 for positive, 0 for negative
        labels = np.zeros(self.num_candidates, dtype=np.float32)
        positive_idx = list(order).index(0)  # Where did positive end up?
        labels[positive_idx] = 1.0

        # Also encode the rating strength
        rating_labels = np.zeros(self.num_candidates, dtype=np.float32)
        rating_labels[positive_idx] = rating.rating / 5.0  # Normalize to [0, 1]

        # Get user history up to this rating (temporal split)
        history = self.dataset.get_user_history(user_id)
        history_before = [r for r in history if r.timestamp < rating.timestamp]

        # Create batch with history before this rating
        batch, embeddings = self.create_batch_for_user(
            user_id,
            candidates,
            history_limit=len(history_before),
        )

        return batch, embeddings, labels

    def get_training_batch(
        self,
        batch_size: int = 32,
        num_negatives: int = 4,
    ) -> Tuple[RecsysBatch, RecsysEmbeddings, np.ndarray]:
        """Get a batch of training examples.

        Args:
            batch_size: Number of examples in batch
            num_negatives: Negative samples per positive

        Returns:
            Tuple of (batched_batch, batched_embeddings, batched_labels)
        """
        # Sample random ratings from training set
        indices = self.rng.choice(
            len(self.dataset.train_ratings),
            size=batch_size,
            replace=False
        )

        batches = []
        embeddings_list = []
        labels_list = []

        for idx in indices:
            rating = self.dataset.train_ratings[idx]
            batch, emb, labels = self.get_training_example(rating, num_negatives)
            batches.append(batch)
            embeddings_list.append(emb)
            labels_list.append(labels)

        # Stack into batched tensors
        return self._stack_batches(batches, embeddings_list, labels_list)

    def _stack_batches(
        self,
        batches: List[RecsysBatch],
        embeddings_list: List[RecsysEmbeddings],
        labels_list: List[np.ndarray],
    ) -> Tuple[RecsysBatch, RecsysEmbeddings, np.ndarray]:
        """Stack list of single-example batches into one batched batch."""

        stacked_batch = RecsysBatch(
            user_hashes=np.concatenate([b.user_hashes for b in batches], axis=0),
            history_post_hashes=np.concatenate([b.history_post_hashes for b in batches], axis=0),
            history_author_hashes=np.concatenate([b.history_author_hashes for b in batches], axis=0),
            history_actions=np.concatenate([b.history_actions for b in batches], axis=0),
            history_product_surface=np.concatenate([b.history_product_surface for b in batches], axis=0),
            candidate_post_hashes=np.concatenate([b.candidate_post_hashes for b in batches], axis=0),
            candidate_author_hashes=np.concatenate([b.candidate_author_hashes for b in batches], axis=0),
            candidate_product_surface=np.concatenate([b.candidate_product_surface for b in batches], axis=0),
        )

        stacked_embeddings = RecsysEmbeddings(
            user_embeddings=np.concatenate([e.user_embeddings for e in embeddings_list], axis=0),
            history_post_embeddings=np.concatenate([e.history_post_embeddings for e in embeddings_list], axis=0),
            candidate_post_embeddings=np.concatenate([e.candidate_post_embeddings for e in embeddings_list], axis=0),
            history_author_embeddings=np.concatenate([e.history_author_embeddings for e in embeddings_list], axis=0),
            candidate_author_embeddings=np.concatenate([e.candidate_author_embeddings for e in embeddings_list], axis=0),
        )

        stacked_labels = np.stack(labels_list, axis=0)

        return stacked_batch, stacked_embeddings, stacked_labels
