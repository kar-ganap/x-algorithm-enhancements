"""Adapter to convert synthetic Twitter data to Phoenix format.

Maps synthetic users, posts, and engagements to Phoenix's RecsysBatch
and RecsysEmbeddings.

Usage:
    from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

    adapter = SyntheticTwitterPhoenixAdapter(dataset, model_config)
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

from enhancements.data.ground_truth import (
    UserArchetype,
    ContentTopic,
)
from enhancements.data.synthetic_twitter import (
    SyntheticTwitterDataset,
    SyntheticEngagement,
)


# Type alias for embedding parameters
EmbeddingParams = Dict[str, jnp.ndarray]

# Embedding dimension for archetype/topic base vectors
NUM_ARCHETYPES = len(UserArchetype)
NUM_TOPICS = len(ContentTopic)


@dataclass
class SyntheticTwitterPhoenixAdapter:
    """Adapter to convert synthetic Twitter data to Phoenix format.

    Creates RecsysBatch and RecsysEmbeddings from synthetic users and posts.

    Mapping:
    - user_hashes: [user_id] (single hash)
    - post_hashes: [post_id] (single hash)
    - author_hashes: [author_id] (actual author tracking)
    - history_actions: from engagement actions dict
    - embeddings: Generated from archetype/topic + learned projection
    """

    dataset: SyntheticTwitterDataset
    model_config: PhoenixModelConfig
    emb_size: int = 128
    history_len: int = 32
    num_candidates: int = 8
    rng: np.random.Generator = None

    # Learned embedding projections (set during training)
    archetype_projection: Optional[np.ndarray] = None  # [NUM_ARCHETYPES, emb_size]
    topic_projection: Optional[np.ndarray] = None  # [NUM_TOPICS, emb_size]
    user_embedding_table: Optional[np.ndarray] = None  # [num_users+1, emb_size]
    author_embedding_table: Optional[np.ndarray] = None  # [num_authors+1, emb_size]

    # Training data splits
    train_engagements: Optional[List[SyntheticEngagement]] = None
    val_engagements: Optional[List[SyntheticEngagement]] = None
    test_engagements: Optional[List[SyntheticEngagement]] = None

    # Index mappings
    _archetype_to_idx: Dict[UserArchetype, int] = None
    _topic_to_idx: Dict[ContentTopic, int] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(42)

        self.emb_size = self.model_config.emb_size
        self.history_len = self.model_config.history_seq_len
        self.num_candidates = self.model_config.candidate_seq_len

        # Build index mappings
        self._archetype_to_idx = {a: i for i, a in enumerate(UserArchetype)}
        self._topic_to_idx = {t: i for i, t in enumerate(ContentTopic)}

        # Initialize random projections
        self._init_embeddings()

        # Build user history index for fast lookup
        self._user_history_cache = {}
        self._build_user_history_cache()

    def _init_embeddings(self):
        """Initialize embedding projections."""
        # Archetype to embedding projection
        self.archetype_projection = self.rng.normal(
            0, 0.1, size=(NUM_ARCHETYPES, self.emb_size)
        ).astype(np.float32)

        # Topic to embedding projection
        self.topic_projection = self.rng.normal(
            0, 0.1, size=(NUM_TOPICS, self.emb_size)
        ).astype(np.float32)

        # User embedding table (user_id 1-N maps to index 1-N)
        self.user_embedding_table = self.rng.normal(
            0, 0.1, size=(self.dataset.num_users + 1, self.emb_size)
        ).astype(np.float32)

        # Author embedding table
        self.author_embedding_table = self.rng.normal(
            0, 0.1, size=(self.dataset.num_authors + 1, self.emb_size)
        ).astype(np.float32)

    def _build_user_history_cache(self):
        """Build cache of user histories for fast training."""
        for user in self.dataset.users:
            history = self.dataset.get_user_history(user.user_id)
            self._user_history_cache[user.user_id] = history

    def set_splits(
        self,
        train: List[SyntheticEngagement],
        val: List[SyntheticEngagement],
        test: List[SyntheticEngagement],
    ):
        """Set train/val/test splits.

        Args:
            train: Training engagements
            val: Validation engagements
            test: Test engagements
        """
        self.train_engagements = train
        self.val_engagements = val
        self.test_engagements = test

        # Rebuild history cache with only train data for training
        self._user_history_cache = {}
        for eng in train:
            if eng.user_id not in self._user_history_cache:
                self._user_history_cache[eng.user_id] = []
            self._user_history_cache[eng.user_id].append(eng)

        # Sort by timestamp
        for user_id in self._user_history_cache:
            self._user_history_cache[user_id].sort(key=lambda e: e.timestamp)

    def get_embedding_params(self) -> EmbeddingParams:
        """Get embedding parameters as a dict for optimization.

        Returns:
            Dict with projection and embedding arrays
        """
        return {
            "archetype_projection": jnp.array(self.archetype_projection),
            "topic_projection": jnp.array(self.topic_projection),
            "user_embeddings": jnp.array(self.user_embedding_table),
            "author_embeddings": jnp.array(self.author_embedding_table),
        }

    def set_embedding_params(self, params: EmbeddingParams):
        """Update internal embedding tables from optimized parameters.

        Args:
            params: Dict with projection and embedding arrays
        """
        self.archetype_projection = np.array(params["archetype_projection"])
        self.topic_projection = np.array(params["topic_projection"])
        self.user_embedding_table = np.array(params["user_embeddings"])
        self.author_embedding_table = np.array(params["author_embeddings"])

    def compute_embeddings_from_params(
        self,
        emb_params: EmbeddingParams,
        batch: RecsysBatch,
    ) -> RecsysEmbeddings:
        """Compute embeddings using learnable parameters (for training with gradients).

        This method uses JAX operations so gradients can flow through.

        Args:
            emb_params: Embedding parameters dict
            batch: The batch containing user/post/author IDs

        Returns:
            RecsysEmbeddings computed from the learnable parameters
        """
        topic_proj = emb_params["topic_projection"]  # [NUM_TOPICS, emb_size]
        user_embs = emb_params["user_embeddings"]    # [num_users+1, emb_size]
        author_embs = emb_params["author_embeddings"]  # [num_authors+1, emb_size]

        batch_size = batch.user_hashes.shape[0]
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes

        # User embeddings: look up from user_embeddings table
        user_ids = batch.user_hashes[:, 0]  # [B]
        user_ids = jnp.clip(user_ids, 0, self.dataset.num_users)
        user_embeddings = user_embs[user_ids]  # [B, emb_size]
        user_embeddings = jnp.tile(
            user_embeddings[:, jnp.newaxis, :],
            (1, num_user_hashes, 1)
        )  # [B, num_user_hashes, emb_size]

        # History post embeddings: use topic projection
        # Each post has a topic, we look up the topic embedding
        history_post_ids = batch.history_post_hashes[:, :, 0]  # [B, history_len]
        history_topics = self._get_topics_for_posts_jax(history_post_ids)  # [B, history_len]
        history_post_embs = topic_proj[history_topics]  # [B, history_len, emb_size]
        history_post_embs = jnp.tile(
            history_post_embs[:, :, jnp.newaxis, :],
            (1, 1, num_item_hashes, 1)
        )  # [B, history_len, num_item_hashes, emb_size]

        # Candidate post embeddings
        cand_post_ids = batch.candidate_post_hashes[:, :, 0]  # [B, num_candidates]
        cand_topics = self._get_topics_for_posts_jax(cand_post_ids)  # [B, num_candidates]
        cand_post_embs = topic_proj[cand_topics]  # [B, num_candidates, emb_size]
        cand_post_embs = jnp.tile(
            cand_post_embs[:, :, jnp.newaxis, :],
            (1, 1, num_item_hashes, 1)
        )  # [B, num_candidates, num_item_hashes, emb_size]

        # History author embeddings
        history_author_ids = batch.history_author_hashes[:, :, 0]  # [B, history_len]
        history_author_ids = jnp.clip(history_author_ids, 0, self.dataset.num_authors)
        history_author_embs = author_embs[history_author_ids]  # [B, history_len, emb_size]
        history_author_embs = jnp.tile(
            history_author_embs[:, :, jnp.newaxis, :],
            (1, 1, num_author_hashes, 1)
        )  # [B, history_len, num_author_hashes, emb_size]

        # Candidate author embeddings
        cand_author_ids = batch.candidate_author_hashes[:, :, 0]  # [B, num_candidates]
        cand_author_ids = jnp.clip(cand_author_ids, 0, self.dataset.num_authors)
        cand_author_embs = author_embs[cand_author_ids]  # [B, num_candidates, emb_size]
        cand_author_embs = jnp.tile(
            cand_author_embs[:, :, jnp.newaxis, :],
            (1, 1, num_author_hashes, 1)
        )  # [B, num_candidates, num_author_hashes, emb_size]

        return RecsysEmbeddings(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embs,
            candidate_post_embeddings=cand_post_embs,
            history_author_embeddings=history_author_embs,
            candidate_author_embeddings=cand_author_embs,
        )

    def _get_topics_for_posts_jax(self, post_ids: jnp.ndarray) -> jnp.ndarray:
        """Get topic indices for post IDs (JAX-compatible).

        Args:
            post_ids: Array of post IDs [...]

        Returns:
            Array of topic indices [...]
        """
        # Build a lookup table: post_id -> topic_idx
        lookup = np.zeros(self.dataset.num_posts + 1, dtype=np.int32)
        for post in self.dataset.posts:
            lookup[post.post_id] = self._topic_to_idx[post.topic]

        lookup_jax = jnp.array(lookup)
        post_ids_clipped = jnp.clip(post_ids, 0, self.dataset.num_posts)
        return lookup_jax[post_ids_clipped]

    def engagement_to_actions(self, engagement: SyntheticEngagement) -> np.ndarray:
        """Convert engagement to action vector.

        Args:
            engagement: The engagement event

        Returns:
            Action vector [num_actions] with engagement signals
        """
        num_actions = self.model_config.num_actions
        actions = np.zeros(num_actions, dtype=np.float32)

        # Action name order in Phoenix
        action_order = [
            "favorite_score",
            "reply_score",
            "repost_score",
            "photo_expand_score",
            "click_score",
            "profile_click_score",
            "vqv_score",
            "share_score",
            "share_via_dm_score",
            "share_via_copy_link_score",
            "dwell_score",
            "quote_score",
            "quoted_click_score",
            "follow_author_score",
            "not_interested_score",
            "block_author_score",
            "mute_author_score",
            "report_score",
        ]

        for i, action_name in enumerate(action_order):
            if i < num_actions and action_name in engagement.actions:
                actions[i] = engagement.actions[action_name]

        return actions

    def create_batch_for_user(
        self,
        user_id: int,
        candidate_post_ids: List[int],
        history_limit: Optional[int] = None,
        num_candidates_override: Optional[int] = None,
    ) -> Tuple[RecsysBatch, RecsysEmbeddings]:
        """Create Phoenix batch for a single user.

        Args:
            user_id: User ID
            candidate_post_ids: Post IDs to rank
            history_limit: Max history items to use
            num_candidates_override: Override number of candidates

        Returns:
            Tuple of (RecsysBatch, RecsysEmbeddings)
        """
        history = self._user_history_cache.get(user_id, [])

        if history_limit is not None:
            history = history[:history_limit]

        # Truncate history to history_len
        if len(history) > self.history_len:
            history = history[-self.history_len:]

        num_cands = num_candidates_override if num_candidates_override else self.num_candidates
        candidates = candidate_post_ids[:num_cands]

        # Build batch arrays
        batch_size = 1
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes
        num_actions = self.model_config.num_actions

        # User hashes
        user_hashes = np.full((batch_size, num_user_hashes), user_id, dtype=np.int32)

        # History post hashes (post_ids)
        history_post_hashes = np.zeros(
            (batch_size, self.history_len, num_item_hashes), dtype=np.int32
        )
        history_author_hashes = np.zeros(
            (batch_size, self.history_len, num_author_hashes), dtype=np.int32
        )
        history_actions = np.zeros(
            (batch_size, self.history_len, num_actions), dtype=np.float32
        )

        for i, eng in enumerate(history):
            post = self.dataset.get_post(eng.post_id)
            if post:
                history_post_hashes[0, i, :] = eng.post_id
                history_author_hashes[0, i, :] = post.author_id
            history_actions[0, i, :] = self.engagement_to_actions(eng)

        # History product surface (constant 0)
        history_product_surface = np.zeros(
            (batch_size, self.history_len), dtype=np.int32
        )

        # Candidate post hashes
        actual_num_candidates = len(candidates)
        candidate_post_hashes = np.zeros(
            (batch_size, actual_num_candidates, num_item_hashes), dtype=np.int32
        )
        candidate_author_hashes = np.zeros(
            (batch_size, actual_num_candidates, num_author_hashes), dtype=np.int32
        )

        for i, post_id in enumerate(candidates):
            post = self.dataset.get_post(post_id)
            if post:
                candidate_post_hashes[0, i, :] = post_id
                candidate_author_hashes[0, i, :] = post.author_id

        candidate_product_surface = np.zeros(
            (batch_size, actual_num_candidates), dtype=np.int32
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

        # Compute embeddings
        embeddings = self._compute_embeddings_numpy(batch)

        return batch, embeddings

    def _compute_embeddings_numpy(self, batch: RecsysBatch) -> RecsysEmbeddings:
        """Compute embeddings using numpy (for inference, not training).

        Args:
            batch: The batch

        Returns:
            RecsysEmbeddings
        """
        batch_size = batch.user_hashes.shape[0]
        num_user_hashes = self.model_config.hash_config.num_user_hashes
        num_item_hashes = self.model_config.hash_config.num_item_hashes
        num_author_hashes = self.model_config.hash_config.num_author_hashes

        # User embeddings
        user_ids = batch.user_hashes[:, 0]  # [B]
        user_ids = np.clip(user_ids, 0, self.dataset.num_users)
        user_embs = self.user_embedding_table[user_ids]  # [B, emb_size]
        user_embeddings = np.tile(
            user_embs[:, np.newaxis, :],
            (1, num_user_hashes, 1)
        )  # [B, num_user_hashes, emb_size]

        # History post embeddings
        history_len = batch.history_post_hashes.shape[1]
        history_post_embs = np.zeros(
            (batch_size, history_len, num_item_hashes, self.emb_size),
            dtype=np.float32
        )
        for b in range(batch_size):
            for h in range(history_len):
                post_id = batch.history_post_hashes[b, h, 0]
                if post_id > 0:
                    post = self.dataset.get_post(post_id)
                    if post:
                        topic_idx = self._topic_to_idx[post.topic]
                        emb = self.topic_projection[topic_idx]
                        for k in range(num_item_hashes):
                            history_post_embs[b, h, k, :] = emb

        # Candidate post embeddings
        num_candidates = batch.candidate_post_hashes.shape[1]
        cand_post_embs = np.zeros(
            (batch_size, num_candidates, num_item_hashes, self.emb_size),
            dtype=np.float32
        )
        for b in range(batch_size):
            for c in range(num_candidates):
                post_id = batch.candidate_post_hashes[b, c, 0]
                if post_id > 0:
                    post = self.dataset.get_post(post_id)
                    if post:
                        topic_idx = self._topic_to_idx[post.topic]
                        emb = self.topic_projection[topic_idx]
                        for k in range(num_item_hashes):
                            cand_post_embs[b, c, k, :] = emb

        # History author embeddings
        history_author_embs = np.zeros(
            (batch_size, history_len, num_author_hashes, self.emb_size),
            dtype=np.float32
        )
        for b in range(batch_size):
            for h in range(history_len):
                author_id = batch.history_author_hashes[b, h, 0]
                if 0 < author_id <= self.dataset.num_authors:
                    emb = self.author_embedding_table[author_id]
                    for k in range(num_author_hashes):
                        history_author_embs[b, h, k, :] = emb

        # Candidate author embeddings
        cand_author_embs = np.zeros(
            (batch_size, num_candidates, num_author_hashes, self.emb_size),
            dtype=np.float32
        )
        for b in range(batch_size):
            for c in range(num_candidates):
                author_id = batch.candidate_author_hashes[b, c, 0]
                if 0 < author_id <= self.dataset.num_authors:
                    emb = self.author_embedding_table[author_id]
                    for k in range(num_author_hashes):
                        cand_author_embs[b, c, k, :] = emb

        return RecsysEmbeddings(
            user_embeddings=user_embeddings,
            history_post_embeddings=history_post_embs,
            candidate_post_embeddings=cand_post_embs,
            history_author_embeddings=history_author_embs,
            candidate_author_embeddings=cand_author_embs,
        )

    def get_training_batch(
        self,
        batch_size: int = 32,
        neg_ratio: int = 4,
    ) -> Tuple[RecsysBatch, EmbeddingParams, np.ndarray]:
        """Get a training batch with positive and negative samples.

        For each positive engagement, samples neg_ratio negative posts
        (posts the user didn't engage with).

        Args:
            batch_size: Number of training examples
            neg_ratio: Number of negatives per positive

        Returns:
            Tuple of (batch, embedding_params, labels)
            - batch: RecsysBatch
            - embedding_params: EmbeddingParams for computing embeddings
            - labels: [batch_size, 1+neg_ratio] binary labels (1 for positive)
        """
        if self.train_engagements is None:
            raise ValueError("Must call set_splits() before get_training_batch()")

        # Sample training engagements
        indices = self.rng.choice(
            len(self.train_engagements), size=batch_size, replace=True
        )

        batches = []
        all_labels = []

        for idx in indices:
            eng = self.train_engagements[idx]
            user_id = eng.user_id
            positive_post_id = eng.post_id

            # Sample negative posts (posts user hasn't engaged with)
            user_engaged_posts = {e.post_id for e in self._user_history_cache.get(user_id, [])}
            candidate_pool = [p for p in self.dataset.all_post_ids if p not in user_engaged_posts]

            if len(candidate_pool) < neg_ratio:
                # Not enough negatives, skip this sample
                continue

            negative_post_ids = self.rng.choice(
                candidate_pool, size=neg_ratio, replace=False
            ).tolist()

            # Candidates: positive first, then negatives
            candidate_ids = [positive_post_id] + negative_post_ids
            labels = [1.0] + [0.0] * neg_ratio

            batch, _ = self.create_batch_for_user(
                user_id, candidate_ids, num_candidates_override=len(candidate_ids)
            )
            batches.append(batch)
            all_labels.append(labels)

        if not batches:
            raise ValueError("Could not create any valid training samples")

        # Stack batches
        combined_batch = self._stack_batches(batches)
        labels_array = np.array(all_labels, dtype=np.float32)

        return combined_batch, self.get_embedding_params(), labels_array

    def _stack_batches(self, batches: List[RecsysBatch]) -> RecsysBatch:
        """Stack multiple single-sample batches into one.

        Args:
            batches: List of RecsysBatch objects

        Returns:
            Combined RecsysBatch
        """
        return RecsysBatch(
            user_hashes=np.concatenate([b.user_hashes for b in batches], axis=0),
            history_post_hashes=np.concatenate([b.history_post_hashes for b in batches], axis=0),
            history_author_hashes=np.concatenate([b.history_author_hashes for b in batches], axis=0),
            history_actions=np.concatenate([b.history_actions for b in batches], axis=0),
            history_product_surface=np.concatenate([b.history_product_surface for b in batches], axis=0),
            candidate_post_hashes=np.concatenate([b.candidate_post_hashes for b in batches], axis=0),
            candidate_author_hashes=np.concatenate([b.candidate_author_hashes for b in batches], axis=0),
            candidate_product_surface=np.concatenate([b.candidate_product_surface for b in batches], axis=0),
        )

    def get_validation_samples(
        self,
        num_samples: int = 100,
    ) -> List[Tuple[int, int, List[int]]]:
        """Get validation samples for evaluation.

        Returns tuples of (user_id, positive_post_id, negative_post_ids)
        for computing NDCG and Hit Rate.

        Args:
            num_samples: Number of validation samples

        Returns:
            List of (user_id, positive_post_id, negative_post_ids) tuples
        """
        if self.val_engagements is None:
            raise ValueError("Must call set_splits() before get_validation_samples()")

        samples = []
        sampled_indices = self.rng.choice(
            len(self.val_engagements),
            size=min(num_samples, len(self.val_engagements)),
            replace=False,
        )

        for idx in sampled_indices:
            eng = self.val_engagements[idx]
            user_id = eng.user_id
            positive_post_id = eng.post_id

            # Get posts user hasn't engaged with
            user_engaged_posts = {e.post_id for e in self._user_history_cache.get(user_id, [])}
            candidate_pool = [p for p in self.dataset.all_post_ids if p not in user_engaged_posts]

            if len(candidate_pool) >= 99:
                negative_post_ids = self.rng.choice(
                    candidate_pool, size=99, replace=False
                ).tolist()
                samples.append((user_id, positive_post_id, negative_post_ids))

        return samples
