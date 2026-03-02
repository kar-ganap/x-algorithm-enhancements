"""Adapter to convert synthetic Twitter data to Phoenix format.

Maps synthetic users, posts, and engagements to Phoenix's RecsysBatch
and RecsysEmbeddings.

Usage:
    from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

    adapter = SyntheticTwitterPhoenixAdapter(dataset, model_config)
    batch, embeddings, labels = adapter.get_training_batch(batch_size=32)
"""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from enhancements.data.ground_truth import (
    ContentTopic,
    UserArchetype,
    get_engagement_probs,
)
from enhancements.data.synthetic_twitter import (
    SyntheticEngagement,
    SyntheticTwitterDataset,
)
from phoenix.recsys_model import (
    PhoenixModelConfig,
    RecsysBatch,
    RecsysEmbeddings,
)

# Type alias for embedding parameters
EmbeddingParams = dict[str, jnp.ndarray]

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
    archetype_projection: np.ndarray | None = None  # [NUM_ARCHETYPES, emb_size]
    topic_projection: np.ndarray | None = None  # [NUM_TOPICS, emb_size]
    user_embedding_table: np.ndarray | None = None  # [num_users+1, emb_size]
    author_embedding_table: np.ndarray | None = None  # [num_authors+1, emb_size]

    # Training data splits
    train_engagements: list[SyntheticEngagement] | None = None
    val_engagements: list[SyntheticEngagement] | None = None
    test_engagements: list[SyntheticEngagement] | None = None

    # Index mappings
    _archetype_to_idx: dict[UserArchetype, int] = None
    _topic_to_idx: dict[ContentTopic, int] = None

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
        """Initialize embedding projections with archetype-specific user embeddings.

        Users are initialized in distinct regions of embedding space based on their
        archetype. This helps the classifier and action predictor learn faster.
        """
        # Archetype base embeddings - well-separated in embedding space
        # Use orthogonal-ish vectors for maximum separation
        self.archetype_projection = np.zeros((NUM_ARCHETYPES, self.emb_size), dtype=np.float32)
        for i in range(NUM_ARCHETYPES):
            # Create a base direction for each archetype
            # Spread across different dimensions to maximize separation
            base_dim = (i * self.emb_size) // NUM_ARCHETYPES
            self.archetype_projection[i, base_dim:base_dim + self.emb_size // NUM_ARCHETYPES] = 1.0
            # Add small random component
            self.archetype_projection[i] += self.rng.normal(0, 0.05, self.emb_size).astype(np.float32)
        # Normalize to unit length
        norms = np.linalg.norm(self.archetype_projection, axis=1, keepdims=True)
        self.archetype_projection = (self.archetype_projection / norms).astype(np.float32)

        # Topic to embedding projection - also well-separated
        self.topic_projection = np.zeros((NUM_TOPICS, self.emb_size), dtype=np.float32)
        for i in range(NUM_TOPICS):
            base_dim = (i * self.emb_size) // NUM_TOPICS
            self.topic_projection[i, base_dim:base_dim + self.emb_size // NUM_TOPICS] = 1.0
            self.topic_projection[i] += self.rng.normal(0, 0.05, self.emb_size).astype(np.float32)
        norms = np.linalg.norm(self.topic_projection, axis=1, keepdims=True)
        self.topic_projection = (self.topic_projection / norms).astype(np.float32)

        # User embedding table - initialize based on user's archetype
        # Each user starts near their archetype's base embedding + small noise
        self.user_embedding_table = np.zeros((self.dataset.num_users + 1, self.emb_size), dtype=np.float32)
        for user in self.dataset.users:
            archetype_idx = self._archetype_to_idx[user.archetype]
            base_emb = self.archetype_projection[archetype_idx]
            # Add small noise (std=0.1) to create within-archetype variation
            noise = self.rng.normal(0, 0.1, self.emb_size).astype(np.float32)
            self.user_embedding_table[user.user_id] = base_emb + noise

        # Author embedding table - initialize based on author's primary topic
        self.author_embedding_table = np.zeros((self.dataset.num_authors + 1, self.emb_size), dtype=np.float32)
        for author in self.dataset.authors:
            topic_idx = self._topic_to_idx[author.primary_topic]
            base_emb = self.topic_projection[topic_idx]
            noise = self.rng.normal(0, 0.1, self.emb_size).astype(np.float32)
            self.author_embedding_table[author.author_id] = base_emb + noise

    def _build_user_history_cache(self):
        """Build cache of user histories for fast training."""
        for user in self.dataset.users:
            history = self.dataset.get_user_history(user.user_id)
            self._user_history_cache[user.user_id] = history

    def set_splits(
        self,
        train: list[SyntheticEngagement],
        val: list[SyntheticEngagement],
        test: list[SyntheticEngagement],
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
        candidate_post_ids: list[int],
        history_limit: int | None = None,
        num_candidates_override: int | None = None,
    ) -> tuple[RecsysBatch, RecsysEmbeddings]:
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
        hard_negative_ratio: float = 0.5,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[RecsysBatch, EmbeddingParams, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a training batch with positive and negative samples.

        For each positive engagement, samples neg_ratio negative posts.
        Uses topic-based hard negatives for a portion of negatives.

        Args:
            batch_size: Number of training examples
            neg_ratio: Number of negatives per positive
            hard_negative_ratio: Fraction of negatives from same topic (hard negatives)
            sample_weights: Optional weights for sampling engagements (for hard example mining)

        Returns:
            Tuple of (batch, embedding_params, labels, action_labels, archetype_labels, sample_indices)
            - batch: RecsysBatch
            - embedding_params: EmbeddingParams for computing embeddings
            - labels: [batch_size, 1+neg_ratio] binary labels (1 for positive)
            - action_labels: [batch_size, num_actions] action labels for positive sample
            - archetype_labels: [batch_size] archetype indices for users
            - sample_indices: [batch_size] indices into train_engagements (for hard example mining)
        """
        if self.train_engagements is None:
            raise ValueError("Must call set_splits() before get_training_batch()")

        # Sample training engagements (with optional weights for hard example mining)
        if sample_weights is not None:
            # Normalize weights to probabilities
            probs = sample_weights / sample_weights.sum()
            indices = self.rng.choice(
                len(self.train_engagements), size=batch_size, replace=True, p=probs
            )
        else:
            indices = self.rng.choice(
                len(self.train_engagements), size=batch_size, replace=True
            )

        batches = []
        all_labels = []
        all_action_labels = []
        all_archetype_labels = []
        valid_indices = []

        # Action names in order
        action_names = [
            "favorite_score", "reply_score", "repost_score", "photo_expand_score",
            "click_score", "profile_click_score", "vqv_score", "share_score",
            "share_via_dm_score", "share_via_copy_link_score", "dwell_score",
            "quote_score", "quoted_click_score", "follow_author_score",
            "not_interested_score", "block_author_score", "mute_author_score",
            "report_score",
        ]

        # Calculate how many hard vs easy negatives
        num_hard_negs = int(neg_ratio * hard_negative_ratio)
        num_easy_negs = neg_ratio - num_hard_negs

        for idx in indices:
            eng = self.train_engagements[idx]
            user_id = eng.user_id
            positive_post_id = eng.post_id

            # Get positive post's topic for hard negative sampling
            positive_post = self.dataset.get_post(positive_post_id)
            if positive_post is None:
                continue
            positive_topic = positive_post.topic

            # Get user's archetype
            user = self.dataset.get_user(user_id)
            if user is None:
                continue
            archetype_idx = self._archetype_to_idx[user.archetype]

            # Sample negative posts (posts user hasn't engaged with)
            user_engaged_posts = {e.post_id for e in self._user_history_cache.get(user_id, [])}

            # Hard negatives: same topic as positive (harder to distinguish)
            same_topic_posts = [
                p.post_id for p in self.dataset.get_posts_by_topic(positive_topic)
                if p.post_id not in user_engaged_posts and p.post_id != positive_post_id
            ]

            # Easy negatives: different topics
            other_posts = [
                p for p in self.dataset.all_post_ids
                if p not in user_engaged_posts and p != positive_post_id
                and self.dataset.get_post(p).topic != positive_topic
            ]

            # Sample hard negatives (same topic)
            if len(same_topic_posts) >= num_hard_negs and num_hard_negs > 0:
                hard_neg_ids = self.rng.choice(
                    same_topic_posts, size=num_hard_negs, replace=False
                ).tolist()
            else:
                hard_neg_ids = same_topic_posts[:num_hard_negs] if same_topic_posts else []

            # Sample easy negatives (different topics)
            remaining_negs = neg_ratio - len(hard_neg_ids)
            if len(other_posts) >= remaining_negs:
                easy_neg_ids = self.rng.choice(
                    other_posts, size=remaining_negs, replace=False
                ).tolist()
            else:
                # Fall back to any available posts
                all_available = [p for p in self.dataset.all_post_ids
                                if p not in user_engaged_posts and p != positive_post_id
                                and p not in hard_neg_ids]
                if len(all_available) >= remaining_negs:
                    easy_neg_ids = self.rng.choice(
                        all_available, size=remaining_negs, replace=False
                    ).tolist()
                else:
                    continue  # Skip if not enough negatives

            negative_post_ids = hard_neg_ids + easy_neg_ids

            if len(negative_post_ids) < neg_ratio:
                continue

            # Candidates: positive first, then negatives
            candidate_ids = [positive_post_id] + negative_post_ids
            labels = [1.0] + [0.0] * neg_ratio

            # Use ground truth probabilities as soft labels based on (archetype, topic)
            # This helps the model learn archetype-specific action rates
            gt_probs = get_engagement_probs(user.archetype, positive_topic)
            action_labels = gt_probs.to_array()[:18]  # First 18 actions

            batch, _ = self.create_batch_for_user(
                user_id, candidate_ids, num_candidates_override=len(candidate_ids)
            )
            batches.append(batch)
            all_labels.append(labels)
            all_action_labels.append(action_labels)
            all_archetype_labels.append(archetype_idx)
            valid_indices.append(idx)

        if not batches:
            raise ValueError("Could not create any valid training samples")

        # Stack batches
        combined_batch = self._stack_batches(batches)
        labels_array = np.array(all_labels, dtype=np.float32)
        action_labels_array = np.array(all_action_labels, dtype=np.float32)
        archetype_labels_array = np.array(all_archetype_labels, dtype=np.int32)
        sample_indices_array = np.array(valid_indices, dtype=np.int32)

        return (combined_batch, self.get_embedding_params(), labels_array,
                action_labels_array, archetype_labels_array, sample_indices_array)

    def _stack_batches(self, batches: list[RecsysBatch]) -> RecsysBatch:
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

    def get_block_aware_batch(
        self,
        batch_size: int = 16,
        synthetic_ratio: float = 0.5,
    ) -> tuple[RecsysBatch, EmbeddingParams, np.ndarray] | None:
        """Get training batch for block-aware contrastive learning.

        For generalizable block semantics, we create two types of pairs:
        1. Actual blocks: Users who blocked authors, testing on those authors
        2. Synthetic blocks: Random users, with injected block for random author

        This teaches the model that block action → author should score lower,
        not just memorize specific (user, author) pairs.

        Args:
            batch_size: Number of samples
            synthetic_ratio: Fraction of batch that should be synthetic blocks

        Returns:
            Tuple of (batch, embedding_params, block_labels) or None if no data
        """
        if self.train_engagements is None:
            raise ValueError("Must call set_splits() before get_block_aware_batch()")

        batches = []
        block_labels = []

        # Calculate split
        num_synthetic = int(batch_size * synthetic_ratio)
        num_actual = batch_size - num_synthetic

        # === Part 1: Actual block pairs ===
        # Find users who have blocked someone
        users_with_blocks = []
        user_blocked_authors = {}

        for eng in self.train_engagements:
            if eng.actions.get("block_author_score", 0) > 0:
                user_id = eng.user_id
                post = self.dataset.get_post(eng.post_id)
                if post:
                    author_id = post.author_id
                    if user_id not in user_blocked_authors:
                        user_blocked_authors[user_id] = set()
                        users_with_blocks.append(user_id)
                    user_blocked_authors[user_id].add(author_id)

        if users_with_blocks and num_actual > 0:
            sampled_users = self.rng.choice(
                users_with_blocks,
                size=min(num_actual, len(users_with_blocks)),
                replace=True,
            )

            for user_id in sampled_users:
                blocked_authors = user_blocked_authors[user_id]

                blocked_posts = []
                for author_id in blocked_authors:
                    blocked_posts.extend(self.dataset.get_posts_by_author(author_id))

                if not blocked_posts:
                    continue

                blocked_post = self.rng.choice(blocked_posts)

                non_blocked_posts = [
                    p for p in self.dataset.posts
                    if p.author_id not in blocked_authors
                ]

                if not non_blocked_posts:
                    continue

                non_blocked_post = self.rng.choice(non_blocked_posts)

                candidate_ids = [blocked_post.post_id, non_blocked_post.post_id]

                batch, _ = self.create_batch_for_user(
                    user_id, candidate_ids, num_candidates_override=2
                )

                history_actions = np.array(batch.history_actions)
                history_author_hashes = np.array(batch.history_author_hashes)

                history_actions[0, 0, :] = 0
                history_actions[0, 0, 15] = 1.0

                blocked_author_id = blocked_post.author_id
                for k in range(history_author_hashes.shape[2]):
                    history_author_hashes[0, 0, k] = blocked_author_id

                batch = RecsysBatch(
                    user_hashes=batch.user_hashes,
                    history_post_hashes=batch.history_post_hashes,
                    history_author_hashes=history_author_hashes,
                    history_actions=history_actions,
                    history_product_surface=batch.history_product_surface,
                    candidate_post_hashes=batch.candidate_post_hashes,
                    candidate_author_hashes=batch.candidate_author_hashes,
                    candidate_product_surface=batch.candidate_product_surface,
                )

                batches.append(batch)
                block_labels.append([1.0, 0.0])

        # === Part 2: Synthetic block pairs ===
        # Random user + random author, inject block action
        # This teaches generalization: block = lower score regardless of user identity
        all_users = [u.user_id for u in self.dataset.users]
        all_authors = [a.author_id for a in self.dataset.authors]

        for _ in range(num_synthetic):
            user_id = int(self.rng.choice(all_users))
            blocked_author_id = int(self.rng.choice(all_authors))

            blocked_posts = self.dataset.get_posts_by_author(blocked_author_id)
            if not blocked_posts:
                continue

            blocked_post = self.rng.choice(blocked_posts)

            non_blocked_posts = [
                p for p in self.dataset.posts
                if p.author_id != blocked_author_id
            ]
            non_blocked_post = self.rng.choice(non_blocked_posts)

            candidate_ids = [blocked_post.post_id, non_blocked_post.post_id]

            batch, _ = self.create_batch_for_user(
                user_id, candidate_ids, num_candidates_override=2
            )

            history_actions = np.array(batch.history_actions)
            history_author_hashes = np.array(batch.history_author_hashes)

            history_actions[0, 0, :] = 0
            history_actions[0, 0, 15] = 1.0

            for k in range(history_author_hashes.shape[2]):
                history_author_hashes[0, 0, k] = blocked_author_id

            batch = RecsysBatch(
                user_hashes=batch.user_hashes,
                history_post_hashes=batch.history_post_hashes,
                history_author_hashes=history_author_hashes,
                history_actions=history_actions,
                history_product_surface=batch.history_product_surface,
                candidate_post_hashes=batch.candidate_post_hashes,
                candidate_author_hashes=batch.candidate_author_hashes,
                candidate_product_surface=batch.candidate_product_surface,
            )

            batches.append(batch)
            block_labels.append([1.0, 0.0])

        if not batches:
            return None

        combined_batch = self._stack_batches(batches)
        block_labels_array = np.array(block_labels, dtype=np.float32)

        return combined_batch, self.get_embedding_params(), block_labels_array

    def get_history_contrastive_batch(
        self,
        batch_size: int = 16,
    ) -> tuple[RecsysBatch, RecsysBatch, EmbeddingParams] | None:
        """Get batch for history-topic contrastive learning.

        Creates pairs where:
        - Same candidate post (from a specific topic)
        - Matching history: from user who engages with that topic
        - Mismatched history: from user who engages with different topic

        Train: score(post | matching_history) > score(post | mismatched_history)

        This teaches the transformer to use history content to influence scores.

        Returns:
            Tuple of (matching_batch, mismatched_batch, embedding_params) or None
            Both batches have same candidates but different histories.
        """
        if self.train_engagements is None:
            raise ValueError("Must call set_splits() before get_history_contrastive_batch()")

        # Define archetype -> preferred topic mapping
        archetype_topic_map = {
            UserArchetype.SPORTS_FAN: ContentTopic.SPORTS,
            UserArchetype.TECH_BRO: ContentTopic.TECH,
            UserArchetype.POLITICAL_L: ContentTopic.POLITICS_L,
            UserArchetype.POLITICAL_R: ContentTopic.POLITICS_R,
        }

        # Reverse: topic -> archetype that prefers it
        topic_archetype_map = {v: k for k, v in archetype_topic_map.items()}

        # Get users by archetype for quick lookup
        users_by_archetype = {
            arch: self.dataset.get_users_by_archetype(arch)
            for arch in archetype_topic_map.keys()
        }

        matching_batches = []
        mismatched_batches = []

        # Topics we can create contrastive pairs for
        contrastive_topics = list(topic_archetype_map.keys())

        for _ in range(batch_size):
            # Pick a random topic that has a preferred archetype
            topic = contrastive_topics[int(self.rng.integers(0, len(contrastive_topics)))]
            preferred_archetype = topic_archetype_map[topic]

            # Get a post from this topic
            topic_posts = self.dataset.get_posts_by_topic(topic)
            if not topic_posts:
                continue
            post = topic_posts[int(self.rng.integers(0, len(topic_posts)))]

            # Get a user who prefers this topic (matching)
            matching_users = users_by_archetype[preferred_archetype]
            if not matching_users:
                continue
            matching_user = matching_users[int(self.rng.integers(0, len(matching_users)))]

            # Get a user who prefers a DIFFERENT topic (mismatched)
            other_archetypes = [a for a in archetype_topic_map.keys() if a != preferred_archetype]
            other_archetype = other_archetypes[int(self.rng.integers(0, len(other_archetypes)))]
            mismatched_users = users_by_archetype[other_archetype]
            if not mismatched_users:
                continue
            mismatched_user = mismatched_users[int(self.rng.integers(0, len(mismatched_users)))]

            # Create batch with MATCHING user's history
            # Use a neutral user_id (or matching user) - the key is the history
            matching_batch, _ = self.create_batch_for_user(
                matching_user.user_id, [post.post_id], num_candidates_override=1
            )

            # Create batch with MISMATCHED user's history but same candidate
            mismatched_batch_raw, _ = self.create_batch_for_user(
                mismatched_user.user_id, [post.post_id], num_candidates_override=1
            )

            # For fair comparison, use same user embedding but swap histories
            # This isolates the effect of history content
            mismatched_batch = RecsysBatch(
                user_hashes=matching_batch.user_hashes,  # Same user identity
                history_post_hashes=mismatched_batch_raw.history_post_hashes,  # Different history
                history_author_hashes=mismatched_batch_raw.history_author_hashes,
                history_actions=mismatched_batch_raw.history_actions,
                history_product_surface=mismatched_batch_raw.history_product_surface,
                candidate_post_hashes=matching_batch.candidate_post_hashes,  # Same candidate
                candidate_author_hashes=matching_batch.candidate_author_hashes,
                candidate_product_surface=matching_batch.candidate_product_surface,
            )

            matching_batches.append(matching_batch)
            mismatched_batches.append(mismatched_batch)

        if not matching_batches:
            return None

        combined_matching = self._stack_batches(matching_batches)
        combined_mismatched = self._stack_batches(mismatched_batches)

        return combined_matching, combined_mismatched, self.get_embedding_params()

    def get_validation_samples(
        self,
        num_samples: int = 100,
    ) -> list[tuple[int, int, list[int]]]:
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
