"""F2 Phase 2b: Full K/V Tensor Caching Runner.

This module provides a Phoenix runner that implements actual K/V tensor
caching for user context, enabling significant speedup when scoring multiple
candidate batches for the same user.

Architecture:
    ┌────────────────────────────────────────────────────────────────────┐
    │                     FullKVCachedRunner                             │
    │                                                                    │
    │  encode_user_context(batch, embeddings):                           │
    │    1. Build user + history embeddings                              │
    │    2. Run through CachingTransformer                               │
    │    3. Return FullKVCache with K,V for all layers                   │
    │                                                                    │
    │  score_with_cache(cache, batch, embeddings):                       │
    │    1. Build candidate embeddings only                              │
    │    2. Run through CachingTransformer with cached K,V               │
    │    3. Return scores (fast!)                                        │
    └────────────────────────────────────────────────────────────────────┘
"""

import hashlib
import time
from typing import NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from phoenix.grok import TransformerConfig, layer_norm
from phoenix.recsys_model import (
    HashConfig,
    PhoenixModelConfig,
    RecsysBatch,
    RecsysEmbeddings,
    RecsysModelOutput,
    block_candidate_reduce,
    block_history_reduce,
    block_user_reduce,
)
from phoenix.runners import ACTIONS, RankingOutput, RecsysInferenceRunner

from enhancements.optimization.caching_attention import LayerKVCache
from enhancements.optimization.caching_transformer import (
    CachingTransformer,
    FullKVCache,
    extract_user_context_from_cache,
)


class CacheStats(NamedTuple):
    """Statistics for full K/V cache performance."""
    hits: int
    misses: int
    encode_time_ms: float
    score_time_ms: float


def compute_user_hash(batch: RecsysBatch) -> int:
    """Compute hash of user context for cache invalidation."""
    user_bytes = np.asarray(batch.user_hashes).tobytes()
    history_bytes = np.asarray(batch.history_post_hashes).tobytes()
    combined = user_bytes + history_bytes
    return int(hashlib.md5(combined).hexdigest(), 16) % (2**63)


class CachingPhoenixModel(hk.Module):
    """Phoenix model with K/V caching support.

    This is a modified version of PhoenixModel that uses CachingTransformer
    and provides methods for incremental inference with cached user context.
    """

    def __init__(
        self,
        config: PhoenixModelConfig,
        fprop_dtype=jnp.bfloat16,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.config = config
        self.fprop_dtype = fprop_dtype

        # Create caching transformer
        tc = config.model
        self.transformer = CachingTransformer(
            num_q_heads=tc.num_q_heads,
            num_kv_heads=tc.num_kv_heads,
            key_size=tc.key_size,
            num_layers=tc.num_layers,
            widening_factor=tc.widening_factor,
            attn_output_multiplier=tc.attn_output_multiplier,
        )

    def _get_action_embeddings(self, actions: jax.Array) -> jax.Array:
        """Convert multi-hot action vectors to embeddings."""
        config = self.config
        _, _, num_actions = actions.shape
        D = config.emb_size

        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        action_projection = hk.get_parameter(
            "action_projection",
            [num_actions, D],
            dtype=jnp.float32,
            init=embed_init,
        )

        actions_signed = (2 * actions - 1).astype(jnp.float32)
        action_emb = jnp.dot(actions_signed.astype(action_projection.dtype), action_projection)
        valid_mask = jnp.any(actions, axis=-1, keepdims=True)
        action_emb = action_emb * valid_mask

        return action_emb.astype(self.fprop_dtype)

    def _single_hot_to_embeddings(
        self, input: jax.Array, vocab_size: int, emb_size: int, name: str
    ) -> jax.Array:
        """Convert single-hot indices to embeddings."""
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        embedding_table = hk.get_parameter(
            name, [vocab_size, emb_size], dtype=jnp.float32, init=embed_init
        )
        input_one_hot = jax.nn.one_hot(input, vocab_size)
        output = jnp.dot(input_one_hot, embedding_table)
        return output.astype(self.fprop_dtype)

    def _get_unembedding(self) -> jax.Array:
        """Get the unembedding matrix."""
        config = self.config
        embed_init = hk.initializers.VarianceScaling(1.0, mode="fan_out")
        unembed_mat = hk.get_parameter(
            "unembeddings",
            [config.emb_size, config.num_actions],
            dtype=jnp.float32,
            init=embed_init,
        )
        return unembed_mat

    def _compute_logits(self, embeddings: jax.Array) -> jax.Array:
        """Compute logits from embeddings - shared by all forward paths.

        This ensures the layer_norm and unembedding parameters have consistent
        paths regardless of which forward method calls this.
        """
        out_embeddings = layer_norm(embeddings)
        unembeddings = self._get_unembedding()
        logits = jnp.dot(out_embeddings.astype(unembeddings.dtype), unembeddings)
        return logits.astype(self.fprop_dtype)

    def build_user_context_embeddings(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> Tuple[jax.Array, jax.Array]:
        """Build embeddings for user + history only.

        Returns:
            user_context_embeddings: [B, 1 + history_len, D]
            user_context_mask: [B, 1 + history_len]
        """
        config = self.config
        hash_config = config.hash_config

        # User embeddings
        user_embeddings, user_padding_mask = block_user_reduce(
            batch.user_hashes,
            recsys_embeddings.user_embeddings,
            hash_config.num_user_hashes,
            config.emb_size,
            1.0,
        )

        # History embeddings
        history_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.history_product_surface,
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )
        history_actions_embeddings = self._get_action_embeddings(batch.history_actions)

        history_embeddings, history_padding_mask = block_history_reduce(
            batch.history_post_hashes,
            recsys_embeddings.history_post_embeddings,
            recsys_embeddings.history_author_embeddings,
            history_product_surface_embeddings,
            history_actions_embeddings,
            hash_config.num_item_hashes,
            hash_config.num_author_hashes,
            1.0,
        )

        # Concatenate user + history
        context_embeddings = jnp.concatenate([user_embeddings, history_embeddings], axis=1)
        context_mask = jnp.concatenate([user_padding_mask, history_padding_mask], axis=1)

        return context_embeddings.astype(self.fprop_dtype), context_mask

    def build_candidate_embeddings(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> Tuple[jax.Array, jax.Array]:
        """Build embeddings for candidates only.

        Returns:
            candidate_embeddings: [B, num_candidates, D]
            candidate_mask: [B, num_candidates]
        """
        config = self.config
        hash_config = config.hash_config

        candidate_product_surface_embeddings = self._single_hot_to_embeddings(
            batch.candidate_product_surface,
            config.product_surface_vocab_size,
            config.emb_size,
            "product_surface_embedding_table",
        )

        candidate_embeddings, candidate_padding_mask = block_candidate_reduce(
            batch.candidate_post_hashes,
            recsys_embeddings.candidate_post_embeddings,
            recsys_embeddings.candidate_author_embeddings,
            candidate_product_surface_embeddings,
            hash_config.num_item_hashes,
            hash_config.num_author_hashes,
            1.0,
        )

        return candidate_embeddings.astype(self.fprop_dtype), candidate_padding_mask

    def encode_user_context(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
        user_hash: int,
    ) -> FullKVCache:
        """Encode user context and return K/V cache.

        This runs the user + history through the transformer and caches
        the K/V tensors for reuse when scoring candidates.

        Args:
            batch: Full batch (only user/history parts used)
            recsys_embeddings: Full embeddings (only user/history parts used)
            user_hash: Hash for cache invalidation

        Returns:
            FullKVCache containing K/V for user context
        """
        context_emb, context_mask = self.build_user_context_embeddings(batch, recsys_embeddings)

        # Run through transformer (no candidate isolation, just causal)
        output = self.transformer(
            context_emb,
            context_mask,
            user_hash=user_hash,
            kv_cache=None,
            position_offset=0,
            candidate_start_offset=None,  # No candidates yet
        )

        return output.kv_cache

    def score_with_cache(
        self,
        cache: FullKVCache,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
    ) -> RecsysModelOutput:
        """Score candidates using cached user context.

        Args:
            cache: K/V cache from encode_user_context
            batch: Full batch (only candidate parts used)
            recsys_embeddings: Full embeddings (only candidate parts used)

        Returns:
            RecsysModelOutput with logits
        """
        cand_emb, cand_mask = self.build_candidate_embeddings(batch, recsys_embeddings)

        # Candidate start offset is the cached context length
        candidate_start_offset = cache.cached_len

        # Run candidates through transformer with cached user context
        output = self.transformer(
            cand_emb,
            cand_mask,
            user_hash=cache.user_hash,
            kv_cache=cache,
            position_offset=cache.cached_len,
            candidate_start_offset=candidate_start_offset,
        )

        # Compute logits using shared method
        logits = self._compute_logits(output.embeddings)

        return RecsysModelOutput(logits=logits)

    def forward_full(
        self,
        batch: RecsysBatch,
        recsys_embeddings: RecsysEmbeddings,
        user_hash: int,
    ) -> Tuple[RecsysModelOutput, FullKVCache]:
        """Full forward pass, returning both output and cache.

        This runs the complete sequence (user + history + candidates) and
        returns both the model output and a cache of user context K/V.

        Args:
            batch: Full batch
            recsys_embeddings: Full embeddings
            user_hash: Hash for cache

        Returns:
            Tuple of (RecsysModelOutput, FullKVCache for user context only)
        """
        # Build all embeddings
        context_emb, context_mask = self.build_user_context_embeddings(batch, recsys_embeddings)
        cand_emb, cand_mask = self.build_candidate_embeddings(batch, recsys_embeddings)

        # Concatenate
        full_emb = jnp.concatenate([context_emb, cand_emb], axis=1)
        full_mask = jnp.concatenate([context_mask, cand_mask], axis=1)

        context_len = context_emb.shape[1]

        # Run through transformer
        output = self.transformer(
            full_emb,
            full_mask,
            user_hash=user_hash,
            kv_cache=None,
            position_offset=0,
            candidate_start_offset=context_len,
        )

        # Extract candidate embeddings and compute logits using shared method
        candidate_embeddings = output.embeddings[:, context_len:, :]
        logits = self._compute_logits(candidate_embeddings)

        # Extract user context cache for future reuse
        user_context_cache = extract_user_context_from_cache(
            output.kv_cache, context_len, user_hash
        )

        return RecsysModelOutput(logits=logits), user_context_cache


class FullKVCachedRunner:
    """Phoenix runner with full K/V tensor caching.

    This runner implements actual K/V caching, storing transformer key/value
    tensors for user context and reusing them when scoring multiple candidate
    batches for the same user.

    Example:
        >>> runner = FullKVCachedRunner(model_config)
        >>> runner.initialize()
        >>>
        >>> # First call - cache miss
        >>> output1 = runner.rank(batch1, embeddings1)  # Encodes user + scores
        >>>
        >>> # Same user, new candidates - cache hit (FAST!)
        >>> output2 = runner.rank(batch2, embeddings2)  # Uses cached K/V
    """

    def __init__(self, model_config: PhoenixModelConfig):
        """Initialize the cached runner.

        Args:
            model_config: Phoenix model configuration
        """
        self.model_config = model_config
        self.params = None
        self._cache: Optional[FullKVCache] = None
        self._stats = CacheStats(hits=0, misses=0, encode_time_ms=0.0, score_time_ms=0.0)

        # Haiku functions will be set during initialize()
        self._encode_fn = None
        self._score_fn = None
        self._full_forward_fn = None

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    @property
    def cache(self) -> Optional[FullKVCache]:
        """Get current cache."""
        return self._cache

    def clear_cache(self) -> None:
        """Clear the K/V cache."""
        self._cache = None

    def initialize(self) -> None:
        """Initialize the runner by creating Haiku functions and params."""
        config = self.model_config
        config.initialize()

        # Create the caching model
        def make_model():
            return CachingPhoenixModel(config, fprop_dtype=jnp.bfloat16)

        # Define Haiku functions
        def hk_encode(batch, embeddings, user_hash):
            return make_model().encode_user_context(batch, embeddings, user_hash)

        def hk_score(cache_layers, cached_len, user_hash, batch, embeddings):
            cache = FullKVCache(
                layer_caches=cache_layers,
                cached_len=cached_len,
                user_hash=user_hash,
            )
            return make_model().score_with_cache(cache, batch, embeddings)

        def hk_full_forward(batch, embeddings, user_hash):
            return make_model().forward_full(batch, embeddings, user_hash)

        # Transform to pure functions
        self._encode_fn = hk.without_apply_rng(hk.transform(hk_encode))
        self._score_fn = hk.without_apply_rng(hk.transform(hk_score))
        self._full_forward_fn = hk.without_apply_rng(hk.transform(hk_full_forward))

        # Initialize parameters
        from phoenix.runners import create_example_batch

        dummy_batch, dummy_emb = create_example_batch(
            batch_size=1,
            emb_size=config.emb_size,
            history_len=config.history_seq_len,
            num_candidates=config.candidate_seq_len,
            num_actions=config.num_actions,
            num_user_hashes=config.hash_config.num_user_hashes,
            num_item_hashes=config.hash_config.num_item_hashes,
            num_author_hashes=config.hash_config.num_author_hashes,
            product_surface_vocab_size=config.product_surface_vocab_size,
        )

        rng = jax.random.PRNGKey(42)
        self.params = self._full_forward_fn.init(rng, dummy_batch, dummy_emb, 0)

    def _should_use_cache(self, batch: RecsysBatch) -> bool:
        """Check if cache is valid for this batch."""
        if self._cache is None:
            return False
        current_hash = compute_user_hash(batch)
        return self._cache.user_hash == current_hash

    def encode_user_context(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> FullKVCache:
        """Encode user context and return K/V cache."""
        user_hash = compute_user_hash(batch)
        return self._encode_fn.apply(self.params, batch, embeddings, user_hash)

    def score_with_cache(
        self,
        cache: FullKVCache,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """Score candidates using cached user context."""
        output = self._score_fn.apply(
            self.params,
            cache.layer_caches,
            cache.cached_len,
            cache.user_hash,
            batch,
            embeddings,
        )
        return self._output_to_ranking(output)

    def rank(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
        use_cache: bool = True,
    ) -> RankingOutput:
        """Rank candidates with automatic caching.

        Args:
            batch: Input batch
            embeddings: Input embeddings
            use_cache: Whether to use caching

        Returns:
            RankingOutput with scores and rankings
        """
        if not use_cache:
            user_hash = compute_user_hash(batch)
            output, _ = self._full_forward_fn.apply(self.params, batch, embeddings, user_hash)
            return self._output_to_ranking(output)

        if self._should_use_cache(batch):
            # Cache hit
            start = time.perf_counter()
            result = self.score_with_cache(self._cache, batch, embeddings)
            elapsed = (time.perf_counter() - start) * 1000

            self._stats = CacheStats(
                hits=self._stats.hits + 1,
                misses=self._stats.misses,
                encode_time_ms=self._stats.encode_time_ms,
                score_time_ms=elapsed,
            )
            return result
        else:
            # Cache miss - full forward and cache user context
            start = time.perf_counter()
            user_hash = compute_user_hash(batch)
            output, cache = self._full_forward_fn.apply(self.params, batch, embeddings, user_hash)
            elapsed = (time.perf_counter() - start) * 1000

            self._cache = cache
            self._stats = CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses + 1,
                encode_time_ms=elapsed,
                score_time_ms=0.0,
            )
            return self._output_to_ranking(output)

    def _output_to_ranking(self, output: RecsysModelOutput) -> RankingOutput:
        """Convert model output to RankingOutput."""
        logits = output.logits
        probs = jax.nn.sigmoid(logits)
        primary_scores = probs[:, :, 0]
        ranked_indices = jnp.argsort(-primary_scores, axis=-1)

        return RankingOutput(
            scores=probs,
            ranked_indices=ranked_indices,
            p_favorite_score=probs[:, :, 0],
            p_reply_score=probs[:, :, 1],
            p_repost_score=probs[:, :, 2],
            p_photo_expand_score=probs[:, :, 3],
            p_click_score=probs[:, :, 4],
            p_profile_click_score=probs[:, :, 5],
            p_vqv_score=probs[:, :, 6],
            p_share_score=probs[:, :, 7],
            p_share_via_dm_score=probs[:, :, 8],
            p_share_via_copy_link_score=probs[:, :, 9],
            p_dwell_score=probs[:, :, 10],
            p_quote_score=probs[:, :, 11],
            p_quoted_click_score=probs[:, :, 12],
            p_follow_author_score=probs[:, :, 13],
            p_not_interested_score=probs[:, :, 14],
            p_block_author_score=probs[:, :, 15],
            p_mute_author_score=probs[:, :, 16],
            p_report_score=probs[:, :, 17],
            p_dwell_time=probs[:, :, 18],
        )
