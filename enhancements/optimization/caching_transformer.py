"""F2 Phase 2b: Caching Transformer Module.

This module provides a transformer wrapper that manages K,V caching across
all layers, enabling efficient incremental inference.

Architecture:
    ┌──────────────────────────────────────────┐
    │ CachingTransformer                       │
    │  ├── Layer 0: CachingDecoderLayer        │
    │  │    └── CachingMHABlock                │
    │  │         └── CachingMultiHeadAttention │
    │  ├── Layer 1: CachingDecoderLayer        │
    │  │    └── ...                            │
    │  └── Layer N-1: ...                      │
    │                                          │
    │  Returns: (output, FullKVCache)          │
    └──────────────────────────────────────────┘
"""

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from enhancements.optimization.caching_attention import (
    CachingMultiHeadAttention,
    LayerKVCache,
)
from phoenix.grok import DenseBlock, RMSNorm, make_recsys_attn_mask


class FullKVCache(NamedTuple):
    """Complete K,V cache across all transformer layers.

    Attributes:
        layer_caches: Tuple of LayerKVCache, one per transformer layer
        cached_len: Number of sequence positions cached (user context length)
        user_hash: Hash of user context for cache invalidation
    """
    layer_caches: tuple[LayerKVCache, ...]
    cached_len: int
    user_hash: int


class CachingTransformerOutput(NamedTuple):
    """Output from caching transformer forward pass."""
    embeddings: jax.Array
    kv_cache: FullKVCache


def hk_rms_norm(x: jax.Array, fixed_scale: bool = False) -> jax.Array:
    """Apply RMS normalization."""
    ln = RMSNorm(axis=-1, create_scale=not fixed_scale)
    return ln(x)


class CachingMHABlock(hk.Module):
    """MHA Block with K,V caching support."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        attn_output_multiplier: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.attn_output_multiplier = attn_output_multiplier

    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        kv_cache: LayerKVCache | None = None,
        position_offset: int = 0,
    ) -> tuple[jax.Array, LayerKVCache]:
        """Forward pass with optional K,V cache.

        Args:
            inputs: Input tensor [batch, seq_len, model_size]
            mask: Attention mask
            kv_cache: Optional K,V cache from previous forward
            position_offset: Position offset for RoPE (when using cache)

        Returns:
            Tuple of (attention output, new K,V cache)
        """
        _, _, model_size = inputs.shape
        side_input = inputs

        attn = CachingMultiHeadAttention(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            model_size=model_size,
            attn_output_multiplier=self.attn_output_multiplier,
        )

        output = attn(
            inputs, side_input, side_input, mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )

        return output.embeddings, output.kv_cache


class CachingDecoderLayer(hk.Module):
    """Transformer decoder layer with K,V caching support."""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        widening_factor: float = 4.0,
        attn_output_multiplier: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.widening_factor = widening_factor
        self.attn_output_multiplier = attn_output_multiplier

    def __call__(
        self,
        inputs: jax.Array,
        mask: jax.Array,
        kv_cache: LayerKVCache | None = None,
        position_offset: int = 0,
    ) -> tuple[jax.Array, LayerKVCache]:
        """Forward pass with optional K,V cache.

        Args:
            inputs: Input tensor [batch, seq_len, model_size]
            mask: Attention mask
            kv_cache: Optional K,V cache
            position_offset: Position offset for RoPE

        Returns:
            Tuple of (layer output, new K,V cache)
        """
        h = inputs

        # Pre-norm attention
        h_normed = hk_rms_norm(h)
        h_attn, new_kv_cache = CachingMHABlock(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            attn_output_multiplier=self.attn_output_multiplier,
        )(h_normed, mask, kv_cache=kv_cache, position_offset=position_offset)

        h_attn = hk_rms_norm(h_attn)
        h = h + h_attn

        # Pre-norm FFN
        h_normed = hk_rms_norm(h)
        h_dense = DenseBlock(
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            key_size=self.key_size,
            widening_factor=self.widening_factor,
        )(h_normed)

        h_dense = hk_rms_norm(h_dense)
        h = h + h_dense

        return h, new_kv_cache


class CachingTransformer(hk.Module):
    """Transformer with full K,V caching across all layers.

    This transformer can:
    1. Run a full forward pass and return K,V cache for all layers
    2. Run an incremental forward using cached K,V

    Example:
        # Full forward (cache miss)
        output, cache = transformer(embeddings, mask, user_hash=123)

        # Incremental forward (cache hit)
        output, cache = transformer(
            new_embeddings, mask,
            kv_cache=cache,
            position_offset=cache.cached_len,
            user_hash=123
        )
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        num_layers: int,
        widening_factor: float = 4.0,
        attn_output_multiplier: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.attn_output_multiplier = attn_output_multiplier

    def __call__(
        self,
        embeddings: jax.Array,
        mask: jax.Array,
        user_hash: int,
        kv_cache: FullKVCache | None = None,
        position_offset: int = 0,
        candidate_start_offset: int | None = None,
    ) -> CachingTransformerOutput:
        """Forward pass with K,V caching.

        Args:
            embeddings: Input embeddings [batch, seq_len, model_size]
            mask: Padding mask [batch, seq_len]
            user_hash: Hash for cache invalidation
            kv_cache: Optional full K,V cache from previous forward
            position_offset: Position offset for RoPE (when using cache)
            candidate_start_offset: Offset for candidate isolation mask

        Returns:
            CachingTransformerOutput with embeddings and updated cache
        """
        fprop_dtype = embeddings.dtype
        _, seq_len, _ = embeddings.shape

        # Build attention mask
        padding_mask = mask[:, None, None, :]  # [B, 1, 1, T]

        if kv_cache is not None:
            # Using cache: need to handle mask for [query_len, full_key_len]
            total_len = kv_cache.cached_len + seq_len

            if candidate_start_offset is not None:
                # Candidates attending to cached context + themselves
                attn_mask = make_recsys_attn_mask(total_len, candidate_start_offset, fprop_dtype)
                # Slice to [query_len, full_len]
                attn_mask = attn_mask[:, :, -seq_len:, :]
            else:
                # Causal mask for incremental decoding
                # New tokens can attend to all previous tokens
                attn_mask = jnp.ones((1, 1, seq_len, total_len), dtype=fprop_dtype)
                # Apply causal constraint for new tokens among themselves
                causal_part = jnp.tril(jnp.ones((seq_len, seq_len), dtype=fprop_dtype))
                attn_mask = attn_mask.at[:, :, :, -seq_len:].set(causal_part)

            # Extend padding mask for full sequence
            # Assume cached positions are valid
            cached_padding = jnp.ones((mask.shape[0], 1, 1, kv_cache.cached_len), dtype=fprop_dtype)
            new_padding = padding_mask
            full_padding_mask = jnp.concatenate([cached_padding, new_padding], axis=-1)
            attn_mask = attn_mask * full_padding_mask
        else:
            # No cache: full attention mask
            if candidate_start_offset is not None:
                attn_mask = make_recsys_attn_mask(seq_len, candidate_start_offset, fprop_dtype)
            else:
                attn_mask = jnp.tril(jnp.ones((1, 1, seq_len, seq_len), dtype=fprop_dtype))

            attn_mask = attn_mask * padding_mask

        # Process through layers
        h = embeddings
        new_layer_caches = []

        for i in range(self.num_layers):
            layer_cache = kv_cache.layer_caches[i] if kv_cache is not None else None

            h, new_cache = CachingDecoderLayer(
                num_q_heads=self.num_q_heads,
                num_kv_heads=self.num_kv_heads,
                key_size=self.key_size,
                widening_factor=self.widening_factor,
                attn_output_multiplier=self.attn_output_multiplier,
                name=f"decoder_layer_{i}",
            )(h, attn_mask, kv_cache=layer_cache, position_offset=position_offset)

            new_layer_caches.append(new_cache)

        # Compute new cached length
        if kv_cache is not None:
            new_cached_len = kv_cache.cached_len + seq_len
        else:
            new_cached_len = seq_len

        full_cache = FullKVCache(
            layer_caches=tuple(new_layer_caches),
            cached_len=new_cached_len,
            user_hash=user_hash,
        )

        return CachingTransformerOutput(embeddings=h, kv_cache=full_cache)


def extract_user_context_from_cache(
    full_cache: FullKVCache,
    context_len: int,
    user_hash: int,
) -> FullKVCache:
    """Extract user context (first context_len positions) from full cache.

    After a full forward pass that includes candidates, this extracts just
    the user context portion for reuse with new candidate batches.

    Args:
        full_cache: Full K,V cache from complete forward pass
        context_len: Length of user context (user + history)
        user_hash: User hash for the new cache

    Returns:
        FullKVCache containing only user context K,V
    """
    extracted_layers = []
    for layer_cache in full_cache.layer_caches:
        extracted_layers.append(LayerKVCache(
            keys=layer_cache.keys[:, :, :context_len, :],
            values=layer_cache.values[:, :, :context_len, :],
        ))

    return FullKVCache(
        layer_caches=tuple(extracted_layers),
        cached_len=context_len,
        user_hash=user_hash,
    )
