"""F2 Phase 2b: Caching Attention Module.

This module provides a modified MultiHeadAttention that supports K,V caching
for incremental inference. When the user context is cached, only candidate
tokens need to be processed through the attention mechanism.

Key modifications from Phoenix's original attention:
1. Returns K,V tensors for caching
2. Accepts pre-computed K,V cache
3. Handles RoPE position offsets for cached tokens
"""

from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp

from phoenix.grok import Linear, RotaryEmbedding


class LayerKVCache(NamedTuple):
    """K,V cache for a single transformer layer.

    Stores pre-computed key and value projections with RoPE already applied.

    Attributes:
        keys: Shape [batch, num_kv_heads, seq_len, head_dim]
        values: Shape [batch, num_kv_heads, seq_len, head_dim]
    """
    keys: jax.Array
    values: jax.Array


class CachingMHAOutput(NamedTuple):
    """Output from caching multi-head attention."""
    embeddings: jax.Array
    kv_cache: LayerKVCache


class CachingMultiHeadAttention(hk.Module):
    """Multi-head attention with K,V caching support.

    This is a modified version of Phoenix's MultiHeadAttention that:
    1. Can return K,V tensors for external caching
    2. Can accept pre-computed K,V and concatenate with new K,V
    3. Correctly handles RoPE position offsets when using cache

    Example usage:
        # First call (cache miss) - encode full sequence
        output, kv_cache = attention(query, key, value, mask)

        # Subsequent calls (cache hit) - only process new tokens
        output, new_cache = attention(
            query_new, key_new, value_new, mask,
            kv_cache=kv_cache,
            position_offset=cached_seq_len
        )
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        key_size: int,
        *,
        with_bias: bool = False,
        value_size: int | None = None,
        model_size: int | None = None,
        attn_output_multiplier: float = 1.0,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.key_size = key_size
        self.value_size = value_size or key_size
        self.model_size = model_size or key_size * num_q_heads
        self.attn_output_multiplier = attn_output_multiplier
        self.with_bias = with_bias

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: jax.Array,
        kv_cache: LayerKVCache | None = None,
        position_offset: int = 0,
    ) -> CachingMHAOutput:
        """Compute attention with optional K,V caching.

        Args:
            query: Query tensor [batch, seq_len, model_size]
            key: Key tensor [batch, seq_len, model_size]
            value: Value tensor [batch, seq_len, model_size]
            mask: Attention mask [batch, 1, seq_len, total_seq_len]
            kv_cache: Optional pre-computed K,V cache
            position_offset: RoPE position offset for new tokens (used with cache)

        Returns:
            CachingMHAOutput with attention output and updated K,V cache
        """
        projection = self._linear_projection

        # Project Q, K, V
        query_heads = projection(query, self.key_size, self.num_q_heads, name="query")
        key_heads = projection(key, self.key_size, self.num_kv_heads, name="key")
        value_heads = projection(value, self.value_size, self.num_kv_heads, name="value")

        # Apply RoPE with position offset
        # When using cache, new tokens start at position_offset
        rotate = RotaryEmbedding(dim=self.key_size, base_exponent=int(1e4))
        key_heads = rotate(key_heads, seq_dim=1, offset=position_offset)
        query_heads = rotate(query_heads, seq_dim=1, offset=position_offset)

        # Transpose to [batch, heads, seq, dim] for caching
        # Current shape: [batch, seq, heads, dim]
        key_heads_transposed = jnp.transpose(key_heads, (0, 2, 1, 3))
        value_heads_transposed = jnp.transpose(value_heads, (0, 2, 1, 3))

        # Handle K,V cache
        if kv_cache is not None:
            # Concatenate cached K,V with new K,V
            # Cached: [batch, heads, cached_len, dim]
            # New: [batch, heads, new_len, dim]
            key_heads_transposed = jnp.concatenate(
                [kv_cache.keys, key_heads_transposed], axis=2
            )
            value_heads_transposed = jnp.concatenate(
                [kv_cache.values, value_heads_transposed], axis=2
            )

        # Create new cache (storing transposed format)
        new_cache = LayerKVCache(
            keys=key_heads_transposed,
            values=value_heads_transposed,
        )

        # Transpose back for attention computation
        # [batch, heads, seq, dim] -> [batch, seq, heads, dim]
        key_heads_for_attn = jnp.transpose(key_heads_transposed, (0, 2, 1, 3))
        value_heads_for_attn = jnp.transpose(value_heads_transposed, (0, 2, 1, 3))

        # Compute attention
        b, t, h, d = query_heads.shape
        _, T, kv_h, _ = key_heads_for_attn.shape  # T is total seq len (cached + new)

        assert h % kv_h == 0, f"query_heads {h} must be a multiple of kv_heads {kv_h}"

        # Reshape query for grouped-query attention
        query_heads = jnp.reshape(query_heads, (b, t, kv_h, h // kv_h, d))

        # Compute attention weights
        attn_logits = jnp.einsum(
            "...thHd,...Thd->...hHtT", query_heads, key_heads_for_attn
        ).astype(jnp.float32)

        attn_logits *= self.attn_output_multiplier

        # Soft cap attention logits
        max_attn_val = jnp.array(30.0, dtype=attn_logits.dtype)
        attn_logits = max_attn_val * jnp.tanh(attn_logits / max_attn_val)

        # Apply mask
        # Mask shape needs to match: [batch, 1, num_groups, query_len, key_len]
        mask = mask[:, :, None, :, :]

        if mask is not None:
            attn_logits = jnp.where(mask, attn_logits, -1e30)

        attn_weights = jax.nn.softmax(attn_logits).astype(query.dtype)

        # Apply attention to values
        attn = jnp.einsum("...hHtT,...Thd->...thHd", attn_weights, value_heads_for_attn)

        # Reshape and project output
        leading_dims = attn.shape[:2]
        attn = jnp.reshape(attn, (*leading_dims, -1))

        final_projection = Linear(self.model_size, with_bias=False)
        output = final_projection(attn)

        return CachingMHAOutput(embeddings=output, kv_cache=new_cache)

    @hk.transparent
    def _linear_projection(
        self,
        x: jax.Array,
        head_size: int,
        num_heads: int,
        name: str | None = None,
    ) -> jax.Array:
        """Project input to multi-head format."""
        y = Linear(num_heads * head_size, with_bias=False, name=name)(x)
        *leading_dims, _ = x.shape
        return y.reshape((*leading_dims, num_heads, head_size))


def extract_user_context_cache(
    full_cache: LayerKVCache,
    context_len: int,
) -> LayerKVCache:
    """Extract only the user context portion from a full K,V cache.

    Args:
        full_cache: K,V cache from full sequence [batch, heads, full_len, dim]
        context_len: Length of user context to extract

    Returns:
        LayerKVCache containing only positions 0:context_len
    """
    return LayerKVCache(
        keys=full_cache.keys[:, :, :context_len, :],
        values=full_cache.values[:, :, :context_len, :],
    )
