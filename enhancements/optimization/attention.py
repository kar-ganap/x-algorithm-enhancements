"""Efficient Attention Implementation for Phoenix.

This module implements memory-efficient attention that exploits Phoenix's
candidate isolation mask structure:
- Context tokens (user + history): attend to everything
- Candidate tokens: attend to context + self only (not other candidates)

By exploiting this structure, we reduce memory from O((context+candidates)²)
to O(context²) + O(context × candidates), and candidate-to-candidate from
O(candidates²) to O(candidates) by only computing the diagonal.

For very long context sequences, we also provide an optional flash-attention
style implementation using online softmax.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class EfficientAttentionOutput(NamedTuple):
    """Output of efficient attention."""
    output: jax.Array  # [batch, seq_len, num_heads, head_dim]
    # Optional debug info
    context_attn_weights: jax.Array | None = None
    candidate_attn_weights: jax.Array | None = None


def standard_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None = None,
    scale: float = 1.0,
    attn_logit_cap: float = 30.0,
) -> jax.Array:
    """Standard scaled dot-product attention.

    Args:
        query: [batch, seq_q, num_heads, head_dim]
        key: [batch, seq_k, num_heads, head_dim]
        value: [batch, seq_k, num_heads, head_dim]
        mask: Optional [batch, 1, seq_q, seq_k] or broadcastable
        scale: Scaling factor for attention logits
        attn_logit_cap: Cap for attention logits (tanh-based)

    Returns:
        Output of shape [batch, seq_q, num_heads, head_dim]
    """
    # Compute attention scores: [batch, num_heads, seq_q, seq_k]
    attn_logits = jnp.einsum('bqhd,bkhd->bhqk', query, key).astype(jnp.float32)
    attn_logits = attn_logits * scale

    # Apply tanh capping for stability (matching Phoenix)
    attn_logits = attn_logit_cap * jnp.tanh(attn_logits / attn_logit_cap)

    # Apply mask
    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -1e30)

    # Softmax
    attn_weights = jax.nn.softmax(attn_logits, axis=-1).astype(query.dtype)

    # Weighted sum of values
    output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)

    return output


def efficient_phoenix_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    context_len: int,
    scale: float = 1.0,
    attn_logit_cap: float = 30.0,
    return_weights: bool = False,
) -> EfficientAttentionOutput:
    """Memory-efficient attention exploiting Phoenix's mask structure.

    Phoenix's attention mask has a special structure:
    - Context tokens (user + history): can attend to all tokens (causal)
    - Candidate tokens: can attend to context + self only (not other candidates)

    This function exploits this structure to reduce memory:
    - Context self-attention: O(context²) - standard attention
    - Candidate attention: O(candidates × context) + O(candidates) for diagonal

    Args:
        query: [batch, seq_len, num_heads, head_dim]
        key: [batch, seq_len, num_heads, head_dim]
        value: [batch, seq_len, num_heads, head_dim]
        context_len: Number of context tokens (user + history)
        scale: Scaling factor for attention logits
        attn_logit_cap: Cap for attention logits (tanh-based)
        return_weights: Whether to return attention weights for debugging

    Returns:
        EfficientAttentionOutput with output shape [batch, seq_len, num_heads, head_dim]
    """
    batch_size, seq_len, num_heads, head_dim = query.shape
    num_candidates = seq_len - context_len

    # Validate inputs
    assert context_len > 0, "context_len must be positive"
    assert num_candidates >= 0, "num_candidates cannot be negative"

    # Split into context and candidate parts
    Q_ctx = query[:, :context_len]      # [batch, context_len, heads, dim]
    Q_cand = query[:, context_len:]     # [batch, num_candidates, heads, dim]

    K_ctx = key[:, :context_len]
    K_cand = key[:, context_len:]

    V_ctx = value[:, :context_len]
    V_cand = value[:, context_len:]

    # ============================================
    # CONTEXT TOKENS: Causal attention (context only)
    # ============================================
    # Context tokens can ONLY attend to earlier context tokens (causal)
    # They CANNOT see candidates (due to causal mask)
    causal_mask_ctx = jnp.tril(jnp.ones((1, 1, context_len, context_len)))

    # Context attending to context (causal)
    ctx_to_ctx_logits = jnp.einsum('bqhd,bkhd->bhqk', Q_ctx, K_ctx).astype(jnp.float32)
    ctx_to_ctx_logits = ctx_to_ctx_logits * scale
    ctx_to_ctx_logits = attn_logit_cap * jnp.tanh(ctx_to_ctx_logits / attn_logit_cap)
    ctx_to_ctx_logits = jnp.where(causal_mask_ctx, ctx_to_ctx_logits, -1e30)

    # Softmax only over context (no candidates visible to context tokens)
    ctx_weights = jax.nn.softmax(ctx_to_ctx_logits, axis=-1).astype(query.dtype)

    # Compute output for context tokens
    out_ctx = jnp.einsum('bhqk,bkhd->bqhd', ctx_weights, V_ctx)

    # ============================================
    # CANDIDATE TOKENS: Context + self only
    # ============================================
    if num_candidates > 0:
        # Candidates attending to context (all allowed)
        cand_to_ctx_logits = jnp.einsum('bqhd,bkhd->bhqk', Q_cand, K_ctx).astype(jnp.float32)
        cand_to_ctx_logits = cand_to_ctx_logits * scale
        cand_to_ctx_logits = attn_logit_cap * jnp.tanh(cand_to_ctx_logits / attn_logit_cap)
        # Shape: [batch, heads, num_candidates, context_len]

        # Candidates attending to self only (diagonal of Q_cand @ K_cand.T)
        # Instead of computing full [num_candidates, num_candidates] matrix,
        # just compute the diagonal: sum(Q_cand * K_cand, axis=-1)
        cand_self_logits = jnp.sum(Q_cand * K_cand, axis=-1).astype(jnp.float32)
        # Shape: [batch, num_candidates, heads]
        cand_self_logits = cand_self_logits * scale
        cand_self_logits = attn_logit_cap * jnp.tanh(cand_self_logits / attn_logit_cap)
        # Transpose to [batch, heads, num_candidates, 1] for concatenation
        cand_self_logits = jnp.transpose(cand_self_logits, (0, 2, 1))[:, :, :, None]

        # Combine: [batch, heads, num_candidates, context_len + 1]
        cand_all_logits = jnp.concatenate([cand_to_ctx_logits, cand_self_logits], axis=-1)
        cand_weights = jax.nn.softmax(cand_all_logits, axis=-1).astype(query.dtype)

        # Split weights
        cand_weights_to_ctx = cand_weights[:, :, :, :context_len]
        cand_weights_to_self = cand_weights[:, :, :, -1:]  # [batch, heads, num_cand, 1]

        # Compute output
        # Contribution from context
        out_cand_from_ctx = jnp.einsum('bhqk,bkhd->bqhd', cand_weights_to_ctx, V_ctx)

        # Contribution from self (element-wise, each candidate uses its own V)
        # cand_weights_to_self: [batch, heads, num_cand, 1]
        # V_cand: [batch, num_cand, heads, dim]
        # Need to multiply weight by corresponding value
        cand_weights_to_self_t = jnp.transpose(cand_weights_to_self[:, :, :, 0], (0, 2, 1))
        # Shape: [batch, num_cand, heads]
        out_cand_from_self = cand_weights_to_self_t[:, :, :, None] * V_cand

        out_cand = out_cand_from_ctx + out_cand_from_self
    else:
        out_cand = jnp.zeros((batch_size, 0, num_heads, head_dim), dtype=query.dtype)
        cand_weights = None

    # ============================================
    # Combine outputs
    # ============================================
    output = jnp.concatenate([out_ctx, out_cand], axis=1)

    if return_weights:
        return EfficientAttentionOutput(
            output=output,
            context_attn_weights=ctx_weights,
            candidate_attn_weights=cand_weights,
        )

    return EfficientAttentionOutput(output=output)


def flash_attention_1d(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None = None,
    scale: float = 1.0,
    block_size: int = 64,
) -> jax.Array:
    """Flash attention using online softmax algorithm.

    This implementation computes attention in blocks without materializing
    the full attention matrix, using the online softmax algorithm.

    NOTE: This is a reference implementation. On CPU, it may not be faster
    than standard attention. The benefits are primarily on GPU with long
    sequences where memory bandwidth is the bottleneck.

    Args:
        query: [batch, seq_q, num_heads, head_dim]
        key: [batch, seq_k, num_heads, head_dim]
        value: [batch, seq_k, num_heads, head_dim]
        mask: Optional [batch, 1, seq_q, seq_k]
        scale: Scaling factor
        block_size: Size of blocks for tiled computation

    Returns:
        Output of shape [batch, seq_q, num_heads, head_dim]
    """
    batch_size, seq_q, num_heads, head_dim = query.shape
    _, seq_k, _, _ = key.shape

    # Initialize output and running statistics
    # Using float32 for numerical stability in running stats
    output = jnp.zeros((batch_size, seq_q, num_heads, head_dim), dtype=query.dtype)
    m = jnp.full((batch_size, seq_q, num_heads), -1e30, dtype=jnp.float32)  # running max
    l = jnp.zeros((batch_size, seq_q, num_heads), dtype=jnp.float32)  # running sum

    # Process key/value in blocks
    num_blocks = (seq_k + block_size - 1) // block_size

    for block_idx in range(num_blocks):
        start_k = block_idx * block_size
        end_k = min(start_k + block_size, seq_k)

        K_block = key[:, start_k:end_k]  # [batch, block_size, heads, dim]
        V_block = value[:, start_k:end_k]

        # Compute attention scores for this block
        # [batch, seq_q, heads, block_size]
        scores = jnp.einsum('bqhd,bkhd->bqhk', query, K_block).astype(jnp.float32)
        scores = scores * scale

        # Apply mask if provided
        # mask shape: [batch, 1, seq_q, seq_k] -> need [batch, seq_q, 1, block_size]
        if mask is not None:
            mask_block = mask[:, :, :, start_k:end_k]  # [batch, 1, seq_q, block_size]
            mask_block = jnp.transpose(mask_block, (0, 2, 1, 3))  # [batch, seq_q, 1, block_size]
            scores = jnp.where(mask_block, scores, -1e30)

        # Online softmax update
        # New max: max of old max and block max
        block_max = jnp.max(scores, axis=-1)  # [batch, seq_q, heads]
        m_new = jnp.maximum(m, block_max)

        # Correction factor for old values
        correction = jnp.exp(m - m_new)

        # Exponentials with new max
        exp_scores = jnp.exp(scores - m_new[:, :, :, None])

        # Update running sum
        l_new = l * correction + jnp.sum(exp_scores, axis=-1)

        # Update output
        # Scale old output by correction factor and normalization change
        output = output * (l * correction / l_new)[:, :, :, None]
        # Add contribution from this block
        output = output + jnp.einsum('bqhk,bkhd->bqhd', exp_scores / l_new[:, :, :, None], V_block)

        m, l = m_new, l_new

    return output.astype(query.dtype)


def compute_attention_memory_bytes(
    batch_size: int,
    seq_len: int,
    num_heads: int,
    dtype_bytes: int = 4,
) -> int:
    """Compute memory needed for full attention matrix.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        num_heads: Number of attention heads
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)

    Returns:
        Memory in bytes for the full attention matrix
    """
    return batch_size * num_heads * seq_len * seq_len * dtype_bytes


def compute_efficient_attention_memory_bytes(
    batch_size: int,
    context_len: int,
    num_candidates: int,
    num_heads: int,
    dtype_bytes: int = 4,
) -> int:
    """Compute memory needed for efficient Phoenix attention.

    Args:
        batch_size: Batch size
        context_len: Context sequence length
        num_candidates: Number of candidates
        num_heads: Number of attention heads
        dtype_bytes: Bytes per element

    Returns:
        Memory in bytes for efficient attention matrices
    """
    # Context self-attention: context × context (causal, but we store full for simplicity)
    ctx_self = context_len * context_len
    # Context does NOT attend to candidates (causal mask prevents this)
    # Candidates to context: candidates × context
    cand_to_ctx = num_candidates * context_len
    # Candidates to self: candidates (diagonal only)
    cand_self = num_candidates

    total_elements = ctx_self + cand_to_ctx + cand_self
    return batch_size * num_heads * total_elements * dtype_bytes


def memory_reduction_factor(
    context_len: int,
    num_candidates: int,
) -> float:
    """Compute memory reduction factor from using efficient attention.

    Args:
        context_len: Context sequence length
        num_candidates: Number of candidates

    Returns:
        Factor by which memory is reduced (standard / efficient)
    """
    seq_len = context_len + num_candidates

    # Standard attention memory
    standard = seq_len * seq_len

    # Efficient attention memory
    # Note: context tokens don't attend to candidates (causal mask)
    efficient = (context_len * context_len +  # context self (causal)
                 num_candidates * context_len +  # cand to context
                 num_candidates)  # cand self (diagonal)

    return standard / efficient
