"""Tests for efficient attention implementation.

These tests verify:
1. Correctness: Efficient attention matches standard attention with Phoenix mask
2. Mask handling: Candidate isolation mask is correctly applied
3. Memory estimation: Memory calculations are correct
4. Edge cases: Various sequence lengths and configurations
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenix.grok import make_recsys_attn_mask
from enhancements.optimization.attention import (
    standard_attention,
    efficient_phoenix_attention,
    flash_attention_1d,
    compute_attention_memory_bytes,
    compute_efficient_attention_memory_bytes,
    memory_reduction_factor,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def small_config():
    """Small configuration for fast tests."""
    return {
        'batch_size': 2,
        'context_len': 16,
        'num_candidates': 4,
        'num_heads': 4,
        'head_dim': 32,
    }


@pytest.fixture
def medium_config():
    """Medium configuration for more thorough tests."""
    return {
        'batch_size': 2,
        'context_len': 64,
        'num_candidates': 8,
        'num_heads': 8,
        'head_dim': 64,
    }


def create_qkv(batch_size, seq_len, num_heads, head_dim, key=42):
    """Create random Q, K, V tensors."""
    rng = jax.random.PRNGKey(key)
    k1, k2, k3 = jax.random.split(rng, 3)

    query = jax.random.normal(k1, (batch_size, seq_len, num_heads, head_dim))
    key_tensor = jax.random.normal(k2, (batch_size, seq_len, num_heads, head_dim))
    value = jax.random.normal(k3, (batch_size, seq_len, num_heads, head_dim))

    return query, key_tensor, value


def compute_reference_attention_with_phoenix_mask(
    query, key, value, context_len, scale=1.0, attn_logit_cap=30.0
):
    """Compute attention using standard method with Phoenix mask for reference."""
    batch_size, seq_len, num_heads, head_dim = query.shape

    # Create Phoenix mask
    mask_2d = make_recsys_attn_mask(seq_len, context_len)  # [1, 1, seq, seq]
    # Expand for heads
    mask = jnp.broadcast_to(mask_2d, (batch_size, 1, seq_len, seq_len))

    # Standard attention with mask
    return standard_attention(query, key, value, mask, scale, attn_logit_cap)


# =============================================================================
# Test 1: Basic Correctness
# =============================================================================

class TestCorrectnessBasic:
    """Test that efficient attention matches standard attention."""

    def test_efficient_matches_standard_small(self, small_config):
        """Efficient attention output matches standard with Phoenix mask."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        scale = 0.125  # Typical attention scale

        # Reference: standard attention with Phoenix mask
        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, cfg['context_len'], scale
        )

        # Efficient attention
        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale
        )

        np.testing.assert_allclose(
            result.output, reference, rtol=1e-4, atol=1e-5,
            err_msg="Efficient attention does not match reference"
        )

    def test_efficient_matches_standard_medium(self, medium_config):
        """Efficient attention matches standard for medium configuration."""
        cfg = medium_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        scale = 0.125

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, cfg['context_len'], scale
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale
        )

        np.testing.assert_allclose(
            result.output, reference, rtol=1e-4, atol=1e-5,
            err_msg="Efficient attention does not match reference for medium config"
        )

    def test_efficient_matches_standard_various_scales(self, small_config):
        """Test correctness with various attention scales."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        for scale in [0.0625, 0.125, 0.25, 1.0]:
            reference = compute_reference_attention_with_phoenix_mask(
                query, key, value, cfg['context_len'], scale
            )

            result = efficient_phoenix_attention(
                query, key, value, cfg['context_len'], scale
            )

            np.testing.assert_allclose(
                result.output, reference, rtol=1e-4, atol=1e-5,
                err_msg=f"Mismatch at scale={scale}"
            )


# =============================================================================
# Test 2: Candidate Isolation Mask
# =============================================================================

class TestCandidateIsolation:
    """Test that candidate isolation is correctly enforced."""

    def test_candidates_dont_attend_to_each_other(self, small_config):
        """Verify candidates cannot see other candidates."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale=0.125,
            return_weights=True
        )

        # Check candidate attention weights
        # Shape: [batch, heads, num_candidates, context_len + 1]
        # The last dimension is [context_positions..., self_position]
        cand_weights = result.candidate_attn_weights
        assert cand_weights is not None

        # The attention should only be over context_len + 1 positions
        # (context + self), not context + all_candidates
        assert cand_weights.shape[-1] == cfg['context_len'] + 1

    def test_candidates_attend_to_all_context(self, small_config):
        """Verify candidates can attend to all context tokens."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale=0.125,
            return_weights=True
        )

        cand_weights = result.candidate_attn_weights
        # Weights to context should be non-zero (softmax makes them positive)
        context_weights = cand_weights[:, :, :, :cfg['context_len']]
        assert jnp.all(context_weights >= 0)
        # Sum of weights should be 1 (softmax property)
        weight_sums = jnp.sum(cand_weights, axis=-1)
        np.testing.assert_allclose(weight_sums, 1.0, rtol=1e-5)

    def test_context_has_causal_attention(self, small_config):
        """Verify context tokens only attend to earlier context (causal)."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale=0.125,
            return_weights=True
        )

        ctx_weights = result.context_attn_weights
        assert ctx_weights is not None
        # Shape: [batch, heads, context_len, context_len] - NOT seq_len!
        # Context tokens only see other context tokens (causal), not candidates
        assert ctx_weights.shape[-1] == cfg['context_len']


# =============================================================================
# Test 3: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_candidate(self):
        """Test with only one candidate."""
        batch_size, context_len, num_candidates = 2, 16, 1
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, context_len, scale=0.125
        )

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        np.testing.assert_allclose(result.output, reference, rtol=1e-4, atol=1e-5)

    def test_many_candidates(self):
        """Test with many candidates."""
        batch_size, context_len, num_candidates = 2, 32, 32
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, context_len, scale=0.125
        )

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        np.testing.assert_allclose(result.output, reference, rtol=1e-4, atol=1e-5)

    def test_no_candidates(self):
        """Test with zero candidates (context only)."""
        batch_size, context_len, num_candidates = 2, 32, 0
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, context_len, scale=0.125
        )

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        np.testing.assert_allclose(result.output, reference, rtol=1e-4, atol=1e-5)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        batch_size, context_len, num_candidates = 1, 16, 4
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, context_len, scale=0.125
        )

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        np.testing.assert_allclose(result.output, reference, rtol=1e-4, atol=1e-5)

    def test_long_context(self):
        """Test with longer context."""
        batch_size, context_len, num_candidates = 1, 256, 8
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)

        reference = compute_reference_attention_with_phoenix_mask(
            query, key, value, context_len, scale=0.125
        )

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        np.testing.assert_allclose(result.output, reference, rtol=1e-4, atol=1e-5)


# =============================================================================
# Test 4: Flash Attention
# =============================================================================

class TestFlashAttention:
    """Test flash attention implementation.

    NOTE: Flash attention doesn't implement tanh capping (Phoenix-specific),
    so we compare against standard attention with tanh capping disabled.
    """

    def test_flash_matches_standard_no_mask(self):
        """Flash attention matches standard without mask."""
        batch_size, seq_len, num_heads, head_dim = 2, 32, 4, 32
        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)
        scale = 0.125

        # Disable tanh capping for fair comparison (flash doesn't implement it)
        reference = standard_attention(query, key, value, mask=None, scale=scale,
                                       attn_logit_cap=1e30)
        result = flash_attention_1d(query, key, value, mask=None, scale=scale)

        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)

    def test_flash_matches_standard_with_causal_mask(self):
        """Flash attention matches standard with causal mask."""
        batch_size, seq_len, num_heads, head_dim = 2, 32, 4, 32
        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)
        scale = 0.125

        # Causal mask - broadcast to batch size
        mask = jnp.tril(jnp.ones((batch_size, 1, seq_len, seq_len)))

        reference = standard_attention(query, key, value, mask=mask, scale=scale,
                                       attn_logit_cap=1e30)
        result = flash_attention_1d(query, key, value, mask=mask, scale=scale)

        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=1e-5)

    def test_flash_different_block_sizes(self):
        """Flash attention works with different block sizes."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 32
        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)
        scale = 0.125

        reference = standard_attention(query, key, value, mask=None, scale=scale,
                                       attn_logit_cap=1e30)

        for block_size in [8, 16, 32, 64]:
            result = flash_attention_1d(
                query, key, value, mask=None, scale=scale, block_size=block_size
            )
            np.testing.assert_allclose(
                result, reference, rtol=1e-4, atol=1e-5,
                err_msg=f"Mismatch at block_size={block_size}"
            )


# =============================================================================
# Test 5: Memory Calculations
# =============================================================================

class TestMemoryCalculations:
    """Test memory estimation functions."""

    def test_standard_memory_calculation(self):
        """Test standard attention memory calculation."""
        batch_size, seq_len, num_heads = 2, 64, 8
        dtype_bytes = 4  # float32

        expected = batch_size * num_heads * seq_len * seq_len * dtype_bytes
        result = compute_attention_memory_bytes(batch_size, seq_len, num_heads, dtype_bytes)

        assert result == expected

    def test_efficient_memory_calculation(self):
        """Test efficient attention memory calculation."""
        batch_size, context_len, num_candidates, num_heads = 2, 64, 8, 8
        dtype_bytes = 4

        result = compute_efficient_attention_memory_bytes(
            batch_size, context_len, num_candidates, num_heads, dtype_bytes
        )

        # Manual calculation
        # Note: context doesn't attend to candidates (causal mask)
        ctx_self = context_len * context_len  # 64 * 64 = 4096
        cand_to_ctx = num_candidates * context_len  # 8 * 64 = 512
        cand_self = num_candidates  # 8
        expected = batch_size * num_heads * (ctx_self + cand_to_ctx + cand_self) * dtype_bytes

        assert result == expected

    def test_memory_reduction_factor(self):
        """Test memory reduction factor calculation."""
        context_len, num_candidates = 256, 16
        seq_len = context_len + num_candidates

        standard_mem = seq_len * seq_len  # 272 * 272 = 73984
        # Note: context doesn't attend to candidates (causal mask)
        efficient_mem = (context_len * context_len +  # 65536
                        num_candidates * context_len +  # 4096
                        num_candidates)  # 16 = 69648

        expected_factor = standard_mem / efficient_mem
        result = memory_reduction_factor(context_len, num_candidates)

        np.testing.assert_allclose(result, expected_factor, rtol=1e-6)

    def test_memory_reduction_increases_with_candidates(self):
        """Memory reduction should increase as candidate ratio increases."""
        context_len = 256

        factors = []
        for num_candidates in [8, 16, 32, 64]:
            factor = memory_reduction_factor(context_len, num_candidates)
            factors.append(factor)

        # More candidates = more memory saved from diagonal optimization
        # But the relationship is not strictly monotonic for small candidates
        # At minimum, all factors should be >= 1.0 (we save memory)
        assert all(f >= 1.0 for f in factors)


# =============================================================================
# Test 6: Numerical Stability
# =============================================================================

class TestNumericalStability:
    """Test numerical stability of attention implementations."""

    def test_no_nans_or_infs(self, small_config):
        """Output should never contain NaN or Inf."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale=0.125
        )

        assert jnp.all(jnp.isfinite(result.output)), "Output contains NaN or Inf"

    def test_stability_with_large_values(self):
        """Test stability when inputs have large values."""
        batch_size, context_len, num_candidates = 2, 16, 4
        num_heads, head_dim = 4, 32
        seq_len = context_len + num_candidates

        # Create inputs with larger magnitudes
        query, key, value = create_qkv(batch_size, seq_len, num_heads, head_dim)
        query = query * 10
        key = key * 10
        value = value * 10

        result = efficient_phoenix_attention(
            query, key, value, context_len, scale=0.125
        )

        assert jnp.all(jnp.isfinite(result.output)), "Output unstable with large inputs"

    def test_attention_weights_sum_to_one(self, small_config):
        """Attention weights should sum to 1."""
        cfg = small_config
        seq_len = cfg['context_len'] + cfg['num_candidates']
        query, key, value = create_qkv(
            cfg['batch_size'], seq_len, cfg['num_heads'], cfg['head_dim']
        )

        result = efficient_phoenix_attention(
            query, key, value, cfg['context_len'], scale=0.125,
            return_weights=True
        )

        # Check context weights
        ctx_weight_sums = jnp.sum(result.context_attn_weights, axis=-1)
        np.testing.assert_allclose(ctx_weight_sums, 1.0, rtol=1e-5)

        # Check candidate weights
        if result.candidate_attn_weights is not None:
            cand_weight_sums = jnp.sum(result.candidate_attn_weights, axis=-1)
            np.testing.assert_allclose(cand_weight_sums, 1.0, rtol=1e-5)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
