"""KV-cache quantization for memory-efficient inference.

Provides quantization for K and V tensors in the attention cache.
This allows significant memory reduction for long context inference
while maintaining good accuracy (K/V tensors are less sensitive than weights).

Usage:
    from enhancements.optimization.quantization.kv_quantize import (
        QuantizedLayerKVCache,
        quantize_kv_cache,
        dequantize_kv_cache,
    )

    # Quantize cache before storing
    quant_cache = quantize_kv_cache(kv_cache, bit_width=BitWidth.INT8)

    # Dequantize when using for attention
    kv_cache = dequantize_kv_cache(quant_cache)
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from enhancements.optimization.quantization.config import BitWidth, Symmetry
from enhancements.optimization.quantization.quantize import (
    QuantizedTensor,
    compute_scale_zp_asymmetric,
    compute_scale_zp_symmetric,
)


class QuantizedLayerKVCache(NamedTuple):
    """Quantized K,V cache for a single transformer layer.

    Stores INT8 quantized key and value projections with their scales.

    Attributes:
        keys: QuantizedTensor containing INT8 keys
        values: QuantizedTensor containing INT8 values
    """
    keys: QuantizedTensor
    values: QuantizedTensor


def quantize_kv(
    tensor: jax.Array,
    bit_width: BitWidth = BitWidth.INT8,
    symmetry: Symmetry = Symmetry.SYMMETRIC,
) -> QuantizedTensor:
    """Quantize a K or V tensor for cache storage.

    Uses per-tensor quantization for simplicity and speed.
    K/V tensors have shape [batch, num_kv_heads, seq_len, head_dim].

    Args:
        tensor: K or V tensor from attention layer
        bit_width: Target bit width (FP16 or INT8)
        symmetry: Symmetric or asymmetric quantization

    Returns:
        QuantizedTensor with quantized data and scale
    """
    original_dtype = tensor.dtype

    # FP16 is just a dtype cast
    if bit_width == BitWidth.FP16:
        return QuantizedTensor(
            data=tensor.astype(jnp.float16),
            scale=jnp.array(1.0, dtype=jnp.float32),
            zero_point=jnp.array(0, dtype=jnp.int32),
            original_dtype=original_dtype,
        )

    # INT8 quantization - use per-tensor for speed
    bits = 8

    # Compute quantization parameters
    if symmetry == Symmetry.SYMMETRIC:
        scale, zero_point = compute_scale_zp_symmetric(tensor, bits, axis=None)
    else:
        scale, zero_point = compute_scale_zp_asymmetric(tensor, bits, axis=None)

    # Quantize
    tensor_fp32 = tensor.astype(jnp.float32)
    quantized = jnp.round(tensor_fp32 / scale) + zero_point

    # Clip to valid range
    if symmetry == Symmetry.SYMMETRIC:
        qmin, qmax = -128, 127
    else:
        qmin, qmax = 0, 255

    quantized = jnp.clip(quantized, qmin, qmax).astype(jnp.int8)

    return QuantizedTensor(
        data=quantized,
        scale=scale,
        zero_point=zero_point,
        original_dtype=original_dtype,
    )


def dequantize_kv(qtensor: QuantizedTensor) -> jax.Array:
    """Dequantize a K or V tensor for use in attention.

    Args:
        qtensor: QuantizedTensor containing quantized K or V

    Returns:
        Recovered floating point tensor
    """
    # FP16 is just a dtype cast back
    if qtensor.data.dtype == jnp.float16:
        return qtensor.data.astype(qtensor.original_dtype)

    # Integer dequantization: (q - zero_point) * scale
    dequantized = (
        qtensor.data.astype(jnp.float32) - qtensor.zero_point.astype(jnp.float32)
    ) * qtensor.scale

    return dequantized.astype(qtensor.original_dtype)


def quantize_kv_cache(
    keys: jax.Array,
    values: jax.Array,
    bit_width: BitWidth = BitWidth.INT8,
    symmetry: Symmetry = Symmetry.SYMMETRIC,
) -> QuantizedLayerKVCache:
    """Quantize both K and V tensors for cache storage.

    Args:
        keys: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        values: Value tensor [batch, num_kv_heads, seq_len, head_dim]
        bit_width: Target bit width (default INT8)
        symmetry: Quantization symmetry (default symmetric)

    Returns:
        QuantizedLayerKVCache containing quantized K and V
    """
    return QuantizedLayerKVCache(
        keys=quantize_kv(keys, bit_width, symmetry),
        values=quantize_kv(values, bit_width, symmetry),
    )


def dequantize_kv_cache(
    quant_cache: QuantizedLayerKVCache,
) -> tuple[jax.Array, jax.Array]:
    """Dequantize K and V tensors for use in attention.

    Args:
        quant_cache: QuantizedLayerKVCache containing quantized K and V

    Returns:
        Tuple of (keys, values) as floating point tensors
    """
    return (
        dequantize_kv(quant_cache.keys),
        dequantize_kv(quant_cache.values),
    )


def get_kv_cache_memory_bytes(quant_cache: QuantizedLayerKVCache) -> int:
    """Compute memory footprint of quantized KV cache.

    Args:
        quant_cache: Quantized KV cache

    Returns:
        Total memory in bytes
    """
    k_data_bytes = quant_cache.keys.data.size * quant_cache.keys.data.dtype.itemsize
    k_scale_bytes = quant_cache.keys.scale.size * 4  # float32
    k_zp_bytes = quant_cache.keys.zero_point.size * 4  # int32

    v_data_bytes = quant_cache.values.data.size * quant_cache.values.data.dtype.itemsize
    v_scale_bytes = quant_cache.values.scale.size * 4
    v_zp_bytes = quant_cache.values.zero_point.size * 4

    return k_data_bytes + k_scale_bytes + k_zp_bytes + v_data_bytes + v_scale_bytes + v_zp_bytes


def get_compression_ratio(
    original_keys: jax.Array,
    original_values: jax.Array,
    quant_cache: QuantizedLayerKVCache,
) -> float:
    """Compute compression ratio for KV cache.

    Args:
        original_keys: Original key tensor
        original_values: Original value tensor
        quant_cache: Quantized KV cache

    Returns:
        Compression ratio (original_bytes / quantized_bytes)
    """
    original_bytes = (
        original_keys.size * original_keys.dtype.itemsize +
        original_values.size * original_values.dtype.itemsize
    )
    quantized_bytes = get_kv_cache_memory_bytes(quant_cache)

    return original_bytes / max(quantized_bytes, 1)
