"""Core quantization functions for Phoenix model.

Provides functions for quantizing and dequantizing tensors with various
configurations (INT8, INT4, FP16, per-tensor, per-channel, symmetric, asymmetric).
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from enhancements.optimization.quantization.config import (
    BitWidth,
    Granularity,
    QuantizationConfig,
    Symmetry,
)


class QuantizedTensor(NamedTuple):
    """A quantized tensor with scale and zero point.

    Attributes:
        data: Quantized values (int8 for INT8/INT4, float16 for FP16)
        scale: Scale factor(s) for dequantization
        zero_point: Zero point(s) for asymmetric quantization (0 for symmetric)
        original_dtype: Original dtype before quantization
        original_shape: Original tensor shape (for per-group quantization reshape)
    """
    data: jax.Array
    scale: jax.Array
    zero_point: jax.Array
    original_dtype: jnp.dtype
    original_shape: tuple[int, ...] | None = None  # For per-group dequantization


def compute_scale_zp_symmetric(
    tensor: jax.Array,
    bit_width: int,
    axis: int | tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute symmetric quantization parameters.

    For symmetric quantization, the range is [-qmax, qmax] and zero_point is 0.

    Args:
        tensor: Input tensor to quantize
        bit_width: Number of bits (8 or 4)
        axis: Axis for per-channel quantization (None for per-tensor)

    Returns:
        Tuple of (scale, zero_point)
    """
    qmax = 2 ** (bit_width - 1) - 1  # 127 for INT8, 7 for INT4

    if axis is None:
        # Per-tensor: single scale for entire tensor
        amax = jnp.max(jnp.abs(tensor))
    else:
        # Per-channel: scale per channel (keep dims for broadcasting)
        amax = jnp.max(jnp.abs(tensor), axis=axis, keepdims=True)

    # Compute scale, avoiding division by zero
    scale = amax / qmax
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).eps)

    # Zero point is always 0 for symmetric
    zero_point = jnp.zeros_like(scale, dtype=jnp.int32)

    return scale, zero_point


def compute_scale_zp_asymmetric(
    tensor: jax.Array,
    bit_width: int,
    axis: int | tuple[int, ...] | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Compute asymmetric quantization parameters.

    For asymmetric quantization, the range is [qmin, qmax] with a non-zero zero_point.

    Args:
        tensor: Input tensor to quantize
        bit_width: Number of bits (8 or 4)
        axis: Axis for per-channel quantization (None for per-tensor)

    Returns:
        Tuple of (scale, zero_point)
    """
    qmin = 0
    qmax = 2 ** bit_width - 1  # 255 for INT8, 15 for INT4

    if axis is None:
        # Per-tensor
        tmin = jnp.min(tensor)
        tmax = jnp.max(tensor)
    else:
        # Per-channel
        tmin = jnp.min(tensor, axis=axis, keepdims=True)
        tmax = jnp.max(tensor, axis=axis, keepdims=True)

    # Compute scale and zero point
    scale = (tmax - tmin) / (qmax - qmin)
    scale = jnp.maximum(scale, jnp.finfo(jnp.float32).eps)

    zero_point = jnp.round(qmin - tmin / scale).astype(jnp.int32)
    zero_point = jnp.clip(zero_point, qmin, qmax)

    return scale, zero_point


def compute_scale_zp_per_group(
    tensor: jax.Array,
    bit_width: int,
    group_size: int,
    symmetric: bool = True,
) -> tuple[jax.Array, jax.Array, tuple[int, ...]]:
    """Compute per-group quantization parameters.

    Reshapes tensor to [num_groups, group_size] and computes scale per group.
    Better accuracy than per-tensor for aggressive quantization like INT4.

    Args:
        tensor: Input tensor to quantize (2D+)
        bit_width: Number of bits (8 or 4)
        group_size: Number of weights per group (typically 128)
        symmetric: Use symmetric quantization (default True)

    Returns:
        Tuple of (scale, zero_point, original_shape)
    """
    original_shape = tensor.shape

    # Flatten and pad to be divisible by group_size
    flat_tensor = tensor.flatten()
    num_elements = flat_tensor.size
    padded_size = ((num_elements + group_size - 1) // group_size) * group_size
    padding_needed = padded_size - num_elements

    if padding_needed > 0:
        flat_tensor = jnp.pad(flat_tensor, (0, padding_needed), mode='constant')

    # Reshape to [num_groups, group_size]
    num_groups = padded_size // group_size
    grouped_tensor = flat_tensor.reshape(num_groups, group_size)

    if symmetric:
        qmax = 2 ** (bit_width - 1) - 1
        # Compute scale per group (axis=1 reduces within each group)
        amax = jnp.max(jnp.abs(grouped_tensor), axis=1, keepdims=True)
        scale = amax / qmax
        scale = jnp.maximum(scale, jnp.finfo(jnp.float32).eps)
        zero_point = jnp.zeros_like(scale, dtype=jnp.int32)
    else:
        qmin_val, qmax_val = 0, 2 ** bit_width - 1
        tmin = jnp.min(grouped_tensor, axis=1, keepdims=True)
        tmax = jnp.max(grouped_tensor, axis=1, keepdims=True)
        scale = (tmax - tmin) / (qmax_val - qmin_val)
        scale = jnp.maximum(scale, jnp.finfo(jnp.float32).eps)
        zero_point = jnp.round(qmin_val - tmin / scale).astype(jnp.int32)
        zero_point = jnp.clip(zero_point, qmin_val, qmax_val)

    return scale, zero_point, original_shape


def quantize_tensor(
    tensor: jax.Array,
    bit_width: BitWidth,
    granularity: Granularity,
    symmetry: Symmetry,
    group_size: int = 128,
) -> QuantizedTensor:
    """Quantize a single tensor.

    Args:
        tensor: Input tensor (float32 or bfloat16)
        bit_width: Target bit width (FP16, INT8, INT4)
        granularity: Per-tensor, per-channel, or per-group
        symmetry: Symmetric or asymmetric
        group_size: Number of weights per group for PER_GROUP (default 128)

    Returns:
        QuantizedTensor with quantized data, scale, and zero_point
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

    # Integer quantization
    bits = 8 if bit_width == BitWidth.INT8 else 4
    is_symmetric = symmetry == Symmetry.SYMMETRIC

    # Handle per-group quantization separately
    if granularity == Granularity.PER_GROUP and tensor.ndim >= 2:
        scale, zero_point, original_shape = compute_scale_zp_per_group(
            tensor, bits, group_size, symmetric=is_symmetric
        )

        # Flatten and pad for quantization
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.size
        padded_size = ((num_elements + group_size - 1) // group_size) * group_size
        padding_needed = padded_size - num_elements

        if padding_needed > 0:
            flat_tensor = jnp.pad(flat_tensor, (0, padding_needed), mode='constant')

        # Reshape to [num_groups, group_size]
        num_groups = padded_size // group_size
        grouped_tensor = flat_tensor.reshape(num_groups, group_size).astype(jnp.float32)

        # Quantize
        quantized = jnp.round(grouped_tensor / scale) + zero_point

        # Clip to valid range
        if is_symmetric:
            qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        else:
            qmin, qmax = 0, 2 ** bits - 1

        quantized = jnp.clip(quantized, qmin, qmax).astype(jnp.int8)

        return QuantizedTensor(
            data=quantized,
            scale=scale,
            zero_point=zero_point,
            original_dtype=original_dtype,
            original_shape=original_shape,
        )

    # Standard per-tensor or per-channel quantization
    axis: int | tuple[int, ...] | None = None
    if granularity == Granularity.PER_CHANNEL and tensor.ndim >= 2:
        # For weight matrices [in, out], quantize per output channel (axis 0)
        # This keeps a scale per row - reduce all dims except first
        if tensor.ndim == 2:
            axis = 1  # Single axis for 2D
        else:
            axis = tuple(range(1, tensor.ndim))  # Multiple axes for higher dims

    # Compute quantization parameters
    if is_symmetric:
        scale, zero_point = compute_scale_zp_symmetric(tensor, bits, axis)
    else:
        scale, zero_point = compute_scale_zp_asymmetric(tensor, bits, axis)

    # Quantize
    tensor_fp32 = tensor.astype(jnp.float32)
    quantized = jnp.round(tensor_fp32 / scale) + zero_point

    # Clip to valid range
    if is_symmetric:
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    else:
        qmin, qmax = 0, 2 ** bits - 1

    quantized = jnp.clip(quantized, qmin, qmax).astype(jnp.int8)

    return QuantizedTensor(
        data=quantized,
        scale=scale,
        zero_point=zero_point,
        original_dtype=original_dtype,
    )


def dequantize_tensor(qtensor: QuantizedTensor) -> jax.Array:
    """Dequantize a tensor back to floating point.

    Args:
        qtensor: QuantizedTensor to dequantize

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

    # Handle per-group quantization: reshape back to original shape
    if qtensor.original_shape is not None:
        # Flatten dequantized and truncate to original size
        flat_dequantized = dequantized.flatten()
        original_size = 1
        for dim in qtensor.original_shape:
            original_size *= dim
        flat_dequantized = flat_dequantized[:original_size]
        dequantized = flat_dequantized.reshape(qtensor.original_shape)

    return dequantized.astype(qtensor.original_dtype)


def quantize_tensor_simple(
    tensor: jax.Array,
    config: QuantizationConfig,
    bit_width_override: BitWidth | None = None,
    granularity_override: Granularity | None = None,
) -> QuantizedTensor:
    """Quantize a tensor using a QuantizationConfig.

    Args:
        tensor: Input tensor
        config: Quantization configuration
        bit_width_override: Override bit width (for mixed precision)
        granularity_override: Override granularity (for layer-specific settings)

    Returns:
        QuantizedTensor
    """
    return quantize_tensor(
        tensor,
        bit_width_override or config.bit_width,
        granularity_override or config.granularity,
        config.symmetry,
        config.group_size,
    )


class LayerType:
    """Identifiers for different layer types in the model."""
    EMBEDDING = "embedding"
    ATTENTION = "attention"
    FFN = "ffn"
    UNEMBEDDING = "unembedding"
    OTHER = "other"


def get_layer_type(path: str) -> str:
    """Determine the layer type from parameter path.

    Args:
        path: Parameter path string (e.g., "decoder_layer_0/mha_block/query")

    Returns:
        Layer type string from LayerType class
    """
    path_lower = path.lower()

    # Embedding projections
    if "proj_mat" in path_lower or "action_projection" in path_lower:
        return LayerType.EMBEDDING

    # Attention layers (Q, K, V, O)
    if any(k in path_lower for k in ["query", "key", "value"]):
        if "mha" in path_lower or "attention" in path_lower:
            return LayerType.ATTENTION
    if "linear" in path_lower and ("mha" in path_lower or "attention" in path_lower):
        return LayerType.ATTENTION

    # FFN layers
    if any(k in path_lower for k in ["linear_v", "linear_w", "linear_out", "dense"]):
        if "decoder" in path_lower or "block" in path_lower:
            return LayerType.FFN

    # Unembedding
    if "unembedding" in path_lower:
        return LayerType.UNEMBEDDING

    return LayerType.OTHER


def get_quant_settings_for_param(
    path: str,
    config: QuantizationConfig,
) -> tuple[bool, BitWidth | None, Granularity | None]:
    """Get quantization settings for a specific parameter.

    Handles both standard quantization and mixed precision configs.

    Args:
        path: Parameter path string
        config: Quantization configuration

    Returns:
        Tuple of (should_quantize, bit_width_override, granularity_override)
    """
    layer_type = get_layer_type(path)

    # Layer norms - never quantize
    path_lower = path.lower()
    if "norm" in path_lower or "scale" in path_lower:
        return (False, None, None)

    # Check if this layer type should be quantized
    should_quantize = False
    if layer_type == LayerType.EMBEDDING:
        should_quantize = config.quantize_embeddings
    elif layer_type == LayerType.ATTENTION:
        should_quantize = config.quantize_attention
    elif layer_type == LayerType.FFN:
        should_quantize = config.quantize_ffn
    elif layer_type == LayerType.UNEMBEDDING:
        should_quantize = config.quantize_unembedding
    # OTHER types are not quantized

    if not should_quantize:
        return (False, None, None)

    # Get layer-specific bit width and granularity for mixed precision
    bit_width_override: BitWidth | None = None
    granularity_override: Granularity | None = None

    if config.use_mixed_precision and config.mixed_precision:
        mp = config.mixed_precision
        if layer_type == LayerType.FFN:
            bit_width_override = mp.ffn_bit_width
            granularity_override = mp.ffn_granularity
        elif layer_type == LayerType.ATTENTION:
            bit_width_override = mp.attention_bit_width
            granularity_override = mp.attention_granularity
        elif layer_type == LayerType.EMBEDDING:
            bit_width_override = mp.embedding_bit_width

    return (True, bit_width_override, granularity_override)


def should_quantize_param(path: str, config: QuantizationConfig) -> bool:
    """Determine if a parameter should be quantized based on config.

    Args:
        path: Parameter path string (e.g., "decoder_layer_0/mha_block/query")
        config: Quantization configuration

    Returns:
        True if parameter should be quantized
    """
    path_lower = path.lower()

    # Embedding projections
    if "proj_mat" in path_lower or "action_projection" in path_lower:
        return config.quantize_embeddings

    # Attention layers (Q, K, V, O)
    if any(k in path_lower for k in ["query", "key", "value", "linear"]):
        if "mha" in path_lower or "attention" in path_lower:
            return config.quantize_attention

    # FFN layers
    if any(k in path_lower for k in ["linear_v", "linear_w", "linear_out", "dense"]):
        if "decoder" in path_lower or "block" in path_lower:
            return config.quantize_ffn

    # Unembedding
    if "unembedding" in path_lower:
        return config.quantize_unembedding

    # Layer norms - typically not quantized
    if "norm" in path_lower or "scale" in path_lower:
        return False

    # Default: quantize if it's a large weight matrix
    return False


def quantize_params(
    params: dict[str, Any],
    config: QuantizationConfig,
) -> dict[str, Any]:
    """Quantize model parameters according to config.

    Supports both uniform quantization and mixed precision with
    layer-specific bit widths and granularities.

    Args:
        params: Model parameters (nested dict)
        config: Quantization configuration

    Returns:
        Dict with same structure, containing QuantizedTensors where appropriate
    """
    def quantize_recursive(obj: Any, path: str = "") -> Any:
        """Recursively quantize parameters."""
        if isinstance(obj, dict):
            return {
                k: quantize_recursive(v, f"{path}/{k}" if path else k)
                for k, v in obj.items()
            }
        elif hasattr(obj, 'ndim'):
            # This is an array-like object
            # Get quantization settings (supports mixed precision)
            should_quant, bit_width_override, granularity_override = \
                get_quant_settings_for_param(path, config)

            if not should_quant:
                return obj

            # Only quantize 2D+ tensors (weight matrices)
            if obj.ndim < 2:
                return obj

            return quantize_tensor_simple(
                obj, config, bit_width_override, granularity_override
            )
        else:
            return obj

    return quantize_recursive(params)


def dequantize_params(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Dequantize all quantized parameters.

    Args:
        params: Parameters with some QuantizedTensors

    Returns:
        Parameters with all tensors as regular arrays
    """
    def dequantize_recursive(obj: Any) -> Any:
        """Recursively dequantize, handling QuantizedTensor as leaf."""
        if isinstance(obj, QuantizedTensor):
            return dequantize_tensor(obj)
        elif isinstance(obj, dict):
            return {k: dequantize_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            result = [dequantize_recursive(v) for v in obj]
            return type(obj)(result) if isinstance(obj, tuple) else result
        else:
            return obj

    return dequantize_recursive(params)


def compute_memory_bytes(params: dict[str, Any]) -> int:
    """Compute theoretical memory footprint of parameters.

    Automatically detects QuantizedTensors and computes their storage size
    including scale/zero_point overhead.

    Args:
        params: Model parameters (may contain QuantizedTensors)

    Returns:
        Total memory in bytes
    """
    def count_recursive(obj: Any) -> int:
        """Recursively count bytes, handling QuantizedTensor as leaf."""
        if isinstance(obj, QuantizedTensor):
            # Quantized: data size + scale/zp overhead
            data_bytes = obj.data.size * obj.data.dtype.itemsize
            scale_bytes = obj.scale.size * 4  # float32 scales
            zp_bytes = obj.zero_point.size * 4  # int32 zero points
            return data_bytes + scale_bytes + zp_bytes
        elif isinstance(obj, dict):
            return sum(count_recursive(v) for v in obj.values())
        elif isinstance(obj, (list, tuple)):
            return sum(count_recursive(v) for v in obj)
        elif hasattr(obj, 'dtype') and hasattr(obj, 'size'):
            return obj.size * obj.dtype.itemsize
        return 0

    return count_recursive(params)
