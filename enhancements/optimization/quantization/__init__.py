"""Quantization module for Phoenix model optimization.

This module provides infrastructure for quantizing Phoenix model parameters
to reduce memory footprint and (potentially) improve inference speed.

Key Components:
    - QuantizationConfig: Configuration for quantization (bit width, granularity, etc.)
    - MixedPrecisionConfig: Layer-specific quantization settings
    - quantize/dequantize: Core functions for tensor quantization
    - QuantizedPhoenixRunner: Runner with quantized parameters
    - QuantizationStudy: Benchmark framework for comparing configurations
    - KV-cache quantization: Memory-efficient K/V caching

Phase 4b Extensions:
    - Mixed INT4/INT8 precision (layer-specific bit widths)
    - Per-group INT4 quantization (128 weights per group)
    - KV-cache quantization (INT8 K/V tensors)
"""

from enhancements.optimization.quantization.config import (
    BitWidth,
    EXTENDED_STUDY_CONFIGS,
    Granularity,
    MixedPrecisionConfig,
    QuantizationConfig,
    STUDY_CONFIGS,
    Symmetry,
)
from enhancements.optimization.quantization.quantize import (
    LayerType,
    QuantizedTensor,
    compute_memory_bytes,
    compute_scale_zp_asymmetric,
    compute_scale_zp_per_group,
    compute_scale_zp_symmetric,
    dequantize_params,
    dequantize_tensor,
    get_layer_type,
    get_quant_settings_for_param,
    quantize_params,
    quantize_tensor,
    quantize_tensor_simple,
    should_quantize_param,
)
from enhancements.optimization.quantization.quantized_runner import (
    QuantizedPhoenixRunner,
    create_quantized_runner,
)
from enhancements.optimization.quantization.study import (
    BenchmarkMetrics,
    QuantizationStudy,
    StudyConfig,
    WinnerSelectionCriteria,
    compute_kendall_tau,
    compute_top3_match,
    format_results_table,
    select_winner,
)
from enhancements.optimization.quantization.kv_quantize import (
    QuantizedLayerKVCache,
    dequantize_kv,
    dequantize_kv_cache,
    get_compression_ratio,
    get_kv_cache_memory_bytes,
    quantize_kv,
    quantize_kv_cache,
)

__all__ = [
    # Config
    "BitWidth",
    "EXTENDED_STUDY_CONFIGS",
    "Granularity",
    "MixedPrecisionConfig",
    "QuantizationConfig",
    "STUDY_CONFIGS",
    "Symmetry",
    # Quantize functions
    "LayerType",
    "QuantizedTensor",
    "compute_memory_bytes",
    "compute_scale_zp_asymmetric",
    "compute_scale_zp_per_group",
    "compute_scale_zp_symmetric",
    "dequantize_params",
    "dequantize_tensor",
    "get_layer_type",
    "get_quant_settings_for_param",
    "quantize_params",
    "quantize_tensor",
    "quantize_tensor_simple",
    "should_quantize_param",
    # Runner
    "QuantizedPhoenixRunner",
    "create_quantized_runner",
    # Study
    "BenchmarkMetrics",
    "QuantizationStudy",
    "StudyConfig",
    "WinnerSelectionCriteria",
    "compute_kendall_tau",
    "compute_top3_match",
    "format_results_table",
    "select_winner",
    # KV-cache quantization
    "QuantizedLayerKVCache",
    "dequantize_kv",
    "dequantize_kv_cache",
    "get_compression_ratio",
    "get_kv_cache_memory_bytes",
    "quantize_kv",
    "quantize_kv_cache",
]
