"""Quantized Phoenix runner for inference.

Provides a runner that uses quantized model parameters for reduced memory
and (potentially) faster inference. Supports simulated quantization
(dequantize on-the-fly) for accuracy evaluation.
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from phoenix.recsys_model import (
    PhoenixModelConfig,
    RecsysBatch,
    RecsysEmbeddings,
)
from phoenix.runners import (
    ModelRunner,
    RankingOutput,
    RecsysInferenceRunner,
)

from enhancements.optimization.quantization.config import QuantizationConfig
from enhancements.optimization.quantization.quantize import (
    QuantizedTensor,
    compute_memory_bytes,
    dequantize_params,
    quantize_params,
)


class QuantizedPhoenixRunner:
    """Phoenix runner with quantized model parameters.

    This runner stores model weights in quantized form (INT8, INT4, or FP16)
    and dequantizes them on-the-fly during inference. This provides:
    - Memory reduction for parameter storage
    - Accuracy evaluation for different quantization configs
    - Foundation for native quantized inference (future work)

    Usage:
        # Create base runner and initialize
        base_runner = RecsysInferenceRunner(ModelRunner(config), "base")
        base_runner.initialize()

        # Create quantized runner from base
        quant_config = QuantizationConfig(bit_width=BitWidth.INT8)
        quant_runner = QuantizedPhoenixRunner(base_runner, quant_config)

        # Use for inference
        output = quant_runner.rank(batch, embeddings)
    """

    def __init__(
        self,
        base_runner: RecsysInferenceRunner,
        quant_config: QuantizationConfig,
    ):
        """Initialize quantized runner from a base runner.

        Args:
            base_runner: Initialized RecsysInferenceRunner with params
            quant_config: Quantization configuration
        """
        self.base_runner = base_runner
        self.quant_config = quant_config
        self.model_config = base_runner.runner.model

        # Quantize parameters
        self.original_params = base_runner.params
        self.quantized_params = quantize_params(base_runner.params, quant_config)

        # Store the rank function from base runner
        self._rank_fn = base_runner.rank_candidates

    def rank(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """Rank candidates using quantized model.

        Uses simulated quantization: parameters are stored quantized and
        dequantized on-the-fly for inference. This accurately simulates
        the accuracy impact of quantization.

        Args:
            batch: Input batch with hashes, actions, product surfaces
            embeddings: Pre-looked-up embeddings

        Returns:
            RankingOutput with scores and ranked indices
        """
        # Dequantize params for inference
        dequantized_params = dequantize_params(self.quantized_params)

        # Run inference with dequantized params
        return self._rank_fn(dequantized_params, batch, embeddings)

    def get_original_memory_bytes(self) -> int:
        """Get memory footprint of original (unquantized) parameters."""
        return compute_memory_bytes(self.original_params)

    def get_quantized_memory_bytes(self) -> int:
        """Get memory footprint of quantized parameters."""
        return compute_memory_bytes(self.quantized_params)

    def get_memory_reduction_ratio(self) -> float:
        """Get memory reduction ratio (0 to 1).

        Returns:
            Ratio of memory saved. E.g., 0.75 means 75% reduction.
        """
        original = self.get_original_memory_bytes()
        quantized = self.get_quantized_memory_bytes()
        return 1.0 - (quantized / original)

    def get_compression_ratio(self) -> float:
        """Get compression ratio (original / quantized).

        Returns:
            Compression ratio. E.g., 4.0 means 4x smaller.
        """
        original = self.get_original_memory_bytes()
        quantized = self.get_quantized_memory_bytes()
        return original / quantized

    def count_quantized_params(self) -> Dict[str, int]:
        """Count parameters by quantization status.

        Returns:
            Dict with 'quantized' and 'unquantized' counts
        """
        quantized_count = 0
        unquantized_count = 0

        def count_leaf(value: Any):
            nonlocal quantized_count, unquantized_count
            if isinstance(value, QuantizedTensor):
                quantized_count += value.data.size
            elif hasattr(value, 'size'):
                unquantized_count += value.size

        jax.tree_util.tree_map(count_leaf, self.quantized_params)

        return {
            'quantized': quantized_count,
            'unquantized': unquantized_count,
            'total': quantized_count + unquantized_count,
            'quantized_ratio': quantized_count / (quantized_count + unquantized_count)
            if (quantized_count + unquantized_count) > 0 else 0.0,
        }


def create_quantized_runner(
    model_config: PhoenixModelConfig,
    quant_config: QuantizationConfig,
    base_runner: Optional[RecsysInferenceRunner] = None,
) -> QuantizedPhoenixRunner:
    """Create a quantized runner, optionally initializing from scratch.

    Args:
        model_config: Phoenix model configuration
        quant_config: Quantization configuration
        base_runner: Optional pre-initialized base runner. If None, creates one.

    Returns:
        QuantizedPhoenixRunner ready for inference
    """
    if base_runner is None:
        # Create and initialize base runner
        model_runner = ModelRunner(model=model_config)
        base_runner = RecsysInferenceRunner(runner=model_runner, name="base")
        base_runner.initialize()

    return QuantizedPhoenixRunner(base_runner, quant_config)
