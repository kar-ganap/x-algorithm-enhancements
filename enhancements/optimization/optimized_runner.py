"""F2 Phase 5: Combined Optimized Phoenix Runner.

This module provides a unified runner that combines all F2 optimizations:
- JIT compilation (Phase 1): ~10x speedup from static-shape compilation
- KV-cache (Phase 2b): ~2-10x speedup from caching user context K/V tensors
- Quantization (Phase 4/4b): ~58% memory reduction with INT8

Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     OptimizedPhoenixRunner                          │
    │                                                                     │
    │  Optimizations (configurable):                                      │
    │    ├── JIT: Static-shape compilation for fast inference             │
    │    ├── KV-Cache: Reuse user context K/V across candidate batches    │
    │    └── Quantization: INT8/INT4 params for memory reduction          │
    │                                                                     │
    │  Usage:                                                             │
    │    runner = OptimizedPhoenixRunner(config, use_jit=True, ...)       │
    │    runner.initialize()                                              │
    │    output = runner.rank(batch, embeddings)                          │
    └─────────────────────────────────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

from enhancements.optimization.full_kv_cache import FullKVCachedRunner
from enhancements.optimization.jit_utils import (
    StaticJITRunner,
    StaticShapeConfig,
)
from enhancements.optimization.quantization import (
    QuantizationConfig,
    compute_memory_bytes,
    dequantize_params,
    quantize_params,
)
from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import (
    ModelRunner,
    RankingOutput,
    RecsysInferenceRunner,
)


class OptimizationStats(NamedTuple):
    """Statistics for optimization performance."""
    jit_compilation_ms: float = 0.0
    jit_warmup_avg_ms: float = 0.0
    kv_cache_hits: int = 0
    kv_cache_misses: int = 0
    memory_original_bytes: int = 0
    memory_optimized_bytes: int = 0
    memory_reduction_ratio: float = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for combined optimizations.

    Attributes:
        use_jit: Enable JIT compilation with static shapes
        use_kv_cache: Enable K/V tensor caching
        use_quantization: Enable parameter quantization
        jit_config: Static shape configuration for JIT
        quant_config: Quantization configuration
    """
    use_jit: bool = True
    use_kv_cache: bool = True
    use_quantization: bool = True

    # JIT configuration
    jit_batch_size: int = 1
    jit_history_len: int = 32
    jit_num_candidates: int = 8

    # Quantization configuration (default: INT8 per-channel with KV-cache)
    quant_config: QuantizationConfig | None = None

    def __post_init__(self):
        if self.quant_config is None and self.use_quantization:
            from enhancements.optimization.quantization import BitWidth, Granularity
            self.quant_config = QuantizationConfig(
                bit_width=BitWidth.INT8,
                granularity=Granularity.PER_CHANNEL,
                quantize_kv_cache=True,
            )


class OptimizedPhoenixRunner:
    """Phoenix runner with all F2 optimizations combined.

    This runner provides a unified interface that combines:
    - JIT compilation for fast inference
    - KV-cache for user context reuse
    - Quantization for memory reduction

    Each optimization can be enabled/disabled independently.

    Example:
        >>> config = PhoenixModelConfig(...)
        >>> opt_config = OptimizationConfig(use_jit=True, use_kv_cache=True)
        >>> runner = OptimizedPhoenixRunner(config, opt_config)
        >>> runner.initialize()
        >>> output = runner.rank(batch, embeddings)
    """

    def __init__(
        self,
        model_config: PhoenixModelConfig,
        optimization_config: OptimizationConfig | None = None,
    ):
        """Initialize the optimized runner.

        Args:
            model_config: Phoenix model configuration
            optimization_config: Optimization settings (defaults if None)
        """
        self.model_config = model_config
        self.opt_config = optimization_config or OptimizationConfig()

        # Runners (set during initialize)
        self._base_runner: RecsysInferenceRunner | None = None
        self._jit_runner: StaticJITRunner | None = None
        self._kv_cache_runner: FullKVCachedRunner | None = None

        # Quantized params (if using quantization without KV-cache)
        self._quantized_params: dict[str, Any] | None = None
        self._original_params: dict[str, Any] | None = None

        # Stats
        self._stats = OptimizationStats()
        self._initialized = False

    @property
    def stats(self) -> OptimizationStats:
        """Get optimization statistics."""
        return self._stats

    @property
    def params(self):
        """Get model parameters."""
        if self._kv_cache_runner is not None:
            return self._kv_cache_runner.params
        elif self._base_runner is not None:
            return self._base_runner.params
        return None

    def initialize(self) -> None:
        """Initialize the runner and apply optimizations."""
        if self._initialized:
            return

        opt = self.opt_config

        # Strategy: Use KV-cache runner as the primary path if enabled,
        # otherwise use base runner with JIT wrapper

        if opt.use_kv_cache:
            # KV-cache runner has its own JIT-compiled functions
            self._kv_cache_runner = FullKVCachedRunner(self.model_config)
            self._kv_cache_runner.initialize()

            # Apply quantization to KV-cache runner params if enabled
            if opt.use_quantization and opt.quant_config:
                self._original_params = self._kv_cache_runner.params
                self._quantized_params = quantize_params(
                    self._kv_cache_runner.params,
                    opt.quant_config
                )
        else:
            # Create base runner
            model_runner = ModelRunner(model=self.model_config)
            self._base_runner = RecsysInferenceRunner(runner=model_runner, name="optimized")
            self._base_runner.initialize()

            # Apply quantization if enabled
            if opt.use_quantization and opt.quant_config:
                self._original_params = self._base_runner.params
                self._quantized_params = quantize_params(
                    self._base_runner.params,
                    opt.quant_config
                )

            # Wrap with JIT if enabled
            if opt.use_jit:
                jit_config = StaticShapeConfig(
                    batch_size=opt.jit_batch_size,
                    history_len=opt.jit_history_len,
                    num_candidates=opt.jit_num_candidates,
                )
                self._jit_runner = StaticJITRunner(
                    self._base_runner,
                    jit_config,
                    warmup_iterations=3,
                )
                # Trigger compilation
                self._jit_runner._ensure_compiled()

                if self._jit_runner.stats:
                    self._stats = self._stats._replace(
                        jit_compilation_ms=self._jit_runner.stats.compilation_time_ms,
                        jit_warmup_avg_ms=self._jit_runner.stats.warmup_avg_time_ms,
                    )

        # Compute memory stats
        self._update_memory_stats()
        self._initialized = True

    def _update_memory_stats(self) -> None:
        """Update memory statistics."""
        if self._original_params is not None:
            original_bytes = compute_memory_bytes(self._original_params)

            if self._quantized_params is not None:
                optimized_bytes = compute_memory_bytes(self._quantized_params)
            else:
                optimized_bytes = original_bytes

            reduction = 1.0 - (optimized_bytes / original_bytes) if original_bytes > 0 else 0.0

            self._stats = self._stats._replace(
                memory_original_bytes=original_bytes,
                memory_optimized_bytes=optimized_bytes,
                memory_reduction_ratio=reduction,
            )
        elif self._kv_cache_runner is not None:
            original_bytes = compute_memory_bytes(self._kv_cache_runner.params)
            self._stats = self._stats._replace(
                memory_original_bytes=original_bytes,
                memory_optimized_bytes=original_bytes,
            )

    def _get_inference_params(self):
        """Get parameters for inference (dequantized if needed)."""
        if self._quantized_params is not None:
            return dequantize_params(self._quantized_params)
        elif self._kv_cache_runner is not None:
            return self._kv_cache_runner.params
        elif self._base_runner is not None:
            return self._base_runner.params
        raise RuntimeError("Runner not initialized")

    def rank(
        self,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> RankingOutput:
        """Rank candidates using optimized inference.

        Automatically uses the best available optimizations based on config.

        Args:
            batch: Input batch with user and candidate data
            embeddings: Pre-computed embeddings

        Returns:
            RankingOutput with scores and rankings
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        opt = self.opt_config

        # Path 1: KV-cache (includes its own JIT)
        if opt.use_kv_cache and self._kv_cache_runner is not None:
            # If using quantization, we need to inject dequantized params
            if opt.use_quantization and self._quantized_params is not None:
                # For KV-cache runner, we need to temporarily replace params
                original_params = self._kv_cache_runner.params
                self._kv_cache_runner.params = dequantize_params(self._quantized_params)
                try:
                    output = self._kv_cache_runner.rank(batch, embeddings)
                finally:
                    self._kv_cache_runner.params = original_params
            else:
                output = self._kv_cache_runner.rank(batch, embeddings)

            # Update cache stats
            stats = self._kv_cache_runner.stats
            self._stats = self._stats._replace(
                kv_cache_hits=stats.hits,
                kv_cache_misses=stats.misses,
            )
            return output

        # Path 2: JIT without KV-cache
        if opt.use_jit and self._jit_runner is not None:
            if opt.use_quantization and self._quantized_params is not None:
                # Inject dequantized params
                original_params = self._base_runner.params
                self._base_runner.params = dequantize_params(self._quantized_params)
                try:
                    output = self._jit_runner.rank(batch, embeddings)
                finally:
                    self._base_runner.params = original_params
            else:
                output = self._jit_runner.rank(batch, embeddings)
            return output

        # Path 3: Base runner (no JIT, no KV-cache)
        if self._base_runner is not None:
            if opt.use_quantization and self._quantized_params is not None:
                params = dequantize_params(self._quantized_params)
            else:
                params = self._base_runner.params
            return self._base_runner.rank_candidates(params, batch, embeddings)

        raise RuntimeError("No runner available")

    def clear_cache(self) -> None:
        """Clear KV-cache if enabled."""
        if self._kv_cache_runner is not None:
            self._kv_cache_runner.clear_cache()

    def get_config_summary(self) -> dict[str, Any]:
        """Get summary of optimization configuration."""
        opt = self.opt_config
        return {
            "use_jit": opt.use_jit,
            "use_kv_cache": opt.use_kv_cache,
            "use_quantization": opt.use_quantization,
            "jit_batch_size": opt.jit_batch_size,
            "jit_history_len": opt.jit_history_len,
            "jit_num_candidates": opt.jit_num_candidates,
            "quant_config": opt.quant_config.name if opt.quant_config else None,
        }

    def benchmark_report(self) -> dict[str, Any]:
        """Generate benchmark report comparing to baseline.

        Returns:
            Dict with benchmark metrics and comparisons
        """
        if not self._initialized:
            raise RuntimeError("Runner not initialized")

        report = {
            "optimization_config": self.get_config_summary(),
            "stats": {
                "jit_compilation_ms": self._stats.jit_compilation_ms,
                "jit_warmup_avg_ms": self._stats.jit_warmup_avg_ms,
                "kv_cache_hits": self._stats.kv_cache_hits,
                "kv_cache_misses": self._stats.kv_cache_misses,
                "memory_original_bytes": self._stats.memory_original_bytes,
                "memory_optimized_bytes": self._stats.memory_optimized_bytes,
                "memory_reduction_ratio": self._stats.memory_reduction_ratio,
            },
        }

        return report


def create_optimized_runner(
    model_config: PhoenixModelConfig,
    use_jit: bool = True,
    use_kv_cache: bool = True,
    use_quantization: bool = True,
    quant_config: QuantizationConfig | None = None,
) -> OptimizedPhoenixRunner:
    """Create and initialize an optimized runner.

    Convenience function that creates and initializes the runner in one call.

    Args:
        model_config: Phoenix model configuration
        use_jit: Enable JIT compilation
        use_kv_cache: Enable KV-cache
        use_quantization: Enable quantization
        quant_config: Optional custom quantization config

    Returns:
        Initialized OptimizedPhoenixRunner
    """
    opt_config = OptimizationConfig(
        use_jit=use_jit,
        use_kv_cache=use_kv_cache,
        use_quantization=use_quantization,
        quant_config=quant_config,
    )

    runner = OptimizedPhoenixRunner(model_config, opt_config)
    runner.initialize()

    return runner
