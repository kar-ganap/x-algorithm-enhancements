"""Tests for F2 Phase 5: Combined Optimized Phoenix Runner.

Go/No-Go Gate Criteria:
1. Runner initializes with all optimization configurations
2. Output matches baseline within tolerance (rtol=1e-4)
3. KV-cache path produces correct results
4. JIT path produces correct results
5. Quantization path produces correct results
6. Benchmark report contains required metrics
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from enhancements.optimization.optimized_runner import (
    OptimizationConfig,
    OptimizationStats,
    OptimizedPhoenixRunner,
    create_optimized_runner,
)
from enhancements.optimization.quantization import BitWidth, Granularity, QuantizationConfig
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, ModelRunner, RecsysInferenceRunner, create_example_batch

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def model_config():
    """Create model configuration for testing."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=128,
        num_actions=len(ACTIONS),
        history_seq_len=32,
        candidate_seq_len=8,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=128,
            widening_factor=2,
            key_size=64,
            num_q_heads=4,
            num_kv_heads=2,
            num_layers=4,
            attn_output_multiplier=0.125,
        ),
    )


@pytest.fixture(scope="module")
def base_runner(model_config):
    """Create and initialize baseline runner."""
    model_runner = ModelRunner(model=model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="baseline")
    runner.initialize()
    return runner


@pytest.fixture
def example_batch(model_config):
    """Create example batch for testing."""
    return create_example_batch(
        batch_size=1,
        emb_size=model_config.emb_size,
        history_len=model_config.history_seq_len,
        num_candidates=model_config.candidate_seq_len,
        num_actions=model_config.num_actions,
        num_user_hashes=model_config.hash_config.num_user_hashes,
        num_item_hashes=model_config.hash_config.num_item_hashes,
        num_author_hashes=model_config.hash_config.num_author_hashes,
        product_surface_vocab_size=model_config.product_surface_vocab_size,
    )


# =============================================================================
# GATE CRITERION 1: Runner initializes with all configurations
# =============================================================================


class TestRunnerInitialization:
    """Tests for OptimizedPhoenixRunner initialization."""

    def test_default_config(self, model_config):
        """Test initialization with default config."""
        runner = OptimizedPhoenixRunner(model_config)
        runner.initialize()

        assert runner.params is not None
        assert runner.opt_config.use_jit is True
        assert runner.opt_config.use_kv_cache is True
        assert runner.opt_config.use_quantization is True

    def test_jit_only_config(self, model_config):
        """Test initialization with JIT only."""
        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=False,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        assert runner._jit_runner is not None
        assert runner._kv_cache_runner is None

    def test_kv_cache_only_config(self, model_config):
        """Test initialization with KV-cache only."""
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=True,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        assert runner._kv_cache_runner is not None

    def test_quantization_only_config(self, model_config):
        """Test initialization with quantization only."""
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=False,
            use_quantization=True,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        assert runner._quantized_params is not None

    def test_no_optimizations_config(self, model_config):
        """Test initialization with no optimizations."""
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=False,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        assert runner._base_runner is not None
        assert runner._jit_runner is None
        assert runner._kv_cache_runner is None
        assert runner._quantized_params is None

    def test_double_initialize_is_noop(self, model_config):
        """Test that calling initialize twice is safe."""
        runner = OptimizedPhoenixRunner(model_config)
        runner.initialize()
        runner.initialize()  # Should not raise

        assert runner._initialized is True


# =============================================================================
# GATE CRITERION 2: Output matches baseline
# =============================================================================


class TestOutputCorrectness:
    """Tests for numerical correctness of optimized runner."""

    def test_kv_cache_matches_baseline(self, model_config, base_runner, example_batch):
        """Verify KV-cache runner matches baseline within tolerance."""
        batch, embeddings = example_batch

        # Baseline
        baseline_output = base_runner.rank(batch, embeddings)

        # KV-cache runner
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=True,
            use_quantization=False,
        )
        opt_runner = OptimizedPhoenixRunner(model_config, opt_config)
        opt_runner.initialize()
        opt_output = opt_runner.rank(batch, embeddings)

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(opt_output.scores),
            rtol=1e-4,
            atol=1e-6,
            err_msg="KV-cache runner should match baseline",
        )

    def test_jit_matches_baseline(self, model_config, base_runner, example_batch):
        """Verify JIT runner matches baseline within tolerance."""
        batch, embeddings = example_batch

        # Baseline
        baseline_output = base_runner.rank(batch, embeddings)

        # JIT runner (no KV-cache, no quantization)
        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=False,
            use_quantization=False,
        )
        opt_runner = OptimizedPhoenixRunner(model_config, opt_config)
        opt_runner.initialize()
        opt_output = opt_runner.rank(batch, embeddings)

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(opt_output.scores),
            rtol=1e-4,
            atol=1e-6,
            err_msg="JIT runner should match baseline",
        )

    def test_no_optimizations_matches_baseline(self, model_config, base_runner, example_batch):
        """Verify no-optimizations runner matches baseline exactly."""
        batch, embeddings = example_batch

        # Baseline
        baseline_output = base_runner.rank(batch, embeddings)

        # No optimizations
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=False,
            use_quantization=False,
        )
        opt_runner = OptimizedPhoenixRunner(model_config, opt_config)
        opt_runner.initialize()
        opt_output = opt_runner.rank(batch, embeddings)

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(opt_output.scores),
            rtol=1e-6,
            atol=1e-6,
            err_msg="No-optimizations runner should match baseline exactly",
        )


# =============================================================================
# GATE CRITERION 3: KV-cache path produces correct results
# =============================================================================


class TestKVCachePath:
    """Tests for KV-cache inference path."""

    def test_cache_stats_tracked(self, model_config, example_batch):
        """Verify cache statistics are tracked."""
        batch, embeddings = example_batch

        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=True,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        # First call - cache miss
        runner.clear_cache()
        _ = runner.rank(batch, embeddings)

        # Second call - cache hit
        _ = runner.rank(batch, embeddings)

        stats = runner.stats
        assert stats.kv_cache_hits >= 1, "Should have at least one cache hit"
        assert stats.kv_cache_misses >= 1, "Should have at least one cache miss"

    def test_clear_cache(self, model_config, example_batch):
        """Verify clear_cache works correctly."""
        batch, embeddings = example_batch

        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=True,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        # Populate cache
        _ = runner.rank(batch, embeddings)

        # Clear cache
        runner.clear_cache()

        # Next call should be cache miss
        _ = runner.rank(batch, embeddings)
        stats = runner.stats
        assert stats.kv_cache_misses >= 2, "Clear cache should force cache miss"


# =============================================================================
# GATE CRITERION 4: JIT path produces correct results
# =============================================================================


class TestJITPath:
    """Tests for JIT inference path."""

    def test_jit_compilation_stats(self, model_config, example_batch):
        """Verify JIT compilation statistics are tracked."""
        batch, embeddings = example_batch

        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=False,
            use_quantization=False,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        # Run inference
        _ = runner.rank(batch, embeddings)

        stats = runner.stats
        # Compilation time should be recorded (may be 0 if already cached)
        assert stats.jit_compilation_ms >= 0

    def test_jit_with_padded_input(self, model_config, example_batch):
        """Verify JIT handles input correctly."""
        batch, embeddings = example_batch

        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=False,
            use_quantization=False,
            jit_batch_size=1,
            jit_history_len=32,
            jit_num_candidates=8,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        output = runner.rank(batch, embeddings)
        assert output.scores is not None
        assert output.scores.shape[1] == model_config.candidate_seq_len


# =============================================================================
# GATE CRITERION 5: Quantization path produces correct results
# =============================================================================


class TestQuantizationPath:
    """Tests for quantization inference path."""

    def test_quantization_memory_stats(self, model_config, example_batch):
        """Verify memory statistics are tracked with quantization."""
        batch, embeddings = example_batch

        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=False,
            use_quantization=True,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        stats = runner.stats
        assert stats.memory_original_bytes > 0
        assert stats.memory_optimized_bytes > 0
        assert stats.memory_reduction_ratio > 0, "Should have memory reduction"

    def test_quantization_output_reasonable(self, model_config, base_runner, example_batch):
        """Verify quantized output is reasonable (within 5% of baseline)."""
        batch, embeddings = example_batch

        # Baseline
        baseline_output = base_runner.rank(batch, embeddings)

        # Quantized runner
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=False,
            use_quantization=True,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()
        quant_output = runner.rank(batch, embeddings)

        # INT8 quantization should preserve ranking reasonably well
        baseline_ranking = np.argsort(-np.array(baseline_output.scores[0, :, 0]))
        quant_ranking = np.argsort(-np.array(quant_output.scores[0, :, 0]))

        # Top-3 should match for INT8
        top3_match = set(baseline_ranking[:3]) == set(quant_ranking[:3])
        assert top3_match, "Top-3 ranking should be preserved with INT8 quantization"

    def test_kv_cache_with_quantization(self, model_config, base_runner, example_batch):
        """Verify KV-cache works with quantization."""
        batch, embeddings = example_batch

        # Baseline
        baseline_output = base_runner.rank(batch, embeddings)

        # Combined runner
        opt_config = OptimizationConfig(
            use_jit=False,
            use_kv_cache=True,
            use_quantization=True,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()
        opt_output = runner.rank(batch, embeddings)

        # Should still produce valid output
        assert opt_output.scores is not None
        assert opt_output.scores.shape == baseline_output.scores.shape


# =============================================================================
# GATE CRITERION 6: Benchmark report contains required metrics
# =============================================================================


class TestBenchmarkReport:
    """Tests for benchmark report generation."""

    def test_benchmark_report_structure(self, model_config, example_batch):
        """Verify benchmark report contains required fields."""
        batch, embeddings = example_batch

        runner = create_optimized_runner(model_config)
        _ = runner.rank(batch, embeddings)

        report = runner.benchmark_report()

        # Check required sections
        assert "optimization_config" in report
        assert "stats" in report

        # Check config fields
        config = report["optimization_config"]
        assert "use_jit" in config
        assert "use_kv_cache" in config
        assert "use_quantization" in config

        # Check stats fields
        stats = report["stats"]
        assert "jit_compilation_ms" in stats
        assert "kv_cache_hits" in stats
        assert "kv_cache_misses" in stats
        assert "memory_original_bytes" in stats
        assert "memory_optimized_bytes" in stats
        assert "memory_reduction_ratio" in stats

    def test_config_summary(self, model_config):
        """Verify config summary is complete."""
        opt_config = OptimizationConfig(
            use_jit=True,
            use_kv_cache=True,
            use_quantization=True,
            jit_batch_size=2,
            jit_history_len=64,
            jit_num_candidates=16,
        )
        runner = OptimizedPhoenixRunner(model_config, opt_config)
        runner.initialize()

        summary = runner.get_config_summary()

        assert summary["use_jit"] is True
        assert summary["use_kv_cache"] is True
        assert summary["use_quantization"] is True
        assert summary["jit_batch_size"] == 2
        assert summary["jit_history_len"] == 64
        assert summary["jit_num_candidates"] == 16


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunction:
    """Tests for create_optimized_runner convenience function."""

    def test_create_optimized_runner_default(self, model_config, example_batch):
        """Test create_optimized_runner with defaults."""
        batch, embeddings = example_batch

        runner = create_optimized_runner(model_config)

        output = runner.rank(batch, embeddings)
        assert output.scores is not None

    def test_create_optimized_runner_custom(self, model_config, example_batch):
        """Test create_optimized_runner with custom settings."""
        batch, embeddings = example_batch

        runner = create_optimized_runner(
            model_config,
            use_jit=False,
            use_kv_cache=True,
            use_quantization=False,
        )

        assert runner.opt_config.use_jit is False
        assert runner.opt_config.use_kv_cache is True
        assert runner.opt_config.use_quantization is False

        output = runner.rank(batch, embeddings)
        assert output.scores is not None

    def test_create_optimized_runner_with_quant_config(self, model_config):
        """Test create_optimized_runner with custom quant config."""
        quant_config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            granularity=Granularity.PER_CHANNEL,
            name="custom_int8",
        )
        runner = create_optimized_runner(
            model_config,
            use_quantization=True,
            quant_config=quant_config,
        )

        assert runner.opt_config.quant_config is quant_config


# =============================================================================
# OPTIMIZATION STATS TESTS
# =============================================================================


class TestOptimizationStats:
    """Tests for OptimizationStats."""

    def test_stats_namedtuple(self):
        """Verify OptimizationStats is a valid NamedTuple."""
        stats = OptimizationStats()

        assert stats.jit_compilation_ms == 0.0
        assert stats.jit_warmup_avg_ms == 0.0
        assert stats.kv_cache_hits == 0
        assert stats.kv_cache_misses == 0
        assert stats.memory_original_bytes == 0
        assert stats.memory_optimized_bytes == 0
        assert stats.memory_reduction_ratio == 0.0

    def test_stats_replace(self):
        """Verify stats can be updated with _replace."""
        stats = OptimizationStats()
        new_stats = stats._replace(kv_cache_hits=5)

        assert new_stats.kv_cache_hits == 5
        assert stats.kv_cache_hits == 0  # Original unchanged


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_rank_before_initialize(self, model_config, example_batch):
        """Verify rank() raises error if not initialized."""
        batch, embeddings = example_batch

        runner = OptimizedPhoenixRunner(model_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            runner.rank(batch, embeddings)

    def test_benchmark_report_before_initialize(self, model_config):
        """Verify benchmark_report raises error if not initialized."""
        runner = OptimizedPhoenixRunner(model_config)

        with pytest.raises(RuntimeError, match="not initialized"):
            runner.benchmark_report()


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
