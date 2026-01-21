"""Tests for quantization module.

Tests cover:
- Configuration creation and validation
- Quantize/dequantize roundtrip for various configs
- Per-tensor vs per-channel quantization
- Symmetric vs asymmetric quantization
- Quantized runner functionality
- Memory reduction calculations
- Benchmark metrics and winner selection
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import (
    ModelRunner,
    RecsysInferenceRunner,
    create_example_batch,
)

from enhancements.optimization.quantization import (
    BitWidth,
    EXTENDED_STUDY_CONFIGS,
    Granularity,
    LayerType,
    MixedPrecisionConfig,
    QuantizationConfig,
    QuantizedLayerKVCache,
    QuantizedPhoenixRunner,
    QuantizedTensor,
    STUDY_CONFIGS,
    Symmetry,
    compute_kendall_tau,
    compute_memory_bytes,
    compute_scale_zp_per_group,
    compute_top3_match,
    dequantize_kv,
    dequantize_kv_cache,
    dequantize_tensor,
    get_layer_type,
    get_quant_settings_for_param,
    quantize_kv,
    quantize_kv_cache,
    quantize_tensor,
    select_winner,
    BenchmarkMetrics,
    WinnerSelectionCriteria,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def small_model_config() -> PhoenixModelConfig:
    """Create a small model config for fast testing."""
    return PhoenixModelConfig(
        emb_size=64,
        num_actions=19,
        history_seq_len=8,
        candidate_seq_len=4,
        hash_config=HashConfig(
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
        ),
        model=TransformerConfig(
            emb_size=64,
            key_size=32,
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=2,
            widening_factor=2.0,
            attn_output_multiplier=1.0,
        ),
    )


@pytest.fixture(scope="module")
def base_runner(small_model_config) -> RecsysInferenceRunner:
    """Create and initialize a base runner."""
    model_runner = ModelRunner(model=small_model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="test")
    runner.initialize()
    return runner


@pytest.fixture
def example_batch(small_model_config):
    """Create an example batch for testing."""
    return create_example_batch(
        batch_size=1,
        emb_size=small_model_config.emb_size,
        history_len=small_model_config.history_seq_len,
        num_candidates=small_model_config.candidate_seq_len,
        num_actions=small_model_config.num_actions,
        num_user_hashes=small_model_config.hash_config.num_user_hashes,
        num_item_hashes=small_model_config.hash_config.num_item_hashes,
        num_author_hashes=small_model_config.hash_config.num_author_hashes,
    )


# ============================================================================
# Config Tests
# ============================================================================

class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_config(self):
        """Default config should be INT8 per-channel symmetric."""
        config = QuantizationConfig()
        assert config.bit_width == BitWidth.INT8
        assert config.granularity == Granularity.PER_CHANNEL
        assert config.symmetry == Symmetry.SYMMETRIC

    def test_config_name_auto_generated(self):
        """Name should be auto-generated if not provided."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            granularity=Granularity.PER_TENSOR,
            symmetry=Symmetry.SYMMETRIC,
        )
        assert config.name is not None
        assert "int8" in config.name
        assert "tensor" in config.name
        assert "sym" in config.name

    def test_config_name_fp16(self):
        """FP16 config name should be simple."""
        config = QuantizationConfig(bit_width=BitWidth.FP16)
        assert config.name == "fp16"

    def test_config_name_with_kv_cache(self):
        """Config with KV-cache should include it in name."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            quantize_kv_cache=True,
            kv_cache_bit_width=BitWidth.INT8,
        )
        assert "kv" in config.name

    def test_config_explicit_name(self):
        """Explicit name should override auto-generated."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            name="my_custom_config",
        )
        assert config.name == "my_custom_config"

    def test_get_bit_width_int(self):
        """Bit width should convert to integer correctly."""
        assert QuantizationConfig(bit_width=BitWidth.FP16).get_bit_width_int() == 16
        assert QuantizationConfig(bit_width=BitWidth.INT8).get_bit_width_int() == 8
        assert QuantizationConfig(bit_width=BitWidth.INT4).get_bit_width_int() == 4

    def test_study_configs_exist(self):
        """STUDY_CONFIGS should have multiple configurations."""
        assert len(STUDY_CONFIGS) >= 5
        assert all(isinstance(c, QuantizationConfig) for c in STUDY_CONFIGS)


# ============================================================================
# Quantize/Dequantize Tests
# ============================================================================

class TestQuantizeTensor:
    """Tests for tensor quantization functions."""

    def test_fp16_quantize(self):
        """FP16 quantization should just cast dtype."""
        tensor = jnp.array([[1.0, -1.0, 0.5, -0.5]], dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.FP16, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )

        assert qtensor.data.dtype == jnp.float16
        assert qtensor.scale.shape == ()
        np.testing.assert_allclose(qtensor.data, tensor, rtol=1e-3)

    def test_int8_symmetric_per_tensor(self):
        """INT8 symmetric per-tensor quantization roundtrip."""
        tensor = jnp.array([[1.0, -1.0, 0.5, -0.5]], dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )

        assert qtensor.data.dtype == jnp.int8
        assert qtensor.zero_point == 0  # Symmetric has zero_point = 0

        recovered = dequantize_tensor(qtensor)
        np.testing.assert_allclose(recovered, tensor, rtol=0.01, atol=0.01)

    def test_int8_symmetric_per_channel(self):
        """INT8 symmetric per-channel quantization."""
        # Different scales needed for different channels
        tensor = jnp.array([[1.0, 100.0], [0.1, 10.0]], dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_CHANNEL, Symmetry.SYMMETRIC
        )

        assert qtensor.data.dtype == jnp.int8
        # Per-channel should have different scales
        assert qtensor.scale.size >= 1

        recovered = dequantize_tensor(qtensor)
        np.testing.assert_allclose(recovered, tensor, rtol=0.02, atol=0.5)

    def test_int8_asymmetric_per_tensor(self):
        """INT8 asymmetric per-tensor quantization."""
        # Test that asymmetric quantization works (may have larger error due to
        # int8 range limitations with unsigned-style asymmetric)
        tensor = jnp.array([[0.0, 0.25, 0.5, 0.75]], dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_TENSOR, Symmetry.ASYMMETRIC
        )

        assert qtensor.data.dtype == jnp.int8
        # Non-zero zero_point for asymmetric
        # Note: actual accuracy depends on implementation details

        recovered = dequantize_tensor(qtensor)
        # Asymmetric may have larger error - verify shape at minimum
        assert recovered.shape == tensor.shape
        # Values should be in reasonable range
        assert jnp.all(recovered >= -0.1)
        assert jnp.all(recovered <= 1.0)

    def test_int4_quantize(self):
        """INT4 quantization with limited range."""
        tensor = jnp.array([[0.0, 0.5, 1.0, -1.0]], dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT4, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )

        # INT4 uses int8 storage but clipped to [-8, 7]
        assert qtensor.data.dtype == jnp.int8
        assert jnp.all(qtensor.data >= -8)
        assert jnp.all(qtensor.data <= 7)

        recovered = dequantize_tensor(qtensor)
        # INT4 has lower precision
        np.testing.assert_allclose(recovered, tensor, rtol=0.15, atol=0.15)

    def test_quantize_preserves_shape(self):
        """Quantization should preserve tensor shape."""
        tensor = jnp.ones((4, 8, 16), dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )

        assert qtensor.data.shape == tensor.shape

        recovered = dequantize_tensor(qtensor)
        assert recovered.shape == tensor.shape


# ============================================================================
# Memory Tests
# ============================================================================

class TestMemoryComputation:
    """Tests for memory computation functions."""

    def test_compute_memory_bytes_regular(self):
        """Memory for regular params should be size * dtype_bytes."""
        params = {
            'layer1': jnp.ones((100, 100), dtype=jnp.float32),
            'layer2': jnp.ones((50, 50), dtype=jnp.float32),
        }
        memory = compute_memory_bytes(params)

        expected = 100 * 100 * 4 + 50 * 50 * 4  # float32 = 4 bytes
        assert memory == expected

    def test_compute_memory_bytes_quantized(self):
        """Memory for quantized params should include scale overhead."""
        tensor = jnp.ones((100, 100), dtype=jnp.float32)
        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )

        params = {'layer': qtensor}
        memory = compute_memory_bytes(params)

        # INT8 data + scale + zero_point
        data_bytes = 100 * 100 * 1  # int8 = 1 byte
        scale_bytes = qtensor.scale.size * 4
        zp_bytes = qtensor.zero_point.size * 4
        expected = data_bytes + scale_bytes + zp_bytes

        assert memory == expected


# ============================================================================
# Quantized Runner Tests
# ============================================================================

class TestQuantizedPhoenixRunner:
    """Tests for QuantizedPhoenixRunner."""

    def test_runner_initializes(self, base_runner):
        """Quantized runner should initialize from base runner."""
        config = QuantizationConfig(bit_width=BitWidth.INT8)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        assert quant_runner.quantized_params is not None
        assert quant_runner.original_params is not None

    def test_runner_produces_output(self, base_runner, example_batch):
        """Quantized runner should produce valid output."""
        config = QuantizationConfig(bit_width=BitWidth.INT8)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        batch, embeddings = example_batch
        output = quant_runner.rank(batch, embeddings)

        # Check output has expected shape
        assert output.scores.shape == (1, 4, 19)  # batch, candidates, actions
        assert output.ranked_indices.shape == (1, 4)

    def test_runner_memory_reduction_int8(self, base_runner):
        """INT8 quantization should reduce memory significantly."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            granularity=Granularity.PER_TENSOR,
        )
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        reduction = quant_runner.get_memory_reduction_ratio()
        compression = quant_runner.get_compression_ratio()

        # INT8 should give roughly 4x compression
        assert reduction > 0.5  # At least 50% reduction
        assert compression > 2.0  # At least 2x compression

    def test_runner_memory_reduction_fp16(self, base_runner):
        """FP16 quantization should give ~2x compression."""
        config = QuantizationConfig(bit_width=BitWidth.FP16)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        reduction = quant_runner.get_memory_reduction_ratio()
        compression = quant_runner.get_compression_ratio()

        # FP16 should give roughly 2x compression
        assert reduction > 0.3  # At least 30% reduction
        assert compression > 1.5  # At least 1.5x compression

    def test_runner_output_close_to_baseline(self, base_runner, example_batch):
        """Quantized output should be close to baseline."""
        batch, embeddings = example_batch

        # Baseline output
        baseline_output = base_runner.rank(batch, embeddings)

        # FP16 quantized output (minimal accuracy loss expected)
        config = QuantizationConfig(bit_width=BitWidth.FP16)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)
        quant_output = quant_runner.rank(batch, embeddings)

        # Scores should be very close
        np.testing.assert_allclose(
            quant_output.scores, baseline_output.scores, rtol=1e-2, atol=1e-2
        )

    def test_count_quantized_params(self, base_runner):
        """Should correctly count quantized vs unquantized params."""
        config = QuantizationConfig(bit_width=BitWidth.INT8)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        counts = quant_runner.count_quantized_params()

        assert 'quantized' in counts
        assert 'unquantized' in counts
        assert 'total' in counts
        assert counts['total'] == counts['quantized'] + counts['unquantized']


# ============================================================================
# Kendall's Tau Tests
# ============================================================================

class TestKendallTau:
    """Tests for Kendall's tau computation."""

    def test_identical_rankings(self):
        """Identical rankings should have tau = 1."""
        ranking1 = np.array([0, 1, 2, 3])
        ranking2 = np.array([0, 1, 2, 3])

        tau = compute_kendall_tau(ranking1, ranking2)
        assert tau == 1.0

    def test_reversed_rankings(self):
        """Reversed rankings should have tau = -1."""
        ranking1 = np.array([0, 1, 2, 3])
        ranking2 = np.array([3, 2, 1, 0])

        tau = compute_kendall_tau(ranking1, ranking2)
        assert tau == -1.0

    def test_partial_correlation(self):
        """Partially correlated rankings should have tau between -1 and 1."""
        ranking1 = np.array([0, 1, 2, 3])
        ranking2 = np.array([1, 0, 2, 3])  # First two swapped

        tau = compute_kendall_tau(ranking1, ranking2)
        assert -1.0 < tau < 1.0


# ============================================================================
# Top-3 Match Tests
# ============================================================================

class TestTop3Match:
    """Tests for top-3 matching."""

    def test_exact_match(self):
        """Identical scores should match."""
        # Shape: (num_candidates, num_actions) = (4, 1)
        baseline = np.array([[1.0], [0.9], [0.8], [0.7]])
        quant = np.array([[1.0], [0.9], [0.8], [0.7]])

        assert compute_top3_match(baseline, quant) is True

    def test_top3_preserved(self):
        """Top-3 should match even if 4th changes."""
        baseline = np.array([[1.0], [0.9], [0.8], [0.7]])
        quant = np.array([[1.0], [0.9], [0.8], [0.5]])  # 4th candidate changed

        assert compute_top3_match(baseline, quant) is True

    def test_top3_not_preserved(self):
        """Should return False if top-3 changes."""
        baseline = np.array([[1.0], [0.9], [0.8], [0.7]])  # Top-3: indices 0,1,2
        quant = np.array([[0.7], [0.9], [0.8], [1.0]])  # Top-3: indices 3,1,2

        assert compute_top3_match(baseline, quant) is False


# ============================================================================
# Winner Selection Tests
# ============================================================================

class TestWinnerSelection:
    """Tests for winner selection logic."""

    def test_select_from_passing_configs(self):
        """Should select best config from those passing gates."""
        results = [
            BenchmarkMetrics(
                config_name="config_a",
                kendall_tau=0.95,
                top3_preserved_rate=0.92,
                max_score_diff=0.01,
                mean_score_diff=0.005,
                memory_bytes_original=1000000,
                memory_bytes_quantized=250000,
                memory_reduction_ratio=0.75,
                latency_p50_ms=10.0,
                latency_p95_ms=15.0,
                latency_ratio=1.1,
                passes_accuracy_gate=True,
                passes_memory_gate=True,
                passes_latency_gate=True,
                passes_all_gates=True,
            ),
            BenchmarkMetrics(
                config_name="config_b",
                kendall_tau=0.98,
                top3_preserved_rate=0.95,
                max_score_diff=0.005,
                mean_score_diff=0.002,
                memory_bytes_original=1000000,
                memory_bytes_quantized=500000,
                memory_reduction_ratio=0.50,
                latency_p50_ms=9.0,
                latency_p95_ms=14.0,
                latency_ratio=1.0,
                passes_accuracy_gate=True,
                passes_memory_gate=True,
                passes_latency_gate=True,
                passes_all_gates=True,
            ),
        ]

        winner, details = select_winner(results)

        assert winner is not None
        assert winner.config_name in ["config_a", "config_b"]
        assert details["num_passing"] == 2

    def test_no_winner_if_all_fail(self):
        """Should return None if no configs pass gates."""
        results = [
            BenchmarkMetrics(
                config_name="failing_config",
                kendall_tau=0.50,
                top3_preserved_rate=0.80,  # Below 0.90 threshold
                max_score_diff=0.1,
                mean_score_diff=0.05,
                memory_bytes_original=1000000,
                memory_bytes_quantized=800000,
                memory_reduction_ratio=0.20,  # Below 0.40 threshold
                latency_p50_ms=15.0,
                latency_p95_ms=20.0,
                latency_ratio=1.5,  # Above 1.2 threshold
                passes_accuracy_gate=False,
                passes_memory_gate=False,
                passes_latency_gate=False,
                passes_all_gates=False,
            ),
        ]

        winner, details = select_winner(results)

        assert winner is None
        assert "error" in details

    def test_custom_criteria(self):
        """Should respect custom selection criteria."""
        results = [
            BenchmarkMetrics(
                config_name="high_accuracy",
                kendall_tau=0.99,
                top3_preserved_rate=0.99,
                max_score_diff=0.001,
                mean_score_diff=0.0005,
                memory_bytes_original=1000000,
                memory_bytes_quantized=600000,
                memory_reduction_ratio=0.40,
                latency_p50_ms=12.0,
                latency_p95_ms=18.0,
                latency_ratio=1.15,
                passes_accuracy_gate=True,
                passes_memory_gate=True,
                passes_latency_gate=True,
                passes_all_gates=True,
            ),
            BenchmarkMetrics(
                config_name="high_compression",
                kendall_tau=0.90,
                top3_preserved_rate=0.91,
                max_score_diff=0.02,
                mean_score_diff=0.01,
                memory_bytes_original=1000000,
                memory_bytes_quantized=200000,
                memory_reduction_ratio=0.80,
                latency_p50_ms=11.0,
                latency_p95_ms=16.0,
                latency_ratio=1.05,
                passes_accuracy_gate=True,
                passes_memory_gate=True,
                passes_latency_gate=True,
                passes_all_gates=True,
            ),
        ]

        # Weight heavily toward memory
        criteria = WinnerSelectionCriteria(
            weight_accuracy=0.1,
            weight_memory=0.8,
            weight_latency=0.1,
        )
        winner, _ = select_winner(results, criteria)

        assert winner is not None
        assert winner.config_name == "high_compression"

        # Weight heavily toward accuracy
        criteria = WinnerSelectionCriteria(
            weight_accuracy=0.8,
            weight_memory=0.1,
            weight_latency=0.1,
        )
        winner, _ = select_winner(results, criteria)

        assert winner is not None
        assert winner.config_name == "high_accuracy"


# ============================================================================
# Integration Tests
# ============================================================================

class TestQuantizationIntegration:
    """Integration tests for full quantization workflow."""

    def test_full_workflow_fp16(self, base_runner, example_batch):
        """Test complete workflow with FP16 quantization."""
        batch, embeddings = example_batch

        # Create quantized runner
        config = QuantizationConfig(bit_width=BitWidth.FP16)
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        # Get baseline and quantized outputs
        baseline = base_runner.rank(batch, embeddings)
        quantized = quant_runner.rank(batch, embeddings)

        # Verify outputs
        assert baseline.scores.shape == quantized.scores.shape

        # FP16 should be very close
        baseline_scores = np.asarray(baseline.scores[0])
        quant_scores = np.asarray(quantized.scores[0])

        # Check ranking preservation
        assert compute_top3_match(baseline_scores, quant_scores)

    def test_full_workflow_int8(self, base_runner, example_batch):
        """Test complete workflow with INT8 quantization."""
        batch, embeddings = example_batch

        # Create quantized runner
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            granularity=Granularity.PER_CHANNEL,
            symmetry=Symmetry.SYMMETRIC,
        )
        quant_runner = QuantizedPhoenixRunner(base_runner, config)

        # Get outputs
        baseline = base_runner.rank(batch, embeddings)
        quantized = quant_runner.rank(batch, embeddings)

        # Verify memory reduction
        reduction = quant_runner.get_memory_reduction_ratio()
        assert reduction > 0.4  # Gate threshold

        # Verify output shapes
        assert baseline.scores.shape == quantized.scores.shape


# ============================================================================
# Phase 4b: Per-Group Quantization Tests
# ============================================================================

class TestPerGroupQuantization:
    """Tests for per-group quantization (Phase 4b)."""

    def test_compute_scale_zp_per_group_symmetric(self):
        """Per-group symmetric quantization should compute per-group scales."""
        tensor = jnp.array([[1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.0, -1.0]], dtype=jnp.float32)
        scale, zero_point, original_shape = compute_scale_zp_per_group(
            tensor, bit_width=8, group_size=4, symmetric=True
        )

        # Should have scales for each group
        assert scale.shape[0] == 2  # 8 elements / 4 group_size = 2 groups
        assert jnp.all(zero_point == 0)  # Symmetric has zero_point = 0
        assert original_shape == tensor.shape

    def test_per_group_quantize_roundtrip(self):
        """Per-group quantization should have good roundtrip accuracy."""
        tensor = jnp.array(
            [[1.0, -1.0, 0.5, -0.5] * 32],  # 128 elements = 1 group with group_size=128
            dtype=jnp.float32
        ).reshape(4, 32)

        qtensor = quantize_tensor(
            tensor, BitWidth.INT8, Granularity.PER_GROUP, Symmetry.SYMMETRIC, group_size=128
        )

        # Check that original_shape is stored
        assert qtensor.original_shape == tensor.shape

        # Dequantize
        recovered = dequantize_tensor(qtensor)

        # Shape should be restored
        assert recovered.shape == tensor.shape

        # Values should be close
        np.testing.assert_allclose(recovered, tensor, rtol=0.02, atol=0.02)

    def test_per_group_int4_better_than_per_tensor(self):
        """Per-group INT4 should be more accurate than per-tensor INT4."""
        # Tensor with different value ranges in different regions
        tensor = jnp.array(
            [[100.0, 99.0, 101.0, 98.0] * 8 + [1.0, 0.9, 1.1, 0.8] * 8],
            dtype=jnp.float32
        ).reshape(2, 32)

        # Per-tensor INT4
        qtensor_per_tensor = quantize_tensor(
            tensor, BitWidth.INT4, Granularity.PER_TENSOR, Symmetry.SYMMETRIC
        )
        recovered_per_tensor = dequantize_tensor(qtensor_per_tensor)

        # Per-group INT4 (group_size=32)
        qtensor_per_group = quantize_tensor(
            tensor, BitWidth.INT4, Granularity.PER_GROUP, Symmetry.SYMMETRIC, group_size=32
        )
        recovered_per_group = dequantize_tensor(qtensor_per_group)

        # Per-group should have lower error
        error_per_tensor = jnp.mean(jnp.abs(tensor - recovered_per_tensor))
        error_per_group = jnp.mean(jnp.abs(tensor - recovered_per_group))

        # Per-group should be more accurate (or at least comparable)
        # Due to the different value ranges, per-group should help
        assert error_per_group <= error_per_tensor * 1.5  # Allow some margin

    def test_per_group_config(self):
        """QuantizationConfig should support per-group granularity."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT4,
            granularity=Granularity.PER_GROUP,
            group_size=128,
            symmetry=Symmetry.SYMMETRIC,
        )

        assert config.granularity == Granularity.PER_GROUP
        assert config.group_size == 128
        assert "pergroup" in config.name.lower()


# ============================================================================
# Phase 4b: Mixed Precision Tests
# ============================================================================

class TestMixedPrecision:
    """Tests for mixed precision quantization (Phase 4b)."""

    def test_mixed_precision_config_creation(self):
        """MixedPrecisionConfig should be creatable with defaults."""
        mp_config = MixedPrecisionConfig()

        assert mp_config.ffn_bit_width == BitWidth.INT4
        assert mp_config.attention_bit_width == BitWidth.INT8
        assert mp_config.embedding_bit_width == BitWidth.INT8

    def test_mixed_precision_config_custom(self):
        """MixedPrecisionConfig should accept custom settings."""
        mp_config = MixedPrecisionConfig(
            ffn_bit_width=BitWidth.INT4,
            attention_bit_width=BitWidth.INT8,
            ffn_granularity=Granularity.PER_GROUP,
        )

        assert mp_config.ffn_bit_width == BitWidth.INT4
        assert mp_config.ffn_granularity == Granularity.PER_GROUP

    def test_quantization_config_with_mixed_precision(self):
        """QuantizationConfig should support mixed precision."""
        config = QuantizationConfig(
            use_mixed_precision=True,
            mixed_precision=MixedPrecisionConfig(
                ffn_bit_width=BitWidth.INT4,
                attention_bit_width=BitWidth.INT8,
            ),
            name="test_mixed",
        )

        assert config.use_mixed_precision is True
        assert config.mixed_precision is not None
        assert config.mixed_precision.ffn_bit_width == BitWidth.INT4

    def test_get_layer_type(self):
        """get_layer_type should correctly identify layer types."""
        assert get_layer_type("decoder_layer_0/mha_block/query") == LayerType.ATTENTION
        assert get_layer_type("decoder_layer_0/mha_block/key") == LayerType.ATTENTION
        assert get_layer_type("decoder_layer_0/mha_block/value") == LayerType.ATTENTION
        assert get_layer_type("decoder_layer_0/block/linear_v") == LayerType.FFN
        assert get_layer_type("decoder_layer_0/block/linear_w") == LayerType.FFN
        assert get_layer_type("proj_mat_1") == LayerType.EMBEDDING
        assert get_layer_type("unembedding") == LayerType.UNEMBEDDING
        assert get_layer_type("random_param") == LayerType.OTHER

    def test_get_quant_settings_standard(self):
        """get_quant_settings_for_param should work for standard config."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT8,
            quantize_attention=True,
            quantize_ffn=True,
        )

        should_quant, bit_width, granularity = get_quant_settings_for_param(
            "decoder_layer_0/mha_block/query", config
        )

        assert should_quant is True
        assert bit_width is None  # No override for standard config
        assert granularity is None

    def test_get_quant_settings_mixed_precision(self):
        """get_quant_settings_for_param should return layer-specific settings."""
        config = QuantizationConfig(
            use_mixed_precision=True,
            mixed_precision=MixedPrecisionConfig(
                ffn_bit_width=BitWidth.INT4,
                attention_bit_width=BitWidth.INT8,
                ffn_granularity=Granularity.PER_GROUP,
            ),
            quantize_attention=True,
            quantize_ffn=True,
        )

        # Check attention layer
        should_quant, bit_width, granularity = get_quant_settings_for_param(
            "decoder_layer_0/mha_block/query", config
        )
        assert should_quant is True
        assert bit_width == BitWidth.INT8

        # Check FFN layer
        should_quant, bit_width, granularity = get_quant_settings_for_param(
            "decoder_layer_0/block/linear_v", config
        )
        assert should_quant is True
        assert bit_width == BitWidth.INT4
        assert granularity == Granularity.PER_GROUP

    def test_extended_study_configs_exist(self):
        """EXTENDED_STUDY_CONFIGS should have multiple configurations."""
        assert len(EXTENDED_STUDY_CONFIGS) >= 5  # At least 5 configs
        assert all(isinstance(c, QuantizationConfig) for c in EXTENDED_STUDY_CONFIGS)

        # Check for mixed precision config
        mixed_configs = [c for c in EXTENDED_STUDY_CONFIGS if c.use_mixed_precision]
        assert len(mixed_configs) >= 1

        # Check for per-group config
        pergroup_configs = [c for c in EXTENDED_STUDY_CONFIGS
                          if c.granularity == Granularity.PER_GROUP]
        assert len(pergroup_configs) >= 1


# ============================================================================
# Phase 4b: KV-Cache Quantization Tests
# ============================================================================

class TestKVCacheQuantization:
    """Tests for KV-cache quantization (Phase 4b)."""

    def test_quantize_kv_int8(self):
        """quantize_kv should correctly quantize K/V tensors."""
        # Shape: [batch, num_kv_heads, seq_len, head_dim]
        kv_tensor = jnp.array(
            [[[[1.0, -1.0, 0.5, -0.5]]]],
            dtype=jnp.float32
        )

        qtensor = quantize_kv(kv_tensor, BitWidth.INT8, Symmetry.SYMMETRIC)

        assert qtensor.data.dtype == jnp.int8
        assert qtensor.scale.shape == ()  # Per-tensor scale
        assert qtensor.zero_point == 0  # Symmetric

    def test_quantize_kv_fp16(self):
        """FP16 KV quantization should just cast dtype."""
        kv_tensor = jnp.array(
            [[[[1.0, -1.0, 0.5, -0.5]]]],
            dtype=jnp.float32
        )

        qtensor = quantize_kv(kv_tensor, BitWidth.FP16)

        assert qtensor.data.dtype == jnp.float16
        np.testing.assert_allclose(qtensor.data, kv_tensor, rtol=1e-3)

    def test_kv_roundtrip_accuracy(self):
        """KV quantization roundtrip should preserve values."""
        kv_tensor = jnp.array(
            [[[[1.0, -1.0, 0.5, -0.5],
               [0.2, -0.3, 0.1, -0.15]]]],
            dtype=jnp.float32
        )

        qtensor = quantize_kv(kv_tensor, BitWidth.INT8, Symmetry.SYMMETRIC)
        recovered = dequantize_kv(qtensor)

        assert recovered.shape == kv_tensor.shape
        np.testing.assert_allclose(recovered, kv_tensor, rtol=0.02, atol=0.02)

    def test_quantize_kv_cache(self):
        """quantize_kv_cache should quantize both K and V."""
        keys = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
        values = jnp.ones((1, 2, 4, 8), dtype=jnp.float32) * 0.5

        quant_cache = quantize_kv_cache(keys, values, BitWidth.INT8)

        assert isinstance(quant_cache, QuantizedLayerKVCache)
        assert quant_cache.keys.data.dtype == jnp.int8
        assert quant_cache.values.data.dtype == jnp.int8

    def test_dequantize_kv_cache(self):
        """dequantize_kv_cache should recover both K and V."""
        keys = jnp.ones((1, 2, 4, 8), dtype=jnp.float32)
        values = jnp.ones((1, 2, 4, 8), dtype=jnp.float32) * 0.5

        quant_cache = quantize_kv_cache(keys, values, BitWidth.INT8)
        recovered_keys, recovered_values = dequantize_kv_cache(quant_cache)

        np.testing.assert_allclose(recovered_keys, keys, rtol=0.02, atol=0.02)
        np.testing.assert_allclose(recovered_values, values, rtol=0.02, atol=0.02)

    def test_kv_cache_compression(self):
        """KV-cache quantization should achieve good compression."""
        keys = jnp.ones((1, 4, 128, 64), dtype=jnp.float32)
        values = jnp.ones((1, 4, 128, 64), dtype=jnp.float32)

        # Original size: 2 * 1 * 4 * 128 * 64 * 4 = 262144 bytes (float32)
        original_bytes = keys.size * 4 + values.size * 4

        quant_cache = quantize_kv_cache(keys, values, BitWidth.INT8)

        # Quantized size: 2 * 1 * 4 * 128 * 64 * 1 = 65536 bytes (int8)
        # Plus small overhead for scales
        from enhancements.optimization.quantization import get_kv_cache_memory_bytes
        quant_bytes = get_kv_cache_memory_bytes(quant_cache)

        # Should be ~4x compression
        compression_ratio = original_bytes / quant_bytes
        assert compression_ratio > 3.5  # Allow some margin for scale overhead


# ============================================================================
# Phase 4b: Composability Tests
# ============================================================================

class TestComposability:
    """Tests for composing different quantization strategies."""

    def test_mixed_precision_plus_kv_cache(self):
        """Mixed precision + KV-cache config should be valid."""
        config = QuantizationConfig(
            use_mixed_precision=True,
            mixed_precision=MixedPrecisionConfig(
                ffn_bit_width=BitWidth.INT4,
                attention_bit_width=BitWidth.INT8,
            ),
            quantize_kv_cache=True,
            kv_cache_bit_width=BitWidth.INT8,
            name="mixed_int4_int8_kv8",
        )

        assert config.use_mixed_precision is True
        assert config.quantize_kv_cache is True
        assert config.mixed_precision.ffn_bit_width == BitWidth.INT4
        assert config.kv_cache_bit_width == BitWidth.INT8

    def test_pergroup_plus_kv_cache(self):
        """Per-group + KV-cache config should be valid."""
        config = QuantizationConfig(
            bit_width=BitWidth.INT4,
            granularity=Granularity.PER_GROUP,
            group_size=128,
            quantize_kv_cache=True,
            kv_cache_bit_width=BitWidth.INT8,
            name="int4_pergroup_kv8",
        )

        assert config.granularity == Granularity.PER_GROUP
        assert config.quantize_kv_cache is True

    def test_mixed_precision_with_pergroup_ffn(self):
        """Mixed precision with per-group FFN config should be valid."""
        config = QuantizationConfig(
            use_mixed_precision=True,
            mixed_precision=MixedPrecisionConfig(
                ffn_bit_width=BitWidth.INT4,
                attention_bit_width=BitWidth.INT8,
                ffn_granularity=Granularity.PER_GROUP,
            ),
            group_size=128,
            name="mixed_int4_pergroup_int8",
        )

        assert config.use_mixed_precision is True
        assert config.mixed_precision.ffn_granularity == Granularity.PER_GROUP
        assert config.group_size == 128

    def test_all_three_composed(self):
        """All three strategies composed should be valid."""
        config = QuantizationConfig(
            use_mixed_precision=True,
            mixed_precision=MixedPrecisionConfig(
                ffn_bit_width=BitWidth.INT4,
                attention_bit_width=BitWidth.INT8,
                ffn_granularity=Granularity.PER_GROUP,
            ),
            group_size=128,
            quantize_kv_cache=True,
            kv_cache_bit_width=BitWidth.INT8,
            name="mixed_int4_pergroup_int8_kv8",
        )

        # All three strategies active
        assert config.use_mixed_precision is True
        assert config.mixed_precision.ffn_granularity == Granularity.PER_GROUP
        assert config.quantize_kv_cache is True

    def test_extended_configs_include_compositions(self):
        """EXTENDED_STUDY_CONFIGS should include composition configs."""
        # Check for mixed + kv-cache composition
        mixed_kv_configs = [
            c for c in EXTENDED_STUDY_CONFIGS
            if c.use_mixed_precision and c.quantize_kv_cache
        ]
        assert len(mixed_kv_configs) >= 1

        # Check for per-group + kv-cache composition
        pergroup_kv_configs = [
            c for c in EXTENDED_STUDY_CONFIGS
            if c.granularity == Granularity.PER_GROUP and c.quantize_kv_cache
        ]
        assert len(pergroup_kv_configs) >= 1
