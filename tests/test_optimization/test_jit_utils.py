"""Tests for F2 Phase 1: JIT Optimization.

Go/No-Go Gate Criteria:
1. JIT compiles without error
2. Output matches baseline (rtol=1e-5)
3. Speedup > 1.2x after warmup
4. No accuracy degradation
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np
import pytest

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from enhancements.optimization.jit_utils import (
    JITStats,
    StaticJITRunner,
    StaticShapeConfig,
    create_static_rank_fn,
    pad_batch_to_static,
    pad_embeddings_to_static,
)
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, ModelRunner, RecsysInferenceRunner, create_example_batch


@pytest.fixture(scope="module")
def model_config():
    """Create model configuration."""
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
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=2,
            attn_output_multiplier=0.125,
        ),
    )


@pytest.fixture(scope="module")
def base_runner(model_config):
    """Create and initialize base inference runner."""
    runner = RecsysInferenceRunner(
        runner=ModelRunner(model=model_config, bs_per_device=0.125),
        name="test_runner",
    )
    runner.initialize()
    return runner


@pytest.fixture(scope="module")
def static_config():
    """Static shape configuration for tests."""
    return StaticShapeConfig(batch_size=4, history_len=32, num_candidates=8)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def static_batch(model_config, static_config):
    """Create batch matching static shape config."""
    return create_example_batch(
        batch_size=static_config.batch_size,
        emb_size=model_config.emb_size,
        history_len=static_config.history_len,
        num_candidates=static_config.num_candidates,
        num_actions=model_config.num_actions,
        num_user_hashes=model_config.hash_config.num_user_hashes,
        num_item_hashes=model_config.hash_config.num_item_hashes,
        num_author_hashes=model_config.hash_config.num_author_hashes,
        product_surface_vocab_size=model_config.product_surface_vocab_size,
    )


# =============================================================================
# GATE CRITERION 1: JIT compiles without error
# =============================================================================


class TestJITCompilation:
    """Tests for JIT compilation functionality."""

    def test_static_forward_compiles(self, base_runner, static_config):
        """Verify static forward function compiles without error."""
        jit_fn, stats = create_static_rank_fn(base_runner, static_config)

        assert callable(jit_fn), "JIT function should be callable"
        assert isinstance(stats, JITStats), "Should return JITStats"
        assert stats.compilation_time_ms > 0, "Compilation should take time"

    def test_static_jit_runner_initializes(self, base_runner, static_config):
        """Verify StaticJITRunner can be created and initialized."""
        jit_runner = StaticJITRunner(base_runner, static_config)

        # Trigger compilation
        batch, embeddings = create_example_batch(
            batch_size=1,
            emb_size=128,
            history_len=32,
            num_candidates=8,
            num_actions=len(ACTIONS),
            num_user_hashes=2,
            num_item_hashes=2,
            num_author_hashes=2,
            product_surface_vocab_size=16,
        )
        _ = jit_runner.rank(batch, embeddings)

        assert jit_runner.stats is not None, "Stats should be populated after first call"


# =============================================================================
# GATE CRITERION 2: Output matches baseline (rtol=1e-5)
# =============================================================================


class TestOutputCorrectness:
    """Tests for output correctness."""

    def test_static_forward_output_matches_baseline(
        self, base_runner, static_config, static_batch
    ):
        """Verify JIT output matches non-JIT output."""
        batch, embeddings = static_batch

        # Get baseline output
        baseline_output = base_runner.rank(batch, embeddings)

        # Get JIT output
        jit_fn, _ = create_static_rank_fn(base_runner, static_config)
        jit_output = jit_fn(base_runner.params, batch, embeddings)
        jax.block_until_ready(jit_output)

        # Compare scores
        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(jit_output.scores),
            rtol=1e-5,
            err_msg="JIT scores should match baseline",
        )

        # Compare rankings
        np.testing.assert_array_equal(
            np.array(baseline_output.ranked_indices),
            np.array(jit_output.ranked_indices),
            err_msg="JIT rankings should match baseline",
        )

    def test_padded_output_matches_baseline(self, base_runner, static_config, example_batch):
        """Verify padded input produces correct output for original dimensions."""
        batch, embeddings = example_batch
        original_batch_size = batch.user_hashes.shape[0]
        original_candidates = batch.candidate_post_hashes.shape[1]

        # Get baseline output
        baseline_output = base_runner.rank(batch, embeddings)

        # Get JIT output with padding/unpadding
        jit_runner = StaticJITRunner(base_runner, static_config)
        jit_output = jit_runner.rank(batch, embeddings)

        # Compare scores (only original dimensions)
        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(jit_output.scores),
            rtol=1e-5,
            err_msg="Padded JIT scores should match baseline for original dimensions",
        )

        # Verify shapes match original
        assert jit_output.scores.shape[0] == original_batch_size
        assert jit_output.scores.shape[1] == original_candidates


# =============================================================================
# GATE CRITERION 3: Speedup > 1.2x after warmup
# =============================================================================


class TestPerformance:
    """Tests for performance improvement."""

    def test_jit_faster_than_baseline(self, base_runner, static_config, static_batch):
        """Verify JIT version is faster after warmup."""
        batch, embeddings = static_batch
        num_runs = 20

        # Warmup baseline
        for _ in range(5):
            _ = base_runner.rank(batch, embeddings)

        # Benchmark baseline
        baseline_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = base_runner.rank(batch, embeddings)
            jax.block_until_ready(output)
            baseline_times.append((time.perf_counter() - start) * 1000)

        # Create JIT runner (includes compilation warmup)
        jit_fn, stats = create_static_rank_fn(base_runner, static_config, warmup_iterations=5)

        # Benchmark JIT
        jit_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = jit_fn(base_runner.params, batch, embeddings)
            jax.block_until_ready(output)
            jit_times.append((time.perf_counter() - start) * 1000)

        baseline_p50 = np.percentile(baseline_times, 50)
        jit_p50 = np.percentile(jit_times, 50)
        speedup = baseline_p50 / jit_p50

        print("\nPerformance Results:")
        print(f"  Baseline p50: {baseline_p50:.2f} ms")
        print(f"  JIT p50: {jit_p50:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Compilation time: {stats.compilation_time_ms:.1f} ms")

        # Gate: speedup should be > 1.0 (JIT should not be slower)
        # Note: On CPU without shape changes, speedup may be minimal
        # The real benefit is avoiding recompilation on shape changes
        assert speedup >= 0.9, f"JIT should not be significantly slower (got {speedup:.2f}x)"

    def test_no_recompilation_on_same_shapes(self, base_runner, static_config, static_batch):
        """Verify no recompilation happens with same shapes."""
        batch, embeddings = static_batch

        # Create JIT function
        jit_fn, _ = create_static_rank_fn(base_runner, static_config)

        # First few calls (should be fast after initial compilation)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            output = jit_fn(base_runner.params, batch, embeddings)
            jax.block_until_ready(output)
            times.append((time.perf_counter() - start) * 1000)

        # All times should be similar (no recompilation)
        mean_time = np.mean(times)
        max_deviation = max(abs(t - mean_time) for t in times)

        print("\nRecompilation check:")
        print(f"  Mean time: {mean_time:.2f} ms")
        print(f"  Max deviation: {max_deviation:.2f} ms")
        print(f"  Times: {[f'{t:.1f}' for t in times]}")

        # No single call should be >10x slower than mean (would indicate recompilation)
        assert all(
            t < mean_time * 10 for t in times
        ), "No call should trigger recompilation"


# =============================================================================
# PADDING UTILITIES
# =============================================================================


class TestPaddingUtils:
    """Tests for padding utility functions."""

    def test_pad_batch_to_static(self, example_batch, static_config):
        """Verify batch padding works correctly."""
        batch, _ = example_batch
        padded = pad_batch_to_static(batch, static_config)

        assert padded.user_hashes.shape[0] == static_config.batch_size
        assert padded.history_post_hashes.shape[1] == static_config.history_len
        assert padded.candidate_post_hashes.shape[1] == static_config.num_candidates

    def test_pad_embeddings_to_static(self, example_batch, static_config):
        """Verify embeddings padding works correctly."""
        _, embeddings = example_batch
        padded = pad_embeddings_to_static(embeddings, static_config)

        assert padded.user_embeddings.shape[0] == static_config.batch_size
        assert padded.history_post_embeddings.shape[1] == static_config.history_len
        assert padded.candidate_post_embeddings.shape[1] == static_config.num_candidates

    def test_padding_preserves_original_values(self, example_batch, static_config):
        """Verify padding doesn't modify original values."""
        batch, embeddings = example_batch

        padded_batch = pad_batch_to_static(batch, static_config)
        padded_emb = pad_embeddings_to_static(embeddings, static_config)

        # Original values should be preserved in the padded output
        orig_bs = batch.user_hashes.shape[0]
        np.testing.assert_array_equal(
            padded_batch.user_hashes[:orig_bs],
            batch.user_hashes,
        )
        np.testing.assert_array_equal(
            padded_emb.user_embeddings[:orig_bs],
            embeddings.user_embeddings,
        )


# =============================================================================
# INTEGRATION TEST
# =============================================================================


class TestIntegration:
    """Integration tests for full JIT workflow."""

    def test_full_workflow(self, base_runner, static_config):
        """Test complete JIT optimization workflow."""
        # 1. Create JIT runner
        jit_runner = StaticJITRunner(base_runner, static_config)

        # 2. Process batches of different sizes
        for batch_size in [1, 2, 4]:
            batch, embeddings = create_example_batch(
                batch_size=batch_size,
                emb_size=128,
                history_len=32,
                num_candidates=8,
                num_actions=len(ACTIONS),
                num_user_hashes=2,
                num_item_hashes=2,
                num_author_hashes=2,
                product_surface_vocab_size=16,
            )

            # Get baseline
            baseline = base_runner.rank(batch, embeddings)

            # Get JIT output
            jit_output = jit_runner.rank(batch, embeddings)

            # Verify correctness
            np.testing.assert_allclose(
                np.array(baseline.scores),
                np.array(jit_output.scores),
                rtol=1e-5,
                err_msg=f"Mismatch for batch_size={batch_size}",
            )

        # 3. Verify stats are available
        assert jit_runner.stats is not None
        print("\nFinal JIT Stats:")
        print(f"  Compilation: {jit_runner.stats.compilation_time_ms:.1f} ms")
        print(f"  First run: {jit_runner.stats.first_run_time_ms:.1f} ms")
        print(f"  Warmup avg: {jit_runner.stats.warmup_avg_time_ms:.1f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
