"""Tests for F2 Phase 2: KV-Cache Implementation.

Go/No-Go Gate Criteria:
1. KV cache creates successfully
2. Cached output matches baseline (rtol=1e-5)
3. Cache hit faster than or equal to cache miss
4. Cache invalidates correctly on user change
"""

import sys
import time
from pathlib import Path

import jax
import numpy as np
import pytest

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from enhancements.optimization.kv_cache import (
    CachedJITRunner,
    CachedPhoenixRunner,
    KVCache,
    compute_user_hash,
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


def create_batch_with_different_user(model_config):
    """Create a batch with different user (different seed)."""
    # Use a different seed to get different user hashes
    rng = np.random.default_rng(9999)

    batch_size = 1
    hash_config = model_config.hash_config
    history_len = model_config.history_seq_len
    num_candidates = model_config.candidate_seq_len

    from phoenix.recsys_model import RecsysBatch, RecsysEmbeddings

    # Generate different user hashes
    user_hashes = rng.integers(1, 100000, size=(batch_size, hash_config.num_user_hashes)).astype(np.int32)
    history_post_hashes = rng.integers(1, 100000, size=(batch_size, history_len, hash_config.num_item_hashes)).astype(np.int32)
    history_author_hashes = rng.integers(1, 100000, size=(batch_size, history_len, hash_config.num_author_hashes)).astype(np.int32)
    history_actions = rng.random(size=(batch_size, history_len, model_config.num_actions)).astype(np.float32)
    history_product_surface = rng.integers(0, 16, size=(batch_size, history_len)).astype(np.int32)
    candidate_post_hashes = rng.integers(1, 100000, size=(batch_size, num_candidates, hash_config.num_item_hashes)).astype(np.int32)
    candidate_author_hashes = rng.integers(1, 100000, size=(batch_size, num_candidates, hash_config.num_author_hashes)).astype(np.int32)
    candidate_product_surface = rng.integers(0, 16, size=(batch_size, num_candidates)).astype(np.int32)

    batch = RecsysBatch(
        user_hashes=user_hashes,
        history_post_hashes=history_post_hashes,
        history_author_hashes=history_author_hashes,
        history_actions=history_actions,
        history_product_surface=history_product_surface,
        candidate_post_hashes=candidate_post_hashes,
        candidate_author_hashes=candidate_author_hashes,
        candidate_product_surface=candidate_product_surface,
    )

    embeddings = RecsysEmbeddings(
        user_embeddings=rng.normal(size=(batch_size, hash_config.num_user_hashes, model_config.emb_size)).astype(np.float32),
        history_post_embeddings=rng.normal(size=(batch_size, history_len, hash_config.num_item_hashes, model_config.emb_size)).astype(np.float32),
        candidate_post_embeddings=rng.normal(size=(batch_size, num_candidates, hash_config.num_item_hashes, model_config.emb_size)).astype(np.float32),
        history_author_embeddings=rng.normal(size=(batch_size, history_len, hash_config.num_author_hashes, model_config.emb_size)).astype(np.float32),
        candidate_author_embeddings=rng.normal(size=(batch_size, num_candidates, hash_config.num_author_hashes, model_config.emb_size)).astype(np.float32),
    )

    return batch, embeddings


# =============================================================================
# GATE CRITERION 1: KV cache creates successfully
# =============================================================================


class TestKVCacheCreation:
    """Tests for KV cache creation."""

    def test_cached_runner_initializes(self, base_runner):
        """Verify CachedPhoenixRunner can be created."""
        cached_runner = CachedPhoenixRunner(base_runner)
        assert cached_runner.cache is None
        assert cached_runner.stats.hits == 0
        assert cached_runner.stats.misses == 0

    def test_kv_cache_created_after_first_call(self, base_runner, example_batch):
        """Verify KV cache is created after first ranking call."""
        batch, embeddings = example_batch
        cached_runner = CachedPhoenixRunner(base_runner)

        _ = cached_runner.rank(batch, embeddings)

        assert cached_runner.cache is not None
        assert isinstance(cached_runner.cache, KVCache)
        assert cached_runner.stats.misses == 1
        assert cached_runner.stats.hits == 0

    def test_user_hash_computation(self, example_batch):
        """Verify user hash is computed deterministically."""
        batch, _ = example_batch

        hash1 = compute_user_hash(batch)
        hash2 = compute_user_hash(batch)

        assert hash1 == hash2
        assert isinstance(hash1, int)


# =============================================================================
# GATE CRITERION 2: Cached output matches baseline (rtol=1e-5)
# =============================================================================


class TestOutputCorrectness:
    """Tests for output correctness."""

    def test_cached_output_matches_baseline(self, base_runner, example_batch):
        """Verify cached scoring matches non-cached scoring."""
        batch, embeddings = example_batch

        baseline_output = base_runner.rank(batch, embeddings)

        cached_runner = CachedPhoenixRunner(base_runner)
        cached_output = cached_runner.rank(batch, embeddings, use_cache=True)

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(cached_output.scores),
            rtol=1e-5,
            err_msg="Cached scores should match baseline",
        )

    def test_cache_hit_output_matches_baseline(self, base_runner, example_batch):
        """Verify cache hit produces correct output."""
        batch, embeddings = example_batch

        baseline_output = base_runner.rank(batch, embeddings)

        cached_runner = CachedPhoenixRunner(base_runner)
        _ = cached_runner.rank(batch, embeddings)  # Miss
        hit_output = cached_runner.rank(batch, embeddings)  # Hit

        assert cached_runner.stats.hits == 1

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(hit_output.scores),
            rtol=1e-5,
        )

    def test_bypass_cache_matches_baseline(self, base_runner, example_batch):
        """Verify use_cache=False produces correct output."""
        batch, embeddings = example_batch

        baseline_output = base_runner.rank(batch, embeddings)

        cached_runner = CachedPhoenixRunner(base_runner)
        no_cache_output = cached_runner.rank(batch, embeddings, use_cache=False)

        np.testing.assert_allclose(
            np.array(baseline_output.scores),
            np.array(no_cache_output.scores),
            rtol=1e-5,
        )


# =============================================================================
# GATE CRITERION 3: Cache hit performance
# =============================================================================


class TestCachePerformance:
    """Tests for cache performance."""

    def test_cache_hit_not_slower_than_miss(self, base_runner, example_batch):
        """Verify cache hit is not slower than cache miss."""
        batch, embeddings = example_batch
        num_runs = 10

        cached_runner = CachedJITRunner(base_runner)

        # Warmup
        for _ in range(3):
            _ = cached_runner.rank(batch, embeddings)

        # Measure cache miss times
        miss_times = []
        for _ in range(num_runs):
            cached_runner.clear_cache()
            start = time.perf_counter()
            output = cached_runner.rank(batch, embeddings)
            jax.block_until_ready(output)
            miss_times.append((time.perf_counter() - start) * 1000)

        # Measure cache hit times
        _ = cached_runner.rank(batch, embeddings)  # Populate cache
        hit_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = cached_runner.rank(batch, embeddings)
            jax.block_until_ready(output)
            hit_times.append((time.perf_counter() - start) * 1000)

        miss_p50 = np.percentile(miss_times, 50)
        hit_p50 = np.percentile(hit_times, 50)

        print("\nCache Performance:")
        print(f"  Miss p50: {miss_p50:.2f} ms")
        print(f"  Hit p50: {hit_p50:.2f} ms")
        print(f"  Ratio: {hit_p50 / miss_p50:.2f}x")

        # Cache hit should not be significantly slower
        assert hit_p50 <= miss_p50 * 1.2, "Cache hit should not be slower than miss"

    def test_stats_tracking(self, base_runner, example_batch):
        """Verify cache statistics are tracked correctly."""
        batch, embeddings = example_batch

        cached_runner = CachedPhoenixRunner(base_runner)

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.misses == 1
        assert cached_runner.stats.hits == 0

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.misses == 1
        assert cached_runner.stats.hits == 1

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.misses == 1
        assert cached_runner.stats.hits == 2


# =============================================================================
# GATE CRITERION 4: Cache invalidates correctly on user change
# =============================================================================


class TestCacheInvalidation:
    """Tests for cache invalidation."""

    def test_cache_invalidation_on_user_change(self, base_runner, model_config, example_batch):
        """Verify cache invalidates when user changes."""
        batch1, emb1 = example_batch
        batch2, emb2 = create_batch_with_different_user(model_config)

        cached_runner = CachedPhoenixRunner(base_runner)

        # First user
        _ = cached_runner.rank(batch1, emb1)
        old_hash = cached_runner.cache.user_hash
        assert cached_runner.stats.misses == 1

        # Different user - should be cache miss
        _ = cached_runner.rank(batch2, emb2)
        new_hash = cached_runner.cache.user_hash

        assert new_hash != old_hash, "Cache should have new user hash"
        assert cached_runner.stats.misses == 2, "Should be a cache miss"
        assert cached_runner.stats.hits == 0

    def test_same_user_is_cache_hit(self, base_runner, example_batch):
        """Verify same user context results in cache hit."""
        batch, embeddings = example_batch

        cached_runner = CachedPhoenixRunner(base_runner)

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.misses == 1

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.hits == 1

    def test_clear_cache(self, base_runner, example_batch):
        """Verify clear_cache works correctly."""
        batch, embeddings = example_batch

        cached_runner = CachedPhoenixRunner(base_runner)

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.cache is not None

        cached_runner.clear_cache()
        assert cached_runner.cache is None

        _ = cached_runner.rank(batch, embeddings)
        assert cached_runner.stats.misses == 2


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestCachedJITRunner:
    """Tests for the combined JIT + Cache runner."""

    def test_cached_jit_runner_initializes(self, base_runner):
        """Verify CachedJITRunner can be created."""
        runner = CachedJITRunner(base_runner)
        assert runner.stats.hits == 0
        assert runner.stats.misses == 0

    def test_cached_jit_output_matches_baseline(self, base_runner, example_batch):
        """Verify CachedJITRunner produces correct output."""
        batch, embeddings = example_batch

        baseline = base_runner.rank(batch, embeddings)
        cached_jit = CachedJITRunner(base_runner)
        output = cached_jit.rank(batch, embeddings)

        np.testing.assert_allclose(
            np.array(baseline.scores),
            np.array(output.scores),
            rtol=1e-5,
        )

    def test_cached_jit_tracks_stats(self, base_runner, example_batch):
        """Verify CachedJITRunner tracks cache stats."""
        batch, embeddings = example_batch

        cached_jit = CachedJITRunner(base_runner)

        _ = cached_jit.rank(batch, embeddings)
        assert cached_jit.stats.misses == 1

        _ = cached_jit.rank(batch, embeddings)
        assert cached_jit.stats.hits == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
