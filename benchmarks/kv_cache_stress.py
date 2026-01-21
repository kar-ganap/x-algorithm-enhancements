"""Stress Tests for KV-Cache Implementation.

Tests:
1. Batch size scaling - speedup with larger batches
2. Memory usage - cache memory consumption
3. Sequential hits - many candidate batches for same user
4. Numerical stability - output consistency over iterations
5. Candidate count impact - varying number of candidates
6. Throughput - candidates scored per second
"""

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner


def create_config(
    num_layers: int = 12,
    emb_size: int = 512,
    history_len: int = 128,
    num_candidates: int = 8,
) -> PhoenixModelConfig:
    """Create model config with given parameters."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=emb_size,
        num_actions=len(ACTIONS),
        history_seq_len=history_len,
        candidate_seq_len=num_candidates,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=emb_size,
            widening_factor=2,
            key_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            num_layers=num_layers,
            attn_output_multiplier=0.125,
        ),
    )


def create_batch_for_config(
    config: PhoenixModelConfig,
    batch_size: int = 1,
    seed: int = 42,
) -> Tuple[RecsysBatch, RecsysEmbeddings]:
    """Create batch with specific batch size."""
    return create_example_batch(
        batch_size=batch_size,
        emb_size=config.emb_size,
        history_len=config.history_seq_len,
        num_candidates=config.candidate_seq_len,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )


def get_memory_usage_mb() -> float:
    """Get current JAX memory usage in MB."""
    try:
        # Force garbage collection
        gc.collect()
        # Get JAX memory stats
        backend = jax.default_backend()
        if backend == 'cpu':
            # CPU doesn't have easy memory tracking
            return 0.0
        devices = jax.devices()
        if devices:
            # Try to get memory info
            try:
                stats = devices[0].memory_stats()
                if stats:
                    return stats.get('bytes_in_use', 0) / 1024 / 1024
            except:
                pass
        return 0.0
    except:
        return 0.0


# =============================================================================
# TEST 1: Batch Size Scaling
# =============================================================================

def test_batch_size_scaling():
    """Test how speedup scales with batch size."""
    print("\n" + "="*80)
    print("TEST 1: Batch Size Scaling")
    print("="*80)
    print("Does KV-cache speedup hold with larger batch sizes?")

    config = create_config(num_layers=12, emb_size=512, history_len=256, num_candidates=8)

    batch_sizes = [1, 2, 4, 8]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Batch size: {bs} ---")

        # Create new runner for each batch size (different static shapes)
        bs_config = create_config(
            num_layers=12, emb_size=512,
            history_len=256, num_candidates=8
        )
        runner = FullKVCachedRunner(bs_config)
        runner.initialize()

        batch, emb = create_batch_for_config(bs_config, batch_size=bs)

        # Warmup
        for _ in range(3):
            runner.clear_cache()
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)

        # Measure miss
        miss_times = []
        for _ in range(5):
            runner.clear_cache()
            start = time.perf_counter()
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)
            miss_times.append((time.perf_counter() - start) * 1000)

        # Measure hit
        runner.clear_cache()
        _ = runner.rank(batch, emb, use_cache=True)
        hit_times = []
        for _ in range(5):
            start = time.perf_counter()
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)
            hit_times.append((time.perf_counter() - start) * 1000)

        miss_p50 = np.median(miss_times)
        hit_p50 = np.median(hit_times)
        speedup = miss_p50 / hit_p50

        # Per-sample metrics
        miss_per_sample = miss_p50 / bs
        hit_per_sample = hit_p50 / bs

        print(f"  Miss: {miss_p50:.1f}ms ({miss_per_sample:.1f}ms/sample)")
        print(f"  Hit:  {hit_p50:.1f}ms ({hit_per_sample:.1f}ms/sample)")
        print(f"  Speedup: {speedup:.2f}x")

        results.append((bs, miss_p50, hit_p50, speedup))

    print("\n--- Summary ---")
    print(f"{'Batch':>6} {'Miss(ms)':>10} {'Hit(ms)':>10} {'Speedup':>8}")
    for bs, miss, hit, spd in results:
        print(f"{bs:>6} {miss:>10.1f} {hit:>10.1f} {spd:>7.2f}x")


# =============================================================================
# TEST 2: Memory Usage
# =============================================================================

def test_memory_usage():
    """Test memory consumption of KV cache."""
    print("\n" + "="*80)
    print("TEST 2: Memory Usage Analysis")
    print("="*80)
    print("How much memory does the KV cache consume?")

    configs = [
        ("Small (4L, 256)", 4, 256, 64, 8),
        ("Medium (12L, 512)", 12, 512, 128, 8),
        ("Large (16L, 768)", 16, 768, 256, 8),
        ("XLarge (24L, 1024)", 24, 1024, 512, 16),
    ]

    print(f"\n{'Config':<25} {'Layers':>7} {'Emb':>6} {'Context':>8} {'Cache Size':>12}")
    print("-"*70)

    for name, layers, emb, history, candidates in configs:
        config = create_config(
            num_layers=layers,
            emb_size=emb,
            history_len=history,
            num_candidates=candidates,
        )

        # Calculate theoretical cache size
        # Cache stores K,V for each layer: [batch, num_kv_heads, context_len, key_size]
        num_kv_heads = 4
        key_size = 64
        context_len = 1 + history
        batch_size = 1

        # K and V each: batch * heads * context * key_size * 2 bytes (bfloat16)
        kv_size_per_layer = batch_size * num_kv_heads * context_len * key_size * 2 * 2  # *2 for K and V
        total_cache_bytes = kv_size_per_layer * layers
        cache_mb = total_cache_bytes / 1024 / 1024

        print(f"{name:<25} {layers:>7} {emb:>6} {context_len:>8} {cache_mb:>10.2f} MB")

    print("\nNote: Cache size scales with layers × context_len × num_kv_heads × key_size")
    print("For batch_size > 1, multiply by batch_size")


# =============================================================================
# TEST 3: Sequential Cache Hits (Many Candidate Batches)
# =============================================================================

def test_sequential_hits():
    """Test scoring many candidate batches for the same user."""
    print("\n" + "="*80)
    print("TEST 3: Sequential Cache Hits")
    print("="*80)
    print("Scoring many candidate batches for the same user (simulating pagination)")

    config = create_config(num_layers=16, emb_size=768, history_len=256, num_candidates=8)
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, emb = create_batch_for_config(config)

    # Warmup
    for _ in range(3):
        runner.clear_cache()
        out = runner.rank(batch, emb)
        jax.block_until_ready(out)

    num_batches = 50

    # First call is cache miss
    runner.clear_cache()
    start = time.perf_counter()
    out = runner.rank(batch, emb, use_cache=True)
    jax.block_until_ready(out)
    first_call_ms = (time.perf_counter() - start) * 1000

    # Subsequent calls are cache hits
    hit_times = []
    for i in range(num_batches):
        # Create new candidates (same user context)
        rng = np.random.default_rng(1000 + i)
        new_cand_emb = rng.normal(size=emb.candidate_post_embeddings.shape).astype(np.float32)
        new_emb = RecsysEmbeddings(
            user_embeddings=emb.user_embeddings,
            history_post_embeddings=emb.history_post_embeddings,
            candidate_post_embeddings=new_cand_emb,
            history_author_embeddings=emb.history_author_embeddings,
            candidate_author_embeddings=emb.candidate_author_embeddings,
        )

        start = time.perf_counter()
        out = runner.rank(batch, new_emb, use_cache=True)
        jax.block_until_ready(out)
        hit_times.append((time.perf_counter() - start) * 1000)

    hit_p50 = np.median(hit_times)
    hit_p99 = np.percentile(hit_times, 99)
    total_candidates = num_batches * config.candidate_seq_len
    total_time = sum(hit_times)
    throughput = total_candidates / (total_time / 1000)

    print(f"\nFirst call (cache miss): {first_call_ms:.1f} ms")
    print(f"Cache hits ({num_batches} batches):")
    print(f"  p50: {hit_p50:.1f} ms")
    print(f"  p99: {hit_p99:.1f} ms")
    print(f"  Total time: {total_time:.1f} ms")
    print(f"\nThroughput: {throughput:.0f} candidates/second")
    print(f"Speedup vs all misses: {(first_call_ms * num_batches) / total_time:.1f}x")


# =============================================================================
# TEST 4: Numerical Stability
# =============================================================================

def test_numerical_stability():
    """Test that cached output remains numerically stable."""
    print("\n" + "="*80)
    print("TEST 4: Numerical Stability")
    print("="*80)
    print("Does the output remain stable over many cache hits?")

    config = create_config(num_layers=12, emb_size=512, history_len=128, num_candidates=8)
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, emb = create_batch_for_config(config)

    # Get baseline (no cache)
    runner.clear_cache()
    baseline = runner.rank(batch, emb, use_cache=False)
    baseline_scores = np.array(baseline.scores, dtype=np.float32)

    # Get cached output
    runner.clear_cache()
    _ = runner.rank(batch, emb, use_cache=True)  # Miss

    num_iterations = 100
    max_diffs = []
    mean_diffs = []

    for _ in range(num_iterations):
        out = runner.rank(batch, emb, use_cache=True)  # Hit
        scores = np.array(out.scores, dtype=np.float32)
        diff = np.abs(scores - baseline_scores)
        max_diffs.append(float(np.max(diff)))
        mean_diffs.append(float(np.mean(diff)))

    max_diff_val = float(np.max(max_diffs))
    mean_diff_val = float(np.mean(mean_diffs))
    std_diff_val = float(np.std(max_diffs))

    print(f"\nComparing {num_iterations} cache hits to baseline (no cache):")
    print(f"  Max absolute diff:  {max_diff_val:.2e}")
    print(f"  Mean absolute diff: {mean_diff_val:.2e}")
    print(f"  Diff std dev:       {std_diff_val:.2e}")

    # Check stability across iterations
    first_hit = runner.rank(batch, emb, use_cache=True)
    first_scores = np.array(first_hit.scores, dtype=np.float32)

    runner.clear_cache()
    _ = runner.rank(batch, emb, use_cache=True)  # New miss
    second_hit = runner.rank(batch, emb, use_cache=True)
    second_scores = np.array(second_hit.scores, dtype=np.float32)

    cross_diff = float(np.max(np.abs(first_scores - second_scores)))
    print(f"\nCross-cache consistency (different cache instances):")
    print(f"  Max diff: {cross_diff:.2e}")

    if max_diff_val < 1e-3 and cross_diff < 1e-3:
        print("\n✓ PASS: Output is numerically stable")
    else:
        print("\n✗ WARNING: Numerical differences detected")


# =============================================================================
# TEST 5: Candidate Count Impact
# =============================================================================

def test_candidate_count_impact():
    """Test speedup with varying number of candidates."""
    print("\n" + "="*80)
    print("TEST 5: Candidate Count Impact")
    print("="*80)
    print("How does speedup change with number of candidates?")

    candidate_counts = [4, 8, 16, 32, 64]
    results = []

    for num_cand in candidate_counts:
        print(f"\n--- {num_cand} candidates ---")

        config = create_config(
            num_layers=12,
            emb_size=512,
            history_len=256,  # Fixed context
            num_candidates=num_cand,
        )
        runner = FullKVCachedRunner(config)
        runner.initialize()

        batch, emb = create_batch_for_config(config)

        # Warmup
        for _ in range(3):
            runner.clear_cache()
            out = runner.rank(batch, emb)
            jax.block_until_ready(out)

        # Measure
        miss_times = []
        for _ in range(5):
            runner.clear_cache()
            start = time.perf_counter()
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)
            miss_times.append((time.perf_counter() - start) * 1000)

        runner.clear_cache()
        _ = runner.rank(batch, emb, use_cache=True)
        hit_times = []
        for _ in range(5):
            start = time.perf_counter()
            out = runner.rank(batch, emb, use_cache=True)
            jax.block_until_ready(out)
            hit_times.append((time.perf_counter() - start) * 1000)

        miss_p50 = np.median(miss_times)
        hit_p50 = np.median(hit_times)
        speedup = miss_p50 / hit_p50
        theoretical = (257 + num_cand) / num_cand

        print(f"  Miss: {miss_p50:.1f}ms, Hit: {hit_p50:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x (theoretical max: {theoretical:.1f}x)")

        results.append((num_cand, miss_p50, hit_p50, speedup, theoretical))

    print("\n--- Summary ---")
    print(f"{'Cand':>6} {'Miss(ms)':>10} {'Hit(ms)':>10} {'Speedup':>8} {'Theory':>8}")
    for cand, miss, hit, spd, theory in results:
        print(f"{cand:>6} {miss:>10.1f} {hit:>10.1f} {spd:>7.2f}x {theory:>7.1f}x")

    print("\nNote: Fewer candidates = higher theoretical speedup (more context cached relative to work)")


# =============================================================================
# TEST 6: Throughput Under Load
# =============================================================================

def test_throughput():
    """Measure sustained throughput."""
    print("\n" + "="*80)
    print("TEST 6: Throughput Under Load")
    print("="*80)
    print("Measuring sustained scoring throughput")

    config = create_config(num_layers=16, emb_size=768, history_len=256, num_candidates=16)
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, emb = create_batch_for_config(config)

    # Warmup
    for _ in range(5):
        runner.clear_cache()
        out = runner.rank(batch, emb)
        jax.block_until_ready(out)

    duration_seconds = 5

    # Without cache
    print(f"\nWithout cache (full forward each time):")
    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        runner.clear_cache()
        out = runner.rank(batch, emb, use_cache=True)
        jax.block_until_ready(out)
        count += 1
    elapsed = time.perf_counter() - start
    no_cache_throughput = (count * config.candidate_seq_len) / elapsed
    print(f"  Batches: {count}, Candidates: {count * config.candidate_seq_len}")
    print(f"  Throughput: {no_cache_throughput:.0f} candidates/sec")

    # With cache (same user)
    print(f"\nWith cache (same user, cache hits):")
    runner.clear_cache()
    _ = runner.rank(batch, emb, use_cache=True)  # Populate cache

    count = 0
    start = time.perf_counter()
    while time.perf_counter() - start < duration_seconds:
        out = runner.rank(batch, emb, use_cache=True)
        jax.block_until_ready(out)
        count += 1
    elapsed = time.perf_counter() - start
    cache_throughput = (count * config.candidate_seq_len) / elapsed
    print(f"  Batches: {count}, Candidates: {count * config.candidate_seq_len}")
    print(f"  Throughput: {cache_throughput:.0f} candidates/sec")

    print(f"\nThroughput improvement: {cache_throughput / no_cache_throughput:.2f}x")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("KV-CACHE STRESS TESTS")
    print("="*80)

    tests = [
        ("Batch Size Scaling", test_batch_size_scaling),
        ("Memory Usage", test_memory_usage),
        ("Sequential Hits", test_sequential_hits),
        ("Numerical Stability", test_numerical_stability),
        ("Candidate Count Impact", test_candidate_count_impact),
        ("Throughput", test_throughput),
    ]

    print(f"\nRunning {len(tests)} stress tests...")

    for name, test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n*** ERROR in {name}: {e} ***")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print("STRESS TESTS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
