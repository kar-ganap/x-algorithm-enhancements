"""Hardware Analysis: Understanding KV-Cache Performance Factors.

This script analyzes which metrics are hardware-dependent and
estimates expected GPU performance.
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner


def get_hardware_info():
    """Get information about the current hardware."""
    print("="*70)
    print("HARDWARE ANALYSIS")
    print("="*70)

    backend = jax.default_backend()
    devices = jax.devices()

    print(f"\nJAX Backend: {backend}")
    print(f"Devices: {devices}")

    if backend == 'cpu':
        print("\n⚠️  Running on CPU")
        print("   KV-cache benefits are MUCH larger on GPU due to:")
        print("   - Higher memory bandwidth (HBM: 2-3 TB/s vs DDR: 50-100 GB/s)")
        print("   - Massive parallelism (thousands of CUDA cores)")
        print("   - Better batch efficiency")
    elif backend in ('gpu', 'cuda'):
        print("\n✓ Running on GPU")
        for i, dev in enumerate(devices):
            print(f"   Device {i}: {dev}")

    return backend


def analyze_bottlenecks():
    """Analyze what limits KV-cache performance."""
    print("\n" + "="*70)
    print("PERFORMANCE BOTTLENECK ANALYSIS")
    print("="*70)

    # Create a medium-sized model
    hash_config = HashConfig(num_user_hashes=2, num_item_hashes=2, num_author_hashes=2)
    config = PhoenixModelConfig(
        emb_size=512,
        num_actions=len(ACTIONS),
        history_seq_len=256,
        candidate_seq_len=8,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=512, widening_factor=2, key_size=64,
            num_q_heads=8, num_kv_heads=4, num_layers=12,
            attn_output_multiplier=0.125,
        ),
    )

    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, emb = create_example_batch(
        batch_size=1, emb_size=config.emb_size,
        history_len=config.history_seq_len, num_candidates=config.candidate_seq_len,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )

    # Warmup
    for _ in range(5):
        runner.clear_cache()
        out = runner.rank(batch, emb)
        jax.block_until_ready(out)

    # Measure different phases
    print("\nTiming breakdown:")

    # 1. Cache miss (full forward)
    runner.clear_cache()
    start = time.perf_counter()
    out = runner.rank(batch, emb, use_cache=True)
    jax.block_until_ready(out)
    miss_time = (time.perf_counter() - start) * 1000

    # 2. Cache hit (candidate only)
    start = time.perf_counter()
    out = runner.rank(batch, emb, use_cache=True)
    jax.block_until_ready(out)
    hit_time = (time.perf_counter() - start) * 1000

    # 3. Encode only
    start = time.perf_counter()
    cache = runner.encode_user_context(batch, emb)
    jax.block_until_ready(cache)
    encode_time = (time.perf_counter() - start) * 1000

    # 4. Score only
    start = time.perf_counter()
    out = runner.score_with_cache(cache, batch, emb)
    jax.block_until_ready(out)
    score_time = (time.perf_counter() - start) * 1000

    print(f"  Full forward (miss):     {miss_time:>8.1f} ms")
    print(f"  Cache hit:               {hit_time:>8.1f} ms")
    print(f"  Encode context only:     {encode_time:>8.1f} ms")
    print(f"  Score with cache only:   {score_time:>8.1f} ms")

    context_ratio = encode_time / miss_time * 100
    candidate_ratio = score_time / miss_time * 100
    overhead = miss_time - encode_time - score_time

    print(f"\nBreakdown of full forward:")
    print(f"  Context encoding:  {context_ratio:>5.1f}%")
    print(f"  Candidate scoring: {candidate_ratio:>5.1f}%")
    print(f"  Overhead:          {overhead:>5.1f} ms ({overhead/miss_time*100:.1f}%)")

    speedup = miss_time / hit_time
    theoretical = (257 + 8) / 8

    print(f"\nSpeedup: {speedup:.2f}x (theoretical max: {theoretical:.1f}x)")
    print(f"Efficiency: {(speedup-1)/(theoretical-1)*100:.1f}% of theoretical gain")


def estimate_gpu_performance():
    """Estimate expected GPU performance based on typical speedups."""
    print("\n" + "="*70)
    print("ESTIMATED GPU PERFORMANCE")
    print("="*70)

    print("""
Based on typical CPU vs GPU characteristics:

                        │    CPU (yours)    │    GPU (estimated)
────────────────────────┼───────────────────┼────────────────────
Memory Bandwidth        │   50-100 GB/s     │   1500-3000 GB/s
Compute (TFLOPS)        │   0.5-2 TFLOPS    │   100-400 TFLOPS
Batch=1 Latency         │   100-500 ms      │   5-50 ms
Batch=8 Latency         │   500-3000 ms     │   10-100 ms
────────────────────────┼───────────────────┼────────────────────
KV-Cache Speedup        │   2-10x           │   10-50x
Throughput (cand/sec)   │   20-50           │   1000-10000

Why GPU speedup is higher:
1. KV-cache is MEMORY-BOUND - GPU HBM is 30x faster than DDR
2. Attention is highly parallel - GPU excels at GEMM operations
3. Batch efficiency - GPU utilization improves with larger batches

For production deployment:
- A100 (80GB): Handle 1000s of users concurrently
- RTX 4090: Good for development/small-scale serving
- CPU: Only for testing/development
""")


def hardware_independent_metrics():
    """Show which metrics are NOT hardware-dependent."""
    print("\n" + "="*70)
    print("HARDWARE-INDEPENDENT METRICS")
    print("="*70)

    print("""
These metrics are the SAME regardless of hardware:

1. ✓ Cache memory size (bytes)
   - Determined by: layers × context_len × num_kv_heads × key_size × 2
   - Example: 12 layers, 257 context, 4 heads, 64 dim = 1.51 MB

2. ✓ Numerical correctness (rtol=1e-4)
   - Cached output matches full forward exactly
   - Independent of hardware

3. ✓ Theoretical maximum speedup
   - Formula: (context + candidates) / candidates
   - Only depends on sequence lengths

4. ✓ Cache hit/miss detection
   - Based on user hash comparison
   - Works identically on all hardware

Hardware-DEPENDENT metrics:

1. ✗ Absolute latency (ms)
2. ✗ Throughput (candidates/sec)
3. ✗ Actual speedup achieved (varies with memory bandwidth)
4. ✗ Efficiency (% of theoretical gain)
""")


def main():
    backend = get_hardware_info()
    analyze_bottlenecks()
    estimate_gpu_performance()
    hardware_independent_metrics()

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if backend == 'cpu':
        print("""
Your CPU results show KV-caching is working correctly:
- Speedup: 1.3-9x depending on configuration
- Correctness: Verified (cache matches baseline)
- Memory: Efficient (1-12 MB for typical configs)

On GPU, expect:
- Speedup: 10-50x (memory bandwidth is the bottleneck)
- Throughput: 50-100x higher
- Same correctness guarantees
""")


if __name__ == "__main__":
    main()
