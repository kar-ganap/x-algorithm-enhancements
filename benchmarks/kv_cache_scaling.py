"""Benchmark: KV-Cache Speedup Scaling with Model Size.

This benchmark tests how KV-cache speedup scales with:
1. Number of transformer layers
2. Embedding/model size
3. Context length (user + history)

Expected: Speedup increases with model size and context length.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import jax
import numpy as np

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    name: str
    num_layers: int
    emb_size: int
    key_size: int
    num_q_heads: int
    num_kv_heads: int
    history_len: int
    num_candidates: int


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    miss_p50_ms: float
    hit_p50_ms: float
    speedup: float
    context_len: int  # user (1) + history_len


def create_model_config(bench_config: BenchmarkConfig) -> PhoenixModelConfig:
    """Create Phoenix model config from benchmark config."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=bench_config.emb_size,
        num_actions=len(ACTIONS),
        history_seq_len=bench_config.history_len,
        candidate_seq_len=bench_config.num_candidates,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=bench_config.emb_size,
            widening_factor=2,
            key_size=bench_config.key_size,
            num_q_heads=bench_config.num_q_heads,
            num_kv_heads=bench_config.num_kv_heads,
            num_layers=bench_config.num_layers,
            attn_output_multiplier=0.125,
        ),
    )


def run_benchmark(bench_config: BenchmarkConfig, num_warmup: int = 3, num_runs: int = 10) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {bench_config.name}")
    print(f"  Layers: {bench_config.num_layers}, Emb: {bench_config.emb_size}, "
          f"History: {bench_config.history_len}, Candidates: {bench_config.num_candidates}")
    print(f"{'='*60}")

    # Create model and runner
    model_config = create_model_config(bench_config)
    runner = FullKVCachedRunner(model_config)

    print("Initializing model...")
    init_start = time.perf_counter()
    runner.initialize()
    init_time = (time.perf_counter() - init_start) * 1000
    print(f"  Init time: {init_time:.1f} ms")

    # Create example batch
    batch, embeddings = create_example_batch(
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

    # Warmup
    print(f"Warming up ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        runner.clear_cache()
        output = runner.rank(batch, embeddings, use_cache=True)
        jax.block_until_ready(output)
        output = runner.rank(batch, embeddings, use_cache=True)
        jax.block_until_ready(output)

    # Measure cache miss times
    print(f"Measuring cache miss ({num_runs} runs)...")
    miss_times = []
    for _ in range(num_runs):
        runner.clear_cache()
        start = time.perf_counter()
        output = runner.rank(batch, embeddings, use_cache=True)
        jax.block_until_ready(output)
        miss_times.append((time.perf_counter() - start) * 1000)

    # Measure cache hit times
    print(f"Measuring cache hit ({num_runs} runs)...")
    runner.clear_cache()
    _ = runner.rank(batch, embeddings, use_cache=True)  # Populate cache

    hit_times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        output = runner.rank(batch, embeddings, use_cache=True)
        jax.block_until_ready(output)
        hit_times.append((time.perf_counter() - start) * 1000)

    miss_p50 = np.percentile(miss_times, 50)
    hit_p50 = np.percentile(hit_times, 50)
    speedup = miss_p50 / hit_p50

    context_len = 1 + bench_config.history_len

    print(f"\nResults:")
    print(f"  Context length: {context_len} tokens")
    print(f"  Cache miss p50: {miss_p50:.2f} ms")
    print(f"  Cache hit p50:  {hit_p50:.2f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    return BenchmarkResult(
        config=bench_config,
        miss_p50_ms=miss_p50,
        hit_p50_ms=hit_p50,
        speedup=speedup,
        context_len=context_len,
    )


def print_summary(results: List[BenchmarkResult]):
    """Print summary table of all results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY: KV-Cache Speedup Scaling")
    print("="*80)

    # Header
    print(f"{'Config':<20} {'Layers':>7} {'Emb':>6} {'Context':>8} {'Miss(ms)':>10} {'Hit(ms)':>10} {'Speedup':>8}")
    print("-"*80)

    for r in results:
        print(f"{r.config.name:<20} {r.config.num_layers:>7} {r.config.emb_size:>6} "
              f"{r.context_len:>8} {r.miss_p50_ms:>10.2f} {r.hit_p50_ms:>10.2f} {r.speedup:>7.2f}x")

    print("-"*80)

    # Analysis
    print("\nAnalysis:")
    if len(results) >= 2:
        first_speedup = results[0].speedup
        last_speedup = results[-1].speedup
        improvement = last_speedup / first_speedup
        print(f"  Speedup improved {improvement:.1f}x from smallest to largest config")

    # Expected behavior
    print("\nExpected behavior:")
    print("  - Speedup should increase with more layers (more computation saved)")
    print("  - Speedup should increase with longer context (more K/V cached)")
    print("  - Speedup should increase with larger embedding size (more compute per token)")


def main():
    """Run the full benchmark suite."""
    # Define benchmark configurations - progressively larger
    configs = [
        # Small (test config)
        BenchmarkConfig(
            name="Small",
            num_layers=4,
            emb_size=128,
            key_size=64,
            num_q_heads=4,
            num_kv_heads=2,
            history_len=32,
            num_candidates=8,
        ),
        # Medium
        BenchmarkConfig(
            name="Medium",
            num_layers=8,
            emb_size=256,
            key_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            history_len=64,
            num_candidates=8,
        ),
        # Large
        BenchmarkConfig(
            name="Large",
            num_layers=12,
            emb_size=512,
            key_size=64,
            num_q_heads=8,
            num_kv_heads=4,
            history_len=128,
            num_candidates=8,
        ),
        # XLarge (closer to production)
        BenchmarkConfig(
            name="XLarge",
            num_layers=16,
            emb_size=768,
            key_size=96,
            num_q_heads=12,
            num_kv_heads=4,
            history_len=256,
            num_candidates=16,
        ),
    ]

    print("="*80)
    print("KV-Cache Scaling Benchmark")
    print("="*80)
    print(f"Testing {len(configs)} configurations to verify speedup scaling")
    print("Note: Larger models show greater speedup from KV-caching")

    results = []
    for config in configs:
        try:
            result = run_benchmark(config)
            results.append(result)
        except Exception as e:
            print(f"\nError running {config.name}: {e}")
            print("Skipping to next configuration...")
            continue

    if results:
        print_summary(results)
    else:
        print("\nNo benchmarks completed successfully.")


if __name__ == "__main__":
    main()
