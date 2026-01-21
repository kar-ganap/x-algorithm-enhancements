"""Benchmark: Finding the Limits of KV-Cache Speedup.

This benchmark pushes configurations to find maximum achievable speedup.

Theoretical maximum speedup ≈ (context_len + candidates) / candidates
- If context=512, candidates=8: theoretical max = 520/8 = 65x
- In practice, limited by: memory bandwidth, fixed overhead, JIT dispatch

We test:
1. Very large models (24 layers, 1024 emb)
2. Extreme context-to-candidate ratios
3. Multiple candidates counts
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

    @property
    def context_len(self) -> int:
        return 1 + self.history_len

    @property
    def theoretical_max_speedup(self) -> float:
        """Theoretical maximum speedup if all context compute is saved."""
        return (self.context_len + self.num_candidates) / self.num_candidates


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config: BenchmarkConfig
    miss_p50_ms: float
    hit_p50_ms: float
    speedup: float
    theoretical_max: float
    efficiency: float  # actual / theoretical


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


def run_benchmark(bench_config: BenchmarkConfig, num_warmup: int = 3, num_runs: int = 10) -> Optional[BenchmarkResult]:
    """Run benchmark for a single configuration."""
    print(f"\n{'='*70}")
    print(f"Config: {bench_config.name}")
    print(f"  Layers={bench_config.num_layers}, Emb={bench_config.emb_size}, "
          f"Context={bench_config.context_len}, Candidates={bench_config.num_candidates}")
    print(f"  Theoretical max speedup: {bench_config.theoretical_max_speedup:.1f}x")
    print(f"{'='*70}")

    try:
        # Create model and runner
        model_config = create_model_config(bench_config)
        runner = FullKVCachedRunner(model_config)

        print("Initializing...", end=" ", flush=True)
        init_start = time.perf_counter()
        runner.initialize()
        init_time = (time.perf_counter() - init_start) * 1000
        print(f"done ({init_time:.0f}ms)")

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
        print(f"Warmup...", end=" ", flush=True)
        for _ in range(num_warmup):
            runner.clear_cache()
            output = runner.rank(batch, embeddings, use_cache=True)
            jax.block_until_ready(output)
            output = runner.rank(batch, embeddings, use_cache=True)
            jax.block_until_ready(output)
        print("done")

        # Measure cache miss
        print(f"Measuring miss...", end=" ", flush=True)
        miss_times = []
        for _ in range(num_runs):
            runner.clear_cache()
            start = time.perf_counter()
            output = runner.rank(batch, embeddings, use_cache=True)
            jax.block_until_ready(output)
            miss_times.append((time.perf_counter() - start) * 1000)
        print("done")

        # Measure cache hit
        print(f"Measuring hit...", end=" ", flush=True)
        runner.clear_cache()
        _ = runner.rank(batch, embeddings, use_cache=True)

        hit_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            output = runner.rank(batch, embeddings, use_cache=True)
            jax.block_until_ready(output)
            hit_times.append((time.perf_counter() - start) * 1000)
        print("done")

        miss_p50 = float(np.percentile(miss_times, 50))
        hit_p50 = float(np.percentile(hit_times, 50))
        speedup = miss_p50 / hit_p50
        theoretical = bench_config.theoretical_max_speedup
        efficiency = (speedup - 1) / (theoretical - 1) * 100 if theoretical > 1 else 0

        print(f"\n  Results: Miss={miss_p50:.1f}ms, Hit={hit_p50:.1f}ms")
        print(f"  Speedup: {speedup:.2f}x (theoretical max: {theoretical:.1f}x)")
        print(f"  Efficiency: {efficiency:.1f}% of theoretical gain")

        return BenchmarkResult(
            config=bench_config,
            miss_p50_ms=miss_p50,
            hit_p50_ms=hit_p50,
            speedup=speedup,
            theoretical_max=theoretical,
            efficiency=efficiency,
        )

    except Exception as e:
        print(f"\n  ERROR: {e}")
        return None


def print_summary(results: List[BenchmarkResult]):
    """Print summary table."""
    print("\n" + "="*90)
    print("BENCHMARK SUMMARY: KV-Cache Speedup Limits")
    print("="*90)

    print(f"{'Config':<25} {'L':>3} {'Emb':>5} {'Ctx':>5} {'Cand':>5} "
          f"{'Miss':>8} {'Hit':>8} {'Speed':>7} {'Theory':>7} {'Eff':>6}")
    print("-"*90)

    for r in results:
        c = r.config
        print(f"{c.name:<25} {c.num_layers:>3} {c.emb_size:>5} {c.context_len:>5} "
              f"{c.num_candidates:>5} {r.miss_p50_ms:>7.1f}ms {r.hit_p50_ms:>7.1f}ms "
              f"{r.speedup:>6.2f}x {r.theoretical_max:>6.1f}x {r.efficiency:>5.1f}%")

    print("-"*90)

    # Find best result
    best = max(results, key=lambda r: r.speedup)
    print(f"\nBest speedup: {best.speedup:.2f}x ({best.config.name})")
    print(f"  Context={best.config.context_len}, Candidates={best.config.num_candidates}")
    print(f"  Efficiency: {best.efficiency:.1f}% of theoretical maximum")


def main():
    print("="*90)
    print("KV-Cache Speedup Limits Benchmark")
    print("="*90)
    print("Testing extreme configurations to find maximum achievable speedup")
    print("Theoretical max = (context + candidates) / candidates")
    print()

    # Test configurations designed to maximize speedup
    configs = [
        # Baseline - moderate size
        BenchmarkConfig(
            name="Baseline",
            num_layers=12, emb_size=512, key_size=64,
            num_q_heads=8, num_kv_heads=4,
            history_len=128, num_candidates=8,
        ),

        # More layers
        BenchmarkConfig(
            name="Deep (24 layers)",
            num_layers=24, emb_size=512, key_size=64,
            num_q_heads=8, num_kv_heads=4,
            history_len=128, num_candidates=8,
        ),

        # Wider model
        BenchmarkConfig(
            name="Wide (1024 emb)",
            num_layers=12, emb_size=1024, key_size=128,
            num_q_heads=16, num_kv_heads=4,
            history_len=128, num_candidates=8,
        ),

        # Long context
        BenchmarkConfig(
            name="Long context (512)",
            num_layers=12, emb_size=512, key_size=64,
            num_q_heads=8, num_kv_heads=4,
            history_len=511, num_candidates=8,  # 512 total context
        ),

        # Very long context
        BenchmarkConfig(
            name="Very long (1024)",
            num_layers=12, emb_size=512, key_size=64,
            num_q_heads=8, num_kv_heads=4,
            history_len=1023, num_candidates=8,  # 1024 total context
        ),

        # Extreme: long context + few candidates
        BenchmarkConfig(
            name="Extreme ratio (512:4)",
            num_layers=16, emb_size=768, key_size=96,
            num_q_heads=12, num_kv_heads=4,
            history_len=511, num_candidates=4,  # 512:4 = 129x theoretical
        ),

        # Large model + long context
        BenchmarkConfig(
            name="Large + Long",
            num_layers=20, emb_size=768, key_size=96,
            num_q_heads=12, num_kv_heads=4,
            history_len=511, num_candidates=8,
        ),

        # Production-like
        BenchmarkConfig(
            name="Production-like",
            num_layers=24, emb_size=1024, key_size=128,
            num_q_heads=16, num_kv_heads=4,
            history_len=255, num_candidates=16,
        ),
    ]

    results = []
    for config in configs:
        result = run_benchmark(config)
        if result:
            results.append(result)

    if results:
        print_summary(results)

        # Additional analysis
        print("\n" + "="*90)
        print("ANALYSIS")
        print("="*90)

        # Speedup vs context length
        print("\nSpeedup vs Context Length:")
        sorted_by_ctx = sorted(results, key=lambda r: r.config.context_len)
        for r in sorted_by_ctx:
            bar = "█" * int(r.speedup * 5)
            print(f"  Context {r.config.context_len:>4}: {r.speedup:>5.2f}x {bar}")

        # Speedup vs model size (layers * emb)
        print("\nSpeedup vs Model Size (layers × emb):")
        sorted_by_size = sorted(results, key=lambda r: r.config.num_layers * r.config.emb_size)
        for r in sorted_by_size:
            size = r.config.num_layers * r.config.emb_size
            bar = "█" * int(r.speedup * 5)
            print(f"  Size {size:>6}: {r.speedup:>5.2f}x {bar}")


if __name__ == "__main__":
    main()
