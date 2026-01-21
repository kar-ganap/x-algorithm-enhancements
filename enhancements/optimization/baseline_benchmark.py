"""F2 Phase 0: Baseline Benchmark for Phoenix Ranking Inference.

This script measures the baseline latency of Phoenix ranking inference
to establish go/no-go gates for KV-cache optimization.

Usage:
    python -m enhancements.optimization.baseline_benchmark

Metrics collected:
    - Latency per batch (ms)
    - Throughput (candidates/sec)
    - JAX compilation time vs runtime
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np

# Add phoenix to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))

from grok import TransformerConfig
from recsys_model import HashConfig, PhoenixModelConfig
from runners import ACTIONS, ModelRunner, RecsysInferenceRunner, create_example_batch


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    emb_size: int = 128
    history_seq_len: int = 32
    candidate_seq_len: int = 8
    warmup_iterations: int = 5
    benchmark_iterations: int = 20
    batch_sizes: tuple = (1, 2, 4, 8)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    batch_size: int
    num_candidates: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_candidates_per_sec: float
    compilation_time_ms: float


def create_model(config: BenchmarkConfig) -> RecsysInferenceRunner:
    """Create and initialize Phoenix ranking model."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )

    recsys_model = PhoenixModelConfig(
        emb_size=config.emb_size,
        num_actions=len(ACTIONS),
        history_seq_len=config.history_seq_len,
        candidate_seq_len=config.candidate_seq_len,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=config.emb_size,
            widening_factor=2,
            key_size=64,
            num_q_heads=2,
            num_kv_heads=2,
            num_layers=2,
            attn_output_multiplier=0.125,
        ),
    )

    inference_runner = RecsysInferenceRunner(
        runner=ModelRunner(
            model=recsys_model,
            bs_per_device=0.125,
        ),
        name="baseline_benchmark",
    )

    return inference_runner


def run_benchmark(
    runner: RecsysInferenceRunner,
    config: BenchmarkConfig,
    batch_size: int,
) -> BenchmarkResult:
    """Run benchmark for a specific batch size."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )

    # Create batch
    batch, embeddings = create_example_batch(
        batch_size=batch_size,
        emb_size=config.emb_size,
        history_len=config.history_seq_len,
        num_candidates=config.candidate_seq_len,
        num_actions=len(ACTIONS),
        num_user_hashes=hash_config.num_user_hashes,
        num_item_hashes=hash_config.num_item_hashes,
        num_author_hashes=hash_config.num_author_hashes,
        product_surface_vocab_size=16,
    )

    # Measure compilation time (first run)
    jax.clear_caches()
    compile_start = time.perf_counter()
    _ = runner.rank(batch, embeddings)
    jax.block_until_ready(_)
    compilation_time_ms = (time.perf_counter() - compile_start) * 1000

    # Warmup runs (already compiled, so these are fast)
    for _ in range(config.warmup_iterations):
        _ = runner.rank(batch, embeddings)
        jax.block_until_ready(_)

    # Benchmark runs
    latencies = []
    for _ in range(config.benchmark_iterations):
        start = time.perf_counter()
        output = runner.rank(batch, embeddings)
        jax.block_until_ready(output)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    total_candidates = batch_size * config.candidate_seq_len

    return BenchmarkResult(
        batch_size=batch_size,
        num_candidates=config.candidate_seq_len,
        mean_latency_ms=float(np.mean(latencies)),
        std_latency_ms=float(np.std(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        throughput_candidates_per_sec=float(total_candidates / (np.mean(latencies) / 1000)),
        compilation_time_ms=compilation_time_ms,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("PHOENIX BASELINE BENCHMARK RESULTS")
    print("=" * 80)

    print(f"\n{'Batch':<8} {'Candidates':<12} {'Mean (ms)':<12} {'Std (ms)':<10} "
          f"{'Min (ms)':<10} {'Max (ms)':<10} {'Throughput':<15}")
    print("-" * 80)

    for r in results:
        print(f"{r.batch_size:<8} {r.num_candidates:<12} {r.mean_latency_ms:<12.2f} "
              f"{r.std_latency_ms:<10.2f} {r.min_latency_ms:<10.2f} {r.max_latency_ms:<10.2f} "
              f"{r.throughput_candidates_per_sec:<15.1f}")

    print("\n" + "-" * 80)
    print("Compilation times:")
    for r in results:
        print(f"  Batch size {r.batch_size}: {r.compilation_time_ms:.1f} ms")

    print("\n" + "=" * 80)


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Save results to JSON file."""
    data = {
        "benchmark": "phoenix_baseline",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "jax_devices": str(jax.devices()),
        "results": [
            {
                "batch_size": r.batch_size,
                "num_candidates": r.num_candidates,
                "mean_latency_ms": r.mean_latency_ms,
                "std_latency_ms": r.std_latency_ms,
                "min_latency_ms": r.min_latency_ms,
                "max_latency_ms": r.max_latency_ms,
                "throughput_candidates_per_sec": r.throughput_candidates_per_sec,
                "compilation_time_ms": r.compilation_time_ms,
            }
            for r in results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    """Run baseline benchmark."""
    print("F2 Phase 0: Phoenix Baseline Benchmark")
    print(f"JAX devices: {jax.devices()}")

    config = BenchmarkConfig()

    print("\nInitializing Phoenix model...")
    runner = create_model(config)
    runner.initialize()
    print("Model initialized!")

    results = []
    for batch_size in config.batch_sizes:
        print(f"\nBenchmarking batch_size={batch_size}...")
        result = run_benchmark(runner, config, batch_size)
        results.append(result)
        print(f"  Mean latency: {result.mean_latency_ms:.2f} ms")
        print(f"  Throughput: {result.throughput_candidates_per_sec:.1f} candidates/sec")

    print_results(results)

    # Save results
    output_path = Path(__file__).parent.parent.parent / "results" / "f2_baseline.json"
    save_results(results, output_path)

    # Go/No-Go gate check
    print("\n" + "=" * 80)
    print("GO/NO-GO GATE: Phase 0 Complete")
    print("=" * 80)
    print("✓ Baseline metrics collected")
    print("✓ Model runs successfully on available hardware")
    print(f"✓ Baseline latency for batch_size=1: {results[0].mean_latency_ms:.2f} ms")
    print("\nProceed to Phase 1: KV-Cache Implementation")
    print("=" * 80)


if __name__ == "__main__":
    main()
