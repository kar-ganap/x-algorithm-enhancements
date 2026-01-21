"""Quantization comparative study framework.

Provides infrastructure for benchmarking different quantization configurations
and selecting the best one based on accuracy, memory, and latency criteria.
"""

import time
from dataclasses import dataclass
from typing import List, NamedTuple, Optional, Tuple

import jax
import numpy as np

from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import (
    ModelRunner,
    RankingOutput,
    RecsysInferenceRunner,
    create_example_batch,
)

from enhancements.optimization.quantization.config import (
    EXTENDED_STUDY_CONFIGS,
    QuantizationConfig,
    STUDY_CONFIGS,
)
from enhancements.optimization.quantization.quantized_runner import (
    QuantizedPhoenixRunner,
    create_quantized_runner,
)


class BenchmarkMetrics(NamedTuple):
    """Metrics for a single quantization configuration.

    Attributes:
        config_name: Name of the quantization config
        kendall_tau: Kendall's tau correlation with baseline ranking
        top3_preserved_rate: Fraction of samples where top-3 is preserved
        max_score_diff: Maximum score difference vs baseline
        mean_score_diff: Mean score difference vs baseline
        memory_bytes_original: Memory of original params
        memory_bytes_quantized: Memory of quantized params
        memory_reduction_ratio: Fraction of memory saved (0 to 1)
        latency_p50_ms: 50th percentile latency
        latency_p95_ms: 95th percentile latency
        latency_ratio: Ratio vs baseline (1.0 = same)
        passes_accuracy_gate: True if top3_preserved > 0.90
        passes_memory_gate: True if memory_reduction > 0.40
        passes_latency_gate: True if latency_ratio < 1.2
        passes_all_gates: True if all gates pass
    """
    config_name: str

    # Accuracy metrics
    kendall_tau: float
    top3_preserved_rate: float
    max_score_diff: float
    mean_score_diff: float

    # Memory metrics
    memory_bytes_original: int
    memory_bytes_quantized: int
    memory_reduction_ratio: float

    # Latency metrics
    latency_p50_ms: float
    latency_p95_ms: float
    latency_ratio: float

    # Gate status
    passes_accuracy_gate: bool
    passes_memory_gate: bool
    passes_latency_gate: bool
    passes_all_gates: bool


@dataclass
class StudyConfig:
    """Configuration for quantization study.

    Attributes:
        num_eval_batches: Number of batches for accuracy evaluation
        num_warmup_runs: Number of warmup runs before timing
        num_timing_runs: Number of runs for latency measurement
        min_top3_preserved: Minimum top-3 preservation rate (accuracy gate)
        min_memory_reduction: Minimum memory reduction ratio (memory gate)
        max_latency_ratio: Maximum latency ratio vs baseline (latency gate)
    """
    num_eval_batches: int = 100
    num_warmup_runs: int = 5
    num_timing_runs: int = 20
    min_top3_preserved: float = 0.90
    min_memory_reduction: float = 0.40
    max_latency_ratio: float = 1.2


def compute_kendall_tau(ranking1: np.ndarray, ranking2: np.ndarray) -> float:
    """Compute Kendall's tau rank correlation coefficient.

    Args:
        ranking1: First ranking (indices in rank order)
        ranking2: Second ranking

    Returns:
        Kendall's tau in range [-1, 1]
    """
    n = len(ranking1)
    if n < 2:
        return 1.0

    # Convert to rank arrays if needed
    if ranking1.ndim == 1 and ranking2.ndim == 1:
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                # Compare pairs
                diff1 = np.sign(ranking1[i] - ranking1[j])
                diff2 = np.sign(ranking2[i] - ranking2[j])

                if diff1 * diff2 > 0:
                    concordant += 1
                elif diff1 * diff2 < 0:
                    discordant += 1
                # Ties don't count

        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            return 1.0

        return (concordant - discordant) / total_pairs

    return 1.0


def compute_top3_match(baseline_scores: np.ndarray, quant_scores: np.ndarray) -> bool:
    """Check if top-3 candidates match between baseline and quantized.

    Args:
        baseline_scores: Baseline scores [num_candidates, num_actions]
        quant_scores: Quantized scores [num_candidates, num_actions]

    Returns:
        True if top-3 candidates (by primary score) match
    """
    # Use primary score (index 0)
    baseline_ranking = np.argsort(-baseline_scores[:, 0])[:3]
    quant_ranking = np.argsort(-quant_scores[:, 0])[:3]

    return set(baseline_ranking) == set(quant_ranking)


class QuantizationStudy:
    """Comparative quantization study framework.

    Runs benchmarks comparing multiple quantization configurations against
    a baseline and selects the best one based on accuracy, memory, and latency.
    """

    def __init__(
        self,
        model_config: PhoenixModelConfig,
        study_config: Optional[StudyConfig] = None,
    ):
        """Initialize quantization study.

        Args:
            model_config: Phoenix model configuration
            study_config: Study parameters (defaults if None)
        """
        self.model_config = model_config
        self.study_config = study_config or StudyConfig()
        self._base_runner: Optional[RecsysInferenceRunner] = None

    def _get_base_runner(self) -> RecsysInferenceRunner:
        """Get or create the base (unquantized) runner."""
        if self._base_runner is None:
            model_runner = ModelRunner(model=self.model_config)
            self._base_runner = RecsysInferenceRunner(runner=model_runner, name="base")
            self._base_runner.initialize()
        return self._base_runner

    def _create_eval_batches(
        self, num_batches: int
    ) -> List[Tuple[RecsysBatch, RecsysEmbeddings]]:
        """Create evaluation batches."""
        batches = []
        for i in range(num_batches):
            # Use different random seeds for variety
            batch, embeddings = create_example_batch(
                batch_size=1,
                emb_size=self.model_config.emb_size,
                history_len=self.model_config.history_seq_len,
                num_candidates=self.model_config.candidate_seq_len,
                num_actions=self.model_config.num_actions,
                num_user_hashes=self.model_config.hash_config.num_user_hashes,
                num_item_hashes=self.model_config.hash_config.num_item_hashes,
                num_author_hashes=self.model_config.hash_config.num_author_hashes,
            )
            batches.append((batch, embeddings))
        return batches

    def _collect_baseline_outputs(
        self, batches: List[Tuple[RecsysBatch, RecsysEmbeddings]]
    ) -> List[RankingOutput]:
        """Collect baseline outputs for all batches."""
        runner = self._get_base_runner()
        outputs = []
        for batch, embeddings in batches:
            output = runner.rank(batch, embeddings)
            outputs.append(output)
        return outputs

    def _measure_latency(
        self,
        runner,
        batch: RecsysBatch,
        embeddings: RecsysEmbeddings,
    ) -> dict:
        """Measure latency statistics for a runner."""
        cfg = self.study_config

        # Warmup
        for _ in range(cfg.num_warmup_runs):
            output = runner.rank(batch, embeddings)
            jax.block_until_ready(output.scores)

        # Timing runs
        latencies = []
        for _ in range(cfg.num_timing_runs):
            start = time.perf_counter()
            output = runner.rank(batch, embeddings)
            jax.block_until_ready(output.scores)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            'p50': float(np.percentile(latencies, 50)),
            'p95': float(np.percentile(latencies, 95)),
            'mean': float(np.mean(latencies)),
        }

    def evaluate_config(
        self,
        quant_config: QuantizationConfig,
        eval_batches: List[Tuple[RecsysBatch, RecsysEmbeddings]],
        baseline_outputs: List[RankingOutput],
        baseline_latency: dict,
    ) -> BenchmarkMetrics:
        """Evaluate a single quantization configuration.

        Args:
            quant_config: Configuration to evaluate
            eval_batches: Evaluation batches
            baseline_outputs: Baseline outputs for comparison
            baseline_latency: Baseline latency stats

        Returns:
            BenchmarkMetrics for this configuration
        """
        cfg = self.study_config
        base_runner = self._get_base_runner()

        # Create quantized runner
        quant_runner = QuantizedPhoenixRunner(base_runner, quant_config)

        # Accuracy evaluation
        top3_matches = 0
        score_diffs = []
        kendall_taus = []

        for i, (batch, embeddings) in enumerate(eval_batches):
            quant_output = quant_runner.rank(batch, embeddings)
            baseline_output = baseline_outputs[i]

            # Get scores as numpy
            baseline_scores = np.asarray(baseline_output.scores[0])
            quant_scores = np.asarray(quant_output.scores[0])

            # Top-3 match
            if compute_top3_match(baseline_scores, quant_scores):
                top3_matches += 1

            # Score difference
            diff = np.abs(baseline_scores - quant_scores)
            score_diffs.append(diff)

            # Kendall's tau (on primary score ranking)
            baseline_ranking = np.argsort(-baseline_scores[:, 0])
            quant_ranking = np.argsort(-quant_scores[:, 0])
            tau = compute_kendall_tau(baseline_ranking, quant_ranking)
            kendall_taus.append(tau)

        top3_rate = top3_matches / len(eval_batches)
        all_diffs = np.concatenate([d.flatten() for d in score_diffs])
        max_diff = float(np.max(all_diffs))
        mean_diff = float(np.mean(all_diffs))
        mean_tau = float(np.mean(kendall_taus))

        # Memory evaluation
        mem_original = quant_runner.get_original_memory_bytes()
        mem_quantized = quant_runner.get_quantized_memory_bytes()
        mem_reduction = quant_runner.get_memory_reduction_ratio()

        # Latency evaluation
        batch, embeddings = eval_batches[0]
        quant_latency = self._measure_latency(quant_runner, batch, embeddings)
        latency_ratio = quant_latency['p50'] / baseline_latency['p50']

        # Gate checks
        passes_accuracy = top3_rate >= cfg.min_top3_preserved
        passes_memory = mem_reduction >= cfg.min_memory_reduction
        passes_latency = latency_ratio <= cfg.max_latency_ratio

        return BenchmarkMetrics(
            config_name=quant_config.name or "unnamed",
            kendall_tau=mean_tau,
            top3_preserved_rate=top3_rate,
            max_score_diff=max_diff,
            mean_score_diff=mean_diff,
            memory_bytes_original=mem_original,
            memory_bytes_quantized=mem_quantized,
            memory_reduction_ratio=mem_reduction,
            latency_p50_ms=quant_latency['p50'],
            latency_p95_ms=quant_latency['p95'],
            latency_ratio=latency_ratio,
            passes_accuracy_gate=passes_accuracy,
            passes_memory_gate=passes_memory,
            passes_latency_gate=passes_latency,
            passes_all_gates=passes_accuracy and passes_memory and passes_latency,
        )

    def run(
        self,
        configs: Optional[List[QuantizationConfig]] = None,
    ) -> List[BenchmarkMetrics]:
        """Run comparative quantization study.

        Args:
            configs: Configurations to test (defaults to STUDY_CONFIGS)

        Returns:
            List of BenchmarkMetrics for each configuration
        """
        configs_to_use: List[QuantizationConfig] = configs if configs is not None else list(STUDY_CONFIGS)
        cfg = self.study_config

        print(f"Running quantization study with {len(configs_to_use)} configurations")
        print(f"  Eval batches: {cfg.num_eval_batches}")
        print(f"  Timing runs: {cfg.num_timing_runs}")
        print()

        # Create evaluation batches
        print("Creating evaluation batches...")
        eval_batches = self._create_eval_batches(cfg.num_eval_batches)

        # Collect baseline outputs
        print("Collecting baseline outputs...")
        baseline_outputs = self._collect_baseline_outputs(eval_batches)

        # Measure baseline latency
        print("Measuring baseline latency...")
        base_runner = self._get_base_runner()
        batch, embeddings = eval_batches[0]
        baseline_latency = self._measure_latency(base_runner, batch, embeddings)
        print(f"  Baseline p50: {baseline_latency['p50']:.2f} ms")
        print()

        # Evaluate each config
        results = []
        for idx, config in enumerate(configs_to_use):
            print(f"[{idx+1}/{len(configs_to_use)}] Evaluating: {config.name}")
            metrics = self.evaluate_config(
                config, eval_batches, baseline_outputs, baseline_latency
            )
            results.append(metrics)

            # Print summary
            status = "PASS" if metrics.passes_all_gates else "FAIL"
            print(f"  Top-3: {metrics.top3_preserved_rate:.1%}, "
                  f"Memory: {metrics.memory_reduction_ratio:.1%}, "
                  f"Latency: {metrics.latency_ratio:.2f}x "
                  f"[{status}]")

        print()
        return results


@dataclass
class WinnerSelectionCriteria:
    """Criteria for selecting the winning quantization config.

    Weights are used to compute a composite score for ranking candidates
    that pass all gates.
    """
    weight_accuracy: float = 0.40   # Kendall's tau importance
    weight_memory: float = 0.35    # Memory reduction importance
    weight_latency: float = 0.25   # Latency (inverse ratio) importance

    require_all_gates: bool = True


def select_winner(
    results: List[BenchmarkMetrics],
    criteria: Optional[WinnerSelectionCriteria] = None,
) -> Tuple[Optional[BenchmarkMetrics], dict]:
    """Select the winning configuration from study results.

    Args:
        results: List of benchmark metrics
        criteria: Selection criteria (defaults if None)

    Returns:
        Tuple of (winner_metrics or None, selection_details)
    """
    if criteria is None:
        criteria = WinnerSelectionCriteria()

    # Filter by gates if required
    if criteria.require_all_gates:
        candidates = [r for r in results if r.passes_all_gates]
    else:
        candidates = list(results)

    if not candidates:
        return None, {
            "error": "No configurations pass all required gates",
            "num_total": len(results),
            "num_passing": 0,
        }

    # Compute weighted scores
    def compute_score(m: BenchmarkMetrics) -> float:
        # Normalize metrics to [0, 1] where higher is better
        accuracy_score = m.kendall_tau  # Already in [0, 1]
        memory_score = m.memory_reduction_ratio  # Higher = better
        latency_score = 1.0 / max(m.latency_ratio, 0.1)  # Lower ratio = better
        latency_score = min(latency_score, 1.0)  # Cap at 1.0

        return (
            criteria.weight_accuracy * accuracy_score +
            criteria.weight_memory * memory_score +
            criteria.weight_latency * latency_score
        )

    scored = [(compute_score(m), m) for m in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)

    winner_score, winner = scored[0]

    return winner, {
        "winner_score": winner_score,
        "num_total": len(results),
        "num_passing": len(candidates),
        "all_scores": [(m.config_name, s) for s, m in scored],
        "criteria": {
            "weight_accuracy": criteria.weight_accuracy,
            "weight_memory": criteria.weight_memory,
            "weight_latency": criteria.weight_latency,
        },
    }


def format_results_table(results: List[BenchmarkMetrics]) -> str:
    """Format results as a table string."""
    lines = []
    lines.append(
        f"{'Config':<25} {'Top3':>6} {'Tau':>6} {'MemRed':>7} "
        f"{'Latency':>8} {'Status':>6}"
    )
    lines.append("-" * 70)

    for r in results:
        status = "PASS" if r.passes_all_gates else "FAIL"
        lines.append(
            f"{r.config_name:<25} {r.top3_preserved_rate:>5.1%} "
            f"{r.kendall_tau:>6.3f} {r.memory_reduction_ratio:>6.1%} "
            f"{r.latency_ratio:>7.2f}x {status:>6}"
        )

    return "\n".join(lines)
