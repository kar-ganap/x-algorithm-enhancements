"""Option C: Sensitivity Analysis.

This script runs many trajectory simulations to quantify:
- How sensitive are final recommendations to early choices?
- How much does "following recommendations" constrain the outcome space?

Usage:
    uv run python enhancements/analysis/sensitivity_analysis.py
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner
from enhancements.analysis.trajectory_simulation import (
    TrajectorySimulator,
    TrajectoryPath,
)


def create_test_config():
    """Create a test model configuration."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=256,
        num_actions=len(ACTIONS),
        history_seq_len=32,
        candidate_seq_len=8,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=256,
            widening_factor=2,
            key_size=64,
            num_q_heads=4,
            num_kv_heads=2,
            num_layers=4,
            attn_output_multiplier=0.125,
        ),
    )


class SensitivityMetrics:
    """Aggregated metrics from many trajectory runs."""

    def __init__(self, num_candidates: int):
        self.num_candidates = num_candidates
        # How often each candidate was engaged across all runs
        self.engagement_counts: Dict[int, int] = defaultdict(int)
        # How often each candidate remained at end
        self.final_remaining_counts: Dict[int, int] = defaultdict(int)
        # Position frequencies: position -> {candidate -> count}
        self.position_frequencies: Dict[int, Dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # All engagement sequences
        self.sequences: List[List[int]] = []
        self.num_runs = 0

    def add_trajectory(self, trajectory: TrajectoryPath):
        """Add a trajectory's data to the metrics."""
        self.num_runs += 1

        # Track engagement sequence
        self.sequences.append(trajectory.engagement_sequence)

        # Count engagements
        for idx in trajectory.engagement_sequence:
            self.engagement_counts[idx] += 1

        # Track position frequencies
        for position, idx in enumerate(trajectory.engagement_sequence):
            self.position_frequencies[position][idx] += 1

        # Track final remaining
        final_step = trajectory.steps[-1]
        for cs in final_step.remaining_scores:
            self.final_remaining_counts[cs.index] += 1

    def compute_engagement_entropy(self) -> float:
        """Compute entropy of engagement distribution.

        Higher entropy = more uniform distribution = less predictable
        Lower entropy = concentrated on few candidates = more predictable
        """
        total = sum(self.engagement_counts.values())
        if total == 0:
            return 0.0

        probs = np.array([c / total for c in self.engagement_counts.values()])
        probs = probs[probs > 0]  # Remove zeros for log
        return float(-np.sum(probs * np.log2(probs)))

    def compute_position_entropy(self, position: int) -> float:
        """Compute entropy at a specific position.

        Lower entropy = same candidate often appears at this position
        Higher entropy = different candidates appear at this position
        """
        if position not in self.position_frequencies:
            return 0.0

        counts = list(self.position_frequencies[position].values())
        total = sum(counts)
        if total == 0:
            return 0.0

        probs = np.array([c / total for c in counts])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def compute_outcome_diversity(self) -> float:
        """Compute how many unique final states were observed.

        Returns ratio of unique sequences to total runs.
        """
        unique_sequences = set(tuple(s) for s in self.sequences)
        return len(unique_sequences) / self.num_runs if self.num_runs > 0 else 0.0

    def get_position_stability(self, position: int) -> Tuple[int, float]:
        """Get most common candidate at position and its frequency."""
        if position not in self.position_frequencies:
            return (-1, 0.0)

        pos_counts = self.position_frequencies[position]
        if not pos_counts:
            return (-1, 0.0)

        total = sum(pos_counts.values())
        most_common = max(pos_counts.items(), key=lambda x: x[1])
        return (most_common[0], most_common[1] / total)


def run_random_trajectories(
    runner,
    batch,
    embeddings,
    num_candidates: int,
    num_runs: int,
    num_engagements: int,
    rng: np.random.Generator,
) -> SensitivityMetrics:
    """Run trajectories with random choices."""
    metrics = SensitivityMetrics(num_candidates)

    for _ in range(num_runs):
        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        # Make random choices
        for _ in range(num_engagements):
            if not simulator._remaining_candidate_indices:
                break
            num_remaining = len(simulator._remaining_candidate_indices)
            choice = rng.integers(0, num_remaining)
            simulator.engage(choice)

        metrics.add_trajectory(simulator.get_trajectory())

    return metrics


def run_top_biased_trajectories(
    runner,
    batch,
    embeddings,
    num_candidates: int,
    num_runs: int,
    num_engagements: int,
    top_probability: float,
    rng: np.random.Generator,
) -> SensitivityMetrics:
    """Run trajectories biased toward top choices.

    With probability `top_probability`, choose the top-ranked candidate.
    Otherwise, choose randomly from remaining.
    """
    metrics = SensitivityMetrics(num_candidates)

    for _ in range(num_runs):
        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        for _ in range(num_engagements):
            if not simulator._remaining_candidate_indices:
                break

            if rng.random() < top_probability:
                # Choose top
                simulator.engage(0)
            else:
                # Choose randomly
                num_remaining = len(simulator._remaining_candidate_indices)
                choice = rng.integers(0, num_remaining)
                simulator.engage(choice)

        metrics.add_trajectory(simulator.get_trajectory())

    return metrics


def visualize_position_heatmap(metrics: SensitivityMetrics, title: str):
    """Create ASCII heatmap of position frequencies."""
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print("=" * 70)
    print("\nPosition Heatmap (rows=candidates, cols=positions)")
    print("Higher values = candidate more likely at that position")

    num_positions = max(metrics.position_frequencies.keys()) + 1
    num_candidates = metrics.num_candidates

    # Header
    print(f"\n{'Cand':<6}", end="")
    for pos in range(num_positions):
        print(f"{'Pos'+str(pos):<8}", end="")
    print(f"{'Total':<8}")
    print("-" * (6 + 8 * (num_positions + 1)))

    # Body
    for cand in range(num_candidates):
        print(f"C{cand:<5}", end="")
        total = 0
        for pos in range(num_positions):
            count = metrics.position_frequencies[pos].get(cand, 0)
            total += count
            # Show as percentage
            pct = count / metrics.num_runs * 100 if metrics.num_runs > 0 else 0
            if pct >= 50:
                print(f"{'█'}{pct:>5.0f}% ", end="")
            elif pct >= 25:
                print(f"{'▓'}{pct:>5.0f}% ", end="")
            elif pct >= 10:
                print(f"{'▒'}{pct:>5.0f}% ", end="")
            elif pct > 0:
                print(f"{'░'}{pct:>5.0f}% ", end="")
            else:
                print(f"{' '}{pct:>5.0f}% ", end="")
        # Total engagements for this candidate
        print(f"{total:<8}")


def visualize_metrics_comparison(
    random_metrics: SensitivityMetrics,
    biased_metrics: SensitivityMetrics,
    top_probability: float,
):
    """Compare metrics between random and top-biased strategies."""
    print(f"\n{'=' * 70}")
    print("STRATEGY COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Random':<15} {'Top-Biased':<15}")
    print("-" * 65)

    # Engagement entropy
    rand_ent = random_metrics.compute_engagement_entropy()
    bias_ent = biased_metrics.compute_engagement_entropy()
    max_ent = np.log2(random_metrics.num_candidates)
    print(f"{'Engagement entropy':<35} {rand_ent:<15.3f} {bias_ent:<15.3f}")
    print(f"{'  (max possible)':<35} {max_ent:<15.3f} {max_ent:<15.3f}")

    # Outcome diversity
    rand_div = random_metrics.compute_outcome_diversity()
    bias_div = biased_metrics.compute_outcome_diversity()
    print(f"{'Outcome diversity':<35} {rand_div*100:<14.1f}% {bias_div*100:<14.1f}%")

    # Position-by-position entropy
    print(f"\n{'Position Entropy (lower = more predictable):':<50}")
    print(f"{'Position':<15} {'Random':<15} {'Top-Biased':<15}")
    print("-" * 45)

    num_positions = max(
        max(random_metrics.position_frequencies.keys(), default=0),
        max(biased_metrics.position_frequencies.keys(), default=0),
    ) + 1

    for pos in range(num_positions):
        rand_pos_ent = random_metrics.compute_position_entropy(pos)
        bias_pos_ent = biased_metrics.compute_position_entropy(pos)
        print(f"{pos:<15} {rand_pos_ent:<15.3f} {bias_pos_ent:<15.3f}")

    # Position stability
    print(f"\n{'Position Stability (most common candidate at each position):':<60}")
    print(f"{'Position':<10} {'Random':<25} {'Top-Biased':<25}")
    print("-" * 60)

    for pos in range(num_positions):
        rand_cand, rand_freq = random_metrics.get_position_stability(pos)
        bias_cand, bias_freq = biased_metrics.get_position_stability(pos)
        rand_str = f"C{rand_cand} ({rand_freq*100:.0f}%)" if rand_cand >= 0 else "—"
        bias_str = f"C{bias_cand} ({bias_freq*100:.0f}%)" if bias_cand >= 0 else "—"
        print(f"{pos:<10} {rand_str:<25} {bias_str:<25}")


def interpret_results(
    random_metrics: SensitivityMetrics,
    biased_metrics: SensitivityMetrics,
):
    """Provide interpretation of the sensitivity analysis results."""
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print("=" * 70)

    rand_div = random_metrics.compute_outcome_diversity()
    bias_div = biased_metrics.compute_outcome_diversity()
    rand_ent = random_metrics.compute_engagement_entropy()
    bias_ent = biased_metrics.compute_engagement_entropy()

    print("\n1. OUTCOME SPACE CONSTRAINT")
    if bias_div < rand_div * 0.5:
        print(f"   → Strong constraint: Following recommendations reduces outcome")
        print(f"     diversity by {(1 - bias_div/rand_div)*100:.0f}%")
        print(f"   → This suggests the system funnels users toward specific outcomes")
    elif bias_div < rand_div * 0.8:
        print(f"   → Moderate constraint: Some reduction in outcome diversity")
        print(f"     ({(1 - bias_div/rand_div)*100:.0f}% reduction)")
    else:
        print(f"   → Weak constraint: Outcome diversity similar regardless of strategy")
        print(f"   → System allows exploration even when following recommendations")

    print("\n2. ENGAGEMENT CONCENTRATION")
    entropy_ratio = bias_ent / rand_ent if rand_ent > 0 else 1.0
    if entropy_ratio < 0.7:
        print(f"   → High concentration: Biased strategy engages with fewer")
        print(f"     distinct candidates (entropy {entropy_ratio:.0%} of random)")
        print(f"   → Potential filter bubble risk")
    elif entropy_ratio < 0.9:
        print(f"   → Moderate concentration: Some narrowing of engagement")
    else:
        print(f"   → Low concentration: Engagement spread similarly across candidates")

    print("\n3. EARLY CHOICE IMPACT")
    # Check if first position is more stable in biased vs random
    _, rand_p0_freq = random_metrics.get_position_stability(0)
    _, bias_p0_freq = biased_metrics.get_position_stability(0)

    if bias_p0_freq > rand_p0_freq * 1.5:
        print(f"   → First choice is significantly more predictable with top-biased")
        print(f"     strategy ({bias_p0_freq*100:.0f}% vs {rand_p0_freq*100:.0f}%)")
        print(f"   → Early recommendations have strong influence")
    else:
        print(f"   → First choice predictability similar between strategies")
        print(f"   → Early choices have moderate influence")


def run_analysis():
    """Run the full sensitivity analysis."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nThis analysis quantifies how sensitive outcomes are to early choices")
    print("by comparing random vs recommendation-following strategies.")

    print("\nInitializing model...")
    config = create_test_config()
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, embeddings = create_example_batch(
        batch_size=1,
        emb_size=config.emb_size,
        history_len=config.history_seq_len,
        num_candidates=config.candidate_seq_len,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )

    print(f"Model initialized with {config.candidate_seq_len} candidates")

    # Configuration
    num_runs = 50
    num_engagements = 5
    top_probability = 0.7

    rng = np.random.default_rng(seed=42)

    num_candidates = config.candidate_seq_len

    # Run random trajectories
    print(f"\nRunning {num_runs} random trajectories...")
    random_metrics = run_random_trajectories(
        runner, batch, embeddings, num_candidates, num_runs, num_engagements, rng
    )

    # Run top-biased trajectories
    print(f"Running {num_runs} top-biased trajectories (p={top_probability})...")
    biased_metrics = run_top_biased_trajectories(
        runner, batch, embeddings, num_candidates, num_runs, num_engagements, top_probability, rng
    )

    # Visualizations
    visualize_position_heatmap(random_metrics, "RANDOM STRATEGY - Position Heatmap")
    visualize_position_heatmap(biased_metrics, f"TOP-BIASED STRATEGY (p={top_probability}) - Position Heatmap")

    visualize_metrics_comparison(random_metrics, biased_metrics, top_probability)
    interpret_results(random_metrics, biased_metrics)

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)
    print(f"\nRan {num_runs} trajectories for each strategy")
    print(f"Each trajectory made {num_engagements} engagements")
    print(f"Top-biased strategy chose top with probability {top_probability}")

    print(f"\nUnique outcomes observed:")
    rand_unique = len(set(tuple(s) for s in random_metrics.sequences))
    bias_unique = len(set(tuple(s) for s in biased_metrics.sequences))
    print(f"  Random strategy: {rand_unique}/{num_runs} unique sequences")
    print(f"  Top-biased strategy: {bias_unique}/{num_runs} unique sequences")


if __name__ == "__main__":
    run_analysis()
