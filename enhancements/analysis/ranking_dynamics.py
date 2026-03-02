"""Option A: Ranking Dynamics Visualization.

This script visualizes how candidate rankings evolve through an engagement sequence.
It shows:
1. Score evolution: How each candidate's score changes step by step
2. Rank volatility: How much rankings shuffle per engagement
3. Winner momentum: Does the top candidate stay on top?

Usage:
    uv run python enhancements/analysis/ranking_dynamics.py
"""

import sys
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancements.analysis.trajectory_simulation import (
    TrajectorySimulator,
    format_trajectory_table,
)
from enhancements.optimization.full_kv_cache import FullKVCachedRunner
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch


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
        candidate_seq_len=8,  # 8 candidates to rank
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


def visualize_score_evolution(trajectory):
    """Create ASCII visualization of score evolution."""
    print("\n" + "=" * 70)
    print("SCORE EVOLUTION")
    print("=" * 70)

    # Track scores for each candidate across steps
    candidate_scores = {}  # candidate_idx -> list of (step, score)

    for step in trajectory.steps:
        for cs in step.remaining_scores:
            if cs.index not in candidate_scores:
                candidate_scores[cs.index] = []
            candidate_scores[cs.index].append((step.step_num, cs.score))

    # Find score range for normalization
    all_scores = [s for scores in candidate_scores.values() for _, s in scores]
    if not all_scores:
        print("No scores to display")
        return

    min_score, max_score = min(all_scores), max(all_scores)
    score_range = max_score - min_score if max_score > min_score else 1

    print(f"\nScore range: [{min_score:.3f}, {max_score:.3f}]")
    print()

    # ASCII bar chart for each step
    bar_width = 40
    for step in trajectory.steps:
        print(f"Step {step.step_num}:", end="")
        if step.engaged_candidate_idx is not None:
            print(f" (engaged: {step.engaged_candidate_idx})")
        else:
            print(" (initial)")

        for cs in sorted(step.remaining_scores, key=lambda x: -x.score)[:5]:
            normalized = (cs.score - min_score) / score_range
            bar_len = int(normalized * bar_width)
            bar = "█" * bar_len + "░" * (bar_width - bar_len)
            print(f"  C{cs.index}: [{bar}] {cs.score:+.4f} (rank {cs.rank})")
        print()


def compute_rank_volatility(trajectory):
    """Compute how much rankings change between steps."""
    print("\n" + "=" * 70)
    print("RANK VOLATILITY")
    print("=" * 70)

    volatilities = []

    for i in range(1, len(trajectory.steps)):
        prev_step = trajectory.steps[i - 1]
        curr_step = trajectory.steps[i]

        # Get rankings at each step
        prev_ranks = {cs.index: cs.rank for cs in prev_step.remaining_scores}
        curr_ranks = {cs.index: cs.rank for cs in curr_step.remaining_scores}

        # Compute rank changes for candidates in both steps
        common = set(prev_ranks.keys()) & set(curr_ranks.keys())
        if common:
            rank_changes = [abs(curr_ranks[idx] - prev_ranks[idx]) for idx in common]
            avg_change = np.mean(rank_changes)
            max_change = max(rank_changes)
            volatilities.append((i, avg_change, max_change))

    print(f"\n{'Step':<6} {'Engaged':<10} {'Avg Rank Δ':<12} {'Max Rank Δ':<12}")
    print("-" * 45)

    for i, (step_num, avg_change, max_change) in enumerate(volatilities):
        engaged = trajectory.steps[step_num].engaged_candidate_idx
        print(f"{step_num:<6} C{engaged:<9} {avg_change:<12.2f} {max_change:<12}")

    if volatilities:
        overall_avg = np.mean([v[1] for v in volatilities])
        print("-" * 45)
        print(f"Overall average rank change: {overall_avg:.2f}")


def analyze_winner_momentum(trajectory):
    """Analyze whether top candidates maintain their position."""
    print("\n" + "=" * 70)
    print("WINNER MOMENTUM ANALYSIS")
    print("=" * 70)

    # Track the top candidate at each step
    top_candidates = []
    for step in trajectory.steps:
        if step.remaining_scores:
            top = step.remaining_scores[0]
            top_candidates.append((step.step_num, top.index, top.score))

    print(f"\n{'Step':<6} {'Top Candidate':<15} {'Score':<12} {'Retained?':<10}")
    print("-" * 50)

    prev_top = None
    retained_count = 0
    total_transitions = 0

    for step_num, top_idx, score in top_candidates:
        retained = "—" if prev_top is None else ("✓" if top_idx == prev_top else "✗")
        if prev_top is not None:
            total_transitions += 1
            if top_idx == prev_top:
                retained_count += 1
        print(f"{step_num:<6} C{top_idx:<14} {score:<12.4f} {retained:<10}")
        prev_top = top_idx

    if total_transitions > 0:
        retention_rate = retained_count / total_transitions * 100
        print("-" * 50)
        print(f"Winner retention rate: {retention_rate:.1f}%")
        print(f"({retained_count}/{total_transitions} transitions maintained top position)")


def run_analysis():
    """Run the full ranking dynamics analysis."""
    print("=" * 70)
    print("RANKING DYNAMICS ANALYSIS")
    print("=" * 70)
    print("\nInitializing model...")

    # Create model and runner
    config = create_test_config()
    runner = FullKVCachedRunner(config)
    runner.initialize()

    # Create example batch
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
    print("\nRunning trajectory simulation...")

    # Create and run simulator
    simulator = TrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
    )
    simulator.initialize()

    # Simulate engaging with top candidate 5 times
    print("\nSimulating 5 engagements (always choosing top-ranked)...")
    simulator.engage_top_n(5)

    # Get trajectory
    trajectory = simulator.get_trajectory()

    # Print full trajectory table
    print(format_trajectory_table(trajectory))

    # Visualizations
    visualize_score_evolution(trajectory)
    compute_rank_volatility(trajectory)
    analyze_winner_momentum(trajectory)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total steps: {len(trajectory.steps)}")
    print(f"Engagement sequence: {' → '.join(f'C{i}' for i in trajectory.engagement_sequence)}")
    print()


if __name__ == "__main__":
    run_analysis()
