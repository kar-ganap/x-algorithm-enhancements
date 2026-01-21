"""Option B: Path Divergence Analysis.

This script analyzes how different initial engagement choices lead to
diverging recommendation trajectories. It answers:

- Do small choices compound? (filter bubble risk)
- Or do paths converge to similar outcomes? (robust system)

Usage:
    uv run python enhancements/analysis/path_divergence.py
"""

import sys
from pathlib import Path
from typing import List, Tuple

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
    compare_trajectories,
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


def run_trajectory_with_first_choice(
    runner, batch, embeddings, first_choice_rank: int, subsequent_steps: int
) -> TrajectoryPath:
    """Run a trajectory starting with a specific first choice."""
    simulator = TrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
    )
    simulator.initialize()

    # Make first choice
    simulator.engage(first_choice_rank)

    # Then always choose top for remaining steps
    simulator.engage_top_n(subsequent_steps)

    return simulator.get_trajectory()


def compute_ranking_overlap(path1: TrajectoryPath, path2: TrajectoryPath, step: int) -> float:
    """Compute overlap in top-K rankings between two paths at a given step."""
    if step >= len(path1.steps) or step >= len(path2.steps):
        return float('nan')

    scores1 = {cs.index: cs.rank for cs in path1.steps[step].remaining_scores}
    scores2 = {cs.index: cs.rank for cs in path2.steps[step].remaining_scores}

    # Common candidates
    common = set(scores1.keys()) & set(scores2.keys())
    if not common:
        return 0.0

    # For each common candidate, check if ranks are similar (within 1)
    similar_ranks = sum(1 for idx in common if abs(scores1[idx] - scores2[idx]) <= 1)
    return similar_ranks / len(common)


def visualize_path_tree(paths: List[TrajectoryPath], labels: List[str]):
    """Create ASCII tree visualization of diverging paths."""
    print("\n" + "=" * 70)
    print("PATH DIVERGENCE TREE")
    print("=" * 70)

    # Get initial rankings (same for all paths)
    initial = paths[0].steps[0]
    print(f"\nInitial state ({len(initial.remaining_scores)} candidates)")
    print("  Rankings: ", end="")
    for cs in initial.remaining_scores[:4]:
        print(f"C{cs.index}", end=" ")
    print("...")

    print("\n  Diverging paths:")

    for label, path in zip(labels, paths):
        engaged = path.engagement_sequence
        print(f"\n  [{label}]: ", end="")

        # Show engagement sequence
        for i, cand in enumerate(engaged):
            if i > 0:
                print(" → ", end="")
            print(f"C{cand}", end="")

        # Show final state
        final_step = path.steps[-1]
        print(f"\n           Final: ", end="")
        for cs in final_step.remaining_scores[:3]:
            print(f"C{cs.index}({cs.score:.2f})", end=" ")
        if len(final_step.remaining_scores) > 3:
            print("...", end="")
        print()


def analyze_divergence(paths: List[TrajectoryPath], labels: List[str]):
    """Analyze divergence between paths over time."""
    print("\n" + "=" * 70)
    print("DIVERGENCE ANALYSIS")
    print("=" * 70)

    max_steps = max(len(p.steps) for p in paths)
    num_paths = len(paths)

    print(f"\nComparing {num_paths} paths over {max_steps} steps")

    # Compute pairwise divergence at each step
    print(f"\n{'Step':<6} {'Avg Score Diff':<16} {'Rank Overlap':<14} {'Common Cands':<12}")
    print("-" * 55)

    for step in range(max_steps):
        score_diffs = []
        overlaps = []
        common_counts = []

        for i in range(num_paths):
            for j in range(i + 1, num_paths):
                if step < len(paths[i].steps) and step < len(paths[j].steps):
                    # Get scores at this step
                    scores_i = {cs.index: cs.score for cs in paths[i].steps[step].remaining_scores}
                    scores_j = {cs.index: cs.score for cs in paths[j].steps[step].remaining_scores}

                    # Common candidates
                    common = set(scores_i.keys()) & set(scores_j.keys())
                    common_counts.append(len(common))

                    if common:
                        # Score difference
                        diffs = [abs(scores_i[idx] - scores_j[idx]) for idx in common]
                        score_diffs.append(np.mean(diffs))

                        # Rank overlap
                        overlap = compute_ranking_overlap(paths[i], paths[j], step)
                        overlaps.append(overlap)

        if score_diffs:
            avg_diff = np.mean(score_diffs)
            avg_overlap = np.mean(overlaps) if overlaps else 0
            avg_common = np.mean(common_counts)
            print(f"{step:<6} {avg_diff:<16.4f} {avg_overlap*100:<13.1f}% {avg_common:<12.1f}")

    # Final comparison
    print("\n" + "-" * 55)
    print("FINAL STATE COMPARISON")
    print("-" * 55)

    for i, (label, path) in enumerate(zip(labels, paths)):
        final = path.steps[-1]
        remaining = [cs.index for cs in final.remaining_scores]
        print(f"\n{label}:")
        print(f"  Engaged: {' → '.join(f'C{c}' for c in path.engagement_sequence)}")
        print(f"  Remaining: {remaining}")
        if final.remaining_scores:
            print(f"  Top remaining: C{final.remaining_scores[0].index} "
                  f"(score: {final.remaining_scores[0].score:.4f})")


def compute_path_similarity_matrix(paths: List[TrajectoryPath], labels: List[str]):
    """Compute and display similarity matrix between final path states."""
    print("\n" + "=" * 70)
    print("PATH SIMILARITY MATRIX (Final State)")
    print("=" * 70)

    n = len(paths)

    # Compute similarity based on final remaining candidates and their order
    similarity = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            final_i = paths[i].steps[-1].remaining_scores
            final_j = paths[j].steps[-1].remaining_scores

            # Jaccard similarity of remaining candidates
            set_i = set(cs.index for cs in final_i)
            set_j = set(cs.index for cs in final_j)

            if set_i or set_j:
                jaccard = len(set_i & set_j) / len(set_i | set_j)
            else:
                jaccard = 1.0

            similarity[i, j] = jaccard

    # Print matrix
    print("\n" + " " * 12, end="")
    for label in labels:
        print(f"{label[:10]:<12}", end="")
    print()

    for i, label in enumerate(labels):
        print(f"{label[:10]:<12}", end="")
        for j in range(n):
            val = similarity[i, j]
            if i == j:
                print(f"{'—':^12}", end="")
            else:
                print(f"{val*100:>10.1f}% ", end="")
        print()

    # Interpretation
    avg_off_diag = np.mean([similarity[i, j] for i in range(n) for j in range(n) if i != j])
    print(f"\nAverage path similarity: {avg_off_diag*100:.1f}%")

    if avg_off_diag > 0.7:
        print("→ Paths CONVERGE: Different choices lead to similar outcomes")
    elif avg_off_diag < 0.3:
        print("→ Paths DIVERGE: Small choices lead to very different outcomes (filter bubble risk)")
    else:
        print("→ Paths show MODERATE divergence")


def run_analysis():
    """Run the full path divergence analysis."""
    print("=" * 70)
    print("PATH DIVERGENCE ANALYSIS")
    print("=" * 70)
    print("\nThis analysis compares trajectories starting with different first choices.")
    print("Question: Do small initial choices compound into very different outcomes?")

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

    # Run trajectories with different first choices
    print("\nRunning trajectories with different first choices...")
    paths = []
    labels = []

    # Path A: Start with rank 0 (top choice)
    print("  Path A: Start with top-ranked candidate...")
    path_a = run_trajectory_with_first_choice(runner, batch, embeddings, 0, 4)
    paths.append(path_a)
    labels.append("Top First")

    # Path B: Start with rank 1 (second choice)
    print("  Path B: Start with 2nd-ranked candidate...")
    path_b = run_trajectory_with_first_choice(runner, batch, embeddings, 1, 4)
    paths.append(path_b)
    labels.append("2nd First")

    # Path C: Start with rank 2 (third choice)
    print("  Path C: Start with 3rd-ranked candidate...")
    path_c = run_trajectory_with_first_choice(runner, batch, embeddings, 2, 4)
    paths.append(path_c)
    labels.append("3rd First")

    # Path D: Start with last ranked
    print("  Path D: Start with bottom-ranked candidate...")
    path_d = run_trajectory_with_first_choice(runner, batch, embeddings, -1, 4)
    paths.append(path_d)
    labels.append("Bottom First")

    # Visualizations
    visualize_path_tree(paths, labels)
    analyze_divergence(paths, labels)
    compute_path_similarity_matrix(paths, labels)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nCompared {len(paths)} trajectories with different first choices")
    print("Each trajectory then followed top-ranked candidates for 4 more steps")

    # Compare engagement sequences
    print("\nEngagement sequences:")
    for label, path in zip(labels, paths):
        seq = ' → '.join(f'C{c}' for c in path.engagement_sequence)
        print(f"  {label}: {seq}")


if __name__ == "__main__":
    run_analysis()
