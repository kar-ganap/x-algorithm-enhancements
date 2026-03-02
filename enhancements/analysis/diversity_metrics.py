"""Diversity Metrics for Recommendation Analysis.

This module computes metrics to detect filter bubble / echo chamber effects:

1. Intra-list diversity: How different are items in a single slate?
2. Temporal diversity: How do recommendations change over engagement steps?
3. Coverage: What fraction of candidates ever get recommended?
4. Embedding distance: Average pairwise distance between candidates

Supports two modes:
- Simulated (default): Uses perturbation-based score updates (fast, but artificial)
- Real re-ranking: Uses actual model inference at each step (slower, but accurate)

Usage:
    # Simulated mode (fast)
    uv run python enhancements/analysis/diversity_metrics.py

    # Real re-ranking mode (accurate, use with trained weights)
    uv run python enhancements/analysis/diversity_metrics.py --real

    # With more candidates
    uv run python enhancements/analysis/diversity_metrics.py --candidates 100 --real
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancements.analysis.real_trajectory_simulation import (
    RealTrajectorySimulator,
)
from enhancements.analysis.trajectory_simulation import (
    CandidateScore,
    TrajectorySimulator,
)
from enhancements.optimization.full_kv_cache import FullKVCachedRunner
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch


@dataclass
class DiversitySnapshot:
    """Diversity metrics at a single point in time."""
    step: int
    num_remaining: int

    # Embedding-based diversity
    avg_pairwise_distance: float  # Mean L2 distance between embeddings
    min_pairwise_distance: float  # Closest pair (redundancy indicator)
    max_pairwise_distance: float  # Furthest pair (spread indicator)

    # Score-based diversity
    score_std: float  # Standard deviation of scores
    score_range: float  # Max - min score

    # Rank concentration
    top_score_gap: float  # Gap between #1 and #2 (dominance indicator)


@dataclass
class TrajectoryDiversity:
    """Diversity metrics tracked across a full trajectory."""
    snapshots: list[DiversitySnapshot] = field(default_factory=list)
    engaged_embeddings: list[np.ndarray] = field(default_factory=list)

    def diversity_trend(self) -> list[float]:
        """Return avg_pairwise_distance at each step."""
        return [s.avg_pairwise_distance for s in self.snapshots]

    def concentration_trend(self) -> list[float]:
        """Return top_score_gap at each step (higher = more concentrated)."""
        return [s.top_score_gap for s in self.snapshots]

    def diversity_change(self) -> float:
        """Compute % change in diversity from start to end."""
        if len(self.snapshots) < 2:
            return 0.0
        initial = self.snapshots[0].avg_pairwise_distance
        final = self.snapshots[-1].avg_pairwise_distance
        if initial == 0:
            return 0.0
        return (final - initial) / initial * 100


@dataclass
class DiversityAnalysisResult:
    """Aggregated diversity analysis across multiple trajectories."""
    num_trajectories: int
    num_candidates: int
    num_engagements: int

    # Per-step aggregates
    mean_diversity_by_step: list[float]
    std_diversity_by_step: list[float]
    mean_concentration_by_step: list[float]

    # Summary statistics
    mean_diversity_change: float  # Average % change across trajectories
    std_diversity_change: float

    # Coverage metrics
    unique_engaged: int  # How many distinct candidates were engaged
    engagement_gini: float  # 0=equal, 1=concentrated on few

    # Baseline comparison
    random_baseline_diversity: float
    diversity_ratio: float  # actual / baseline (< 1 means less diverse)


def compute_pairwise_distances(embeddings: np.ndarray) -> tuple[float, float, float]:
    """Compute pairwise L2 distances between embeddings.

    Args:
        embeddings: [num_items, emb_dim] array

    Returns:
        (mean_distance, min_distance, max_distance)
    """
    n = embeddings.shape[0]
    if n < 2:
        return (0.0, 0.0, 0.0)

    # Compute all pairwise distances
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(embeddings[i] - embeddings[j])
            distances.append(dist)

    distances = np.array(distances)
    return (float(np.mean(distances)), float(np.min(distances)), float(np.max(distances)))


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """Compute Gini coefficient for engagement distribution.

    0 = perfectly equal distribution
    1 = all engagement on single item
    """
    if len(counts) == 0 or np.sum(counts) == 0:
        return 0.0

    # Sort counts
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)

    # Compute Gini
    cumsum = np.cumsum(sorted_counts)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


def compute_snapshot(
    remaining_scores: list[CandidateScore],
    candidate_embeddings: np.ndarray,
    step: int,
) -> DiversitySnapshot:
    """Compute diversity snapshot for current state.

    Args:
        remaining_scores: Current candidate scores
        candidate_embeddings: [num_candidates, emb_dim] all candidate embeddings
        step: Current step number
    """
    if not remaining_scores:
        return DiversitySnapshot(
            step=step,
            num_remaining=0,
            avg_pairwise_distance=0.0,
            min_pairwise_distance=0.0,
            max_pairwise_distance=0.0,
            score_std=0.0,
            score_range=0.0,
            top_score_gap=0.0,
        )

    # Get embeddings for remaining candidates
    remaining_indices = [cs.index for cs in remaining_scores]
    remaining_embeddings = candidate_embeddings[remaining_indices]

    # Embedding diversity
    avg_dist, min_dist, max_dist = compute_pairwise_distances(remaining_embeddings)

    # Score diversity
    scores = np.array([cs.score for cs in remaining_scores])
    score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
    score_range = float(np.max(scores) - np.min(scores)) if len(scores) > 1 else 0.0

    # Top score gap (concentration indicator)
    sorted_scores = sorted(scores, reverse=True)
    top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    return DiversitySnapshot(
        step=step,
        num_remaining=len(remaining_scores),
        avg_pairwise_distance=avg_dist,
        min_pairwise_distance=min_dist,
        max_pairwise_distance=max_dist,
        score_std=score_std,
        score_range=score_range,
        top_score_gap=float(top_gap),
    )


def run_trajectory_with_diversity(
    runner,
    batch,
    embeddings,
    candidate_embeddings: np.ndarray,
    num_engagements: int,
    strategy: str = "top",  # "top" or "random"
    rng: np.random.Generator | None = None,
    use_real_reranking: bool = False,
    num_item_hashes: int = 2,
    num_author_hashes: int = 2,
) -> TrajectoryDiversity:
    """Run a trajectory and track diversity at each step.

    Args:
        runner: FullKVCachedRunner
        batch: RecsysBatch
        embeddings: RecsysEmbeddings
        candidate_embeddings: [num_candidates, emb_dim] numpy array
        num_engagements: How many steps to simulate
        strategy: "top" (always pick top) or "random"
        rng: Random generator for random strategy
        use_real_reranking: If True, use actual model inference at each step.
            This is slower but reflects true model dynamics. Recommended when
            using trained weights. If False, uses perturbation-based simulation.
        num_item_hashes: Number of item hashes (needed for real re-ranking)
        num_author_hashes: Number of author hashes (needed for real re-ranking)
    """
    if use_real_reranking:
        return _run_real_trajectory_with_diversity(
            runner, batch, embeddings, candidate_embeddings,
            num_engagements, strategy, rng,
            num_item_hashes, num_author_hashes,
        )

    # Simulated mode (original behavior)
    simulator = TrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
    )
    simulator.initialize()

    diversity = TrajectoryDiversity()

    # Record initial diversity
    initial_snapshot = compute_snapshot(
        simulator.current_scores(),
        candidate_embeddings,
        step=0,
    )
    diversity.snapshots.append(initial_snapshot)

    # Run engagements
    for step in range(1, num_engagements + 1):
        if not simulator._remaining_candidate_indices:
            break

        # Choose engagement
        if strategy == "top":
            choice = 0
        else:  # random
            num_remaining = len(simulator._remaining_candidate_indices)
            choice = rng.integers(0, num_remaining) if rng else 0

        # Get engaged candidate embedding before engaging
        engaged_idx = simulator.current_scores()[choice].index
        diversity.engaged_embeddings.append(candidate_embeddings[engaged_idx])

        # Engage
        simulator.engage(choice)

        # Record diversity after engagement
        snapshot = compute_snapshot(
            simulator.current_scores(),
            candidate_embeddings,
            step=step,
        )
        diversity.snapshots.append(snapshot)

    return diversity


def _run_real_trajectory_with_diversity(
    runner,
    batch,
    embeddings,
    candidate_embeddings: np.ndarray,
    num_engagements: int,
    strategy: str,
    rng: np.random.Generator | None,
    num_item_hashes: int,
    num_author_hashes: int,
) -> TrajectoryDiversity:
    """Run trajectory with actual model re-ranking at each step.

    This uses RealTrajectorySimulator which runs actual model inference
    when a candidate is engaged, properly extending the history context.
    """
    simulator = RealTrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
        num_item_hashes=num_item_hashes,
        num_author_hashes=num_author_hashes,
    )
    simulator.initialize()

    diversity = TrajectoryDiversity()

    # Record initial diversity
    initial_snapshot = compute_snapshot(
        simulator.current_scores(),
        candidate_embeddings,
        step=0,
    )
    diversity.snapshots.append(initial_snapshot)

    # Run engagements
    for step in range(1, num_engagements + 1):
        if not simulator._remaining_indices:
            break

        # Choose engagement
        if strategy == "top":
            choice = 0
        else:  # random
            num_remaining = len(simulator._remaining_indices)
            choice = rng.integers(0, num_remaining) if rng else 0

        # Get engaged candidate embedding before engaging
        engaged_idx = simulator.current_scores()[choice].index
        diversity.engaged_embeddings.append(candidate_embeddings[engaged_idx])

        # Engage (this triggers actual model re-ranking)
        simulator.engage(choice)

        # Record diversity after engagement
        snapshot = compute_snapshot(
            simulator.current_scores(),
            candidate_embeddings,
            step=step,
        )
        diversity.snapshots.append(snapshot)

    return diversity


def analyze_diversity(
    runner,
    batch,
    embeddings,
    candidate_embeddings: np.ndarray,
    num_candidates: int,
    num_trajectories: int = 50,
    num_engagements: int = 10,
    strategy: str = "top",
    seed: int = 42,
    use_real_reranking: bool = False,
    num_item_hashes: int = 2,
    num_author_hashes: int = 2,
) -> DiversityAnalysisResult:
    """Run multiple trajectories and aggregate diversity metrics.

    Args:
        runner: FullKVCachedRunner
        batch: RecsysBatch
        embeddings: RecsysEmbeddings
        candidate_embeddings: [num_candidates, emb_dim] numpy array
        num_candidates: Total candidates available
        num_trajectories: How many trajectories to run
        num_engagements: Engagements per trajectory
        strategy: "top" or "random"
        seed: Random seed
        use_real_reranking: If True, use actual model inference at each step.
            Slower but accurate. Recommended with trained weights.
        num_item_hashes: Number of item hashes (for real re-ranking)
        num_author_hashes: Number of author hashes (for real re-ranking)
    """
    rng = np.random.default_rng(seed=seed)

    all_diversity_trends = []
    all_concentration_trends = []
    all_diversity_changes = []
    engagement_counts = np.zeros(num_candidates)

    for _ in range(num_trajectories):
        traj_div = run_trajectory_with_diversity(
            runner, batch, embeddings, candidate_embeddings,
            num_engagements=num_engagements,
            strategy=strategy,
            rng=rng,
            use_real_reranking=use_real_reranking,
            num_item_hashes=num_item_hashes,
            num_author_hashes=num_author_hashes,
        )

        all_diversity_trends.append(traj_div.diversity_trend())
        all_concentration_trends.append(traj_div.concentration_trend())
        all_diversity_changes.append(traj_div.diversity_change())

        # Track which candidates were engaged
        for emb in traj_div.engaged_embeddings:
            # Find matching candidate
            for i in range(num_candidates):
                if np.allclose(emb, candidate_embeddings[i]):
                    engagement_counts[i] += 1
                    break

    # Aggregate by step
    max_steps = max(len(t) for t in all_diversity_trends)
    mean_diversity_by_step = []
    std_diversity_by_step = []
    mean_concentration_by_step = []

    for step in range(max_steps):
        step_diversities = [t[step] for t in all_diversity_trends if step < len(t)]
        step_concentrations = [t[step] for t in all_concentration_trends if step < len(t)]

        mean_diversity_by_step.append(float(np.mean(step_diversities)))
        std_diversity_by_step.append(float(np.std(step_diversities)))
        mean_concentration_by_step.append(float(np.mean(step_concentrations)))

    # Compute baseline: diversity of full candidate set
    baseline_diversity = compute_pairwise_distances(candidate_embeddings)[0]

    # Coverage metrics
    unique_engaged = int(np.sum(engagement_counts > 0))
    gini = compute_gini_coefficient(engagement_counts)

    # Diversity ratio (final / baseline)
    final_diversity = mean_diversity_by_step[-1] if mean_diversity_by_step else 0
    diversity_ratio = final_diversity / baseline_diversity if baseline_diversity > 0 else 1.0

    return DiversityAnalysisResult(
        num_trajectories=num_trajectories,
        num_candidates=num_candidates,
        num_engagements=num_engagements,
        mean_diversity_by_step=mean_diversity_by_step,
        std_diversity_by_step=std_diversity_by_step,
        mean_concentration_by_step=mean_concentration_by_step,
        mean_diversity_change=float(np.mean(all_diversity_changes)),
        std_diversity_change=float(np.std(all_diversity_changes)),
        unique_engaged=unique_engaged,
        engagement_gini=gini,
        random_baseline_diversity=baseline_diversity,
        diversity_ratio=diversity_ratio,
    )


def visualize_diversity_comparison(
    top_result: DiversityAnalysisResult,
    random_result: DiversityAnalysisResult,
):
    """Visualize comparison between top-following and random strategies."""
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS COMPARISON")
    print("=" * 70)

    print("\nConfiguration:")
    print(f"  Candidates: {top_result.num_candidates}")
    print(f"  Trajectories: {top_result.num_trajectories}")
    print(f"  Engagements per trajectory: {top_result.num_engagements}")
    print(f"  Baseline diversity (full pool): {top_result.random_baseline_diversity:.4f}")

    # Diversity trend comparison
    print(f"\n{'DIVERSITY BY STEP (avg pairwise embedding distance)':^70}")
    print("-" * 70)
    print(f"{'Step':<6} {'Top Strategy':<20} {'Random Strategy':<20} {'Difference':<15}")
    print("-" * 70)

    max_steps = max(len(top_result.mean_diversity_by_step),
                    len(random_result.mean_diversity_by_step))

    for step in range(max_steps):
        top_div = top_result.mean_diversity_by_step[step] if step < len(top_result.mean_diversity_by_step) else float('nan')
        rand_div = random_result.mean_diversity_by_step[step] if step < len(random_result.mean_diversity_by_step) else float('nan')
        diff = top_div - rand_div if not (np.isnan(top_div) or np.isnan(rand_div)) else float('nan')

        top_std = top_result.std_diversity_by_step[step] if step < len(top_result.std_diversity_by_step) else 0
        rand_std = random_result.std_diversity_by_step[step] if step < len(random_result.std_diversity_by_step) else 0

        print(f"{step:<6} {top_div:>8.4f} ± {top_std:<8.4f} {rand_div:>8.4f} ± {rand_std:<8.4f} {diff:>+10.4f}")

    # Summary metrics
    print(f"\n{'SUMMARY METRICS':^70}")
    print("-" * 70)
    print(f"{'Metric':<40} {'Top':<15} {'Random':<15}")
    print("-" * 70)

    print(f"{'Diversity change (%)':<40} {top_result.mean_diversity_change:>+10.1f}%    {random_result.mean_diversity_change:>+10.1f}%")
    print(f"{'Final diversity ratio (vs baseline)':<40} {top_result.diversity_ratio:>10.3f}     {random_result.diversity_ratio:>10.3f}")
    print(f"{'Unique candidates engaged':<40} {top_result.unique_engaged:>10}     {random_result.unique_engaged:>10}")
    print(f"{'Engagement Gini (0=equal, 1=concentrated)':<40} {top_result.engagement_gini:>10.3f}     {random_result.engagement_gini:>10.3f}")

    # Interpretation
    print(f"\n{'INTERPRETATION':^70}")
    print("-" * 70)

    div_diff = top_result.mean_diversity_change - random_result.mean_diversity_change
    if div_diff < -10:
        print("⚠️  TOP STRATEGY shows STRONGER diversity reduction than random")
        print(f"   ({abs(div_diff):.1f}% more reduction)")
        print("   → Evidence of filter bubble / echo chamber effect")
    elif div_diff < -5:
        print("⚡ TOP STRATEGY shows MODERATE diversity reduction vs random")
        print(f"   ({abs(div_diff):.1f}% more reduction)")
        print("   → Some narrowing when following recommendations")
    else:
        print("✓  TOP STRATEGY diversity similar to random")
        print("   → No strong evidence of filter bubble")

    gini_diff = top_result.engagement_gini - random_result.engagement_gini
    if gini_diff > 0.1:
        print(f"\n⚠️  ENGAGEMENT CONCENTRATION: Top strategy Gini {gini_diff:.3f} higher")
        print("   → Recommendations concentrate on fewer candidates")

    coverage_ratio = top_result.unique_engaged / random_result.unique_engaged if random_result.unique_engaged > 0 else 1
    if coverage_ratio < 0.7:
        print(f"\n⚠️  COVERAGE GAP: Top strategy engages {(1-coverage_ratio)*100:.0f}% fewer unique candidates")


def create_test_config(candidate_seq_len: int = 8):
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
        candidate_seq_len=candidate_seq_len,
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


def run_analysis(
    num_candidates: int = 8,
    num_trajectories: int = 50,
    num_engagements: int | None = None,
    use_real_reranking: bool = False,
):
    """Run the full diversity analysis.

    Args:
        num_candidates: Number of candidates to rank
        num_trajectories: Number of trajectories to simulate
        num_engagements: Engagements per trajectory (default: scales with candidates)
        use_real_reranking: If True, use actual model inference at each step.
            This is slower but reflects true model dynamics. With random weights,
            all candidates will have similar scores. Use with trained weights
            to observe real filter bubble dynamics.
    """
    # Default engagements scales with candidates
    if num_engagements is None:
        num_engagements = min(num_candidates - 1, max(5, num_candidates // 5))

    mode_str = "REAL RE-RANKING" if use_real_reranking else "SIMULATED"
    print("=" * 70)
    print(f"DIVERSITY METRICS ANALYSIS ({mode_str})")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Candidates: {num_candidates}")
    print(f"  Trajectories: {num_trajectories}")
    print(f"  Engagements per trajectory: {num_engagements}")
    print(f"  Mode: {mode_str}")
    if use_real_reranking:
        print("  Note: Real re-ranking uses actual model inference at each step.")
        print("        With random weights, expect uniform scores (~0.5).")
        print("        Use trained weights to observe real dynamics.")

    print("\nInitializing model...")
    config = create_test_config(candidate_seq_len=num_candidates)
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, embeddings = create_example_batch(
        batch_size=1,
        emb_size=config.emb_size,
        history_len=config.history_seq_len,
        num_candidates=num_candidates,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )

    # Extract candidate embeddings as numpy array
    # candidate_post_embeddings has shape [B, C, num_item_hashes, D]
    # We average over the hash dimension to get [B, C, D], then take first batch
    raw_cand_emb = np.array(embeddings.candidate_post_embeddings[0])  # [C, num_hashes, D]
    candidate_embeddings = np.mean(raw_cand_emb, axis=1)  # [C, D]

    print("Model initialized")
    print(f"Candidate embedding shape: {candidate_embeddings.shape}")

    # Run top-following strategy
    print(f"\nRunning {num_trajectories} TOP-FOLLOWING trajectories...")
    top_result = analyze_diversity(
        runner, batch, embeddings, candidate_embeddings,
        num_candidates=num_candidates,
        num_trajectories=num_trajectories,
        num_engagements=num_engagements,
        strategy="top",
        seed=42,
        use_real_reranking=use_real_reranking,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
    )

    # Run random strategy
    print(f"Running {num_trajectories} RANDOM trajectories...")
    random_result = analyze_diversity(
        runner, batch, embeddings, candidate_embeddings,
        num_candidates=num_candidates,
        num_trajectories=num_trajectories,
        num_engagements=num_engagements,
        strategy="random",
        seed=123,
        use_real_reranking=use_real_reranking,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
    )

    # Visualize comparison
    visualize_diversity_comparison(top_result, random_result)

    # Confidence interval info
    print(f"\n{'CONFIDENCE INTERVAL NOTES':^70}")
    print("-" * 70)
    print(f"With {num_candidates} candidates and {num_trajectories} trajectories:")

    # Standard error of mean diversity change
    se_top = top_result.std_diversity_change / np.sqrt(num_trajectories)
    se_rand = random_result.std_diversity_change / np.sqrt(num_trajectories)

    print(f"  Top strategy diversity change: {top_result.mean_diversity_change:+.1f}% ± {1.96*se_top:.1f}% (95% CI)")
    print(f"  Random strategy diversity change: {random_result.mean_diversity_change:+.1f}% ± {1.96*se_rand:.1f}% (95% CI)")

    # Effect size
    pooled_std = np.sqrt((top_result.std_diversity_change**2 + random_result.std_diversity_change**2) / 2)
    if pooled_std > 0:
        cohens_d = (top_result.mean_diversity_change - random_result.mean_diversity_change) / pooled_std
        print(f"\n  Effect size (Cohen's d): {cohens_d:.2f}")
        if abs(cohens_d) < 0.2:
            print("    → Small effect")
        elif abs(cohens_d) < 0.8:
            print("    → Medium effect")
        else:
            print("    → Large effect")

    return top_result, random_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run diversity metrics analysis")
    parser.add_argument("--candidates", type=int, default=8, help="Number of candidates")
    parser.add_argument("--trajectories", type=int, default=50, help="Number of trajectories")
    parser.add_argument("--engagements", type=int, default=None, help="Engagements per trajectory")
    parser.add_argument(
        "--real", action="store_true",
        help="Use real model re-ranking instead of simulated perturbations. "
             "Slower but accurate. Recommended with trained weights."
    )
    args = parser.parse_args()

    run_analysis(
        num_candidates=args.candidates,
        num_trajectories=args.trajectories,
        num_engagements=args.engagements,
        use_real_reranking=args.real,
    )
