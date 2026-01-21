"""Real Trajectory Simulation with Actual Model Re-ranking.

Unlike the simulated trajectory (which uses perturbation-based score updates),
this module runs actual model inference at each step to see how rankings
truly evolve as engagement history grows.

The key insight: when a user engages with candidate X, that candidate becomes
part of their history context. Future candidates are scored against this
extended history, which may cause ranking shifts that reflect the model's
actual attention dynamics.

Usage:
    uv run python enhancements/analysis/real_trajectory_simulation.py
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner


class CandidateScore(NamedTuple):
    """Score for a single candidate."""
    index: int  # Original candidate index (in initial pool)
    score: float  # Ranking score
    rank: int  # Rank among remaining candidates (1 = top)


class RealTrajectoryStep(NamedTuple):
    """Record of a single step with actual model scores."""
    step_num: int
    engaged_candidate_idx: Optional[int]
    engaged_candidate_score: Optional[float]
    remaining_scores: List[CandidateScore]
    history_length: int  # Current history length after engagement


class RealTrajectoryPath(NamedTuple):
    """Complete trajectory with actual model re-ranking."""
    steps: List[RealTrajectoryStep]
    engagement_sequence: List[int]


def create_modified_batch(
    original_batch: RecsysBatch,
    original_embeddings: RecsysEmbeddings,
    engaged_indices: List[int],
    remaining_indices: List[int],
    num_item_hashes: int,
    num_author_hashes: int,
) -> Tuple[RecsysBatch, RecsysEmbeddings]:
    """Create a new batch with engaged candidates added to history.

    When a user engages with a candidate, that candidate becomes part of their
    history for future recommendations. This function:
    1. Appends engaged candidate hashes/embeddings to history
    2. Creates a new candidate set with only remaining candidates

    Args:
        original_batch: Original batch
        original_embeddings: Original embeddings
        engaged_indices: Indices of candidates that have been engaged (in order)
        remaining_indices: Indices of candidates still available
        num_item_hashes: Number of item hashes per candidate
        num_author_hashes: Number of author hashes per candidate

    Returns:
        Modified (batch, embeddings) with extended history
    """
    # Extract dimensions
    B = original_batch.user_hashes.shape[0]
    orig_history_len = original_batch.history_post_hashes.shape[1]
    num_engaged = len(engaged_indices)
    num_remaining = len(remaining_indices)

    # New history = original history + engaged candidates
    new_history_len = orig_history_len + num_engaged

    # === Build new history hashes ===
    # Original history hashes
    orig_history_post = np.array(original_batch.history_post_hashes)  # [B, H, num_item_hashes]
    orig_history_author = np.array(original_batch.history_author_hashes)  # [B, H, num_author_hashes]

    # Engaged candidate hashes (to append to history)
    engaged_post = np.array(original_batch.candidate_post_hashes)[:, engaged_indices, :]  # [B, E, num_item_hashes]
    engaged_author = np.array(original_batch.candidate_author_hashes)[:, engaged_indices, :]  # [B, E, num_author_hashes]

    # Concatenate to form new history
    new_history_post = np.concatenate([orig_history_post, engaged_post], axis=1)
    new_history_author = np.concatenate([orig_history_author, engaged_author], axis=1)

    # === Build new history product surface ===
    orig_history_ps = np.array(original_batch.history_product_surface)  # [B, H]
    engaged_ps = np.array(original_batch.candidate_product_surface)[:, engaged_indices]  # [B, E]
    new_history_ps = np.concatenate([orig_history_ps, engaged_ps], axis=1)

    # === Build new history actions ===
    # For engaged candidates, we create a "positive engagement" action vector
    # Using action 0 (favorite) as the default engagement signal
    orig_history_actions = np.array(original_batch.history_actions)  # [B, H, num_actions]
    num_actions = orig_history_actions.shape[-1]

    # Create engagement action: set first action (favorite) to 1
    engaged_actions = np.zeros((B, num_engaged, num_actions), dtype=orig_history_actions.dtype)
    engaged_actions[:, :, 0] = 1  # Mark as favorited

    new_history_actions = np.concatenate([orig_history_actions, engaged_actions], axis=1)

    # === Build new candidate hashes (remaining only) ===
    new_cand_post = np.array(original_batch.candidate_post_hashes)[:, remaining_indices, :]
    new_cand_author = np.array(original_batch.candidate_author_hashes)[:, remaining_indices, :]
    new_cand_ps = np.array(original_batch.candidate_product_surface)[:, remaining_indices]

    # === Build new batch ===
    new_batch = RecsysBatch(
        user_hashes=original_batch.user_hashes,
        history_post_hashes=new_history_post,
        history_author_hashes=new_history_author,
        history_product_surface=new_history_ps,
        history_actions=new_history_actions,
        candidate_post_hashes=new_cand_post,
        candidate_author_hashes=new_cand_author,
        candidate_product_surface=new_cand_ps,
    )

    # === Build new embeddings ===
    # Original history embeddings
    orig_hist_post_emb = np.array(original_embeddings.history_post_embeddings)  # [B, H, num_item_hashes, D]
    orig_hist_author_emb = np.array(original_embeddings.history_author_embeddings)  # [B, H, num_author_hashes, D]

    # Engaged candidate embeddings (to append to history)
    engaged_post_emb = np.array(original_embeddings.candidate_post_embeddings)[:, engaged_indices, :, :]
    engaged_author_emb = np.array(original_embeddings.candidate_author_embeddings)[:, engaged_indices, :, :]

    # Concatenate
    new_hist_post_emb = np.concatenate([orig_hist_post_emb, engaged_post_emb], axis=1)
    new_hist_author_emb = np.concatenate([orig_hist_author_emb, engaged_author_emb], axis=1)

    # Remaining candidate embeddings
    new_cand_post_emb = np.array(original_embeddings.candidate_post_embeddings)[:, remaining_indices, :, :]
    new_cand_author_emb = np.array(original_embeddings.candidate_author_embeddings)[:, remaining_indices, :, :]

    new_embeddings = RecsysEmbeddings(
        user_embeddings=original_embeddings.user_embeddings,
        history_post_embeddings=new_hist_post_emb,
        history_author_embeddings=new_hist_author_emb,
        candidate_post_embeddings=new_cand_post_emb,
        candidate_author_embeddings=new_cand_author_emb,
    )

    return new_batch, new_embeddings


@dataclass
class RealTrajectorySimulator:
    """Simulate trajectories with actual model re-ranking.

    Unlike the perturbation-based TrajectorySimulator, this class runs
    actual model inference at each step. When a candidate is engaged:
    1. It's added to the user's history
    2. A new batch is created with remaining candidates
    3. The model re-scores remaining candidates against extended history

    This reveals the true dynamics of how the model's rankings shift.
    """
    runner: FullKVCachedRunner
    initial_batch: RecsysBatch
    initial_embeddings: RecsysEmbeddings
    num_item_hashes: int
    num_author_hashes: int

    # State
    _current_batch: Optional[RecsysBatch] = field(default=None, init=False)
    _current_embeddings: Optional[RecsysEmbeddings] = field(default=None, init=False)
    _engaged_indices: List[int] = field(default_factory=list, init=False)
    _remaining_indices: List[int] = field(default_factory=list, init=False)
    _trajectory_steps: List[RealTrajectoryStep] = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)
    _index_mapping: dict = field(default_factory=dict, init=False)  # Maps current position to original index

    def initialize(self):
        """Initialize with first forward pass."""
        if self._initialized:
            return

        # Start with original batch
        self._current_batch = self.initial_batch
        self._current_embeddings = self.initial_embeddings

        # Get number of candidates
        num_candidates = self.initial_batch.candidate_post_hashes.shape[1]
        self._remaining_indices = list(range(num_candidates))
        self._engaged_indices = []

        # Initialize index mapping (position -> original index)
        self._index_mapping = {i: i for i in range(num_candidates)}

        # Run initial forward pass
        self.runner.clear_cache()
        output = self.runner.rank(self._current_batch, self._current_embeddings, use_cache=False)

        # Record initial state
        scores = self._extract_scores(output, self._remaining_indices)
        initial_step = RealTrajectoryStep(
            step_num=0,
            engaged_candidate_idx=None,
            engaged_candidate_score=None,
            remaining_scores=scores,
            history_length=self.initial_batch.history_post_hashes.shape[1],
        )
        self._trajectory_steps.append(initial_step)
        self._initialized = True

    def _extract_scores(self, output, original_indices: List[int]) -> List[CandidateScore]:
        """Extract and rank scores from model output."""
        # output.scores: [batch, num_candidates, num_actions]
        # Aggregate to single score (mean across actions)
        raw_scores = np.array(output.scores[0])  # [num_candidates, num_actions]
        agg_scores = np.mean(raw_scores, axis=-1)  # [num_candidates]

        # Create (original_index, score) pairs
        scored = [(orig_idx, float(agg_scores[pos]))
                  for pos, orig_idx in enumerate(original_indices)]

        # Sort by score descending
        scored.sort(key=lambda x: -x[1])

        # Create CandidateScore objects with ranks
        return [
            CandidateScore(index=idx, score=score, rank=rank + 1)
            for rank, (idx, score) in enumerate(scored)
        ]

    def current_scores(self) -> List[CandidateScore]:
        """Get current ranking of remaining candidates."""
        if not self._initialized:
            self.initialize()
        return self._trajectory_steps[-1].remaining_scores

    def engage(self, rank: int = 0) -> RealTrajectoryStep:
        """Engage with candidate at given rank, re-rank remaining with actual model.

        Args:
            rank: Which candidate to engage with (0 = top-ranked)

        Returns:
            RealTrajectoryStep with new rankings from actual model inference
        """
        if not self._initialized:
            self.initialize()

        if not self._remaining_indices:
            raise ValueError("No candidates remaining")

        if rank >= len(self._remaining_indices):
            raise ValueError(f"Rank {rank} out of range")

        # Get the candidate to engage with (by current ranking)
        current_scores = self._trajectory_steps[-1].remaining_scores
        engaged = current_scores[rank]
        engaged_original_idx = engaged.index

        # Update tracking
        self._engaged_indices.append(engaged_original_idx)
        self._remaining_indices.remove(engaged_original_idx)

        # If no candidates remaining, record final state
        if not self._remaining_indices:
            final_step = RealTrajectoryStep(
                step_num=len(self._trajectory_steps),
                engaged_candidate_idx=engaged_original_idx,
                engaged_candidate_score=engaged.score,
                remaining_scores=[],
                history_length=self.initial_batch.history_post_hashes.shape[1] + len(self._engaged_indices),
            )
            self._trajectory_steps.append(final_step)
            return final_step

        # Create new batch with engaged candidate added to history
        new_batch, new_embeddings = create_modified_batch(
            self.initial_batch,
            self.initial_embeddings,
            self._engaged_indices,
            self._remaining_indices,
            self.num_item_hashes,
            self.num_author_hashes,
        )

        self._current_batch = new_batch
        self._current_embeddings = new_embeddings

        # Run actual model inference
        self.runner.clear_cache()
        output = self.runner.rank(new_batch, new_embeddings, use_cache=False)

        # Extract new scores
        new_scores = self._extract_scores(output, self._remaining_indices)

        new_step = RealTrajectoryStep(
            step_num=len(self._trajectory_steps),
            engaged_candidate_idx=engaged_original_idx,
            engaged_candidate_score=engaged.score,
            remaining_scores=new_scores,
            history_length=self.initial_batch.history_post_hashes.shape[1] + len(self._engaged_indices),
        )
        self._trajectory_steps.append(new_step)

        return new_step

    def engage_top_n(self, n: int) -> List[RealTrajectoryStep]:
        """Engage with top candidate n times."""
        steps = []
        for _ in range(n):
            if not self._remaining_indices:
                break
            step = self.engage(0)
            steps.append(step)
        return steps

    def get_trajectory(self) -> RealTrajectoryPath:
        """Get the complete trajectory."""
        return RealTrajectoryPath(
            steps=list(self._trajectory_steps),
            engagement_sequence=list(self._engaged_indices),
        )

    def reset(self):
        """Reset to initial state."""
        self._current_batch = None
        self._current_embeddings = None
        self._engaged_indices = []
        self._remaining_indices = []
        self._trajectory_steps = []
        self._index_mapping = {}
        self._initialized = False


def compare_real_vs_simulated(
    runner: FullKVCachedRunner,
    batch: RecsysBatch,
    embeddings: RecsysEmbeddings,
    num_item_hashes: int,
    num_author_hashes: int,
    num_engagements: int = 5,
) -> dict:
    """Compare real model re-ranking vs simulated perturbation.

    Returns dict with comparison metrics.
    """
    from enhancements.analysis.trajectory_simulation import TrajectorySimulator

    # Run real trajectory
    real_sim = RealTrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
        num_item_hashes=num_item_hashes,
        num_author_hashes=num_author_hashes,
    )
    real_sim.initialize()
    real_sim.engage_top_n(num_engagements)
    real_traj = real_sim.get_trajectory()

    # Run simulated trajectory
    sim_sim = TrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
    )
    sim_sim.initialize()
    sim_sim.engage_top_n(num_engagements)
    sim_traj = sim_sim.get_trajectory()

    # Compare engagement sequences
    same_sequence = real_traj.engagement_sequence == sim_traj.engagement_sequence

    # Compare rankings at each step
    rank_correlations = []
    for step_idx in range(min(len(real_traj.steps), len(sim_traj.steps))):
        real_step = real_traj.steps[step_idx]
        sim_step = sim_traj.steps[step_idx]

        if real_step.remaining_scores and sim_step.remaining_scores:
            # Get ranking order
            real_order = [cs.index for cs in real_step.remaining_scores]
            sim_order = [cs.index for cs in sim_step.remaining_scores]

            # Compute rank correlation (simple: count matching positions)
            matches = sum(1 for r, s in zip(real_order, sim_order) if r == s)
            correlation = matches / len(real_order) if real_order else 0
            rank_correlations.append(correlation)

    return {
        'real_sequence': real_traj.engagement_sequence,
        'simulated_sequence': sim_traj.engagement_sequence,
        'same_sequence': same_sequence,
        'rank_correlations': rank_correlations,
        'avg_rank_correlation': np.mean(rank_correlations) if rank_correlations else 0,
    }


def visualize_real_trajectory(trajectory: RealTrajectoryPath):
    """Visualize a real trajectory."""
    print("\n" + "=" * 70)
    print("REAL TRAJECTORY (Actual Model Re-ranking)")
    print("=" * 70)

    print(f"\nEngagement sequence: {' → '.join(f'C{i}' for i in trajectory.engagement_sequence)}")

    for step in trajectory.steps:
        print(f"\n--- Step {step.step_num} ---")
        if step.engaged_candidate_idx is not None:
            print(f"Engaged: C{step.engaged_candidate_idx} (score: {step.engaged_candidate_score:.4f})")
        else:
            print("Initial state")
        print(f"History length: {step.history_length}")

        if step.remaining_scores:
            print(f"Remaining ({len(step.remaining_scores)}):")
            for cs in step.remaining_scores[:5]:
                print(f"  Rank {cs.rank}: C{cs.index} (score: {cs.score:.4f})")
            if len(step.remaining_scores) > 5:
                print(f"  ... and {len(step.remaining_scores) - 5} more")


def analyze_ranking_shifts(trajectory: RealTrajectoryPath):
    """Analyze how rankings shift due to actual model dynamics."""
    print("\n" + "=" * 70)
    print("RANKING SHIFT ANALYSIS")
    print("=" * 70)

    if len(trajectory.steps) < 2:
        print("Not enough steps to analyze shifts")
        return

    print(f"\n{'Step':<6} {'Engaged':<10} {'Top Changed?':<15} {'Rank Changes':<20}")
    print("-" * 55)

    for i in range(1, len(trajectory.steps)):
        prev_step = trajectory.steps[i - 1]
        curr_step = trajectory.steps[i]

        engaged = curr_step.engaged_candidate_idx

        if not prev_step.remaining_scores or not curr_step.remaining_scores:
            continue

        # Check if top candidate changed (excluding the engaged one)
        prev_remaining = [cs for cs in prev_step.remaining_scores if cs.index != engaged]
        if prev_remaining and curr_step.remaining_scores:
            prev_top = prev_remaining[0].index
            curr_top = curr_step.remaining_scores[0].index
            top_changed = "Yes" if prev_top != curr_top else "No"
        else:
            top_changed = "N/A"

        # Count rank changes
        prev_ranks = {cs.index: cs.rank for cs in prev_step.remaining_scores if cs.index != engaged}
        curr_ranks = {cs.index: cs.rank for cs in curr_step.remaining_scores}

        common = set(prev_ranks.keys()) & set(curr_ranks.keys())
        if common:
            # Adjust prev ranks (account for removed candidate)
            changes = []
            for idx in common:
                # How many higher-ranked candidates were removed?
                removed_above = sum(1 for cs in prev_step.remaining_scores
                                   if cs.index == engaged and cs.rank < prev_ranks[idx])
                adjusted_prev = prev_ranks[idx] - removed_above
                change = curr_ranks[idx] - adjusted_prev
                if change != 0:
                    changes.append((idx, change))

            if changes:
                change_str = ", ".join(f"C{idx}:{c:+d}" for idx, c in changes[:3])
                if len(changes) > 3:
                    change_str += f" (+{len(changes)-3} more)"
            else:
                change_str = "None"
        else:
            change_str = "N/A"

        print(f"{i:<6} C{engaged:<9} {top_changed:<15} {change_str:<20}")


def create_test_config(candidate_seq_len: int = 8):
    """Create test configuration."""
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


def run_analysis(num_candidates: int = 8, num_engagements: int = 5):
    """Run the real trajectory analysis."""
    print("=" * 70)
    print("REAL TRAJECTORY SIMULATION")
    print("=" * 70)
    print("\nThis analysis uses ACTUAL MODEL INFERENCE at each step,")
    print("not simulated perturbations. Rankings reflect true model dynamics.")

    print(f"\nConfiguration:")
    print(f"  Candidates: {num_candidates}")
    print(f"  Engagements: {num_engagements}")

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

    print("Model initialized")

    # Run real trajectory
    print("\nRunning real trajectory simulation...")
    simulator = RealTrajectorySimulator(
        runner=runner,
        initial_batch=batch,
        initial_embeddings=embeddings,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
    )
    simulator.initialize()
    simulator.engage_top_n(num_engagements)
    trajectory = simulator.get_trajectory()

    # Visualize
    visualize_real_trajectory(trajectory)
    analyze_ranking_shifts(trajectory)

    # Compare to simulated
    print("\n" + "=" * 70)
    print("REAL vs SIMULATED COMPARISON")
    print("=" * 70)

    comparison = compare_real_vs_simulated(
        runner, batch, embeddings,
        config.hash_config.num_item_hashes,
        config.hash_config.num_author_hashes,
        num_engagements,
    )

    print(f"\nReal engagement sequence:      {' → '.join(f'C{i}' for i in comparison['real_sequence'])}")
    print(f"Simulated engagement sequence: {' → '.join(f'C{i}' for i in comparison['simulated_sequence'])}")
    print(f"Sequences match: {comparison['same_sequence']}")
    print(f"\nRank correlation by step: {[f'{c:.2f}' for c in comparison['rank_correlations']]}")
    print(f"Average rank correlation: {comparison['avg_rank_correlation']:.2f}")

    if comparison['avg_rank_correlation'] < 0.5:
        print("\n⚠️  Low correlation suggests simulated dynamics don't match real model behavior")
    elif comparison['avg_rank_correlation'] < 0.8:
        print("\n⚡ Moderate correlation - simulation captures some but not all dynamics")
    else:
        print("\n✓  High correlation - simulation reasonably approximates real dynamics")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--engagements", type=int, default=5)
    args = parser.parse_args()

    run_analysis(num_candidates=args.candidates, num_engagements=args.engagements)
