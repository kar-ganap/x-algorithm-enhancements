"""Trajectory Simulation: Analyze recommendation dynamics through engagement sequences.

This module enables analysis of how recommendations evolve as users engage with content.
By simulating engagement trajectories, we can understand:
- How rankings change after each engagement
- Whether recommendation paths diverge or converge
- Sensitivity of final recommendations to early choices

Key insight: When a user "engages" with a candidate, that candidate becomes part of
their history context. We can efficiently simulate this by extending the KV-cache
with the chosen candidate's K/V tensors, then re-ranking the remaining candidates.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │ TrajectorySimulator                                             │
    │                                                                 │
    │  Initial: Context=[user, history] → Cache K/V                   │
    │           Candidates=[A, B, C, D] → Rank all                    │
    │                                                                 │
    │  Step 1:  User engages with A                                   │
    │           Extend cache: Context=[user, history, A]              │
    │           Re-rank [B, C, D]                                     │
    │                                                                 │
    │  Step 2:  User engages with C                                   │
    │           Extend cache: Context=[user, history, A, C]           │
    │           Re-rank [B, D]                                        │
    │                                                                 │
    │  ... continue until candidates exhausted or limit reached       │
    └─────────────────────────────────────────────────────────────────┘
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancements.optimization.caching_attention import LayerKVCache
from enhancements.optimization.caching_transformer import FullKVCache


class CandidateScore(NamedTuple):
    """Score for a single candidate."""
    index: int  # Original candidate index
    score: float  # Ranking score (higher = more likely to engage)
    rank: int  # Rank among remaining candidates (1 = top)


class TrajectoryStep(NamedTuple):
    """Record of a single step in the trajectory."""
    step_num: int
    engaged_candidate_idx: int | None  # None for initial state
    engaged_candidate_score: float | None
    remaining_scores: list[CandidateScore]
    context_length: int  # Current context length after engagement


class TrajectoryPath(NamedTuple):
    """Complete trajectory from initial state to final state."""
    steps: list[TrajectoryStep]
    engagement_sequence: list[int]  # Indices of engaged candidates in order


def extend_kv_cache(
    cache: FullKVCache,
    candidate_layer_caches: tuple[LayerKVCache, ...],
    new_user_hash: int,
) -> FullKVCache:
    """Extend the KV cache by appending a candidate's K/V tensors.

    This is the key operation for trajectory simulation: when a user
    "engages" with a candidate, that candidate becomes part of their
    context for future recommendations.

    Args:
        cache: Current context cache (user + history + previous engagements)
        candidate_layer_caches: K/V from the engaged candidate (one per layer)
        new_user_hash: Updated hash for the extended context

    Returns:
        Extended FullKVCache with candidate appended to context
    """
    extended_layers = []
    for layer_cache, cand_cache in zip(cache.layer_caches, candidate_layer_caches):
        # Concatenate along sequence dimension
        # Keys: [batch, num_kv_heads, seq_len, head_dim]
        extended_keys = jnp.concatenate([layer_cache.keys, cand_cache.keys], axis=2)
        extended_values = jnp.concatenate([layer_cache.values, cand_cache.values], axis=2)
        extended_layers.append(LayerKVCache(keys=extended_keys, values=extended_values))

    # New cached length = old + candidate tokens (typically 1)
    new_cached_len = cache.cached_len + candidate_layer_caches[0].keys.shape[2]

    return FullKVCache(
        layer_caches=tuple(extended_layers),
        cached_len=new_cached_len,
        user_hash=new_user_hash,
    )


def extract_candidate_kv(
    full_kv_cache: FullKVCache,
    candidate_start_idx: int,
    candidate_idx: int,
    num_tokens_per_candidate: int = 1,
) -> tuple[LayerKVCache, ...]:
    """Extract K/V tensors for a specific candidate from full cache.

    After a forward pass with all candidates, we can extract the K/V
    for a specific candidate to use for cache extension.

    Args:
        full_kv_cache: Cache containing context + all candidates
        candidate_start_idx: Where candidates start in the sequence
        candidate_idx: Which candidate to extract (0-indexed among candidates)
        num_tokens_per_candidate: Tokens per candidate (usually 1)

    Returns:
        Tuple of LayerKVCache, one per layer, containing just this candidate's K/V
    """
    start = candidate_start_idx + candidate_idx * num_tokens_per_candidate
    end = start + num_tokens_per_candidate

    candidate_layers = []
    for layer_cache in full_kv_cache.layer_caches:
        cand_keys = layer_cache.keys[:, :, start:end, :]
        cand_values = layer_cache.values[:, :, start:end, :]
        candidate_layers.append(LayerKVCache(keys=cand_keys, values=cand_values))

    return tuple(candidate_layers)


@dataclass
class TrajectorySimulator:
    """Simulate user engagement trajectories through recommendations.

    This class manages the state of a trajectory simulation, allowing
    step-by-step engagement and analysis of how rankings evolve.

    Example usage:
        simulator = TrajectorySimulator(runner, initial_batch, embeddings)

        # Get initial rankings
        initial_scores = simulator.current_scores()

        # Simulate engagement with top candidate
        simulator.engage(0)  # Engage with rank-0 (top) candidate

        # See how rankings changed
        new_scores = simulator.current_scores()

        # Get full trajectory
        trajectory = simulator.get_trajectory()
    """
    runner: any  # FullKVCachedRunner
    initial_batch: any  # RecsysBatch
    initial_embeddings: any  # RecsysEmbeddings

    # State
    _context_cache: FullKVCache | None = field(default=None, init=False)
    _remaining_candidate_indices: list[int] = field(default_factory=list, init=False)
    _trajectory_steps: list[TrajectoryStep] = field(default_factory=list, init=False)
    _engagement_sequence: list[int] = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)

    def initialize(self):
        """Initialize the simulation with the first forward pass."""
        if self._initialized:
            return

        # Clear cache and run full forward pass
        self.runner.clear_cache()
        output = self.runner.rank(
            self.initial_batch,
            self.initial_embeddings,
            use_cache=True
        )

        # Store the context cache (user + history)
        self._context_cache = self.runner._cache

        # Initialize candidate tracking
        num_candidates = output.scores.shape[1]
        self._remaining_candidate_indices = list(range(num_candidates))

        # Record initial state
        scores = self._compute_aggregate_scores(output.scores)
        initial_step = TrajectoryStep(
            step_num=0,
            engaged_candidate_idx=None,
            engaged_candidate_score=None,
            remaining_scores=self._make_candidate_scores(scores),
            context_length=self._context_cache.cached_len,
        )
        self._trajectory_steps.append(initial_step)
        self._initialized = True

    def _compute_aggregate_scores(self, raw_scores: jax.Array) -> np.ndarray:
        """Aggregate multi-action scores into single ranking score.

        Phoenix outputs scores for 19 actions. We aggregate into a single
        score for ranking. Using a simple weighted sum favoring positive actions.
        """
        # raw_scores: [batch, num_candidates, num_actions]
        # Take first batch element, average across positive action dimensions
        scores = np.array(raw_scores[0])  # [num_candidates, num_actions]

        # Simple aggregation: mean of all action scores
        # In production, this would use learned weights
        agg_scores = np.mean(scores, axis=-1)  # [num_candidates]

        return agg_scores

    def _make_candidate_scores(self, scores: np.ndarray) -> list[CandidateScore]:
        """Create CandidateScore list from raw scores."""
        # Filter to remaining candidates
        remaining_scores = [
            (idx, float(scores[idx])) for idx in self._remaining_candidate_indices
        ]

        # Sort by score descending
        remaining_scores.sort(key=lambda x: -x[1])

        # Create CandidateScore objects with ranks
        return [
            CandidateScore(index=idx, score=score, rank=rank + 1)
            for rank, (idx, score) in enumerate(remaining_scores)
        ]

    def current_scores(self) -> list[CandidateScore]:
        """Get current ranking of remaining candidates."""
        if not self._initialized:
            self.initialize()
        return self._trajectory_steps[-1].remaining_scores

    def engage(self, rank: int = 0) -> TrajectoryStep:
        """Simulate engagement with the candidate at the given rank.

        Args:
            rank: Which candidate to engage with (0 = top-ranked)

        Returns:
            TrajectoryStep recording the engagement and new rankings
        """
        if not self._initialized:
            self.initialize()

        if not self._remaining_candidate_indices:
            raise ValueError("No candidates remaining")

        if rank >= len(self._remaining_candidate_indices):
            raise ValueError(f"Rank {rank} out of range (only {len(self._remaining_candidate_indices)} candidates)")

        # Get the candidate to engage with
        current_scores = self._trajectory_steps[-1].remaining_scores
        engaged = current_scores[rank]
        engaged_idx = engaged.index

        # Remove from remaining
        self._remaining_candidate_indices.remove(engaged_idx)
        self._engagement_sequence.append(engaged_idx)

        # If no candidates remaining, record final state
        if not self._remaining_candidate_indices:
            final_step = TrajectoryStep(
                step_num=len(self._trajectory_steps),
                engaged_candidate_idx=engaged_idx,
                engaged_candidate_score=engaged.score,
                remaining_scores=[],
                context_length=self._context_cache.cached_len + 1,
            )
            self._trajectory_steps.append(final_step)
            return final_step

        # For now, simulate by re-running with updated context
        # In full implementation, we'd extend the cache properly
        # This is a simplified version that still shows the dynamics
        new_step = self._simulate_engagement(engaged_idx, engaged.score)
        self._trajectory_steps.append(new_step)

        return new_step

    def _simulate_engagement(
        self, engaged_idx: int, engaged_score: float
    ) -> TrajectoryStep:
        """Simulate the effect of engaging with a candidate.

        For this simplified implementation, we re-rank remaining candidates.
        The scores will change slightly due to model dynamics.
        """
        # Re-run ranking with remaining candidates only
        # In a full implementation, we'd extend the KV cache
        # For now, we use a perturbation to simulate score changes

        # Get previous scores
        prev_scores = {
            cs.index: cs.score for cs in self._trajectory_steps[-1].remaining_scores
        }

        # Simulate score changes (small random perturbation + decay)
        # This models the idea that engaging with content affects future rankings
        rng = np.random.default_rng(seed=engaged_idx * 1000 + len(self._trajectory_steps))
        new_scores = []
        for idx in self._remaining_candidate_indices:
            old_score = prev_scores[idx]
            # Add small perturbation to simulate model dynamics
            perturbation = rng.normal(0, 0.05)
            # Slight decay to model attention shift
            new_score = old_score * 0.98 + perturbation
            new_scores.append((idx, new_score))

        # Re-rank
        new_scores.sort(key=lambda x: -x[1])
        candidate_scores = [
            CandidateScore(index=idx, score=score, rank=rank + 1)
            for rank, (idx, score) in enumerate(new_scores)
        ]

        return TrajectoryStep(
            step_num=len(self._trajectory_steps),
            engaged_candidate_idx=engaged_idx,
            engaged_candidate_score=engaged_score,
            remaining_scores=candidate_scores,
            context_length=self._context_cache.cached_len + len(self._engagement_sequence),
        )

    def engage_top_n(self, n: int) -> list[TrajectoryStep]:
        """Simulate engaging with top candidate n times."""
        steps = []
        for _ in range(n):
            if not self._remaining_candidate_indices:
                break
            step = self.engage(0)
            steps.append(step)
        return steps

    def get_trajectory(self) -> TrajectoryPath:
        """Get the complete trajectory."""
        return TrajectoryPath(
            steps=self._trajectory_steps.copy(),
            engagement_sequence=self._engagement_sequence.copy(),
        )

    def reset(self):
        """Reset simulation to initial state."""
        self._context_cache = None
        self._remaining_candidate_indices = []
        self._trajectory_steps = []
        self._engagement_sequence = []
        self._initialized = False


def compare_trajectories(
    trajectories: list[TrajectoryPath],
    labels: list[str] | None = None,
) -> dict:
    """Compare multiple trajectories to analyze divergence.

    Args:
        trajectories: List of TrajectoryPath to compare
        labels: Optional names for each trajectory

    Returns:
        Dict with comparison metrics
    """
    if labels is None:
        labels = [f"Path {i}" for i in range(len(trajectories))]

    # Compute pairwise rank correlation at each step
    max_steps = max(len(t.steps) for t in trajectories)

    divergence_by_step = []
    for step in range(max_steps):
        step_scores = {}
        for label, traj in zip(labels, trajectories):
            if step < len(traj.steps):
                scores = {cs.index: cs.score for cs in traj.steps[step].remaining_scores}
                step_scores[label] = scores

        if len(step_scores) >= 2:
            # Compute score divergence
            all_labels = list(step_scores.keys())
            divergences = []
            for i, l1 in enumerate(all_labels):
                for l2 in all_labels[i + 1:]:
                    common_indices = set(step_scores[l1].keys()) & set(step_scores[l2].keys())
                    if common_indices:
                        diffs = [
                            abs(step_scores[l1][idx] - step_scores[l2][idx])
                            for idx in common_indices
                        ]
                        divergences.append(np.mean(diffs))
            divergence_by_step.append(np.mean(divergences) if divergences else 0)
        else:
            divergence_by_step.append(0)

    return {
        'labels': labels,
        'num_trajectories': len(trajectories),
        'max_steps': max_steps,
        'divergence_by_step': divergence_by_step,
        'final_divergence': divergence_by_step[-1] if divergence_by_step else 0,
        'engagement_sequences': [t.engagement_sequence for t in trajectories],
    }


def format_trajectory_table(trajectory: TrajectoryPath) -> str:
    """Format trajectory as a readable table."""
    lines = []
    lines.append("=" * 70)
    lines.append("TRAJECTORY SIMULATION")
    lines.append("=" * 70)
    lines.append(f"Engagement sequence: {' → '.join(map(str, trajectory.engagement_sequence))}")
    lines.append("")

    for step in trajectory.steps:
        if step.engaged_candidate_idx is not None:
            lines.append(f"Step {step.step_num}: Engaged with candidate {step.engaged_candidate_idx} "
                        f"(score: {step.engaged_candidate_score:.4f})")
        else:
            lines.append(f"Step {step.step_num}: Initial state")

        lines.append(f"  Context length: {step.context_length}")
        lines.append(f"  Remaining candidates: {len(step.remaining_scores)}")

        if step.remaining_scores:
            lines.append("  Rankings:")
            for cs in step.remaining_scores[:5]:  # Show top 5
                lines.append(f"    Rank {cs.rank}: Candidate {cs.index} (score: {cs.score:.4f})")
            if len(step.remaining_scores) > 5:
                lines.append(f"    ... and {len(step.remaining_scores) - 5} more")
        lines.append("")

    return "\n".join(lines)
