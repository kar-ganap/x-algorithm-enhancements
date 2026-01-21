"""Counterfactual Analysis: How do rankings change with different user contexts?

This module answers questions like:
- Which history items most influence rankings? (ablation)
- How sensitive are rankings to recent vs old history? (recency)
- Do different engagement types (favorite vs retweet) matter? (action sensitivity)
- How do rankings differ across user segments? (fairness)

Unlike trajectory simulation (which varies candidates with fixed context),
counterfactual analysis varies context with fixed candidates.

Key insight: Candidate INPUT embeddings are context-independent (hash lookups).
Only the transformer representations depend on context. So we can:
1. Fix candidate embeddings
2. Vary user/history embeddings
3. Re-run transformer to see ranking changes

Usage:
    uv run python enhancements/analysis/counterfactual_analysis.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ACTIONS, create_example_batch

from enhancements.optimization.full_kv_cache import FullKVCachedRunner


@dataclass
class RankingSnapshot:
    """Snapshot of rankings for a specific context."""
    context_label: str
    ranked_indices: List[int]  # Candidate indices in rank order
    scores: Dict[int, float]  # candidate_idx -> score


@dataclass
class AblationResult:
    """Result of ablating (removing) a history item."""
    removed_position: int
    original_ranking: List[int]
    ablated_ranking: List[int]
    rank_changes: Dict[int, int]  # candidate_idx -> rank change
    kendall_tau: float  # Rank correlation


@dataclass
class CounterfactualResult:
    """Result of counterfactual analysis."""
    baseline_ranking: List[int]
    counterfactual_rankings: Dict[str, List[int]]  # label -> ranking
    ranking_stability: float  # How stable are rankings across contexts


def compute_kendall_tau(ranking1: List[int], ranking2: List[int]) -> float:
    """Compute Kendall's tau rank correlation coefficient.

    Returns value in [-1, 1] where:
    - 1 = identical rankings
    - 0 = uncorrelated
    - -1 = reversed rankings
    """
    n = len(ranking1)
    if n != len(ranking2) or n < 2:
        return 0.0

    # Create rank mappings
    rank1 = {item: i for i, item in enumerate(ranking1)}
    rank2 = {item: i for i, item in enumerate(ranking2)}

    # Count concordant and discordant pairs
    concordant = 0
    discordant = 0

    items = list(set(ranking1) & set(ranking2))
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i], items[j]
            # Compare relative order in both rankings
            order1 = rank1[a] - rank1[b]
            order2 = rank2[a] - rank2[b]

            if order1 * order2 > 0:
                concordant += 1
            elif order1 * order2 < 0:
                discordant += 1

    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 1.0

    return (concordant - discordant) / total_pairs


def compute_rank_changes(original: List[int], modified: List[int]) -> Dict[int, int]:
    """Compute how much each candidate's rank changed."""
    orig_ranks = {item: i for i, item in enumerate(original)}
    mod_ranks = {item: i for i, item in enumerate(modified)}

    changes = {}
    for item in set(original) & set(modified):
        changes[item] = mod_ranks[item] - orig_ranks[item]

    return changes


class CounterfactualAnalyzer:
    """Analyze how rankings change with different user contexts.

    This class enables counterfactual questions:
    - "What if the user had different history?"
    - "Which history items matter most for these rankings?"
    - "How sensitive are rankings to history length?"
    """

    def __init__(
        self,
        runner: FullKVCachedRunner,
        base_batch: RecsysBatch,
        base_embeddings: RecsysEmbeddings,
    ):
        self.runner = runner
        self.base_batch = base_batch
        self.base_embeddings = base_embeddings
        self._baseline_ranking: Optional[List[int]] = None
        self._baseline_scores: Optional[Dict[int, float]] = None

    def _run_ranking(self, batch: RecsysBatch, embeddings: RecsysEmbeddings) -> Tuple[List[int], Dict[int, float]]:
        """Run ranking and return (ranked_indices, scores_dict)."""
        self.runner.clear_cache()
        output = self.runner.rank(batch, embeddings, use_cache=False)

        # Extract rankings
        ranked_indices = list(np.array(output.ranked_indices[0]))

        # Extract scores (mean across actions)
        raw_scores = np.array(output.scores[0])  # [num_candidates, num_actions]
        agg_scores = np.mean(raw_scores, axis=-1)
        scores_dict = {i: float(agg_scores[i]) for i in range(len(agg_scores))}

        return ranked_indices, scores_dict

    def get_baseline(self) -> RankingSnapshot:
        """Get baseline ranking with full context."""
        if self._baseline_ranking is None:
            self._baseline_ranking, self._baseline_scores = self._run_ranking(
                self.base_batch, self.base_embeddings
            )

        return RankingSnapshot(
            context_label="baseline",
            ranked_indices=self._baseline_ranking,
            scores=self._baseline_scores,
        )

    def ablate_history_item(self, position: int) -> AblationResult:
        """Remove a single history item and measure ranking change.

        Args:
            position: Which history position to remove (0 = oldest)

        Returns:
            AblationResult with ranking changes
        """
        baseline = self.get_baseline()

        history_len = self.base_batch.history_post_hashes.shape[1]
        if position < 0 or position >= history_len:
            raise ValueError(f"Position {position} out of range [0, {history_len})")

        # Create mask for positions to keep
        keep_mask = [i for i in range(history_len) if i != position]

        # Create ablated batch
        ablated_batch = RecsysBatch(
            user_hashes=self.base_batch.user_hashes,
            history_post_hashes=np.array(self.base_batch.history_post_hashes)[:, keep_mask, :],
            history_author_hashes=np.array(self.base_batch.history_author_hashes)[:, keep_mask, :],
            history_product_surface=np.array(self.base_batch.history_product_surface)[:, keep_mask],
            history_actions=np.array(self.base_batch.history_actions)[:, keep_mask, :],
            candidate_post_hashes=self.base_batch.candidate_post_hashes,
            candidate_author_hashes=self.base_batch.candidate_author_hashes,
            candidate_product_surface=self.base_batch.candidate_product_surface,
        )

        ablated_embeddings = RecsysEmbeddings(
            user_embeddings=self.base_embeddings.user_embeddings,
            history_post_embeddings=np.array(self.base_embeddings.history_post_embeddings)[:, keep_mask, :, :],
            history_author_embeddings=np.array(self.base_embeddings.history_author_embeddings)[:, keep_mask, :, :],
            candidate_post_embeddings=self.base_embeddings.candidate_post_embeddings,
            candidate_author_embeddings=self.base_embeddings.candidate_author_embeddings,
        )

        # Run ranking with ablated context
        ablated_ranking, _ = self._run_ranking(ablated_batch, ablated_embeddings)

        # Compute changes
        rank_changes = compute_rank_changes(baseline.ranked_indices, ablated_ranking)
        tau = compute_kendall_tau(baseline.ranked_indices, ablated_ranking)

        return AblationResult(
            removed_position=position,
            original_ranking=baseline.ranked_indices,
            ablated_ranking=ablated_ranking,
            rank_changes=rank_changes,
            kendall_tau=tau,
        )

    def ablate_all_history(self) -> List[AblationResult]:
        """Ablate each history item and measure impact."""
        history_len = self.base_batch.history_post_hashes.shape[1]
        results = []

        for pos in range(history_len):
            result = self.ablate_history_item(pos)
            results.append(result)

        return results

    def truncate_history(self, keep_last_n: int) -> RankingSnapshot:
        """Keep only the N most recent history items.

        Args:
            keep_last_n: Number of recent items to keep

        Returns:
            RankingSnapshot with truncated history
        """
        history_len = self.base_batch.history_post_hashes.shape[1]
        if keep_last_n >= history_len:
            return self.get_baseline()

        # Keep last N items
        start_idx = history_len - keep_last_n

        truncated_batch = RecsysBatch(
            user_hashes=self.base_batch.user_hashes,
            history_post_hashes=np.array(self.base_batch.history_post_hashes)[:, start_idx:, :],
            history_author_hashes=np.array(self.base_batch.history_author_hashes)[:, start_idx:, :],
            history_product_surface=np.array(self.base_batch.history_product_surface)[:, start_idx:],
            history_actions=np.array(self.base_batch.history_actions)[:, start_idx:, :],
            candidate_post_hashes=self.base_batch.candidate_post_hashes,
            candidate_author_hashes=self.base_batch.candidate_author_hashes,
            candidate_product_surface=self.base_batch.candidate_product_surface,
        )

        truncated_embeddings = RecsysEmbeddings(
            user_embeddings=self.base_embeddings.user_embeddings,
            history_post_embeddings=np.array(self.base_embeddings.history_post_embeddings)[:, start_idx:, :, :],
            history_author_embeddings=np.array(self.base_embeddings.history_author_embeddings)[:, start_idx:, :, :],
            candidate_post_embeddings=self.base_embeddings.candidate_post_embeddings,
            candidate_author_embeddings=self.base_embeddings.candidate_author_embeddings,
        )

        ranking, scores = self._run_ranking(truncated_batch, truncated_embeddings)

        return RankingSnapshot(
            context_label=f"last_{keep_last_n}",
            ranked_indices=ranking,
            scores=scores,
        )

    def analyze_recency_sensitivity(self) -> Dict[int, float]:
        """Analyze how rankings change as we truncate history.

        Returns:
            Dict mapping history_length -> kendall_tau with baseline
        """
        baseline = self.get_baseline()
        history_len = self.base_batch.history_post_hashes.shape[1]

        results = {}
        for keep_n in range(1, history_len + 1):
            snapshot = self.truncate_history(keep_n)
            tau = compute_kendall_tau(baseline.ranked_indices, snapshot.ranked_indices)
            results[keep_n] = tau

        return results

    def modify_history_actions(self, position: int, new_actions: np.ndarray) -> RankingSnapshot:
        """Change the action vector for a history item.

        Args:
            position: Which history item to modify
            new_actions: New action vector [num_actions]

        Returns:
            RankingSnapshot with modified history
        """
        history_len = self.base_batch.history_post_hashes.shape[1]
        if position < 0 or position >= history_len:
            raise ValueError(f"Position {position} out of range")

        # Copy and modify actions
        modified_actions = np.array(self.base_batch.history_actions).copy()
        modified_actions[0, position, :] = new_actions

        modified_batch = RecsysBatch(
            user_hashes=self.base_batch.user_hashes,
            history_post_hashes=self.base_batch.history_post_hashes,
            history_author_hashes=self.base_batch.history_author_hashes,
            history_product_surface=self.base_batch.history_product_surface,
            history_actions=modified_actions,
            candidate_post_hashes=self.base_batch.candidate_post_hashes,
            candidate_author_hashes=self.base_batch.candidate_author_hashes,
            candidate_product_surface=self.base_batch.candidate_product_surface,
        )

        ranking, scores = self._run_ranking(modified_batch, self.base_embeddings)

        return RankingSnapshot(
            context_label=f"modified_action_pos{position}",
            ranked_indices=ranking,
            scores=scores,
        )


def visualize_ablation_results(results: List[AblationResult], history_len: int):
    """Visualize which history items matter most."""
    print("\n" + "=" * 70)
    print("HISTORY ITEM IMPORTANCE (Ablation Analysis)")
    print("=" * 70)
    print("\nRemoving each history item and measuring ranking change:")
    print("Lower Kendall's τ = more important item (bigger ranking change)")

    print(f"\n{'Position':<10} {'Kendall τ':<12} {'Importance':<15} {'Visual':<30}")
    print("-" * 70)

    # Sort by importance (lower tau = more important)
    sorted_results = sorted(results, key=lambda r: r.kendall_tau)

    for result in results:
        tau = result.kendall_tau
        importance = 1 - tau  # Higher = more important

        # Recency indicator
        recency = "recent" if result.removed_position >= history_len - 5 else "older"

        # Visual bar
        bar_len = int(importance * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)

        print(f"{result.removed_position:<10} {tau:<12.3f} {importance:<15.3f} [{bar}]")

    # Summary
    most_important = sorted_results[0]
    least_important = sorted_results[-1]

    print("\n" + "-" * 70)
    print(f"Most important: Position {most_important.removed_position} (τ={most_important.kendall_tau:.3f})")
    print(f"Least important: Position {least_important.removed_position} (τ={least_important.kendall_tau:.3f})")

    # Check recency bias
    recent_positions = [r.removed_position for r in sorted_results[:5] if r.removed_position >= history_len - 10]
    if len(recent_positions) >= 3:
        print("\n⚠️  Recent history items appear more important (recency bias)")


def visualize_recency_sensitivity(sensitivity: Dict[int, float], full_history_len: int):
    """Visualize how rankings change with history truncation."""
    print("\n" + "=" * 70)
    print("RECENCY SENSITIVITY ANALYSIS")
    print("=" * 70)
    print("\nHow do rankings change as we use less history?")
    print("Kendall's τ = correlation with full-history ranking")

    print(f"\n{'History Length':<15} {'Kendall τ':<12} {'Stability':<30}")
    print("-" * 60)

    for length in sorted(sensitivity.keys()):
        tau = sensitivity[length]
        bar_len = int(tau * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"{length:<15} {tau:<12.3f} [{bar}]")

    # Analysis
    recent_only = sensitivity.get(5, sensitivity.get(min(sensitivity.keys())))
    half_history = sensitivity.get(full_history_len // 2, 0)

    print("\n" + "-" * 60)
    if recent_only > 0.8:
        print("✓ Rankings stable with just recent history")
        print("  → Older history has limited influence")
    elif recent_only < 0.5:
        print("⚠️ Rankings highly sensitive to history length")
        print("  → Full history context matters significantly")


def create_test_config(history_len: int = 32, candidate_seq_len: int = 8):
    """Create test configuration."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=256,
        num_actions=len(ACTIONS),
        history_seq_len=history_len,
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


def run_analysis(history_len: int = 32, num_candidates: int = 8):
    """Run counterfactual analysis."""
    print("=" * 70)
    print("COUNTERFACTUAL ANALYSIS")
    print("=" * 70)
    print("\nThis analysis asks: 'What if the user had different history?'")
    print("We fix candidates and vary the user context.")

    print(f"\nConfiguration:")
    print(f"  History length: {history_len}")
    print(f"  Candidates: {num_candidates}")

    print("\nInitializing model...")
    config = create_test_config(history_len=history_len, candidate_seq_len=num_candidates)
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, embeddings = create_example_batch(
        batch_size=1,
        emb_size=config.emb_size,
        history_len=history_len,
        num_candidates=num_candidates,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )

    print("Model initialized")

    # Create analyzer
    analyzer = CounterfactualAnalyzer(runner, batch, embeddings)

    # Get baseline
    baseline = analyzer.get_baseline()
    print(f"\nBaseline ranking: {' → '.join(f'C{i}' for i in baseline.ranked_indices[:5])}...")

    # Ablation analysis (sample if history is long)
    print("\nRunning ablation analysis...")
    if history_len <= 16:
        ablation_results = analyzer.ablate_all_history()
    else:
        # Sample positions for long history
        sample_positions = list(range(0, history_len, history_len // 16))
        ablation_results = [analyzer.ablate_history_item(pos) for pos in sample_positions]

    visualize_ablation_results(ablation_results, history_len)

    # Recency sensitivity
    print("\nRunning recency sensitivity analysis...")
    recency_sensitivity = analyzer.analyze_recency_sensitivity()
    visualize_recency_sensitivity(recency_sensitivity, history_len)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Average ablation impact
    avg_tau = np.mean([r.kendall_tau for r in ablation_results])
    print(f"\nAverage ranking stability (single item removal): τ = {avg_tau:.3f}")

    if avg_tau > 0.95:
        print("→ Rankings very stable - individual history items have minimal impact")
    elif avg_tau > 0.8:
        print("→ Rankings moderately stable - some history items matter")
    else:
        print("→ Rankings sensitive to history - specific items strongly influence output")

    # Recency impact
    recent_tau = recency_sensitivity.get(5, recency_sensitivity.get(min(recency_sensitivity.keys())))
    print(f"\nRanking correlation with 5 most recent items only: τ = {recent_tau:.3f}")

    if recent_tau > 0.8:
        print("→ Recent history sufficient - older items add little value")
    else:
        print("→ Full history matters - truncation significantly changes rankings")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run counterfactual analysis")
    parser.add_argument("--history", type=int, default=32, help="History length")
    parser.add_argument("--candidates", type=int, default=8, help="Number of candidates")
    args = parser.parse_args()

    run_analysis(history_len=args.history, num_candidates=args.candidates)
