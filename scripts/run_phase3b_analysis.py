#!/usr/bin/env python3
"""Run Phase 3b analysis tools with trained MovieLens model.

This script runs:
1. Trajectory Simulation - How rankings evolve as users engage
2. Counterfactual Analysis - Which history items influence rankings

Usage:
    uv run python scripts/run_phase3b_analysis.py
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

import jax.numpy as jnp
import numpy as np

from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ModelRunner, RecsysInferenceRunner

from enhancements.data.movielens import MovieLensDataset
from enhancements.data.movielens_adapter import MovieLensPhoenixAdapter


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, PhoenixModelConfig]:
    """Load trained model checkpoint."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["params"], checkpoint["model_config"]


def compute_kendall_tau(ranking1: List[int], ranking2: List[int]) -> float:
    """Compute Kendall's tau rank correlation coefficient."""
    n = len(ranking1)
    if n != len(ranking2) or n < 2:
        return 0.0

    rank1 = {item: i for i, item in enumerate(ranking1)}
    rank2 = {item: i for i, item in enumerate(ranking2)}

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i + 1, n):
            item_i, item_j = ranking1[i], ranking1[j]
            diff1 = rank1[item_i] - rank1[item_j]
            diff2 = rank2[item_i] - rank2[item_j]

            if diff1 * diff2 > 0:
                concordant += 1
            elif diff1 * diff2 < 0:
                discordant += 1

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return 0.0

    return (concordant - discordant) / total_pairs


def run_trajectory_analysis(
    runner: RecsysInferenceRunner,
    adapter: MovieLensPhoenixAdapter,
    model_params: Dict,
    dataset: MovieLensDataset,
    num_users: int = 10,
    num_steps: int = 5,
) -> Dict:
    """Run trajectory simulation: How do rankings evolve as user engages?

    Simulates user engaging with top-ranked items and tracks ranking changes.
    """
    print("\n" + "=" * 60)
    print("Trajectory Simulation")
    print("=" * 60)
    print(f"\nSimulating {num_users} users over {num_steps} engagement steps")
    print()

    results = []

    for user_idx in range(num_users):
        # Get a user with enough history
        user_ids = list(dataset.users.keys())
        user_id = user_ids[user_idx % len(user_ids)]
        history = dataset.get_user_history(user_id)

        if len(history) < 10:
            continue

        # Get initial batch with history
        # Use first half of history as "context", second half as "future"
        split_point = len(history) // 2
        context_history = history[:split_point]

        # Sample candidates from movies not in history
        user_movies = {r.movie_id for r in history}
        all_movie_ids = list(dataset.movies.keys())
        available_movies = [m for m in all_movie_ids if m not in user_movies]

        if len(available_movies) < 8:
            continue

        # Initial ranking
        candidates = list(np.random.choice(available_movies, size=8, replace=False))

        trajectory = []
        current_history_len = split_point

        for step in range(num_steps):
            # Create batch with current history length
            batch, embeddings = adapter.create_batch_for_user(
                user_id, candidates, history_limit=current_history_len
            )

            # Convert to JAX
            batch = RecsysBatch(
                user_hashes=jnp.array(batch.user_hashes),
                history_post_hashes=jnp.array(batch.history_post_hashes),
                history_author_hashes=jnp.array(batch.history_author_hashes),
                history_actions=jnp.array(batch.history_actions),
                history_product_surface=jnp.array(batch.history_product_surface),
                candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
                candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
                candidate_product_surface=jnp.array(batch.candidate_product_surface),
            )

            batch_embeddings = RecsysEmbeddings(
                user_embeddings=jnp.array(embeddings.user_embeddings),
                history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
                candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
                history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
                candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
            )

            # Get rankings
            output = runner.rank_candidates(model_params, batch, batch_embeddings)
            scores = np.array(output.scores[0, :, 0])
            ranking = list(np.argsort(-scores))

            trajectory.append({
                "step": step,
                "history_len": current_history_len,
                "ranking": ranking,
                "top_score": float(scores[ranking[0]]),
                "score_spread": float(scores.max() - scores.min()),
            })

            # Simulate engagement: add more history
            current_history_len = min(current_history_len + 1, len(history))

        # Compute trajectory stability
        if len(trajectory) >= 2:
            taus = []
            for i in range(len(trajectory) - 1):
                tau = compute_kendall_tau(
                    trajectory[i]["ranking"],
                    trajectory[i + 1]["ranking"]
                )
                taus.append(tau)

            results.append({
                "user_id": user_id,
                "trajectory": trajectory,
                "avg_tau": np.mean(taus),
                "min_tau": np.min(taus),
            })

    # Summarize
    if results:
        avg_stability = np.mean([r["avg_tau"] for r in results])
        print(f"Analyzed {len(results)} user trajectories")
        print(f"Average ranking stability (Kendall's tau): {avg_stability:.3f}")
        print(f"  1.0 = perfectly stable rankings")
        print(f"  0.0 = random changes")
        print()

        # Show example trajectory
        example = results[0]
        print(f"Example trajectory (User {example['user_id']}):")
        for t in example["trajectory"]:
            print(f"  Step {t['step']}: history_len={t['history_len']}, "
                  f"top_score={t['top_score']:.2e}, spread={t['score_spread']:.2e}")

    return {"trajectories": results}


def run_counterfactual_analysis(
    runner: RecsysInferenceRunner,
    adapter: MovieLensPhoenixAdapter,
    model_params: Dict,
    dataset: MovieLensDataset,
    num_users: int = 10,
) -> Dict:
    """Run counterfactual analysis: Which history items influence rankings?

    Ablates (removes) history items one at a time and measures ranking changes.
    """
    print("\n" + "=" * 60)
    print("Counterfactual Analysis (History Ablation)")
    print("=" * 60)
    print(f"\nAnalyzing history importance for {num_users} users")
    print()

    results = []

    for user_idx in range(num_users):
        user_ids = list(dataset.users.keys())
        user_id = user_ids[user_idx % len(user_ids)]
        history = dataset.get_user_history(user_id)

        if len(history) < 8:
            continue

        # Sample candidates
        user_movies = {r.movie_id for r in history}
        all_movie_ids = list(dataset.movies.keys())
        available_movies = [m for m in all_movie_ids if m not in user_movies]

        if len(available_movies) < 8:
            continue

        candidates = list(np.random.choice(available_movies, size=8, replace=False))

        # Baseline ranking with full history
        batch, embeddings = adapter.create_batch_for_user(user_id, candidates)

        batch = RecsysBatch(
            user_hashes=jnp.array(batch.user_hashes),
            history_post_hashes=jnp.array(batch.history_post_hashes),
            history_author_hashes=jnp.array(batch.history_author_hashes),
            history_actions=jnp.array(batch.history_actions),
            history_product_surface=jnp.array(batch.history_product_surface),
            candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
            candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
            candidate_product_surface=jnp.array(batch.candidate_product_surface),
        )

        batch_embeddings = RecsysEmbeddings(
            user_embeddings=jnp.array(embeddings.user_embeddings),
            history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
            candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
            history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
            candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
        )

        output = runner.rank_candidates(model_params, batch, batch_embeddings)
        baseline_scores = np.array(output.scores[0, :, 0])
        baseline_ranking = list(np.argsort(-baseline_scores))

        # Ablate each history position and measure impact
        ablation_impacts = []
        history_len = min(len(history), adapter.history_len)

        for pos in range(min(8, history_len)):  # Ablate first 8 positions
            # Create batch with ablated history (skip position pos)
            ablated_batch, ablated_embeddings = adapter.create_batch_for_user(
                user_id, candidates, history_limit=history_len
            )

            # Zero out the position we're ablating
            ablated_embeddings.history_post_embeddings[0, pos, :] = 0.0
            ablated_embeddings.history_author_embeddings[0, pos, :] = 0.0

            ablated_batch_jax = RecsysBatch(
                user_hashes=jnp.array(ablated_batch.user_hashes),
                history_post_hashes=jnp.array(ablated_batch.history_post_hashes),
                history_author_hashes=jnp.array(ablated_batch.history_author_hashes),
                history_actions=jnp.array(ablated_batch.history_actions),
                history_product_surface=jnp.array(ablated_batch.history_product_surface),
                candidate_post_hashes=jnp.array(ablated_batch.candidate_post_hashes),
                candidate_author_hashes=jnp.array(ablated_batch.candidate_author_hashes),
                candidate_product_surface=jnp.array(ablated_batch.candidate_product_surface),
            )

            ablated_emb_jax = RecsysEmbeddings(
                user_embeddings=jnp.array(ablated_embeddings.user_embeddings),
                history_post_embeddings=jnp.array(ablated_embeddings.history_post_embeddings),
                candidate_post_embeddings=jnp.array(ablated_embeddings.candidate_post_embeddings),
                history_author_embeddings=jnp.array(ablated_embeddings.history_author_embeddings),
                candidate_author_embeddings=jnp.array(ablated_embeddings.candidate_author_embeddings),
            )

            ablated_output = runner.rank_candidates(model_params, ablated_batch_jax, ablated_emb_jax)
            ablated_scores = np.array(ablated_output.scores[0, :, 0])
            ablated_ranking = list(np.argsort(-ablated_scores))

            # Compute impact
            tau = compute_kendall_tau(baseline_ranking, ablated_ranking)
            score_change = float(np.mean(np.abs(baseline_scores - ablated_scores)))

            ablation_impacts.append({
                "position": pos,
                "tau": tau,
                "score_change": score_change,
                "top1_changed": baseline_ranking[0] != ablated_ranking[0],
            })

        # Find most important positions
        impacts_by_change = sorted(ablation_impacts, key=lambda x: x["score_change"], reverse=True)

        results.append({
            "user_id": user_id,
            "history_len": history_len,
            "ablation_impacts": ablation_impacts,
            "most_important_pos": impacts_by_change[0]["position"] if impacts_by_change else 0,
            "avg_tau": np.mean([a["tau"] for a in ablation_impacts]),
        })

    # Summarize
    if results:
        avg_tau = np.mean([r["avg_tau"] for r in results])
        top1_changes = sum(
            1 for r in results
            for a in r["ablation_impacts"]
            if a["top1_changed"]
        )
        total_ablations = sum(len(r["ablation_impacts"]) for r in results)

        print(f"Analyzed {len(results)} users")
        print(f"Total ablations: {total_ablations}")
        print(f"Average tau after ablation: {avg_tau:.3f}")
        print(f"Top-1 changed in {top1_changes}/{total_ablations} ablations ({100*top1_changes/total_ablations:.1f}%)")
        print()

        # Analyze recency effect (are recent items more important?)
        recent_impacts = []
        old_impacts = []
        for r in results:
            for a in r["ablation_impacts"]:
                if a["position"] < 4:  # Recent (first 4 in sequence)
                    recent_impacts.append(a["score_change"])
                else:
                    old_impacts.append(a["score_change"])

        if recent_impacts and old_impacts:
            print(f"Recency analysis:")
            print(f"  Recent items (pos 0-3) avg impact: {np.mean(recent_impacts):.2e}")
            print(f"  Older items (pos 4+) avg impact: {np.mean(old_impacts):.2e}")
            if np.mean(recent_impacts) > np.mean(old_impacts):
                print(f"  -> Model shows recency bias (recent items matter more)")
            else:
                print(f"  -> Model values older context (no recency bias)")

    return {"ablations": results}


def main():
    print("=" * 60)
    print("Phase 3b Analysis with Trained MovieLens Model")
    print("=" * 60)

    # Load checkpoint
    checkpoint_path = "models/movielens_phoenix/best_model.pkl"
    print(f"\nLoading model: {checkpoint_path}")
    params, model_config = load_checkpoint(checkpoint_path)
    model_params = params["model"]
    print(f"  Model: {model_config.emb_size}d, {model_config.model.num_layers} layers")

    # Load dataset
    print("\nLoading MovieLens dataset...")
    dataset = MovieLensDataset("data/ml-100k")
    adapter = MovieLensPhoenixAdapter(
        dataset=dataset,
        model_config=model_config,
        emb_size=model_config.emb_size,
        history_len=model_config.history_seq_len,
        num_candidates=model_config.candidate_seq_len,
    )
    adapter.set_embedding_params(params["embeddings"])
    print(f"  Users: {dataset.num_users}")
    print(f"  Movies: {dataset.num_movies}")

    # Create runner
    print("\nCreating model runner...")
    model_runner = ModelRunner(model=model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="analysis")
    runner.initialize()
    print("  Ready!")

    # Run analyses
    trajectory_results = run_trajectory_analysis(
        runner, adapter, model_params, dataset,
        num_users=20, num_steps=8
    )

    counterfactual_results = run_counterfactual_analysis(
        runner, adapter, model_params, dataset,
        num_users=20
    )

    # Summary
    print("\n" + "=" * 60)
    print("Phase 3b Analysis Complete")
    print("=" * 60)
    print()
    print("Key findings with trained weights:")
    print("1. Trajectory simulation shows how rankings evolve with user engagement")
    print("2. Counterfactual analysis reveals which history items influence rankings")
    print("3. Recency analysis shows whether recent or older history matters more")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
