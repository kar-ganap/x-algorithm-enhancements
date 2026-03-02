#!/usr/bin/env python3
"""Ablation study: Transformer vs Embeddings contribution.

This script measures what proportion of model performance comes from:
1. Learned embeddings (item representations)
2. Transformer architecture (contextual reasoning)

We compare:
- Full model (learned embeddings + transformer)
- Random embeddings + transformer (transformer-only contribution)
- Learned embeddings + dot-product scoring (embedding-only contribution)

Usage:
    uv run python scripts/ablation_study.py
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

import jax.numpy as jnp
import numpy as np

from enhancements.data.movielens import MovieLensDataset
from enhancements.data.movielens_adapter import MovieLensPhoenixAdapter
from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ModelRunner, RecsysInferenceRunner


def load_checkpoint(checkpoint_path: str) -> tuple[dict, PhoenixModelConfig]:
    """Load trained model checkpoint."""
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint["params"], checkpoint["model_config"]


def compute_ndcg(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute NDCG@K."""
    top_k_indices = np.argsort(-scores)[:k]

    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        rel = labels[idx]
        dcg += rel / np.log2(i + 2)

    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_labels):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_hit_rate(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute Hit@K (1 if positive in top-k, else 0)."""
    top_k_indices = np.argsort(-scores)[:k]
    return 1.0 if any(labels[idx] > 0 for idx in top_k_indices) else 0.0


def dot_product_scoring(
    user_emb: np.ndarray,
    history_emb: np.ndarray,
    candidate_emb: np.ndarray,
) -> np.ndarray:
    """Simple dot-product scoring without transformer.

    Uses average of history embeddings as user context,
    then dot-product with each candidate.
    """
    # Flatten all embeddings to [batch, ..., emb_size] and average to get context
    # user_emb: [batch, num_hashes, emb_size] -> [batch, emb_size]
    user_context = np.mean(user_emb, axis=1) if user_emb.ndim > 2 else user_emb

    # history_emb: [batch, history_len, num_hashes, emb_size] -> [batch, emb_size]
    if history_emb.ndim == 4:
        history_context = np.mean(history_emb, axis=(1, 2))
    else:
        history_context = np.mean(history_emb, axis=1)

    # Combine user and history context
    context = (user_context + history_context) / 2  # [batch, emb_size]

    # candidate_emb: [batch, num_candidates, num_hashes, emb_size] -> [batch, num_candidates, emb_size]
    if candidate_emb.ndim == 4:
        candidate_avg = np.mean(candidate_emb, axis=2)
    else:
        candidate_avg = candidate_emb

    # Ensure context is 2D
    if context.ndim == 1:
        context = context[np.newaxis, :]

    # Dot product: [batch, num_candidates]
    scores = np.einsum('be,bce->bc', context, candidate_avg)

    return scores[0]  # Return [num_candidates]


def main():
    print("=" * 70)
    print("Ablation Study: Transformer vs Embeddings Contribution")
    print("=" * 70)
    print()

    # Load checkpoint
    checkpoint_path = "models/movielens_phoenix/best_model.pkl"
    print(f"Loading model: {checkpoint_path}")
    params, model_config = load_checkpoint(checkpoint_path)
    model_params = params["model"]
    embedding_params = params["embeddings"]
    print(f"  Model: {model_config.emb_size}d, {model_config.model.num_layers} layers")
    print()

    # Load dataset
    print("Loading MovieLens dataset...")
    dataset = MovieLensDataset("data/ml-100k")

    # Create adapter with learned embeddings
    adapter = MovieLensPhoenixAdapter(
        dataset=dataset,
        model_config=model_config,
        emb_size=model_config.emb_size,
        history_len=model_config.history_seq_len,
        num_candidates=model_config.candidate_seq_len,
    )
    adapter.set_embedding_params(embedding_params)
    print(f"  Test samples: {len(dataset.test_ratings)}")
    print()

    # Create runner
    print("Creating model runner...")
    model_runner = ModelRunner(model=model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="ablation")
    runner.initialize()
    print("  Ready!")
    print()

    # Evaluate on test set
    num_samples = min(500, len(dataset.test_ratings))
    print(f"Evaluating {num_samples} test samples...")
    print()

    # Metrics accumulators
    full_model_ndcg = []
    full_model_hit = []
    random_emb_ndcg = []
    random_emb_hit = []
    learned_emb_ndcg = []
    learned_emb_hit = []
    random_baseline_ndcg = []
    random_baseline_hit = []

    np.random.seed(42)

    for i, rating in enumerate(dataset.test_ratings[:num_samples]):
        # Get test example with learned embeddings
        batch, embeddings, labels = adapter.get_training_example(
            rating, num_negatives=7
        )
        labels_np = np.array(labels)

        # Convert to JAX
        batch_jax = RecsysBatch(
            user_hashes=jnp.array(batch.user_hashes),
            history_post_hashes=jnp.array(batch.history_post_hashes),
            history_author_hashes=jnp.array(batch.history_author_hashes),
            history_actions=jnp.array(batch.history_actions),
            history_product_surface=jnp.array(batch.history_product_surface),
            candidate_post_hashes=jnp.array(batch.candidate_post_hashes),
            candidate_author_hashes=jnp.array(batch.candidate_author_hashes),
            candidate_product_surface=jnp.array(batch.candidate_product_surface),
        )

        embeddings_jax = RecsysEmbeddings(
            user_embeddings=jnp.array(embeddings.user_embeddings),
            history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
            candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
            history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
            candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
        )

        # 1. Full model (learned embeddings + transformer)
        output = runner.rank_candidates(model_params, batch_jax, embeddings_jax)
        full_scores = np.array(output.scores[0, :, 0])
        full_model_ndcg.append(compute_ndcg(full_scores, labels_np))
        full_model_hit.append(compute_hit_rate(full_scores, labels_np))

        # 2. Random embeddings + transformer
        emb_shape = embeddings.user_embeddings.shape
        random_embeddings = RecsysEmbeddings(
            user_embeddings=jnp.array(np.random.randn(*embeddings.user_embeddings.shape).astype(np.float32) * 0.02),
            history_post_embeddings=jnp.array(np.random.randn(*embeddings.history_post_embeddings.shape).astype(np.float32) * 0.02),
            candidate_post_embeddings=jnp.array(np.random.randn(*embeddings.candidate_post_embeddings.shape).astype(np.float32) * 0.02),
            history_author_embeddings=jnp.array(np.random.randn(*embeddings.history_author_embeddings.shape).astype(np.float32) * 0.02),
            candidate_author_embeddings=jnp.array(np.random.randn(*embeddings.candidate_author_embeddings.shape).astype(np.float32) * 0.02),
        )
        output_random = runner.rank_candidates(model_params, batch_jax, random_embeddings)
        random_scores = np.array(output_random.scores[0, :, 0])
        random_emb_ndcg.append(compute_ndcg(random_scores, labels_np))
        random_emb_hit.append(compute_hit_rate(random_scores, labels_np))

        # 3. Learned embeddings + dot-product (no transformer)
        dot_scores = dot_product_scoring(
            np.array(embeddings.user_embeddings),
            np.array(embeddings.history_post_embeddings),
            np.array(embeddings.candidate_post_embeddings),
        )
        learned_emb_ndcg.append(compute_ndcg(dot_scores, labels_np))
        learned_emb_hit.append(compute_hit_rate(dot_scores, labels_np))

        # 4. Random baseline (random scores)
        random_scores_baseline = np.random.randn(len(labels_np))
        random_baseline_ndcg.append(compute_ndcg(random_scores_baseline, labels_np))
        random_baseline_hit.append(compute_hit_rate(random_scores_baseline, labels_np))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples}")

    # Results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)
    print()

    results = {
        "Random Baseline": (np.mean(random_baseline_ndcg), np.mean(random_baseline_hit)),
        "Learned Emb + Dot-Product": (np.mean(learned_emb_ndcg), np.mean(learned_emb_hit)),
        "Random Emb + Transformer": (np.mean(random_emb_ndcg), np.mean(random_emb_hit)),
        "Full Model (Learned Emb + Transformer)": (np.mean(full_model_ndcg), np.mean(full_model_hit)),
    }

    print(f"{'Configuration':<40} {'NDCG@3':<12} {'Hit@3':<12}")
    print("-" * 64)
    for name, (ndcg, hit) in results.items():
        print(f"{name:<40} {ndcg:.4f}       {hit:.4f}")

    print()
    print("=" * 70)
    print("Attribution Analysis")
    print("=" * 70)
    print()

    random_ndcg = results["Random Baseline"][0]
    emb_only_ndcg = results["Learned Emb + Dot-Product"][0]
    trans_only_ndcg = results["Random Emb + Transformer"][0]
    full_ndcg = results["Full Model (Learned Emb + Transformer)"][0]

    # Compute contributions
    total_improvement = full_ndcg - random_ndcg
    emb_contribution = emb_only_ndcg - random_ndcg
    trans_contribution = trans_only_ndcg - random_ndcg
    synergy = total_improvement - emb_contribution - trans_contribution

    print(f"Total improvement over random: {total_improvement:.4f}")
    print()
    print(f"Embedding contribution:   {emb_contribution:.4f} ({100*emb_contribution/total_improvement:.1f}%)")
    print(f"Transformer contribution: {trans_contribution:.4f} ({100*trans_contribution/total_improvement:.1f}%)")
    print(f"Synergy (interaction):    {synergy:.4f} ({100*synergy/total_improvement:.1f}%)")
    print()

    # Interpretation
    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print()

    if emb_contribution > trans_contribution:
        print("→ Learned embeddings contribute MORE than the transformer")
        print("  This suggests the item representations capture most of the signal.")
    else:
        print("→ Transformer contributes MORE than learned embeddings")
        print("  This suggests contextual reasoning is the main driver.")

    if synergy > 0.01:
        print(f"\n→ Positive synergy ({100*synergy/total_improvement:.1f}%): Embeddings and transformer work better together")
    elif synergy < -0.01:
        print("\n→ Negative synergy: Some redundancy between embeddings and transformer")
    else:
        print("\n→ Contributions are roughly additive (minimal interaction)")

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Full model NDCG@3:     {full_ndcg:.4f}")
    print(f"Embedding lift:        +{emb_contribution:.4f} ({100*emb_contribution/total_improvement:.1f}% of total)")
    print(f"Transformer lift:      +{trans_contribution:.4f} ({100*trans_contribution/total_improvement:.1f}% of total)")
    print(f"Synergy:               {'+' if synergy >= 0 else ''}{synergy:.4f} ({100*synergy/total_improvement:.1f}% of total)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
