#!/usr/bin/env python3
"""Analyze learned Phoenix model with quantization.

This script:
1. Loads the trained model checkpoint
2. Evaluates on MovieLens test set
3. Compares FP32 vs INT8 quantized rankings
4. Reports accuracy preservation

Usage:
    uv run python scripts/analyze_learned_model.py [--checkpoint PATH]
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

import jax
import jax.numpy as jnp
import numpy as np

from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch

from enhancements.data.movielens import MovieLensDataset
from enhancements.data.movielens_adapter import MovieLensPhoenixAdapter
from enhancements.optimization.quantization import (
    BitWidth,
    Granularity,
    Symmetry,
    QuantizationConfig,
    quantize_params,
    dequantize_params,
)
from phoenix.runners import ModelRunner, RecsysInferenceRunner


def load_checkpoint(checkpoint_path: str) -> Tuple[Dict, PhoenixModelConfig]:
    """Load trained model checkpoint.

    Returns:
        Tuple of (params dict, model config)
    """
    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint["params"], checkpoint["model_config"]


def compute_ndcg(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute NDCG@K."""
    top_k_indices = np.argsort(-scores)[:k]

    # DCG
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        rel = labels[idx]
        dcg += rel / np.log2(i + 2)

    # Ideal DCG
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_labels):
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_ranking_agreement(
    fp32_scores: np.ndarray,
    quant_scores: np.ndarray,
    k: int = 3,
) -> float:
    """Compute top-K ranking agreement between FP32 and quantized."""
    fp32_top_k = set(np.argsort(-fp32_scores)[:k])
    quant_top_k = set(np.argsort(-quant_scores)[:k])

    return len(fp32_top_k & quant_top_k) / k


def main():
    parser = argparse.ArgumentParser(description="Analyze learned Phoenix model")
    parser.add_argument(
        "--checkpoint", type=str, default="models/movielens_phoenix/best_model.pkl",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/ml-100k",
        help="Path to MovieLens data"
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
        help="Number of test samples to evaluate"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phoenix Learned Model Analysis")
    print("=" * 60)
    print()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    params, model_config = load_checkpoint(args.checkpoint)
    print(f"  Model: {model_config.emb_size}d, {model_config.model.num_layers} layers")
    print(f"  Params: model + embeddings")
    print()

    # Load dataset and adapter
    print("Loading dataset...")
    dataset = MovieLensDataset(args.data_dir)
    adapter = MovieLensPhoenixAdapter(
        dataset=dataset,
        model_config=model_config,
        emb_size=model_config.emb_size,
        history_len=model_config.history_seq_len,
        num_candidates=model_config.candidate_seq_len,
    )

    # Sync learned embeddings to adapter
    adapter.set_embedding_params(params["embeddings"])
    print(f"  Test samples: {len(dataset.test_ratings)}")
    print()

    # Create runner
    print("Creating model runner...")
    model_runner = ModelRunner(model=model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="analysis")
    runner.initialize()

    # Get model params
    model_params = params["model"]
    print("  Loaded learned weights")
    print()

    # Quantize params
    print("Quantizing model...")
    quant_config = QuantizationConfig(
        bit_width=BitWidth.INT8,
        granularity=Granularity.PER_CHANNEL,
        symmetry=Symmetry.SYMMETRIC,
    )
    quantized_params = quantize_params(model_params, quant_config)
    dequantized_params = dequantize_params(quantized_params)
    print(f"  Config: INT8 per-channel symmetric")
    print()

    # Evaluate on test set
    print("Evaluating on test set...")
    print()

    test_ratings = dataset.test_ratings[:args.num_samples]

    fp32_ndcg_total = 0.0
    quant_ndcg_total = 0.0
    agreement_total = 0.0
    num_samples = 0

    for i, rating in enumerate(test_ratings):
        # Get test example
        batch, embeddings, labels = adapter.get_training_example(
            rating, num_negatives=7  # 1 positive + 7 negatives = 8 candidates
        )

        # Convert to JAX arrays
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

        # Embeddings from adapter (learned embeddings already synced)
        from phoenix.recsys_model import RecsysEmbeddings as RcEmbeddings
        batch_embeddings = RcEmbeddings(
            user_embeddings=jnp.array(embeddings.user_embeddings),
            history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
            candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
            history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
            candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
        )

        # FP32 inference
        fp32_output = runner.rank_candidates(model_params, batch, batch_embeddings)
        fp32_scores = np.array(fp32_output.scores[0, :, 0])

        # Quantized inference (using dequantized params)
        quant_output = runner.rank_candidates(dequantized_params, batch, batch_embeddings)
        quant_scores = np.array(quant_output.scores[0, :, 0])

        # Compute metrics
        labels_np = np.array(labels)
        fp32_ndcg = compute_ndcg(fp32_scores, labels_np)
        quant_ndcg = compute_ndcg(quant_scores, labels_np)
        agreement = compute_ranking_agreement(fp32_scores, quant_scores)

        fp32_ndcg_total += fp32_ndcg
        quant_ndcg_total += quant_ndcg
        agreement_total += agreement
        num_samples += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(test_ratings)} samples")
            sys.stdout.flush()

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    avg_fp32_ndcg = fp32_ndcg_total / num_samples
    avg_quant_ndcg = quant_ndcg_total / num_samples
    avg_agreement = agreement_total / num_samples

    print(f"FP32 NDCG@3:      {avg_fp32_ndcg:.4f}")
    print(f"INT8 NDCG@3:      {avg_quant_ndcg:.4f}")
    print(f"NDCG Retention:   {(avg_quant_ndcg / avg_fp32_ndcg * 100):.1f}%")
    print()
    print(f"Top-3 Agreement:  {avg_agreement * 100:.1f}%")
    print()

    # Go/No-Go gate
    passed = avg_agreement >= 0.95
    print("=" * 60)
    print(f"Go/No-Go Gate: {'PASSED' if passed else 'FAILED'}")
    print(f"  Requirement: Top-3 agreement >= 95%")
    print(f"  Result: {avg_agreement * 100:.1f}%")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
