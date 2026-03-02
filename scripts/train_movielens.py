#!/usr/bin/env python3
"""Train Phoenix model on MovieLens dataset.

Usage:
    uv run python scripts/train_movielens.py [--epochs N] [--quick]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

from enhancements.data.movielens import MovieLensDataset
from enhancements.training.trainer import LossType, PhoenixTrainer, TrainingConfig
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig


def create_model_config(size: str = "small") -> PhoenixModelConfig:
    """Create model configuration.

    Args:
        size: "tiny" for quick tests, "small" for actual training

    Returns:
        PhoenixModelConfig
    """
    if size == "tiny":
        return PhoenixModelConfig(
            emb_size=32,
            num_actions=19,
            history_seq_len=8,
            candidate_seq_len=8,
            hash_config=HashConfig(
                num_user_hashes=1,
                num_item_hashes=1,
                num_author_hashes=1,
            ),
            model=TransformerConfig(
                emb_size=32,
                key_size=16,
                num_q_heads=2,
                num_kv_heads=1,
                num_layers=2,
                widening_factor=2.0,
                attn_output_multiplier=0.125,
            ),
        )
    else:  # small
        return PhoenixModelConfig(
            emb_size=64,
            num_actions=19,
            history_seq_len=16,
            candidate_seq_len=8,
            hash_config=HashConfig(
                num_user_hashes=1,
                num_item_hashes=1,
                num_author_hashes=1,
            ),
            model=TransformerConfig(
                emb_size=64,
                key_size=32,
                num_q_heads=4,
                num_kv_heads=2,
                num_layers=4,
                widening_factor=2.0,
                attn_output_multiplier=0.125,
            ),
        )


def main():
    parser = argparse.ArgumentParser(description="Train Phoenix on MovieLens")
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=5e-4,
        help="Learning rate (default 0.0005)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick training with tiny model (for testing)"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/ml-100k",
        help="Path to MovieLens data"
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Maximum batches per epoch (for quick testing)"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Checkpoint filename to resume from (e.g., 'epoch_5.pkl')"
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early stopping patience (epochs without improvement)"
    )
    parser.add_argument(
        "--loss", type=str, default="bpr", choices=["bpr", "bce"],
        help="Loss function: bpr (pairwise ranking) or bce (pointwise)"
    )
    parser.add_argument(
        "--negatives", type=int, default=7,
        help="Number of random negative samples (when not using in-batch)"
    )
    parser.add_argument(
        "--in-batch", action="store_true", default=True,
        help="Use in-batch negatives (other positives as negatives)"
    )
    parser.add_argument(
        "--no-in-batch", action="store_true",
        help="Disable in-batch negatives, use random sampling only"
    )
    args = parser.parse_args()

    # Handle in-batch flag
    use_in_batch = args.in_batch and not args.no_in_batch

    print("=" * 60)
    print("Phoenix Training on MovieLens")
    print("=" * 60)
    print()

    # Load dataset
    print("Loading MovieLens dataset...")
    dataset = MovieLensDataset(args.data_dir)
    print(f"  Users: {dataset.num_users}")
    print(f"  Movies: {dataset.num_movies}")
    print(f"  Training ratings: {len(dataset.train_ratings)}")
    print(f"  Validation ratings: {len(dataset.val_ratings)}")
    print()

    # Create model
    model_size = "tiny" if args.quick else "small"
    model_config = create_model_config(model_size)
    print(f"Model config ({model_size}):")
    print(f"  Embedding size: {model_config.emb_size}")
    print(f"  Layers: {model_config.model.num_layers}")
    print(f"  History length: {model_config.history_seq_len}")
    print()

    # Create training config
    loss_type = LossType.BPR if args.loss == "bpr" else LossType.BCE
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        max_batches_per_epoch=args.max_batches or (100 if args.quick else None),
        log_every_n_steps=20 if args.quick else 50,
        early_stopping_patience=args.patience,
        loss_type=loss_type,
        num_negatives=args.negatives,
        use_in_batch_negatives=use_in_batch,
    )

    # Create trainer
    print("Initializing trainer...")
    import sys; sys.stdout.flush()
    trainer = PhoenixTrainer(model_config, dataset, training_config)
    print("Trainer initialized!")
    sys.stdout.flush()
    print()

    # Train
    metrics = trainer.train(resume_from=args.resume)

    # Summary
    print()
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print()
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'NDCG@3':<10} {'Hit@3':<10}")
    print("-" * 52)
    for m in metrics:
        print(f"{m.epoch:<8} {m.train_loss:<12.4f} {m.val_loss:<12.4f} {m.val_ndcg:<10.4f} {m.val_hit_rate:<10.4f}")

    print()
    print(f"Best NDCG@3: {trainer.best_val_ndcg:.4f}")
    print(f"Model saved to: {training_config.checkpoint_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
