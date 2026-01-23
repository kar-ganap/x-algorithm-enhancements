#!/usr/bin/env python3
"""Train Phoenix model on synthetic Twitter-like dataset.

Usage:
    uv run python scripts/train_synthetic.py [--epochs N] [--quick]
"""

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ModelRunner, RecsysInferenceRunner

from enhancements.data.synthetic_twitter import (
    SyntheticTwitterDataset,
    SyntheticTwitterGenerator,
    create_train_val_test_split,
)
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter


# Register RecsysEmbeddings as a JAX pytree
def _recsys_embeddings_flatten(emb: RecsysEmbeddings):
    return (
        emb.user_embeddings,
        emb.history_post_embeddings,
        emb.candidate_post_embeddings,
        emb.history_author_embeddings,
        emb.candidate_author_embeddings,
    ), None


def _recsys_embeddings_unflatten(aux, children):
    return RecsysEmbeddings(
        user_embeddings=children[0],
        history_post_embeddings=children[1],
        candidate_post_embeddings=children[2],
        history_author_embeddings=children[3],
        candidate_author_embeddings=children[4],
    )


try:
    jax.tree_util.register_pytree_node(
        RecsysEmbeddings,
        _recsys_embeddings_flatten,
        _recsys_embeddings_unflatten,
    )
except ValueError:
    pass


def create_model_config(size: str = "small") -> PhoenixModelConfig:
    """Create model configuration."""
    if size == "tiny":
        return PhoenixModelConfig(
            emb_size=32,
            num_actions=19,  # All 19 Twitter actions
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


def make_bpr_loss_fn(
    forward_fn: Callable,
    compute_embeddings_fn: Callable,
):
    """Create BPR loss function with learnable embeddings."""

    def loss_fn(
        params: Dict[str, Any],
        batch: RecsysBatch,
        labels: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute BPR loss.

        Args:
            params: {'model': model_params, 'embeddings': emb_params}
            batch: Input batch
            labels: [batch_size, num_candidates] binary labels

        Returns:
            (loss, metrics_dict)
        """
        model_params = params["model"]
        emb_params = params["embeddings"]

        # Compute embeddings from learnable params
        embeddings = compute_embeddings_fn(emb_params, batch)

        # Forward pass
        output = forward_fn(model_params, batch, embeddings)
        logits = output.scores  # [batch, candidates, actions]

        # Use mean across actions as relevance score
        scores = jnp.mean(logits, axis=-1)  # [batch, candidates]

        # BPR loss: maximize positive score over negative scores
        # Positive is always the first candidate
        pos_scores = scores[:, 0:1]  # [batch, 1]
        neg_scores = scores[:, 1:]   # [batch, num_negs]

        # Log sigmoid of difference
        diff = pos_scores - neg_scores
        bpr_loss = -jnp.mean(jax.nn.log_sigmoid(diff))

        # Accuracy: positive ranked higher than all negatives
        pos_wins = jnp.all(pos_scores > neg_scores, axis=-1)
        accuracy = jnp.mean(pos_wins.astype(jnp.float32))

        return bpr_loss, {"loss": bpr_loss, "accuracy": accuracy}

    return loss_fn


def make_train_step(loss_fn, optimizer):
    """Create JIT-compiled training step."""

    @jax.jit
    def train_step(params, opt_state, batch, labels):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, labels
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return train_step


def compute_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute NDCG@K."""
    batch_size = scores.shape[0]
    ndcgs = []

    for i in range(batch_size):
        # Get top-k indices by score
        top_k_idx = np.argsort(-scores[i])[:k]
        top_k_labels = labels[i][top_k_idx]

        # DCG
        discounts = 1.0 / np.log2(np.arange(k) + 2)
        dcg = np.sum(top_k_labels * discounts)

        # Ideal DCG
        ideal_labels = np.sort(labels[i])[::-1][:k]
        idcg = np.sum(ideal_labels * discounts)

        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)

    return float(np.mean(ndcgs))


def compute_hit_rate_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 3) -> float:
    """Compute Hit Rate@K (was any positive in top-k)."""
    batch_size = scores.shape[0]
    hits = 0

    for i in range(batch_size):
        top_k_idx = np.argsort(-scores[i])[:k]
        if np.any(labels[i][top_k_idx] > 0):
            hits += 1

    return hits / batch_size


class SyntheticTrainer:
    """Trainer for Phoenix on synthetic Twitter data."""

    def __init__(
        self,
        model_config: PhoenixModelConfig,
        dataset: SyntheticTwitterDataset,
        adapter: SyntheticTwitterPhoenixAdapter,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        neg_ratio: int = 7,
    ):
        self.model_config = model_config
        self.dataset = dataset
        self.adapter = adapter
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio

        # Create model
        print("  Creating model runner...")
        self.model_runner = ModelRunner(model=model_config)
        self.runner = RecsysInferenceRunner(runner=self.model_runner, name="training")
        print("  Initializing runner...")
        self.runner.initialize()
        print("  Runner initialized!")

        # Combined params
        self.params = {
            "model": self.runner.params,
            "embeddings": self.adapter.get_embedding_params(),
        }

        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Loss function
        self._loss_fn = make_bpr_loss_fn(
            self.runner.rank_candidates,
            self.adapter.compute_embeddings_from_params,
        )
        self._train_step = make_train_step(self._loss_fn, self.optimizer)

        # Tracking
        self.best_val_ndcg = 0.0

    def train_epoch(self, max_batches: Optional[int] = None) -> Tuple[float, float]:
        """Train for one epoch. Returns (avg_loss, avg_accuracy)."""
        num_train = len(self.adapter.train_engagements)
        num_batches = num_train // self.batch_size
        if max_batches:
            num_batches = min(num_batches, max_batches)

        total_loss = 0.0
        total_acc = 0.0

        for step in range(num_batches):
            batch, _, labels = self.adapter.get_training_batch(
                batch_size=self.batch_size,
                neg_ratio=self.neg_ratio,
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
            labels = jnp.array(labels)

            self.params, self.opt_state, metrics = self._train_step(
                self.params, self.opt_state, batch, labels
            )

            total_loss += float(metrics["loss"])
            total_acc += float(metrics["accuracy"])

            if (step + 1) % 50 == 0:
                print(f"    Step {step+1}/{num_batches}: loss={float(metrics['loss']):.4f}")

        return total_loss / num_batches, total_acc / num_batches

    def evaluate(self, num_samples: int = 200) -> Tuple[float, float]:
        """Evaluate on validation set. Returns (ndcg@3, hit_rate@3)."""
        samples = self.adapter.get_validation_samples(num_samples)

        all_scores = []
        all_labels = []

        for user_id, pos_post_id, neg_post_ids in samples:
            candidates = [pos_post_id] + neg_post_ids
            batch, embeddings = self.adapter.create_batch_for_user(
                user_id, candidates, num_candidates_override=len(candidates)
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

            # Compute embeddings from learned params
            emb_params = self.params["embeddings"]
            embeddings = self.adapter.compute_embeddings_from_params(emb_params, batch)

            # Forward pass
            output = self.runner.rank_candidates(self.params["model"], batch, embeddings)
            scores = np.array(jnp.mean(output.scores[0], axis=-1))  # [num_candidates]

            labels = np.array([1.0] + [0.0] * len(neg_post_ids))
            all_scores.append(scores)
            all_labels.append(labels)

        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        ndcg = compute_ndcg_at_k(all_scores, all_labels, k=3)
        hit_rate = compute_hit_rate_at_k(all_scores, all_labels, k=3)

        return ndcg, hit_rate

    def save(self, path: str):
        """Save model parameters."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": jax.device_get(self.params["model"]),
                "embeddings": jax.device_get(self.params["embeddings"]),
                "model_config": self.model_config,
            }, f)


def main():
    parser = argparse.ArgumentParser(description="Train Phoenix on synthetic Twitter data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--quick", action="store_true", help="Quick training with tiny model")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_twitter")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--generate", action="store_true", help="Generate new dataset first")
    args = parser.parse_args()

    print("=" * 70)
    print("Phoenix Training on Synthetic Twitter Data")
    print("=" * 70)
    print()

    # Load or generate dataset
    data_path = Path(args.data_dir)
    dataset_file = data_path / "dataset.pkl"
    splits_file = data_path / "splits.pkl"

    if args.generate or not dataset_file.exists():
        print("Generating new dataset...")
        generator = SyntheticTwitterGenerator(seed=42)
        dataset = generator.generate(
            num_users=1000,
            num_posts=50000,
            num_engagements=200000,
        )
        train, val, test = create_train_val_test_split(dataset)

        data_path.mkdir(parents=True, exist_ok=True)
        with open(dataset_file, "wb") as f:
            pickle.dump(dataset, f)
        with open(splits_file, "wb") as f:
            pickle.dump({"train": train, "val": val, "test": test}, f)
        print(f"  Generated and saved to {data_path}")
    else:
        print(f"Loading dataset from {data_path}...")
        with open(dataset_file, "rb") as f:
            dataset = pickle.load(f)
        with open(splits_file, "rb") as f:
            splits = pickle.load(f)
            train, val, test = splits["train"], splits["val"], splits["test"]

    print(f"  Users: {dataset.num_users}")
    print(f"  Posts: {dataset.num_posts}")
    print(f"  Train engagements: {len(train)}")
    print(f"  Val engagements: {len(val)}")
    print()

    # Create model
    model_size = "tiny" if args.quick else "small"
    model_config = create_model_config(model_size)
    print(f"Model ({model_size}):")
    print(f"  Embedding size: {model_config.emb_size}")
    print(f"  Layers: {model_config.model.num_layers}")
    print()

    # Create adapter
    print("Creating adapter...")
    adapter = SyntheticTwitterPhoenixAdapter(dataset, model_config)
    adapter.set_splits(train, val, test)
    print("  Adapter ready!")
    print()

    # Create trainer
    print("Initializing trainer...")
    trainer = SyntheticTrainer(
        model_config, dataset, adapter,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )
    print("  Trainer ready!")
    print()

    # Training loop
    max_batches = args.max_batches or (50 if args.quick else None)
    checkpoint_dir = Path("models/synthetic_twitter")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    print()

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc = trainer.train_epoch(max_batches=max_batches)

        # Evaluate
        val_ndcg, val_hit = trainer.evaluate(num_samples=200)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f}, acc={train_acc:.4f}, "
              f"val_ndcg={val_ndcg:.4f}, val_hit={val_hit:.4f} "
              f"({epoch_time:.1f}s)")

        # Save best
        if val_ndcg > trainer.best_val_ndcg:
            trainer.best_val_ndcg = val_ndcg
            trainer.save(str(checkpoint_dir / "best_model.pkl"))
            print(f"  -> New best! Saved to {checkpoint_dir}/best_model.pkl")

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Val NDCG@3: {trainer.best_val_ndcg:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
