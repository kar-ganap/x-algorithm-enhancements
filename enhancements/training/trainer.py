"""Phoenix model trainer for MovieLens.

Trains Phoenix recommendation model on MovieLens data with:
- Binary cross-entropy loss on candidate selection
- BPR (Bayesian Personalized Ranking) loss for better ranking
- Validation metrics: NDCG@K, Hit Rate@K
"""

import pickle
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enhancements.data.movielens import MovieLensDataset
from enhancements.data.movielens_adapter import MovieLensPhoenixAdapter
from phoenix.recsys_model import PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ModelRunner, RecsysInferenceRunner


# Register RecsysEmbeddings as a JAX pytree so it can be used with JIT
def _recsys_embeddings_flatten(emb: RecsysEmbeddings):
    """Flatten RecsysEmbeddings for JAX pytree."""
    return (
        emb.user_embeddings,
        emb.history_post_embeddings,
        emb.candidate_post_embeddings,
        emb.history_author_embeddings,
        emb.candidate_author_embeddings,
    ), None


def _recsys_embeddings_unflatten(aux, children):
    """Unflatten RecsysEmbeddings from JAX pytree."""
    return RecsysEmbeddings(
        user_embeddings=children[0],
        history_post_embeddings=children[1],
        candidate_post_embeddings=children[2],
        history_author_embeddings=children[3],
        candidate_author_embeddings=children[4],
    )


# Register the pytree (only if not already registered)
try:
    jax.tree_util.register_pytree_node(
        RecsysEmbeddings,
        _recsys_embeddings_flatten,
        _recsys_embeddings_unflatten,
    )
except ValueError:
    pass  # Already registered


class LossType(Enum):
    """Loss function type for training."""
    BCE = "bce"  # Binary Cross-Entropy (original)
    BPR = "bpr"  # Bayesian Personalized Ranking (pairwise)


def _make_loss_fn_with_embeddings(
    forward_fn: Callable,
    compute_embeddings_fn: Callable,
    loss_type: LossType = LossType.BCE,
):
    """Create a pure loss function that also learns embeddings.

    Args:
        forward_fn: The haiku-transformed forward function (apply method)
        compute_embeddings_fn: Function to compute embeddings from params
        loss_type: Type of loss function (BCE or BPR)

    Returns:
        A pure function that computes loss and metrics
    """
    def loss_fn(
        params: dict[str, Any],
        batch: RecsysBatch,
        labels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute training loss with learnable embeddings.

        Args:
            params: Combined params dict with 'model' and 'embeddings' keys
            batch: Input batch (contains IDs for embedding lookup)
            labels: Binary labels [batch_size, num_candidates]

        Returns:
            Tuple of (loss, metrics_dict)
        """
        model_params = params["model"]
        emb_params = params["embeddings"]

        # Compute embeddings from learnable parameters
        embeddings = compute_embeddings_fn(emb_params, batch)

        # Forward pass through the model
        output = forward_fn(model_params, batch, embeddings)

        # Get scores for primary action (action 0)
        logits = output.scores[..., 0]  # [batch_size, num_candidates]

        if loss_type == LossType.BPR:
            # BPR Loss: -log(sigmoid(positive_score - negative_score))
            # For each sample, compare positive against all negatives
            # labels has shape [batch_size, num_candidates] with one 1 per row

            # Get positive scores: where label == 1
            # positive_mask: [batch_size, num_candidates]
            positive_mask = labels > 0.5

            # Get the positive score for each sample
            # Use sum with mask (since there's exactly one positive per row)
            positive_scores = jnp.sum(logits * positive_mask, axis=-1, keepdims=True)  # [batch_size, 1]

            # Negative mask: where label == 0
            negative_mask = labels < 0.5

            # Compute score differences: positive - negative for all negatives
            # Shape: [batch_size, num_candidates]
            score_diff = positive_scores - logits  # broadcast: [batch_size, 1] - [batch_size, num_candidates]

            # BPR loss: -log(sigmoid(diff)) for negatives only
            # log_sigmoid is numerically stable
            bpr_loss_per_pair = -jax.nn.log_sigmoid(score_diff)

            # Mask to only include negative pairs and average
            # Sum over negatives, divide by number of negatives
            num_negatives = jnp.sum(negative_mask, axis=-1, keepdims=True)  # [batch_size, 1]
            masked_loss = bpr_loss_per_pair * negative_mask
            loss_per_sample = jnp.sum(masked_loss, axis=-1) / (num_negatives.squeeze(-1) + 1e-8)
            loss = jnp.mean(loss_per_sample)
        else:
            # BCE Loss (original)
            # Binary cross-entropy loss (numerically stable version)
            # Use log-sum-exp trick: BCE = max(logits, 0) - logits * labels + log(1 + exp(-|logits|))
            # Clamp logits to prevent overflow
            logits_clipped = jnp.clip(logits, -20, 20)
            max_val = jnp.maximum(logits_clipped, 0)
            bce = max_val - logits_clipped * labels + jnp.log1p(jnp.exp(-jnp.abs(logits_clipped)))
            loss = jnp.mean(bce)

        # Compute accuracy (which candidate has highest score)
        predictions = jnp.argmax(logits, axis=-1)
        targets = jnp.argmax(labels, axis=-1)
        accuracy = jnp.mean(predictions == targets)

        metrics = {
            "loss": loss,
            "accuracy": accuracy,
        }

        return loss, metrics

    return loss_fn


def _make_train_step_with_embeddings(loss_fn: Callable, optimizer: optax.GradientTransformation):
    """Create a JIT-compiled training step that learns embeddings.

    Args:
        loss_fn: The loss function (takes combined params)
        optimizer: The optax optimizer

    Returns:
        A JIT-compiled training step function
    """
    @jax.jit
    def train_step(
        params: dict[str, Any],
        opt_state: Any,
        batch: RecsysBatch,
        labels: jnp.ndarray,
    ) -> tuple[dict[str, Any], Any, dict[str, jnp.ndarray]]:
        """Single training step with learnable embeddings."""
        def compute_loss(p):
            return loss_fn(p, batch, labels)

        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(params)

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return train_step


class TrainingMetrics(NamedTuple):
    """Metrics from training."""
    epoch: int
    train_loss: float
    val_loss: float
    val_ndcg: float
    val_hit_rate: float
    epoch_time_s: float


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    learning_rate: float = 5e-4  # Lower LR for stability
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 20
    num_negatives: int = 7  # Random negatives (used when not using in-batch)
    use_in_batch_negatives: bool = True  # Use other positives in batch as negatives
    max_batches_per_epoch: int | None = None  # Limit batches for testing

    # Loss function
    loss_type: LossType = LossType.BPR  # BPR for ranking, BCE for classification

    # Evaluation
    eval_batch_size: int = 64
    eval_every_n_epochs: int = 1

    # Early stopping
    early_stopping_patience: int | None = None  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 0.001  # Minimum improvement to count as progress

    # Checkpointing
    checkpoint_dir: str = "models/movielens_phoenix"
    save_best_only: bool = True

    # Logging
    log_every_n_steps: int = 50


class PhoenixTrainer:
    """Trainer for Phoenix model on MovieLens."""

    def __init__(
        self,
        model_config: PhoenixModelConfig,
        dataset: MovieLensDataset,
        training_config: TrainingConfig | None = None,
    ):
        """Initialize trainer.

        Args:
            model_config: Phoenix model configuration
            dataset: MovieLens dataset
            training_config: Training configuration
        """
        self.model_config = model_config
        self.dataset = dataset
        self.config = training_config or TrainingConfig()

        print("  Creating adapter...")
        # Create adapter
        self.adapter = MovieLensPhoenixAdapter(dataset, model_config)

        print("  Creating model runner...")
        # Create model runner
        self.model_runner = ModelRunner(model=model_config)
        self.runner = RecsysInferenceRunner(runner=self.model_runner, name="training")
        print("  Initializing runner (this may take a while for JIT compilation)...")
        self.runner.initialize()
        print("  Runner initialized!")

        # Create combined parameters (model + embeddings)
        print("  Initializing learnable embeddings...")
        self.params = {
            "model": self.runner.params,
            "embeddings": self.adapter.get_embedding_params(),
        }

        print("  Creating optimizer...")
        # Create optimizer for combined params with gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),  # Prevent gradient explosion
            optax.adamw(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            ),
        )
        self.opt_state = self.optimizer.init(self.params)
        print("  Optimizer initialized!")

        # Create JIT-compiled training functions with learnable embeddings
        print(f"  Creating JIT-compiled training step (loss={self.config.loss_type.value})...")
        self._loss_fn = _make_loss_fn_with_embeddings(
            self.runner.rank_candidates,
            self.adapter.compute_embeddings_from_params,
            loss_type=self.config.loss_type,
        )
        self._train_step = _make_train_step_with_embeddings(self._loss_fn, self.optimizer)
        print("  Training functions ready!")

        # Training state
        self.current_epoch = 0
        self.best_val_ndcg = 0.0
        self.metrics_history: list[TrainingMetrics] = []

    def train_epoch(self, max_batches: int | None = None) -> float:
        """Train for one epoch.

        Args:
            max_batches: Optional limit on number of batches (for quick testing)

        Returns:
            Average training loss for the epoch
        """
        num_batches = len(self.dataset.train_ratings) // self.config.batch_size
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)
        total_loss = 0.0
        total_acc = 0.0

        for step in range(num_batches):
            # Get training batch (embeddings computed from learnable params now)
            batch, _, labels = self.adapter.get_training_batch(
                batch_size=self.config.batch_size,
                num_negatives=self.config.num_negatives,
                use_in_batch_negatives=self.config.use_in_batch_negatives,
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
            labels = jnp.array(labels)

            # Training step (embeddings computed inside from learnable params)
            self.params, self.opt_state, metrics = self._train_step(
                self.params,
                self.opt_state,
                batch,
                labels,
            )

            total_loss += float(metrics["loss"])
            total_acc += float(metrics["accuracy"])

            if (step + 1) % self.config.log_every_n_steps == 0:
                avg_loss = total_loss / (step + 1)
                avg_acc = total_acc / (step + 1)
                print(f"  Step {step + 1}/{num_batches}: loss={avg_loss:.4f}, acc={avg_acc:.4f}")
                sys.stdout.flush()

        # Sync learned embeddings back to adapter (for evaluation)
        self.adapter.set_embedding_params(self.params["embeddings"])

        return total_loss / num_batches

    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate on validation set.

        Returns:
            Tuple of (val_loss, ndcg@k, hit_rate@k)
        """
        total_loss = 0.0
        total_ndcg = 0.0
        total_hit = 0.0
        num_samples = 0

        # Evaluate on validation ratings
        val_ratings = self.dataset.val_ratings
        num_batches = min(100, len(val_ratings) // self.config.eval_batch_size)

        for i in range(num_batches):
            start_idx = i * self.config.eval_batch_size
            batch_ratings = val_ratings[start_idx:start_idx + self.config.eval_batch_size]

            for rating in batch_ratings:
                # Create eval batch (uses learned embeddings synced from params)
                batch, embeddings, labels = self.adapter.get_training_example(
                    rating, num_negatives=self.config.num_negatives
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
                embeddings = RecsysEmbeddings(
                    user_embeddings=jnp.array(embeddings.user_embeddings),
                    history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
                    candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
                    history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
                    candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
                )

                # Forward pass using learned model params
                output = self.runner.rank_candidates(
                    self.params["model"], batch, embeddings
                )
                scores = np.array(output.scores[0, :, 0])

                # Compute loss for validation
                logits = output.scores[0, :, 0]
                probs = jax.nn.sigmoid(logits)
                eps = 1e-7
                bce = -labels * jnp.log(probs + eps) - (1 - labels) * jnp.log(1 - probs + eps)
                loss = float(jnp.mean(bce))

                # Compute metrics
                total_loss += loss

                # NDCG@K (K=3 for 8 candidates)
                k = 3
                ndcg = self._compute_ndcg(scores, labels, k)
                total_ndcg += ndcg

                # Hit Rate@K
                hit = self._compute_hit_rate(scores, labels, k)
                total_hit += hit

                num_samples += 1

        if num_samples == 0:
            return 0.0, 0.0, 0.0

        return (
            total_loss / num_samples,
            total_ndcg / num_samples,
            total_hit / num_samples,
        )

    def _compute_ndcg(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> float:
        """Compute NDCG@K."""
        # Get top-k indices by score
        top_k_indices = np.argsort(-scores)[:k]

        # DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            rel = labels[idx]
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG (labels sorted in descending order)
        ideal_labels = np.sort(labels)[::-1][:k]
        idcg = 0.0
        for i, rel in enumerate(ideal_labels):
            idcg += rel / np.log2(i + 2)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def _compute_hit_rate(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        k: int,
    ) -> float:
        """Compute Hit Rate@K (is positive in top-k?)."""
        top_k_indices = np.argsort(-scores)[:k]
        positive_idx = np.argmax(labels)
        return 1.0 if positive_idx in top_k_indices else 0.0

    def evaluate_test(self) -> tuple[float, float, float]:
        """Evaluate on test set.

        Returns:
            Tuple of (test_loss, ndcg@k, hit_rate@k)
        """
        total_loss = 0.0
        total_ndcg = 0.0
        total_hit = 0.0
        num_samples = 0

        # Evaluate on test ratings
        test_ratings = self.dataset.test_ratings
        num_batches = min(100, len(test_ratings) // self.config.eval_batch_size)

        for i in range(num_batches):
            start_idx = i * self.config.eval_batch_size
            batch_ratings = test_ratings[start_idx:start_idx + self.config.eval_batch_size]

            for rating in batch_ratings:
                # Create eval batch (uses learned embeddings synced from params)
                batch, embeddings, labels = self.adapter.get_training_example(
                    rating, num_negatives=self.config.num_negatives
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
                embeddings = RecsysEmbeddings(
                    user_embeddings=jnp.array(embeddings.user_embeddings),
                    history_post_embeddings=jnp.array(embeddings.history_post_embeddings),
                    candidate_post_embeddings=jnp.array(embeddings.candidate_post_embeddings),
                    history_author_embeddings=jnp.array(embeddings.history_author_embeddings),
                    candidate_author_embeddings=jnp.array(embeddings.candidate_author_embeddings),
                )

                # Forward pass using learned model params
                output = self.runner.rank_candidates(
                    self.params["model"], batch, embeddings
                )
                scores = np.array(output.scores[0, :, 0])

                # Compute loss for test
                logits = output.scores[0, :, 0]
                probs = jax.nn.sigmoid(logits)
                eps = 1e-7
                bce = -labels * jnp.log(probs + eps) - (1 - labels) * jnp.log(1 - probs + eps)
                loss = float(jnp.mean(bce))

                # Compute metrics
                total_loss += loss

                # NDCG@K (K=3 for 8 candidates)
                k = 3
                ndcg = self._compute_ndcg(scores, labels, k)
                total_ndcg += ndcg

                # Hit Rate@K
                hit = self._compute_hit_rate(scores, labels, k)
                total_hit += hit

                num_samples += 1

        if num_samples == 0:
            return 0.0, 0.0, 0.0

        return (
            total_loss / num_samples,
            total_ndcg / num_samples,
            total_hit / num_samples,
        )

    def train(self, resume_from: str | None = None) -> list[TrainingMetrics]:
        """Run full training loop.

        Args:
            resume_from: Optional checkpoint filename to resume from

        Returns:
            List of metrics for each epoch
        """
        start_epoch = 0
        epochs_without_improvement = 0

        # Resume from checkpoint if specified
        if resume_from is not None:
            checkpoint_path = Path(self.config.checkpoint_dir) / resume_from
            if checkpoint_path.exists():
                self.load_checkpoint(checkpoint_path)
                start_epoch = self.current_epoch + 1
                print(f"Resuming from epoch {start_epoch}")
            else:
                print(f"Checkpoint {resume_from} not found, starting fresh")

        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Weight decay: {self.config.weight_decay}")
        print(f"  Loss function: {self.config.loss_type.value.upper()}")
        if self.config.use_in_batch_negatives:
            print(f"  Negatives: in-batch ({self.config.batch_size - 1} per positive)")
        else:
            print(f"  Negatives: random ({self.config.num_negatives} per positive)")
        if self.config.early_stopping_patience:
            print(f"  Early stopping patience: {self.config.early_stopping_patience} epochs")
        if start_epoch > 0:
            print(f"  Resuming from epoch: {start_epoch + 1}")
        print()
        sys.stdout.flush()

        # Epoch 0: Evaluate before any training
        if start_epoch == 0:
            print("Epoch 0 (before training):")
            val_loss, val_ndcg, val_hit_rate = self.evaluate()
            print(f"  Val loss: {val_loss:.4f}, NDCG@3: {val_ndcg:.4f}, Hit@3: {val_hit_rate:.4f}")
            metrics = TrainingMetrics(
                epoch=0,
                train_loss=0.0,
                val_loss=val_loss,
                val_ndcg=val_ndcg,
                val_hit_rate=val_hit_rate,
                epoch_time_s=0.0,
            )
            self.metrics_history.append(metrics)
            self.best_val_ndcg = val_ndcg
            print()
            sys.stdout.flush()

        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            sys.stdout.flush()

            # Training
            train_loss = self.train_epoch(max_batches=self.config.max_batches_per_epoch)

            # Evaluation
            if (epoch + 1) % self.config.eval_every_n_epochs == 0:
                val_loss, val_ndcg, val_hit_rate = self.evaluate()

                epoch_time = time.time() - epoch_start

                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_ndcg=val_ndcg,
                    val_hit_rate=val_hit_rate,
                    epoch_time_s=epoch_time,
                )
                self.metrics_history.append(metrics)

                print(f"  Train loss: {train_loss:.4f}")
                print(f"  Val loss: {val_loss:.4f}, NDCG@3: {val_ndcg:.4f}, Hit@3: {val_hit_rate:.4f}")
                print(f"  Time: {epoch_time:.1f}s")
                sys.stdout.flush()

                # Check for improvement
                improvement = val_ndcg - self.best_val_ndcg
                if improvement > self.config.early_stopping_min_delta:
                    self.best_val_ndcg = val_ndcg
                    epochs_without_improvement = 0
                    if self.config.save_best_only:
                        self.save_checkpoint("best_model.pkl")
                        print(f"  Saved best model (NDCG: {val_ndcg:.4f})")
                        sys.stdout.flush()
                else:
                    epochs_without_improvement += 1
                    if self.config.early_stopping_patience:
                        print(f"  No improvement for {epochs_without_improvement}/{self.config.early_stopping_patience} epochs")

                # Save epoch checkpoint for resume capability
                self.save_checkpoint(f"epoch_{epoch + 1}.pkl")

                # Early stopping check
                if (self.config.early_stopping_patience and
                    epochs_without_improvement >= self.config.early_stopping_patience):
                    print()
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"No improvement in validation NDCG for {epochs_without_improvement} epochs")
                    break

            print()

        # Save final model
        self.save_checkpoint("final_model.pkl")
        print(f"Training complete. Best val NDCG: {self.best_val_ndcg:.4f}")

        # Evaluate on test set
        print()
        print("Evaluating on test set...")
        test_loss, test_ndcg, test_hit_rate = self.evaluate_test()
        print(f"  Test loss: {test_loss:.4f}")
        print(f"  Test NDCG@3: {test_ndcg:.4f}")
        print(f"  Test Hit@3: {test_hit_rate:.4f}")

        return self.metrics_history

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filepath = checkpoint_dir / filename

        checkpoint = {
            "params": self.params,  # Combined model + embedding params
            "opt_state": self.opt_state,
            "epoch": self.current_epoch,
            "best_val_ndcg": self.best_val_ndcg,
            "metrics_history": self.metrics_history,
            "model_config": self.model_config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)

        return filepath

    def load_checkpoint(self, filepath: str | Path) -> None:
        """Load model checkpoint.

        Args:
            filepath: Path to checkpoint file
        """
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]  # Combined model + embedding params
        self.opt_state = checkpoint["opt_state"]
        self.current_epoch = checkpoint["epoch"]
        self.best_val_ndcg = checkpoint["best_val_ndcg"]
        self.metrics_history = checkpoint.get("metrics_history", [])

        # Sync embedding params back to adapter
        self.adapter.set_embedding_params(self.params["embeddings"])

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best NDCG: {self.best_val_ndcg:.4f}")
