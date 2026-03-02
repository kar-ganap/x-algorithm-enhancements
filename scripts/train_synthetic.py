#!/usr/bin/env python3
"""Train Phoenix model on synthetic Twitter-like dataset.

Usage:
    uv run python scripts/train_synthetic.py [--epochs N] [--quick]
"""

import argparse
import pickle
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

import jax
import jax.numpy as jnp
import numpy as np
import optax

from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter
from enhancements.data.synthetic_twitter import (
    SyntheticTwitterDataset,
    SyntheticTwitterGenerator,
    create_train_val_test_split,
)
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig, RecsysBatch, RecsysEmbeddings
from phoenix.runners import ModelRunner, RecsysInferenceRunner


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


def make_block_aware_loss_fn(
    forward_fn: Callable,
    compute_embeddings_fn: Callable,
    margin: float = 1.0,
):
    """Create block-aware contrastive loss function.

    Enforces: score(non_blocked_post) > score(blocked_post) + margin

    Args:
        forward_fn: Model forward function
        compute_embeddings_fn: Function to compute embeddings
        margin: Margin for contrastive loss
    """

    def loss_fn(
        params: dict[str, Any],
        batch: RecsysBatch,
        block_labels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute block-aware contrastive loss.

        Args:
            params: {'model': ..., 'embeddings': ...}
            batch: Input batch with 2 candidates per user:
                   candidate 0 = blocked author's post
                   candidate 1 = non-blocked author's post
            block_labels: [batch_size, 2] where [:, 0]=1 (blocked), [:, 1]=0 (non-blocked)

        Returns:
            (loss, metrics_dict)
        """
        model_params = params["model"]
        emb_params = params["embeddings"]

        # Compute embeddings
        embeddings = compute_embeddings_fn(emb_params, batch)

        # Forward pass
        output = forward_fn(model_params, batch, embeddings)
        logits = output.scores  # [batch, 2, actions]

        # Get scores (mean across actions)
        scores = jnp.mean(logits, axis=-1)  # [batch, 2]

        blocked_scores = scores[:, 0]     # Score for blocked author's post
        non_blocked_scores = scores[:, 1]  # Score for non-blocked author's post

        # Margin ranking loss: non_blocked should be > blocked + margin
        # Loss = max(0, margin - (non_blocked - blocked))
        diff = non_blocked_scores - blocked_scores
        loss = jnp.mean(jnp.maximum(0.0, margin - diff))

        # Track accuracy (% where non-blocked > blocked)
        correct = non_blocked_scores > blocked_scores
        accuracy = jnp.mean(correct.astype(jnp.float32))

        # Track margin satisfaction
        margin_satisfied = diff > margin
        margin_rate = jnp.mean(margin_satisfied.astype(jnp.float32))

        return loss, {
            "block_loss": loss,
            "block_acc": accuracy,
            "margin_rate": margin_rate,
            "avg_diff": jnp.mean(diff),
        }

    return loss_fn


def make_history_contrastive_loss_fn(
    forward_fn: Callable,
    compute_embeddings_fn: Callable,
    margin: float = 0.5,
):
    """Create history-topic contrastive loss function.

    Enforces: score(post | matching_history) > score(post | mismatched_history) + margin

    This teaches the transformer to use history content to influence scores.
    """

    def loss_fn(
        params: dict[str, Any],
        matching_batch: RecsysBatch,
        mismatched_batch: RecsysBatch,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute history contrastive loss."""
        model_params = params["model"]
        emb_params = params["embeddings"]

        # Forward pass with matching history
        matching_emb = compute_embeddings_fn(emb_params, matching_batch)
        matching_output = forward_fn(model_params, matching_batch, matching_emb)
        matching_scores = jnp.mean(matching_output.scores, axis=-1)[:, 0]  # [batch]

        # Forward pass with mismatched history
        mismatched_emb = compute_embeddings_fn(emb_params, mismatched_batch)
        mismatched_output = forward_fn(model_params, mismatched_batch, mismatched_emb)
        mismatched_scores = jnp.mean(mismatched_output.scores, axis=-1)[:, 0]  # [batch]

        # Margin ranking loss: matching should score higher
        diff = matching_scores - mismatched_scores
        loss = jnp.mean(jnp.maximum(0.0, margin - diff))

        # Track accuracy
        correct = matching_scores > mismatched_scores
        accuracy = jnp.mean(correct.astype(jnp.float32))

        return loss, {
            "history_loss": loss,
            "history_acc": accuracy,
            "history_diff": jnp.mean(diff),
        }

    return loss_fn


def make_multitask_loss_fn(
    forward_fn: Callable,
    compute_embeddings_fn: Callable,
    num_archetypes: int = 6,
    num_actions: int = 18,
    bpr_weight: float = 1.0,
    bce_weight: float = 1.0,
    cls_weight: float = 1.0,
    action_pred_weight: float = 2.0,
):
    """Create multi-task loss function with BPR + BCE + Classification + Action Prediction.

    Args:
        forward_fn: Model forward function
        compute_embeddings_fn: Function to compute embeddings
        num_archetypes: Number of user archetypes to classify
        num_actions: Number of actions to predict
        bpr_weight: Weight for BPR ranking loss
        bce_weight: Weight for BCE action prediction loss (from model output)
        cls_weight: Weight for archetype classification loss
        action_pred_weight: Weight for user-based action rate prediction
    """

    def loss_fn(
        params: dict[str, Any],
        batch: RecsysBatch,
        labels: jnp.ndarray,
        action_labels: jnp.ndarray,
        archetype_labels: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Compute multi-task loss.

        Args:
            params: {'model': ..., 'embeddings': ..., 'classifier': ..., 'action_predictor': ...}
            batch: Input batch
            labels: [batch_size, num_candidates] binary labels
            action_labels: [batch_size, num_actions] soft action probability labels
            archetype_labels: [batch_size] archetype indices

        Returns:
            (loss, metrics_dict)
        """
        model_params = params["model"]
        emb_params = params["embeddings"]
        cls_params = params["classifier"]
        action_pred_params = params["action_predictor"]

        # Compute embeddings from learnable params
        embeddings = compute_embeddings_fn(emb_params, batch)

        # Forward pass
        output = forward_fn(model_params, batch, embeddings)
        logits = output.scores  # [batch, candidates, actions]

        # === BPR Loss for ranking ===
        scores = jnp.mean(logits, axis=-1)  # [batch, candidates]
        pos_scores = scores[:, 0:1]  # [batch, 1]
        neg_scores = scores[:, 1:]   # [batch, num_negs]

        diff = pos_scores - neg_scores
        bpr_loss = -jnp.mean(jax.nn.log_sigmoid(diff))

        pos_wins = jnp.all(pos_scores > neg_scores, axis=-1)
        accuracy = jnp.mean(pos_wins.astype(jnp.float32))

        # === BCE Loss for model's action prediction ===
        pos_logits = logits[:, 0, :18]  # [batch, 18 actions]
        bce_loss = -jnp.mean(
            action_labels * jax.nn.log_sigmoid(pos_logits)
            + (1 - action_labels) * jax.nn.log_sigmoid(-pos_logits)
        )

        pred_actions = jax.nn.sigmoid(pos_logits) > 0.5
        action_acc = jnp.mean((pred_actions == (action_labels > 0.5)).astype(jnp.float32))

        # === Archetype Classification Loss ===
        user_emb = embeddings.user_embeddings[:, 0, :]  # [batch, emb_size]

        cls_weight_matrix = cls_params["weight"]  # [emb_size, num_archetypes]
        cls_bias = cls_params["bias"]  # [num_archetypes]
        archetype_logits = jnp.dot(user_emb, cls_weight_matrix) + cls_bias

        cls_loss = optax.softmax_cross_entropy_with_integer_labels(
            archetype_logits, archetype_labels
        )
        cls_loss = jnp.mean(cls_loss)

        pred_archetypes = jnp.argmax(archetype_logits, axis=-1)
        cls_acc = jnp.mean((pred_archetypes == archetype_labels).astype(jnp.float32))

        # === User-based Action Rate Prediction ===
        # This head predicts action rates directly from user embeddings
        # This forces the model to learn archetype-specific action patterns
        action_weight = action_pred_params["weight"]  # [emb_size, num_actions]
        action_bias = action_pred_params["bias"]  # [num_actions]
        action_logits_from_user = jnp.dot(user_emb, action_weight) + action_bias  # [batch, num_actions]

        # MSE loss between predicted and ground truth action rates
        pred_action_rates = jax.nn.sigmoid(action_logits_from_user)
        action_pred_loss = jnp.mean((pred_action_rates - action_labels) ** 2)

        # Combined loss
        total_loss = (
            bpr_weight * bpr_loss
            + bce_weight * bce_loss
            + cls_weight * cls_loss
            + action_pred_weight * action_pred_loss
        )

        # Per-sample loss for hard example mining
        per_sample_bpr = -jnp.mean(jax.nn.log_sigmoid(diff), axis=-1)
        per_sample_loss = per_sample_bpr

        return total_loss, {
            "loss": total_loss,
            "bpr_loss": bpr_loss,
            "bce_loss": bce_loss,
            "cls_loss": cls_loss,
            "action_pred_loss": action_pred_loss,
            "accuracy": accuracy,
            "action_acc": action_acc,
            "cls_acc": cls_acc,
            "per_sample_loss": per_sample_loss,
        }

    return loss_fn


def make_train_step(loss_fn, optimizer):
    """Create JIT-compiled training step."""

    @jax.jit
    def train_step(params, opt_state, batch, labels, action_labels, archetype_labels):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, batch, labels, action_labels, archetype_labels
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return train_step


def make_block_train_step(block_loss_fn, optimizer):
    """Create JIT-compiled training step for block-aware loss."""

    @jax.jit
    def train_step(params, opt_state, batch, block_labels):
        (loss, metrics), grads = jax.value_and_grad(block_loss_fn, has_aux=True)(
            params, batch, block_labels
        )

        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, metrics

    return train_step


def make_history_train_step(history_loss_fn, optimizer):
    """Create JIT-compiled training step for history contrastive loss."""

    @jax.jit
    def train_step(params, opt_state, matching_batch, mismatched_batch):
        (loss, metrics), grads = jax.value_and_grad(history_loss_fn, has_aux=True)(
            params, matching_batch, mismatched_batch
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
    """Trainer for Phoenix on synthetic Twitter data with enhanced learning."""

    def __init__(
        self,
        model_config: PhoenixModelConfig,
        dataset: SyntheticTwitterDataset,
        adapter: SyntheticTwitterPhoenixAdapter,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        neg_ratio: int = 7,
        hard_negative_ratio: float = 0.5,
        num_archetypes: int = 6,
        bpr_weight: float = 1.0,
        bce_weight: float = 0.5,
        cls_weight: float = 1.0,
    ):
        self.model_config = model_config
        self.dataset = dataset
        self.adapter = adapter
        self.batch_size = batch_size
        self.neg_ratio = neg_ratio
        self.hard_negative_ratio = hard_negative_ratio
        self.num_archetypes = num_archetypes

        # Create model
        print("  Creating model runner...")
        self.model_runner = ModelRunner(model=model_config)
        self.runner = RecsysInferenceRunner(runner=self.model_runner, name="training")
        print("  Initializing runner...")
        self.runner.initialize()
        print("  Runner initialized!")

        # Initialize classifier head parameters
        emb_size = model_config.emb_size
        num_actions = 18  # First 18 actions for prediction
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        cls_weight_init = jax.random.normal(subkey, (emb_size, num_archetypes)) * 0.01
        cls_bias_init = jnp.zeros(num_archetypes)

        # Initialize action predictor head (predicts action rates from user embeddings)
        rng_key, subkey = jax.random.split(rng_key)
        action_weight_init = jax.random.normal(subkey, (emb_size, num_actions)) * 0.01
        action_bias_init = jnp.zeros(num_actions)

        # Combined params
        self.params = {
            "model": self.runner.params,
            "embeddings": self.adapter.get_embedding_params(),
            "classifier": {
                "weight": cls_weight_init,
                "bias": cls_bias_init,
            },
            "action_predictor": {
                "weight": action_weight_init,
                "bias": action_bias_init,
            },
        }

        # Optimizer
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=learning_rate, weight_decay=1e-4),
        )
        self.opt_state = self.optimizer.init(self.params)

        # Loss function (multi-task: BPR + BCE + Classification + Action Prediction)
        self._loss_fn = make_multitask_loss_fn(
            self.runner.rank_candidates,
            self.adapter.compute_embeddings_from_params,
            num_archetypes=num_archetypes,
            num_actions=num_actions,
            bpr_weight=bpr_weight,
            bce_weight=bce_weight,
            cls_weight=cls_weight,
            action_pred_weight=2.0,  # Weight for user-based action rate prediction
        )
        self._train_step = make_train_step(self._loss_fn, self.optimizer)

        # Block-aware contrastive loss
        self._block_loss_fn = make_block_aware_loss_fn(
            self.runner.rank_candidates,
            self.adapter.compute_embeddings_from_params,
            margin=0.5,  # Require blocked posts to score 0.5 lower
        )
        self._block_train_step = make_block_train_step(self._block_loss_fn, self.optimizer)

        # History-topic contrastive loss
        self._history_loss_fn = make_history_contrastive_loss_fn(
            self.runner.rank_candidates,
            self.adapter.compute_embeddings_from_params,
            margin=0.3,  # Require matching history to score higher
        )
        self._history_train_step = make_history_train_step(self._history_loss_fn, self.optimizer)

        # Hard example mining: sample weights
        num_train = len(adapter.train_engagements)
        self.sample_weights = np.ones(num_train, dtype=np.float32)
        self.sample_losses = np.zeros(num_train, dtype=np.float32)

        # Tracking
        self.best_val_ndcg = 0.0

    def train_epoch(
        self,
        max_batches: int | None = None,
        use_hard_mining: bool = True,
        use_block_training: bool = True,
        use_history_training: bool = True,
        block_steps_per_batch: int = 1,
        history_steps_per_batch: int = 1,
    ) -> tuple[float, float, float, float, float]:
        """Train for one epoch.

        Returns:
            (avg_loss, avg_accuracy, avg_cls_acc, avg_block_acc, avg_history_acc)
        """
        num_train = len(self.adapter.train_engagements)
        num_batches = num_train // self.batch_size
        if max_batches:
            num_batches = min(num_batches, max_batches)

        total_loss = 0.0
        total_acc = 0.0
        total_cls_acc = 0.0
        total_block_acc = 0.0
        total_history_acc = 0.0
        block_steps = 0
        history_steps = 0

        # Use sample weights for hard example mining
        weights = self.sample_weights if use_hard_mining else None

        for step in range(num_batches):
            # === Regular training step ===
            result = self.adapter.get_training_batch(
                batch_size=self.batch_size,
                neg_ratio=self.neg_ratio,
                hard_negative_ratio=self.hard_negative_ratio,
                sample_weights=weights,
            )
            batch, _, labels, action_labels, archetype_labels, sample_indices = result

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
            action_labels = jnp.array(action_labels)
            archetype_labels = jnp.array(archetype_labels)

            self.params, self.opt_state, metrics = self._train_step(
                self.params, self.opt_state, batch, labels, action_labels, archetype_labels
            )

            # Update sample weights for hard example mining
            if use_hard_mining and "per_sample_loss" in metrics:
                per_sample_loss = np.array(metrics["per_sample_loss"])
                for i, idx in enumerate(sample_indices):
                    if i < len(per_sample_loss):
                        # Exponential moving average of sample loss
                        self.sample_losses[idx] = 0.9 * self.sample_losses[idx] + 0.1 * per_sample_loss[i]

            total_loss += float(metrics["loss"])
            total_acc += float(metrics["accuracy"])
            total_cls_acc += float(metrics.get("cls_acc", 0))

            # === Block-aware training steps ===
            if use_block_training:
                for _ in range(block_steps_per_batch):
                    block_result = self.adapter.get_block_aware_batch(batch_size=self.batch_size)
                    if block_result is not None:
                        block_batch, _, block_labels = block_result

                        # Convert to JAX
                        block_batch = RecsysBatch(
                            user_hashes=jnp.array(block_batch.user_hashes),
                            history_post_hashes=jnp.array(block_batch.history_post_hashes),
                            history_author_hashes=jnp.array(block_batch.history_author_hashes),
                            history_actions=jnp.array(block_batch.history_actions),
                            history_product_surface=jnp.array(block_batch.history_product_surface),
                            candidate_post_hashes=jnp.array(block_batch.candidate_post_hashes),
                            candidate_author_hashes=jnp.array(block_batch.candidate_author_hashes),
                            candidate_product_surface=jnp.array(block_batch.candidate_product_surface),
                        )
                        block_labels = jnp.array(block_labels)

                        self.params, self.opt_state, block_metrics = self._block_train_step(
                            self.params, self.opt_state, block_batch, block_labels
                        )

                        total_block_acc += float(block_metrics.get("block_acc", 0))
                        block_steps += 1

            # === History-topic contrastive training steps ===
            if use_history_training:
                for _ in range(history_steps_per_batch):
                    history_result = self.adapter.get_history_contrastive_batch(batch_size=self.batch_size)
                    if history_result is not None:
                        matching_batch, mismatched_batch, _ = history_result

                        # Convert to JAX
                        matching_batch = RecsysBatch(
                            user_hashes=jnp.array(matching_batch.user_hashes),
                            history_post_hashes=jnp.array(matching_batch.history_post_hashes),
                            history_author_hashes=jnp.array(matching_batch.history_author_hashes),
                            history_actions=jnp.array(matching_batch.history_actions),
                            history_product_surface=jnp.array(matching_batch.history_product_surface),
                            candidate_post_hashes=jnp.array(matching_batch.candidate_post_hashes),
                            candidate_author_hashes=jnp.array(matching_batch.candidate_author_hashes),
                            candidate_product_surface=jnp.array(matching_batch.candidate_product_surface),
                        )
                        mismatched_batch = RecsysBatch(
                            user_hashes=jnp.array(mismatched_batch.user_hashes),
                            history_post_hashes=jnp.array(mismatched_batch.history_post_hashes),
                            history_author_hashes=jnp.array(mismatched_batch.history_author_hashes),
                            history_actions=jnp.array(mismatched_batch.history_actions),
                            history_product_surface=jnp.array(mismatched_batch.history_product_surface),
                            candidate_post_hashes=jnp.array(mismatched_batch.candidate_post_hashes),
                            candidate_author_hashes=jnp.array(mismatched_batch.candidate_author_hashes),
                            candidate_product_surface=jnp.array(mismatched_batch.candidate_product_surface),
                        )

                        self.params, self.opt_state, history_metrics = self._history_train_step(
                            self.params, self.opt_state, matching_batch, mismatched_batch
                        )

                        total_history_acc += float(history_metrics.get("history_acc", 0))
                        history_steps += 1

            if (step + 1) % 50 == 0:
                bce = float(metrics.get("bce_loss", 0))
                cls = float(metrics.get("cls_loss", 0))
                cls_acc = float(metrics.get("cls_acc", 0))
                block_acc = total_block_acc / max(1, block_steps)
                history_acc = total_history_acc / max(1, history_steps)
                print(f"    Step {step+1}/{num_batches}: loss={float(metrics['loss']):.4f}, "
                      f"cls_acc={cls_acc:.2%}, block_acc={block_acc:.2%}, hist_acc={history_acc:.2%}")

        # Update sample weights based on accumulated losses (for next epoch)
        if use_hard_mining:
            # Higher loss = higher weight (sample more frequently)
            # Use softmax-like weighting to prevent extreme values
            loss_scaled = self.sample_losses / (self.sample_losses.max() + 1e-6)
            self.sample_weights = 1.0 + loss_scaled  # Range [1, 2]

        avg_block_acc = total_block_acc / max(1, block_steps)
        avg_history_acc = total_history_acc / max(1, history_steps)
        return total_loss / num_batches, total_acc / num_batches, total_cls_acc / num_batches, avg_block_acc, avg_history_acc

    def evaluate(self, num_samples: int = 200) -> tuple[float, float]:
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
                "classifier": jax.device_get(self.params["classifier"]),
                "action_predictor": jax.device_get(self.params["action_predictor"]),
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
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--bpr-weight", type=float, default=1.0, help="BPR loss weight")
    parser.add_argument("--bce-weight", type=float, default=0.5, help="BCE loss weight")
    parser.add_argument("--cls-weight", type=float, default=1.0, help="Classification loss weight")
    parser.add_argument("--hard-neg-ratio", type=float, default=0.5, help="Fraction of hard negatives")
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
    print(f"  Loss weights: BPR={args.bpr_weight}, BCE={args.bce_weight}, CLS={args.cls_weight}")
    print(f"  Hard negative ratio: {args.hard_neg_ratio}")
    trainer = SyntheticTrainer(
        model_config, dataset, adapter,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        hard_negative_ratio=args.hard_neg_ratio,
        bpr_weight=args.bpr_weight,
        bce_weight=args.bce_weight,
        cls_weight=args.cls_weight,
    )
    print("  Trainer ready!")
    print()

    # Training loop
    max_batches = args.max_batches or (50 if args.quick else None)
    checkpoint_dir = Path("models/synthetic_twitter")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    print(f"Early stopping patience: {args.patience} epochs")
    print()

    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss, train_acc, train_cls_acc, train_block_acc, train_hist_acc = trainer.train_epoch(
            max_batches=max_batches,
            use_block_training=True,
            use_history_training=True,
            block_steps_per_batch=1,
            history_steps_per_batch=1,
        )

        # Evaluate
        val_ndcg, val_hit = trainer.evaluate(num_samples=200)

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch}/{args.epochs}: "
              f"loss={train_loss:.4f}, cls_acc={train_cls_acc:.2%}, "
              f"block={train_block_acc:.2%}, hist={train_hist_acc:.2%}, "
              f"val_ndcg={val_ndcg:.4f} "
              f"({epoch_time:.1f}s)")

        # Save best
        if val_ndcg > trainer.best_val_ndcg:
            trainer.best_val_ndcg = val_ndcg
            trainer.save(str(checkpoint_dir / "best_model.pkl"))
            print(f"  -> New best! Saved to {checkpoint_dir}/best_model.pkl")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: No improvement for {args.patience} epochs")
                break

    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Best Val NDCG@3: {trainer.best_val_ndcg:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
