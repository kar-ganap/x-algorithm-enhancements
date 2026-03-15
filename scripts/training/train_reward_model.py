#!/usr/bin/env python3
"""Train contextual reward model on F2 synthetic data.

Establishes Phase 1 baseline by training archetype-specific reward weights
using Bradley-Terry preference learning on F2's ground truth data.

Usage:
    uv run python scripts/train_reward_model.py

Outputs:
    results/f4_phase1/
    ├── baseline_weights.npy       # Trained weights [K, 18]
    ├── training_metrics.json      # Loss/accuracy history
    ├── weight_analysis.json       # Per-archetype weight interpretation
    └── training_curves.png        # Loss/accuracy plots
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "phoenix"))

from enhancements.data import (
    ContentTopic,
    UserArchetype,
    get_engagement_probs,
)
from enhancements.reward_modeling import (
    ACTION_NAMES,
    NUM_ACTIONS,
    RewardWeights,
    compute_preference_accuracy,
    contextual_bradley_terry_loss,
)

# =============================================================================
# Data Generation
# =============================================================================


def create_preference_pairs_from_ground_truth(
    num_pairs_per_archetype: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create preference pairs directly from F2's ground truth engagement rules.

    For each archetype, generates pairs where the preferred item has higher
    expected engagement based on the ground truth rules.

    Returns:
        Tuple of (probs_preferred, probs_rejected, archetype_ids, confidences)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    for arch_idx, archetype in enumerate(archetypes):
        pairs_created = 0
        attempts = 0
        max_attempts = num_pairs_per_archetype * 10

        while pairs_created < num_pairs_per_archetype and attempts < max_attempts:
            attempts += 1

            # Sample two different topics
            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a = topics[topic_indices[0]]
            topic_b = topics[topic_indices[1]]

            # Get engagement probabilities
            probs_a = get_engagement_probs(archetype, topic_a)
            probs_b = get_engagement_probs(archetype, topic_b)

            # Convert to arrays
            arr_a = np.array(probs_a.to_array(), dtype=np.float32)
            arr_b = np.array(probs_b.to_array(), dtype=np.float32)

            # Compute "preference score" based on positive vs negative actions
            # Positive: indices 0-13, Negative: indices 14-17
            pos_a = arr_a[:14].sum()
            neg_a = arr_a[14:].sum()
            score_a = pos_a - neg_a * 2  # Weight negatives more heavily

            pos_b = arr_b[:14].sum()
            neg_b = arr_b[14:].sum()
            score_b = pos_b - neg_b * 2

            # Only create pair if there's a clear preference
            if abs(score_a - score_b) > 0.1:
                if score_a > score_b:
                    all_probs_pref.append(arr_a)
                    all_probs_rej.append(arr_b)
                else:
                    all_probs_pref.append(arr_b)
                    all_probs_rej.append(arr_a)

                all_arch_ids.append(arch_idx)
                all_confidences.append(min(1.0, abs(score_a - score_b)))
                pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


def create_within_topic_preference_pairs(
    num_pairs_per_archetype: int = 500,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create preference pairs within the same topic (harder task).

    For each archetype, creates pairs from the same topic where
    preference is based on action type emphasis.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    for arch_idx, archetype in enumerate(archetypes):
        for _ in range(num_pairs_per_archetype):
            # Sample one topic
            topic = topics[rng.integers(0, len(topics))]

            # Get base engagement probabilities
            base_probs = get_engagement_probs(archetype, topic)
            arr_base = np.array(base_probs.to_array(), dtype=np.float32)

            # Create two variants with noise
            noise_scale = 0.15
            arr_a = np.clip(arr_base + rng.normal(0, noise_scale, NUM_ACTIONS), 0, 1).astype(np.float32)
            arr_b = np.clip(arr_base + rng.normal(0, noise_scale, NUM_ACTIONS), 0, 1).astype(np.float32)

            # Determine preference
            pos_a = arr_a[:14].sum()
            neg_a = arr_a[14:].sum()
            score_a = pos_a - neg_a * 2

            pos_b = arr_b[:14].sum()
            neg_b = arr_b[14:].sum()
            score_b = pos_b - neg_b * 2

            if score_a > score_b:
                all_probs_pref.append(arr_a)
                all_probs_rej.append(arr_b)
            else:
                all_probs_pref.append(arr_b)
                all_probs_rej.append(arr_a)

            all_arch_ids.append(arch_idx)
            all_confidences.append(min(1.0, abs(score_a - score_b) + 0.3))

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


# =============================================================================
# Training
# =============================================================================


def train_contextual_reward_model(
    num_archetypes: int = 6,
    num_epochs: int = 100,
    learning_rate: float = 0.05,
    batch_size: int = 64,
    verbose: bool = True,
) -> tuple[jnp.ndarray, dict]:
    """Train contextual reward weights on ground truth preferences.

    Args:
        num_archetypes: Number of user archetypes
        num_epochs: Training epochs
        learning_rate: Adam learning rate
        batch_size: Mini-batch size
        verbose: Print progress

    Returns:
        Tuple of (trained_weights, metrics_dict)
    """
    import optax

    rng = np.random.default_rng(42)

    # Generate training data
    print("Generating preference pairs from ground truth...")
    cross_topic_data = create_preference_pairs_from_ground_truth(
        num_pairs_per_archetype=1000, rng=rng
    )
    within_topic_data = create_within_topic_preference_pairs(
        num_pairs_per_archetype=500, rng=rng
    )

    # Combine datasets
    probs_pref = np.concatenate([cross_topic_data[0], within_topic_data[0]], axis=0)
    probs_rej = np.concatenate([cross_topic_data[1], within_topic_data[1]], axis=0)
    arch_ids = np.concatenate([cross_topic_data[2], within_topic_data[2]], axis=0)
    confidences = np.concatenate([cross_topic_data[3], within_topic_data[3]], axis=0)

    print(f"Total training pairs: {len(probs_pref)}")

    # Split into train/val
    n_total = len(probs_pref)
    n_train = int(0.8 * n_total)
    indices = rng.permutation(n_total)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_data = (
        jnp.array(probs_pref[train_idx]),
        jnp.array(probs_rej[train_idx]),
        jnp.array(arch_ids[train_idx]),
        jnp.array(confidences[train_idx]),
    )
    val_data = (
        jnp.array(probs_pref[val_idx]),
        jnp.array(probs_rej[val_idx]),
        jnp.array(arch_ids[val_idx]),
        jnp.array(confidences[val_idx]),
    )

    print(f"Train pairs: {len(train_data[0])}, Val pairs: {len(val_data[0])}")

    # Initialize weights from default
    default = RewardWeights.default()
    weights = jnp.tile(default.weights[jnp.newaxis, :], (num_archetypes, 1))

    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    # Training loop
    loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    def loss_fn(w, probs_p, probs_r, arch_ids, conf):
        return contextual_bradley_terry_loss(w, probs_p, probs_r, arch_ids, conf)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    n_train_samples = len(train_data[0])

    for epoch in range(num_epochs):
        # Shuffle training data
        perm = rng.permutation(n_train_samples)

        epoch_losses = []

        # Mini-batch training
        for i in range(0, n_train_samples, batch_size):
            batch_idx = perm[i:i + batch_size]

            batch_probs_p = train_data[0][batch_idx]
            batch_probs_r = train_data[1][batch_idx]
            batch_arch = train_data[2][batch_idx]
            batch_conf = train_data[3][batch_idx]

            loss, grads = grad_fn(weights, batch_probs_p, batch_probs_r, batch_arch, batch_conf)
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

            epoch_losses.append(float(loss))

        # Compute epoch metrics
        avg_train_loss = np.mean(epoch_losses)
        loss_history.append(avg_train_loss)

        # Validation loss
        val_loss = float(loss_fn(weights, *val_data))
        val_loss_history.append(val_loss)

        # Training accuracy
        train_acc = compute_preference_accuracy(
            weights, train_data[0], train_data[1], train_data[2]
        )
        train_acc_history.append(train_acc)

        # Validation accuracy
        val_acc = compute_preference_accuracy(
            weights, val_data[0], val_data[1], val_data[2]
        )
        val_acc_history.append(val_acc)

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:3d}/{num_epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
            )

    metrics = {
        "loss_history": loss_history,
        "val_loss_history": val_loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
        "final_train_loss": loss_history[-1],
        "final_val_loss": val_loss_history[-1],
        "final_train_acc": train_acc_history[-1],
        "final_val_acc": val_acc_history[-1],
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_train_pairs": len(train_data[0]),
        "num_val_pairs": len(val_data[0]),
    }

    return weights, metrics


# =============================================================================
# Analysis
# =============================================================================


def analyze_learned_weights(
    weights: jnp.ndarray,
    archetypes: list[UserArchetype] = None,
) -> dict:
    """Analyze learned weights to verify they match expected patterns."""
    if archetypes is None:
        archetypes = list(UserArchetype)

    analysis = {}

    for arch_idx, archetype in enumerate(archetypes):
        arch_weights = np.array(weights[arch_idx])

        # Get top positive and negative weights
        sorted_idx = np.argsort(arch_weights)
        top_positive = [(ACTION_NAMES[i], float(arch_weights[i])) for i in sorted_idx[-5:][::-1]]
        top_negative = [(ACTION_NAMES[i], float(arch_weights[i])) for i in sorted_idx[:5]]

        # Compute some summary stats
        positive_sum = float(arch_weights[:14].sum())
        negative_sum = float(arch_weights[14:].sum())

        analysis[archetype.value] = {
            "top_positive_weights": top_positive,
            "top_negative_weights": top_negative,
            "positive_actions_sum": positive_sum,
            "negative_actions_sum": negative_sum,
            "weight_range": [float(arch_weights.min()), float(arch_weights.max())],
        }

    # Compute pairwise cosine similarities
    norms = np.linalg.norm(weights, axis=1, keepdims=True)
    normalized = weights / (norms + 1e-8)
    similarities = np.dot(normalized, normalized.T)

    sim_dict = {}
    for i, arch_i in enumerate(archetypes):
        for j, arch_j in enumerate(archetypes):
            if i < j:
                key = f"{arch_i.value}_vs_{arch_j.value}"
                sim_dict[key] = float(similarities[i, j])

    analysis["pairwise_similarities"] = sim_dict
    analysis["avg_pairwise_similarity"] = float(
        (similarities.sum() - len(archetypes)) / (len(archetypes) * (len(archetypes) - 1))
    )

    return analysis


def plot_training_curves(metrics: dict, output_path: Path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(metrics["loss_history"]) + 1)

    # Loss curves
    axes[0].plot(epochs, metrics["loss_history"], label="Train Loss", color="blue")
    axes[0].plot(epochs, metrics["val_loss_history"], label="Val Loss", color="orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Bradley-Terry Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, metrics["train_acc_history"], label="Train Acc", color="blue")
    axes[1].plot(epochs, metrics["val_acc_history"], label="Val Acc", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Preference Accuracy")
    axes[1].set_title("Preference Prediction Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0.4, 1.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_weight_heatmap(weights: jnp.ndarray, output_path: Path):
    """Plot heatmap of learned weights."""
    archetypes = list(UserArchetype)

    fig, ax = plt.subplots(figsize=(14, 6))

    im = ax.imshow(weights, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)

    ax.set_xticks(range(NUM_ACTIONS))
    ax.set_xticklabels(ACTION_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(archetypes)))
    ax.set_yticklabels([a.value for a in archetypes])

    ax.set_xlabel("Action")
    ax.set_ylabel("Archetype")
    ax.set_title("Learned Reward Weights by Archetype")

    plt.colorbar(im, ax=ax, label="Weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved weight heatmap to {output_path}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Train and save baseline contextual reward model."""
    output_dir = Path("results/f4_phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("F4 Phase 1: Training Contextual Reward Model")
    print("=" * 60)

    # Train model
    weights, metrics = train_contextual_reward_model(
        num_archetypes=6,
        num_epochs=100,
        learning_rate=0.05,
        batch_size=64,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {metrics['final_train_loss']:.4f}")
    print(f"Final val loss:   {metrics['final_val_loss']:.4f}")
    print(f"Final train acc:  {metrics['final_train_acc']:.4f}")
    print(f"Final val acc:    {metrics['final_val_acc']:.4f}")

    # Analyze weights
    print("\nAnalyzing learned weights...")
    analysis = analyze_learned_weights(weights)

    # Print per-archetype analysis
    print("\n" + "-" * 60)
    print("Per-Archetype Weight Analysis")
    print("-" * 60)
    for archetype in UserArchetype:
        arch_analysis = analysis[archetype.value]
        print(f"\n{archetype.value}:")
        print(f"  Top positive: {arch_analysis['top_positive_weights'][:3]}")
        print(f"  Top negative: {arch_analysis['top_negative_weights'][:3]}")

    print(f"\nAverage pairwise similarity: {analysis['avg_pairwise_similarity']:.4f}")

    # Save outputs
    print("\n" + "=" * 60)
    print("Saving Results")
    print("=" * 60)

    # Save weights
    weights_path = output_dir / "baseline_weights.npy"
    np.save(weights_path, np.array(weights))
    print(f"Saved weights to {weights_path}")

    # Save metrics
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

    # Save analysis
    analysis_path = output_dir / "weight_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    # Plot training curves
    plot_training_curves(metrics, output_dir / "training_curves.png")

    # Plot weight heatmap
    plot_weight_heatmap(weights, output_dir / "weight_heatmap.png")

    print("\n" + "=" * 60)
    print("Phase 1 Baseline Complete!")
    print("=" * 60)

    # Summary for Go/No-Go gates
    print("\nGo/No-Go Gate Summary:")
    print(f"  [{'PASS' if metrics['final_val_acc'] > 0.6 else 'FAIL'}] Val accuracy > 60%: {metrics['final_val_acc']:.2%}")
    print(f"  [{'PASS' if metrics['final_val_loss'] < metrics['loss_history'][0] else 'FAIL'}] Loss decreased during training")

    # Note: High similarity is EXPECTED in Phase 1 because all archetypes
    # share the same action preferences (positive > negative).
    # Archetype differentiation happens at the TOPIC level, which Phase 2 addresses.
    print(f"  [INFO] Weight similarity across archetypes: {analysis['avg_pairwise_similarity']:.4f}")
    print("         (High similarity expected - archetypes differ by TOPIC preference, not action weighting)")

    # Verify the weights learned correct action semantics
    weights_np = np.array(weights)
    positive_action_avg = weights_np[:, :14].mean()
    negative_action_avg = weights_np[:, 14:].mean()
    action_semantics_correct = positive_action_avg > negative_action_avg

    print(f"  [{'PASS' if action_semantics_correct else 'FAIL'}] Action semantics correct (positive > negative): {positive_action_avg:.2f} > {negative_action_avg:.2f}")


if __name__ == "__main__":
    main()
