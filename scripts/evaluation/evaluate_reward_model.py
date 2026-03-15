#!/usr/bin/env python3
"""Comprehensive evaluation of Bradley-Terry reward model.

Tests model robustness across multiple challenging scenarios:
1. Held-out archetypes (generalization)
2. Noisy labels (robustness)
3. Hard negatives - same topic (fine discrimination)
4. Adversarial pairs (edge cases)

This reveals limitations of simple Bradley-Terry and motivates
more sophisticated models in later phases.

Usage:
    uv run python scripts/evaluate_reward_model.py
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

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
    NUM_ACTIONS,
    RewardWeights,
    compute_preference_accuracy,
    contextual_bradley_terry_loss,
)

# =============================================================================
# Evaluation Data Generators
# =============================================================================


def create_standard_test_set(
    num_pairs_per_archetype: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standard test set - same as training distribution.

    Note: Some archetypes (lurker, power_user) have topic-independent behavior,
    so we add small noise to create variation for those.
    """
    if rng is None:
        rng = np.random.default_rng(999)  # Different seed from training

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    for arch_idx, archetype in enumerate(archetypes):
        pairs_created = 0
        max_attempts = num_pairs_per_archetype * 10

        for attempt in range(max_attempts):
            if pairs_created >= num_pairs_per_archetype:
                break

            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a, topic_b = topics[topic_indices[0]], topics[topic_indices[1]]

            probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
            probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

            # Add small noise for topic-independent archetypes (lurker, power_user)
            noise_std = 0.05
            probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
            probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)

            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            # Use lower threshold since we added noise
            if abs(score_a - score_b) > 0.05:
                if score_a > score_b:
                    all_probs_pref.append(probs_a)
                    all_probs_rej.append(probs_b)
                else:
                    all_probs_pref.append(probs_b)
                    all_probs_rej.append(probs_a)
                all_arch_ids.append(arch_idx)
                all_confidences.append(1.0)
                pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


def create_held_out_archetype_test(
    held_out_archetypes: list[UserArchetype],
    num_pairs_per_archetype: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Test set with only held-out archetypes (never seen during training)."""
    if rng is None:
        rng = np.random.default_rng(888)

    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    # Map held-out archetypes to indices (will be wrong during eval - that's the point)
    # We'll use index 0 for all held-out to simulate "unknown archetype"
    for archetype in held_out_archetypes:
        pairs_created = 0
        max_attempts = num_pairs_per_archetype * 10

        for attempt in range(max_attempts):
            if pairs_created >= num_pairs_per_archetype:
                break

            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a, topic_b = topics[topic_indices[0]], topics[topic_indices[1]]

            probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
            probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

            # Add small noise for topic-independent archetypes
            noise_std = 0.05
            probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
            probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)

            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            if abs(score_a - score_b) > 0.05:
                if score_a > score_b:
                    all_probs_pref.append(probs_a)
                    all_probs_rej.append(probs_b)
                else:
                    all_probs_pref.append(probs_b)
                    all_probs_rej.append(probs_a)
                # Use archetype index 0 (pretend we don't know the archetype)
                all_arch_ids.append(0)
                all_confidences.append(1.0)
                pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
        [a.value for a in held_out_archetypes],
    )


def create_noisy_test_set(
    noise_std: float = 0.15,
    label_flip_rate: float = 0.1,
    num_pairs_per_archetype: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Test set with noisy action probabilities and flipped labels."""
    if rng is None:
        rng = np.random.default_rng(777)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    for arch_idx, archetype in enumerate(archetypes):
        pairs_created = 0
        while pairs_created < num_pairs_per_archetype:
            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a, topic_b = topics[topic_indices[0]], topics[topic_indices[1]]

            # Get base probs and add noise
            probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
            probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

            # Add Gaussian noise
            probs_a = np.clip(probs_a + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)
            probs_b = np.clip(probs_b + rng.normal(0, noise_std, NUM_ACTIONS), 0, 1).astype(np.float32)

            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            if score_a > score_b:
                pref, rej = probs_a, probs_b
            else:
                pref, rej = probs_b, probs_a

            # Randomly flip labels
            if rng.random() < label_flip_rate:
                pref, rej = rej, pref

            all_probs_pref.append(pref)
            all_probs_rej.append(rej)
            all_arch_ids.append(arch_idx)
            all_confidences.append(1.0)
            pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


def create_hard_negatives_test(
    num_pairs_per_archetype: int = 50,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Test set with hard negatives - same topic, subtle differences."""
    if rng is None:
        rng = np.random.default_rng(666)

    archetypes = list(UserArchetype)
    topics = list(ContentTopic)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    for arch_idx, archetype in enumerate(archetypes):
        pairs_created = 0
        attempts = 0
        max_attempts = num_pairs_per_archetype * 20

        while pairs_created < num_pairs_per_archetype and attempts < max_attempts:
            attempts += 1

            # Same topic for both!
            topic = topics[rng.integers(0, len(topics))]

            # Get base probs
            base_probs = np.array(get_engagement_probs(archetype, topic).to_array(), dtype=np.float32)

            # Create two variants with small perturbations
            # This simulates "same topic, different quality/author"
            noise_scale = 0.08  # Small noise
            probs_a = np.clip(base_probs + rng.normal(0, noise_scale, NUM_ACTIONS), 0, 1).astype(np.float32)
            probs_b = np.clip(base_probs + rng.normal(0, noise_scale, NUM_ACTIONS), 0, 1).astype(np.float32)

            # Compute scores
            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            # Only keep if there's SOME difference (but it will be small)
            if abs(score_a - score_b) > 0.02:  # Very small threshold
                if score_a > score_b:
                    all_probs_pref.append(probs_a)
                    all_probs_rej.append(probs_b)
                else:
                    all_probs_pref.append(probs_b)
                    all_probs_rej.append(probs_a)
                all_arch_ids.append(arch_idx)
                all_confidences.append(min(1.0, abs(score_a - score_b) * 5))  # Lower confidence for close pairs
                pairs_created += 1

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


def create_adversarial_test(
    num_pairs: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Adversarial pairs where naive scoring fails.

    Creates pairs where rejected has higher positive actions BUT also has
    negative signals that should make it worse overall.
    """
    if rng is None:
        rng = np.random.default_rng(555)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    num_archetypes = len(list(UserArchetype))

    for i in range(num_pairs):
        # Preferred: moderate positive, NO negative
        pref = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pref[0] = rng.uniform(0.4, 0.6)  # favorite
        pref[1] = rng.uniform(0.2, 0.4)  # reply
        pref[2] = rng.uniform(0.3, 0.5)  # repost
        pref[4] = rng.uniform(0.3, 0.5)  # click
        pref[10] = rng.uniform(0.4, 0.6)  # dwell
        # No negative actions

        # Rejected: HIGHER positive, but HAS negative signals
        rej = np.zeros(NUM_ACTIONS, dtype=np.float32)
        rej[0] = rng.uniform(0.6, 0.8)  # favorite - HIGHER
        rej[1] = rng.uniform(0.3, 0.5)  # reply - HIGHER
        rej[2] = rng.uniform(0.4, 0.6)  # repost - HIGHER
        rej[4] = rng.uniform(0.4, 0.6)  # click - HIGHER
        rej[10] = rng.uniform(0.5, 0.7)  # dwell - HIGHER
        # BUT also has negative signals
        rej[14] = rng.uniform(0.15, 0.3)  # not_interested
        rej[15] = rng.uniform(0.1, 0.25)  # block_author
        rej[16] = rng.uniform(0.05, 0.15)  # mute_author

        # Naive sum of positive: rej wins
        # Correct (with negative penalty): pref should win

        all_probs_pref.append(pref)
        all_probs_rej.append(rej)
        all_arch_ids.append(i % num_archetypes)
        all_confidences.append(1.0)

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


def create_conflicting_signals_test(
    num_pairs: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pairs with mixed signals - some positive AND some negative.

    Tests: Can model handle nuanced cases where content is polarizing?
    """
    if rng is None:
        rng = np.random.default_rng(444)

    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []
    all_confidences = []

    num_archetypes = len(list(UserArchetype))

    for i in range(num_pairs):
        # Both have mixed signals, but preferred has better ratio

        # Preferred: good positive/negative ratio
        pref = np.zeros(NUM_ACTIONS, dtype=np.float32)
        pref[0] = rng.uniform(0.5, 0.7)  # favorite
        pref[2] = rng.uniform(0.3, 0.5)  # repost
        pref[10] = rng.uniform(0.4, 0.6)  # dwell
        pref[14] = rng.uniform(0.05, 0.1)  # small not_interested

        # Rejected: worse ratio (more negative relative to positive)
        rej = np.zeros(NUM_ACTIONS, dtype=np.float32)
        rej[0] = rng.uniform(0.4, 0.6)  # favorite (similar)
        rej[2] = rng.uniform(0.2, 0.4)  # repost (similar)
        rej[10] = rng.uniform(0.3, 0.5)  # dwell (similar)
        rej[14] = rng.uniform(0.15, 0.25)  # higher not_interested
        rej[15] = rng.uniform(0.1, 0.2)  # some block

        all_probs_pref.append(pref)
        all_probs_rej.append(rej)
        all_arch_ids.append(i % num_archetypes)
        all_confidences.append(0.7)  # Lower confidence - nuanced cases

    return (
        np.array(all_probs_pref, dtype=np.float32),
        np.array(all_probs_rej, dtype=np.float32),
        np.array(all_arch_ids, dtype=np.int32),
        np.array(all_confidences, dtype=np.float32),
    )


# =============================================================================
# Training with Held-Out Archetypes
# =============================================================================


def train_with_held_out_archetypes(
    train_archetypes: list[UserArchetype],
    num_epochs: int = 100,
    learning_rate: float = 0.05,
    rng: np.random.Generator | None = None,
) -> jnp.ndarray:
    """Train only on specified archetypes."""
    import optax

    if rng is None:
        rng = np.random.default_rng(42)

    topics = list(ContentTopic)
    num_archetypes = len(train_archetypes)

    # Generate training data for only these archetypes
    all_probs_pref = []
    all_probs_rej = []
    all_arch_ids = []

    for arch_idx, archetype in enumerate(train_archetypes):
        for _ in range(200):
            topic_indices = rng.choice(len(topics), size=2, replace=False)
            topic_a, topic_b = topics[topic_indices[0]], topics[topic_indices[1]]

            probs_a = np.array(get_engagement_probs(archetype, topic_a).to_array(), dtype=np.float32)
            probs_b = np.array(get_engagement_probs(archetype, topic_b).to_array(), dtype=np.float32)

            score_a = probs_a[:14].sum() - probs_a[14:].sum() * 2
            score_b = probs_b[:14].sum() - probs_b[14:].sum() * 2

            if abs(score_a - score_b) > 0.1:
                if score_a > score_b:
                    all_probs_pref.append(probs_a)
                    all_probs_rej.append(probs_b)
                else:
                    all_probs_pref.append(probs_b)
                    all_probs_rej.append(probs_a)
                all_arch_ids.append(arch_idx)

    probs_pref = jnp.array(all_probs_pref)
    probs_rej = jnp.array(all_probs_rej)
    arch_ids = jnp.array(all_arch_ids)

    # Initialize and train
    default = RewardWeights.default()
    weights = jnp.tile(default.weights[jnp.newaxis, :], (num_archetypes, 1))

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(weights)

    def loss_fn(w, p_pref, p_rej, a_ids):
        return contextual_bradley_terry_loss(w, p_pref, p_rej, a_ids, None)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    n_samples = len(probs_pref)
    batch_size = 64

    for epoch in range(num_epochs):
        perm = rng.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            loss, grads = grad_fn(weights, probs_pref[idx], probs_rej[idx], arch_ids[idx])
            updates, opt_state = optimizer.update(grads, opt_state, weights)
            weights = optax.apply_updates(weights, updates)

    return weights


# =============================================================================
# Evaluation
# =============================================================================


@dataclass
class EvalResult:
    """Results from an evaluation."""
    name: str
    accuracy: float
    loss: float
    num_samples: int
    description: str


def evaluate_on_test_set(
    weights: jnp.ndarray,
    test_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    name: str,
    description: str,
) -> EvalResult:
    """Evaluate weights on a test set."""
    probs_pref, probs_rej, arch_ids, conf = test_data

    # Convert to JAX
    probs_pref = jnp.array(probs_pref)
    probs_rej = jnp.array(probs_rej)
    arch_ids = jnp.array(arch_ids)
    conf = jnp.array(conf)

    # Compute accuracy
    acc = compute_preference_accuracy(weights, probs_pref, probs_rej, arch_ids)

    # Compute loss
    loss = float(contextual_bradley_terry_loss(weights, probs_pref, probs_rej, arch_ids, conf))

    return EvalResult(
        name=name,
        accuracy=acc,
        loss=loss,
        num_samples=len(probs_pref),
        description=description,
    )


def run_comprehensive_evaluation(
    weights: jnp.ndarray,
    verbose: bool = True,
) -> dict[str, EvalResult]:
    """Run all evaluation tests."""
    results = {}

    # 1. Standard test (same distribution as training)
    if verbose:
        flush_print("\n1. Standard Test (same distribution)...")
    std_data = create_standard_test_set()
    results["standard"] = evaluate_on_test_set(
        weights, std_data,
        "Standard",
        "Same distribution as training - baseline sanity check"
    )

    # 2. Noisy labels (10% flip, 0.15 std noise)
    if verbose:
        flush_print("2. Noisy Labels Test...")
    noisy_data = create_noisy_test_set(noise_std=0.15, label_flip_rate=0.10)
    results["noisy_10pct"] = evaluate_on_test_set(
        weights, noisy_data,
        "Noisy (10% flip)",
        "Action probs with noise + 10% label flips"
    )

    # 3. Higher noise (20% flip)
    if verbose:
        flush_print("3. High Noise Test...")
    high_noise_data = create_noisy_test_set(noise_std=0.20, label_flip_rate=0.20)
    results["noisy_20pct"] = evaluate_on_test_set(
        weights, high_noise_data,
        "Noisy (20% flip)",
        "Higher noise + 20% label flips"
    )

    # 4. Hard negatives (same topic)
    if verbose:
        flush_print("4. Hard Negatives Test (same topic)...")
    hard_data = create_hard_negatives_test()
    results["hard_negatives"] = evaluate_on_test_set(
        weights, hard_data,
        "Hard Negatives",
        "Same topic, subtle quality differences"
    )

    # 5. Adversarial pairs
    if verbose:
        flush_print("5. Adversarial Test...")
    adv_data = create_adversarial_test()
    results["adversarial"] = evaluate_on_test_set(
        weights, adv_data,
        "Adversarial",
        "Rejected has higher positive but also negative signals"
    )

    # 6. Conflicting signals
    if verbose:
        flush_print("6. Conflicting Signals Test...")
    conflict_data = create_conflicting_signals_test()
    results["conflicting"] = evaluate_on_test_set(
        weights, conflict_data,
        "Conflicting Signals",
        "Both have mixed positive/negative signals"
    )

    return results


def run_held_out_evaluation(verbose: bool = True) -> dict[str, float]:
    """Train with held-out archetypes and evaluate generalization."""

    results = {}

    # Test 1: Hold out POLITICAL_R and POWER_USER
    if verbose:
        flush_print("\n7. Held-Out Archetypes Test...")
        print("   Training on: SPORTS_FAN, POLITICAL_L, TECH_BRO, LURKER")
        print("   Testing on: POLITICAL_R, POWER_USER")

    train_archs = [
        UserArchetype.SPORTS_FAN,
        UserArchetype.POLITICAL_L,
        UserArchetype.TECH_BRO,
        UserArchetype.LURKER,
    ]
    test_archs = [
        UserArchetype.POLITICAL_R,
        UserArchetype.POWER_USER,
    ]

    # Train on subset (reduced epochs for faster evaluation)
    weights_subset = train_with_held_out_archetypes(train_archs, num_epochs=20)

    # Evaluate on held-out (using archetype 0 weights for all)
    held_out_data = create_held_out_archetype_test(test_archs)
    probs_pref, probs_rej, arch_ids, conf, arch_names = held_out_data

    # Use first archetype's weights (simulating "unknown user type")
    single_weights = weights_subset[0:1]  # [1, 18]
    # Broadcast to match batch
    acc = compute_preference_accuracy(
        single_weights,
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        jnp.zeros(len(probs_pref), dtype=jnp.int32),  # All use weight 0
    )
    results["held_out_political_r_power_user"] = acc

    # Test 2: Hold out opposite political (train L, test R)
    if verbose:
        flush_print("\n8. Political Generalization Test...")
        print("   Training on: POLITICAL_L only")
        print("   Testing on: POLITICAL_R")

    weights_pol_l = train_with_held_out_archetypes(
        [UserArchetype.POLITICAL_L], num_epochs=20
    )
    held_out_pol_r = create_held_out_archetype_test([UserArchetype.POLITICAL_R])
    probs_pref, probs_rej, _, _, _ = held_out_pol_r

    acc_pol = compute_preference_accuracy(
        weights_pol_l,
        jnp.array(probs_pref),
        jnp.array(probs_rej),
        jnp.zeros(len(probs_pref), dtype=jnp.int32),
    )
    results["political_l_to_r"] = acc_pol

    return results


# =============================================================================
# Main
# =============================================================================


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def save_checkpoint(path: Path, results: dict[str, "EvalResult"], held_out: dict[str, float]):
    """Save evaluation checkpoint."""
    checkpoint = {
        "results": {k: {"name": v.name, "accuracy": v.accuracy, "loss": v.loss,
                        "num_samples": v.num_samples, "description": v.description}
                    for k, v in results.items()},
        "held_out_results": held_out,
    }
    with open(path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    flush_print(f"Checkpoint saved to {path}")


def main():
    """Run comprehensive evaluation."""

    output_dir = Path("results/f4_phase1")
    output_dir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 70)
    flush_print("F4 Phase 1: Comprehensive Bradley-Terry Evaluation")
    flush_print("=" * 70)

    # Check for checkpoint
    checkpoint_path = output_dir / "eval_checkpoint.json"
    results = {}
    held_out_results = {}

    if checkpoint_path.exists():
        flush_print("Loading checkpoint...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
            results = {k: EvalResult(**v) for k, v in checkpoint.get("results", {}).items()}
            held_out_results = checkpoint.get("held_out_results", {})
        flush_print(f"Resumed from checkpoint with {len(results)} standard tests done")

    # Load trained weights
    weights_path = output_dir / "baseline_weights.npy"
    if not weights_path.exists():
        flush_print(f"ERROR: Trained weights not found at {weights_path}")
        flush_print("Please run train_reward_model.py first.")
        return

    weights = jnp.array(np.load(weights_path))
    flush_print(f"Loaded weights from {weights_path}")
    flush_print(f"Weights shape: {weights.shape}")

    # Run standard evaluations
    flush_print("\n" + "=" * 70)
    flush_print("Running Evaluation Suite")
    flush_print("=" * 70)

    if len(results) < 6:  # 6 standard tests
        results = run_comprehensive_evaluation(weights, verbose=True)
        # Save checkpoint after standard tests
        save_checkpoint(checkpoint_path, results, held_out_results)
    else:
        flush_print("Standard tests already completed (from checkpoint)")

    # Run held-out evaluations
    if len(held_out_results) < 2:  # 2 held-out tests
        held_out_results = run_held_out_evaluation(verbose=True)
        save_checkpoint(checkpoint_path, results, held_out_results)
    else:
        flush_print("Held-out tests already completed (from checkpoint)")

    # Print results table
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"{'Test':<30} {'Accuracy':>10} {'Loss':>10} {'Samples':>10}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{result.name:<30} {result.accuracy:>10.2%} {result.loss:>10.4f} {result.num_samples:>10}")

    print("-" * 70)
    print("Held-Out Archetype Tests:")
    for name, acc in held_out_results.items():
        print(f"  {name:<40} {acc:>10.2%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    std_acc = results["standard"].accuracy
    hard_acc = results["hard_negatives"].accuracy
    adv_acc = results["adversarial"].accuracy
    noisy_acc = results["noisy_10pct"].accuracy

    print(f"\nAccuracy Degradation from Standard ({std_acc:.1%}):")
    print(f"  - Hard Negatives:    {std_acc - hard_acc:>+6.1%} → {hard_acc:.1%}")
    print(f"  - Adversarial:       {std_acc - adv_acc:>+6.1%} → {adv_acc:.1%}")
    print(f"  - Noisy (10%):       {std_acc - noisy_acc:>+6.1%} → {noisy_acc:.1%}")
    print(f"  - Held-out archs:    {std_acc - held_out_results['held_out_political_r_power_user']:>+6.1%} → {held_out_results['held_out_political_r_power_user']:.1%}")

    # Determine if model is robust
    print("\n" + "-" * 70)
    print("Go/No-Go Assessment:")

    gates = [
        ("Standard accuracy > 95%", std_acc > 0.95, std_acc),
        ("Hard negatives > 60%", hard_acc > 0.60, hard_acc),
        ("Adversarial > 75%", adv_acc > 0.75, adv_acc),
        ("Noisy (10%) > 85%", noisy_acc > 0.85, noisy_acc),
        ("Held-out generalization > 70%", held_out_results['held_out_political_r_power_user'] > 0.70, held_out_results['held_out_political_r_power_user']),
    ]

    all_pass = True
    for name, passed, value in gates:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}: {value:.1%}")
        if not passed:
            all_pass = False

    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)

    eval_results = {
        "standard_tests": {k: {"accuracy": v.accuracy, "loss": v.loss, "samples": v.num_samples}
                          for k, v in results.items()},
        "held_out_tests": held_out_results,
        "summary": {
            "standard_accuracy": std_acc,
            "hard_negatives_accuracy": hard_acc,
            "adversarial_accuracy": adv_acc,
            "noisy_accuracy": noisy_acc,
            "held_out_accuracy": held_out_results['held_out_political_r_power_user'],
            "all_gates_passed": all_pass,
        }
    }

    eval_path = output_dir / "comprehensive_evaluation.json"
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Saved evaluation results to {eval_path}")

    # Plot comparison
    plot_evaluation_comparison(results, held_out_results, output_dir / "evaluation_comparison.png")

    print("\n" + "=" * 70)
    if all_pass:
        print("All gates PASSED - Bradley-Terry baseline is robust!")
    else:
        print("Some gates FAILED - Model has limitations (expected)")
        print("This motivates more sophisticated models in Phase 2+")
    print("=" * 70)


def plot_evaluation_comparison(
    results: dict[str, EvalResult],
    held_out_results: dict[str, float],
    output_path: Path,
):
    """Plot comparison of evaluation results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Combine all results
    names = []
    accuracies = []
    colors = []

    for name, result in results.items():
        names.append(result.name)
        accuracies.append(result.accuracy)
        colors.append("steelblue")

    for name, acc in held_out_results.items():
        short_name = name.replace("held_out_", "Held: ").replace("_", " ")
        names.append(short_name)
        accuracies.append(acc)
        colors.append("coral")

    # Create bar chart
    x = np.arange(len(names))
    bars = ax.bar(x, accuracies, color=colors)

    # Add threshold lines
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.7, label='95% (excellent)')
    ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.7, label='80% (good)')
    ax.axhline(y=0.65, color='red', linestyle='--', alpha=0.7, label='65% (minimum)')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Bradley-Terry Model: Comprehensive Evaluation')
    ax.set_ylim([0.4, 1.05])
    ax.legend(loc='lower right')

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved evaluation plot to {output_path}")


if __name__ == "__main__":
    import jax
    main()
