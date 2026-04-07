"""Verify held-out evaluation on synthetic data (Phase 1.5).

Checks that existing synthetic results (cosine similarities, accuracy)
are robust to train/test splitting. Trains BT models on 80% of
preference pairs and evaluates on 20% held-out.

Usage:
    uv run python scripts/experiments/verify_held_out.py

Output:
    results/held_out_verification.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


alt_losses = _load(
    "alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py"
)

LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss
NUM_ACTIONS = alt_losses.NUM_ACTIONS
ACTION_INDICES = alt_losses.ACTION_INDICES
POSITIVE_INDICES = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alt_losses.NEGATIVE_INDICES

# Stakeholder utility functions (same as run_loss_experiments.py)
STAKEHOLDER_UTILITY = {
    "user": lambda pos, neg: pos - neg,
    "platform": lambda pos, neg: pos - 0.3 * neg,
    "society": lambda pos, neg: pos - 4.0 * neg,
}


def generate_content_pool(n_content: int, seed: int) -> tuple:
    """Generate content pool (inlined from run_loss_experiments.py)."""
    rng = np.random.default_rng(seed)
    content_probs = np.zeros((n_content, NUM_ACTIONS), dtype=np.float32)
    content_topics = rng.integers(0, 6, size=n_content)
    for i in range(n_content):
        topic = content_topics[i]
        base_probs = rng.uniform(0.05, 0.3, NUM_ACTIONS)
        if topic == 0:  # sports
            base_probs[ACTION_INDICES["favorite"]] *= 1.5
            base_probs[ACTION_INDICES["repost"]] *= 1.3
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.3
        elif topic == 1:  # tech
            base_probs[ACTION_INDICES["favorite"]] *= 1.2
            base_probs[ACTION_INDICES["reply"]] *= 1.4
        elif topic in [2, 3]:  # politics
            base_probs[ACTION_INDICES["favorite"]] *= 1.3
            base_probs[ACTION_INDICES["repost"]] *= 1.8
            base_probs[ACTION_INDICES["reply"]] *= 2.0
            base_probs[ACTION_INDICES["block_author"]] *= 2.5
            base_probs[ACTION_INDICES["report"]] *= 2.0
        elif topic == 4:  # entertainment
            base_probs[ACTION_INDICES["favorite"]] *= 1.8
            base_probs[ACTION_INDICES["share"]] *= 1.5
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.2
        elif topic == 5:  # news
            base_probs[ACTION_INDICES["follow_author"]] *= 1.3
            base_probs[ACTION_INDICES["reply"]] *= 1.2
        content_probs[i] = np.clip(base_probs, 0, 1)
    return content_probs, content_topics


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def generate_preferences_synthetic(
    content_probs: np.ndarray,
    stakeholder: str,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic preference pairs (same logic as run_loss_experiments)."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)
    utility_fn = STAKEHOLDER_UTILITY[stakeholder]

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    utility = utility_fn(pos_scores, neg_scores)

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]
        noise = rng.normal(0, 0.05)
        if (diff + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    return probs_pref, probs_rej


def split_data(
    pref: np.ndarray, rej: np.ndarray, eval_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split preference pairs into train and eval."""
    rng = np.random.default_rng(seed)
    n = len(pref)
    n_eval = int(n * eval_fraction)
    indices = rng.permutation(n)
    eval_idx = indices[:n_eval]
    train_idx = indices[n_eval:]
    return pref[train_idx], rej[train_idx], pref[eval_idx], rej[eval_idx]


def main() -> None:
    print("=" * 70)
    print("Phase 1.5: Held-Out Evaluation Verification (Synthetic)")
    print("=" * 70)

    results: dict = {"config": {"n_pairs": 2500, "eval_fraction": 0.2, "seed": 42}}

    # Generate content pool (same as loss experiments)
    content_probs, content_topics = generate_content_pool(500, seed=42)
    print(f"\nContent pool: {content_probs.shape}")

    stakeholders = {"user": StakeholderType.USER, "platform": StakeholderType.PLATFORM, "society": StakeholderType.SOCIETY}
    trained_weights = {}
    results["per_stakeholder"] = {}

    for name, st_type in stakeholders.items():
        print(f"\n--- {name} ---")

        # Generate 2500 pairs, split 80/20
        pref, rej = generate_preferences_synthetic(content_probs, name, 2500, seed=42)
        train_pref, train_rej, eval_pref, eval_rej = split_data(pref, rej, 0.2, seed=42)
        print(f"  Train: {len(train_pref)}, Eval: {len(eval_pref)}")

        config = LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=st_type,
            learning_rate=0.01,
            num_epochs=50,
            batch_size=64,
        )

        # Train with held-out eval
        model = train_with_loss(
            config, train_pref, train_rej,
            verbose=False,
            eval_probs_preferred=eval_pref,
            eval_probs_rejected=eval_rej,
        )

        trained_weights[name] = model.weights
        gap = abs(model.accuracy - (model.eval_accuracy or 0))
        status = "✓" if gap < 0.03 else "⚠"

        print(f"  {status} Train acc: {model.accuracy:.1%}, Held-out acc: {model.eval_accuracy:.1%}, Gap: {gap:.1%}")

        results["per_stakeholder"][name] = {
            "train_accuracy": round(model.accuracy, 4),
            "eval_accuracy": round(model.eval_accuracy, 4) if model.eval_accuracy else None,
            "gap": round(gap, 4),
            "loss_start": round(model.loss_history[0], 4),
            "loss_end": round(model.loss_history[-1], 4),
        }

    # Cosine similarity matrix
    print("\n--- Cosine Similarity Matrix ---")
    pairs = [("user", "platform"), ("user", "society"), ("platform", "society")]
    results["cosine_similarity"] = {}

    # Load existing comparison values
    comparison_path = ROOT / "results" / "loss_experiments" / "comparison.json"
    existing = {}
    if comparison_path.exists():
        with open(comparison_path) as f:
            existing = json.load(f).get("bradley", {}).get("cosine_similarities", {})

    for name_a, name_b in pairs:
        sim = cosine_sim(trained_weights[name_a], trained_weights[name_b])
        key = f"{name_a}_{name_b}"
        existing_val = existing.get(key, None)
        delta = abs(sim - existing_val) if existing_val else None
        delta_str = f", Δ={delta:.3f}" if delta is not None else ""
        status = "✓" if delta is None or delta < 0.05 else "⚠"

        existing_str = f"{existing_val:.3f}" if existing_val is not None else "N/A"
        print(f"  {status} {name_a}-{name_b}: {sim:.3f} (existing: {existing_str}{delta_str})")

        results["cosine_similarity"][key] = {
            "held_out_split": round(sim, 4),
            "existing": round(existing_val, 4) if existing_val else None,
            "delta": round(delta, 4) if delta is not None else None,
        }

    # Summary
    print("\n" + "=" * 70)
    print("Phase 1.5 Success Criteria:")

    all_gaps = [r["gap"] for r in results["per_stakeholder"].values()]
    all_deltas = [r["delta"] for r in results["cosine_similarity"].values() if r["delta"] is not None]

    checks = [
        ("Held-out accuracy within 3% of training", max(all_gaps) < 0.03),
        ("Cosine similarities within 0.05 of existing", max(all_deltas) < 0.05 if all_deltas else True),
    ]
    all_pass = True
    for desc, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {desc}")
        if not passed:
            all_pass = False

    print("=" * 70)

    # Save
    out_path = ROOT / "results" / "held_out_verification.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
