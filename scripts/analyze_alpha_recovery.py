#!/usr/bin/env python3
"""α-Recovery analysis: can BT weight vectors recover the utility parameter α?

For the utility family U = pos − α·neg, trains BT models across a range of α
values and tests whether the neg/pos weight ratio recovers α.

Closes out Direction 1 (Identifiability limits) in the F4 retro.

Usage:
    uv run python scripts/analyze_alpha_recovery.py
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Module loading (same pattern as other analysis scripts)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_module(
    module_name: str, module_path: Path, register_as: str | None = None
):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if register_as:
        sys.modules[register_as] = module
    spec.loader.exec_module(module)
    return module


alt_losses = load_module(
    "alternative_losses",
    project_root / "enhancements" / "reward_modeling" / "alternative_losses.py",
    register_as="enhancements.reward_modeling.alternative_losses",
)

POSITIVE_INDICES: list[int] = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES: list[int] = alt_losses.NEGATIVE_INDICES
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss

loss_exp_mod = load_module(
    "run_loss_experiments",
    project_root / "scripts" / "run_loss_experiments.py",
)
generate_content_pool = loss_exp_mod.generate_content_pool

# Also load the preference generation from disagreement bound script
disag_mod = load_module(
    "analyze_disagreement_bound",
    project_root / "scripts" / "analyze_disagreement_bound.py",
)
generate_preference_data = disag_mod.generate_preference_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_CONTENT = 500
N_TRAINING_PAIRS = 5000
NUM_EPOCHS = 150
LEARNING_RATE = 0.01

# 13 α values spanning the range from the disagreement bound sweep
ALPHA_VALUES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]

# The three main stakeholders and their known α values
STAKEHOLDER_ALPHAS = {
    "platform": 0.3,
    "user": 1.0,
    "society": 4.0,
}


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# α recovery methods
# ---------------------------------------------------------------------------


def compute_alpha_ratio(weights: np.ndarray) -> float:
    """Recover α as -mean(w_neg) / mean(w_pos)."""
    pos_mean = float(np.mean(weights[POSITIVE_INDICES]))
    neg_mean = float(np.mean(weights[NEGATIVE_INDICES]))
    if abs(pos_mean) < 1e-10:
        return float("inf")
    return -neg_mean / pos_mean


def compute_alpha_sum_ratio(weights: np.ndarray) -> float:
    """Recover α as -sum(w_neg) / sum(w_pos).

    Note: this has a systematic bias of n_neg/n_pos relative to the true α.
    """
    pos_sum = float(np.sum(weights[POSITIVE_INDICES]))
    neg_sum = float(np.sum(weights[NEGATIVE_INDICES]))
    if abs(pos_sum) < 1e-10:
        return float("inf")
    return -neg_sum / pos_sum


def compute_alpha_regression(weights: np.ndarray) -> float:
    """Recover α via regression: w_i = β_pos if positive, β_neg if negative.

    α_eff = -β_neg / β_pos.
    """
    n = len(POSITIVE_INDICES) + len(NEGATIVE_INDICES)
    x_mat = np.zeros((n, 2))
    y = np.zeros(n)

    for i, idx in enumerate(POSITIVE_INDICES):
        x_mat[i, 0] = 1.0
        y[i] = weights[idx]
    for i, idx in enumerate(NEGATIVE_INDICES):
        x_mat[len(POSITIVE_INDICES) + i, 1] = 1.0
        y[len(POSITIVE_INDICES) + i] = weights[idx]

    beta, _, _, _ = np.linalg.lstsq(x_mat, y, rcond=None)
    if abs(beta[0]) < 1e-10:
        return float("inf")
    return float(-beta[1] / beta[0])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_bt_for_alpha(
    content_probs: np.ndarray, alpha: float, seed: int = SEED
) -> np.ndarray:
    """Train a BT model with utility U = pos − α·neg, return weight vector."""
    probs_pref, probs_rej = generate_preference_data(
        content_probs, neg_penalty=alpha, n_pairs=N_TRAINING_PAIRS, seed=seed
    )
    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    model = train_with_loss(config, probs_pref, probs_rej, verbose=False)
    return np.array(model.weights)


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    start_time = time.time()

    flush_print("=" * 60)
    flush_print("α-RECOVERY ANALYSIS")
    flush_print("=" * 60)
    flush_print(f"α values: {ALPHA_VALUES}")
    flush_print(f"Training pairs: {N_TRAINING_PAIRS}")
    flush_print(f"Content pool: {N_CONTENT}")
    flush_print(f"Epochs: {NUM_EPOCHS}")

    # Generate content pool
    flush_print("\nGenerating content pool...")
    content_probs, _ = generate_content_pool(N_CONTENT, SEED)

    # Train BT models for each α
    flush_print("\nTraining BT models...")
    results: list[dict[str, Any]] = []
    for alpha in ALPHA_VALUES:
        weights = train_bt_for_alpha(content_probs, alpha)

        alpha_ratio = compute_alpha_ratio(weights)
        alpha_sum = compute_alpha_sum_ratio(weights)
        alpha_reg = compute_alpha_regression(weights)

        pos_mean = float(np.mean(weights[POSITIVE_INDICES]))
        neg_mean = float(np.mean(weights[NEGATIVE_INDICES]))

        result = {
            "alpha_true": alpha,
            "alpha_ratio": alpha_ratio,
            "alpha_sum_ratio": alpha_sum,
            "alpha_regression": alpha_reg,
            "pos_mean": pos_mean,
            "neg_mean": neg_mean,
            "weight_norm": float(np.linalg.norm(weights)),
        }
        results.append(result)
        flush_print(
            f"  α={alpha:5.1f} → ratio={alpha_ratio:6.3f}  "
            f"sum={alpha_sum:6.3f}  reg={alpha_reg:6.3f}  "
            f"pos_mean={pos_mean:+.3f}  neg_mean={neg_mean:+.3f}"
        )

    # Also check the three main stakeholders
    flush_print("\nStakeholder validation...")
    stakeholder_results: dict[str, dict[str, Any]] = {}
    for name, alpha in STAKEHOLDER_ALPHAS.items():
        weights = train_bt_for_alpha(content_probs, alpha, seed=SEED + 100)
        alpha_ratio = compute_alpha_ratio(weights)
        stakeholder_results[name] = {
            "alpha_true": alpha,
            "alpha_ratio": alpha_ratio,
            "alpha_regression": compute_alpha_regression(weights),
        }
        flush_print(
            f"  {name:10s}: α_true={alpha:4.1f}  α_recovered={alpha_ratio:.3f}"
        )

    # Compute correlations
    flush_print("\n" + "=" * 60)
    flush_print("CORRELATION ANALYSIS")
    flush_print("=" * 60)

    alpha_true = np.array([r["alpha_true"] for r in results])

    methods = {
        "ratio": np.array([r["alpha_ratio"] for r in results]),
        "sum_ratio": np.array([r["alpha_sum_ratio"] for r in results]),
        "regression": np.array([r["alpha_regression"] for r in results]),
    }

    correlations: dict[str, dict[str, float]] = {}
    for name, alpha_rec in methods.items():
        spearman = float(spearmanr(alpha_true, alpha_rec)[0])
        pearson = float(pearsonr(alpha_true, alpha_rec)[0])

        # Optimal affine transform: α_rec ≈ a + b·α_true
        x_mat = np.column_stack([np.ones_like(alpha_true), alpha_true])
        beta, _, _, _ = np.linalg.lstsq(x_mat, alpha_rec, rcond=None)
        pred = x_mat @ beta
        mae_affine = float(np.mean(np.abs(alpha_rec - pred)))

        # R² of affine fit
        ss_res = float(np.sum((alpha_rec - pred) ** 2))
        ss_tot = float(np.sum((alpha_rec - np.mean(alpha_rec)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        correlations[name] = {
            "spearman": spearman,
            "pearson": pearson,
            "r_squared": r2,
            "mae_after_affine": mae_affine,
            "affine_intercept": float(beta[0]),
            "affine_slope": float(beta[1]),
        }
        flush_print(
            f"  {name:12s}: Spearman={spearman:.4f}  Pearson={pearson:.4f}  "
            f"R²={r2:.4f}  MAE(affine)={mae_affine:.4f}"
        )
        flush_print(
            f"               α_rec ≈ {beta[0]:.3f} + {beta[1]:.3f} · α_true"
        )

    # Summary
    best_method = max(correlations, key=lambda k: correlations[k]["spearman"])
    flush_print(f"\n  Best method: {best_method} "
                f"(Spearman={correlations[best_method]['spearman']:.4f})")

    wall_time = time.time() - start_time
    flush_print(f"  Wall time: {wall_time:.1f}s")

    # Save results
    output_path = project_root / "results" / "alpha_recovery.json"
    output = {
        "config": {
            "n_content": N_CONTENT,
            "n_training_pairs": N_TRAINING_PAIRS,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "alpha_values": ALPHA_VALUES,
        },
        "sweep_results": results,
        "stakeholder_validation": stakeholder_results,
        "correlations": correlations,
        "best_method": best_method,
        "wall_time_seconds": wall_time,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
