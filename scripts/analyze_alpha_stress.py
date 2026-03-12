#!/usr/bin/env python3
"""α-Recovery stress tests: finding the limits of BT weight ratio recovery.

The baseline analysis (Spearman = 1.0) recovers α perfectly under ideal
conditions.  This script tests four stress dimensions to find where recovery
actually breaks down:

  1. Label noise     — random preference flips (annotation quality)
  2. Sample size     — fewer training pairs (data budget)
  3. Temperature     — BT-probabilistic labels (signal strength)
  4. Content corr.   — correlated pos/neg actions (signal ambiguity)

Each dimension is swept independently (others at baseline).  For each
condition: 13 α values × 5 seeds → Spearman, Pearson, per-α MAE.

Usage:
    uv run python scripts/analyze_alpha_stress.py
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, pearsonr, spearmanr

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
NUM_ACTIONS: int = alt_losses.NUM_ACTIONS
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss

loss_exp_mod = load_module(
    "run_loss_experiments",
    project_root / "scripts" / "run_loss_experiments.py",
)
generate_content_pool = loss_exp_mod.generate_content_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_CONTENT = 500
N_PAIRS_BASELINE = 2000
NUM_EPOCHS = 50
LEARNING_RATE = 0.01
N_SEEDS = 5

ALPHA_VALUES = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0]

# Stress dimension sweep values
LABEL_NOISE_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
SAMPLE_SIZE_VALUES = [50, 100, 250, 500, 1000, 2000]
TEMPERATURE_VALUES = [1e6, 2.0, 1.0, 0.5, 0.2]  # 1e6 ≈ hard labels
CORRELATION_VALUES = [0.0, 0.3, 0.6, 0.8]


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Stress condition helpers
# ---------------------------------------------------------------------------


def generate_preference_data_stressed(
    content_probs: np.ndarray,
    alpha: float,
    n_pairs: int,
    p_flip: float = 0.0,
    beta: float = 1e6,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs with optional label noise and BT temperature.

    Args:
        content_probs: (N_content, 18) action probability matrix
        alpha: utility parameter (U = pos - α·neg)
        n_pairs: number of preference pairs to generate
        p_flip: probability of flipping each label (uniform noise)
        beta: BT temperature (∞ = hard, lower = softer preferences)
        seed: random seed
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    utility = pos_scores - alpha * neg_scores

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]

        if beta >= 1e6:
            # Hard label with tiny noise for tie-breaking
            noise = rng.normal(0, 0.05)
            prefer_c1 = (diff + noise) > 0
        else:
            # BT-probabilistic: P(prefer c1) = σ(β · diff)
            z = float(beta * diff)
            z = max(-500.0, min(500.0, z))
            prob_c1 = 1.0 / (1.0 + np.exp(-z))
            prefer_c1 = rng.random() < prob_c1

        if prefer_c1:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    # Apply label noise (uniform flips)
    if p_flip > 0:
        flip_mask = rng.random(n_pairs) < p_flip
        temp = probs_pref[flip_mask].copy()
        probs_pref[flip_mask] = probs_rej[flip_mask]
        probs_rej[flip_mask] = temp

    return probs_pref, probs_rej


def generate_correlated_content(
    n_content: int,
    rho: float,
    seed: int = SEED,
) -> np.ndarray:
    """Generate content with controlled pos-neg score correlation.

    Uses bivariate normal copula to generate correlated multipliers
    for positive and negative action probabilities.
    """
    rng = np.random.default_rng(seed)

    # Correlated latent factors
    cov = [[1.0, rho], [rho, 1.0]]
    z = rng.multivariate_normal([0.0, 0.0], cov, size=n_content)

    # Map to [0.5, 2.0] multiplier range via normal CDF
    pos_mult = 0.5 + 1.5 * norm.cdf(z[:, 0])
    neg_mult = 0.5 + 1.5 * norm.cdf(z[:, 1])

    content_probs = np.zeros((n_content, NUM_ACTIONS), dtype=np.float32)
    for i in range(n_content):
        base = rng.uniform(0.05, 0.3, NUM_ACTIONS).astype(np.float32)
        for idx in POSITIVE_INDICES:
            base[idx] *= pos_mult[i]
        for idx in NEGATIVE_INDICES:
            base[idx] *= neg_mult[i]
        content_probs[i] = np.clip(base, 0, 1)

    return content_probs


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def compute_alpha_ratio(weights: np.ndarray) -> float:
    """Recover α as -mean(w_neg) / mean(w_pos)."""
    pos_mean = float(np.mean(weights[POSITIVE_INDICES]))
    neg_mean = float(np.mean(weights[NEGATIVE_INDICES]))
    if abs(pos_mean) < 1e-10:
        return float("inf")
    return -neg_mean / pos_mean


def train_and_recover(
    content_probs: np.ndarray,
    alpha: float,
    n_pairs: int = N_PAIRS_BASELINE,
    p_flip: float = 0.0,
    beta: float = 1e6,
    seed: int = SEED,
) -> float:
    """Train BT model under stress conditions and return recovered α."""
    probs_pref, probs_rej = generate_preference_data_stressed(
        content_probs, alpha, n_pairs, p_flip=p_flip, beta=beta, seed=seed,
    )
    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    model = train_with_loss(config, probs_pref, probs_rej, verbose=False)
    return compute_alpha_ratio(np.array(model.weights))


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------


def run_condition(
    content_probs: np.ndarray,
    n_pairs: int = N_PAIRS_BASELINE,
    p_flip: float = 0.0,
    beta: float = 1e6,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run all α values for one (condition, seed).  Return metrics."""
    alphas_true = np.array(ALPHA_VALUES)
    alphas_recovered = np.zeros(len(ALPHA_VALUES))

    for i, alpha in enumerate(ALPHA_VALUES):
        alphas_recovered[i] = train_and_recover(
            content_probs, alpha, n_pairs=n_pairs, p_flip=p_flip,
            beta=beta, seed=seed + i * 1000,
        )

    # Metrics
    spearman = float(spearmanr(alphas_true, alphas_recovered).statistic)
    pearson = float(pearsonr(alphas_true, alphas_recovered).statistic)

    # Per-α MAE after affine transform
    x_mat = np.column_stack([np.ones_like(alphas_true), alphas_true])
    beta_fit, _, _, _ = np.linalg.lstsq(x_mat, alphas_recovered, rcond=None)
    pred = x_mat @ beta_fit
    mae_affine = float(np.mean(np.abs(alphas_recovered - pred)))

    return {
        "alphas_recovered": alphas_recovered.tolist(),
        "spearman": spearman,
        "pearson": pearson,
        "mae_affine": mae_affine,
        "affine_intercept": float(beta_fit[0]),
        "affine_slope": float(beta_fit[1]),
    }


def run_dimension_sweep(
    dimension_name: str,
    values: list[Any],
    content_pools: dict[Any, np.ndarray],
    make_kwargs: Any,
    n_seeds: int = N_SEEDS,
) -> dict[str, Any]:
    """Sweep one stress dimension across all values and seeds.

    Args:
        dimension_name: name for logging
        values: parameter values to sweep
        content_pools: map from value → content_probs array
        make_kwargs: callable(value) → dict of kwargs for run_condition
        n_seeds: number of seeds per condition
    """
    flush_print(f"\n{'='*60}")
    flush_print(f"DIMENSION: {dimension_name}")
    flush_print(f"{'='*60}")

    results: list[dict[str, Any]] = []

    for val in values:
        seed_results: list[dict[str, Any]] = []
        content = content_pools[val]
        kwargs = make_kwargs(val)

        for s in range(n_seeds):
            seed = SEED + s * 100
            kwargs["seed"] = seed
            r = run_condition(content, **kwargs)
            seed_results.append(r)

        # Aggregate across seeds
        spearmans = [r["spearman"] for r in seed_results]
        pearsons = [r["pearson"] for r in seed_results]
        maes = [r["mae_affine"] for r in seed_results]

        agg = {
            "value": val,
            "spearman_mean": float(np.mean(spearmans)),
            "spearman_std": float(np.std(spearmans)),
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_std": float(np.std(pearsons)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "seed_results": seed_results,
        }
        results.append(agg)

        flush_print(
            f"  {dimension_name}={val!s:>6s}  "
            f"Spearman={agg['spearman_mean']:.4f}±{agg['spearman_std']:.4f}  "
            f"Pearson={agg['pearson_mean']:.4f}±{agg['pearson_std']:.4f}  "
            f"MAE={agg['mae_mean']:.4f}±{agg['mae_std']:.4f}"
        )

    # Find breaking point (Spearman < 0.95)
    breaking_point = None
    for r in results:
        if r["spearman_mean"] < 0.95:
            breaking_point = r["value"]
            break

    flush_print(
        f"  Breaking point (Spearman < 0.95): "
        f"{breaking_point if breaking_point is not None else 'none (all pass)'}"
    )

    return {
        "dimension": dimension_name,
        "results": results,
        "breaking_point": breaking_point,
    }


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
    flush_print("α-RECOVERY STRESS TESTS")
    flush_print("=" * 60)
    flush_print(f"α values: {ALPHA_VALUES} ({len(ALPHA_VALUES)} points)")
    flush_print(f"Seeds per condition: {N_SEEDS}")
    flush_print(f"Baseline: {N_PAIRS_BASELINE} pairs, {NUM_EPOCHS} epochs")

    # Generate baseline content pool
    flush_print("\nGenerating baseline content pool...")
    baseline_content, _ = generate_content_pool(N_CONTENT, SEED)

    dimensions: list[dict[str, Any]] = []

    # --- Dimension 1: Label noise ---
    dim1 = run_dimension_sweep(
        dimension_name="label_noise",
        values=LABEL_NOISE_VALUES,
        content_pools={v: baseline_content for v in LABEL_NOISE_VALUES},
        make_kwargs=lambda v: {"n_pairs": N_PAIRS_BASELINE, "p_flip": v, "beta": 1e6},
    )
    dimensions.append(dim1)

    # --- Dimension 2: Sample size ---
    dim2 = run_dimension_sweep(
        dimension_name="sample_size",
        values=SAMPLE_SIZE_VALUES,
        content_pools={v: baseline_content for v in SAMPLE_SIZE_VALUES},
        make_kwargs=lambda v: {"n_pairs": v, "p_flip": 0.0, "beta": 1e6},
    )
    dimensions.append(dim2)

    # --- Dimension 3: Preference temperature ---
    dim3 = run_dimension_sweep(
        dimension_name="temperature",
        values=TEMPERATURE_VALUES,
        content_pools={v: baseline_content for v in TEMPERATURE_VALUES},
        make_kwargs=lambda v: {"n_pairs": N_PAIRS_BASELINE, "p_flip": 0.0, "beta": v},
    )
    dimensions.append(dim3)

    # --- Dimension 4: Content correlation ---
    flush_print("\nGenerating correlated content pools...")
    corr_content_pools = {}
    for rho in CORRELATION_VALUES:
        if rho == 0.0:
            corr_content_pools[rho] = baseline_content
        else:
            corr_content_pools[rho] = generate_correlated_content(
                N_CONTENT, rho, seed=SEED,
            )

    dim4 = run_dimension_sweep(
        dimension_name="content_correlation",
        values=CORRELATION_VALUES,
        content_pools=corr_content_pools,
        make_kwargs=lambda _v: {"n_pairs": N_PAIRS_BASELINE, "p_flip": 0.0, "beta": 1e6},
    )
    dimensions.append(dim4)

    # --- Summary ---
    wall_time = time.time() - start_time

    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)

    summary_rows: list[dict[str, Any]] = []
    for dim in dimensions:
        bp = dim["breaking_point"]
        # Find the best and worst Spearman
        best = max(dim["results"], key=lambda r: r["spearman_mean"])
        worst = min(dim["results"], key=lambda r: r["spearman_mean"])
        row = {
            "dimension": dim["dimension"],
            "breaking_point": bp,
            "best_spearman": best["spearman_mean"],
            "best_value": best["value"],
            "worst_spearman": worst["spearman_mean"],
            "worst_value": worst["value"],
        }
        summary_rows.append(row)
        flush_print(
            f"  {dim['dimension']:22s}  "
            f"break={str(bp):>6s}  "
            f"best={best['spearman_mean']:.4f} @{best['value']}  "
            f"worst={worst['spearman_mean']:.4f} @{worst['value']}"
        )

    flush_print(f"\n  Wall time: {wall_time:.1f}s ({wall_time/60:.1f}m)")

    total_runs = sum(
        len(d["results"]) * N_SEEDS * len(ALPHA_VALUES) for d in dimensions
    )
    flush_print(f"  Total training runs: {total_runs}")

    # Save results
    output_path = project_root / "results" / "alpha_recovery_stress.json"
    output = {
        "config": {
            "n_content": N_CONTENT,
            "n_pairs_baseline": N_PAIRS_BASELINE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "n_seeds": N_SEEDS,
            "alpha_values": ALPHA_VALUES,
        },
        "dimensions": dimensions,
        "summary": summary_rows,
        "wall_time_seconds": wall_time,
        "total_training_runs": total_runs,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
