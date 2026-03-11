#!/usr/bin/env python3
"""Disagreement → differentiation bound analysis for F4 reward modeling.

Answers: what is the functional form relating label disagreement rate between
two stakeholders to the cosine similarity of their learned BT weight vectors?
Is the relationship monotonic? Can we establish a usable lower bound?

Approach: sweep the negative-penalty parameter α in U(pos, neg) = pos - α·neg
to generate a continuous range of disagreement rates, train BT models for each,
and fit curves to (disagreement_rate, cosine_similarity) pairs.

Usage:
    uv run python scripts/analyze_disagreement_bound.py
"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import optimize
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Module loading (same pattern as run_loss_experiments.py)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_module(module_name: str, module_path: Path, register_as: str | None = None):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if register_as:
        sys.modules[register_as] = module
    spec.loader.exec_module(module)
    return module


# Load alternative_losses
alt_losses = load_module(
    "alternative_losses",
    project_root / "enhancements" / "reward_modeling" / "alternative_losses.py",
    register_as="enhancements.reward_modeling.alternative_losses",
)

train_with_loss = alt_losses.train_with_loss
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
POSITIVE_INDICES = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alt_losses.NEGATIVE_INDICES
NUM_ACTIONS = alt_losses.NUM_ACTIONS
ACTION_INDICES = alt_losses.ACTION_INDICES

# Load generate_content_pool from run_loss_experiments
loss_exp_mod = load_module(
    "run_loss_experiments",
    project_root / "scripts" / "run_loss_experiments.py",
)
generate_content_pool = loss_exp_mod.generate_content_pool


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CONTENT = 500
N_PAIRS = 5000
SEED = 42
NUM_EPOCHS = 200
LEARNING_RATE = 0.01


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def compute_disagreement_rate(
    content_probs: np.ndarray,
    neg_penalty_a: float,
    neg_penalty_b: float,
    n_pairs: int,
    seed: int = 42,
) -> float:
    """Compute label disagreement rate between two utility functions.

    Each utility is U(pos, neg) = pos - α·neg. Disagreement = fraction of
    content pairs where the two utilities prefer different items.
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    # Pre-compute per-content utility scores
    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])

    utility_a = pos_scores - neg_penalty_a * neg_scores
    utility_b = pos_scores - neg_penalty_b * neg_scores

    # Sample random pairs and count disagreements
    disagree_count = 0
    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        prefer_a = utility_a[c1] > utility_a[c2]
        prefer_b = utility_b[c1] > utility_b[c2]
        if prefer_a != prefer_b:
            disagree_count += 1

    return disagree_count / n_pairs


def compute_disagreement_with_margins(
    content_probs: np.ndarray,
    neg_penalty_a: float,
    neg_penalty_b: float,
    n_pairs: int,
    seed: int = 42,
) -> dict[str, float]:
    """Compute disagreement rate AND margin statistics on disagreed pairs.

    For each disagreed pair, tracks:
    - margin_a: |U_a(c1) - U_a(c2)| (stakeholder A's confidence)
    - margin_b: |U_b(c1) - U_b(c2)| (stakeholder B's confidence)
    - total_margin: margin_a + margin_b = |Δα| · |Δneg| for our utility family
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])

    utility_a = pos_scores - neg_penalty_a * neg_scores
    utility_b = pos_scores - neg_penalty_b * neg_scores

    disagree_count = 0
    margins_total: list[float] = []
    margins_a: list[float] = []
    margins_b: list[float] = []
    delta_negs: list[float] = []

    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        prefer_a = utility_a[c1] > utility_a[c2]
        prefer_b = utility_b[c1] > utility_b[c2]
        if prefer_a != prefer_b:
            disagree_count += 1
            ma = float(abs(utility_a[c1] - utility_a[c2]))
            mb = float(abs(utility_b[c1] - utility_b[c2]))
            margins_a.append(ma)
            margins_b.append(mb)
            margins_total.append(ma + mb)
            delta_negs.append(float(abs(neg_scores[c1] - neg_scores[c2])))

    rate = disagree_count / n_pairs
    if margins_total:
        result = {
            "rate": rate,
            "mean_margin_a": float(np.mean(margins_a)),
            "mean_margin_b": float(np.mean(margins_b)),
            "mean_total_margin": float(np.mean(margins_total)),
            "margin_std": float(np.std(margins_total)),
            "mean_delta_neg": float(np.mean(delta_negs)),
        }
    else:
        result = {
            "rate": rate,
            "mean_margin_a": 0.0,
            "mean_margin_b": 0.0,
            "mean_total_margin": 0.0,
            "margin_std": 0.0,
            "mean_delta_neg": 0.0,
        }
    return result


def generate_preference_data(
    content_probs: np.ndarray,
    neg_penalty: float,
    n_pairs: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs for a utility function U = pos - α·neg.

    Returns: (probs_preferred[N, 18], probs_rejected[N, 18])
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])

    utility = pos_scores - neg_penalty * neg_scores

    # Add small noise for tie-breaking (matching run_loss_experiments.py)
    noise = rng.normal(0, 0.05, (n_pairs, 2))

    probs_preferred = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rejected = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        score1 = utility[c1] + noise[i, 0]
        score2 = utility[c2] + noise[i, 1]

        if score1 > score2:
            probs_preferred[i] = content_probs[c1]
            probs_rejected[i] = content_probs[c2]
        else:
            probs_preferred[i] = content_probs[c2]
            probs_rejected[i] = content_probs[c1]

    return probs_preferred, probs_rejected


def train_bt_weights(
    probs_preferred: np.ndarray,
    probs_rejected: np.ndarray,
    label: str = "model",
) -> np.ndarray:
    """Train standard BT model and return weight vector."""
    config = LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=StakeholderType.USER,  # Label doesn't affect training
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    model = train_with_loss(config, probs_preferred, probs_rejected, verbose=False)
    return np.array(model.weights)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_sweep(
    content_probs: np.ndarray,
    fixed_penalty: float,
    sweep_penalties: list[float],
    n_pairs: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Run a full sweep: for each α_sweep, compute disagreement and train BT.

    Returns list of {neg_penalty_fixed, neg_penalty_sweep, disagreement_rate,
    cosine_similarity, accuracy_fixed, accuracy_sweep}.
    """
    results = []

    # Train the fixed-penalty model once
    flush_print(f"  Training fixed model (α={fixed_penalty})...")
    pref_fixed, rej_fixed = generate_preference_data(
        content_probs, fixed_penalty, n_pairs, seed
    )
    weights_fixed = train_bt_weights(pref_fixed, rej_fixed, f"fixed_{fixed_penalty}")

    for i, alpha_sweep in enumerate(sweep_penalties):
        flush_print(f"  [{i+1}/{len(sweep_penalties)}] α_sweep={alpha_sweep:.1f}...", end=" ")
        t0 = time.time()

        # Disagreement rate
        dis_rate = compute_disagreement_rate(
            content_probs, fixed_penalty, alpha_sweep, n_pairs, seed
        )

        # Train sweep model
        pref_sweep, rej_sweep = generate_preference_data(
            content_probs, alpha_sweep, n_pairs, seed
        )
        weights_sweep = train_bt_weights(pref_sweep, rej_sweep, f"sweep_{alpha_sweep}")

        # Cosine similarity
        cos = cosine_sim(weights_fixed, weights_sweep)

        elapsed = time.time() - t0
        flush_print(f"disagreement={dis_rate:.1%}, cosine={cos:.4f} ({elapsed:.1f}s)")

        results.append({
            "neg_penalty_fixed": fixed_penalty,
            "neg_penalty_sweep": alpha_sweep,
            "disagreement_rate": dis_rate,
            "cosine_similarity": cos,
        })

    return results


# ---------------------------------------------------------------------------
# Curve fitting
# ---------------------------------------------------------------------------


def fit_curves(
    disagreement_rates: np.ndarray,
    cosine_similarities: np.ndarray,
) -> dict[str, Any]:
    """Fit multiple functional forms and return best fit with R² values.

    Curve forms:
    - Linear: cos = a - b*d
    - Logarithmic: cos = a - b*log(d + ε)
    - Power law: cos = a*(1-d)^b
    - Exponential decay: cos = a*exp(-b*d)
    """
    d = disagreement_rates
    c = cosine_similarities

    # Filter out zero-disagreement points (identical utilities)
    mask = d > 0.001
    d_fit = d[mask]
    c_fit = c[mask]

    if len(d_fit) < 3:
        return {"error": "Too few non-zero disagreement points for curve fitting"}

    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    fits = {}

    # Linear: cos = a - b*d
    try:
        def linear(x, a, b):
            return a - b * x
        popt, _ = optimize.curve_fit(linear, d_fit, c_fit, p0=[1.0, 1.0])
        pred = linear(d_fit, *popt)
        fits["linear"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r_squared": r_squared(c_fit, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * d",
        }
    except (RuntimeError, ValueError):
        pass

    # Logarithmic: cos = a - b*log(d + ε)
    try:
        def logarithmic(x, a, b):
            return a - b * np.log(x + 1e-6)
        popt, _ = optimize.curve_fit(logarithmic, d_fit, c_fit, p0=[0.5, 0.1])
        pred = logarithmic(d_fit, *popt)
        fits["logarithmic"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r_squared": r_squared(c_fit, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * log(d)",
        }
    except (RuntimeError, ValueError):
        pass

    # Power law: cos = a*(1-d)^b
    try:
        def power_law(x, a, b):
            return a * np.power(np.clip(1 - x, 1e-8, 1.0), b)
        popt, _ = optimize.curve_fit(power_law, d_fit, c_fit, p0=[1.0, 1.0],
                                     bounds=([0, 0], [2, 20]))
        pred = power_law(d_fit, *popt)
        fits["power_law"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r_squared": r_squared(c_fit, pred),
            "formula": f"cos = {popt[0]:.4f} * (1-d)^{popt[1]:.4f}",
        }
    except (RuntimeError, ValueError):
        pass

    # Exponential decay: cos = a*exp(-b*d)
    try:
        def exp_decay(x, a, b):
            return a * np.exp(-b * x)
        popt, _ = optimize.curve_fit(exp_decay, d_fit, c_fit, p0=[1.0, 5.0],
                                     bounds=([0, 0], [2, 100]))
        pred = exp_decay(d_fit, *popt)
        fits["exponential"] = {
            "params": {"a": float(popt[0]), "b": float(popt[1])},
            "r_squared": r_squared(c_fit, pred),
            "formula": f"cos = {popt[0]:.4f} * exp(-{popt[1]:.4f} * d)",
        }
    except (RuntimeError, ValueError):
        pass

    if not fits:
        return {"error": "All curve fits failed"}

    # Find best fit
    best_name = max(fits, key=lambda k: fits[k]["r_squared"])
    best_fit = fits[best_name]

    # Compute thresholds: at what disagreement does cosine drop below 0.95, 0.80, 0.50?
    thresholds = {}
    best_params = fits[best_name]["params"]

    def _predict(x):
        a, b = best_params["a"], best_params["b"]
        if best_name == "linear":
            return a - b * x
        elif best_name == "logarithmic":
            return a - b * np.log(x + 1e-6)
        elif best_name == "power_law":
            return a * np.power(np.clip(1 - x, 1e-8, 1.0), b)
        elif best_name == "exponential":
            return a * np.exp(-b * x)
        return np.full_like(x, np.nan)

    for threshold in [0.95, 0.80, 0.50]:
        d_test = np.linspace(0.001, 0.5, 1000)
        cos_test = _predict(d_test)
        crossings = np.where(cos_test < threshold)[0]
        if len(crossings) > 0:
            thresholds[f"cos_below_{threshold}"] = float(
                d_test[crossings[0]]
            )
        else:
            thresholds[f"cos_below_{threshold}"] = ">50%"

    return {
        "fits": fits,
        "best_fit": best_name,
        "best_r_squared": best_fit["r_squared"],
        "best_formula": best_fit["formula"],
        "thresholds": thresholds,
    }


def fit_augmented_models(
    disagreement_rates: np.ndarray,
    cosine_similarities: np.ndarray,
    mean_margins: np.ndarray,
    alpha_distances: np.ndarray,
) -> dict[str, Any]:
    """Fit models using margin as additional predictor. Compare R² values.

    Tests:
    1. disagreement_rate only (baseline)
    2. mean_margin only
    3. |Δα| only
    4. 2-variable: d + m
    5. product: d × m
    """
    d = disagreement_rates
    c = cosine_similarities
    m = mean_margins
    da = alpha_distances

    # Filter zero-disagreement points
    mask = d > 0.001
    d_f, c_f, m_f, da_f = d[mask], c[mask], m[mask], da[mask]

    if len(d_f) < 4:
        return {"error": "Too few points for augmented fitting"}

    def r_squared(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    results: dict[str, Any] = {}

    # 1. Disagreement-only (baseline)
    try:
        def lin1(x, a, b):
            return a - b * x
        popt, _ = optimize.curve_fit(lin1, d_f, c_f, p0=[1.0, 1.0])
        pred = lin1(d_f, *popt)
        results["disagreement_only"] = {
            "r_squared": r_squared(c_f, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * d",
            "params": {"a": float(popt[0]), "b": float(popt[1])},
        }
    except (RuntimeError, ValueError):
        pass

    # 2. Margin-only
    try:
        popt, _ = optimize.curve_fit(lin1, m_f, c_f, p0=[1.0, 0.5])
        pred = lin1(m_f, *popt)
        results["margin_only"] = {
            "r_squared": r_squared(c_f, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * m",
            "params": {"a": float(popt[0]), "b": float(popt[1])},
        }
    except (RuntimeError, ValueError):
        pass

    # 3. |Δα| only
    try:
        popt, _ = optimize.curve_fit(lin1, da_f, c_f, p0=[1.0, 0.1])
        pred = lin1(da_f, *popt)
        results["alpha_distance_only"] = {
            "r_squared": r_squared(c_f, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * |Δα|",
            "params": {"a": float(popt[0]), "b": float(popt[1])},
        }
    except (RuntimeError, ValueError):
        pass

    # 4. 2-variable: cos = a - b1*d - b2*m
    try:
        X = np.column_stack([np.ones_like(d_f), d_f, m_f])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, c_f, rcond=None)
        pred = X @ coeffs
        results["two_variable"] = {
            "r_squared": r_squared(c_f, pred),
            "formula": (
                f"cos = {coeffs[0]:.4f} "
                f"+ {coeffs[1]:+.4f} * d "
                f"+ {coeffs[2]:+.4f} * m"
            ),
            "params": {
                "intercept": float(coeffs[0]),
                "b_disagreement": float(coeffs[1]),
                "b_margin": float(coeffs[2]),
            },
        }
    except np.linalg.LinAlgError:
        pass

    # 5. Product: cos = a - b * (d × m)
    try:
        product = d_f * m_f
        popt, _ = optimize.curve_fit(lin1, product, c_f, p0=[1.0, 1.0])
        pred = lin1(product, *popt)
        results["product_d_m"] = {
            "r_squared": r_squared(c_f, pred),
            "formula": f"cos = {popt[0]:.4f} - {popt[1]:.4f} * (d × m)",
            "params": {"a": float(popt[0]), "b": float(popt[1])},
        }
    except (RuntimeError, ValueError):
        pass

    # Find best
    if results:
        best = max(results, key=lambda k: results[k]["r_squared"])
        return {
            "models": results,
            "best_model": best,
            "best_r_squared": results[best]["r_squared"],
        }
    return {"error": "All augmented fits failed"}


# ---------------------------------------------------------------------------
# Go/No-Go evaluation
# ---------------------------------------------------------------------------


def evaluate_go_no_go(
    all_d: np.ndarray,
    all_c: np.ndarray,
    best_r_squared: float,
    known_points: list[dict[str, float]],
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate the go/no-go criteria from the plan."""
    results = {}

    # 1. Best-fit R² ≥ 0.80
    results["r_squared"] = {
        "value": best_r_squared,
        "threshold": 0.80,
        "pass": best_r_squared >= 0.80,
    }

    # 2. Monotonicity: Spearman ρ(d, cos) ≤ -0.85
    mask = all_d > 0.001  # exclude identical-utility points
    if mask.sum() >= 3:
        rho, _ = spearmanr(all_d[mask], all_c[mask])
        results["monotonicity"] = {
            "spearman_rho": float(rho),
            "threshold": -0.85,
            "pass": rho <= -0.85,
        }
    else:
        results["monotonicity"] = {"pass": False, "reason": "insufficient data"}

    # 3. Consistency: known points within 0.10 of fitted curve
    if "fits" in fit_result and fit_result["best_fit"] in fit_result["fits"]:
        best_name = fit_result["best_fit"]
        params = fit_result["fits"][best_name]["params"]
        residuals = []

        for kp in known_points:
            d_known = kp["disagreement_rate"]
            c_known = kp["cosine_similarity"]

            if best_name == "linear":
                c_pred = params["a"] - params["b"] * d_known
            elif best_name == "logarithmic":
                c_pred = params["a"] - params["b"] * np.log(d_known + 1e-6)
            elif best_name == "power_law":
                c_pred = params["a"] * np.power(max(1 - d_known, 1e-8), params["b"])
            elif best_name == "exponential":
                c_pred = params["a"] * np.exp(-params["b"] * d_known)
            else:
                c_pred = c_known  # fallback

            residual = abs(c_known - c_pred)
            residuals.append({
                "pair": kp["pair"],
                "actual": c_known,
                "predicted": c_pred,
                "residual": residual,
            })

        max_residual = max(r["residual"] for r in residuals)
        results["consistency"] = {
            "residuals": residuals,
            "max_residual": max_residual,
            "threshold": 0.10,
            "pass": max_residual <= 0.10,
        }
    else:
        results["consistency"] = {"pass": False, "reason": "no fit available"}

    # 4. Non-trivial range: disagreement spans ≥ 25 percentage points
    d_range = float(np.max(all_d[mask]) - np.min(all_d[mask])) if mask.sum() >= 2 else 0.0
    results["range"] = {
        "disagreement_range_pp": d_range * 100,
        "threshold_pp": 25.0,
        "pass": d_range * 100 >= 25.0,
    }

    # Overall
    all_pass = all(v["pass"] for v in results.values())
    results["overall"] = "PASS" if all_pass else "FAIL"

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    flush_print("=" * 70)
    flush_print("F4 Disagreement → Differentiation Bound Analysis")
    flush_print("=" * 70)
    flush_print("\nQuestion: What functional form relates label disagreement")
    flush_print("rate to learned weight vector cosine similarity?")

    # 1. Generate content pool
    flush_print("\n[1/5] Generating content pool...")
    content_probs, content_topics = generate_content_pool(N_CONTENT, SEED)
    flush_print(f"  Content items: {N_CONTENT}, Pairs per sweep point: {N_PAIRS}")

    # 2. Sweep A: Fix Platform (α=0.3), sweep partner
    flush_print("\n[2/5] Sweep A: Fixed Platform (α=0.3)")
    sweep_a_penalties = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0,
                         3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
    sweep_a = run_sweep(content_probs, 0.3, sweep_a_penalties, N_PAIRS, SEED)

    # 3. Sweep B: Fix User (α=1.0), sweep partner
    flush_print("\n[3/5] Sweep B: Fixed User (α=1.0)")
    sweep_b_penalties = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0,
                         2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0]
    sweep_b = run_sweep(content_probs, 1.0, sweep_b_penalties, N_PAIRS, SEED)

    # 4. Curve fitting
    flush_print("\n[4/5] Fitting curves...")

    # Combine all sweep points (excluding duplicates at identical alphas)
    all_points = sweep_a + sweep_b
    all_d = np.array([p["disagreement_rate"] for p in all_points])
    all_c = np.array([p["cosine_similarity"] for p in all_points])

    fit_result = fit_curves(all_d, all_c)

    if "error" in fit_result:
        flush_print(f"  Error: {fit_result['error']}")
    else:
        flush_print(f"\n  Best fit: {fit_result['best_fit']}")
        flush_print(f"  R²: {fit_result['best_r_squared']:.4f}")
        flush_print(f"  Formula: {fit_result['best_formula']}")

        flush_print("\n  All fits:")
        for name, fit in fit_result["fits"].items():
            flush_print(f"    {name:<15} R²={fit['r_squared']:.4f}  {fit['formula']}")

        if "thresholds" in fit_result:
            flush_print("\n  Differentiation thresholds (best fit):")
            for key, val in fit_result["thresholds"].items():
                if isinstance(val, float):
                    flush_print(f"    {key}: {val:.1%} disagreement needed")
                else:
                    flush_print(f"    {key}: {val}")

    # 5. Go/No-Go evaluation
    flush_print("\n[5/5] Go/No-Go Evaluation")
    flush_print("=" * 70)

    # Known data points from comparison.json (bradley loss)
    known_points = [
        {"pair": "U-P", "disagreement_rate": 0.12, "cosine_similarity": 0.830},
        {"pair": "U-S", "disagreement_rate": 0.23, "cosine_similarity": 0.884},
        {"pair": "P-S", "disagreement_rate": 0.35, "cosine_similarity": 0.478},
    ]

    # Recompute actual disagreement rates for known utility pairs
    flush_print("\n  Recomputing known-point disagreement rates for consistency...")
    actual_rates = {
        "U-P": compute_disagreement_rate(content_probs, 1.0, 0.3, N_PAIRS, SEED),
        "U-S": compute_disagreement_rate(content_probs, 1.0, 4.0, N_PAIRS, SEED),
        "P-S": compute_disagreement_rate(content_probs, 0.3, 4.0, N_PAIRS, SEED),
    }
    for pair, rate in actual_rates.items():
        flush_print(f"    {pair}: {rate:.1%}")
        # Update known points with actual rates
        for kp in known_points:
            if kp["pair"] == pair:
                kp["disagreement_rate"] = rate

    go_no_go = evaluate_go_no_go(
        all_d, all_c,
        fit_result.get("best_r_squared", 0.0),
        known_points,
        fit_result,
    )

    flush_print(f"\n  {'Criterion':<25} {'Value':>10} {'Threshold':>10} {'Status':>8}")
    flush_print("  " + "-" * 58)

    r2_info = go_no_go["r_squared"]
    flush_print(
        f"  {'R²':<25} {r2_info['value']:>10.4f} {'≥0.80':>10} "
        f"{'PASS' if r2_info['pass'] else 'FAIL':>8}"
    )

    mono_info = go_no_go["monotonicity"]
    if "spearman_rho" in mono_info:
        flush_print(
            f"  {'Monotonicity (ρ)':<25} {mono_info['spearman_rho']:>10.4f} {'≤-0.85':>10} "
            f"{'PASS' if mono_info['pass'] else 'FAIL':>8}"
        )

    cons_info = go_no_go["consistency"]
    if "max_residual" in cons_info:
        flush_print(
            f"  {'Known-point residual':<25} {cons_info['max_residual']:>10.4f} {'≤0.10':>10} "
            f"{'PASS' if cons_info['pass'] else 'FAIL':>8}"
        )
        for r in cons_info["residuals"]:
            flush_print(
                f"    {r['pair']}: actual={r['actual']:.3f}, "
                f"predicted={r['predicted']:.3f}, Δ={r['residual']:.3f}"
            )

    range_info = go_no_go["range"]
    flush_print(
        f"  {'Disagree range (pp)':<25} {range_info['disagreement_range_pp']:>10.1f} {'≥25.0':>10} "
        f"{'PASS' if range_info['pass'] else 'FAIL':>8}"
    )

    flush_print(f"\n  Overall: {go_no_go['overall']}")
    flush_print("=" * 70)

    # Print sweep data table
    flush_print("\n  Sweep data (sorted by disagreement rate):")
    flush_print(f"  {'Sweep':>6} {'α_fixed':>8} {'α_sweep':>8} {'Disagree':>10} {'Cosine':>10}")
    flush_print("  " + "-" * 48)
    for p in sorted(all_points, key=lambda x: x["disagreement_rate"]):
        sweep_label = "A" if p["neg_penalty_fixed"] == 0.3 else "B"
        flush_print(
            f"  {sweep_label:>6} {p['neg_penalty_fixed']:>8.1f} {p['neg_penalty_sweep']:>8.1f} "
            f"{p['disagreement_rate']:>10.1%} {p['cosine_similarity']:>10.4f}"
        )

    # -----------------------------------------------------------------------
    # Margin-augmented analysis (no new training — just post-processing)
    # -----------------------------------------------------------------------
    flush_print("\n" + "=" * 70)
    flush_print("MARGIN-AUGMENTED ANALYSIS")
    flush_print("=" * 70)
    flush_print("\nHypothesis: mean margin on disagreed pairs explains")
    flush_print("the cross-sweep variance that disagreement rate alone misses.")

    flush_print("\nComputing margin stats for all sweep points...")
    margin_stats = []
    for p in all_points:
        ms = compute_disagreement_with_margins(
            content_probs,
            p["neg_penalty_fixed"],
            p["neg_penalty_sweep"],
            N_PAIRS,
            SEED,
        )
        ms["alpha_distance"] = abs(
            p["neg_penalty_sweep"] - p["neg_penalty_fixed"]
        )
        margin_stats.append(ms)

    # Extract arrays for fitting
    all_m = np.array([ms["mean_total_margin"] for ms in margin_stats])
    all_da = np.array([ms["alpha_distance"] for ms in margin_stats])

    # Print margin data table
    flush_print(
        f"\n  {'Sweep':>6} {'α_fixed':>8} {'α_sweep':>8} "
        f"{'Disagree':>10} {'Margin':>10} {'|Δα|':>8} {'Cosine':>10}"
    )
    flush_print("  " + "-" * 72)
    sorted_idx = np.argsort(all_d)
    for i in sorted_idx:
        p = all_points[i]
        ms = margin_stats[i]
        sweep_label = "A" if p["neg_penalty_fixed"] == 0.3 else "B"
        flush_print(
            f"  {sweep_label:>6} "
            f"{p['neg_penalty_fixed']:>8.1f} "
            f"{p['neg_penalty_sweep']:>8.1f} "
            f"{p['disagreement_rate']:>10.1%} "
            f"{ms['mean_total_margin']:>10.4f} "
            f"{ms['alpha_distance']:>8.1f} "
            f"{p['cosine_similarity']:>10.4f}"
        )

    # Fit augmented models
    flush_print("\nFitting augmented models...")
    aug_result = fit_augmented_models(all_d, all_c, all_m, all_da)

    if "error" in aug_result:
        flush_print(f"  Error: {aug_result['error']}")
    else:
        flush_print(f"\n  {'Model':<25} {'R²':>10}")
        flush_print("  " + "-" * 38)
        for name, model in sorted(
            aug_result["models"].items(),
            key=lambda x: x[1]["r_squared"],
            reverse=True,
        ):
            marker = " ← best" if name == aug_result["best_model"] else ""
            flush_print(
                f"  {name:<25} {model['r_squared']:>10.4f}{marker}"
            )
            flush_print(f"    {model['formula']}")

        flush_print(f"\n  Best model: {aug_result['best_model']}")
        flush_print(f"  Best R²: {aug_result['best_r_squared']:.4f}")

        # Compare to baseline
        baseline_r2 = 0.898  # from disagreement-only fit
        improvement = aug_result["best_r_squared"] - baseline_r2
        flush_print(
            f"  Improvement over disagreement-only: "
            f"{improvement:+.4f} ({improvement/baseline_r2*100:+.1f}%)"
        )

        # Hypothesis evaluation
        flush_print("\n  Hypothesis evaluation:")
        best_r2 = aug_result["best_r_squared"]
        flush_print(
            f"    2-var R² improvement ≥ 0.05: "
            f"{'PASS' if improvement >= 0.05 else 'FAIL'} ({improvement:+.4f})"
        )
        flush_print(
            f"    Best R² ≥ 0.95: "
            f"{'PASS' if best_r2 >= 0.95 else 'FAIL'} ({best_r2:.4f})"
        )

        # Check product model specifically
        if "product_d_m" in aug_result["models"]:
            prod_r2 = aug_result["models"]["product_d_m"]["r_squared"]
            flush_print(
                f"    Product R² ≥ 0.95: "
                f"{'PASS' if prod_r2 >= 0.95 else 'FAIL'} ({prod_r2:.4f})"
            )

    flush_print("=" * 70)

    # Print sweep data table (original)
    flush_print("\n  Sweep data (sorted by disagreement rate):")
    flush_print(
        f"  {'Sweep':>6} {'α_fixed':>8} {'α_sweep':>8} "
        f"{'Disagree':>10} {'Cosine':>10}"
    )
    flush_print("  " + "-" * 48)
    for p in sorted(all_points, key=lambda x: x["disagreement_rate"]):
        sweep_label = "A" if p["neg_penalty_fixed"] == 0.3 else "B"
        flush_print(
            f"  {sweep_label:>6} {p['neg_penalty_fixed']:>8.1f} "
            f"{p['neg_penalty_sweep']:>8.1f} "
            f"{p['disagreement_rate']:>10.1%} "
            f"{p['cosine_similarity']:>10.4f}"
        )

    # Save results
    results = {
        "config": {
            "n_content": N_CONTENT,
            "n_pairs": N_PAIRS,
            "seed": SEED,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "sweep_a": {
            "fixed_penalty": 0.3,
            "description": "Platform-fixed sweep",
            "points": sweep_a,
        },
        "sweep_b": {
            "fixed_penalty": 1.0,
            "description": "User-fixed sweep",
            "points": sweep_b,
        },
        "curve_fitting": fit_result,
        "go_no_go": go_no_go,
        "known_points": known_points,
        "margin_analysis": {
            "margin_stats": margin_stats,
            "augmented_fits": aug_result,
        },
    }

    output_path = "results/disagreement_bound_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.bool_, np.integer)):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
