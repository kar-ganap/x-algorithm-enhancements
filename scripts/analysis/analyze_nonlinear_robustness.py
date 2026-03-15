#!/usr/bin/env python3
"""Nonlinear robustness audit for F4 reward modeling.

Tests whether F4's key results hold under nonlinear utility families:
- Concave (prospect theory): U = pos^γ - α·neg^γ
- Threshold (ignore low negativity): U = pos - α·max(neg - τ, 0)

Experiments:
  A — Labels vs Loss: do 4 loss functions still converge to same weights?
  B — Parameter Recovery: is α recoverable under nonlinear utility?
  C — Proxy Recovery: which LOSO proxy methods survive nonlinearity?
  D — Stress × Nonlinearity: do stress thresholds tighten under nonlinear utility?

Usage:
    uv run python scripts/analyze_nonlinear_robustness.py
    uv run python scripts/analyze_nonlinear_robustness.py --exp A
    uv run python scripts/analyze_nonlinear_robustness.py --exp D
"""

import importlib.util
import json
import os
import sys
import time
import types
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Module loading (same pattern as analyze_partial_observation.py)
# ---------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


def load_module_direct(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Create package structure
enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg

reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [
    os.path.join(project_root, "enhancements/reward_modeling")
]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

weights_mod = load_module_direct(
    "enhancements.reward_modeling.weights",
    os.path.join(project_root, "enhancements/reward_modeling/weights.py"),
)
NUM_ACTIONS = weights_mod.NUM_ACTIONS

load_module_direct(
    "enhancements.reward_modeling.pluralistic",
    os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py"),
)

stakeholder_mod = load_module_direct(
    "enhancements.reward_modeling.stakeholder_utilities",
    os.path.join(
        project_root, "enhancements/reward_modeling/stakeholder_utilities.py"
    ),
)
compute_user_utility = stakeholder_mod.compute_user_utility
compute_platform_utility = stakeholder_mod.compute_platform_utility
compute_society_utility = stakeholder_mod.compute_society_utility

alt_losses = load_module_direct(
    "enhancements.reward_modeling.alternative_losses",
    os.path.join(
        project_root, "enhancements/reward_modeling/alternative_losses.py"
    ),
)
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss
POSITIVE_INDICES = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alt_losses.NEGATIVE_INDICES
ACTION_INDICES = alt_losses.ACTION_INDICES

loss_exp_mod = load_module_direct(
    "run_loss_experiments",
    os.path.join(project_root, "scripts/experiments/run_loss_experiments.py"),
)
generate_content_pool = loss_exp_mod.generate_content_pool

# For Exp C: reuse partial observation infrastructure
partial_obs_mod = load_module_direct(
    "analyze_partial_observation",
    os.path.join(project_root, "scripts/analysis/analyze_partial_observation.py"),
)
compute_learned_frontier = partial_obs_mod.compute_learned_frontier
compute_oracle_linear_proxy = partial_obs_mod.compute_oracle_linear_proxy
synthesize_weights_interpolation = partial_obs_mod.synthesize_weights_interpolation
find_best_hidden_utility = partial_obs_mod.find_best_hidden_utility
compute_recovery_rate = partial_obs_mod.compute_recovery_rate
recover_alpha_from_weights = partial_obs_mod.recover_alpha_from_weights
build_base_action_probs = partial_obs_mod.build_base_action_probs
extract_pareto_front_2d = partial_obs_mod.extract_pareto_front_2d
is_dominated = partial_obs_mod.is_dominated
compute_full_frontier = partial_obs_mod.compute_full_frontier
_evaluate_proxy_on_seed = partial_obs_mod._evaluate_proxy_on_seed
DIVERSITY_WEIGHTS = partial_obs_mod.DIVERSITY_WEIGHTS
UTILITY_DIMS = partial_obs_mod.UTILITY_DIMS

analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analysis/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

alpha_stress_mod = load_module_direct(
    "analyze_alpha_stress",
    os.path.join(project_root, "scripts/analysis/analyze_alpha_stress.py"),
)
generate_correlated_content = alpha_stress_mod.generate_correlated_content


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
N_CONTENT = 500
N_TRAINING_PAIRS = 5000
NUM_EPOCHS = 150
LEARNING_RATE = 0.01

# Exp C constants (match analyze_partial_observation.py)
EXP_C_SEEDS = 5
EXP_C_BASE_SEED = 42
EXP_C_NUM_USERS = 600
EXP_C_NUM_CONTENT = 500
EXP_C_NUM_TOPICS = 6
EXP_C_N_PAIRS = 2000
EXP_C_NUM_EPOCHS = 50
EXP_C_TOP_K = 10

STAKEHOLDERS = ["user", "platform", "society"]

STAKEHOLDER_TYPE_MAP = {
    "user": StakeholderType.USER,
    "platform": StakeholderType.PLATFORM,
    "society": StakeholderType.SOCIETY,
}

# Affine calibration from results/alpha_recovery.json (Direction 1)
ALPHA_AFFINE_SLOPE = 1.3207
ALPHA_AFFINE_INTERCEPT = -0.0619

# Exp D constants (match analyze_alpha_stress.py)
EXP_D_N_PAIRS = 2000
EXP_D_NUM_EPOCHS = 50
EXP_D_N_SEEDS = 5
EXP_D_ALPHA_VALUES = [
    0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0,
]
EXP_D_LABEL_NOISE_VALUES = [0.0, 0.05, 0.10, 0.20, 0.30]
EXP_D_SAMPLE_SIZE_VALUES = [50, 100, 250, 500, 1000, 2000]
EXP_D_TEMPERATURE_VALUES = [1e6, 2.0, 1.0, 0.5, 0.2]  # 1e6 ≈ hard labels
EXP_D_CORRELATION_VALUES = [0.0, 0.3, 0.6, 0.8]


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Utility families
# ---------------------------------------------------------------------------


def linear_utility(
    pos: np.ndarray, neg: np.ndarray, alpha: float
) -> np.ndarray:
    """U = pos - α·neg."""
    return pos - alpha * neg


def concave_utility(
    pos: np.ndarray, neg: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    """U = pos^γ - α·neg^γ.  Diminishing returns (prospect theory)."""
    return np.power(pos, gamma) - alpha * np.power(neg, gamma)


def threshold_utility(
    pos: np.ndarray, neg: np.ndarray, alpha: float, tau: float
) -> np.ndarray:
    """U = pos - α·max(neg - τ, 0).  Ignores low negativity."""
    return pos - alpha * np.maximum(neg - tau, 0.0)


# Family configurations: for each family × stakeholder, the utility params
UTILITY_FAMILIES: dict[str, dict[str, Any]] = {
    "linear": {
        "fn": linear_utility,
        "stakeholder_params": {
            "user": {"alpha": 1.0},
            "platform": {"alpha": 0.3},
            "society": {"alpha": 4.0},
        },
    },
    "concave": {
        "fn": concave_utility,
        "stakeholder_params": {
            "user": {"alpha": 1.0, "gamma": 0.7},
            "platform": {"alpha": 0.3, "gamma": 0.9},
            "society": {"alpha": 4.0, "gamma": 0.5},
        },
    },
    "threshold": {
        "fn": threshold_utility,
        "stakeholder_params": {
            "user": {"alpha": 1.0, "tau": 0.1},
            "platform": {"alpha": 0.3, "tau": 0.3},
            "society": {"alpha": 4.0, "tau": 0.05},
        },
    },
}

# Exp D family configs: utility fn + nonlinear params (α is swept separately)
EXP_D_FAMILIES: dict[str, dict[str, Any]] = {
    "linear": {"fn": linear_utility, "extra_params": {}},
    "concave": {"fn": concave_utility, "extra_params": {"gamma": 0.7}},
    "threshold": {"fn": threshold_utility, "extra_params": {"tau": 0.1}},
}


# ---------------------------------------------------------------------------
# Preference generation
# ---------------------------------------------------------------------------


def generate_nonlinear_preferences(
    content_probs: np.ndarray,
    utility_fn: Callable[..., Any],
    utility_params: dict[str, Any],
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs using a given utility function.

    Generalizes generate_stakeholder_preferences() from
    analyze_partial_observation.py to support nonlinear utilities.
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    utility = utility_fn(pos_scores, neg_scores, **utility_params)

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        noise = rng.normal(0, 0.05)
        if (utility[c1] - utility[c2] + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]

    return probs_pref, probs_rej


def generate_nonlinear_preferences_stressed(
    content_probs: np.ndarray,
    utility_fn: Callable[..., Any],
    alpha: float,
    extra_params: dict[str, Any],
    n_pairs: int,
    p_flip: float = 0.0,
    beta: float = 1e6,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference pairs with nonlinear utility + stress.

    Combines nonlinear utility computation with BT temperature
    and label noise from the stress test framework.
    """
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    utility = utility_fn(pos_scores, neg_scores, alpha=alpha, **extra_params)

    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)

    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]

        if beta >= 1e6:
            noise = rng.normal(0, 0.05)
            prefer_c1 = (diff + noise) > 0
        else:
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


# ---------------------------------------------------------------------------
# Exp A: Labels vs Loss Robustness
# ---------------------------------------------------------------------------

LOSS_TYPES = [
    LossType.BRADLEY_TERRY,
    LossType.MARGIN_BT,
    LossType.CALIBRATED_BT,
    LossType.CONSTRAINED_BT,
]


def run_exp_a() -> dict[str, Any]:
    """Exp A: Do 4 loss functions still converge to same weights under
    nonlinear utility?"""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT A: LABELS VS LOSS ROBUSTNESS")
    flush_print("=" * 60)

    content_probs, _ = generate_content_pool(N_CONTENT, SEED)
    results: dict[str, Any] = {}

    for family_name, family_config in UTILITY_FAMILIES.items():
        flush_print(f"\n  Family: {family_name}")
        family_results: dict[str, Any] = {}

        for stakeholder in STAKEHOLDERS:
            params = family_config["stakeholder_params"][stakeholder]
            utility_fn = family_config["fn"]

            # Generate preferences with this utility
            probs_pref, probs_rej = generate_nonlinear_preferences(
                content_probs, utility_fn, params,
                n_pairs=N_TRAINING_PAIRS, seed=SEED,
            )

            # Compute engagement targets for calibrated BT
            pos_scores = np.sum(probs_pref[:, POSITIVE_INDICES], axis=1)
            eng_pref = np.clip(pos_scores / 5 + 0.3, 0, 1).astype(np.float32)
            pos_scores_rej = np.sum(probs_rej[:, POSITIVE_INDICES], axis=1)
            eng_rej = np.clip(pos_scores_rej / 5 + 0.3, 0, 1).astype(
                np.float32
            )

            # Train with 4 losses (filter diverged models)
            trained_weights: dict[str, np.ndarray] = {}
            diverged: list[str] = []
            for loss_type in LOSS_TYPES:
                config = LossConfig(
                    loss_type=loss_type,
                    stakeholder=STAKEHOLDER_TYPE_MAP[stakeholder],
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    config, probs_pref, probs_rej,
                    target_engagement_pref=eng_pref,
                    target_engagement_rej=eng_rej,
                    verbose=False,
                )
                w = np.array(model.weights)
                if np.any(np.isnan(w)) or np.linalg.norm(w) < 1e-8:
                    diverged.append(loss_type.value)
                else:
                    trained_weights[loss_type.value] = w

            if diverged:
                flush_print(
                    f"    {stakeholder}: diverged={diverged}"
                )

            # Pairwise cosine sims (converged models only)
            loss_names = list(trained_weights.keys())
            cos_sims: list[float] = []
            for i in range(len(loss_names)):
                for j in range(i + 1, len(loss_names)):
                    w1 = trained_weights[loss_names[i]]
                    w2 = trained_weights[loss_names[j]]
                    cos = float(
                        np.dot(w1, w2)
                        / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-12)
                    )
                    cos_sims.append(cos)

            mean_cos = float(np.mean(cos_sims)) if cos_sims else float("nan")
            min_cos = float(np.min(cos_sims)) if cos_sims else float("nan")

            family_results[stakeholder] = {
                "mean_cosine_sim": mean_cos,
                "min_cosine_sim": min_cos,
                "all_cosine_sims": cos_sims,
                "n_converged": len(loss_names),
                "diverged_losses": diverged,
            }
            flush_print(
                f"    {stakeholder}: mean_cos={mean_cos:.4f}"
                f"  min_cos={min_cos:.4f}"
                f"  ({len(loss_names)}/{len(LOSS_TYPES)} converged)"
            )

        results[family_name] = family_results

    return results


# ---------------------------------------------------------------------------
# Exp B: Parameter Recovery
# ---------------------------------------------------------------------------

# Sweep 1 configs: vary nonlinear param, fix α=1.0
SWEEP1_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "concave": [
        {"alpha": 1.0, "gamma": g} for g in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    ],
    "threshold": [
        {"alpha": 1.0, "tau": t}
        for t in [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    ],
}

# Sweep 2: vary α, fix nonlinear param at stakeholder defaults
SWEEP2_ALPHAS = [0.1, 0.3, 0.5, 1.0, 2.0, 4.0]
SWEEP2_NONLINEAR_PARAMS: dict[str, dict[str, float]] = {
    "concave": {"gamma": 0.7},
    "threshold": {"tau": 0.1},
}

EXP_B_SEEDS = 5


def run_exp_b() -> dict[str, Any]:
    """Exp B: α-recovery under nonlinear utility."""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT B: PARAMETER RECOVERY")
    flush_print("=" * 60)

    content_probs, _ = generate_content_pool(N_CONTENT, SEED)
    results: dict[str, Any] = {}

    for family_name in ["concave", "threshold"]:
        flush_print(f"\n  Family: {family_name}")
        utility_fn = UTILITY_FAMILIES[family_name]["fn"]
        family_results: dict[str, Any] = {}

        # --- Sweep 1: vary nonlinear param, α=1.0 fixed ---
        flush_print("    Sweep 1: varying nonlinear param (α=1.0)")
        sweep1_results: list[dict[str, Any]] = []

        for params in SWEEP1_CONFIGS[family_name]:
            seed_alphas: list[float] = []
            for s in range(EXP_B_SEEDS):
                seed = SEED + s * 100
                probs_pref, probs_rej = generate_nonlinear_preferences(
                    content_probs, utility_fn, params,
                    n_pairs=N_TRAINING_PAIRS, seed=seed,
                )
                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    config, probs_pref, probs_rej, verbose=False,
                )
                alpha_rec = recover_alpha_from_weights(
                    np.array(model.weights), calibrate=True,
                )
                seed_alphas.append(alpha_rec)

            point = {
                "params": params,
                "alpha_recovered_mean": float(np.mean(seed_alphas)),
                "alpha_recovered_std": float(np.std(seed_alphas)),
            }
            sweep1_results.append(point)

            # Identify the nonlinear param for display
            nl_param = {k: v for k, v in params.items() if k != "alpha"}
            flush_print(
                f"      {nl_param}: α_rec={point['alpha_recovered_mean']:.3f}"
                f"±{point['alpha_recovered_std']:.3f}"
            )

        family_results["sweep1"] = sweep1_results

        # Sensitivity: std of α_recovered across nonlinear param values
        alpha_means = [r["alpha_recovered_mean"] for r in sweep1_results]
        sensitivity = float(np.std(alpha_means))
        flush_print(f"    α-sensitivity (std across nl param): {sensitivity:.4f}")
        family_results["sweep1_sensitivity"] = sensitivity

        # --- Sweep 2: vary α, nonlinear param fixed ---
        nl_defaults = SWEEP2_NONLINEAR_PARAMS[family_name]
        flush_print(f"    Sweep 2: varying α (nl param={nl_defaults})")
        sweep2_results: list[dict[str, Any]] = []

        for alpha in SWEEP2_ALPHAS:
            params = {"alpha": alpha, **nl_defaults}
            seed_alphas_s2: list[float] = []
            for s in range(EXP_B_SEEDS):
                seed = SEED + s * 100
                probs_pref, probs_rej = generate_nonlinear_preferences(
                    content_probs, utility_fn, params,
                    n_pairs=N_TRAINING_PAIRS, seed=seed,
                )
                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    num_epochs=NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    config, probs_pref, probs_rej, verbose=False,
                )
                alpha_rec = recover_alpha_from_weights(
                    np.array(model.weights), calibrate=True,
                )
                seed_alphas_s2.append(alpha_rec)

            point = {
                "alpha_true": alpha,
                "alpha_recovered_mean": float(np.mean(seed_alphas_s2)),
                "alpha_recovered_std": float(np.std(seed_alphas_s2)),
            }
            sweep2_results.append(point)
            flush_print(
                f"      α={alpha:.1f}: α_rec="
                f"{point['alpha_recovered_mean']:.3f}"
                f"±{point['alpha_recovered_std']:.3f}"
            )

        family_results["sweep2"] = sweep2_results

        # Spearman correlation
        true_alphas = [r["alpha_true"] for r in sweep2_results]
        rec_alphas = [r["alpha_recovered_mean"] for r in sweep2_results]
        spearman = float(spearmanr(true_alphas, rec_alphas)[0])  # type: ignore[arg-type]
        flush_print(f"    α-sweep Spearman: {spearman:.4f}")
        family_results["sweep2_spearman"] = spearman

        results[family_name] = family_results

    return results


# ---------------------------------------------------------------------------
# Exp C: Proxy Recovery Under Nonlinearity
# ---------------------------------------------------------------------------

LOSO_CONFIGS = [
    {
        "hidden": "society",
        "hidden_dim": "society_utility",
        "observed_dims": ["user_utility", "platform_utility"],
    },
    {
        "hidden": "platform",
        "hidden_dim": "platform_utility",
        "observed_dims": ["user_utility", "society_utility"],
    },
    {
        "hidden": "user",
        "hidden_dim": "user_utility",
        "observed_dims": ["platform_utility", "society_utility"],
    },
]


def run_exp_c() -> dict[str, Any]:
    """Exp C: proxy recovery under nonlinear utility."""
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT C: PROXY RECOVERY UNDER NONLINEARITY")
    flush_print("=" * 60)

    seeds = [EXP_C_BASE_SEED + i * 100 for i in range(EXP_C_SEEDS)]
    results: dict[str, Any] = {}

    for family_name, family_config in UTILITY_FAMILIES.items():
        flush_print(f"\n  Family: {family_name}")
        utility_fn = family_config["fn"]
        family_results: dict[str, Any] = {}

        for loso_config in LOSO_CONFIGS:
            hidden = loso_config["hidden"]
            hidden_dim = loso_config["hidden_dim"]
            observed_dims = loso_config["observed_dims"]
            observed = [s for s in STAKEHOLDERS if s != hidden]

            flush_print(f"    Hidden: {hidden}")
            seed_results: list[dict[str, Any]] = []

            for seed_idx, seed in enumerate(seeds):
                # Generate content pool for training
                content_probs, _ = generate_content_pool(
                    n_content=N_CONTENT, seed=seed,
                )

                # Generate evaluation data
                data = generate_synthetic_data(
                    num_users=EXP_C_NUM_USERS,
                    num_content=EXP_C_NUM_CONTENT,
                    num_topics=EXP_C_NUM_TOPICS,
                    seed=seed,
                )
                base_probs = build_base_action_probs(data)

                # Train BT for observed stakeholders with nonlinear prefs
                learned_weights: dict[str, np.ndarray] = {}
                for s in observed:
                    s_params = family_config["stakeholder_params"][s]
                    probs_pref, probs_rej = generate_nonlinear_preferences(
                        content_probs, utility_fn, s_params,
                        n_pairs=EXP_C_N_PAIRS,
                        seed=seed + hash(s) % 10000,
                    )
                    bt_config = LossConfig(
                        loss_type=LossType.BRADLEY_TERRY,
                        stakeholder=STAKEHOLDER_TYPE_MAP[s],
                        num_epochs=EXP_C_NUM_EPOCHS,
                        learning_rate=LEARNING_RATE,
                    )
                    model = train_with_loss(
                        bt_config, probs_pref, probs_rej, verbose=False,
                    )
                    learned_weights[s] = np.array(model.weights)

                # Full frontier (uses hardcoded utilities, not nonlinear)
                full_frontier = compute_full_frontier(
                    base_probs,
                    data["user_archetypes"],
                    data["content_topics"],
                )
                full_pareto = [
                    p for p in full_frontier
                    if not is_dominated(p, full_frontier, UTILITY_DIMS)
                ]
                loso_frontier = extract_pareto_front_2d(
                    full_frontier,
                    observed_dims[0],
                    observed_dims[1],
                )

                seed_data: dict[str, Any] = {}
                w_hidden: np.ndarray | None = None

                # Oracle linear proxy
                if len(observed) == 2:
                    oracle = compute_oracle_linear_proxy(
                        learned_weights[observed[0]],
                        learned_weights[observed[1]],
                        learned_weights.get(
                            hidden,
                            np.zeros(NUM_ACTIONS),
                        ),
                    )
                    # Use the proxy weights (even without true hidden)
                    # For oracle: train hidden stakeholder too (cheating)
                    hidden_params = family_config["stakeholder_params"][hidden]
                    h_pref, h_rej = generate_nonlinear_preferences(
                        content_probs, utility_fn, hidden_params,
                        n_pairs=EXP_C_N_PAIRS,
                        seed=seed + hash(hidden) % 10000,
                    )
                    h_config = LossConfig(
                        loss_type=LossType.BRADLEY_TERRY,
                        stakeholder=STAKEHOLDER_TYPE_MAP[hidden],
                        num_epochs=EXP_C_NUM_EPOCHS,
                        learning_rate=LEARNING_RATE,
                    )
                    h_model = train_with_loss(
                        h_config, h_pref, h_rej, verbose=False,
                    )
                    w_hidden = np.array(h_model.weights)

                    oracle = compute_oracle_linear_proxy(
                        learned_weights[observed[0]],
                        learned_weights[observed[1]],
                        w_hidden,
                    )
                    seed_data["oracle_linear"] = _evaluate_proxy_on_seed(
                        oracle["proxy_weights"],
                        base_probs,
                        data["user_archetypes"],
                        data["content_topics"],
                        full_pareto,
                        loso_frontier,
                        hidden_dim,
                        observed_dims,
                    )
                    # Weight-space cosine: oracle proxy vs hidden
                    oracle_w = np.array(oracle["proxy_weights"])
                    seed_data["oracle_cosine"] = float(
                        np.dot(oracle_w, w_hidden)
                        / (np.linalg.norm(oracle_w) * np.linalg.norm(w_hidden)
                           + 1e-12)
                    )

                # α-interpolation proxy
                alpha_obs: dict[str, float] = {}
                for s in observed:
                    alpha_obs[s] = recover_alpha_from_weights(
                        learned_weights[s], calibrate=True,
                    )

                alpha_blind = 2.0 * max(alpha_obs.values())

                s1, s2 = observed
                interp_blind = synthesize_weights_interpolation(
                    learned_weights[s1], alpha_obs[s1],
                    learned_weights[s2], alpha_obs[s2],
                    alpha_blind,
                )
                seed_data["interp_blind_alpha"] = _evaluate_proxy_on_seed(
                    interp_blind, base_probs,
                    data["user_archetypes"], data["content_topics"],
                    full_pareto, loso_frontier, hidden_dim, observed_dims,
                )
                # Weight-space cosine: interp proxy vs hidden
                if w_hidden is not None:
                    seed_data["interp_cosine"] = float(
                        np.dot(interp_blind, w_hidden)
                        / (np.linalg.norm(interp_blind)
                           * np.linalg.norm(w_hidden) + 1e-12)
                    )

                # Diversity knob
                full_best_hidden = max(
                    p[hidden_dim] for p in full_pareto
                )
                _, loso_best_hidden = find_best_hidden_utility(
                    loso_frontier, hidden_dim,
                )
                dw_results: dict[str, Any] = {}
                for fixed_dw in [0.3, 0.5, 0.7]:
                    closest = min(
                        full_frontier,
                        key=lambda p: abs(
                            p["diversity_weight"] - fixed_dw
                        ),
                    )
                    dw_results[f"dw_{fixed_dw}"] = {
                        "hidden_utility": closest[hidden_dim],
                        "recovery_rate": compute_recovery_rate(
                            closest[hidden_dim],
                            loso_best_hidden,
                            full_best_hidden,
                        ),
                    }
                seed_data["diversity_knob"] = dw_results

                seed_results.append(seed_data)
                flush_print(
                    f"      seed {seed_idx + 1}/{len(seeds)} done"
                )

            # Aggregate
            agg: dict[str, Any] = {}
            for method in ["oracle_linear", "interp_blind_alpha"]:
                if method in seed_results[0]:
                    rec_vals = [
                        s[method]["recovery_rate"] for s in seed_results
                    ]
                    agg[method] = {
                        "recovery_mean": float(np.mean(rec_vals)),
                        "recovery_std": float(np.std(rec_vals)),
                    }
                    flush_print(
                        f"      {method}: recovery="
                        f"{agg[method]['recovery_mean']:.3f}"
                        f"±{agg[method]['recovery_std']:.3f}"
                    )

            for cos_key in ["oracle_cosine", "interp_cosine"]:
                if cos_key in seed_results[0]:
                    cos_vals = [s[cos_key] for s in seed_results]
                    agg[cos_key] = {
                        "mean": float(np.mean(cos_vals)),
                        "std": float(np.std(cos_vals)),
                    }
                    flush_print(
                        f"      {cos_key}: "
                        f"{agg[cos_key]['mean']:.4f}"
                        f"±{agg[cos_key]['std']:.4f}"
                    )

            for fixed_dw in [0.3, 0.5, 0.7]:
                key = f"dw_{fixed_dw}"
                rec_vals = [
                    s["diversity_knob"][key]["recovery_rate"]
                    for s in seed_results
                ]
                agg[key] = {
                    "recovery_mean": float(np.mean(rec_vals)),
                    "recovery_std": float(np.std(rec_vals)),
                }
                flush_print(
                    f"      {key}: recovery="
                    f"{agg[key]['recovery_mean']:.3f}"
                    f"±{agg[key]['recovery_std']:.3f}"
                )

            family_results[f"hide_{hidden}"] = {
                "aggregated": agg,
                "per_seed": seed_results,
            }

        results[family_name] = family_results

    return results


# ---------------------------------------------------------------------------
# Exp D: Stress × Nonlinearity
# ---------------------------------------------------------------------------


def _compute_alpha_ratio(weights: np.ndarray) -> float:
    """Recover α as -mean(w_neg) / mean(w_pos) (raw, no calibration)."""
    pos_mean = float(np.mean(weights[POSITIVE_INDICES]))
    neg_mean = float(np.mean(weights[NEGATIVE_INDICES]))
    if abs(pos_mean) < 1e-10:
        return float("inf")
    return -neg_mean / pos_mean


def _run_exp_d_dimension(
    dimension_name: str,
    values: list[Any],
    content_pools: dict[Any, np.ndarray],
    make_kwargs: Callable[[Any], dict[str, Any]],
    utility_fn: Callable[..., Any],
    extra_params: dict[str, Any],
) -> dict[str, Any]:
    """Run one stress dimension for one family in Exp D."""
    flush_print(f"    Dimension: {dimension_name}")

    results: list[dict[str, Any]] = []

    for val in values:
        content = content_pools[val]
        kwargs = make_kwargs(val)

        seed_spearmans: list[float] = []
        seed_pearsons: list[float] = []

        for s in range(EXP_D_N_SEEDS):
            seed = SEED + s * 100

            alphas_true = np.array(EXP_D_ALPHA_VALUES)
            alphas_recovered = np.zeros(len(EXP_D_ALPHA_VALUES))

            for i, alpha in enumerate(EXP_D_ALPHA_VALUES):
                probs_pref, probs_rej = (
                    generate_nonlinear_preferences_stressed(
                        content, utility_fn, alpha, extra_params,
                        n_pairs=kwargs["n_pairs"],
                        p_flip=kwargs["p_flip"],
                        beta=kwargs["beta"],
                        seed=seed + i * 1000,
                    )
                )
                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    num_epochs=EXP_D_NUM_EPOCHS,
                    learning_rate=LEARNING_RATE,
                )
                model = train_with_loss(
                    config, probs_pref, probs_rej, verbose=False,
                )
                alphas_recovered[i] = _compute_alpha_ratio(
                    np.array(model.weights)
                )

            sp = float(
                spearmanr(alphas_true, alphas_recovered).statistic  # type: ignore[union-attr]
            )
            if np.isnan(sp):
                sp = 0.0
            seed_spearmans.append(sp)

            finite_mask = np.isfinite(alphas_recovered)
            if np.sum(finite_mask) >= 3:
                pr = float(
                    pearsonr(
                        alphas_true[finite_mask],
                        alphas_recovered[finite_mask],
                    )[0]
                )
                if np.isnan(pr):
                    pr = 0.0
            else:
                pr = 0.0
            seed_pearsons.append(pr)

        agg = {
            "value": val,
            "spearman_mean": float(np.mean(seed_spearmans)),
            "spearman_std": float(np.std(seed_spearmans)),
            "pearson_mean": float(np.mean(seed_pearsons)),
            "pearson_std": float(np.std(seed_pearsons)),
        }
        results.append(agg)

        flush_print(
            f"      {dimension_name}={val!s:>6s}  "
            f"Spearman={agg['spearman_mean']:.4f}"
            f"±{agg['spearman_std']:.4f}"
        )

    # Breaking point (first value where Spearman < 0.95)
    breaking_point = None
    for r in results:
        if r["spearman_mean"] < 0.95:
            breaking_point = r["value"]
            break

    flush_print(
        f"      Breaking point: "
        f"{breaking_point if breaking_point is not None else 'none'}"
    )

    return {
        "results": results,
        "breaking_point": breaking_point,
    }


def _checkpoint_exp_d(families_results: dict[str, Any]) -> None:
    """Save partial Exp D results after each family completes."""
    checkpoint = {
        "config": {
            "n_content": N_CONTENT,
            "n_pairs": EXP_D_N_PAIRS,
            "num_epochs": EXP_D_NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "n_seeds": EXP_D_N_SEEDS,
            "alpha_values": EXP_D_ALPHA_VALUES,
        },
        "families": families_results,
        "breaking_points": {
            fname: {
                dim: dim_res["breaking_point"]
                for dim, dim_res in fam_res.items()
            }
            for fname, fam_res in families_results.items()
        },
        "_partial": True,
    }
    cached = load_cached_results()
    cached["exp_d_stress_nonlinearity"] = checkpoint
    save_results(cached)
    flush_print(
        f"  [checkpoint] Saved {len(families_results)} families"
    )


def run_exp_d() -> dict[str, Any]:
    """Exp D: Stress × Nonlinearity cross-experiment.

    Runs the same 4D stress sweep from analyze_alpha_stress.py
    for Linear, Concave (γ=0.7), and Threshold (τ=0.1) utility families.
    """
    flush_print("\n" + "=" * 60)
    flush_print("EXPERIMENT D: STRESS × NONLINEARITY")
    flush_print("=" * 60)
    flush_print(
        f"  Families: {list(EXP_D_FAMILIES.keys())}"
    )
    flush_print(
        f"  α values: {len(EXP_D_ALPHA_VALUES)}, "
        f"seeds: {EXP_D_N_SEEDS}, "
        f"pairs: {EXP_D_N_PAIRS}, "
        f"epochs: {EXP_D_NUM_EPOCHS}"
    )

    baseline_content, _ = generate_content_pool(N_CONTENT, SEED)

    # Pre-generate correlated content pools
    flush_print("  Generating correlated content pools...")
    corr_content_pools: dict[float, np.ndarray] = {}
    for rho in EXP_D_CORRELATION_VALUES:
        if rho == 0.0:
            corr_content_pools[rho] = baseline_content
        else:
            corr_content_pools[rho] = generate_correlated_content(
                N_CONTENT, rho, seed=SEED,
            )

    families_results: dict[str, Any] = {}

    for family_name, family_config in EXP_D_FAMILIES.items():
        flush_print(f"\n  Family: {family_name}")
        utility_fn = family_config["fn"]
        extra_params = family_config["extra_params"]

        family_results: dict[str, Any] = {}

        # Dimension 1: Label noise
        family_results["label_noise"] = _run_exp_d_dimension(
            "label_noise",
            EXP_D_LABEL_NOISE_VALUES,
            {v: baseline_content for v in EXP_D_LABEL_NOISE_VALUES},
            lambda v: {
                "n_pairs": EXP_D_N_PAIRS, "p_flip": v, "beta": 1e6,
            },
            utility_fn,
            extra_params,
        )

        # Dimension 2: Sample size
        family_results["sample_size"] = _run_exp_d_dimension(
            "sample_size",
            EXP_D_SAMPLE_SIZE_VALUES,
            {v: baseline_content for v in EXP_D_SAMPLE_SIZE_VALUES},
            lambda v: {
                "n_pairs": v, "p_flip": 0.0, "beta": 1e6,
            },
            utility_fn,
            extra_params,
        )

        # Dimension 3: Temperature
        family_results["temperature"] = _run_exp_d_dimension(
            "temperature",
            EXP_D_TEMPERATURE_VALUES,
            {v: baseline_content for v in EXP_D_TEMPERATURE_VALUES},
            lambda v: {
                "n_pairs": EXP_D_N_PAIRS, "p_flip": 0.0, "beta": v,
            },
            utility_fn,
            extra_params,
        )

        # Dimension 4: Content correlation
        family_results["correlation"] = _run_exp_d_dimension(
            "correlation",
            EXP_D_CORRELATION_VALUES,
            corr_content_pools,
            lambda _v: {
                "n_pairs": EXP_D_N_PAIRS, "p_flip": 0.0, "beta": 1e6,
            },
            utility_fn,
            extra_params,
        )

        families_results[family_name] = family_results

        # Checkpoint after each family (~4 hours per family)
        _checkpoint_exp_d(families_results)

    # Extract breaking points summary
    breaking_points: dict[str, dict[str, Any]] = {}
    for family_name, fam_res in families_results.items():
        bp: dict[str, Any] = {}
        for dim_name, dim_res in fam_res.items():
            bp[dim_name] = dim_res["breaking_point"]
        breaking_points[family_name] = bp

    # Print summary table
    flush_print("\n" + "-" * 60)
    flush_print("BREAKING POINTS (Spearman < 0.95)")
    flush_print("-" * 60)
    dims = ["label_noise", "sample_size", "temperature", "correlation"]
    flush_print(
        f"  {'Dimension':<18} "
        + "  ".join(f"{f:<12}" for f in EXP_D_FAMILIES)
    )
    flush_print("  " + "-" * 54)
    for dim in dims:
        vals = []
        for f in EXP_D_FAMILIES:
            bp = breaking_points[f][dim]
            vals.append(str(bp) if bp is not None else "none")
        flush_print(
            f"  {dim:<18} "
            + "  ".join(f"{v:<12}" for v in vals)
        )

    return {
        "config": {
            "n_content": N_CONTENT,
            "n_pairs": EXP_D_N_PAIRS,
            "num_epochs": EXP_D_NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "n_seeds": EXP_D_N_SEEDS,
            "alpha_values": EXP_D_ALPHA_VALUES,
        },
        "families": families_results,
        "breaking_points": breaking_points,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_PATH = os.path.join(
    project_root, "results/nonlinear_robustness.json"
)


def load_cached_results() -> dict[str, Any]:
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            return json.load(f)
    return {}


def save_results(output: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to: {OUTPUT_PATH}")


def parse_experiments(args: list[str]) -> set[str]:
    """Parse --exp flag. Usage: --exp A, --exp A,B, (no flag) = all."""
    for i, arg in enumerate(args):
        if arg == "--exp" and i + 1 < len(args):
            return {x.strip().upper() for x in args[i + 1].split(",")}
    return {"A", "B", "C", "D"}


def main() -> None:
    t0 = time.time()
    requested = parse_experiments(sys.argv[1:])

    flush_print("=" * 60)
    flush_print("NONLINEAR ROBUSTNESS AUDIT")
    flush_print("=" * 60)
    flush_print("Families: linear, concave, threshold")
    flush_print(f"Experiments: {sorted(requested)}")

    cached = load_cached_results()

    # Exp A
    if "A" in requested:
        exp_a_results = run_exp_a()
    elif "exp_a_labels_vs_loss" in cached:
        exp_a_results = cached["exp_a_labels_vs_loss"]
        flush_print("\n  Exp A: loaded from cache")
    else:
        exp_a_results = {}

    # Exp B
    if "B" in requested:
        exp_b_results = run_exp_b()
    elif "exp_b_parameter_recovery" in cached:
        exp_b_results = cached["exp_b_parameter_recovery"]
        flush_print("  Exp B: loaded from cache")
    else:
        exp_b_results = {}

    # Exp C
    if "C" in requested:
        exp_c_results = run_exp_c()
    elif "exp_c_proxy_recovery" in cached:
        exp_c_results = cached["exp_c_proxy_recovery"]
        flush_print("  Exp C: loaded from cache")
    else:
        exp_c_results = {}

    # Exp D
    if "D" in requested:
        exp_d_results = run_exp_d()
    elif "exp_d_stress_nonlinearity" in cached:
        exp_d_results = cached["exp_d_stress_nonlinearity"]
        flush_print("  Exp D: loaded from cache")
    else:
        exp_d_results = {}

    # Summary
    flush_print("\n" + "=" * 60)
    flush_print("SUMMARY")
    flush_print("=" * 60)

    if exp_a_results:
        flush_print("\nExp A — Labels vs Loss (mean cosine sim across 4 losses):")
        flush_print(
            f"  {'Family':<12} {'User':>10} {'Platform':>10} {'Society':>10}"
        )
        flush_print("  " + "-" * 42)
        for family in UTILITY_FAMILIES:
            if family in exp_a_results:
                fr = exp_a_results[family]
                flush_print(
                    f"  {family:<12} "
                    f"{fr['user']['mean_cosine_sim']:>10.4f} "
                    f"{fr['platform']['mean_cosine_sim']:>10.4f} "
                    f"{fr['society']['mean_cosine_sim']:>10.4f}"
                )

    if exp_b_results:
        flush_print("\nExp B — Parameter Recovery:")
        for family in ["concave", "threshold"]:
            if family in exp_b_results:
                fr = exp_b_results[family]
                flush_print(
                    f"  {family}: sensitivity={fr['sweep1_sensitivity']:.4f}"
                    f"  α-Spearman={fr['sweep2_spearman']:.4f}"
                )

    if exp_c_results:
        flush_print("\nExp C — Proxy Recovery (hiding society, recovery rate):")
        flush_print(
            f"  {'Family':<12} {'Oracle':>8} {'IntBlind':>10} {'DW0.7':>8}"
        )
        flush_print("  " + "-" * 40)
        for family in UTILITY_FAMILIES:
            if family in exp_c_results:
                fr = exp_c_results[family]
                hs = fr.get("hide_society", {}).get("aggregated", {})
                oracle = hs.get("oracle_linear", {}).get(
                    "recovery_mean", 0
                )
                interp = hs.get("interp_blind_alpha", {}).get(
                    "recovery_mean", 0
                )
                dw07 = hs.get("dw_0.7", {}).get("recovery_mean", 0)
                flush_print(
                    f"  {family:<12} "
                    f"{oracle:>8.3f} "
                    f"{interp:>10.3f} "
                    f"{dw07:>8.3f}"
                )

    if exp_d_results:
        flush_print("\nExp D — Stress × Nonlinearity (breaking points):")
        dims = ["label_noise", "sample_size", "temperature", "correlation"]
        bp = exp_d_results.get("breaking_points", {})
        flush_print(
            f"  {'Dimension':<18} "
            + "  ".join(f"{f:<12}" for f in ["linear", "concave", "threshold"])
        )
        flush_print("  " + "-" * 54)
        for dim in dims:
            vals = []
            for f in ["linear", "concave", "threshold"]:
                v = bp.get(f, {}).get(dim)
                vals.append(str(v) if v is not None else "none")
            flush_print(
                f"  {dim:<18} "
                + "  ".join(f"{v:<12}" for v in vals)
            )

    wall_time = time.time() - t0
    flush_print(f"\n  Wall time: {wall_time:.1f}s ({wall_time / 60:.1f}m)")

    # Save results
    output = cached.copy()
    if exp_a_results:
        output["exp_a_labels_vs_loss"] = exp_a_results
    if exp_b_results:
        output["exp_b_parameter_recovery"] = exp_b_results
    if exp_c_results:
        output["exp_c_proxy_recovery"] = exp_c_results
    if exp_d_results:
        output["exp_d_stress_nonlinearity"] = exp_d_results

    save_results(output)


if __name__ == "__main__":
    main()
