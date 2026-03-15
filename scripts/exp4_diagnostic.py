#!/usr/bin/env python3
"""Diagnostic: why does n=2000 recovery (0.744) < n=1000 (0.931)?

Investigates per-seed breakdown of the anomaly.
"""
import importlib.util
import os
import sys
import types
from typing import Any

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def load_module_direct(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Package structure
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
    os.path.join(project_root, "enhancements/reward_modeling/stakeholder_utilities.py"),
)

alt_losses = load_module_direct(
    "enhancements.reward_modeling.alternative_losses",
    os.path.join(project_root, "enhancements/reward_modeling/alternative_losses.py"),
)
LossConfig = alt_losses.LossConfig
LossType = alt_losses.LossType
StakeholderType = alt_losses.StakeholderType
train_with_loss = alt_losses.train_with_loss
POSITIVE_INDICES = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alt_losses.NEGATIVE_INDICES

loss_exp_mod = load_module_direct(
    "run_loss_experiments",
    os.path.join(project_root, "scripts/run_loss_experiments.py"),
)
generate_content_pool = loss_exp_mod.generate_content_pool

partial_obs_mod = load_module_direct(
    "analyze_partial_observation",
    os.path.join(project_root, "scripts/analyze_partial_observation.py"),
)
compute_learned_frontier = partial_obs_mod.compute_learned_frontier
compute_full_frontier = partial_obs_mod.compute_full_frontier
_evaluate_proxy_on_seed = partial_obs_mod._evaluate_proxy_on_seed
extract_pareto_front_2d = partial_obs_mod.extract_pareto_front_2d
find_best_hidden_utility = partial_obs_mod.find_best_hidden_utility
compute_recovery_rate = partial_obs_mod.compute_recovery_rate
is_dominated = partial_obs_mod.is_dominated
generate_stakeholder_preferences = partial_obs_mod.generate_stakeholder_preferences
build_base_action_probs = partial_obs_mod.build_base_action_probs
check_convergence = partial_obs_mod.check_convergence
DIVERSITY_WEIGHTS = partial_obs_mod.DIVERSITY_WEIGHTS
UTILITY_DIMS = partial_obs_mod.UTILITY_DIMS
STAKEHOLDER_TYPE_MAP = partial_obs_mod.STAKEHOLDER_TYPE_MAP

analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

NUM_USERS = 600
NUM_CONTENT = 100
NUM_TOPICS = 6
N_PAIRS = 2000
NUM_EPOCHS = 50
LEARNING_RATE = 0.01


def main() -> None:
    seeds = [42, 142, 242, 342, 442]
    hidden_dim = "society_utility"
    observed_dims = ["user_utility", "platform_utility"]
    test_ns = [500, 1000, 2000]

    print("=" * 70)
    print("EXP 4 DIAGNOSTIC: n=2000 vs n=1000 anomaly")
    print("=" * 70)

    for seed_idx, seed in enumerate(seeds):
        content_probs, _ = generate_content_pool(n_content=500, seed=seed)
        data = generate_synthetic_data(
            num_users=NUM_USERS, num_content=NUM_CONTENT,
            num_topics=NUM_TOPICS, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        archetypes = data["user_archetypes"]
        topics = data["content_topics"]

        full_frontier = compute_full_frontier(base_probs, archetypes, topics)
        full_pareto = [
            p for p in full_frontier
            if not is_dominated(p, full_frontier, UTILITY_DIMS)
        ]
        full_best = max(p[hidden_dim] for p in full_pareto)

        loso_frontier = extract_pareto_front_2d(
            full_frontier, observed_dims[0], observed_dims[1],
        )
        _, loso_best = find_best_hidden_utility(loso_frontier, hidden_dim)
        gap = full_best - loso_best

        print(f"\n{'='*70}")
        print(f"Seed {seed} (full_best={full_best:.4f}, loso_best={loso_best:.4f}, gap={gap:.4f})")
        print(f"{'='*70}")

        weights_by_n: dict[int, np.ndarray] = {}

        for n in test_ns:
            probs_pref, probs_rej = generate_stakeholder_preferences(
                content_probs, "society", n,
                seed=seed + hash("society") % 10000,
            )
            bt_config = LossConfig(
                loss_type=LossType.BRADLEY_TERRY,
                stakeholder=STAKEHOLDER_TYPE_MAP["society"],
                num_epochs=NUM_EPOCHS,
                learning_rate=LEARNING_RATE,
            )
            model = train_with_loss(
                bt_config, probs_pref, probs_rej, verbose=False,
            )
            w = np.array(model.weights)
            weights_by_n[n] = w

            converged = check_convergence(model.loss_history)

            # Full evaluation via _evaluate_proxy_on_seed
            eval_result = _evaluate_proxy_on_seed(
                w, base_probs, archetypes, topics,
                full_pareto, loso_frontier, hidden_dim, observed_dims,
            )

            # Also check: best society utility across ALL diversity weights
            # (before 2D Pareto extraction)
            proxy_frontier = compute_learned_frontier(
                w, base_probs, DIVERSITY_WEIGHTS, archetypes, topics,
            )
            best_soc_all = max(p[hidden_dim] for p in proxy_frontier)
            best_soc_all_dw = max(proxy_frontier, key=lambda p: p[hidden_dim])["diversity_weight"]

            # After 2D Pareto extraction
            proxy_loso = extract_pareto_front_2d(
                proxy_frontier, observed_dims[0], observed_dims[1],
            )
            _, best_soc_pareto = find_best_hidden_utility(proxy_loso, hidden_dim)

            # Weight vector diagnostics
            pos_mean = float(np.mean(w[POSITIVE_INDICES]))
            neg_mean = float(np.mean(w[NEGATIVE_INDICES]))
            alpha_ratio = -neg_mean / pos_mean if abs(pos_mean) > 1e-10 else float("inf")

            print(f"\n  n={n}: converged={converged}")
            print(f"    recovery_rate = {eval_result['recovery_rate']:.4f}")
            print(f"    best_society(all 21 dw) = {best_soc_all:.4f} @ dw={best_soc_all_dw}")
            print(f"    best_society(2D pareto) = {best_soc_pareto:.4f}")
            print(f"    |2D pareto| = {len(proxy_loso)} / {len(proxy_frontier)} points")
            print(f"    w: norm={np.linalg.norm(w):.4f}, α_ratio={alpha_ratio:.3f}")
            print(f"    w_pos_mean={pos_mean:.4f}, w_neg_mean={neg_mean:.4f}")

            # Show all 21 points on the frontier (society utility)
            dw_soc = [(p["diversity_weight"], p[hidden_dim]) for p in proxy_frontier]
            pareto_dws = {p["diversity_weight"] for p in proxy_loso}
            print(f"    Frontier (dw → society):")
            for dw, soc in dw_soc:
                marker = " *" if dw in pareto_dws else ""
                print(f"      dw={dw:.2f} → soc={soc:.4f}{marker}")

        # Cosine similarity between weight vectors
        print(f"\n  Weight cosines:")
        for i, n1 in enumerate(test_ns):
            for n2 in test_ns[i + 1:]:
                w1, w2 = weights_by_n[n1], weights_by_n[n2]
                cos = float(
                    np.dot(w1, w2)
                    / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-12)
                )
                print(f"    cos(w_{n1}, w_{n2}) = {cos:.4f}")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
