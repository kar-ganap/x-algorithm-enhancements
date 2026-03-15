#!/usr/bin/env python3
"""Compute per-dw frontier data for Exp 4 visualization. Run once, save to JSON.

Usage:
    uv run python scripts/compute_exp4_frontiers.py
"""
import importlib.util
import json
import os
import sys
import time
import types

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


enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg
reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [
    os.path.join(project_root, "enhancements/reward_modeling")
]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

load_module_direct(
    "enhancements.reward_modeling.weights",
    os.path.join(project_root, "enhancements/reward_modeling/weights.py"),
)
load_module_direct(
    "enhancements.reward_modeling.pluralistic",
    os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py"),
)
load_module_direct(
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
generate_stakeholder_preferences = partial_obs_mod.generate_stakeholder_preferences
build_base_action_probs = partial_obs_mod.build_base_action_probs
DIVERSITY_WEIGHTS = partial_obs_mod.DIVERSITY_WEIGHTS
STAKEHOLDER_TYPE_MAP = partial_obs_mod.STAKEHOLDER_TYPE_MAP

analyze_mod = load_module_direct(
    "analyze_stakeholder_utilities",
    os.path.join(project_root, "scripts/analyze_stakeholder_utilities.py"),
)
generate_synthetic_data = analyze_mod.generate_synthetic_data

BASE_SEED = 42
N_SEEDS = 20
N_PAIRS_VALUES = [25, 50, 100, 200, 500, 2000]
OUTPUT_PATH = os.path.join(project_root, "results/exp4_frontiers.json")


def main() -> None:
    t0 = time.time()
    seeds = [BASE_SEED + i * 100 for i in range(N_SEEDS)]

    print(f"Computing frontiers: {N_SEEDS} seeds x "
          f"{2 + len(N_PAIRS_VALUES)} scorers...", flush=True)

    all_data: dict[str, list[list[float]]] = {
        "full": [], "LOSO": [],
    }
    for n in N_PAIRS_VALUES:
        all_data[f"n={n}"] = []

    for i, seed in enumerate(seeds):
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  seed {i + 1}/{N_SEEDS}...", flush=True)

        content_probs, _ = generate_content_pool(n_content=500, seed=seed)
        data = generate_synthetic_data(
            num_users=600, num_content=100, num_topics=6, seed=seed,
        )
        base_probs = build_base_action_probs(data)
        archetypes = data["user_archetypes"]
        topics = data["content_topics"]

        # Full frontier
        full_frontier = compute_full_frontier(base_probs, archetypes, topics)
        all_data["full"].append(
            [p["society_utility"] for p in full_frontier]
        )

        # LOSO: user scorer
        probs_pref, probs_rej = generate_stakeholder_preferences(
            content_probs, "user", 2000,
            seed=seed + hash("user") % 10000,
        )
        model = train_with_loss(
            LossConfig(
                loss_type=LossType.BRADLEY_TERRY,
                stakeholder=StakeholderType.USER,
                num_epochs=50, learning_rate=0.01,
            ),
            probs_pref, probs_rej, verbose=False,
        )
        frontier = compute_learned_frontier(
            np.array(model.weights), base_probs, DIVERSITY_WEIGHTS,
            archetypes, topics,
        )
        all_data["LOSO"].append(
            [p["society_utility"] for p in frontier]
        )

        # Society scorer at each n
        for n in N_PAIRS_VALUES:
            probs_pref, probs_rej = generate_stakeholder_preferences(
                content_probs, "society", n,
                seed=seed + hash("society") % 10000,
            )
            model = train_with_loss(
                LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.SOCIETY,
                    num_epochs=50, learning_rate=0.01,
                ),
                probs_pref, probs_rej, verbose=False,
            )
            frontier = compute_learned_frontier(
                np.array(model.weights), base_probs, DIVERSITY_WEIGHTS,
                archetypes, topics,
            )
            all_data[f"n={n}"].append(
                [p["society_utility"] for p in frontier]
            )

    output = {
        "diversity_weights": DIVERSITY_WEIGHTS,
        "n_seeds": N_SEEDS,
        "seeds": seeds,
        "frontiers": all_data,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nSaved: {OUTPUT_PATH} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
