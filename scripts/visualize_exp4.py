#!/usr/bin/env python3
"""Visualize Exp 4: Partial observation sampling.

Reads pre-computed data (no training). Instant.

Requires:
    results/exp4_frontiers.json   (from compute_exp4_frontiers.py)
    results/partial_observation.json  (from analyze_partial_observation.py --exp 4)

Usage:
    uv run python scripts/visualize_exp4.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

N_BOOTSTRAP = 2000


def bootstrap_ci(
    data: np.ndarray, n_boot: int = N_BOOTSTRAP, ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap mean and CI. data: (n_seeds, ...). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(42)
    n = data.shape[0]
    boot_means = np.array([
        np.mean(data[rng.choice(n, size=n, replace=True)], axis=0)
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha, axis=0)
    hi = np.percentile(boot_means, 100 * (1 - alpha), axis=0)
    return np.mean(data, axis=0), lo, hi


def main() -> None:
    # --- Load pre-computed frontier data ---
    frontiers_path = os.path.join(project_root, "results/exp4_frontiers.json")
    if not os.path.exists(frontiers_path):
        print(f"Missing {frontiers_path}")
        print("Run: uv run python scripts/compute_exp4_frontiers.py")
        return

    with open(frontiers_path) as f:
        frontier_data = json.load(f)

    dw = np.array(frontier_data["diversity_weights"])
    n_seeds_frontier = frontier_data["n_seeds"]
    frontiers = frontier_data["frontiers"]

    # --- Load regret data ---
    results_path = os.path.join(
        project_root, "results/partial_observation.json",
    )
    with open(results_path) as f:
        results = json.load(f)
    exp4 = results.get("exp4_partial_sampling", {})
    n_seeds_regret = exp4.get("config", {}).get("n_seeds", "?")

    # --- Bootstrap CIs for frontiers ---
    keys_to_plot = [
        "LOSO", "n=25", "n=100", "n=200", "n=500", "n=2000", "full",
    ]
    frontier_stats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key in keys_to_plot:
        if key in frontiers:
            arr = np.array(frontiers[key])
            frontier_stats[key] = bootstrap_ci(arr)

    # --- Bootstrap CIs for regret ---
    sweep = exp4.get("sweep", [])
    regret_ns: list[int] = []
    regret_means: list[float] = []
    regret_los: list[float] = []
    regret_his: list[float] = []
    for entry in sweep:
        per_seed = entry.get("per_seed", [])
        if per_seed:
            vals = np.array([s["avg_regret"] for s in per_seed])
            mean, lo, hi = bootstrap_ci(vals)
            regret_ns.append(entry["n_society_pairs"])
            regret_means.append(float(mean))
            regret_los.append(float(lo))
            regret_his.append(float(hi))

    # --- Create figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = {
        "LOSO": "#999999",
        "n=25": "#fdae6b",
        "n=100": "#fd8d3c",
        "n=200": "#e6550d",
        "n=500": "#a63603",
        "n=2000": "#4a1503",
        "full": "#2171b5",
    }

    # --- Plot 1: Frontier curves with CI bands ---
    for key in keys_to_plot:
        if key not in frontier_stats:
            continue
        mean, lo, hi = frontier_stats[key]
        color = colors.get(key, "gray")
        is_ref = key in ("LOSO", "full")

        ax1.plot(
            dw, mean, label=key, color=color,
            linewidth=2.0 if is_ref else 1.3,
            linestyle="--" if is_ref else "-",
        )
        ax1.fill_between(dw, lo, hi, color=color, alpha=0.12)

    ax1.set_xlabel("Diversity Weight", fontsize=12)
    ax1.set_ylabel("Society Utility", fontsize=12)
    ax1.set_title(
        f"Society Frontier vs Data Budget\n"
        f"(mean + 95% bootstrap CI, {n_seeds_frontier} seeds)",
        fontsize=12,
    )
    ax1.legend(loc="lower right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # --- Plot 2: Regret with bootstrap CIs ---
    if regret_ns:
        yerr_lo = [m - l for m, l in zip(regret_means, regret_los)]
        yerr_hi = [h - m for m, h in zip(regret_means, regret_his)]

        ax2.errorbar(
            regret_ns, regret_means, yerr=[yerr_lo, yerr_hi],
            marker="o", linewidth=2, capsize=4, capthick=1.5,
            color="#d62728", label="Society scorer",
            markersize=6, zorder=5,
        )

        # LOSO baseline band
        if regret_ns[0] == 0:
            ax2.axhspan(regret_los[0], regret_his[0], color="#cccccc", alpha=0.3)
            ax2.axhline(
                y=regret_means[0], color="#999999", linewidth=1.5,
                linestyle="--",
                label=f"LOSO baseline ({regret_means[0]:.3f})",
            )

        ax2.set_xlabel("Society Preference Pairs (N)", fontsize=12)
        ax2.set_ylabel("Avg Regret (lower is better)", fontsize=12)
        ax2.set_title(
            f"Frontier Quality vs Data Budget\n"
            f"(mean + 95% bootstrap CI, {n_seeds_regret} seeds)",
            fontsize=12,
        )
        ax2.set_xscale("symlog", linthresh=10)
        ax2.set_xticks(regret_ns)
        ax2.set_xticklabels([str(n) for n in regret_ns], fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(project_root, "results/exp4_partial_sampling.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
