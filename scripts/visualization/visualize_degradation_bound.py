#!/usr/bin/env python3
"""Visualize Fig 5: Degradation bound (cos vs regret scatter).

Reads pre-computed data from results/partial_observation.json. Instant.

Usage:
    uv run python scripts/visualization/visualize_degradation_bound.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Gridline style matching other figures
MINOR_GRID_COLOR = "#888888"
MINOR_GRID_LW = 0.7
MINOR_GRID_LS = "--"
MAJOR_GRID_ALPHA = 0.3


def main() -> None:
    results_path = os.path.join(project_root, "results/partial_observation.json")
    with open(results_path) as f:
        data = json.load(f)

    exp5 = data["exp5_degradation_bound"]
    sweep = exp5["sweep"]

    cos_vals = [s["cos_sim_nearest_mean"] for s in sweep]
    regret_vals = [s["avg_regret_mean"] for s in sweep]
    alpha_vals = [s["alpha"] for s in sweep]

    # Stakeholder reference points (from LOSO Exp 1 results)
    stakeholder_points = {
        "Society": {"alpha": 4.0, "cos": 0.885, "regret": 0.413},
        "Platform": {"alpha": 0.3, "cos": 0.993, "regret": 0.211},
        "User": {"alpha": 1.0, "cos": 0.995, "regret": 0.091},
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter with colormap
    sc = ax.scatter(
        cos_vals, regret_vals, c=alpha_vals, cmap="RdYlBu_r",
        s=100, zorder=5, edgecolors="white", linewidth=0.5,
        vmin=0, vmax=10,
    )
    cbar = plt.colorbar(sc, ax=ax, label=r"$\alpha_{\mathrm{hidden}}$", pad=0.02)
    cbar.ax.tick_params(labelsize=10)

    # Stakeholder labels — Platform moved northwest to avoid data cluster
    label_configs = {
        "Society": {"xytext": (-60, 10), "ha": "right"},
        "Platform": {"xytext": (-70, 30), "ha": "right"},   # northwest
        "User": {"xytext": (-50, -30), "ha": "right"},
    }
    for name, info in stakeholder_points.items():
        cfg = label_configs[name]
        ax.annotate(
            name, xy=(info["cos"], info["regret"]),
            xytext=cfg["xytext"], textcoords="offset points",
            fontsize=11, fontweight="bold", color="#2171b5",
            ha=cfg["ha"],
            arrowprops=dict(arrowstyle="->", color="#2171b5", lw=1.2),
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"$\cos(\mathbf{w}_{\mathrm{hidden}}, \mathbf{w}_{\mathrm{nearest}})$", fontsize=13)
    ax.set_ylabel("Avg Regret", fontsize=13)
    ax.set_title("Degradation vs Correlation (Spearman = $-$0.96)", fontsize=13)

    # Single consistent grid style (dashed, matching other figures)
    ax.grid(True, which="major", linestyle="--", linewidth=MINOR_GRID_LW,
            color=MINOR_GRID_COLOR, alpha=0.6)

    plt.tight_layout()

    pdf_path = os.path.join(project_root, "docs/f2/paper/figures/fig5_degradation_bound.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    main()
