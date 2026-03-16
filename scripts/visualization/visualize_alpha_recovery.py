#!/usr/bin/env python3
"""Visualize Fig 3: α-recovery (Spearman = 1.0).

Reads pre-computed data from results/alpha_recovery.json. Instant.

Usage:
    uv run python scripts/visualization/visualize_alpha_recovery.py
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
    results_path = os.path.join(project_root, "results/alpha_recovery.json")
    with open(results_path) as f:
        data = json.load(f)

    sweep = data["sweep_results"]
    validation = data["stakeholder_validation"]

    alpha_true = [r["alpha_true"] for r in sweep]
    alpha_recovered = [r["alpha_ratio"] for r in sweep]

    # Affine fit
    coeffs = np.polyfit(alpha_true, alpha_recovered, 1)
    fit_x = np.linspace(0, max(alpha_true) * 1.05, 100)
    fit_y = np.polyval(coeffs, fit_x)
    ss_res = sum((ar - np.polyval(coeffs, at)) ** 2 for at, ar in zip(alpha_true, alpha_recovered))
    ss_tot = sum((ar - np.mean(alpha_recovered)) ** 2 for ar in alpha_recovered)
    r2 = 1 - ss_res / ss_tot

    # Stakeholder points
    stakeholders = {
        "Platform": (validation["platform"]["alpha_true"], validation["platform"]["alpha_ratio"]),
        "User": (validation["user"]["alpha_true"], validation["user"]["alpha_ratio"]),
        "Society": (validation["society"]["alpha_true"], validation["society"]["alpha_ratio"]),
    }
    markers = {"Platform": "^", "User": "s", "Society": "D"}

    fig, ax = plt.subplots(figsize=(7, 6))

    # Identity line
    ax.plot(fit_x, fit_x, ":", color="#cccccc", linewidth=1.5, label="Identity")
    # Affine fit
    ax.plot(fit_x, fit_y, "--", color="#999999", linewidth=2,
            label=f"Affine fit ($R^2$ = {r2:.3f})")
    # Sweep points
    ax.scatter(alpha_true, alpha_recovered, s=80, color="#d62728", zorder=5,
               label=r"Recovered $\alpha$")

    # Stakeholder markers
    for name, (at, ar) in stakeholders.items():
        ax.scatter([at], [ar], s=120, marker=markers[name], color="#2171b5",
                   zorder=6, edgecolors="white", linewidth=0.5)

    # Labels in empty right region with curved arrows
    label_x = 6.5
    label_configs = [
        ("Society", r"($\alpha$=4.0)", stakeholders["Society"], label_x, 3.5),
        ("User", r"($\alpha$=1.0)", stakeholders["User"], label_x, 2.5),
        ("Platform", r"($\alpha$=0.3)", stakeholders["Platform"], label_x, 1.5),
    ]
    for name, alpha_str, (at, ar), lx, ly in label_configs:
        ax.annotate(
            f"{name}\n{alpha_str}", xy=(at, ar), xytext=(lx, ly),
            fontsize=10, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d4e6f1", edgecolor="#2171b5", alpha=0.9),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2",
                            color="#2171b5", lw=1.5),
        )

    ax.set_xlabel(r"True $\alpha$", fontsize=13)
    ax.set_ylabel(r"Recovered $\hat{\alpha}$", fontsize=13)
    ax.set_title(r"$\alpha$-Recovery: Spearman $\rho$ = 1.0", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)

    # Gridlines matching other figures
    ax.grid(True, which="major", alpha=MAJOR_GRID_ALPHA)
    # Manual minor gridlines at integer positions
    for xv in [1, 2, 3, 4, 5, 6, 7, 8]:
        ax.axvline(xv, color=MINOR_GRID_COLOR, linewidth=MINOR_GRID_LW,
                   linestyle=MINOR_GRID_LS, zorder=0)
    for yv in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        ax.axhline(yv, color=MINOR_GRID_COLOR, linewidth=MINOR_GRID_LW,
                   linestyle=MINOR_GRID_LS, zorder=0)

    ax.set_xlim(-0.3, max(alpha_true) + 0.5)
    ax.set_ylim(-0.3, max(alpha_recovered) + 1)

    plt.tight_layout()

    pdf_path = os.path.join(project_root, "docs/f2/paper/figures/fig3_alpha_recovery.pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close()


if __name__ == "__main__":
    main()
