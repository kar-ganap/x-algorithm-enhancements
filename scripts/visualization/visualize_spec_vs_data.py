#!/usr/bin/env python3
"""Visualize Direction 2 Angle C: Specification vs Data (Goodhart effect).

Reads pre-computed data from results/utility_sensitivity.json. Instant.

Usage:
    uv run python scripts/visualization/visualize_spec_vs_data.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    results_path = os.path.join(project_root, "results/utility_sensitivity.json")
    with open(results_path) as f:
        data = json.load(f)

    svd = data["specification_vs_data"]

    # --- Strategy 1: Better spec (reduce sigma at fixed N=2000) ---
    s1 = svd["strategy1_spec"]
    sigmas = sorted(s1.keys(), key=float, reverse=True)
    sigma_vals = [float(s) for s in sigmas]
    s1_means = [s1[s]["hausdorff_mean"] for s in sigmas]
    s1_stds = [s1[s]["hausdorff_std"] for s in sigmas]

    # --- Strategy 2: More data (increase N at fixed sigma=0.3) ---
    s2 = svd["strategy2_data"]
    ns = sorted(s2.keys(), key=int)
    n_vals = [int(n) for n in ns]
    s2_means = [s2[n]["hausdorff_mean"] for n in ns]
    s2_stds = [s2[n]["hausdorff_std"] for n in ns]

    # --- Create figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel (a): Better specification
    ax1.errorbar(
        sigma_vals, s1_means, yerr=s1_stds,
        marker="s", linewidth=2, capsize=4, capthick=1.5,
        color="#2171b5", markersize=7, label="Hausdorff distance",
    )
    ax1.set_xlabel(r"Weight noise $\sigma$", fontsize=12)
    ax1.set_ylabel("Hausdorff Distance (lower is better)", fontsize=12)
    ax1.set_title("(a) Better Specification (fixed N = 2,000)", fontsize=12)
    ax1.invert_xaxis()
    ax1.grid(True, alpha=0.3)
    # Match minor gridlines from panel (b)
    for xv in [0.05, 0.15, 0.25, 0.35, 0.45]:
        ax1.axvline(xv, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
    ax1.legend(fontsize=10)

    # Panel (b): More data — log scale x-axis
    ax2.errorbar(
        n_vals, s2_means, yerr=s2_stds,
        marker="o", linewidth=2, capsize=4, capthick=1.5,
        color="#d62728", markersize=7, label="Hausdorff distance",
    )

    # Find and annotate the Goodhart minimum
    min_idx = int(np.argmin(s2_means))
    min_n = n_vals[min_idx]
    min_val = s2_means[min_idx]
    ax2.annotate(
        f"min at N={min_n}\n({min_val:.2f})",
        (min_n, min_val),
        textcoords="offset points", xytext=(25, -20),
        fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

    # Shade the Goodhart region (where more data hurts)
    goodhart_ns = [n for n, m in zip(n_vals, s2_means) if n > min_n]
    goodhart_means = [m for n, m in zip(n_vals, s2_means) if n > min_n]
    if goodhart_ns:
        ax2.axvspan(min_n, max(n_vals) * 1.1, color="#ffcccc", alpha=0.25,
                     label="Goodhart region")

    # Simple log scale with readable tick labels
    ax2.set_xscale("log")
    ax2.set_xlim(15, 3000)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.ticklabel_format(axis="x", style="plain")  # "100" not "10^2"
    # Major gridlines
    ax2.grid(True, which="major", alpha=0.3)
    # Vertical minor gridlines drawn explicitly
    for xv in [20, 30, 40, 50, 60, 70, 80, 90,
                200, 300, 400, 500, 600, 700, 800, 900, 2000]:
        ax2.axvline(xv, color="#888888", linewidth=0.7, linestyle="--", zorder=0)

    ax2.set_xlabel("Training Pairs (N)", fontsize=12)
    ax2.set_ylabel("Hausdorff Distance (lower is better)", fontsize=12)
    ax2.set_title(r"(b) More Data (fixed $\sigma$ = 0.3)", fontsize=12)
    ax2.legend(fontsize=10)

    # Add User/Platform labels in empty regions
    # (positioned carefully to not overlap data)
    ax1.text(
        0.95, 0.95, "User + Platform observed",
        transform=ax1.transAxes, fontsize=11, ha="right", va="top",
        style="italic", color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    ax2.text(
        0.05, 0.82, "User + Platform observed",
        transform=ax2.transAxes, fontsize=11, ha="left", va="top",
        style="italic", color="#666666",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    pdf_path = os.path.join(
        project_root, "docs/f2/paper/figures/fig6_spec_vs_data.pdf",
    )
    png_path = os.path.join(project_root, "results/fig6_spec_vs_data.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
