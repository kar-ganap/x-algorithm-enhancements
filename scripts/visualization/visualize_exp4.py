#!/usr/bin/env python3
"""Visualize Exp 4: Partial observation sampling.

Reads pre-computed data (no training). Instant.

Requires:
    results/exp4_frontiers.json   (from compute_exp4_frontiers.py)
    results/partial_observation.json  (from analyze_partial_observation.py --exp 4)

Usage:
    uv run python scripts/visualization/visualize_exp4.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        print("Run: uv run python scripts/experiments/compute_exp4_frontiers.py")
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

    # --- Plot 1 (a): Frontier curves with CI bands ---
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
        f"(a) Society Frontier vs Data Budget\n"
        f"(mean + 95% bootstrap CI, {n_seeds_frontier} seeds)",
        fontsize=12,
    )
    ax1.legend(loc="lower right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    # Match vertical minor gridlines from panel (b)
    for xv in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ax1.axvline(xv, color="#888888", linewidth=0.7, linestyle="--", zorder=0)
    ax1.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)

    # --- Plot 2 (b): Regret with bootstrap CIs ---
    if regret_ns:
        # Separate N=0 (LOSO baseline) from actual data points
        has_baseline = regret_ns[0] == 0
        if has_baseline:
            baseline_mean = regret_means[0]
            baseline_lo = regret_los[0]
            baseline_hi = regret_his[0]
            plot_ns = regret_ns[1:]
            plot_means = regret_means[1:]
            plot_los = regret_los[1:]
            plot_his = regret_his[1:]
        else:
            baseline_mean = None
            plot_ns = regret_ns
            plot_means = regret_means
            plot_los = regret_los
            plot_his = regret_his

        yerr_lo = [m - l for m, l in zip(plot_means, plot_los)]
        yerr_hi = [h - m for m, h in zip(plot_means, plot_his)]

        ax2.errorbar(
            plot_ns, plot_means, yerr=[yerr_lo, yerr_hi],
            marker="o", linewidth=2, capsize=4, capthick=1.5,
            color="#d62728", label="Society scorer",
            markersize=6, zorder=5,
        )

        # N=0 LOSO baseline as horizontal line (not a data point on log axis)
        if baseline_mean is not None:
            ax2.axhspan(baseline_lo, baseline_hi, color="#cccccc", alpha=0.3)
            ax2.axhline(
                y=baseline_mean, color="#999999", linewidth=1.5,
                linestyle="--",
                label=f"LOSO baseline ({baseline_mean:.2f})",
            )

        # Sparing annotations for key data points only
        annotate_ns = {25: (0, 16), 200: (0, -18), 2000: (0, 16)}
        for n, m in zip(plot_ns, plot_means):
            if n in annotate_ns:
                ox, oy = annotate_ns[n]
                ax2.annotate(
                    f"N={n}", (n, m),
                    textcoords="offset points", xytext=(ox, oy),
                    ha="center", fontsize=9, fontweight="bold",
                )

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

        ax2.set_xlabel("Society Preference Pairs (N)", fontsize=12)
        ax2.set_ylabel("Avg Regret (lower is better)", fontsize=12)
        ax2.set_title(
            f"(b) Frontier Quality vs Data Budget\n"
            f"(mean + 95% bootstrap CI, {n_seeds_regret} seeds)",
            fontsize=12,
        )
        ax2.legend(fontsize=10)

    plt.tight_layout()

    # Save both PDF (for paper) and PNG (for quick preview)
    pdf_path = os.path.join(
        project_root, "docs/f2/paper/figures/fig4_partial_sampling.pdf",
    )
    png_path = os.path.join(project_root, "results/exp4_partial_sampling.png")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    plt.close()


if __name__ == "__main__":
    main()
