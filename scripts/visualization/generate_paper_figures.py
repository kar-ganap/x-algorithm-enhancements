"""Generate publication-quality figures for the arXiv preprint.

Fig 1: X transition diagram (TikZ — generated separately)
Fig 2: Goodhart utility curves (normalized % change, 2 panels)
Fig 3: Direction condition validation scatter (6 points)
Fig 4: Audit threshold curve (platform neg-weight vs cos)
Fig 5: Data budget recovery with bootstrap CIs

Usage:
    uv run python scripts/visualization/generate_paper_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parent.parent.parent
OUT_DIR = ROOT / "docs" / "f2" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

COLORS = {
    "user": "#2196F3",
    "platform": "#4CAF50",
    "diversity": "#F44336",
    "society": "#F44336",
    "ml100k": "#2196F3",
    "ml1m": "#FF9800",
    "synthetic": "#9C27B0",
}


# ═══════════════════════════════════════════════════════════════
# Figure 2: Goodhart Utility Curves
# ═══════════════════════════════════════════════════════════════

def fig2_goodhart_curves():
    """Normalized % change from N=25 baseline for all 3 stakeholders."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))

    datasets = [
        ("ML-100K", "results/movielens_goodhart.json", "(a)"),
        ("ML-1M", "results/ml-1m_goodhart.json", "(b)"),
    ]

    cosines = {
        "ML-100K": {"user": 1.0, "platform": +0.96, "diversity": -0.31},
        "ML-1M": {"user": 1.0, "platform": +0.93, "diversity": -0.16},
    }

    for ax, (ds_name, path, panel_label) in zip(axes, datasets):
        d = json.load(open(ROOT / path))
        s2 = d["strategy2_more_data"]
        n_keys = sorted(s2.keys(), key=int)
        N = np.array([int(k) for k in n_keys])

        baseline = {s: s2[n_keys[0]][f"{s}_utility_mean"] for s in ["user", "platform", "diversity"]}

        for s, color, marker in [
            ("user", COLORS["user"], "o"),
            ("platform", COLORS["platform"], "s"),
            ("diversity", COLORS["diversity"], "D"),
        ]:
            values = np.array([s2[k][f"{s}_utility_mean"] for k in n_keys])
            pct_change = (values - baseline[s]) / abs(baseline[s]) * 100

            cos = cosines[ds_name][s]
            if s == "user":
                label = f"user (target)"
            else:
                label = f"{s} (cos = {cos:+.2f})"

            ax.plot(N, pct_change, color=color, marker=marker, markersize=4.5,
                    linewidth=1.8, label=label, zorder=3)

        # Shade the Goodhart zone (below 0)
        ax.axhspan(-55, 0, alpha=0.04, color="red", zorder=0)
        ax.axhline(y=0, color="gray", linewidth=0.7, linestyle="--", zorder=1)

        ax.set_xscale("log")
        ax.set_xlabel("Training pairs (N)", fontsize=9)
        ax.set_title(f"{panel_label} {ds_name}", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([25, 100, 500, 2000])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(-52, 22)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel("Change from N = 25 baseline (%)", fontsize=9)
    # Legend on the first panel, positioned to avoid lines
    axes[0].legend(loc="lower left", framealpha=0.95, fontsize=8,
                   edgecolor="lightgray")

    fig.tight_layout(w_pad=2.5)
    path = OUT_DIR / "fig2_goodhart_curves.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"  Fig 2 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3: Direction Condition Scatter
# ═══════════════════════════════════════════════════════════════

def fig3_direction_scatter():
    """6-point scatter: cos vs total change, colored by dataset."""
    fig, ax = plt.subplots(figsize=(5, 3.8))

    # Collect all points
    points = []

    # MovieLens datasets
    for ds, path, cos_path, marker in [
        ("ML-100K", "results/movielens_goodhart.json", "results/movielens_foundation.json", "s"),
        ("ML-1M", "results/ml-1m_goodhart.json", "results/ml-1m_foundation.json", "^"),
    ]:
        d = json.load(open(ROOT / path))
        f = json.load(open(ROOT / cos_path))
        s2 = d["strategy2_more_data"]
        n_keys = sorted(s2.keys(), key=int)
        cos_sims = f["cosine_similarity"]

        for hidden in ["platform", "diversity"]:
            cos_key = f"user-{hidden}"
            cos_val = cos_sims.get(cos_key, 0)
            first_u = s2[n_keys[0]][f"{hidden}_utility_mean"]
            last_u = s2[n_keys[-1]][f"{hidden}_utility_mean"]
            total_change = (last_u - first_u) / abs(first_u) * 100
            points.append((cos_val, total_change, ds, marker, hidden))

    # Synthetic
    d_syn = json.load(open(ROOT / "results" / "synthetic_goodhart_utility.json"))
    cos_syn = d_syn["cosine_similarities"]
    s2_syn = d_syn["strategy2_more_data"]
    n_keys_syn = sorted(s2_syn.keys(), key=int)
    for hidden in ["platform", "society"]:
        cos_key = f"{hidden}-user"
        cos_val = cos_syn.get(cos_key, cos_syn.get(f"user-{hidden}", 0))
        first_u = s2_syn[n_keys_syn[0]][f"{hidden}_utility_mean"]
        last_u = s2_syn[n_keys_syn[-1]][f"{hidden}_utility_mean"]
        total_change = (last_u - first_u) / abs(first_u) * 100
        points.append((cos_val, total_change, "Synthetic", "o", hidden))

    # Shaded quadrants using fill_between for proper data-coordinate alignment
    ax.fill_between([-0.5, 0], -60, 0, alpha=0.08, color="red", zorder=0)    # cos<0, degrades
    ax.fill_between([0, 1.1], 0, 80, alpha=0.06, color="green", zorder=0)     # cos>0, improves

    # Quadrant labels — positioned in empty space
    ax.text(-0.28, -40, "GOODHART", fontsize=9, fontweight="bold",
            ha="center", va="center", color="#C62828")
    ax.text(-0.28, -50, "cos < 0, utility degrades", fontsize=7,
            ha="center", va="center", color="#C62828", style="italic")
    ax.text(0.55, 60, "SAFE", fontsize=9, fontweight="bold",
            ha="center", va="center", color="#2E7D32")
    ax.text(0.55, 52, "cos > 0, utility improves", fontsize=7,
            ha="center", va="center", color="#2E7D32", style="italic")

    # Reference lines
    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

    # Plot points with specific label offsets to avoid overlap
    label_offsets = {
        ("ML-100K", "diversity"): (-12, -14),
        ("ML-1M", "diversity"): (10, 8),
        ("ML-100K", "platform"): (8, -12),
        ("ML-1M", "platform"): (8, 8),
        ("Synthetic", "platform"): (-15, -14),
        ("Synthetic", "society"): (8, -12),
    }

    ds_plotted = set()
    for cos_val, change, ds, marker, hidden in points:
        color = COLORS.get(ds.lower().replace("-", ""), "gray")
        label = ds if ds not in ds_plotted else None
        ds_plotted.add(ds)
        ax.scatter(cos_val, change, c=color, marker=marker, s=80,
                   edgecolors="black", linewidths=0.6, zorder=5, label=label)
        # Per-point label offset
        offset = label_offsets.get((ds, hidden), (8, 5))
        ax.annotate(hidden, (cos_val, change), fontsize=7.5,
                    xytext=offset, textcoords="offset points",
                    color="#555555", fontweight="bold")

    ax.set_xlabel(r"cos($\mathbf{w}_{\mathrm{target}}$, $\mathbf{w}_{\mathrm{hidden}}$)",
                  fontsize=10)
    ax.set_ylabel("Utility change, N = 25 → 2000 (%)", fontsize=9)
    ax.set_xlim(-0.45, 1.05)
    ax.set_ylim(-58, 78)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.95,
              edgecolor="lightgray")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # "6/6 validated" annotation
    ax.text(0.97, 0.03, "6/6 match\n0 violations",
            transform=ax.transAxes, fontsize=8,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9",
                      edgecolor="#4CAF50", alpha=0.9))

    fig.tight_layout()
    path = OUT_DIR / "fig3_direction_scatter.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"  Fig 3 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 4: Audit Threshold
# ═══════════════════════════════════════════════════════════════

def fig4_audit_threshold():
    """Platform negative-weight vs cos(platform, society)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Build stakeholder weights and sweep
    D = 18
    N_POS, N_NEUTRAL, N_NEG = 5, 9, 4

    def make_weights(neg_weight, neutral_weight=0.0):
        w = np.zeros(D)
        w[:N_POS] = 1.0
        w[N_POS:N_POS + N_NEUTRAL] = neutral_weight
        w[N_POS + N_NEUTRAL:] = neg_weight
        return w

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    w_society = make_weights(-4.0)

    # Sweep: negative weight from +1.0 to -4.0
    neg_weights = np.linspace(1.0, -4.0, 200)
    cos_values = [cos(make_weights(nw), w_society) for nw in neg_weights]

    # Plot curve
    cos_arr = np.array(cos_values)
    ax.plot(neg_weights, cos_arr, color="#333333", linewidth=2, zorder=3)

    # Shaded regions
    ax.fill_between(neg_weights, cos_arr, -0.5,
                     where=(cos_arr < 0), alpha=0.15, color="red",
                     label="Goodhart zone (cos < 0)", zorder=1)
    ax.fill_between(neg_weights, cos_arr, 1.1,
                     where=(cos_arr > 0), alpha=0.1, color="green",
                     label="Safe zone (cos > 0)", zorder=1)

    ax.axhline(y=0, color="gray", linewidth=0.8, linestyle="--", zorder=2)

    # Annotate specific scenarios
    # Pure engagement: all actions = +1
    w_pure = np.ones(D)
    c_pure = cos(w_pure, w_society)
    ax.scatter(1.0, c_pure, c=COLORS["diversity"], s=80, edgecolors="black",
               linewidths=0.8, zorder=5, marker="X")
    ax.annotate("Pure engagement\n(all actions = +1)", (1.0, c_pure),
                fontsize=7.5, xytext=(-30, -30), textcoords="offset points",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    # 2023 Phoenix
    c_2023 = cos(make_weights(-0.3), w_society)
    ax.scatter(-0.3, c_2023, c=COLORS["platform"], s=80, edgecolors="black",
               linewidths=0.8, zorder=5, marker="o")
    ax.annotate("2023 Phoenix\n(α = 0.3)", (-0.3, c_2023),
                fontsize=7.5, xytext=(-15, 15), textcoords="offset points",
                ha="right", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.8))

    # User-aligned
    c_user = cos(make_weights(-1.0), w_society)
    ax.scatter(-1.0, c_user, c=COLORS["user"], s=80, edgecolors="black",
               linewidths=0.8, zorder=5, marker="o")
    ax.annotate("User-aligned\n(α = 1.0)", (-1.0, c_user),
                fontsize=7.5, xytext=(-15, -15), textcoords="offset points",
                ha="right", va="top",
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=0.8))

    # X-axis labels
    ax.set_xlabel("Platform weight on negative actions",
                  fontsize=9)
    # Add secondary description below
    ax.text(0.5, -0.18, "blocks as engagement  ←———→  blocks as harm",
            transform=ax.transAxes, fontsize=7.5, ha="center", va="top",
            color="gray", style="italic")
    ax.set_ylabel("cos(platform, society)", fontsize=9)
    ax.set_xlim(1.3, -4.3)  # reversed: engagement on left, harm on right
    ax.set_ylim(-0.42, 1.05)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.95,
              edgecolor="lightgray")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    path = OUT_DIR / "fig4_audit_threshold.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"  Fig 4 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 5: Data Budget Recovery with CIs
# ═══════════════════════════════════════════════════════════════

def fig5_data_budget():
    """Recovery curves with bootstrap CI bands for both datasets."""
    fig, ax = plt.subplots(figsize=(5, 3.5))

    datasets = [
        ("ML-100K", "results/ml-100k_deep_analysis.json", COLORS["ml100k"], "s"),
        ("ML-1M", "results/ml-1m_deep_analysis.json", COLORS["ml1m"], "^"),
    ]

    for ds_name, path, color, marker in datasets:
        d = json.load(open(ROOT / path))
        if "bootstrap" not in d:
            continue

        entries = d["bootstrap"]["per_n"]
        N = np.array([e["n_pairs"] for e in entries])
        rec = np.array([e["recovery_mean"] * 100 for e in entries])
        lo = np.array([e["recovery_ci_lo"] * 100 for e in entries])
        hi = np.array([e["recovery_ci_hi"] * 100 for e in entries])

        # Only show up to 120% (cap for readability)
        mask = N <= 500  # avoid the extreme >200% values cluttering
        N_plot = N[mask]
        rec_plot = np.clip(rec[mask], -10, 120)
        lo_plot = np.clip(lo[mask], -10, 120)
        hi_plot = np.clip(hi[mask], -10, 120)

        ax.plot(N_plot, rec_plot, color=color, marker=marker, markersize=5,
                linewidth=1.5, label=ds_name, zorder=3)
        ax.fill_between(N_plot, lo_plot, hi_plot, alpha=0.2, color=color, zorder=2)

    # Reference lines
    ax.axhline(y=0, color="gray", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(y=100, color="#888888", linewidth=0.8, linestyle=":",
               zorder=1, label="Full recovery (100%)")

    # Annotate N=25 region
    ax.annotate("N = 25\n46–56% recovery\n95% CI: [40%, 64%]",
                xy=(25, 56), xytext=(100, 95),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8,
                                connectionstyle="arc3,rad=0.2"),
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#888888", alpha=0.95))

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([25, 50, 100, 200, 500])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Hidden stakeholder preference pairs (N)", fontsize=9)
    ax.set_ylabel("Recovery of hidden stakeholder harm (%)", fontsize=9)
    ax.set_ylim(-10, 125)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95,
              edgecolor="lightgray")
    ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.tight_layout()
    path = OUT_DIR / "fig5_data_budget.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    print(f"  Fig 5 saved: {path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Generating paper figures...")
    print(f"Output: {OUT_DIR}")
    print()

    fig2_goodhart_curves()
    fig3_direction_scatter()
    fig4_audit_threshold()
    fig5_data_budget()

    print()
    print(f"Done. Fig 1 (X transition) is TikZ — generated in LaTeX.")
    print(f"All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
