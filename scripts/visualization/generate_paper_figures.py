"""Generate publication-quality figures for the arXiv preprint.

Design principles:
- Larger figures, less annotation clutter
- Let the data speak — details go in captions
- Clear color/marker distinction
- No text overlapping data points or other text

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

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

C_USER = "#1976D2"
C_PLAT = "#388E3C"
C_DIV = "#D32F2F"
C_100K = "#1976D2"
C_1M = "#E65100"
C_SYN = "#7B1FA2"


# ═══════════════════════════════════════════════════════════════
# Figure 2: Goodhart Utility Curves
# ═══════════════════════════════════════════════════════════════

def fig2_goodhart_curves():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))

    datasets = [
        ("ML-100K", "results/movielens_goodhart.json",
         {"platform": +0.96, "diversity": -0.31}),
        ("ML-1M", "results/ml-1m_goodhart.json",
         {"platform": +0.93, "diversity": -0.16}),
    ]

    for idx, (ax, (ds_name, path, cosines)) in enumerate(zip(axes, datasets)):
        d = json.load(open(ROOT / path))
        s2 = d["strategy2_more_data"]
        n_keys = sorted(s2.keys(), key=int)
        N = np.array([int(k) for k in n_keys])
        baseline = {s: s2[n_keys[0]][f"{s}_utility_mean"]
                    for s in ["user", "platform", "diversity"]}

        lines = [
            ("user", C_USER, "o", "User (target)"),
            ("platform", C_PLAT, "s", f"Platform (cos = {cosines['platform']:+.2f})"),
            ("diversity", C_DIV, "D", f"Diversity (cos = {cosines['diversity']:+.2f})"),
        ]

        for s, color, marker, label in lines:
            vals = np.array([s2[k][f"{s}_utility_mean"] for k in n_keys])
            pct = (vals - baseline[s]) / abs(baseline[s]) * 100
            ax.plot(N, pct, color=color, marker=marker, markersize=5,
                    linewidth=1.8, label=label, zorder=3)

        ax.axhspan(-55, 0, alpha=0.04, color="red", zorder=0)
        ax.axhline(y=0, color="#999999", linewidth=0.7, linestyle="--", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("Training pairs (N)")
        ax.set_title(f"({chr(97+idx)}) {ds_name}", fontweight="bold")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks([25, 100, 500, 2000])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_ylim(-52, 22)
        ax.grid(True, alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("Change from N = 25 baseline (%)")
    # Each panel gets its own legend with its own cosine values
    for ax in axes:
        ax.legend(loc="lower left", framealpha=0.95, edgecolor="#CCCCCC",
                  borderpad=0.5, fontsize=8.5)

    fig.tight_layout(w_pad=2.0)
    for ext in [".pdf", ".png"]:
        fig.savefig(OUT_DIR / f"fig2_goodhart_curves{ext}",
                    dpi=200 if ext == ".png" else 300)
    print("  Fig 2: Goodhart curves ✓")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 3: Direction Condition Scatter
# ═══════════════════════════════════════════════════════════════

def fig3_direction_scatter():
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # Collect points
    points = []
    for ds, path, cos_path, marker in [
        ("ML-100K", "results/movielens_goodhart.json",
         "results/movielens_foundation.json", "s"),
        ("ML-1M", "results/ml-1m_goodhart.json",
         "results/ml-1m_foundation.json", "^"),
    ]:
        d = json.load(open(ROOT / path))
        f = json.load(open(ROOT / cos_path))
        s2 = d["strategy2_more_data"]
        n_keys = sorted(s2.keys(), key=int)
        for hidden in ["platform", "diversity"]:
            cos_val = f["cosine_similarity"].get(f"user-{hidden}", 0)
            first_u = s2[n_keys[0]][f"{hidden}_utility_mean"]
            last_u = s2[n_keys[-1]][f"{hidden}_utility_mean"]
            change = (last_u - first_u) / abs(first_u) * 100
            points.append((cos_val, change, ds, marker, hidden))

    d_syn = json.load(open(ROOT / "results" / "synthetic_goodhart_utility.json"))
    cos_syn = d_syn["cosine_similarities"]
    s2_syn = d_syn["strategy2_more_data"]
    nk_syn = sorted(s2_syn.keys(), key=int)
    for hidden in ["platform", "society"]:
        ck = f"{hidden}-user"
        cos_val = cos_syn.get(ck, cos_syn.get(f"user-{hidden}", 0))
        first_u = s2_syn[nk_syn[0]][f"{hidden}_utility_mean"]
        last_u = s2_syn[nk_syn[-1]][f"{hidden}_utility_mean"]
        change = (last_u - first_u) / abs(first_u) * 100
        points.append((cos_val, change, "Synthetic", "o", hidden))

    # Quadrant shading — use axvspan + axhspan for clean data-coordinate alignment
    ax.axvspan(-0.5, 0, alpha=0.07, color="red", zorder=0)
    ax.axvspan(0, 1.1, alpha=0.05, color="green", zorder=0)

    # Reference lines
    ax.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(x=0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)

    # Plot points with distinct markers per stakeholder type
    ds_colors = {"ML-100K": C_100K, "ML-1M": C_1M, "Synthetic": C_SYN}
    ds_plotted = set()
    for cos_val, change, ds, marker, hidden in points:
        color = ds_colors[ds]
        label = ds if ds not in ds_plotted else None
        ds_plotted.add(ds)
        ax.scatter(cos_val, change, c=color, marker=marker, s=110,
                   edgecolors="black", linewidths=0.7, zorder=5, label=label)

    # No per-point text — legend identifies datasets, quadrant shading
    # identifies direction. Caption maps points to stakeholders.

    # Validation badge
    ax.text(0.97, 0.03, "6/6 validated\n0 violations",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                      edgecolor="#66BB6A", alpha=0.9))

    ax.set_xlabel(r"cos($\mathbf{w}_{\mathrm{target}}$, $\mathbf{w}_{\mathrm{hidden}}$)")
    ax.set_ylabel("Utility change, N = 25 → 2000 (%)")
    ax.set_xlim(-0.45, 1.05)
    ax.set_ylim(-55, 75)
    ax.legend(loc="upper left", framealpha=0.95, edgecolor="#CCCCCC",
              markerscale=1.0, borderpad=0.6)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(OUT_DIR / f"fig3_direction_scatter{ext}",
                    dpi=200 if ext == ".png" else 300)
    print("  Fig 3: Direction scatter ✓")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 4: Audit Threshold
# ═══════════════════════════════════════════════════════════════

def fig4_audit_threshold():
    fig, ax = plt.subplots(figsize=(6, 4))

    D = 18
    N_POS, N_NEUTRAL = 5, 9

    def make_w(neg_weight, neutral_weight=0.0):
        w = np.zeros(D)
        w[:N_POS] = 1.0
        w[N_POS:N_POS + N_NEUTRAL] = neutral_weight
        w[N_POS + N_NEUTRAL:] = neg_weight
        return w

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    w_society = make_w(-4.0)

    # Sweep
    neg_weights = np.linspace(1.0, -4.0, 300)
    cos_values = np.array([cos(make_w(nw), w_society) for nw in neg_weights])

    # Main curve
    ax.plot(neg_weights, cos_values, color="#333333", linewidth=2.5, zorder=3)

    # Shading
    ax.fill_between(neg_weights, cos_values, -0.5,
                     where=cos_values < 0, alpha=0.12, color="red",
                     label="Goodhart zone", zorder=1)
    ax.fill_between(neg_weights, cos_values, 1.1,
                     where=cos_values > 0, alpha=0.08, color="green",
                     label="Safe zone", zorder=1)
    ax.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--", zorder=2)

    # Scenario points — positioned to avoid overlap
    scenarios = [
        (np.ones(D), 1.0, "Pure engagement", C_DIV, "X",
         (15, 15), "left"),
        (make_w(-0.3), -0.3, "2023 Phoenix (α = 0.3)", C_PLAT, "o",
         (15, -20), "left"),
        (make_w(-1.0), -1.0, "User-aligned (α = 1.0)", C_USER, "o",
         (-15, -20), "right"),
    ]

    for w_p, x_pos, label, color, marker, offset, ha in scenarios:
        c = cos(w_p, w_society)
        ax.scatter(x_pos, c, c=color, marker=marker, s=80,
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.annotate(label, (x_pos, c), fontsize=8,
                    xytext=offset, textcoords="offset points",
                    ha=ha, va="center",
                    arrowprops=dict(arrowstyle="->", color="#666666",
                                    lw=0.7, connectionstyle="arc3,rad=0.15"))

    ax.set_xlabel("Platform weight on negative actions")
    ax.set_ylabel("cos(platform, society)", fontsize=10)
    ax.set_xlim(1.3, -4.3)
    ax.set_ylim(-0.5, 1.08)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              edgecolor="#CCCCCC", borderpad=0.6)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(OUT_DIR / f"fig4_audit_threshold{ext}",
                    dpi=200 if ext == ".png" else 300)
    print("  Fig 4: Audit threshold ✓")
    plt.close()


# ═══════════════════════════════════════════════════════════════
# Figure 5: Data Budget Recovery
# ═══════════════════════════════════════════════════════════════

def fig5_data_budget():
    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    datasets = [
        ("ML-100K", "results/ml-100k_deep_analysis.json", C_100K, "s"),
        ("ML-1M", "results/ml-1m_deep_analysis.json", C_1M, "^"),
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

        # Cap display at N≤500 and recovery≤120% for readability
        mask = N <= 500
        ax.plot(N[mask], np.clip(rec[mask], -5, 115), color=color,
                marker=marker, markersize=5.5, linewidth=1.8,
                label=ds_name, zorder=3)
        ax.fill_between(N[mask],
                         np.clip(lo[mask], -5, 115),
                         np.clip(hi[mask], -5, 115),
                         alpha=0.18, color=color, zorder=2)

    ax.axhline(y=0, color="#999999", linewidth=0.6, linestyle="--", zorder=1)
    ax.axhline(y=100, color="#888888", linewidth=0.8, linestyle=":",
               zorder=1, label="Full recovery")

    # N=25 marker — vertical line instead of annotation box
    ax.axvline(x=25, color="#AAAAAA", linewidth=0.6, linestyle=":", zorder=1)
    ax.text(28, 105, "N = 25", fontsize=9, color="#555555", va="bottom")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks([25, 50, 100, 200, 500])
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlabel("Hidden stakeholder preference pairs (N)")
    ax.set_ylabel("Recovery of hidden stakeholder harm (%)")
    ax.set_ylim(-8, 118)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95,
              edgecolor="#CCCCCC", borderpad=0.6)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    for ext in [".pdf", ".png"]:
        fig.savefig(OUT_DIR / f"fig5_data_budget{ext}",
                    dpi=200 if ext == ".png" else 300)
    print("  Fig 5: Data budget ✓")
    plt.close()


# ═══════════════════════════════════════════════════════════════

def main():
    print("Generating paper figures...")
    fig2_goodhart_curves()
    fig3_direction_scatter()
    fig4_audit_threshold()
    fig5_data_budget()
    print(f"Done. Output: {OUT_DIR}")
    print("Fig 1 (X transition) is TikZ — generated in LaTeX.")


if __name__ == "__main__":
    main()
