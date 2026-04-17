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
    """Direction condition scatter using BT-trained cosines across 3 datasets.

    Uses selection_mechanism_ablation.json for softmax T=1.0 utility changes,
    combined with trained cosines from labels_not_loss.json per dataset.
    Falls back to expanded_direction_validation.json if ablation not available.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    C_MIND = "#9C27B0"
    C_AMAZON = "#00897B"
    DS_STYLE = {
        "ml-100k": ("ML-100K", "o", C_100K),
        "ml-1m": ("ML-1M", "D", C_1M),
        "mind-small": ("MIND", "s", C_MIND),
        "amazon-kindle": ("Amazon Kindle", "^", C_AMAZON),
    }

    points = []
    for ds_name in DS_STYLE:
        ev_path = ROOT / "results" / f"{ds_name}_expanded_direction_validation.json"
        lnl_path = ROOT / "results" / f"{ds_name}_labels_not_loss.json"
        if not ev_path.exists() or not lnl_path.exists():
            continue
        with open(ev_path) as f:
            ev = json.load(f)
        with open(lnl_path) as f:
            lnl = json.load(f)
        bt_cos = lnl["group_a_core"]["across_stakeholder_similarity"]["bradley_terry"]
        for p in ev["points"]:
            trn = None
            for k in (f"{p['target']}-{p['hidden']}", f"{p['hidden']}-{p['target']}"):
                if k in bt_cos:
                    trn = bt_cos[k]["mean"]
                    break
            if trn is None:
                continue
            points.append({
                "dataset": ds_name,
                "trained_cos": trn,
                "change_pct": p["change_pct"],
                "match": (trn > 0) == p["improving"],
            })

    if not points:
        print("  WARNING: No direction scatter data found")
        plt.close()
        return

    # Transition zone shading (|cos| < 0.2)
    ax.axvspan(-0.2, 0.2, alpha=0.06, color="#FFC107", zorder=0)
    ax.axvspan(-1.1, -0.2, alpha=0.06, color="red", zorder=0)
    ax.axvspan(0.2, 1.1, alpha=0.05, color="green", zorder=0)

    ax.axhline(y=0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(x=0, color="#999999", linewidth=0.6, linestyle=":", zorder=1)
    ax.axvline(x=-0.2, color="#E0A000", linewidth=0.5, linestyle=":", zorder=1, alpha=0.5)
    ax.axvline(x=0.2, color="#E0A000", linewidth=0.5, linestyle=":", zorder=1, alpha=0.5)

    from matplotlib.lines import Line2D

    datasets_present = set()
    for p in points:
        ds = p["dataset"]
        if ds not in DS_STYLE:
            continue
        datasets_present.add(ds)

        edge = "black" if p["match"] else "red"
        lw = 0.5 if p["match"] else 2.0
        _, marker, color = DS_STYLE[ds]
        ax.scatter(p["trained_cos"], p["change_pct"], c=color, marker=marker,
                   s=70, edgecolors=edge, linewidths=lw,
                   zorder=5, alpha=0.85)

    # Build legend with proxy artists so entries always show clean markers
    handles = []
    for ds in ["ml-100k", "ml-1m", "mind-small", "amazon-kindle"]:
        if ds not in datasets_present:
            continue
        label_text, marker, color = DS_STYLE[ds]
        handles.append(Line2D([0], [0], marker=marker, color="w",
                              markerfacecolor=color, markeredgecolor="black",
                              markeredgewidth=0.5, markersize=8, label=label_text))
    # Add violation indicator
    handles.append(Line2D([0], [0], marker="o", color="w",
                          markerfacecolor="none", markeredgecolor="red",
                          markeredgewidth=2.0, markersize=8, label="Violation"))

    strong = [p for p in points if abs(p["trained_cos"]) > 0.2]
    strong_match = sum(1 for p in strong if p["match"])
    ax.text(0.97, 0.03,
            f"{strong_match}/{len(strong)} for |cos| > 0.2\n"
            f"{len(points)} points, {len(datasets_present)} datasets\n"
            f"trained cosine, softmax T=1.0",
            transform=ax.transAxes, fontsize=7.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                      edgecolor="#66BB6A", alpha=0.9))

    ax.set_xlabel(r"BT-trained cos($\hat{\mathbf{w}}_{\mathrm{target}}$, "
                  r"$\hat{\mathbf{w}}_{\mathrm{hidden}}$)")
    ax.set_ylabel("Utility change, N = 25 \u2192 2000 (%)")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-120, 120)
    ax.legend(handles=handles, loc="upper left", fontsize=8.5, framealpha=0.95,
              edgecolor="#CCCCCC", borderpad=0.6)
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

    # Scenario markers
    w_pure = np.ones(D)
    c_pure = cos(w_pure, w_society)
    ax.scatter(1.0, c_pure, c=C_DIV, marker="X", s=80,
               edgecolors="black", linewidths=0.8, zorder=5)

    w_2023 = make_w(-0.3)
    c_2023 = cos(w_2023, w_society)
    ax.scatter(-0.3, c_2023, c=C_PLAT, marker="o", s=80,
               edgecolors="black", linewidths=0.8, zorder=5)

    w_ua = make_w(-1.0)
    c_ua = cos(w_ua, w_society)
    ax.scatter(-1.0, c_ua, c=C_USER, marker="o", s=80,
               edgecolors="black", linewidths=0.8, zorder=5)

    # Annotations in the white space, arrows pointing to markers
    ax.annotate("Pure engagement",
                xy=(1.0, c_pure), xytext=(-0.8, 0.18),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=-0.2"))

    ax.annotate("2023 Phoenix (α = 0.3)",
                xy=(-0.3, c_2023), xytext=(-2.0, 0.25),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=0.2"))

    ax.annotate("User-aligned (α = 1.0)",
                xy=(-1.0, c_ua), xytext=(-2.8, 0.55),
                fontsize=9, ha="center",
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8,
                                connectionstyle="arc3,rad=0.2"))

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

    # Plot raw regret — no >100% confusion
    datasets = [
        ("ML-100K", "results/ml-100k_loso.json", C_100K, "s"),
        ("ML-1M", "results/ml-1m_loso.json", C_1M, "^"),
    ]

    for ds_name, path, color, marker in datasets:
        d = json.load(open(ROOT / path))
        budget = d.get("data_budget", {})
        sweep = budget.get("sweep", [])
        if not sweep:
            continue

        N = np.array([e["n_pairs"] for e in sweep], dtype=float)
        regret = np.array([e["avg_regret_mean"] for e in sweep])
        std = np.array([e["avg_regret_std"] for e in sweep])

        # Show up to N=500
        mask = N <= 500
        ax.plot(N[mask], regret[mask], color=color,
                marker=marker, markersize=5.5, linewidth=1.8,
                label=ds_name, zorder=3)
        ax.fill_between(N[mask], (regret - std)[mask], (regret + std)[mask],
                         alpha=0.18, color=color, zorder=2)

    # Zero regret line
    ax.axhline(y=0, color="#888888", linewidth=0.8, linestyle=":",
               zorder=1, label="Zero regret")

    # Subtle shading: regret > 0 = harm zone
    ax.axhspan(0, 6, alpha=0.02, color="red", zorder=0)

    # Linear x-axis with evenly spaced ticks at actual N values
    ax.set_xlim(-10, 520)
    ax.set_xticks([0, 25, 50, 100, 200, 500])
    ax.set_xlabel("Hidden stakeholder preference pairs (N)")
    ax.set_ylabel("Regret on hidden stakeholder")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95,
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
