"""Fit functional forms to multi-stakeholder Goodhart data.

Tests whether Gao et al.'s R_gold(d) = α√d − βd applies to our
BT preference learning setting when parameterized by N (pairs).

Three forms:
  Form 1: U(N) = U_0 + α√N − βN          (Gao's, N as proxy for d)
  Form 2: U(N) = U_0 + α√d(N) − βd(N)    (Gao + BT convergence, d(N) = 1-exp(-N/τ))
  Form 3: U(N) = U_0 + αN^γ − βN^δ        (Power law, flexible)

Usage:
    uv run python scripts/analysis/analyze_functional_form.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).parent.parent.parent


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Functional forms
# ---------------------------------------------------------------------------

def form1_gao(N, U0, alpha, beta):
    """Gao's form: U = U_0 + α√N - βN"""
    return U0 + alpha * np.sqrt(N) - beta * N


def form1_learning_only(N, U0, alpha):
    """Learning only (no Goodhart): U = U_0 + α√N"""
    return U0 + alpha * np.sqrt(N)


def form2_convergence(N, U0, alpha, beta, tau):
    """Gao + BT convergence: U = U_0 + α√d(N) - βd(N), d(N) = 1 - exp(-N/τ)"""
    d = 1.0 - np.exp(-N / tau)
    return U0 + alpha * np.sqrt(d) - beta * d


def form2_learning_convergence(N, U0, alpha, tau):
    """Learning with convergence (no Goodhart): U = U_0 + α√(1 - exp(-N/τ))"""
    d = 1.0 - np.exp(-N / tau)
    return U0 + alpha * np.sqrt(d)


def form3_powerlaw(N, U0, alpha, beta, gamma, delta):
    """Power law: U = U_0 + αN^γ - βN^δ"""
    return U0 + alpha * np.power(N, gamma) - beta * np.power(N, delta)


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def fit_and_report(name, N, U, form_func, p0, param_names, bounds=(-np.inf, np.inf)):
    """Fit a functional form and report results."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(form_func, N, U, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = form_func(N, *popt)
        r2 = compute_r2(U, y_pred)
        residuals = U - y_pred

        params = {pn: round(float(v), 6) for pn, v in zip(param_names, popt)}

        # Confidence intervals from covariance diagonal
        perr = np.sqrt(np.diag(pcov))
        ci = {pn: round(float(e), 6) for pn, e in zip(param_names, perr)}

        return {
            "params": params,
            "ci_1sigma": ci,
            "R2": round(r2, 4),
            "residuals": [round(float(r), 4) for r in residuals],
            "converged": True,
        }
    except (RuntimeError, ValueError) as e:
        return {"converged": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_goodhart_data():
    """Load all Goodhart utility data across datasets."""
    datasets = {}

    # ML-100K
    path = ROOT / "results" / "movielens_goodhart.json"
    if path.exists():
        d = json.load(open(path))
        s2 = d["strategy2_more_data"]
        for stakeholder in ["user", "platform", "diversity"]:
            key = f"{stakeholder}_utility_mean"
            n_vals = sorted(s2.keys(), key=int)
            N = np.array([int(n) for n in n_vals], dtype=float)
            U = np.array([s2[n][key] for n in n_vals])
            datasets[f"ml-100k_{stakeholder}"] = (N, U)

    # ML-1M
    path = ROOT / "results" / "ml-1m_goodhart.json"
    if path.exists():
        d = json.load(open(path))
        s2 = d["strategy2_more_data"]
        for stakeholder in ["user", "platform", "diversity"]:
            key = f"{stakeholder}_utility_mean"
            n_vals = sorted(s2.keys(), key=int)
            N = np.array([int(n) for n in n_vals], dtype=float)
            U = np.array([s2[n][key] for n in n_vals])
            datasets[f"ml-1m_{stakeholder}"] = (N, U)

    # Synthetic
    path = ROOT / "results" / "synthetic_goodhart_utility.json"
    if path.exists():
        d = json.load(open(path))
        s2 = d["strategy2_more_data"]
        for stakeholder in ["user", "platform", "society"]:
            key = f"{stakeholder}_utility_mean"
            n_vals = sorted(s2.keys(), key=int)
            N = np.array([int(n) for n in n_vals], dtype=float)
            U = np.array([s2[n][key] for n in n_vals])
            datasets[f"synthetic_{stakeholder}"] = (N, U)

    return datasets


def load_cosines():
    """Load cosine similarities for each dataset's stakeholders vs user target."""
    cosines = {}

    # ML-100K
    path = ROOT / "results" / "movielens_foundation.json"
    if path.exists():
        d = json.load(open(path))
        cos = d["cosine_similarity"]
        cosines["ml-100k_platform"] = cos.get("user-platform", 0)
        cosines["ml-100k_diversity"] = cos.get("user-diversity", 0)
        cosines["ml-100k_user"] = 1.0  # target with itself

    # ML-1M
    path = ROOT / "results" / "ml-1m_foundation.json"
    if path.exists():
        d = json.load(open(path))
        cos = d["cosine_similarity"]
        cosines["ml-1m_platform"] = cos.get("user-platform", 0)
        cosines["ml-1m_diversity"] = cos.get("user-diversity", 0)
        cosines["ml-1m_user"] = 1.0

    # Synthetic
    path = ROOT / "results" / "synthetic_goodhart_utility.json"
    if path.exists():
        d = json.load(open(path))
        cos = d["cosine_similarities"]
        cosines["synthetic_platform"] = cos.get("platform-user", cos.get("user-platform", 0))
        cosines["synthetic_society"] = cos.get("society-user", cos.get("user-society", 0))
        cosines["synthetic_user"] = 1.0

    return cosines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Functional Form Fitting: Gao et al. on BT Goodhart Data")
    print("=" * 60)

    datasets = load_goodhart_data()
    cosines = load_cosines()

    if not datasets:
        print("ERROR: No Goodhart data found")
        return

    results = {}

    # Categorize: degrading (cos < 0) vs improving (cos > 0)
    degrading = {k: v for k, v in datasets.items() if cosines.get(k, 0) < 0}
    improving = {k: v for k, v in datasets.items() if cosines.get(k, 0) > 0 and "user" not in k}

    # --- Fit degrading stakeholders (Goodhart present) ---
    print(f"\n{'='*60}")
    print("DEGRADING STAKEHOLDERS (cos < 0)")
    print(f"{'='*60}")

    for name, (N, U) in sorted(degrading.items()):
        cos = cosines.get(name, 0)
        print(f"\n--- {name} (cos = {cos:+.3f}) ---")
        print(f"  N:  {N.astype(int).tolist()}")
        print(f"  U:  {[round(u, 3) for u in U]}")

        entry = {"cosine": cos, "N": N.tolist(), "U": U.tolist()}

        # Form 1: Gao's U = U_0 + α√N - βN
        r1 = fit_and_report(
            "Form 1 (Gao)", N, U, form1_gao,
            p0=[U[0], 0.1, 0.001],
            param_names=["U0", "alpha", "beta"],
        )
        entry["form1_gao"] = r1
        if r1["converged"]:
            p = r1["params"]
            n_star = p["alpha"]**2 / (4 * p["beta"]**2) if p["beta"] > 0 else float("inf")
            r1["N_star_predicted"] = round(n_star, 1)
            r1["N_star_observed"] = int(N[np.argmax(U)])
            print(f"  Form 1 (α√N - βN): R²={r1['R2']:.3f}, α={p['alpha']:.4f}, β={p['beta']:.6f}, N*_pred={n_star:.0f}, N*_obs={r1['N_star_observed']}")
        else:
            print(f"  Form 1: FAILED — {r1.get('error', 'unknown')}")

        # Form 2: Gao + convergence U = U_0 + α√d(N) - βd(N)
        r2 = fit_and_report(
            "Form 2 (convergence)", N, U, form2_convergence,
            p0=[U[0], 5.0, 5.0, 200.0],
            param_names=["U0", "alpha", "beta", "tau"],
            bounds=([-50, 0, 0, 1], [50, 100, 100, 10000]),
        )
        entry["form2_convergence"] = r2
        if r2["converged"]:
            p = r2["params"]
            d_star = p["alpha"]**2 / (4 * p["beta"]**2) if p["beta"] > 0 else float("inf")
            if d_star < 1:
                n_star2 = -p["tau"] * np.log(1 - d_star)
            else:
                n_star2 = float("inf")
            r2["N_star_predicted"] = round(n_star2, 1)
            print(f"  Form 2 (conv):      R²={r2['R2']:.3f}, α={p['alpha']:.4f}, β={p['beta']:.4f}, τ={p['tau']:.1f}, N*_pred={n_star2:.0f}")
        else:
            print(f"  Form 2: FAILED — {r2.get('error', 'unknown')}")

        # Form 3: Power law (only if enough data points)
        if len(N) >= 6:
            r3 = fit_and_report(
                "Form 3 (power law)", N, U, form3_powerlaw,
                p0=[U[0], 0.1, 0.001, 0.5, 1.0],
                param_names=["U0", "alpha", "beta", "gamma", "delta"],
                bounds=([-50, 0, 0, 0.01, 0.01], [50, 100, 100, 3.0, 3.0]),
            )
            entry["form3_powerlaw"] = r3
            if r3["converged"]:
                p = r3["params"]
                print(f"  Form 3 (power):     R²={r3['R2']:.3f}, γ={p['gamma']:.3f}, δ={p['delta']:.3f} (Gao: γ=0.5, δ=1.0)")
            else:
                print(f"  Form 3: FAILED — {r3.get('error', 'unknown')}")

        results[name] = entry

    # --- Fit improving stakeholders (learning only) ---
    print(f"\n{'='*60}")
    print("IMPROVING STAKEHOLDERS (cos > 0, learning only)")
    print(f"{'='*60}")

    for name, (N, U) in sorted(improving.items()):
        cos = cosines.get(name, 0)
        print(f"\n--- {name} (cos = {cos:+.3f}) ---")

        entry = {"cosine": cos, "N": N.tolist(), "U": U.tolist()}

        # Form 1 learning only: U = U_0 + α√N
        r_learn = fit_and_report(
            "Learning only (√N)", N, U, form1_learning_only,
            p0=[U[0], 0.1],
            param_names=["U0", "alpha"],
        )
        entry["form1_learning_only"] = r_learn
        if r_learn["converged"]:
            p = r_learn["params"]
            print(f"  √N learning:   R²={r_learn['R2']:.3f}, U_0={p['U0']:.3f}, α={p['alpha']:.4f}")

        # Form 2 learning with convergence: U = U_0 + α√(1-exp(-N/τ))
        r_conv = fit_and_report(
            "Learning + conv", N, U, form2_learning_convergence,
            p0=[U[0], U[-1] - U[0], 200.0],
            param_names=["U0", "alpha", "tau"],
            bounds=([-50, 0, 1], [50, 100, 10000]),
        )
        entry["form2_learning_convergence"] = r_conv
        if r_conv["converged"]:
            p = r_conv["params"]
            print(f"  Conv learning: R²={r_conv['R2']:.3f}, U_0={p['U0']:.3f}, α={p['alpha']:.4f}, τ={p['tau']:.1f}")

        results[name] = entry

    # --- Cross-dataset comparison ---
    print(f"\n{'='*60}")
    print("CROSS-DATASET COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'Name':>25} | {'cos':>7} | {'Form1 R²':>8} | {'Form2 R²':>8} | {'α':>8} | {'β':>8}")
    print(f"  {'-'*25}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for name, entry in sorted(results.items()):
        cos = entry["cosine"]
        f1r2 = entry.get("form1_gao", entry.get("form1_learning_only", {})).get("R2", "—")
        f2r2 = entry.get("form2_convergence", entry.get("form2_learning_convergence", {})).get("R2", "—")
        f1 = entry.get("form1_gao", entry.get("form1_learning_only", {}))
        alpha = f1.get("params", {}).get("alpha", "—") if f1.get("converged") else "—"
        beta = entry.get("form1_gao", {}).get("params", {}).get("beta", "—")
        alpha_s = f"{alpha:.4f}" if isinstance(alpha, float) else alpha
        beta_s = f"{beta:.6f}" if isinstance(beta, float) else str(beta)
        f1r2_s = f"{f1r2:.3f}" if isinstance(f1r2, float) else f1r2
        f2r2_s = f"{f2r2:.3f}" if isinstance(f2r2, float) else f2r2
        print(f"  {name:>25} | {cos:>+7.3f} | {f1r2_s:>8} | {f2r2_s:>8} | {alpha_s:>8} | {beta_s:>8}")

    # Save
    out_path = ROOT / "results" / "functional_form_fit.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
