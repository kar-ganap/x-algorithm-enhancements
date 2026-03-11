#!/usr/bin/env python3
"""LLM confidence as margin proxy experiment for F4 reward modeling.

Tests whether an LLM's confidence when making pairwise preferences can
substitute for analytic margin in predicting BT weight differentiation.

Approach: for 15 sweep points from the existing disagreement bound analysis,
prompt Claude Haiku to make pairwise preferences from two stakeholder
perspectives, collect disagreement rate and confidence, then fit regression
models against ground truth cosine similarities.

Usage:
    uv pip install anthropic  # one-time
    uv run python scripts/llm_margin_proxy.py
"""

import asyncio
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Module loading (same pattern as analyze_disagreement_bound.py)
# ---------------------------------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_module(
    module_name: str, module_path: Path, register_as: str | None = None
):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if register_as:
        sys.modules[register_as] = module
    spec.loader.exec_module(module)
    return module


try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not found. Install: uv pip install anthropic")
    sys.exit(1)

alt_losses = load_module(
    "alternative_losses",
    project_root / "enhancements" / "reward_modeling" / "alternative_losses.py",
    register_as="enhancements.reward_modeling.alternative_losses",
)

POSITIVE_INDICES: list[int] = alt_losses.POSITIVE_INDICES
NEGATIVE_INDICES: list[int] = alt_losses.NEGATIVE_INDICES
ACTION_INDICES: dict[str, int] = alt_losses.ACTION_INDICES

loss_exp_mod = load_module(
    "run_loss_experiments",
    project_root / "scripts" / "run_loss_experiments.py",
)
generate_content_pool = loss_exp_mod.generate_content_pool

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POSITIVE_ACTIONS = ["favorite", "repost", "follow_author", "share", "reply"]
NEGATIVE_ACTIONS = ["block_author", "mute_author", "report", "not_interested"]

MODEL = "claude-haiku-4-5-20251001"
N_CONTENT = 500
BATCH_SIZE = 10
CONCURRENCY = 5
SEED = 42

# 15 sweep points spanning disagreement range 0.04–0.41, from both sweeps
SWEEP_CONFIGS = [
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 0.8},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 1.5},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 2.0},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 3.0},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 0.5},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 5.0},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 7.0},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 0.8},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 1.0},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 0.2},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 1.5},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 2.5},
    {"sweep": "B", "alpha_fixed": 1.0, "alpha_sweep": 0.1},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 4.0},
    {"sweep": "A", "alpha_fixed": 0.3, "alpha_sweep": 8.0},
]


def flush_print(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Content formatting and prompt construction
# ---------------------------------------------------------------------------


def compute_scaled_scores(
    content_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute engagement and negativity scores scaled to 0-100."""
    n_content = len(content_probs)
    pos_raw = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_raw = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])
    pos_max = pos_raw.max() if pos_raw.max() > 0 else 1.0
    neg_max = neg_raw.max() if neg_raw.max() > 0 else 1.0
    engagement = (pos_raw / pos_max * 100).astype(int)
    negativity = (neg_raw / neg_max * 100).astype(int)
    return engagement, negativity


def stakeholder_description(alpha: float) -> str:
    """Map alpha to a natural language stakeholder perspective."""
    if alpha <= 0.3:
        return (
            "You are an engagement-focused platform operator. "
            "User activity is what matters most to you. "
            "Some negative reactions are normal and acceptable — "
            "they barely factor into your decisions."
        )
    if alpha <= 0.8:
        return (
            "You are a platform manager who values engagement but keeps "
            "an eye on negative reactions. Both matter, but engagement "
            "is somewhat more important to you."
        )
    if alpha <= 1.5:
        return (
            "You balance engagement with user safety equally. "
            "Negative reactions concern you just as much as "
            "positive engagement."
        )
    if alpha <= 3.0:
        return (
            "You prioritize a healthy platform. Negative reactions "
            "weigh heavily in your decisions — more than raw "
            "engagement numbers. You'd rather show less engaging "
            "content than risk harmful reactions."
        )
    return (
        "You are a safety-first content moderator. Reducing harmful "
        "content is your top priority. High negativity is a dealbreaker "
        "regardless of how much engagement a post gets. "
        "You strongly prefer low-negativity content."
    )


def build_batch_prompt(
    engagement: np.ndarray,
    negativity: np.ndarray,
    pair_indices: list[tuple[int, int]],
    alpha: float,
    rng: np.random.Generator,
) -> tuple[str, str, list[bool]]:
    """Build system and user prompts for a batch of content pairs.

    Uses natural language stakeholder descriptions and simplified
    engagement/negativity scores. The LLM applies its own implicit
    utility function — no formula provided.

    Returns (system_prompt, user_prompt, swap_flags).
    """
    perspective = stakeholder_description(alpha)

    system = (
        "You evaluate content pairs for a social media recommendation system.\n"
        "Each post has two scores (0-100):\n"
        "  - engagement: how much users interact positively "
        "(likes, shares, replies, follows)\n"
        "  - negativity: how much users react negatively "
        "(blocks, mutes, reports, not-interested)\n\n"
        f"Your perspective: {perspective}\n\n"
        "For each pair, pick the post YOU would rather recommend "
        "given your perspective.\n"
        "Then rate your confidence: 0.50 = coin flip, 1.00 = certain.\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"pairs": [{"pair_id": 1, "preferred": "A", "confidence": 0.85},'
        " ...]}"
    )

    user_parts = [f"Evaluate these {len(pair_indices)} content pairs:\n"]
    swap_flags = []
    for i, (c1, c2) in enumerate(pair_indices, 1):
        swap = bool(rng.integers(2))
        swap_flags.append(swap)
        if swap:
            a_idx, b_idx = c2, c1
        else:
            a_idx, b_idx = c1, c2

        user_parts.append(
            f"Pair {i}: "
            f"Post A: engagement={engagement[a_idx]} negativity={negativity[a_idx]} | "
            f"Post B: engagement={engagement[b_idx]} negativity={negativity[b_idx]}"
        )

    return system, "\n".join(user_parts), swap_flags


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_llm_response(
    response_text: str, expected_count: int
) -> list[dict[str, Any]]:
    """Parse and validate LLM JSON response."""
    text = response_text.strip()
    # Sometimes LLM wraps in ```json ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    # Try direct parse first
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Extract the first JSON object {...} from the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        data = json.loads(match.group())
    results: list[dict[str, Any]] = []
    for entry in data["pairs"]:
        conf = float(entry["confidence"])
        conf = max(0.50, min(1.00, conf))
        preferred = str(entry["preferred"]).upper()
        if preferred not in ("A", "B"):
            raise ValueError(f"Invalid preferred: {preferred}")
        results.append(
            {
                "pair_id": int(entry["pair_id"]),
                "preferred": preferred,
                "confidence": conf,
            }
        )
    if len(results) != expected_count:
        flush_print(
            f"  WARNING: expected {expected_count} responses, got {len(results)}"
        )
    return results


def unswap_preference(preferred: str, swapped: bool) -> str:
    """Undo A/B swap to get preference in original c1/c2 terms."""
    if not swapped:
        return preferred
    return "B" if preferred == "A" else "A"


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------


async def query_llm_batch(
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
    user_prompt: str,
    batch_size: int,
    semaphore: asyncio.Semaphore,
    batch_id: int,
    max_retries: int = 3,
) -> list[dict[str, Any]]:
    """Send a single batch to Haiku, parse structured response."""
    for attempt in range(max_retries):
        try:
            async with semaphore:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            # Find the text content block (skip thinking blocks)
            text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    text = block.text
                    break
            if not text:
                raise ValueError("No text content in response")
            return parse_llm_response(text, batch_size)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2**attempt
                flush_print(
                    f"  Batch {batch_id} attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                await asyncio.sleep(wait)
            else:
                flush_print(f"  Batch {batch_id} FAILED after {max_retries} attempts: {e}")
                return []


async def run_sweep_point(
    client: anthropic.AsyncAnthropic,
    engagement: np.ndarray,
    negativity: np.ndarray,
    alpha_fixed: float,
    alpha_sweep: float,
    n_pairs: int,
    semaphore: asyncio.Semaphore,
    point_idx: int,
) -> dict[str, Any]:
    """Run all LLM queries for one sweep point (both stakeholder perspectives)."""
    rng = np.random.default_rng(SEED + point_idx)
    n_content = len(engagement)

    # Sample pairs
    pair_indices = [
        (int(rng.choice(n_content)), int(rng.choice(n_content)))
        for _ in range(n_pairs)
    ]
    # Ensure c1 != c2
    pair_indices = [
        (c1, c2) if c1 != c2 else (c1, (c2 + 1) % n_content)
        for c1, c2 in pair_indices
    ]

    # Split into batches
    batches = [
        pair_indices[i : i + BATCH_SIZE]
        for i in range(0, len(pair_indices), BATCH_SIZE)
    ]

    # Query both stakeholder perspectives
    all_responses_fixed: list[dict[str, Any]] = []
    all_responses_sweep: list[dict[str, Any]] = []

    for batch_idx, batch_pairs in enumerate(batches):
        # Use same A/B randomization for both perspectives
        batch_rng = np.random.default_rng(SEED + point_idx * 1000 + batch_idx)
        sys_fixed, user_fixed, swaps = build_batch_prompt(
            engagement, negativity, batch_pairs, alpha_fixed, batch_rng
        )
        # Reset rng to get same swaps for sweep perspective
        batch_rng2 = np.random.default_rng(SEED + point_idx * 1000 + batch_idx)
        sys_sweep, user_sweep, _ = build_batch_prompt(
            engagement, negativity, batch_pairs, alpha_sweep, batch_rng2
        )

        resp_fixed, resp_sweep = await asyncio.gather(
            query_llm_batch(
                client, sys_fixed, user_fixed, len(batch_pairs),
                semaphore, batch_idx * 2,
            ),
            query_llm_batch(
                client, sys_sweep, user_sweep, len(batch_pairs),
                semaphore, batch_idx * 2 + 1,
            ),
        )

        # Unswap preferences back to original c1/c2 ordering
        for r, s in zip(resp_fixed, swaps):
            r["preferred"] = unswap_preference(r["preferred"], s)
        for r, s in zip(resp_sweep, swaps):
            r["preferred"] = unswap_preference(r["preferred"], s)

        all_responses_fixed.extend(resp_fixed)
        all_responses_sweep.extend(resp_sweep)

    # Compute LLM disagreement and confidence
    n_valid = min(len(all_responses_fixed), len(all_responses_sweep))
    resp_f = all_responses_fixed[:n_valid]
    resp_s = all_responses_sweep[:n_valid]

    disagree_count = 0
    confidence_sums: list[float] = []

    for rf, rs in zip(resp_f, resp_s):
        if rf["preferred"] != rs["preferred"]:
            disagree_count += 1
            conf_sum = float(rf["confidence"]) + float(rs["confidence"])
            confidence_sums.append(conf_sum)

    llm_d = disagree_count / n_valid if n_valid > 0 else 0.0
    llm_conf = float(np.mean(confidence_sums)) if confidence_sums else 0.0

    flush_print(
        f"  Point {point_idx:2d}: α=({alpha_fixed},{alpha_sweep:4.1f}) "
        f"d_llm={llm_d:.3f} conf={llm_conf:.3f} "
        f"n_valid={n_valid} n_disagree={disagree_count}"
    )

    return {
        "alpha_fixed": alpha_fixed,
        "alpha_sweep": alpha_sweep,
        "llm_disagreement_rate": llm_d,
        "llm_mean_confidence_sum": llm_conf,
        "n_pairs_valid": n_valid,
        "n_disagreed": disagree_count,
        "n_api_failures": n_pairs - n_valid,
    }


async def run_all_sweep_points(
    engagement: np.ndarray,
    negativity: np.ndarray,
    configs: list[dict[str, Any]],
    n_pairs: int,
) -> list[dict[str, Any]]:
    """Run the full experiment across all sweep points."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    results = []
    for idx, cfg in enumerate(configs):
        result = await run_sweep_point(
            client,
            engagement,
            negativity,
            cfg["alpha_fixed"],
            cfg["alpha_sweep"],
            n_pairs,
            semaphore,
            idx,
        )
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Analytic baseline computation
# ---------------------------------------------------------------------------


def compute_analytic_metrics(
    content_probs: np.ndarray,
    alpha_fixed: float,
    alpha_sweep: float,
    n_pairs: int,
    seed: int,
) -> dict[str, float]:
    """Compute analytic disagreement rate and margin (no training)."""
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)

    pos_scores = np.array([
        np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)
    ])
    neg_scores = np.array([
        np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)
    ])

    utility_a = pos_scores - alpha_fixed * neg_scores
    utility_b = pos_scores - alpha_sweep * neg_scores

    disagree_count = 0
    margins_total: list[float] = []

    for _ in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        prefer_a = utility_a[c1] > utility_a[c2]
        prefer_b = utility_b[c1] > utility_b[c2]
        if prefer_a != prefer_b:
            disagree_count += 1
            ma = float(abs(utility_a[c1] - utility_a[c2]))
            mb = float(abs(utility_b[c1] - utility_b[c2]))
            margins_total.append(ma + mb)

    rate = disagree_count / n_pairs
    mean_margin = float(np.mean(margins_total)) if margins_total else 0.0

    return {
        "rate": rate,
        "mean_total_margin": mean_margin,
        "alpha_distance": abs(alpha_fixed - alpha_sweep),
    }


# ---------------------------------------------------------------------------
# Regression fitting
# ---------------------------------------------------------------------------


def fit_r2(x_matrix: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit y = x_matrix @ beta via least squares, return (R², coefficients)."""
    result, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
    y_pred = x_matrix @ result
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return r2, result


def fit_all_models(
    d_llm: np.ndarray,
    conf_llm: np.ndarray,
    d_analytic: np.ndarray,
    m_analytic: np.ndarray,
    cos_true: np.ndarray,
) -> dict[str, Any]:
    """Fit 6 regression models comparing LLM vs analytic features."""
    n = len(cos_true)
    ones = np.ones(n)
    models: dict[str, Any] = {}

    # 1. LLM disagreement only
    X = np.column_stack([ones, d_llm])
    r2, beta = fit_r2(X, cos_true)
    models["llm_d_only"] = {
        "r_squared": r2,
        "formula": f"cos = {beta[0]:.4f} + {beta[1]:.4f} * d_llm",
        "params": {"intercept": float(beta[0]), "slope": float(beta[1])},
    }

    # 2. LLM confidence only
    X = np.column_stack([ones, conf_llm])
    r2, beta = fit_r2(X, cos_true)
    models["llm_conf_only"] = {
        "r_squared": r2,
        "formula": f"cos = {beta[0]:.4f} + {beta[1]:.4f} * conf_llm",
        "params": {"intercept": float(beta[0]), "slope": float(beta[1])},
    }

    # 3. LLM 2-variable
    X = np.column_stack([ones, d_llm, conf_llm])
    r2, beta = fit_r2(X, cos_true)
    models["llm_two_variable"] = {
        "r_squared": r2,
        "formula": (
            f"cos = {beta[0]:.4f} + {beta[1]:.4f} * d_llm"
            f" + {beta[2]:.4f} * conf_llm"
        ),
        "params": {
            "intercept": float(beta[0]),
            "b_d": float(beta[1]),
            "b_conf": float(beta[2]),
        },
    }

    # 4. Hybrid: analytic d + LLM confidence
    X = np.column_stack([ones, d_analytic, conf_llm])
    r2, beta = fit_r2(X, cos_true)
    models["hybrid_analytic_d_llm_conf"] = {
        "r_squared": r2,
        "formula": (
            f"cos = {beta[0]:.4f} + {beta[1]:.4f} * d_analytic"
            f" + {beta[2]:.4f} * conf_llm"
        ),
        "params": {
            "intercept": float(beta[0]),
            "b_d": float(beta[1]),
            "b_conf": float(beta[2]),
        },
    }

    # 5. LLM product
    product = d_llm * conf_llm
    X = np.column_stack([ones, product])
    r2, beta = fit_r2(X, cos_true)
    models["llm_product"] = {
        "r_squared": r2,
        "formula": (
            f"cos = {beta[0]:.4f} + {beta[1]:.4f} * (d_llm * conf_llm)"
        ),
        "params": {"intercept": float(beta[0]), "slope": float(beta[1])},
    }

    # 6. Analytic baseline (2-variable, for comparison)
    X = np.column_stack([ones, d_analytic, m_analytic])
    r2, beta = fit_r2(X, cos_true)
    models["analytic_baseline"] = {
        "r_squared": r2,
        "formula": (
            f"cos = {beta[0]:.4f} + {beta[1]:.4f} * d_analytic"
            f" + {beta[2]:.4f} * m_analytic"
        ),
        "params": {
            "intercept": float(beta[0]),
            "b_d": float(beta[1]),
            "b_m": float(beta[2]),
        },
    }

    best = max(models, key=lambda k: models[k]["r_squared"])
    return {"models": models, "best_model": best, "best_r_squared": models[best]["r_squared"]}


# ---------------------------------------------------------------------------
# Go/no-go evaluation
# ---------------------------------------------------------------------------


def evaluate_go_no_go(
    llm_two_var_spearman: float,
    conf_margin_spearman: float,
    spearman_gap: float,
    d_correlation: float,
) -> dict[str, Any]:
    """Evaluate go/no-go criteria.

    v2: Uses Spearman rank correlation instead of R². The research question
    is whether the model correctly *ranks* stakeholder differentiation,
    not whether the relationship is linear. A monotone nonlinear relationship
    is equally useful — practitioners can always fit a calibration curve.
    """
    criteria: dict[str, Any] = {
        "llm_two_var_spearman": {
            "value": llm_two_var_spearman,
            "threshold": 0.85,
            "type": "MUST",
            "pass": llm_two_var_spearman >= 0.85,
        },
        "confidence_margin_spearman": {
            "value": conf_margin_spearman,
            "threshold": 0.70,
            "type": "SHOULD",
            "pass": conf_margin_spearman >= 0.70,
        },
        "spearman_gap": {
            "value": spearman_gap,
            "threshold": 0.10,
            "type": "NICE",
            "pass": spearman_gap <= 0.10,
        },
        "disagreement_correlation": {
            "value": d_correlation,
            "threshold": 0.90,
            "type": "MUST",
            "pass": d_correlation >= 0.90,
        },
    }
    must_pass = all(
        v["pass"] for v in criteria.values() if v["type"] == "MUST"
    )
    should_pass = all(
        v["pass"] for v in criteria.values() if v["type"] == "SHOULD"
    )
    if must_pass and should_pass:
        overall = "GO"
    elif must_pass:
        overall = "CONDITIONAL_GO"
    else:
        overall = "NO_GO"
    return {**criteria, "overall": overall}


# ---------------------------------------------------------------------------
# JSON encoder
# ---------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.bool_, np.integer)):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def lookup_ground_truth(
    existing_results: dict[str, Any],
    alpha_fixed: float,
    alpha_sweep: float,
) -> float | None:
    """Look up cosine similarity from existing sweep results."""
    sweep_key = "sweep_a" if abs(alpha_fixed - 0.3) < 0.01 else "sweep_b"
    for pt in existing_results[sweep_key]["points"]:
        if abs(pt["neg_penalty_sweep"] - alpha_sweep) < 0.01:
            return float(pt["cosine_similarity"])
    return None


def main() -> None:
    start_time = time.time()

    # Load existing results for ground truth cosines
    results_path = project_root / "results" / "disagreement_bound_analysis.json"
    if not results_path.exists():
        flush_print(f"ERROR: {results_path} not found. Run analyze_disagreement_bound.py first.")
        sys.exit(1)
    with open(results_path) as f:
        existing_results = json.load(f)

    flush_print("=" * 60)
    flush_print("LLM CONFIDENCE AS MARGIN PROXY EXPERIMENT")
    flush_print("=" * 60)
    flush_print(f"Model: {MODEL}")
    flush_print(f"Sweep points: {len(SWEEP_CONFIGS)}")
    flush_print("Pairs per point: 200")
    flush_print(f"Batch size: {BATCH_SIZE}")
    flush_print(f"Concurrency: {CONCURRENCY}")

    # Generate content pool (same as existing analysis)
    flush_print("\nGenerating content pool...")
    content_probs, _ = generate_content_pool(N_CONTENT, SEED)

    # Look up ground truth cosine similarities
    flush_print("\nLooking up ground truth cosines...")
    ground_truths: list[float] = []
    for cfg in SWEEP_CONFIGS:
        cos = lookup_ground_truth(
            existing_results, cfg["alpha_fixed"], cfg["alpha_sweep"]
        )
        if cos is None:
            flush_print(
                f"  WARNING: no ground truth for α=({cfg['alpha_fixed']}, "
                f"{cfg['alpha_sweep']})"
            )
            cos = 0.0
        ground_truths.append(cos)
        flush_print(
            f"  α=({cfg['alpha_fixed']}, {cfg['alpha_sweep']:4.1f}) → "
            f"cos={cos:.4f}"
        )

    # Compute analytic baselines
    flush_print("\nComputing analytic baselines...")
    analytic_metrics: list[dict[str, float]] = []
    for cfg in SWEEP_CONFIGS:
        am = compute_analytic_metrics(
            content_probs,
            cfg["alpha_fixed"],
            cfg["alpha_sweep"],
            n_pairs=5000,
            seed=SEED,
        )
        analytic_metrics.append(am)
        flush_print(
            f"  α=({cfg['alpha_fixed']}, {cfg['alpha_sweep']:4.1f}) → "
            f"d={am['rate']:.3f} margin={am['mean_total_margin']:.3f}"
        )

    # Compute engagement / negativity scores for LLM prompts
    engagement, negativity = compute_scaled_scores(content_probs)
    flush_print(f"\nContent scores: engagement [{engagement.min()}-{engagement.max()}], "
                f"negativity [{negativity.min()}-{negativity.max()}]")

    # Run LLM queries
    flush_print("\nRunning LLM queries...")
    n_pairs = 200
    llm_results = asyncio.run(
        run_all_sweep_points(engagement, negativity, SWEEP_CONFIGS, n_pairs)
    )

    # Assemble arrays for regression
    d_llm = np.array([r["llm_disagreement_rate"] for r in llm_results])
    conf_llm = np.array([r["llm_mean_confidence_sum"] for r in llm_results])
    d_analytic = np.array([m["rate"] for m in analytic_metrics])
    m_analytic = np.array([m["mean_total_margin"] for m in analytic_metrics])
    cos_true = np.array(ground_truths)

    # Fit models
    flush_print("\n" + "=" * 60)
    flush_print("REGRESSION RESULTS")
    flush_print("=" * 60)
    model_fits = fit_all_models(d_llm, conf_llm, d_analytic, m_analytic, cos_true)

    flush_print(f"\n{'Model':<35} {'R²':>8}")
    flush_print("-" * 45)
    for name, info in model_fits["models"].items():
        marker = " ← best" if name == model_fits["best_model"] else ""
        flush_print(f"  {name:<33} {info['r_squared']:.4f}{marker}")
        flush_print(f"    {info['formula']}")

    # Compute diagnostic correlations
    # Filter out points with zero disagreement for Spearman
    mask = conf_llm > 0
    if np.sum(mask) > 3:
        conf_margin_rho = float(spearmanr(conf_llm[mask], m_analytic[mask])[0])
    else:
        conf_margin_rho = 0.0

    d_corr = float(pearsonr(d_llm, d_analytic)[0])

    # Compute Spearman of model predictions vs true cosines
    n = len(cos_true)
    ones = np.ones(n)

    X_llm = np.column_stack([ones, d_llm, conf_llm])
    beta_llm, _, _, _ = np.linalg.lstsq(X_llm, cos_true, rcond=None)
    pred_llm = X_llm @ beta_llm
    llm_spearman = float(spearmanr(pred_llm, cos_true)[0])

    X_ana = np.column_stack([ones, d_analytic, m_analytic])
    beta_ana, _, _, _ = np.linalg.lstsq(X_ana, cos_true, rcond=None)
    pred_ana = X_ana @ beta_ana
    analytic_spearman = float(spearmanr(pred_ana, cos_true)[0])

    spearman_gap = analytic_spearman - llm_spearman

    llm_two_var_r2 = model_fits["models"]["llm_two_variable"]["r_squared"]
    analytic_r2 = model_fits["models"]["analytic_baseline"]["r_squared"]

    # Go/no-go evaluation
    flush_print("\n" + "=" * 60)
    flush_print("GO/NO-GO EVALUATION")
    flush_print("=" * 60)

    go_no_go = evaluate_go_no_go(
        llm_two_var_spearman=llm_spearman,
        conf_margin_spearman=conf_margin_rho,
        spearman_gap=spearman_gap,
        d_correlation=d_corr,
    )

    for name, info in go_no_go.items():
        if name == "overall":
            continue
        flush_print(
            f"  {name}: {info['value']:.4f} "
            f"{'≥' if name != 'spearman_gap' else '≤'} {info['threshold']} "
            f"→ {'PASS' if info['pass'] else 'FAIL'} "
            f"({info['type']})"
        )
    flush_print(f"\n  OVERALL: {go_no_go['overall']}")

    # Diagnostics
    total_api_failures = sum(r["n_api_failures"] for r in llm_results)
    wall_time = time.time() - start_time

    flush_print(f"\n  Confidence std across points: {float(np.std(conf_llm)):.4f}")
    flush_print(f"  Confidence range: [{float(np.min(conf_llm)):.3f}, {float(np.max(conf_llm)):.3f}]")
    flush_print(f"  Total API failures: {total_api_failures}")
    flush_print(f"  Wall time: {wall_time:.1f}s")

    # Save results
    output_path = project_root / "results" / "llm_margin_proxy.json"
    output = {
        "config": {
            "model": MODEL,
            "n_content": N_CONTENT,
            "n_pairs_per_point": n_pairs,
            "batch_size": BATCH_SIZE,
            "concurrency": CONCURRENCY,
            "n_sweep_points": len(SWEEP_CONFIGS),
            "seed": SEED,
        },
        "sweep_points": [
            {
                **cfg,
                **llm_results[i],
                "analytic_disagreement_rate": analytic_metrics[i]["rate"],
                "analytic_mean_margin": analytic_metrics[i]["mean_total_margin"],
                "analytic_alpha_distance": analytic_metrics[i]["alpha_distance"],
                "ground_truth_cosine": ground_truths[i],
            }
            for i, cfg in enumerate(SWEEP_CONFIGS)
        ],
        "regression_models": model_fits,
        "go_no_go": go_no_go,
        "diagnostics": {
            "llm_two_var_spearman": llm_spearman,
            "analytic_two_var_spearman": analytic_spearman,
            "spearman_gap": spearman_gap,
            "llm_two_var_r2": llm_two_var_r2,
            "analytic_two_var_r2": analytic_r2,
            "confidence_margin_spearman": float(conf_margin_rho),
            "disagreement_pearson": float(d_corr),
            "confidence_std": float(np.std(conf_llm)),
            "confidence_range": [float(np.min(conf_llm)), float(np.max(conf_llm))],
            "total_api_calls": len(SWEEP_CONFIGS) * (n_pairs // BATCH_SIZE) * 2,
            "total_api_failures": total_api_failures,
            "wall_time_seconds": wall_time,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    flush_print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
