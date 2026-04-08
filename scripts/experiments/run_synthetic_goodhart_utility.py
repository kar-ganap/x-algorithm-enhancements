"""Goodhart experiment on synthetic data with utility-based metrics.

Same design as Phase 6 MovieLens Goodhart (utility metrics, not Hausdorff)
but on the 18-action synthetic data. This enables direct comparison
with MovieLens results for the direction condition validation.

Usage:
    uv run python scripts/experiments/run_synthetic_goodhart_utility.py
"""

from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_al = _load("alternative_losses", ROOT / "enhancements" / "reward_modeling" / "alternative_losses.py")
_kf = _load("k_stakeholder_frontier", ROOT / "enhancements" / "reward_modeling" / "k_stakeholder_frontier.py")
_st = _load("movielens_stakeholders", ROOT / "enhancements" / "data" / "movielens_stakeholders.py")

LossConfig = _al.LossConfig
LossType = _al.LossType
StakeholderType = _al.StakeholderType
train_with_loss = _al.train_with_loss
NUM_ACTIONS = _al.NUM_ACTIONS
ACTION_INDICES = _al.ACTION_INDICES
POSITIVE_INDICES = _al.POSITIVE_INDICES
NEGATIVE_INDICES = _al.NEGATIVE_INDICES

compute_scorer_eval_frontier = _kf.compute_scorer_eval_frontier
split_preferences = _st.split_preferences

DIVERSITY_WEIGHTS = [round(x * 0.05, 2) for x in range(21)]
TOP_K = 10
N_SWEEP = [25, 50, 100, 200, 500, 1000, 2000]
FIXED_SIGMA = 0.3
N_SEEDS = 5
N_NOISE_SAMPLES = 5


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
# Synthetic data generation (same as verify_held_out.py)
# ---------------------------------------------------------------------------

STAKEHOLDER_UTILITY = {
    "user": lambda pos, neg: pos - neg,
    "platform": lambda pos, neg: pos - 0.3 * neg,
    "society": lambda pos, neg: pos - 4.0 * neg,
}


def generate_content_pool(n_content: int, seed: int):
    rng = np.random.default_rng(seed)
    content_probs = np.zeros((n_content, NUM_ACTIONS), dtype=np.float32)
    content_topics = rng.integers(0, 6, size=n_content)
    for i in range(n_content):
        topic = content_topics[i]
        base_probs = rng.uniform(0.05, 0.3, NUM_ACTIONS)
        if topic == 0:
            base_probs[ACTION_INDICES["favorite"]] *= 1.5
            base_probs[ACTION_INDICES["repost"]] *= 1.3
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.3
        elif topic == 1:
            base_probs[ACTION_INDICES["favorite"]] *= 1.2
            base_probs[ACTION_INDICES["reply"]] *= 1.4
        elif topic in [2, 3]:
            base_probs[ACTION_INDICES["favorite"]] *= 1.3
            base_probs[ACTION_INDICES["repost"]] *= 1.8
            base_probs[ACTION_INDICES["reply"]] *= 2.0
            base_probs[ACTION_INDICES["block_author"]] *= 2.5
            base_probs[ACTION_INDICES["report"]] *= 2.0
        elif topic == 4:
            base_probs[ACTION_INDICES["favorite"]] *= 1.8
            base_probs[ACTION_INDICES["share"]] *= 1.5
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.2
        elif topic == 5:
            base_probs[ACTION_INDICES["follow_author"]] *= 1.3
            base_probs[ACTION_INDICES["reply"]] *= 1.2
        content_probs[i] = np.clip(base_probs, 0, 1)
    return content_probs, content_topics


def generate_synthetic_preferences(content_probs, stakeholder, n_pairs, seed):
    rng = np.random.default_rng(seed)
    n_content = len(content_probs)
    utility_fn = STAKEHOLDER_UTILITY[stakeholder]
    pos_scores = np.array([np.sum(content_probs[c, POSITIVE_INDICES]) for c in range(n_content)])
    neg_scores = np.array([np.sum(content_probs[c, NEGATIVE_INDICES]) for c in range(n_content)])
    utility = utility_fn(pos_scores, neg_scores)
    probs_pref = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    probs_rej = np.zeros((n_pairs, NUM_ACTIONS), dtype=np.float32)
    for i in range(n_pairs):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        diff = utility[c1] - utility[c2]
        noise = rng.normal(0, 0.05)
        if (diff + noise) > 0:
            probs_pref[i] = content_probs[c1]
            probs_rej[i] = content_probs[c2]
        else:
            probs_pref[i] = content_probs[c2]
            probs_rej[i] = content_probs[c1]
    return probs_pref, probs_rej


def build_synthetic_stakeholder_weights():
    """Build true stakeholder weight vectors for synthetic data.

    user: pos - 1.0*neg → alpha=1.0
    platform: pos - 0.3*neg → alpha=0.3
    society: pos - 4.0*neg → alpha=4.0
    """
    alphas = {"user": 1.0, "platform": 0.3, "society": 4.0}
    weights = {}
    for name, alpha in alphas.items():
        w = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for idx in POSITIVE_INDICES:
            w[idx] = 1.0
        for idx in NEGATIVE_INDICES:
            w[idx] = -alpha
        weights[name] = w
    return weights


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def perturb_weights(w, sigma, rng):
    return w * (1 + rng.normal(0, sigma, len(w)))


def aggregate(results_list):
    agg = {}
    for key in results_list[0]:
        vals = [r[key] for r in results_list if r[key] is not None]
        if vals and isinstance(vals[0], (int, float)):
            agg[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{key}_std"] = round(float(np.std(vals)), 4)
    return agg


def main():
    print("=" * 60)
    print("Synthetic Goodhart with Utility-Based Metrics")
    print("=" * 60)

    t0 = time.time()

    # Generate content pool
    content_probs, content_topics = generate_content_pool(500, seed=42)
    base_probs = content_probs[np.newaxis, :, :]

    # Build true stakeholder weights
    true_weights = build_synthetic_stakeholder_weights()
    stakeholder_names = sorted(true_weights.keys())

    # Report cosine similarities
    print("\nStakeholder cosine similarities:")
    cos_sims = {}
    for i, s1 in enumerate(stakeholder_names):
        for s2 in stakeholder_names[i+1:]:
            cos = cosine_sim(true_weights[s1], true_weights[s2])
            cos_sims[f"{s1}-{s2}"] = round(cos, 4)
            print(f"  {s1}-{s2}: {cos:.4f}")

    # Target = user (same as MovieLens experiments)
    target = "user"
    target_weights = true_weights[target]

    print(f"\nTarget stakeholder: {target}")
    print(f"Strategy 2: More data at σ={FIXED_SIGMA}")
    print("-" * 50)

    strategy2 = {}
    for n_pairs in N_SWEEP:
        condition_results = []
        for seed_idx in range(N_SEEDS):
            for noise_idx in range(N_NOISE_SAMPLES):
                seed = 42 + seed_idx * 1000 + noise_idx * 100
                rng = np.random.default_rng(seed)

                perturbed = perturb_weights(target_weights, FIXED_SIGMA, rng)
                pref, rej = generate_synthetic_preferences(content_probs, target, n_pairs, seed)
                tp, tr, ep, er = split_preferences(pref, rej, 0.2, seed)

                config = LossConfig(
                    loss_type=LossType.BRADLEY_TERRY,
                    stakeholder=StakeholderType.USER,
                    learning_rate=0.01, num_epochs=50, batch_size=64,
                )
                model = train_with_loss(config, tp, tr, verbose=False,
                                        eval_probs_preferred=ep, eval_probs_rejected=er)

                # Use learned weights as scorer, true weights as evaluator
                frontier = compute_scorer_eval_frontier(
                    base_probs, content_topics, model.weights, true_weights,
                    DIVERSITY_WEIGHTS, top_k=TOP_K,
                )

                # Extract mean utilities
                result = {}
                for s in stakeholder_names:
                    key = f"{s}_utility"
                    result[key] = float(np.mean([p[key] for p in frontier]))
                result["eval_accuracy"] = float(model.eval_accuracy) if model.eval_accuracy else None
                condition_results.append(result)

        agg = aggregate(condition_results)
        utils_str = ", ".join(f"{s}={agg.get(f'{s}_utility_mean', 0):.4f}" for s in stakeholder_names)
        print(f"  N={n_pairs:>4}: {utils_str}")
        strategy2[str(n_pairs)] = agg

    elapsed = time.time() - t0

    # Check direction condition
    print(f"\nDirection condition validation:")
    for hidden in stakeholder_names:
        if hidden == target:
            continue
        cos_key = f"{target}-{hidden}" if f"{target}-{hidden}" in cos_sims else f"{hidden}-{target}"
        cos_val = cos_sims.get(cos_key, 0)
        utilities = [strategy2[str(n)].get(f"{hidden}_utility_mean", 0) for n in N_SWEEP]
        improving = utilities[-1] > utilities[0]
        prediction = "IMPROVE" if cos_val > 0 else "DEGRADE"
        actual = "improving" if improving else "degrading"
        match = (cos_val > 0 and improving) or (cos_val < 0 and not improving)
        status = "✓" if match else "✗"
        peak_idx = int(np.argmax(utilities))
        peak_n = N_SWEEP[peak_idx]
        degradation = (utilities[peak_idx] - utilities[-1]) / abs(utilities[peak_idx]) if abs(utilities[peak_idx]) > 1e-6 else 0
        print(f"  {status} {hidden}: cos={cos_val:+.4f}, predict={prediction}, actual={actual}, degradation={degradation:+.1%}")

    results = {
        "config": {
            "dataset": "synthetic",
            "feature_dim": NUM_ACTIONS,
            "n_content": 500,
            "target": target,
            "fixed_sigma": FIXED_SIGMA,
            "n_sweep": N_SWEEP,
            "n_seeds": N_SEEDS,
            "n_noise_samples": N_NOISE_SAMPLES,
        },
        "cosine_similarities": cos_sims,
        "strategy2_more_data": strategy2,
        "total_time_seconds": round(elapsed, 1),
    }

    out_path = ROOT / "results" / "synthetic_goodhart_utility.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_path}")
    print(f"Complete in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
