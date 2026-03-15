#!/usr/bin/env python3
"""Run alternative loss function experiments.

NOTE: Running with unbuffered output for progress monitoring.

This script:
1. Generates training data
2. Runs all loss function experiments (or a subset)
3. Evaluates each trained model
4. Saves results incrementally
5. Generates comparison summary

Usage:
    # Quick sanity test (4 experiments, 10 epochs each)
    uv run python scripts/run_loss_experiments.py --quick-test

    # Run all experiments
    uv run python scripts/run_loss_experiments.py --all

    # Run specific loss type
    uv run python scripts/run_loss_experiments.py --loss-type margin_bt

    # Resume from checkpoint
    uv run python scripts/run_loss_experiments.py --all --resume
"""

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Direct Module Loading (bypasses __init__.py to avoid grok dependency)
# =============================================================================

def load_module(module_name: str, module_path: Path, register_as: str = None):
    """Load a Python module directly from file path.

    Args:
        module_name: Name to use for the module
        module_path: Path to the .py file
        register_as: Optional package path to also register the module under
    """
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    if register_as:
        sys.modules[register_as] = module
    spec.loader.exec_module(module)
    return module


# Load alternative_losses first and register under package path
# so experiment_config's import resolves without triggering __init__.py
alternative_losses = load_module(
    "alternative_losses",
    project_root / "enhancements" / "reward_modeling" / "alternative_losses.py",
    register_as="enhancements.reward_modeling.alternative_losses"
)

# Now load experiment_config - its import will find the already-loaded module
experiment_config = load_module(
    "experiment_config",
    project_root / "enhancements" / "reward_modeling" / "experiment_config.py"
)

# Import from loaded modules
LossType = alternative_losses.LossType
StakeholderType = alternative_losses.StakeholderType
LossConfig = alternative_losses.LossConfig
TrainedModel = alternative_losses.TrainedModel
train_with_loss = alternative_losses.train_with_loss
PostHocReranker = alternative_losses.PostHocReranker
compute_weight_similarity = alternative_losses.compute_weight_similarity
compute_ranking_correlation = alternative_losses.compute_ranking_correlation
POSITIVE_INDICES = alternative_losses.POSITIVE_INDICES
NEGATIVE_INDICES = alternative_losses.NEGATIVE_INDICES
NUM_ACTIONS = alternative_losses.NUM_ACTIONS
ACTION_INDICES = alternative_losses.ACTION_INDICES

ExperimentConfig = experiment_config.ExperimentConfig
generate_all_configs = experiment_config.generate_all_configs
generate_quick_test_configs = experiment_config.generate_quick_test_configs
get_config_name = experiment_config.get_config_name
get_rerank_configs = experiment_config.get_rerank_configs
count_experiments = experiment_config.count_experiments
STAKEHOLDERS = experiment_config.STAKEHOLDERS


# =============================================================================
# Data Generation
# =============================================================================

TOPIC_NAMES = ["sports", "tech", "politics_L", "politics_R", "entertainment", "news"]
ARCHETYPE_NAMES = ["casual", "power_user", "political_L", "political_R", "explorer", "passive"]


def generate_content_pool(n_content: int, seed: int) -> tuple:
    """Generate content pool with topic-specific characteristics.

    Returns (content_probs, content_topics).
    """
    rng = np.random.default_rng(seed)

    content_probs = np.zeros((n_content, NUM_ACTIONS), dtype=np.float32)
    content_topics = rng.integers(0, len(TOPIC_NAMES), size=n_content)

    for i in range(n_content):
        topic = content_topics[i]
        base_probs = rng.uniform(0.05, 0.3, NUM_ACTIONS)

        # Topic-specific engagement patterns
        if topic == 0:  # sports
            base_probs[ACTION_INDICES["favorite"]] *= 1.5
            base_probs[ACTION_INDICES["repost"]] *= 1.3
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.3
        elif topic == 1:  # tech
            base_probs[ACTION_INDICES["favorite"]] *= 1.2
            base_probs[ACTION_INDICES["reply"]] *= 1.4
        elif topic in [2, 3]:  # politics - high engagement AND high negative
            base_probs[ACTION_INDICES["favorite"]] *= 1.3
            base_probs[ACTION_INDICES["repost"]] *= 1.8
            base_probs[ACTION_INDICES["reply"]] *= 2.0
            base_probs[ACTION_INDICES["block_author"]] *= 2.5
            base_probs[ACTION_INDICES["report"]] *= 2.0
        elif topic == 4:  # entertainment
            base_probs[ACTION_INDICES["favorite"]] *= 1.8
            base_probs[ACTION_INDICES["share"]] *= 1.5
            for idx in NEGATIVE_INDICES:
                base_probs[idx] *= 0.2
        elif topic == 5:  # news
            base_probs[ACTION_INDICES["follow_author"]] *= 1.3
            base_probs[ACTION_INDICES["reply"]] *= 1.2

        content_probs[i] = np.clip(base_probs, 0, 1)

    return content_probs, content_topics


# Stakeholder-specific utility functions
# These determine which content each stakeholder prefers
STAKEHOLDER_UTILITY = {
    # User: balanced - positive engagement minus negative signals
    StakeholderType.USER: lambda pos, neg: pos - neg,

    # Platform: engagement-focused - tolerates some negativity for higher engagement
    StakeholderType.PLATFORM: lambda pos, neg: pos - 0.3 * neg,

    # Society: harm-averse - heavily penalizes divisive/harmful content
    StakeholderType.SOCIETY: lambda pos, neg: pos - 4.0 * neg,
}


def generate_training_data(
    n_samples: int = 5000,
    n_content: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Generate synthetic preference data for all stakeholders.

    All stakeholders see the SAME content pairs but may disagree on which
    is preferred based on their different utility functions. This creates
    genuine preference disagreement that enables model differentiation.
    """
    # Generate shared content pool
    content_probs, content_topics = generate_content_pool(n_content, seed)
    rng = np.random.default_rng(seed)

    # Generate shared content pair indices (same pairs for all stakeholders)
    pair_c1 = np.zeros(n_samples, dtype=np.int32)
    pair_c2 = np.zeros(n_samples, dtype=np.int32)
    for i in range(n_samples):
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        pair_c1[i] = c1
        pair_c2[i] = c2

    # Pre-compute positive and negative scores for all content pairs
    pos_scores_c1 = np.array([np.sum(content_probs[c, POSITIVE_INDICES]) for c in pair_c1])
    neg_scores_c1 = np.array([np.sum(content_probs[c, NEGATIVE_INDICES]) for c in pair_c1])
    pos_scores_c2 = np.array([np.sum(content_probs[c, POSITIVE_INDICES]) for c in pair_c2])
    neg_scores_c2 = np.array([np.sum(content_probs[c, NEGATIVE_INDICES]) for c in pair_c2])

    # Small noise for tie-breaking (same noise across stakeholders)
    noise1 = rng.normal(0, 0.05, n_samples)
    noise2 = rng.normal(0, 0.05, n_samples)

    # Generate per-stakeholder preference labels
    stakeholder_data = {}
    for stakeholder in StakeholderType:
        utility_fn = STAKEHOLDER_UTILITY[stakeholder]

        # Stakeholder-specific utility for each content item
        score1 = utility_fn(pos_scores_c1, neg_scores_c1) + noise1
        score2 = utility_fn(pos_scores_c2, neg_scores_c2) + noise2

        prefer_c1 = score1 > score2

        probs_preferred = np.zeros((n_samples, NUM_ACTIONS), dtype=np.float32)
        probs_rejected = np.zeros((n_samples, NUM_ACTIONS), dtype=np.float32)
        sample_topics = np.zeros(n_samples, dtype=np.int32)
        engagement_pref = np.zeros(n_samples, dtype=np.float32)
        engagement_rej = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            c1, c2 = pair_c1[i], pair_c2[i]
            if prefer_c1[i]:
                probs_preferred[i] = content_probs[c1]
                probs_rejected[i] = content_probs[c2]
                sample_topics[i] = content_topics[c1]
                engagement_pref[i] = float(np.clip((pos_scores_c1[i] / 5) + 0.3, 0, 1))
                engagement_rej[i] = float(np.clip((pos_scores_c2[i] / 5) + 0.3, 0, 1))
            else:
                probs_preferred[i] = content_probs[c2]
                probs_rejected[i] = content_probs[c1]
                sample_topics[i] = content_topics[c2]
                engagement_pref[i] = float(np.clip((pos_scores_c2[i] / 5) + 0.3, 0, 1))
                engagement_rej[i] = float(np.clip((pos_scores_c1[i] / 5) + 0.3, 0, 1))

        stakeholder_data[stakeholder] = {
            "probs_preferred": probs_preferred,
            "probs_rejected": probs_rejected,
            "sample_topics": sample_topics,
            "engagement_pref": engagement_pref,
            "engagement_rej": engagement_rej,
        }

    # Compute label disagreement: on how many pairs do stakeholders disagree?
    user_prefers_c1 = STAKEHOLDER_UTILITY[StakeholderType.USER](pos_scores_c1, neg_scores_c1) + noise1 > \
                      STAKEHOLDER_UTILITY[StakeholderType.USER](pos_scores_c2, neg_scores_c2) + noise2
    platform_prefers_c1 = STAKEHOLDER_UTILITY[StakeholderType.PLATFORM](pos_scores_c1, neg_scores_c1) + noise1 > \
                          STAKEHOLDER_UTILITY[StakeholderType.PLATFORM](pos_scores_c2, neg_scores_c2) + noise2
    society_prefers_c1 = STAKEHOLDER_UTILITY[StakeholderType.SOCIETY](pos_scores_c1, neg_scores_c1) + noise1 > \
                         STAKEHOLDER_UTILITY[StakeholderType.SOCIETY](pos_scores_c2, neg_scores_c2) + noise2

    user_platform_agree = np.mean(user_prefers_c1 == platform_prefers_c1)
    user_society_agree = np.mean(user_prefers_c1 == society_prefers_c1)
    platform_society_agree = np.mean(platform_prefers_c1 == society_prefers_c1)

    print("  Label agreement rates (same pair, different utility):")
    print(f"    User-Platform: {user_platform_agree:.1%}")
    print(f"    User-Society: {user_society_agree:.1%}")
    print(f"    Platform-Society: {platform_society_agree:.1%}")
    print("  Disagreement rates:")
    print(f"    User-Platform: {1-user_platform_agree:.1%}")
    print(f"    User-Society: {1-user_society_agree:.1%}")
    print(f"    Platform-Society: {1-platform_society_agree:.1%}")

    return {
        "content_probs": content_probs,
        "content_topics": content_topics,
        "stakeholder_data": stakeholder_data,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: TrainedModel,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a trained model.

    Returns metrics compatible with the 8-test framework.
    """
    weights = model.weights

    # Basic metrics
    metrics = {
        "loss_type": model.loss_type.value,
        "stakeholder": model.stakeholder.value,
        "accuracy": model.accuracy,
        "final_loss": model.loss_history[-1] if model.loss_history else None,
    }

    # Weight analysis
    metrics["weights"] = {
        "positive_mean": float(np.mean(weights[POSITIVE_INDICES])),
        "negative_mean": float(np.mean(weights[NEGATIVE_INDICES])),
        "weight_std": float(np.std(weights)),
        "weight_range": float(np.max(weights) - np.min(weights)),
    }

    # Scores on content
    scores = content_probs @ weights
    metrics["score_stats"] = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }

    # Per-topic average scores
    topic_scores = {}
    for topic_idx, topic_name in enumerate(TOPIC_NAMES):
        mask = content_topics == topic_idx
        if np.any(mask):
            topic_scores[topic_name] = float(np.mean(scores[mask]))
    metrics["topic_scores"] = topic_scores

    return metrics


def evaluate_posthoc_reranker(
    reranker: PostHocReranker,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    alpha: float,
    stakeholder: StakeholderType,
) -> dict[str, Any]:
    """Evaluate post-hoc reranker."""
    scores = reranker.score(content_probs, stakeholder, alpha)

    metrics = {
        "approach": "posthoc",
        "stakeholder": stakeholder.value,
        "alpha": alpha,
    }

    # Score stats
    metrics["score_stats"] = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }

    # Per-topic scores
    topic_scores = {}
    for topic_idx, topic_name in enumerate(TOPIC_NAMES):
        mask = content_topics == topic_idx
        if np.any(mask):
            topic_scores[topic_name] = float(np.mean(scores[mask]))
    metrics["topic_scores"] = topic_scores

    return metrics


def compare_stakeholder_models(
    models: dict[StakeholderType, TrainedModel],
    content_probs: np.ndarray,
) -> dict[str, Any]:
    """Compare models across stakeholders."""
    comparison = {}

    # Weight similarities
    comparison["cosine_similarities"] = compute_weight_similarity(models)

    # Ranking correlations
    comparison["ranking_correlations"] = compute_ranking_correlation(models, content_probs)

    # Check success criteria
    min_cos_sim = min(comparison["cosine_similarities"].values())
    min_tau = min(comparison["ranking_correlations"].values())

    comparison["success"] = {
        "cos_sim_below_095": min_cos_sim < 0.95,
        "tau_below_080": min_tau < 0.80,
        "min_cos_sim": min_cos_sim,
        "min_tau": min_tau,
    }

    return comparison


# =============================================================================
# Result Management
# =============================================================================

def save_result(result: dict, output_dir: Path, name: str):
    """Save individual experiment result."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{name}.json"

    # Convert numpy types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj

    with open(filepath, "w") as f:
        json.dump(convert(result), f, indent=2)


def load_completed_experiments(output_dir: Path) -> set:
    """Load names of already completed experiments."""
    completed = set()
    if output_dir.exists():
        for f in output_dir.glob("*.json"):
            if f.stem not in ["summary", "best_configs", "comparison"]:
                completed.add(f.stem)
    return completed


def generate_summary(output_dir: Path) -> dict:
    """Generate summary of all experiments."""
    results = {}

    for f in output_dir.glob("*.json"):
        if f.stem in ["summary", "best_configs", "comparison"]:
            continue
        with open(f) as fp:
            results[f.stem] = json.load(fp)

    # Group by loss type
    by_loss_type = {}
    for name, result in results.items():
        loss_type = result.get("loss_type", "posthoc")
        if loss_type not in by_loss_type:
            by_loss_type[loss_type] = []
        by_loss_type[loss_type].append({
            "name": name,
            "accuracy": result.get("accuracy"),
            "stakeholder": result.get("stakeholder"),
            **result.get("weights", {}),
        })

    summary = {
        "total_experiments": len(results),
        "by_loss_type": by_loss_type,
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# =============================================================================
# Main Runner
# =============================================================================

def run_experiments(
    configs: list[LossConfig],
    data: dict[str, Any],
    output_dir: Path,
    resume: bool = False,
    verbose: bool = True,
) -> dict[str, TrainedModel]:
    """Run all training experiments.

    Each experiment uses the stakeholder-specific preference data
    from data["stakeholder_data"][stakeholder].
    """
    completed = load_completed_experiments(output_dir) if resume else set()
    models = {}

    print(f"\nRunning {len(configs)} training experiments")
    print(f"Output: {output_dir}")
    if resume and completed:
        print(f"Resuming - {len(completed)} already completed")
    print("-" * 60)

    for i, config in enumerate(configs):
        name = get_config_name(config)

        if name in completed:
            if verbose:
                print(f"[{i+1}/{len(configs)}] Skipping {name} (completed)")
            continue

        if verbose:
            print(f"[{i+1}/{len(configs)}] Training {name}...", flush=True)

        start_time = time.time()

        # Get stakeholder-specific training data
        s_data = data["stakeholder_data"][config.stakeholder]

        # Train model
        model = train_with_loss(
            config,
            s_data["probs_preferred"],
            s_data["probs_rejected"],
            target_engagement_pref=s_data["engagement_pref"],
            target_engagement_rej=s_data["engagement_rej"],
            verbose=False,
        )

        elapsed = time.time() - start_time

        # Evaluate
        metrics = evaluate_model(
            model,
            data["content_probs"],
            data["content_topics"],
        )

        # Add config info
        metrics["config"] = {
            "margin": config.margin,
            "calibration_weight": config.calibration_weight,
            "constraint_weight": config.constraint_weight,
        }
        metrics["training_time_seconds"] = elapsed
        metrics["weights_vector"] = model.weights.tolist()

        # Save incrementally
        save_result(metrics, output_dir, name)

        models[name] = model

        if verbose:
            print(f"    Accuracy: {model.accuracy:.1%}, Time: {elapsed:.1f}s", flush=True)

    return models


def run_posthoc_experiments(
    base_model: TrainedModel,
    data: dict[str, np.ndarray],
    output_dir: Path,
    verbose: bool = True,
) -> dict[str, dict]:
    """Run post-hoc reranking experiments."""
    reranker = PostHocReranker(base_model.weights)
    rerank_configs = get_rerank_configs()
    results = {}

    print(f"\nRunning {len(rerank_configs)} post-hoc reranking experiments")
    print("-" * 60)

    for i, config in enumerate(rerank_configs):
        name = config["name"]

        if verbose:
            print(f"[{i+1}/{len(rerank_configs)}] Evaluating {name}...")

        metrics = evaluate_posthoc_reranker(
            reranker,
            data["content_probs"],
            data["content_topics"],
            config["alpha"],
            config["stakeholder"],
        )

        save_result(metrics, output_dir, name)
        results[name] = metrics

    return results


def run_comparison(
    models: dict[str, TrainedModel],
    data: dict[str, np.ndarray],
    output_dir: Path,
):
    """Compare stakeholder models for each loss type."""
    # Group models by loss type and hyperparams
    grouped = {}
    for name, model in models.items():
        # Extract loss type and hyperparams from name
        parts = name.split("_")
        loss_type = "_".join(parts[:2]) if parts[1] in ["bt"] else parts[0]

        if loss_type not in grouped:
            grouped[loss_type] = {}
        grouped[loss_type][model.stakeholder] = model

    comparisons = {}
    print("\n" + "=" * 60)
    print("STAKEHOLDER COMPARISONS")
    print("=" * 60)

    for loss_type, stakeholder_models in grouped.items():
        if len(stakeholder_models) < 2:
            continue

        comparison = compare_stakeholder_models(
            stakeholder_models,
            data["content_probs"],
        )

        comparisons[loss_type] = comparison

        success = comparison["success"]
        print(f"\n{loss_type}:")
        print(f"  Min cosine sim: {success['min_cos_sim']:.4f} (< 0.95: {'PASS' if success['cos_sim_below_095'] else 'FAIL'})")
        print(f"  Min Kendall τ:  {success['min_tau']:.4f} (< 0.80: {'PASS' if success['tau_below_080'] else 'FAIL'})")

    # Save comparisons
    with open(output_dir / "comparison.json", "w") as f:
        json.dump(comparisons, f, indent=2)

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Run alternative loss experiments")
    parser.add_argument("--quick-test", action="store_true", help="Quick sanity test (4 experiments)")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--loss-type", type=str, help="Run only specific loss type")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/loss_experiments",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print experiment counts
    counts = count_experiments()
    print("=" * 60)
    print("ALTERNATIVE LOSS FUNCTION EXPERIMENTS")
    print("=" * 60)
    print(f"Total training experiments: {counts['total_training']}")
    print(f"Total evaluation-only: {counts['total_evaluation']}")

    # Generate training data
    print("\nGenerating training data...")
    base_config = ExperimentConfig()
    data = generate_training_data(
        n_samples=base_config.n_training_samples,
        n_content=base_config.n_content_items,
        seed=base_config.random_seed,
    )
    print(f"  Training pairs per stakeholder: {base_config.n_training_samples}")
    print(f"  Content pool: {len(data['content_probs'])}")

    # Generate configs
    if args.quick_test:
        configs = generate_quick_test_configs(base_config)
        print(f"\nQuick test mode: {len(configs)} experiments")
    elif args.loss_type:
        all_configs = generate_all_configs(base_config)
        configs = [c for c in all_configs if c.loss_type.value == args.loss_type]
        print(f"\nFiltered to {len(configs)} {args.loss_type} experiments")
    else:
        configs = generate_all_configs(base_config)
        print(f"\nFull experiment mode: {len(configs)} experiments")

    # Run training experiments
    models = run_experiments(
        configs,
        data,
        output_dir,
        resume=args.resume,
        verbose=True,
    )

    # Run post-hoc experiments if we have a baseline
    baseline_model = None
    for name, model in models.items():
        if model.loss_type == LossType.BRADLEY_TERRY:
            baseline_model = model
            break

    if baseline_model:
        run_posthoc_experiments(baseline_model, data, output_dir)

    # Run comparisons
    if len(models) > 3:
        run_comparison(models, data, output_dir)

    # Generate summary
    summary = generate_summary(output_dir)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments: {summary['total_experiments']}")
    for loss_type, exps in summary["by_loss_type"].items():
        print(f"  {loss_type}: {len(exps)} experiments")

    print(f"\nResults saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
