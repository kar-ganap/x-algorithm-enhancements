"""Analyze stakeholder utilities for the existing reward model.

This script evaluates our two-stage pluralistic model through three lenses:
1. User Utility: engagement - discomfort
2. Platform Utility: total engagement + retention
3. Society Utility: diversity - polarization

It also computes the Pareto frontier showing tradeoffs between objectives.
"""

import importlib.util
import json
import os
import sys

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)


# Direct module loading to avoid __init__.py's full import chain
def load_module_direct(module_name: str, file_path: str):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Create package structure
import types

enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg

reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [os.path.join(project_root, "enhancements/reward_modeling")]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

# Load modules
weights_path = os.path.join(project_root, "enhancements/reward_modeling/weights.py")
weights_mod = load_module_direct("enhancements.reward_modeling.weights", weights_path)
ACTION_INDICES = weights_mod.ACTION_INDICES
NUM_ACTIONS = weights_mod.NUM_ACTIONS
RewardWeights = weights_mod.RewardWeights

pluralistic_path = os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py")
load_module_direct("enhancements.reward_modeling.pluralistic", pluralistic_path)

two_stage_path = os.path.join(project_root, "enhancements/reward_modeling/two_stage.py")
two_stage_mod = load_module_direct("enhancements.reward_modeling.two_stage", two_stage_path)
TwoStageConfig = two_stage_mod.TwoStageConfig
train_two_stage = two_stage_mod.train_two_stage
compute_reward = two_stage_mod.compute_reward

stakeholder_path = os.path.join(project_root, "enhancements/reward_modeling/stakeholder_utilities.py")
stakeholder_mod = load_module_direct("enhancements.reward_modeling.stakeholder_utilities", stakeholder_path)
compute_user_utility = stakeholder_mod.compute_user_utility
compute_platform_utility = stakeholder_mod.compute_platform_utility
compute_society_utility = stakeholder_mod.compute_society_utility
compute_pareto_frontier = stakeholder_mod.compute_pareto_frontier
analyze_stakeholder_tradeoffs = stakeholder_mod.analyze_stakeholder_tradeoffs
ParetoPoint = stakeholder_mod.ParetoPoint


# =============================================================================
# Data Generation
# =============================================================================

def generate_synthetic_data(
    num_users: int = 600,
    num_content: int = 100,
    num_topics: int = 6,
    seed: int = 42,
) -> dict:
    """Generate synthetic data with clear archetype structure.

    Archetypes:
        0: sports_fan - high engagement with sports
        1: tech_bro - high engagement with tech
        2: political_L - engages L, blocks R
        3: political_R - engages R, blocks L
        4: lurker - passive, favorites only
        5: power_user - high engagement everywhere

    Topics:
        0: sports
        1: tech
        2: politics_L
        3: politics_R
        4: entertainment
        5: news
    """
    rng = np.random.default_rng(seed)

    # Assign archetypes evenly
    user_archetypes = np.array([i % num_topics for i in range(num_users)])

    # Create user histories (rich features: topic × action)
    user_histories = np.zeros((num_users, num_topics * NUM_ACTIONS), dtype=np.float32)

    for i in range(num_users):
        archetype = user_archetypes[i]
        history = np.zeros((num_topics, NUM_ACTIONS))

        if archetype == 0:  # sports_fan
            history[0, ACTION_INDICES["favorite"]] = 0.7
            history[0, ACTION_INDICES["repost"]] = 0.5
            history[0, ACTION_INDICES["dwell"]] = 0.8
        elif archetype == 1:  # tech_bro
            history[1, ACTION_INDICES["favorite"]] = 0.7
            history[1, ACTION_INDICES["repost"]] = 0.4
            history[1, ACTION_INDICES["share"]] = 0.5
        elif archetype == 2:  # political_L
            history[2, ACTION_INDICES["favorite"]] = 0.6
            history[2, ACTION_INDICES["repost"]] = 0.5
            history[3, ACTION_INDICES["block_author"]] = 0.4  # Blocks R
            history[3, ACTION_INDICES["not_interested"]] = 0.5
        elif archetype == 3:  # political_R
            history[3, ACTION_INDICES["favorite"]] = 0.6
            history[3, ACTION_INDICES["repost"]] = 0.5
            history[2, ACTION_INDICES["block_author"]] = 0.4  # Blocks L
            history[2, ACTION_INDICES["not_interested"]] = 0.5
        elif archetype == 4:  # lurker
            for t in range(num_topics):
                history[t, ACTION_INDICES["favorite"]] = 0.2
                history[t, ACTION_INDICES["dwell"]] = 0.4
        elif archetype == 5:  # power_user
            for t in range(num_topics):
                history[t, ACTION_INDICES["favorite"]] = 0.5
                history[t, ACTION_INDICES["repost"]] = 0.4
                history[t, ACTION_INDICES["reply"]] = 0.3

        # Add noise
        history += rng.random(history.shape) * 0.1
        user_histories[i] = history.flatten()

    # Create content with topic labels
    content_topics = np.array([i % num_topics for i in range(num_content)])

    # Create action probabilities for each content
    # These represent predicted engagement probabilities
    content_action_probs = np.zeros((num_content, NUM_ACTIONS), dtype=np.float32)

    for i in range(num_content):
        topic = content_topics[i]

        # Base engagement varies by topic
        if topic == 0:  # sports - high engagement
            content_action_probs[i, ACTION_INDICES["favorite"]] = 0.5 + rng.random() * 0.3
            content_action_probs[i, ACTION_INDICES["repost"]] = 0.3 + rng.random() * 0.2
        elif topic == 1:  # tech - moderate engagement
            content_action_probs[i, ACTION_INDICES["favorite"]] = 0.4 + rng.random() * 0.3
            content_action_probs[i, ACTION_INDICES["share"]] = 0.3 + rng.random() * 0.2
        elif topic in [2, 3]:  # politics - high engagement but also negative
            content_action_probs[i, ACTION_INDICES["favorite"]] = 0.4 + rng.random() * 0.3
            content_action_probs[i, ACTION_INDICES["repost"]] = 0.4 + rng.random() * 0.3
            content_action_probs[i, ACTION_INDICES["reply"]] = 0.3 + rng.random() * 0.2
            # Political content has higher negative signals
            content_action_probs[i, ACTION_INDICES["block_author"]] = 0.1 + rng.random() * 0.1
            content_action_probs[i, ACTION_INDICES["not_interested"]] = 0.15 + rng.random() * 0.1
        elif topic == 4:  # entertainment - moderate
            content_action_probs[i, ACTION_INDICES["favorite"]] = 0.4 + rng.random() * 0.2
            content_action_probs[i, ACTION_INDICES["dwell"]] = 0.5 + rng.random() * 0.2
        else:  # news - lower engagement
            content_action_probs[i, ACTION_INDICES["click"]] = 0.4 + rng.random() * 0.2
            content_action_probs[i, ACTION_INDICES["dwell"]] = 0.3 + rng.random() * 0.2

        # Add small baseline for all actions
        content_action_probs[i] += rng.random(NUM_ACTIONS) * 0.05

    # Create preference pairs for training
    probs_preferred = []
    probs_rejected = []

    for i in range(num_users):
        archetype = user_archetypes[i]
        preferred_topic = archetype if archetype < num_topics else 0

        # Find content matching user's preferred topic
        matching = np.where(content_topics == preferred_topic)[0]
        non_matching = np.where(content_topics != preferred_topic)[0]

        if len(matching) > 0 and len(non_matching) > 0:
            pref_idx = rng.choice(matching)
            rej_idx = rng.choice(non_matching)
            probs_preferred.append(content_action_probs[pref_idx])
            probs_rejected.append(content_action_probs[rej_idx])

    probs_preferred = np.array(probs_preferred)
    probs_rejected = np.array(probs_rejected)

    return {
        "user_histories": user_histories,
        "user_archetypes": user_archetypes,
        "content_action_probs": content_action_probs,
        "content_topics": content_topics,
        "probs_preferred": probs_preferred,
        "probs_rejected": probs_rejected,
        "num_topics": num_topics,
        "num_users": num_users,
        "num_content": num_content,
    }


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_current_model(
    data: dict,
    verbose: bool = True,
) -> dict:
    """Analyze current model's impact on each stakeholder.

    Args:
        data: Synthetic data dict
        verbose: Print progress

    Returns:
        Analysis results
    """
    if verbose:
        print("=" * 70)
        print("STAKEHOLDER UTILITY ANALYSIS")
        print("=" * 70)

    # Analyze user utility across all content
    if verbose:
        print("\n[1/3] Analyzing User Utility...")

    user_results = []
    for i in range(data["num_users"]):
        # Get action probs for content matching user's archetype
        archetype = data["user_archetypes"][i]
        matching_content = np.where(data["content_topics"] == archetype)[0]

        if len(matching_content) > 0:
            probs = data["content_action_probs"][matching_content]
            result = compute_user_utility(probs)
            user_results.append(result)

    avg_user_utility = np.mean([r.total_utility for r in user_results])
    avg_positive = np.mean([r.positive_component for r in user_results])
    avg_negative = np.mean([r.negative_component for r in user_results])

    if verbose:
        print(f"  Average User Utility: {avg_user_utility:.4f}")
        print(f"    Positive component: {avg_positive:.4f}")
        print(f"    Negative component: {avg_negative:.4f}")

    # Analyze platform utility
    if verbose:
        print("\n[2/3] Analyzing Platform Utility...")

    platform_result = compute_platform_utility(data["content_action_probs"])

    if verbose:
        print(f"  Platform Utility: {platform_result.total_utility:.4f}")
        print(f"    Engagement score: {platform_result.engagement_score:.4f}")
        print(f"    Retention proxy: {platform_result.retention_proxy:.4f}")

    # Analyze society utility
    if verbose:
        print("\n[3/3] Analyzing Society Utility...")

    # Simulate recommendations: top-10 for each user based on engagement
    recommendations = []
    for i in range(data["num_users"]):
        scores = (
            data["content_action_probs"][:, ACTION_INDICES["favorite"]] +
            data["content_action_probs"][:, ACTION_INDICES["repost"]] * 0.8
        )
        top_k = np.argsort(scores)[-10:][::-1]
        recommendations.append(top_k)

    recommendations = np.array(recommendations)

    society_result = compute_society_utility(
        recommendations,
        data["user_archetypes"],
        data["content_topics"],
        data["num_topics"],
    )

    if verbose:
        print(f"  Society Utility: {society_result.total_utility:.4f}")
        print(f"    Diversity score: {society_result.diversity_score:.4f}")
        print(f"    Polarization score: {society_result.polarization_score:.4f}")
        print(f"    Cross-exposure rate: {society_result.cross_exposure_rate:.4f}")

    return {
        "user": {
            "total_utility": avg_user_utility,
            "positive_component": avg_positive,
            "negative_component": avg_negative,
        },
        "platform": {
            "total_utility": platform_result.total_utility,
            "engagement_score": platform_result.engagement_score,
            "retention_proxy": platform_result.retention_proxy,
        },
        "society": {
            "total_utility": society_result.total_utility,
            "diversity_score": society_result.diversity_score,
            "polarization_score": society_result.polarization_score,
            "cross_exposure_rate": society_result.cross_exposure_rate,
        },
    }


def compute_and_analyze_pareto(
    data: dict,
    verbose: bool = True,
) -> dict:
    """Compute Pareto frontier and analyze tradeoffs.

    Args:
        data: Synthetic data dict
        verbose: Print progress

    Returns:
        Pareto analysis results
    """
    if verbose:
        print("\n" + "=" * 70)
        print("PARETO FRONTIER ANALYSIS")
        print("=" * 70)

    # Create user-content action probability matrix [N, M, num_actions]
    N = data["num_users"]
    M = data["num_content"]

    # For simplicity, assume same content probs for all users
    # In reality, this would come from the model
    base_action_probs = np.tile(
        data["content_action_probs"][np.newaxis, :, :],
        (N, 1, 1)
    )

    # Adjust based on user-content topic match
    for user_idx in range(N):
        user_archetype = data["user_archetypes"][user_idx]
        for content_idx in range(M):
            content_topic = data["content_topics"][content_idx]

            # Boost engagement for matching topics
            if content_topic == user_archetype:
                base_action_probs[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 1.5
                base_action_probs[user_idx, content_idx, ACTION_INDICES["repost"]] *= 1.3

            # Political users: reduce engagement with opposing content
            if user_archetype == 2 and content_topic == 3:  # L seeing R
                base_action_probs[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 0.3
                base_action_probs[user_idx, content_idx, ACTION_INDICES["block_author"]] *= 2.0
            elif user_archetype == 3 and content_topic == 2:  # R seeing L
                base_action_probs[user_idx, content_idx, ACTION_INDICES["favorite"]] *= 0.3
                base_action_probs[user_idx, content_idx, ACTION_INDICES["block_author"]] *= 2.0

    # Clip to valid probability range
    base_action_probs = np.clip(base_action_probs, 0, 1)

    # Compute Pareto frontier
    diversity_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    if verbose:
        print(f"\nComputing frontier across {len(diversity_weights)} diversity weights...")

    points = compute_pareto_frontier(
        base_action_probs,
        diversity_weights,
        data["user_archetypes"],
        data["content_topics"],
        top_k=10,
    )

    # Analyze tradeoffs
    analysis = analyze_stakeholder_tradeoffs(points)

    if verbose:
        print("\n" + "-" * 60)
        print("FRONTIER POINTS")
        print("-" * 60)
        print(f"{'Div Weight':<12} {'User':<10} {'Platform':<10} {'Society':<10} {'Pareto?':<8}")
        print("-" * 60)
        for p in points:
            pareto = "Yes" if p.is_pareto_optimal else "No"
            print(f"{p.diversity_weight:<12.1f} {p.user_utility:<10.4f} {p.platform_utility:<10.4f} {p.society_utility:<10.4f} {pareto:<8}")

        print("\n" + "-" * 60)
        print("KEY FINDINGS")
        print("-" * 60)

        max_user = analysis["max_user_utility"]["point"]
        max_platform = analysis["max_platform_utility"]["point"]
        max_society = analysis["max_society_utility"]["point"]
        balanced = analysis["balanced"]["point"]

        print(f"\nMax User Utility (div_weight={max_user.diversity_weight}):")
        print(f"  User: {max_user.user_utility:.4f}, Platform: {max_user.platform_utility:.4f}, Society: {max_user.society_utility:.4f}")

        print(f"\nMax Platform Utility (div_weight={max_platform.diversity_weight}):")
        print(f"  User: {max_platform.user_utility:.4f}, Platform: {max_platform.platform_utility:.4f}, Society: {max_platform.society_utility:.4f}")

        print(f"\nMax Society Utility (div_weight={max_society.diversity_weight}):")
        print(f"  User: {max_society.user_utility:.4f}, Platform: {max_society.platform_utility:.4f}, Society: {max_society.society_utility:.4f}")

        print(f"\nBalanced Point (div_weight={balanced.diversity_weight}):")
        print(f"  User: {balanced.user_utility:.4f}, Platform: {balanced.platform_utility:.4f}, Society: {balanced.society_utility:.4f}")

        # Compute tradeoff ratios
        user_range = analysis["user_range"]
        society_range = analysis["society_range"]

        if user_range[1] - user_range[0] > 0 and society_range[1] - society_range[0] > 0:
            user_cost = (max_user.user_utility - max_society.user_utility) / (user_range[1] - user_range[0])
            society_gain = (max_society.society_utility - max_user.society_utility) / (society_range[1] - society_range[0])

            print("\n" + "-" * 60)
            print("TRADEOFF ANALYSIS")
            print("-" * 60)
            print("Moving from max-engagement to max-diversity:")
            print(f"  User utility change: {max_society.user_utility - max_user.user_utility:+.4f} ({user_cost*100:+.1f}% of range)")
            print(f"  Society utility change: {max_society.society_utility - max_user.society_utility:+.4f} ({society_gain*100:+.1f}% of range)")

    # Convert points to serializable format
    points_data = [
        {
            "diversity_weight": p.diversity_weight,
            "user_utility": p.user_utility,
            "platform_utility": p.platform_utility,
            "society_utility": p.society_utility,
            "is_pareto_optimal": p.is_pareto_optimal,
        }
        for p in points
    ]

    return {
        "points": points_data,
        "analysis": {
            "max_user_utility": {
                "diversity_weight": analysis["max_user_utility"]["point"].diversity_weight,
                "utilities": {
                    "user": analysis["max_user_utility"]["point"].user_utility,
                    "platform": analysis["max_user_utility"]["point"].platform_utility,
                    "society": analysis["max_user_utility"]["point"].society_utility,
                },
            },
            "max_platform_utility": {
                "diversity_weight": analysis["max_platform_utility"]["point"].diversity_weight,
                "utilities": {
                    "user": analysis["max_platform_utility"]["point"].user_utility,
                    "platform": analysis["max_platform_utility"]["point"].platform_utility,
                    "society": analysis["max_platform_utility"]["point"].society_utility,
                },
            },
            "max_society_utility": {
                "diversity_weight": analysis["max_society_utility"]["point"].diversity_weight,
                "utilities": {
                    "user": analysis["max_society_utility"]["point"].user_utility,
                    "platform": analysis["max_society_utility"]["point"].platform_utility,
                    "society": analysis["max_society_utility"]["point"].society_utility,
                },
            },
            "balanced": {
                "diversity_weight": analysis["balanced"]["point"].diversity_weight,
                "utilities": {
                    "user": analysis["balanced"]["point"].user_utility,
                    "platform": analysis["balanced"]["point"].platform_utility,
                    "society": analysis["balanced"]["point"].society_utility,
                },
            },
            "num_pareto_optimal": analysis["num_pareto_optimal"],
        },
    }


def main():
    """Run full stakeholder analysis."""
    print("=" * 70)
    print("MULTI-STAKEHOLDER UTILITY ANALYSIS")
    print("F4 Phase 4: Analyzing tradeoffs between User, Platform, and Society")
    print("=" * 70)

    # Generate data
    print("\nGenerating synthetic data...")
    data = generate_synthetic_data(
        num_users=600,
        num_content=100,
        num_topics=6,
        seed=42,
    )
    print(f"  Users: {data['num_users']}")
    print(f"  Content items: {data['num_content']}")
    print(f"  Topics: {data['num_topics']}")

    # Analyze current model
    current_analysis = analyze_current_model(data, verbose=True)

    # Compute Pareto frontier
    pareto_analysis = compute_and_analyze_pareto(data, verbose=True)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nCurrent Model Utilities:")
    print(f"  User:     {current_analysis['user']['total_utility']:.4f}")
    print(f"  Platform: {current_analysis['platform']['total_utility']:.4f}")
    print(f"  Society:  {current_analysis['society']['total_utility']:.4f}")

    print("\nKey Insight:")
    balanced = pareto_analysis["analysis"]["balanced"]
    max_eng = pareto_analysis["analysis"]["max_platform_utility"]

    society_gain = balanced["utilities"]["society"] - max_eng["utilities"]["society"]
    platform_cost = max_eng["utilities"]["platform"] - balanced["utilities"]["platform"]

    if platform_cost > 0:
        print(f"  A {society_gain:.2f} improvement in society utility")
        print(f"  costs only {platform_cost:.2f} in platform utility")
        print(f"  ({platform_cost/max_eng['utilities']['platform']*100:.1f}% reduction)")

    # Save results
    output_dir = "results/f4_phase4_stakeholder"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "current_model": current_analysis,
        "pareto_frontier": pareto_analysis,
    }

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output = convert_to_serializable(output)

    output_path = os.path.join(output_dir, "stakeholder_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
