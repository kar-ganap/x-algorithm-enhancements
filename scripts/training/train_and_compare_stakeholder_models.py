"""Train and compare per-stakeholder reward models.

Trains three models optimized for different stakeholders:
1. User-Optimized: engagement - discomfort
2. Platform-Optimized: total engagement
3. Society-Optimized: diversity - polarization

Then runs all comparison tests from the Phase 4b test plan.
"""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Direct module loading to avoid grok dependency issues
project_root = Path(__file__).parent.parent


def load_module(module_name: str, module_path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Load required modules
weights_module = load_module(
    "weights",
    project_root / "enhancements" / "reward_modeling" / "weights.py"
)
stakeholder_module = load_module(
    "stakeholder_models",
    project_root / "enhancements" / "reward_modeling" / "stakeholder_models.py"
)
stakeholder_utils = load_module(
    "stakeholder_utilities",
    project_root / "enhancements" / "reward_modeling" / "stakeholder_utilities.py"
)
causal_module = load_module(
    "causal_verification",
    project_root / "enhancements" / "reward_modeling" / "causal_verification.py"
)

# Import needed classes
RewardWeights = weights_module.RewardWeights
ACTION_NAMES = weights_module.ACTION_NAMES
ACTION_INDICES = weights_module.ACTION_INDICES

StakeholderType = stakeholder_module.StakeholderType
StakeholderTrainingConfig = stakeholder_module.StakeholderTrainingConfig
train_all_stakeholder_models = stakeholder_module.train_all_stakeholder_models
compare_weights = stakeholder_module.compare_weights
compute_ranking_correlation = stakeholder_module.compute_ranking_correlation
identify_contested_content = stakeholder_module.identify_contested_content
compute_cross_exposure = stakeholder_module.compute_cross_exposure

compute_user_utility = stakeholder_utils.compute_user_utility
compute_platform_utility = stakeholder_utils.compute_platform_utility
compute_society_utility = stakeholder_utils.compute_society_utility


# =============================================================================
# Data Generation
# =============================================================================

TOPIC_NAMES = ["sports", "tech", "politics_L", "politics_R", "entertainment", "news"]
ARCHETYPE_NAMES = ["casual", "power_user", "political_L", "political_R", "explorer", "passive"]


def generate_training_data(
    n_samples: int = 5000,
    n_users: int = 100,
    n_content: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic preference pairs with topics and archetypes.

    Returns:
        probs_preferred: [n_samples, num_actions]
        probs_rejected: [n_samples, num_actions]
        content_topics: [n_samples] topic of preferred content
        user_archetypes: [n_samples] archetype of user
        content_probs: [n_content, num_actions] action probs per content
        content_topic_array: [n_content] topic per content item
    """
    rng = np.random.default_rng(seed)
    num_actions = len(ACTION_NAMES)

    # Generate content pool with topic-specific characteristics
    content_probs = np.zeros((n_content, num_actions), dtype=np.float32)
    content_topics = rng.integers(0, len(TOPIC_NAMES), size=n_content)

    for i in range(n_content):
        topic = content_topics[i]
        base_probs = rng.uniform(0.05, 0.3, num_actions)

        # Topic-specific patterns
        if topic == 0:  # sports - high engagement, low controversy
            base_probs[ACTION_INDICES["favorite"]] *= 1.5
            base_probs[ACTION_INDICES["repost"]] *= 1.3
            base_probs[ACTION_INDICES["block_author"]] *= 0.3
            base_probs[ACTION_INDICES["report"]] *= 0.2
        elif topic == 1:  # tech - moderate engagement
            base_probs[ACTION_INDICES["favorite"]] *= 1.2
            base_probs[ACTION_INDICES["reply"]] *= 1.4
        elif topic in [2, 3]:  # politics - high engagement but also high negative
            base_probs[ACTION_INDICES["favorite"]] *= 1.3
            base_probs[ACTION_INDICES["repost"]] *= 1.8  # Viral
            base_probs[ACTION_INDICES["reply"]] *= 2.0
            base_probs[ACTION_INDICES["block_author"]] *= 2.5
            base_probs[ACTION_INDICES["report"]] *= 2.0
            base_probs[ACTION_INDICES["not_interested"]] *= 1.5
        elif topic == 4:  # entertainment - high positive, low negative
            base_probs[ACTION_INDICES["favorite"]] *= 1.8
            base_probs[ACTION_INDICES["share"]] *= 1.5  # bookmark not in actions, use share
            base_probs[ACTION_INDICES["block_author"]] *= 0.2
        elif topic == 5:  # news - moderate all around
            base_probs[ACTION_INDICES["follow_author"]] *= 1.3
            base_probs[ACTION_INDICES["reply"]] *= 1.2

        content_probs[i] = np.clip(base_probs, 0, 1)

    # Generate users with archetype-specific preferences
    user_archetypes_pool = rng.integers(0, len(ARCHETYPE_NAMES), size=n_users)

    # Generate preference pairs
    probs_preferred = np.zeros((n_samples, num_actions), dtype=np.float32)
    probs_rejected = np.zeros((n_samples, num_actions), dtype=np.float32)
    sample_topics = np.zeros(n_samples, dtype=np.int32)
    sample_archetypes = np.zeros(n_samples, dtype=np.int32)

    for i in range(n_samples):
        user_idx = i % n_users
        archetype = user_archetypes_pool[user_idx]

        # Pick two content items
        c1, c2 = rng.choice(n_content, size=2, replace=False)
        topic1, topic2 = content_topics[c1], content_topics[c2]

        # Determine preference based on archetype
        score1, score2 = compute_preference_scores(
            archetype, topic1, topic2,
            content_probs[c1], content_probs[c2],
            rng
        )

        if score1 > score2:
            probs_preferred[i] = content_probs[c1]
            probs_rejected[i] = content_probs[c2]
            sample_topics[i] = topic1
        else:
            probs_preferred[i] = content_probs[c2]
            probs_rejected[i] = content_probs[c1]
            sample_topics[i] = topic2

        sample_archetypes[i] = archetype

    return (
        probs_preferred, probs_rejected,
        sample_topics, sample_archetypes,
        content_probs, content_topics
    )


def compute_preference_scores(
    archetype: int,
    topic1: int, topic2: int,
    probs1: np.ndarray, probs2: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Compute preference scores based on user archetype."""
    base_score1 = np.sum(probs1)
    base_score2 = np.sum(probs2)

    # Archetype-specific topic preferences
    if archetype == 0:  # casual - prefers entertainment
        if topic1 == 4: base_score1 *= 1.5
        if topic2 == 4: base_score2 *= 1.5
    elif archetype == 1:  # power_user - prefers tech
        if topic1 == 1: base_score1 *= 1.5
        if topic2 == 1: base_score2 *= 1.5
    elif archetype == 2:  # political_L - prefers politics_L
        if topic1 == 2: base_score1 *= 2.0
        if topic2 == 2: base_score2 *= 2.0
        if topic1 == 3: base_score1 *= 0.5  # Dislikes politics_R
        if topic2 == 3: base_score2 *= 0.5
    elif archetype == 3:  # political_R - prefers politics_R
        if topic1 == 3: base_score1 *= 2.0
        if topic2 == 3: base_score2 *= 2.0
        if topic1 == 2: base_score1 *= 0.5  # Dislikes politics_L
        if topic2 == 2: base_score2 *= 0.5
    elif archetype == 4:  # explorer - prefers variety
        pass  # No strong preferences
    elif archetype == 5:  # passive - prefers easy content
        if topic1 == 4: base_score1 *= 1.3
        if topic2 == 4: base_score2 *= 1.3

    # Add noise
    base_score1 += rng.normal(0, 0.1)
    base_score2 += rng.normal(0, 0.1)

    return base_score1, base_score2


# =============================================================================
# Test Implementations
# =============================================================================

def test_1_weight_comparison(models: dict) -> dict[str, Any]:
    """Test 1: Compare learned weights across models."""
    print("\n" + "=" * 60)
    print("TEST 1: LEARNED WEIGHT COMPARISON")
    print("=" * 60)

    weight_comparison = compare_weights(models)

    # Key actions to highlight
    key_actions = ["favorite", "repost", "block_author", "report", "follow_author"]

    print("\n" + "-" * 60)
    print(f"{'Action':<20} {'User':>12} {'Platform':>12} {'Society':>12}")
    print("-" * 60)

    results = {}
    for action in key_actions:
        user_w = weight_comparison[action]["user"]
        platform_w = weight_comparison[action]["platform"]
        society_w = weight_comparison[action]["society"]

        print(f"{action:<20} {user_w:>12.4f} {platform_w:>12.4f} {society_w:>12.4f}")
        results[action] = {"user": user_w, "platform": platform_w, "society": society_w}

    # Compute weight divergence (cosine similarity)
    user_weights = models[StakeholderType.USER].weights
    platform_weights = models[StakeholderType.PLATFORM].weights
    society_weights = models[StakeholderType.SOCIETY].weights

    def cosine_sim(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    up_sim = cosine_sim(user_weights, platform_weights)
    us_sim = cosine_sim(user_weights, society_weights)
    ps_sim = cosine_sim(platform_weights, society_weights)

    print("\n" + "-" * 60)
    print("Weight Cosine Similarity:")
    print(f"  User-Platform: {up_sim:.3f}")
    print(f"  User-Society:  {us_sim:.3f}")
    print(f"  Platform-Society: {ps_sim:.3f}")

    # Success: cosine sim < 0.9 indicates meaningful divergence
    success = min(up_sim, us_sim, ps_sim) < 0.95

    print(f"\nSuccess (cosine sim < 0.95): {'PASS' if success else 'FAIL'}")

    return {
        "weights": results,
        "cosine_similarities": {
            "user_platform": up_sim,
            "user_society": us_sim,
            "platform_society": ps_sim,
        },
        "success": success,
    }


def test_2_ranking_correlation(
    models: dict,
    content_probs: np.ndarray,
) -> dict[str, Any]:
    """Test 2: Compute ranking correlation between models."""
    print("\n" + "=" * 60)
    print("TEST 2: RANKING CORRELATION ANALYSIS")
    print("=" * 60)

    correlations = compute_ranking_correlation(models, content_probs)

    print("\nKendall's τ Correlation Matrix:")
    print("-" * 40)
    print(f"{'':>15} {'User':>10} {'Platform':>10} {'Society':>10}")
    print("-" * 40)

    # Build full matrix
    types = ["user", "platform", "society"]
    matrix = {t1: {t2: 1.0 if t1 == t2 else 0.0 for t2 in types} for t1 in types}

    for (t1, t2), tau in correlations.items():
        matrix[t1][t2] = tau
        matrix[t2][t1] = tau

    for t1 in types:
        row = f"{t1:>15}"
        for t2 in types:
            row += f" {matrix[t1][t2]:>10.3f}"
        print(row)

    # Success criteria
    user_platform = correlations.get(("user", "platform"), matrix["user"]["platform"])
    platform_society = correlations.get(("platform", "society"), matrix["platform"]["society"])

    up_pass = user_platform > 0.5  # Moderate overlap expected
    ps_pass = platform_society < 0.8  # Meaningful divergence

    print(f"\nUser-Platform τ > 0.5: {'PASS' if up_pass else 'FAIL'} ({user_platform:.3f})")
    print(f"Platform-Society τ < 0.8: {'PASS' if ps_pass else 'FAIL'} ({platform_society:.3f})")

    return {
        "correlations": correlations,
        "matrix": matrix,
        "success": up_pass and ps_pass,
    }


def test_3_contested_content(
    models: dict,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
) -> dict[str, Any]:
    """Test 3: Identify most contested content."""
    print("\n" + "=" * 60)
    print("TEST 3: CONTESTED CONTENT IDENTIFICATION")
    print("=" * 60)

    contested = identify_contested_content(
        models, content_probs, content_topics, top_k=10
    )

    print("\nTop 10 Most Contested Content Items:")
    print("-" * 70)
    print(f"{'Idx':>5} {'Topic':<15} {'Variance':>10} {'User Rank':>10} {'Plat Rank':>10} {'Soc Rank':>10}")
    print("-" * 70)

    topic_counts = {}
    for item in contested:
        topic = item["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

        print(f"{item['content_idx']:>5} {topic:<15} {item['rank_variance']:>10.1f} "
              f"{item['ranks']['user']:>10} {item['ranks']['platform']:>10} {item['ranks']['society']:>10}")

    print("\nContested content by topic:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    # Success: identified at least 10 contested items
    success = len(contested) >= 10

    print(f"\nSuccess (≥10 items): {'PASS' if success else 'FAIL'}")

    return {
        "contested_items": contested,
        "topic_distribution": topic_counts,
        "success": success,
    }


def test_4_content_type_analysis(
    models: dict,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
) -> dict[str, Any]:
    """Test 4: Analyze how models rank different content types."""
    print("\n" + "=" * 60)
    print("TEST 4: CONTENT TYPE ANALYSIS")
    print("=" * 60)

    # Compute scores and ranks for each model
    model_ranks = {}
    for stype, model in models.items():
        scores = content_probs @ model.weights
        # Percentile rank (0-100, higher = better)
        ranks = (np.argsort(np.argsort(scores)) / len(scores)) * 100
        model_ranks[stype.value] = ranks

    # Compute average rank per topic
    print("\nAverage Percentile Rank by Topic (higher = ranked better):")
    print("-" * 60)
    print(f"{'Topic':<15} {'User':>12} {'Platform':>12} {'Society':>12}")
    print("-" * 60)

    results = {}
    political_user = []
    political_society = []

    for topic_idx, topic_name in enumerate(TOPIC_NAMES):
        mask = content_topics == topic_idx
        if not np.any(mask):
            continue

        user_avg = float(np.mean(model_ranks["user"][mask]))
        platform_avg = float(np.mean(model_ranks["platform"][mask]))
        society_avg = float(np.mean(model_ranks["society"][mask]))

        print(f"{topic_name:<15} {user_avg:>12.1f} {platform_avg:>12.1f} {society_avg:>12.1f}")

        results[topic_name] = {
            "user": user_avg,
            "platform": platform_avg,
            "society": society_avg,
        }

        if topic_name in ["politics_L", "politics_R"]:
            political_user.append(user_avg)
            political_society.append(society_avg)

    # Success: Society model ranks political content lower than user
    if political_user and political_society:
        avg_political_user = np.mean(political_user)
        avg_political_society = np.mean(political_society)
        politics_success = avg_political_society < avg_political_user
    else:
        politics_success = True  # No political content to compare

    print(f"\nSociety ranks politics lower than User: {'PASS' if politics_success else 'FAIL'}")

    return {
        "rank_by_topic": results,
        "success": politics_success,
    }


def test_5_cross_exposure(
    models: dict,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    n_political_users: int = 50,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Test 5: Measure cross-partisan exposure for political users."""
    print("\n" + "=" * 60)
    print("TEST 5: CROSS-PARTISAN EXPOSURE ANALYSIS")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    cross_exposure = {}

    for stype, model in models.items():
        scores = content_probs @ model.weights

        total_cross = 0
        total_opportunities = 0

        # Simulate political users
        for archetype in [2, 3]:  # political_L, political_R
            for _ in range(n_political_users):
                # Add user-specific noise to simulate personalization
                user_scores = scores + rng.normal(0, 0.05, len(scores))

                # Get top-K recommendations
                top_indices = np.argsort(-user_scores)[:top_k]
                top_topics = content_topics[top_indices]

                # Count cross-exposure
                opposing_topic = 3 if archetype == 2 else 2
                cross = np.sum(top_topics == opposing_topic)

                total_cross += cross
                total_opportunities += top_k

        cross_exposure[stype.value] = total_cross / total_opportunities if total_opportunities > 0 else 0

    print("\nCross-Partisan Exposure Rate:")
    print("-" * 40)
    for model_name in ["user", "platform", "society"]:
        rate = cross_exposure[model_name]
        print(f"  {model_name.capitalize():<10}: {rate:.1%}")

    # Success: Society has higher cross-exposure than Platform
    society_rate = cross_exposure["society"]
    platform_rate = cross_exposure["platform"]
    success = society_rate > platform_rate

    gap = society_rate - platform_rate
    print(f"\nSociety-Platform gap: {gap:.1%}")
    print(f"Society > Platform: {'PASS' if success else 'FAIL'}")

    return {
        "cross_exposure": cross_exposure,
        "success": success,
    }


def test_6_win_win_content(
    models: dict,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    top_k: int = 20,
) -> dict[str, Any]:
    """Test 6: Identify content that all stakeholders agree is good."""
    print("\n" + "=" * 60)
    print("TEST 6: WIN-WIN CONTENT IDENTIFICATION")
    print("=" * 60)

    # Get top-K for each model
    top_sets = {}
    for stype, model in models.items():
        scores = content_probs @ model.weights
        top_indices = set(np.argsort(-scores)[:top_k])
        top_sets[stype.value] = top_indices

    # Find intersection
    win_win = top_sets["user"] & top_sets["platform"] & top_sets["society"]

    print(f"\nContent in top-{top_k} for ALL models: {len(win_win)} items")

    # Analyze characteristics
    win_win_topics = {}
    for idx in win_win:
        topic = TOPIC_NAMES[content_topics[idx]] if content_topics[idx] < len(TOPIC_NAMES) else f"topic_{content_topics[idx]}"
        win_win_topics[topic] = win_win_topics.get(topic, 0) + 1

    print("\nWin-win content by topic:")
    for topic, count in sorted(win_win_topics.items(), key=lambda x: -x[1]):
        print(f"  {topic}: {count}")

    # Analyze action patterns of win-win content
    if win_win:
        win_win_probs = content_probs[list(win_win)]
        mean_probs = np.mean(win_win_probs, axis=0)

        print("\nAverage action probs for win-win content:")
        key_actions = ["favorite", "repost", "block_author", "report"]
        for action in key_actions:
            idx = ACTION_INDICES[action]
            print(f"  {action}: {mean_probs[idx]:.3f}")

    # Success: Found at least some win-win content
    success = len(win_win) >= 3

    print(f"\nSuccess (≥3 win-win items): {'PASS' if success else 'FAIL'}")

    return {
        "win_win_count": len(win_win),
        "win_win_indices": list(win_win),
        "topic_distribution": win_win_topics,
        "success": success,
    }


def test_7_policy_simulation(
    models: dict,
    content_probs: np.ndarray,
    content_topics: np.ndarray,
    n_users: int = 100,
    top_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    """Test 7: Simulate policy switch from Platform to Society model."""
    print("\n" + "=" * 60)
    print("TEST 7: POLICY SIMULATION")
    print("=" * 60)
    print("Scenario: What if platform switched from Platform-Optimized to Society-Optimized?")

    rng = np.random.default_rng(seed)

    platform_model = models[StakeholderType.PLATFORM]
    society_model = models[StakeholderType.SOCIETY]

    platform_scores = content_probs @ platform_model.weights
    society_scores = content_probs @ society_model.weights

    # Track metrics before/after
    total_recs_changed = 0
    total_recs = 0

    platform_utility_before = []
    platform_utility_after = []
    society_utility_before = []
    society_utility_after = []

    for _ in range(n_users):
        # User-specific scores
        user_platform = platform_scores + rng.normal(0, 0.05, len(platform_scores))
        user_society = society_scores + rng.normal(0, 0.05, len(society_scores))

        # Top-K before (Platform model)
        top_before = set(np.argsort(-user_platform)[:top_k])

        # Top-K after (Society model)
        top_after = set(np.argsort(-user_society)[:top_k])

        # Count changes
        total_recs_changed += len(top_before - top_after)
        total_recs += top_k

        # Compute utilities for this user
        before_probs = content_probs[list(top_before)]
        after_probs = content_probs[list(top_after)]

        # Platform utility: total engagement
        platform_utility_before.append(np.sum(before_probs))
        platform_utility_after.append(np.sum(after_probs))

        # Society utility: diversity (topic variety in recommendations)
        before_topics = [content_topics[i] for i in top_before]
        after_topics = [content_topics[i] for i in top_after]
        society_utility_before.append(len(set(before_topics)))
        society_utility_after.append(len(set(after_topics)))

    pct_changed = total_recs_changed / total_recs * 100
    platform_change = (np.mean(platform_utility_after) - np.mean(platform_utility_before)) / np.mean(platform_utility_before) * 100
    society_change = (np.mean(society_utility_after) - np.mean(society_utility_before)) / np.mean(society_utility_before) * 100

    print("\nPolicy Switch Impact:")
    print("-" * 40)
    print(f"  Recommendations changed: {pct_changed:.1f}%")
    print(f"  Platform utility change: {platform_change:+.1f}%")
    print(f"  Society utility change:  {society_change:+.1f}%")

    # Success: Can quantify meaningful impact
    success = pct_changed > 10  # At least 10% of recs change

    print(f"\nSuccess (>10% recs change): {'PASS' if success else 'FAIL'}")

    return {
        "recommendations_changed_pct": pct_changed,
        "platform_utility_change_pct": platform_change,
        "society_utility_change_pct": society_change,
        "success": success,
    }


def test_8_causal_verification(
    models: dict,
) -> dict[str, Any]:
    """Test 8: Run causal verification tests on all models."""
    print("\n" + "=" * 60)
    print("TEST 8: CAUSAL VERIFICATION")
    print("=" * 60)

    CausalTestConfig = causal_module.CausalTestConfig
    run_block_intervention_test = causal_module.run_block_intervention_test
    run_follow_intervention_test = causal_module.run_follow_intervention_test
    create_reward_fn_from_weights = causal_module.create_reward_fn_from_weights

    config = CausalTestConfig(
        num_samples=100,
        min_effect_size=0.05,
        min_pass_rate=0.85,
    )

    results = {}
    all_pass = True

    for stype, model in models.items():
        print(f"\n{stype.value.upper()} Model:")
        print("-" * 40)

        # Create reward function from learned weights
        reward_fn = create_reward_fn_from_weights(model.weights)

        # Generate test action probs
        rng = np.random.default_rng(42)
        test_probs = rng.uniform(0.1, 0.5, (100, len(model.weights))).astype(np.float32)
        test_histories = np.zeros((100, 108), dtype=np.float32)  # Dummy histories

        # Run key tests
        block_result = run_block_intervention_test(
            reward_fn=reward_fn,
            action_probs_batch=test_probs,
            user_histories=test_histories,
            config=config,
        )
        follow_result = run_follow_intervention_test(
            reward_fn=reward_fn,
            action_probs_batch=test_probs,
            user_histories=test_histories,
            config=config,
        )

        block_pass = block_result.passed
        follow_pass = follow_result.passed

        print(f"  Block intervention: {'PASS' if block_pass else 'FAIL'} (effect: {block_result.mean_effect_size:.2f})")
        print(f"  Follow intervention: {'PASS' if follow_pass else 'FAIL'} (effect: {follow_result.mean_effect_size:.2f})")

        results[stype.value] = {
            "block_passed": block_pass,
            "block_effect": float(block_result.mean_effect_size),
            "follow_passed": follow_pass,
            "follow_effect": float(follow_result.mean_effect_size),
        }

        if not (block_pass and follow_pass):
            all_pass = False

    print(f"\nAll models pass causal tests: {'PASS' if all_pass else 'FAIL'}")

    return {
        "model_results": results,
        "success": all_pass,
    }


# =============================================================================
# Main Runner
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        # Convert tuple keys to strings for JSON compatibility
        return {
            (str(k) if isinstance(k, tuple) else k): convert_to_serializable(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        # Convert tuples to lists for JSON
        return [convert_to_serializable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def main():
    """Train stakeholder models and run all comparison tests."""
    print("=" * 70)
    print("PHASE 4b: PER-STAKEHOLDER MODEL TRAINING AND COMPARISON")
    print("=" * 70)

    # Generate training data
    print("\n[1/10] Generating training data...")
    (
        probs_preferred, probs_rejected,
        sample_topics, sample_archetypes,
        content_probs, content_topics
    ) = generate_training_data(n_samples=5000, n_content=500)

    print(f"  Training pairs: {len(probs_preferred)}")
    print(f"  Content pool: {len(content_probs)}")

    # Train all models
    # Use stronger penalty weights to create meaningful differentiation
    print("\n[2/10] Training stakeholder models...")
    config = StakeholderTrainingConfig(
        stakeholder=StakeholderType.USER,
        learning_rate=0.01,
        num_epochs=150,  # More epochs for convergence
        user_discomfort_weight=10.0,  # Strong penalty for negative signals
        society_diversity_weight=5.0,  # Strong diversity bonus
        society_polarization_weight=10.0,  # Strong penalty for echo chambers
    )

    models = train_all_stakeholder_models(
        probs_preferred, probs_rejected,
        sample_topics, sample_archetypes,
        config, verbose=True
    )

    # Run all tests
    all_results = {}

    print("\n[3/10] Running Test 1: Weight Comparison")
    all_results["test_1_weights"] = test_1_weight_comparison(models)

    print("\n[4/10] Running Test 2: Ranking Correlation")
    all_results["test_2_correlation"] = test_2_ranking_correlation(models, content_probs)

    print("\n[5/10] Running Test 3: Contested Content")
    all_results["test_3_contested"] = test_3_contested_content(models, content_probs, content_topics)

    print("\n[6/10] Running Test 4: Content Type Analysis")
    all_results["test_4_content_type"] = test_4_content_type_analysis(models, content_probs, content_topics)

    print("\n[7/10] Running Test 5: Cross-Exposure")
    all_results["test_5_cross_exposure"] = test_5_cross_exposure(
        models, content_probs, content_topics
    )

    print("\n[8/10] Running Test 6: Win-Win Content")
    all_results["test_6_win_win"] = test_6_win_win_content(models, content_probs, content_topics)

    print("\n[9/10] Running Test 7: Policy Simulation")
    all_results["test_7_policy"] = test_7_policy_simulation(models, content_probs, content_topics)

    print("\n[10/10] Running Test 8: Causal Verification")
    all_results["test_8_causal"] = test_8_causal_verification(models)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: TEST RESULTS")
    print("=" * 70)

    tests_passed = 0
    total_tests = 8

    test_names = [
        ("test_1_weights", "Weight Comparison"),
        ("test_2_correlation", "Ranking Correlation"),
        ("test_3_contested", "Contested Content"),
        ("test_4_content_type", "Content Type Analysis"),
        ("test_5_cross_exposure", "Cross-Partisan Exposure"),
        ("test_6_win_win", "Win-Win Content"),
        ("test_7_policy", "Policy Simulation"),
        ("test_8_causal", "Causal Verification"),
    ]

    for test_key, test_name in test_names:
        passed = all_results[test_key]["success"]
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<30}: {status}")
        if passed:
            tests_passed += 1

    print("-" * 70)
    print(f"  Total: {tests_passed}/{total_tests} tests passed")
    print("=" * 70)

    # Save results
    output_path = project_root / "outputs" / "stakeholder_comparison_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_results = convert_to_serializable(all_results)

    # Add model weights to results
    serializable_results["model_weights"] = {
        stype.value: model.weights.tolist()
        for stype, model in models.items()
    }
    serializable_results["model_accuracy"] = {
        stype.value: model.training_accuracy
        for stype, model in models.items()
    }

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
