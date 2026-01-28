"""Test causal verification framework with two-stage reward model.

Validates that our reward model captures causal relationships:
1. Block intervention → score decreases
2. Follow intervention → score increases
3. Matching history → higher score than mismatched history

From design doc F4 Tier 1B: Causal Verification
"""

import importlib.util
import json
import os
import sys

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# Direct module loading to avoid __init__.py's full import chain
def load_module_direct(module_name: str, file_path: str):
    """Load a module directly from file path, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Create package structure without triggering __init__.py
import types
enhancements_pkg = types.ModuleType("enhancements")
enhancements_pkg.__path__ = [os.path.join(project_root, "enhancements")]
sys.modules["enhancements"] = enhancements_pkg

reward_modeling_pkg = types.ModuleType("enhancements.reward_modeling")
reward_modeling_pkg.__path__ = [os.path.join(project_root, "enhancements/reward_modeling")]
sys.modules["enhancements.reward_modeling"] = reward_modeling_pkg

# Load weights module first (no external deps)
weights_path = os.path.join(project_root, "enhancements/reward_modeling/weights.py")
weights_mod = load_module_direct("enhancements.reward_modeling.weights", weights_path)
ACTION_INDICES = weights_mod.ACTION_INDICES
NUM_ACTIONS = weights_mod.NUM_ACTIONS
RewardWeights = weights_mod.RewardWeights

# Load pluralistic module (needed by two_stage)
pluralistic_path = os.path.join(project_root, "enhancements/reward_modeling/pluralistic.py")
pluralistic_mod = load_module_direct("enhancements.reward_modeling.pluralistic", pluralistic_path)

# Load two_stage module
two_stage_path = os.path.join(project_root, "enhancements/reward_modeling/two_stage.py")
two_stage_mod = load_module_direct("enhancements.reward_modeling.two_stage", two_stage_path)
TwoStageConfig = two_stage_mod.TwoStageConfig
train_two_stage = two_stage_mod.train_two_stage

# Load causal_verification module
causal_path = os.path.join(project_root, "enhancements/reward_modeling/causal_verification.py")
causal_mod = load_module_direct("enhancements.reward_modeling.causal_verification", causal_path)
CausalTestConfig = causal_mod.CausalTestConfig
CausalVerificationSuite = causal_mod.CausalVerificationSuite
create_reward_fn_from_two_stage = causal_mod.create_reward_fn_from_two_stage
create_reward_fn_from_weights = causal_mod.create_reward_fn_from_weights


def generate_synthetic_data(
    num_users: int = 1000,
    num_topics: int = 6,
    seed: int = 42,
) -> dict:
    """Generate synthetic data for causal verification testing.

    Creates users with topic-specific preferences and content with
    topic-specific action probabilities.

    Args:
        num_users: Number of users
        num_topics: Number of content topics
        seed: Random seed

    Returns:
        Dict with user_histories, action_probs, topics, etc.
    """
    rng = np.random.default_rng(seed)

    # User histories: rich features [num_users, num_topics * num_actions]
    # Each user has strong engagement on their primary topic
    user_histories = np.zeros((num_users, num_topics * NUM_ACTIONS), dtype=np.float32)
    user_topics = np.zeros(num_users, dtype=np.int32)

    for i in range(num_users):
        # Assign primary topic
        primary_topic = i % num_topics
        user_topics[i] = primary_topic

        # Create history matrix [num_topics, num_actions]
        history = np.zeros((num_topics, NUM_ACTIONS))

        # Strong positive engagement on primary topic
        history[primary_topic, ACTION_INDICES["favorite"]] = 0.7 + rng.random() * 0.2
        history[primary_topic, ACTION_INDICES["repost"]] = 0.5 + rng.random() * 0.2
        history[primary_topic, ACTION_INDICES["follow_author"]] = 0.4 + rng.random() * 0.2
        history[primary_topic, ACTION_INDICES["click"]] = 0.6 + rng.random() * 0.2

        # Weak engagement on other topics
        for t in range(num_topics):
            if t != primary_topic:
                history[t, ACTION_INDICES["click"]] = rng.random() * 0.2
                history[t, ACTION_INDICES["dwell"]] = rng.random() * 0.1

        user_histories[i] = history.flatten()

    # Content action probabilities: [num_content, num_actions]
    # Each content piece has topic-specific engagement patterns
    num_content = num_users  # Match for simplicity
    action_probs = np.zeros((num_content, NUM_ACTIONS), dtype=np.float32)
    content_topics = np.zeros(num_content, dtype=np.int32)

    for i in range(num_content):
        topic = i % num_topics
        content_topics[i] = topic

        # Base probabilities
        action_probs[i, ACTION_INDICES["dwell"]] = 0.3 + rng.random() * 0.2
        action_probs[i, ACTION_INDICES["click"]] = 0.2 + rng.random() * 0.2

        # Topic-specific engagement (varies by topic)
        engagement_level = 0.3 + (topic / num_topics) * 0.4
        action_probs[i, ACTION_INDICES["favorite"]] = engagement_level * (0.8 + rng.random() * 0.4)
        action_probs[i, ACTION_INDICES["repost"]] = engagement_level * (0.4 + rng.random() * 0.3)
        action_probs[i, ACTION_INDICES["reply"]] = engagement_level * (0.3 + rng.random() * 0.2)

        # Small negative action probabilities (usually close to 0)
        action_probs[i, ACTION_INDICES["not_interested"]] = rng.random() * 0.05
        action_probs[i, ACTION_INDICES["block_author"]] = rng.random() * 0.02
        action_probs[i, ACTION_INDICES["mute_author"]] = rng.random() * 0.02

    # Create preference data: users prefer content on their topic
    # Pairs: (user_idx, preferred_content_idx, rejected_content_idx)
    probs_preferred = []
    probs_rejected = []

    for i in range(num_users):
        user_topic = user_topics[i]

        # Find content on user's topic (preferred)
        matching_content = np.where(content_topics == user_topic)[0]
        # Find content on different topic (rejected)
        other_topic = (user_topic + 1) % num_topics
        other_content = np.where(content_topics == other_topic)[0]

        if len(matching_content) > 0 and len(other_content) > 0:
            pref_idx = rng.choice(matching_content)
            rej_idx = rng.choice(other_content)
            probs_preferred.append(action_probs[pref_idx])
            probs_rejected.append(action_probs[rej_idx])

    probs_preferred = np.array(probs_preferred)
    probs_rejected = np.array(probs_rejected)

    return {
        "user_histories": user_histories,
        "user_topics": user_topics,
        "action_probs": action_probs,
        "content_topics": content_topics,
        "probs_preferred": probs_preferred,
        "probs_rejected": probs_rejected,
        "num_topics": num_topics,
    }


def test_default_weights():
    """Test causal verification with default (hand-tuned) weights.

    Expected results:
    - Block intervention: PASS (weights have negative block_author weight)
    - Follow intervention: PASS (weights have positive follow_author weight)
    - History intervention: FAIL (simple weights don't use user history at all)

    The history test failure is expected and reveals a limitation: simple
    reward functions that only look at action probabilities cannot capture
    the causal effect of user history on preferences.
    """
    print("\n" + "=" * 70)
    print("TEST 1: Default Weights (no training)")
    print("=" * 70)
    print("Note: History test expected to FAIL - simple weights ignore user history")

    # Generate synthetic data
    data = generate_synthetic_data(num_users=500, num_topics=6)

    # Create reward function from default weights
    default_weights = RewardWeights.default()
    reward_fn = create_reward_fn_from_weights(np.array(default_weights.weights))

    # Run causal verification
    config = CausalTestConfig(
        num_samples=100,
        block_strength=0.8,
        history_strength=0.7,
        min_effect_size=0.05,  # Lower threshold for default weights
        min_pass_rate=0.8,  # 80% pass rate
    )
    suite = CausalVerificationSuite(config)

    results = suite.run_all(
        reward_fn=reward_fn,
        user_histories=data["user_histories"],
        action_probs_batch=data["action_probs"],
        post_topics=data["content_topics"],
        num_topics=data["num_topics"],
        verbose=True,
    )

    return results


def test_trained_two_stage():
    """Test causal verification with trained two-stage model.

    Expected results:
    - Block intervention: PASS (learned weights should penalize blocks)
    - Follow intervention: PASS (learned weights should reward follows)
    - History intervention: PARTIAL (~50%) - The two-stage model clusters
      users by history but uses the same per-cluster weights regardless
      of content topic. This reveals a limitation: the model captures
      USER-level preferences but not USER×CONTENT topic matching.

    Insight: To fully pass history intervention, we'd need a model that
    explicitly scores content based on how well it matches user's
    historical topic interests (e.g., topic-aware reward weights).
    """
    print("\n" + "=" * 70)
    print("TEST 2: Trained Two-Stage Model")
    print("=" * 70)

    # Generate synthetic data
    data = generate_synthetic_data(num_users=1000, num_topics=6)

    # Train two-stage model
    print("\nTraining two-stage model...")
    config = TwoStageConfig(
        num_clusters=6,
        learning_rate=0.01,
        num_epochs=50,
        batch_size=64,
    )

    state, metrics = train_two_stage(
        user_histories=data["user_histories"],
        probs_preferred=data["probs_preferred"],
        probs_rejected=data["probs_rejected"],
        config=config,
        verbose=True,
    )

    print(f"\nTraining accuracy: {metrics.overall_accuracy:.1%}")

    # Create reward function from trained model
    reward_fn = create_reward_fn_from_two_stage(state)

    # Run causal verification
    print("\n" + "-" * 60)
    print("Running causal verification on trained model...")
    print("-" * 60)

    test_config = CausalTestConfig(
        num_samples=200,
        block_strength=0.8,
        history_strength=0.7,
        min_effect_size=0.1,
        min_pass_rate=0.85,
    )
    suite = CausalVerificationSuite(test_config)

    results = suite.run_all(
        reward_fn=reward_fn,
        user_histories=data["user_histories"],
        action_probs_batch=data["action_probs"],
        post_topics=data["content_topics"],
        num_topics=data["num_topics"],
        verbose=True,
    )

    return results, state, metrics


def test_adversarial_weights():
    """Test causal verification with adversarial (wrong) weights.

    This should FAIL - verifying our tests catch broken models.
    """
    print("\n" + "=" * 70)
    print("TEST 3: Adversarial Weights (should FAIL)")
    print("=" * 70)

    # Generate synthetic data
    data = generate_synthetic_data(num_users=500, num_topics=6)

    # Create adversarial weights: block/mute have POSITIVE weights!
    adversarial_weights = np.zeros(NUM_ACTIONS)
    adversarial_weights[ACTION_INDICES["block_author"]] = +1.5  # Should be negative!
    adversarial_weights[ACTION_INDICES["mute_author"]] = +1.0   # Should be negative!
    adversarial_weights[ACTION_INDICES["favorite"]] = -0.5      # Should be positive!
    adversarial_weights[ACTION_INDICES["follow_author"]] = -1.0  # Should be positive!

    reward_fn = create_reward_fn_from_weights(adversarial_weights)

    # Run causal verification - should fail!
    config = CausalTestConfig(
        num_samples=100,
        min_effect_size=0.05,
        min_pass_rate=0.8,
    )
    suite = CausalVerificationSuite(config)

    results = suite.run_all(
        reward_fn=reward_fn,
        user_histories=data["user_histories"],
        action_probs_batch=data["action_probs"],
        post_topics=data["content_topics"],
        num_topics=data["num_topics"],
        verbose=True,
    )

    # Verify it failed
    block_failed = not results["block_intervention"].passed
    follow_failed = not results["follow_intervention"].passed

    if block_failed and follow_failed:
        print("\n[EXPECTED] Adversarial weights correctly detected as broken!")
    else:
        print("\n[WARNING] Adversarial weights not fully detected")

    return results


def main():
    """Run all causal verification tests."""
    print("=" * 70)
    print("CAUSAL VERIFICATION TEST SUITE")
    print("F4 Phase 3: Verifying reward models capture causation")
    print("=" * 70)

    # Test 1: Default weights
    results_default = test_default_weights()

    # Test 2: Trained two-stage model
    results_trained, state, metrics = test_trained_two_stage()

    # Test 3: Adversarial weights (should fail)
    results_adversarial = test_adversarial_weights()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    def summarize_results(name: str, results: dict) -> dict:
        all_passed = all(r.passed for r in results.values())
        return {
            "name": name,
            "all_passed": all_passed,
            "tests": {
                k: {
                    "passed": v.passed,
                    "pass_rate": v.pass_rate,
                    "mean_effect": v.mean_effect_size,
                }
                for k, v in results.items()
            }
        }

    summary = [
        summarize_results("Default Weights", results_default),
        summarize_results("Trained Two-Stage", results_trained),
        summarize_results("Adversarial Weights", results_adversarial),
    ]

    for s in summary:
        status = "PASS" if s["all_passed"] else "FAIL"
        if s["name"] == "Adversarial Weights":
            # Adversarial should fail
            status = "EXPECTED FAIL" if not s["all_passed"] else "UNEXPECTED PASS"
        print(f"\n{s['name']}: {status}")
        for test_name, test_res in s["tests"].items():
            test_status = "PASS" if test_res["passed"] else "FAIL"
            print(f"  - {test_name}: {test_status} ({test_res['pass_rate']:.1%})")

    # Interpretation
    print("\n" + "-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print("""
Block/Follow Tests:
  - PASS means the model correctly captures that blocking/following
    causally affects reward (negative for blocks, positive for follows)

History Test:
  - Default weights FAIL expected: simple R=w·P(actions) ignores history
  - Two-stage ~50% reveals limitation: clusters by user but doesn't
    explicitly model topic matching

Adversarial Detection:
  - FAIL expected: verifies our tests catch broken reward models
""")

    # Key insight for Phase 3
    print("KEY INSIGHT FOR PHASE 3:")
    print("  The two-stage model captures action-level causality (block→-reward)")
    print("  but only partially captures history-level causality (topic→score).")
    print("  To fully pass history tests, need topic-aware content scoring.")

    # Save results
    output_dir = "results/f4_phase3_causal"
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "default_weights": {
            k: {
                "passed": v.passed,
                "pass_rate": v.pass_rate,
                "mean_effect_size": v.mean_effect_size,
                "std_effect_size": v.std_effect_size,
            }
            for k, v in results_default.items()
        },
        "trained_two_stage": {
            "training_accuracy": metrics.overall_accuracy,
            "tests": {
                k: {
                    "passed": v.passed,
                    "pass_rate": v.pass_rate,
                    "mean_effect_size": v.mean_effect_size,
                    "std_effect_size": v.std_effect_size,
                }
                for k, v in results_trained.items()
            },
        },
        "adversarial_weights": {
            "correctly_detected": not all(r.passed for r in results_adversarial.values()),
            "tests": {
                k: {
                    "passed": v.passed,
                    "pass_rate": v.pass_rate,
                    "mean_effect_size": v.mean_effect_size,
                }
                for k, v in results_adversarial.items()
            },
        },
    }

    output_path = os.path.join(output_dir, "causal_verification_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
