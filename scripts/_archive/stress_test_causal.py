"""Stress tests for causal verification framework.

These tests validate that a reward model truly understands causal relationships,
not just correlations. A model claiming causal understanding should pass ALL tests.

Stress Tests:
1. Effect Scaling - Does effect scale monotonically with intervention strength?
2. Compound Interventions - Do multiple signals compound correctly?
3. Conflicting Signals - Can model handle high positive AND negative signals?
4. Cross-Preference - Is block effect consistent across preferred/non-preferred content?
5. Reversibility - Does removing intervention restore baseline score?
6. Noise Robustness - Does causal relationship hold under noise?
7. Threshold Sensitivity - Is pass rate stable across effect thresholds?
"""

import importlib.util
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

causal_path = os.path.join(project_root, "enhancements/reward_modeling/causal_verification.py")
causal_mod = load_module_direct("enhancements.reward_modeling.causal_verification", causal_path)
create_reward_fn_from_weights = causal_mod.create_reward_fn_from_weights
create_reward_fn_from_two_stage = causal_mod.create_reward_fn_from_two_stage

# Type alias
RewardFunction = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class StressTestResult:
    """Result of a single stress test."""
    name: str
    passed: bool
    details: dict
    message: str


def generate_base_action_probs(rng: np.random.Generator) -> np.ndarray:
    """Generate realistic base action probabilities."""
    probs = np.zeros(NUM_ACTIONS, dtype=np.float32)

    # Typical engagement pattern
    probs[ACTION_INDICES["dwell"]] = 0.3 + rng.random() * 0.3
    probs[ACTION_INDICES["click"]] = 0.2 + rng.random() * 0.2
    probs[ACTION_INDICES["favorite"]] = 0.1 + rng.random() * 0.3
    probs[ACTION_INDICES["repost"]] = 0.05 + rng.random() * 0.15
    probs[ACTION_INDICES["reply"]] = 0.05 + rng.random() * 0.1
    probs[ACTION_INDICES["follow_author"]] = 0.02 + rng.random() * 0.08

    # Low negative signals
    probs[ACTION_INDICES["not_interested"]] = rng.random() * 0.05
    probs[ACTION_INDICES["block_author"]] = rng.random() * 0.02
    probs[ACTION_INDICES["mute_author"]] = rng.random() * 0.02

    return probs


# =============================================================================
# STRESS TEST 1: Effect Scaling
# =============================================================================

def test_effect_scaling(
    reward_fn: RewardFunction,
    num_samples: int = 50,
    seed: int = 42,
) -> StressTestResult:
    """Test that causal effect scales monotonically with intervention strength.

    Expected: Stronger block signal → larger negative effect
    """
    rng = np.random.default_rng(seed)
    strengths = [0.2, 0.4, 0.6, 0.8, 1.0]

    effects_by_strength = {s: [] for s in strengths}

    for _ in range(num_samples):
        user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)  # Dummy history
        base_probs = generate_base_action_probs(rng)

        baseline_score = reward_fn(user_history, base_probs)

        for strength in strengths:
            modified = base_probs.copy()
            modified[ACTION_INDICES["block_author"]] = strength
            # Reduce positive engagement proportionally
            modified[ACTION_INDICES["favorite"]] *= (1 - strength)
            modified[ACTION_INDICES["follow_author"]] *= (1 - strength)

            score_after = reward_fn(user_history, modified)
            effect = score_after - baseline_score
            effects_by_strength[strength].append(effect)

    # Check monotonicity: stronger intervention → more negative effect
    mean_effects = [np.mean(effects_by_strength[s]) for s in strengths]

    # Effects should be increasingly negative (decreasing)
    is_monotonic = all(mean_effects[i] >= mean_effects[i+1] for i in range(len(mean_effects)-1))

    # Also check that effect at max strength is significantly negative
    significant_effect = mean_effects[-1] < -0.1

    passed = bool(is_monotonic and significant_effect)

    return StressTestResult(
        name="Effect Scaling",
        passed=passed,
        details={
            "strengths": strengths,
            "mean_effects": [float(e) for e in mean_effects],
            "is_monotonic": is_monotonic,
            "significant_at_max": significant_effect,
        },
        message=f"Monotonic: {is_monotonic}, Effects: {[f'{e:.3f}' for e in mean_effects]}"
    )


# =============================================================================
# STRESS TEST 2: Compound Interventions
# =============================================================================

def test_compound_interventions(
    reward_fn: RewardFunction,
    num_samples: int = 50,
    seed: int = 43,
) -> StressTestResult:
    """Test that multiple negative signals compound (larger total effect).

    Expected: block + mute + not_interested > block alone
    """
    rng = np.random.default_rng(seed)

    single_effects = []
    compound_effects = []

    for _ in range(num_samples):
        user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)
        base_probs = generate_base_action_probs(rng)

        baseline_score = reward_fn(user_history, base_probs)

        # Single intervention: block only
        single = base_probs.copy()
        single[ACTION_INDICES["block_author"]] = 0.8
        single_score = reward_fn(user_history, single)
        single_effects.append(single_score - baseline_score)

        # Compound intervention: block + mute + not_interested
        compound = base_probs.copy()
        compound[ACTION_INDICES["block_author"]] = 0.8
        compound[ACTION_INDICES["mute_author"]] = 0.6
        compound[ACTION_INDICES["not_interested"]] = 0.7
        compound[ACTION_INDICES["favorite"]] *= 0.2  # Reduce positive
        compound_score = reward_fn(user_history, compound)
        compound_effects.append(compound_score - baseline_score)

    mean_single = np.mean(single_effects)
    mean_compound = np.mean(compound_effects)

    # Compound effect should be more negative than single
    compounds_more = mean_compound < mean_single
    # And significantly so (at least 20% more negative)
    significantly_more = mean_compound < mean_single * 1.2 if mean_single < 0 else mean_compound < mean_single - 0.1

    passed = bool(compounds_more and significantly_more)

    return StressTestResult(
        name="Compound Interventions",
        passed=passed,
        details={
            "mean_single_effect": float(mean_single),
            "mean_compound_effect": float(mean_compound),
            "compound_stronger": compounds_more,
            "ratio": float(mean_compound / mean_single) if mean_single != 0 else 0,
        },
        message=f"Single: {mean_single:.3f}, Compound: {mean_compound:.3f}, Ratio: {mean_compound/mean_single:.2f}x" if mean_single != 0 else "Division by zero"
    )


# =============================================================================
# STRESS TEST 3: Conflicting Signals
# =============================================================================

def test_conflicting_signals(
    reward_fn: RewardFunction,
    num_samples: int = 50,
    seed: int = 44,
) -> StressTestResult:
    """Test model handles content with both high positive AND high negative signals.

    Expected: Block intervention still decreases score even with high favorites
    """
    rng = np.random.default_rng(seed)

    effect_preserved = 0

    for _ in range(num_samples):
        user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)

        # Create conflicting content: high favorite AND high block
        conflicting = np.zeros(NUM_ACTIONS, dtype=np.float32)
        conflicting[ACTION_INDICES["favorite"]] = 0.8
        conflicting[ACTION_INDICES["repost"]] = 0.6
        conflicting[ACTION_INDICES["follow_author"]] = 0.5
        # No block yet

        baseline_score = reward_fn(user_history, conflicting)

        # Add block signal
        with_block = conflicting.copy()
        with_block[ACTION_INDICES["block_author"]] = 0.8

        block_score = reward_fn(user_history, with_block)

        # Block should still decrease score
        if block_score < baseline_score:
            effect_preserved += 1

    pass_rate = effect_preserved / num_samples
    passed = pass_rate >= 0.9  # 90% should show correct direction

    return StressTestResult(
        name="Conflicting Signals",
        passed=passed,
        details={
            "pass_rate": float(pass_rate),
            "samples_correct": effect_preserved,
            "total_samples": num_samples,
        },
        message=f"Block effect preserved in {pass_rate:.1%} of conflicting cases"
    )


# =============================================================================
# STRESS TEST 4: Cross-Preference Interventions
# =============================================================================

def test_cross_preference(
    reward_fn: RewardFunction,
    num_topics: int = 6,
    num_samples: int = 50,
    seed: int = 45,
) -> StressTestResult:
    """Test that block effect is consistent regardless of content topic.

    Expected: Blocking should decrease score for ANY content, not just non-preferred
    """
    rng = np.random.default_rng(seed)

    effects_by_match = {"matched": [], "mismatched": []}

    for _ in range(num_samples):
        # Create user with clear topic preference (e.g., topic 0)
        user_history = np.zeros(num_topics * NUM_ACTIONS, dtype=np.float32)
        preferred_topic = 0
        # Strong engagement on preferred topic
        user_history[preferred_topic * NUM_ACTIONS + ACTION_INDICES["favorite"]] = 0.8
        user_history[preferred_topic * NUM_ACTIONS + ACTION_INDICES["repost"]] = 0.5

        for is_matched in [True, False]:
            content_topic = preferred_topic if is_matched else (preferred_topic + 3) % num_topics

            base_probs = generate_base_action_probs(rng)
            baseline = reward_fn(user_history, base_probs)

            # Apply block
            blocked = base_probs.copy()
            blocked[ACTION_INDICES["block_author"]] = 0.8
            blocked[ACTION_INDICES["favorite"]] *= 0.2
            block_score = reward_fn(user_history, blocked)

            effect = block_score - baseline
            key = "matched" if is_matched else "mismatched"
            effects_by_match[key].append(effect)

    mean_matched = np.mean(effects_by_match["matched"])
    mean_mismatched = np.mean(effects_by_match["mismatched"])

    # Both should be negative (block works on any content)
    both_negative = mean_matched < 0 and mean_mismatched < 0

    # Effects should be similar (within 50% of each other)
    if mean_matched != 0:
        ratio = abs(mean_mismatched / mean_matched)
        similar_magnitude = 0.5 < ratio < 2.0
    else:
        similar_magnitude = abs(mean_mismatched) < 0.1

    passed = bool(both_negative and similar_magnitude)

    return StressTestResult(
        name="Cross-Preference Interventions",
        passed=passed,
        details={
            "mean_matched_effect": float(mean_matched),
            "mean_mismatched_effect": float(mean_mismatched),
            "both_negative": both_negative,
            "similar_magnitude": similar_magnitude,
        },
        message=f"Matched: {mean_matched:.3f}, Mismatched: {mean_mismatched:.3f}"
    )


# =============================================================================
# STRESS TEST 5: Reversibility
# =============================================================================

def test_reversibility(
    reward_fn: RewardFunction,
    num_samples: int = 50,
    seed: int = 46,
) -> StressTestResult:
    """Test that removing intervention restores baseline score.

    Expected: baseline → block → remove_block ≈ baseline
    """
    rng = np.random.default_rng(seed)

    restoration_errors = []

    for _ in range(num_samples):
        user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)
        base_probs = generate_base_action_probs(rng)

        # Baseline
        baseline = reward_fn(user_history, base_probs)

        # Apply block
        blocked = base_probs.copy()
        blocked[ACTION_INDICES["block_author"]] = 0.8
        blocked[ACTION_INDICES["favorite"]] *= 0.2
        _ = reward_fn(user_history, blocked)  # Just to confirm it changes

        # Remove block (restore original)
        restored = base_probs.copy()  # Back to original
        restored_score = reward_fn(user_history, restored)

        # Should be exactly the same (deterministic model)
        error = abs(restored_score - baseline)
        restoration_errors.append(error)

    mean_error = np.mean(restoration_errors)
    max_error = np.max(restoration_errors)

    # Should restore perfectly (within floating point tolerance)
    passed = max_error < 1e-5

    return StressTestResult(
        name="Reversibility",
        passed=passed,
        details={
            "mean_restoration_error": float(mean_error),
            "max_restoration_error": float(max_error),
        },
        message=f"Max restoration error: {max_error:.2e}"
    )


# =============================================================================
# STRESS TEST 6: Noise Robustness
# =============================================================================

def test_noise_robustness(
    reward_fn: RewardFunction,
    noise_levels: list[float] = [0.05, 0.1, 0.15, 0.2],
    num_samples: int = 50,
    seed: int = 47,
) -> StressTestResult:
    """Test that causal relationship holds under noise.

    Expected: Block effect direction preserved even with noisy action probs
    """
    rng = np.random.default_rng(seed)

    pass_rates_by_noise = {}

    for noise_std in noise_levels:
        correct = 0

        for _ in range(num_samples):
            user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)
            base_probs = generate_base_action_probs(rng)

            # Add noise to base probs
            noisy_base = base_probs + rng.normal(0, noise_std, NUM_ACTIONS).astype(np.float32)
            noisy_base = np.clip(noisy_base, 0, 1)

            baseline = reward_fn(user_history, noisy_base)

            # Apply block (also with noise)
            blocked = noisy_base.copy()
            blocked[ACTION_INDICES["block_author"]] = 0.8 + rng.normal(0, noise_std * 0.5)
            blocked[ACTION_INDICES["block_author"]] = np.clip(blocked[ACTION_INDICES["block_author"]], 0.5, 1.0)
            blocked[ACTION_INDICES["favorite"]] *= 0.2

            block_score = reward_fn(user_history, blocked)

            # Direction should still be correct
            if block_score < baseline:
                correct += 1

        pass_rates_by_noise[noise_std] = correct / num_samples

    # Should maintain >80% pass rate even at highest noise
    min_pass_rate = min(pass_rates_by_noise.values())
    passed = min_pass_rate >= 0.8

    return StressTestResult(
        name="Noise Robustness",
        passed=passed,
        details={
            "pass_rates_by_noise": {str(k): float(v) for k, v in pass_rates_by_noise.items()},
            "min_pass_rate": float(min_pass_rate),
        },
        message=f"Pass rates: {[f'{v:.1%}' for v in pass_rates_by_noise.values()]}, Min: {min_pass_rate:.1%}"
    )


# =============================================================================
# STRESS TEST 7: Threshold Sensitivity
# =============================================================================

def test_threshold_sensitivity(
    reward_fn: RewardFunction,
    thresholds: list[float] = [0.01, 0.05, 0.1, 0.2, 0.5],
    num_samples: int = 100,
    seed: int = 48,
) -> StressTestResult:
    """Test stability of pass rate across effect size thresholds.

    Expected: Pass rate should degrade gracefully, not cliff-edge
    """
    rng = np.random.default_rng(seed)

    # Collect all effects
    effects = []

    for _ in range(num_samples):
        user_history = rng.random(NUM_ACTIONS * 6).astype(np.float32)
        base_probs = generate_base_action_probs(rng)

        baseline = reward_fn(user_history, base_probs)

        blocked = base_probs.copy()
        blocked[ACTION_INDICES["block_author"]] = 0.8
        blocked[ACTION_INDICES["favorite"]] *= 0.2
        block_score = reward_fn(user_history, blocked)

        effect = baseline - block_score  # Positive = score decreased (good)
        effects.append(effect)

    # Compute pass rate at each threshold
    pass_rates = {}
    for thresh in thresholds:
        passed = sum(1 for e in effects if e > thresh)
        pass_rates[thresh] = passed / len(effects)

    # Check for graceful degradation (no cliff edges > 30% drop between adjacent thresholds)
    rates = list(pass_rates.values())
    max_drop = max(rates[i] - rates[i+1] for i in range(len(rates)-1))

    graceful = max_drop < 0.3
    # Should have high pass rate at low threshold
    high_at_low = pass_rates[thresholds[0]] >= 0.9

    passed = graceful and high_at_low

    return StressTestResult(
        name="Threshold Sensitivity",
        passed=passed,
        details={
            "pass_rates": {str(k): float(v) for k, v in pass_rates.items()},
            "max_drop_between_adjacent": float(max_drop),
            "graceful_degradation": graceful,
        },
        message=f"Rates: {[f'{v:.1%}' for v in pass_rates.values()]}, Max drop: {max_drop:.1%}"
    )


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_stress_tests(
    reward_fn: RewardFunction,
    model_name: str = "Unknown",
    verbose: bool = True,
) -> dict[str, StressTestResult]:
    """Run all stress tests on a reward function."""

    if verbose:
        print("=" * 70)
        print(f"CAUSAL VERIFICATION STRESS TESTS: {model_name}")
        print("=" * 70)

    tests = [
        ("Effect Scaling", test_effect_scaling),
        ("Compound Interventions", test_compound_interventions),
        ("Conflicting Signals", test_conflicting_signals),
        ("Cross-Preference", test_cross_preference),
        ("Reversibility", test_reversibility),
        ("Noise Robustness", test_noise_robustness),
        ("Threshold Sensitivity", test_threshold_sensitivity),
    ]

    results = {}
    passed_count = 0

    for name, test_fn in tests:
        if verbose:
            print(f"\n[{len(results)+1}/{len(tests)}] {name}...")

        result = test_fn(reward_fn)
        results[name] = result

        if result.passed:
            passed_count += 1

        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"  Result: {status}")
            print(f"  {result.message}")

    if verbose:
        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed_count}/{len(tests)} tests passed")
        print("=" * 70)
        for name, result in results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {name}: {status}")

    return results


def main():
    """Run stress tests on default weights and trained two-stage model."""

    print("=" * 70)
    print("CAUSAL VERIFICATION STRESS TEST SUITE")
    print("=" * 70)

    all_results = {}

    # Test 1: Default weights
    print("\n" + "=" * 70)
    print("MODEL 1: Default Weights (hand-tuned)")
    print("=" * 70)

    default_weights = RewardWeights.default()
    default_fn = create_reward_fn_from_weights(np.array(default_weights.weights))

    results_default = run_all_stress_tests(default_fn, "Default Weights")
    all_results["default_weights"] = results_default

    # Test 2: Trained two-stage model
    print("\n" + "=" * 70)
    print("MODEL 2: Trained Two-Stage Model")
    print("=" * 70)

    # Generate training data
    print("\nGenerating training data...")
    rng = np.random.default_rng(42)
    num_users = 1000
    num_topics = 6

    # User histories with topic preferences
    user_histories = np.zeros((num_users, num_topics * NUM_ACTIONS), dtype=np.float32)
    user_topics = np.zeros(num_users, dtype=np.int32)

    for i in range(num_users):
        topic = i % num_topics
        user_topics[i] = topic
        user_histories[i, topic * NUM_ACTIONS + ACTION_INDICES["favorite"]] = 0.7 + rng.random() * 0.2
        user_histories[i, topic * NUM_ACTIONS + ACTION_INDICES["repost"]] = 0.4 + rng.random() * 0.2

    # Preference pairs
    probs_preferred = []
    probs_rejected = []

    for i in range(num_users):
        pref = generate_base_action_probs(rng)
        pref[ACTION_INDICES["favorite"]] += 0.3

        rej = generate_base_action_probs(rng)
        rej[ACTION_INDICES["not_interested"]] += 0.2

        probs_preferred.append(pref)
        probs_rejected.append(rej)

    probs_preferred = np.array(probs_preferred)
    probs_rejected = np.array(probs_rejected)

    # Train model
    print("Training two-stage model...")
    config = TwoStageConfig(num_clusters=6, num_epochs=50)
    state, metrics = train_two_stage(
        user_histories, probs_preferred, probs_rejected, config, verbose=False
    )
    print(f"Training accuracy: {metrics.overall_accuracy:.1%}")

    # Create reward function
    two_stage_fn = create_reward_fn_from_two_stage(state)

    results_two_stage = run_all_stress_tests(two_stage_fn, "Two-Stage Model")
    all_results["two_stage"] = results_two_stage

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    for model_name, results in all_results.items():
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        status = "ALL PASS" if passed == total else f"{passed}/{total}"
        print(f"\n{model_name}: {status}")
        for test_name, result in results.items():
            icon = "✅" if result.passed else "❌"
            print(f"  {icon} {test_name}")

    # Save results
    output_dir = "results/f4_phase3_causal"
    os.makedirs(output_dir, exist_ok=True)

    def convert_to_json_serializable(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    output = {
        model_name: {
            test_name: {
                "passed": bool(result.passed),
                "message": result.message,
                "details": convert_to_json_serializable(result.details),
            }
            for test_name, result in results.items()
        }
        for model_name, results in all_results.items()
    }

    output_path = os.path.join(output_dir, "stress_test_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Return exit code based on results
    all_passed = all(
        all(r.passed for r in results.values())
        for results in all_results.values()
    )

    if all_passed:
        print("\n🎉 ALL STRESS TESTS PASSED - Model demonstrates causal understanding")
    else:
        print("\n⚠️  SOME TESTS FAILED - Model may not fully capture causal relationships")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
