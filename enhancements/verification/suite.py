"""Verification suite orchestrator.

Runs all verification tests and generates a comprehensive report.

Usage:
    from enhancements.verification.suite import run_verification_suite

    results = run_verification_suite(adapter, dataset, runner, params)
    print(results.report())
"""

from dataclasses import dataclass
from typing import Dict, Optional
import time

from enhancements.data.synthetic_twitter import SyntheticTwitterDataset
from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter

from enhancements.verification.embedding_probes import (
    EmbeddingProbeResults,
    run_embedding_probes,
)
from enhancements.verification.behavioral_tests import (
    BehavioralTestResults,
    run_behavioral_tests,
)
from enhancements.verification.action_tests import (
    ActionTestResults,
    run_action_tests,
)
from enhancements.verification.counterfactual_tests import (
    CounterfactualTestResults,
    run_counterfactual_tests,
)


@dataclass
class VerificationConfig:
    """Configuration for verification suite."""
    # Sample sizes
    user_sample_size: int = 500
    post_sample_size: int = 1000
    behavioral_samples: int = 50
    action_samples: int = 100
    block_tests: int = 50
    flip_tests: int = 30

    # Thresholds
    silhouette_threshold: float = 0.2
    behavioral_tolerance: float = 0.15


@dataclass
class VerificationResults:
    """Results from the full verification suite."""
    embedding_probes: Optional[EmbeddingProbeResults]
    behavioral_tests: Optional[BehavioralTestResults]
    action_tests: Optional[ActionTestResults]
    counterfactual_tests: Optional[CounterfactualTestResults]

    # Timing
    total_time_s: float
    embedding_time_s: float
    behavioral_time_s: float
    action_time_s: float
    counterfactual_time_s: float

    # Overall pass/fail
    all_tests_passed: bool

    def report(self) -> str:
        """Generate a human-readable report."""
        lines = [
            "=" * 70,
            "VERIFICATION SUITE RESULTS",
            "=" * 70,
            "",
        ]

        # Embedding probes
        lines.append("1. EMBEDDING PROBES")
        lines.append("-" * 40)
        if self.embedding_probes:
            user_status = "PASS" if self.embedding_probes.user_clustering_pass else "FAIL"
            topic_status = "PASS" if self.embedding_probes.topic_clustering_pass else "FAIL"
            lines.append(f"  User archetype clustering: {self.embedding_probes.user_silhouette:.4f} [{user_status}]")
            lines.append(f"  Topic clustering: {self.embedding_probes.topic_silhouette:.4f} [{topic_status}]")
        else:
            lines.append("  [SKIPPED]")
        lines.append(f"  Time: {self.embedding_time_s:.1f}s")
        lines.append("")

        # Behavioral tests
        lines.append("2. BEHAVIORAL TESTS")
        lines.append("-" * 40)
        if self.behavioral_tests:
            lines.append(f"  Tests passed: {sum(1 for t in self.behavioral_tests.individual_tests if t.passed)}/{len(self.behavioral_tests.individual_tests)}")
            lines.append(f"  Overall accuracy: {self.behavioral_tests.overall_accuracy:.2%}")
            lines.append(f"  Mean error: {self.behavioral_tests.mean_error:.4f}")
            lines.append(f"  Correlation: {self.behavioral_tests.correlation:.4f}")
        else:
            lines.append("  [SKIPPED]")
        lines.append(f"  Time: {self.behavioral_time_s:.1f}s")
        lines.append("")

        # Action tests
        lines.append("3. ACTION DIFFERENTIATION TESTS")
        lines.append("-" * 40)
        if self.action_tests:
            lines.append(f"  Tests passed: {self.action_tests.tests_passed}/{self.action_tests.tests_total}")
            lines.append(f"  Lurker repost ratio: {self.action_tests.lurker_distribution.repost_ratio:.4f}")
            lines.append(f"  Power user repost ratio: {self.action_tests.power_user_distribution.repost_ratio:.4f}")
            for test in self.action_tests.tests:
                status = "PASS" if test.passed else "FAIL"
                lines.append(f"    {test.test_name}: {test.actual} [{status}]")
        else:
            lines.append("  [SKIPPED]")
        lines.append(f"  Time: {self.action_time_s:.1f}s")
        lines.append("")

        # Counterfactual tests
        lines.append("4. COUNTERFACTUAL TESTS")
        lines.append("-" * 40)
        if self.counterfactual_tests:
            lines.append(f"  Block effect rate: {self.counterfactual_tests.block_effect_rate:.2%}")
            lines.append(f"  Archetype flip rate: {self.counterfactual_tests.archetype_flip_rate:.2%}")
        else:
            lines.append("  [SKIPPED]")
        lines.append(f"  Time: {self.counterfactual_time_s:.1f}s")
        lines.append("")

        # Summary
        lines.append("=" * 70)
        lines.append("SUMMARY")
        lines.append("=" * 70)
        lines.append(f"Total time: {self.total_time_s:.1f}s")
        lines.append(f"Overall: {'PASS' if self.all_tests_passed else 'FAIL'}")
        lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "embedding_probes": {
                "user_silhouette": self.embedding_probes.user_silhouette if self.embedding_probes else None,
                "topic_silhouette": self.embedding_probes.topic_silhouette if self.embedding_probes else None,
                "user_pass": self.embedding_probes.user_clustering_pass if self.embedding_probes else None,
                "topic_pass": self.embedding_probes.topic_clustering_pass if self.embedding_probes else None,
            },
            "behavioral_tests": {
                "accuracy": self.behavioral_tests.overall_accuracy if self.behavioral_tests else None,
                "mean_error": self.behavioral_tests.mean_error if self.behavioral_tests else None,
                "correlation": self.behavioral_tests.correlation if self.behavioral_tests else None,
            },
            "action_tests": {
                "tests_passed": self.action_tests.tests_passed if self.action_tests else None,
                "tests_total": self.action_tests.tests_total if self.action_tests else None,
                "lurker_rt_ratio": self.action_tests.lurker_distribution.repost_ratio if self.action_tests else None,
                "power_rt_ratio": self.action_tests.power_user_distribution.repost_ratio if self.action_tests else None,
            },
            "counterfactual_tests": {
                "block_effect_rate": self.counterfactual_tests.block_effect_rate if self.counterfactual_tests else None,
                "flip_rate": self.counterfactual_tests.archetype_flip_rate if self.counterfactual_tests else None,
            },
            "timing": {
                "total_s": self.total_time_s,
                "embedding_s": self.embedding_time_s,
                "behavioral_s": self.behavioral_time_s,
                "action_s": self.action_time_s,
                "counterfactual_s": self.counterfactual_time_s,
            },
            "all_passed": self.all_tests_passed,
        }


def run_verification_suite(
    adapter: SyntheticTwitterPhoenixAdapter,
    dataset: SyntheticTwitterDataset,
    runner,
    params: Dict,
    config: Optional[VerificationConfig] = None,
    skip_embedding: bool = False,
    skip_behavioral: bool = False,
    skip_action: bool = False,
    skip_counterfactual: bool = False,
) -> VerificationResults:
    """Run the full verification suite.

    Args:
        adapter: Adapter with learned embeddings
        dataset: Synthetic dataset
        runner: Phoenix inference runner
        params: Model parameters
        config: Verification configuration
        skip_*: Flags to skip specific test categories

    Returns:
        VerificationResults with all test results
    """
    if config is None:
        config = VerificationConfig()

    total_start = time.time()

    # Embedding probes
    embedding_start = time.time()
    if not skip_embedding:
        print("Running embedding probes...")
        embedding_results = run_embedding_probes(
            adapter, dataset,
            user_sample_size=config.user_sample_size,
            post_sample_size=config.post_sample_size,
            silhouette_threshold=config.silhouette_threshold,
        )
    else:
        embedding_results = None
    embedding_time = time.time() - embedding_start

    # Behavioral tests
    behavioral_start = time.time()
    if not skip_behavioral:
        print("Running behavioral tests...")
        behavioral_results = run_behavioral_tests(
            adapter, dataset, runner, params,
            tolerance=config.behavioral_tolerance,
            num_samples=config.behavioral_samples,
        )
    else:
        behavioral_results = None
    behavioral_time = time.time() - behavioral_start

    # Action tests
    action_start = time.time()
    if not skip_action:
        print("Running action differentiation tests...")
        action_results = run_action_tests(
            adapter, dataset, runner, params,
            num_samples=config.action_samples,
        )
    else:
        action_results = None
    action_time = time.time() - action_start

    # Counterfactual tests
    counterfactual_start = time.time()
    if not skip_counterfactual:
        print("Running counterfactual tests...")
        counterfactual_results = run_counterfactual_tests(
            adapter, dataset, runner, params,
            num_block_tests=config.block_tests,
            num_flip_tests=config.flip_tests,
        )
    else:
        counterfactual_results = None
    counterfactual_time = time.time() - counterfactual_start

    total_time = time.time() - total_start

    # Determine overall pass/fail
    all_passed = True
    if embedding_results:
        all_passed &= embedding_results.user_clustering_pass
        all_passed &= embedding_results.topic_clustering_pass
    if behavioral_results:
        all_passed &= behavioral_results.overall_accuracy >= 0.7
    if action_results:
        all_passed &= action_results.tests_passed >= action_results.tests_total * 0.5
    if counterfactual_results:
        all_passed &= counterfactual_results.block_effect_rate >= 0.5

    return VerificationResults(
        embedding_probes=embedding_results,
        behavioral_tests=behavioral_results,
        action_tests=action_results,
        counterfactual_tests=counterfactual_results,
        total_time_s=total_time,
        embedding_time_s=embedding_time,
        behavioral_time_s=behavioral_time,
        action_time_s=action_time,
        counterfactual_time_s=counterfactual_time,
        all_tests_passed=all_passed,
    )
