"""Verification suite for synthetic data training.

Tools for verifying that the model learns the planted patterns
in synthetic Twitter data.
"""

from enhancements.verification.embedding_probes import (
    EmbeddingProbeResults,
    run_embedding_probes,
    test_user_archetype_clustering,
    test_topic_clustering,
)

from enhancements.verification.behavioral_tests import (
    BehavioralTestResults,
    run_behavioral_tests,
    test_topic_preference,
)

from enhancements.verification.action_tests import (
    ActionTestResults,
    ActionDistribution,
    run_action_tests,
)

from enhancements.verification.counterfactual_tests import (
    CounterfactualTestResults,
    run_counterfactual_tests,
    test_block_effect,
    test_archetype_flip,
)

from enhancements.verification.suite import (
    VerificationConfig,
    VerificationResults,
    run_verification_suite,
)

__all__ = [
    # Embedding probes
    "EmbeddingProbeResults",
    "run_embedding_probes",
    "test_user_archetype_clustering",
    "test_topic_clustering",
    # Behavioral tests
    "BehavioralTestResults",
    "run_behavioral_tests",
    "test_topic_preference",
    # Action tests
    "ActionTestResults",
    "ActionDistribution",
    "run_action_tests",
    # Counterfactual tests
    "CounterfactualTestResults",
    "run_counterfactual_tests",
    "test_block_effect",
    "test_archetype_flip",
    # Suite
    "VerificationConfig",
    "VerificationResults",
    "run_verification_suite",
]
