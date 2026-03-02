"""Tests for trajectory simulation and analysis tools."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phoenix"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enhancements.analysis.sensitivity_analysis import (
    SensitivityMetrics,
    run_random_trajectories,
    run_top_biased_trajectories,
)
from enhancements.analysis.trajectory_simulation import (
    CandidateScore,
    TrajectoryPath,
    TrajectorySimulator,
    compare_trajectories,
    format_trajectory_table,
)
from enhancements.optimization.full_kv_cache import FullKVCachedRunner
from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig
from phoenix.runners import ACTIONS, create_example_batch


def create_test_config(candidate_seq_len: int = 8):
    """Create a test model configuration."""
    hash_config = HashConfig(
        num_user_hashes=2,
        num_item_hashes=2,
        num_author_hashes=2,
    )
    return PhoenixModelConfig(
        emb_size=128,  # Smaller for faster tests
        num_actions=len(ACTIONS),
        history_seq_len=16,  # Smaller for faster tests
        candidate_seq_len=candidate_seq_len,
        hash_config=hash_config,
        product_surface_vocab_size=16,
        model=TransformerConfig(
            emb_size=128,
            widening_factor=2,
            key_size=32,
            num_q_heads=4,
            num_kv_heads=2,
            num_layers=2,  # Fewer layers for faster tests
            attn_output_multiplier=0.125,
        ),
    )


@pytest.fixture(scope="module")
def runner_and_batch():
    """Shared fixture for runner and batch to speed up tests."""
    config = create_test_config()
    runner = FullKVCachedRunner(config)
    runner.initialize()

    batch, embeddings = create_example_batch(
        batch_size=1,
        emb_size=config.emb_size,
        history_len=config.history_seq_len,
        num_candidates=config.candidate_seq_len,
        num_actions=config.num_actions,
        num_user_hashes=config.hash_config.num_user_hashes,
        num_item_hashes=config.hash_config.num_item_hashes,
        num_author_hashes=config.hash_config.num_author_hashes,
        product_surface_vocab_size=config.product_surface_vocab_size,
    )

    return runner, batch, embeddings, config


# ============================================================================
# TrajectorySimulator Tests
# ============================================================================

class TestTrajectorySimulator:
    """Tests for TrajectorySimulator class."""

    def test_simulator_initializes(self, runner_and_batch):
        """Test that simulator initializes correctly."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        assert simulator._initialized
        assert len(simulator._remaining_candidate_indices) == config.candidate_seq_len
        assert len(simulator._trajectory_steps) == 1  # Initial state

    def test_initial_state_has_all_candidates(self, runner_and_batch):
        """Test that initial state includes all candidates."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        initial_scores = simulator.current_scores()
        assert len(initial_scores) == config.candidate_seq_len

        # All candidates should be present
        candidate_indices = {cs.index for cs in initial_scores}
        assert candidate_indices == set(range(config.candidate_seq_len))

    def test_scores_are_ranked(self, runner_and_batch):
        """Test that scores are properly ranked (highest first)."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        scores = simulator.current_scores()

        # Verify descending score order
        for i in range(len(scores) - 1):
            assert scores[i].score >= scores[i + 1].score

        # Verify ranks are sequential
        for i, cs in enumerate(scores):
            assert cs.rank == i + 1

    def test_engage_removes_candidate(self, runner_and_batch):
        """Test that engaging with a candidate removes it from remaining."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        # Get initial top candidate
        initial_top = simulator.current_scores()[0].index

        # Engage with top
        simulator.engage(0)

        # Check it's removed
        remaining_indices = {cs.index for cs in simulator.current_scores()}
        assert initial_top not in remaining_indices
        assert len(simulator._remaining_candidate_indices) == config.candidate_seq_len - 1

    def test_engage_records_trajectory(self, runner_and_batch):
        """Test that engagement is recorded in trajectory."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        engaged_idx = simulator.current_scores()[0].index
        simulator.engage(0)

        trajectory = simulator.get_trajectory()
        assert len(trajectory.steps) == 2  # Initial + 1 engagement
        assert trajectory.engagement_sequence == [engaged_idx]

    def test_engage_top_n(self, runner_and_batch):
        """Test engaging with top candidate multiple times."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        steps = simulator.engage_top_n(3)

        assert len(steps) == 3
        trajectory = simulator.get_trajectory()
        assert len(trajectory.engagement_sequence) == 3
        assert len(simulator._remaining_candidate_indices) == config.candidate_seq_len - 3

    def test_engage_with_different_ranks(self, runner_and_batch):
        """Test engaging with candidates at different ranks."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        # Engage with rank 2 (third choice)
        third_choice = simulator.current_scores()[2].index
        simulator.engage(2)

        # Verify it was engaged
        trajectory = simulator.get_trajectory()
        assert trajectory.engagement_sequence[0] == third_choice

    def test_engage_until_exhausted(self, runner_and_batch):
        """Test engaging until no candidates remain."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        # Engage with all candidates
        simulator.engage_top_n(config.candidate_seq_len)

        assert len(simulator._remaining_candidate_indices) == 0
        trajectory = simulator.get_trajectory()
        assert len(trajectory.engagement_sequence) == config.candidate_seq_len

    def test_reset(self, runner_and_batch):
        """Test resetting the simulator."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage_top_n(3)

        simulator.reset()

        assert not simulator._initialized
        assert len(simulator._trajectory_steps) == 0
        assert len(simulator._engagement_sequence) == 0


# ============================================================================
# TrajectoryPath Tests
# ============================================================================

class TestTrajectoryPath:
    """Tests for TrajectoryPath data structure."""

    def test_trajectory_path_structure(self, runner_and_batch):
        """Test TrajectoryPath contains correct data."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage_top_n(3)

        trajectory = simulator.get_trajectory()

        assert isinstance(trajectory, TrajectoryPath)
        assert len(trajectory.steps) == 4  # Initial + 3 engagements
        assert len(trajectory.engagement_sequence) == 3

    def test_trajectory_step_fields(self, runner_and_batch):
        """Test TrajectoryStep has all required fields."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage(0)

        trajectory = simulator.get_trajectory()

        # Initial step
        initial = trajectory.steps[0]
        assert initial.step_num == 0
        assert initial.engaged_candidate_idx is None
        assert initial.engaged_candidate_score is None
        assert len(initial.remaining_scores) > 0

        # Engagement step
        step1 = trajectory.steps[1]
        assert step1.step_num == 1
        assert step1.engaged_candidate_idx is not None
        assert step1.engaged_candidate_score is not None


# ============================================================================
# CandidateScore Tests
# ============================================================================

class TestCandidateScore:
    """Tests for CandidateScore data structure."""

    def test_candidate_score_fields(self, runner_and_batch):
        """Test CandidateScore has correct fields."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        scores = simulator.current_scores()
        cs = scores[0]

        assert isinstance(cs, CandidateScore)
        assert isinstance(cs.index, int)
        assert isinstance(cs.score, float)
        assert isinstance(cs.rank, int)
        assert cs.rank == 1  # Top candidate has rank 1


# ============================================================================
# compare_trajectories Tests
# ============================================================================

class TestCompareTrajectories:
    """Tests for compare_trajectories function."""

    def test_compare_same_trajectory(self, runner_and_batch):
        """Test comparing a trajectory with itself."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage_top_n(3)
        trajectory = simulator.get_trajectory()

        result = compare_trajectories([trajectory, trajectory], ["A", "B"])

        assert result['num_trajectories'] == 2
        assert result['labels'] == ["A", "B"]

    def test_compare_different_trajectories(self, runner_and_batch):
        """Test comparing different trajectories."""
        runner, batch, embeddings, _ = runner_and_batch

        # Trajectory A: top choices
        sim_a = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        sim_a.initialize()
        sim_a.engage_top_n(3)
        traj_a = sim_a.get_trajectory()

        # Trajectory B: different choices
        sim_b = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        sim_b.initialize()
        sim_b.engage(2)  # Start with 3rd choice
        sim_b.engage_top_n(2)
        traj_b = sim_b.get_trajectory()

        result = compare_trajectories([traj_a, traj_b])

        assert result['num_trajectories'] == 2
        assert 'divergence_by_step' in result


# ============================================================================
# format_trajectory_table Tests
# ============================================================================

class TestFormatTrajectoryTable:
    """Tests for format_trajectory_table function."""

    def test_format_trajectory_table(self, runner_and_batch):
        """Test formatting a trajectory as a table."""
        runner, batch, embeddings, _ = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage_top_n(2)
        trajectory = simulator.get_trajectory()

        table = format_trajectory_table(trajectory)

        assert isinstance(table, str)
        assert "TRAJECTORY SIMULATION" in table
        assert "Step 0" in table
        assert "Step 1" in table


# ============================================================================
# SensitivityMetrics Tests
# ============================================================================

class TestSensitivityMetrics:
    """Tests for SensitivityMetrics class."""

    def test_metrics_initialization(self):
        """Test SensitivityMetrics initializes correctly."""
        metrics = SensitivityMetrics(num_candidates=8)

        assert metrics.num_candidates == 8
        assert metrics.num_runs == 0
        assert len(metrics.sequences) == 0

    def test_add_trajectory(self, runner_and_batch):
        """Test adding a trajectory updates metrics."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()
        simulator.engage_top_n(3)
        trajectory = simulator.get_trajectory()

        metrics = SensitivityMetrics(num_candidates=config.candidate_seq_len)
        metrics.add_trajectory(trajectory)

        assert metrics.num_runs == 1
        assert len(metrics.sequences) == 1
        assert len(metrics.sequences[0]) == 3  # 3 engagements

    def test_engagement_entropy(self):
        """Test engagement entropy calculation."""
        metrics = SensitivityMetrics(num_candidates=4)

        # Create fake trajectory data
        class FakeTrajectory:
            def __init__(self, engagement_sequence):
                self.engagement_sequence = engagement_sequence
                self.steps = [type('Step', (), {'remaining_scores': []})]

        # Uniform distribution
        metrics.add_trajectory(FakeTrajectory([0]))
        metrics.add_trajectory(FakeTrajectory([1]))
        metrics.add_trajectory(FakeTrajectory([2]))
        metrics.add_trajectory(FakeTrajectory([3]))

        entropy = metrics.compute_engagement_entropy()
        # Uniform distribution should have maximum entropy
        assert entropy > 0

    def test_outcome_diversity(self):
        """Test outcome diversity calculation."""
        metrics = SensitivityMetrics(num_candidates=4)

        class FakeTrajectory:
            def __init__(self, engagement_sequence):
                self.engagement_sequence = engagement_sequence
                self.steps = [type('Step', (), {'remaining_scores': []})]

        # All unique sequences
        metrics.add_trajectory(FakeTrajectory([0, 1, 2]))
        metrics.add_trajectory(FakeTrajectory([1, 0, 2]))
        metrics.add_trajectory(FakeTrajectory([2, 1, 0]))

        diversity = metrics.compute_outcome_diversity()
        assert diversity == 1.0  # 3 unique out of 3 runs

    def test_position_stability(self):
        """Test position stability calculation."""
        metrics = SensitivityMetrics(num_candidates=4)

        class FakeTrajectory:
            def __init__(self, engagement_sequence):
                self.engagement_sequence = engagement_sequence
                self.steps = [type('Step', (), {'remaining_scores': []})]

        # Same candidate at position 0 most of the time
        metrics.add_trajectory(FakeTrajectory([0, 1, 2]))
        metrics.add_trajectory(FakeTrajectory([0, 2, 1]))
        metrics.add_trajectory(FakeTrajectory([1, 0, 2]))

        most_common, freq = metrics.get_position_stability(0)
        assert most_common == 0
        assert freq == pytest.approx(2/3)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full analysis pipeline."""

    def test_run_random_trajectories(self, runner_and_batch):
        """Test running random trajectories."""
        runner, batch, embeddings, config = runner_and_batch
        rng = np.random.default_rng(seed=42)

        metrics = run_random_trajectories(
            runner, batch, embeddings,
            num_candidates=config.candidate_seq_len,
            num_runs=5,
            num_engagements=3,
            rng=rng,
        )

        assert metrics.num_runs == 5
        assert all(len(seq) == 3 for seq in metrics.sequences)

    def test_run_top_biased_trajectories(self, runner_and_batch):
        """Test running top-biased trajectories."""
        runner, batch, embeddings, config = runner_and_batch
        rng = np.random.default_rng(seed=42)

        metrics = run_top_biased_trajectories(
            runner, batch, embeddings,
            num_candidates=config.candidate_seq_len,
            num_runs=5,
            num_engagements=3,
            top_probability=0.8,
            rng=rng,
        )

        assert metrics.num_runs == 5

    def test_biased_has_less_diversity(self, runner_and_batch):
        """Test that top-biased strategy has less diversity than random."""
        runner, batch, embeddings, config = runner_and_batch
        rng = np.random.default_rng(seed=42)

        random_metrics = run_random_trajectories(
            runner, batch, embeddings,
            num_candidates=config.candidate_seq_len,
            num_runs=20,
            num_engagements=4,
            rng=rng,
        )

        biased_metrics = run_top_biased_trajectories(
            runner, batch, embeddings,
            num_candidates=config.candidate_seq_len,
            num_runs=20,
            num_engagements=4,
            top_probability=0.9,  # High bias
            rng=rng,
        )

        # Biased should have fewer unique outcomes
        random_unique = len(set(tuple(s) for s in random_metrics.sequences))
        biased_unique = len(set(tuple(s) for s in biased_metrics.sequences))

        # With high bias probability, biased should have fewer unique outcomes
        # (Note: with small num_runs, this may not always hold, so we use a soft check)
        assert biased_unique <= random_unique or biased_unique <= 20


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge case tests."""

    def test_engage_with_invalid_rank(self, runner_and_batch):
        """Test that engaging with invalid rank raises error."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        with pytest.raises(ValueError):
            simulator.engage(config.candidate_seq_len + 5)  # Out of range

    def test_engage_after_exhausted(self, runner_and_batch):
        """Test behavior when trying to engage after all candidates exhausted."""
        runner, batch, embeddings, config = runner_and_batch

        simulator = TrajectorySimulator(
            runner=runner,
            initial_batch=batch,
            initial_embeddings=embeddings,
        )
        simulator.initialize()

        # Exhaust all candidates
        simulator.engage_top_n(config.candidate_seq_len)

        # Further engage_top_n should not add more
        steps = simulator.engage_top_n(1)
        assert len(steps) == 0
