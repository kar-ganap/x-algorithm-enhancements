"""Analysis tools for Phoenix recommendation system."""

from enhancements.analysis.counterfactual_analysis import (
    AblationResult,
    CounterfactualAnalyzer,
    RankingSnapshot,
    compute_kendall_tau,
)
from enhancements.analysis.diversity_metrics import (
    DiversityAnalysisResult,
    DiversitySnapshot,
    TrajectoryDiversity,
    analyze_diversity,
    compute_gini_coefficient,
)
from enhancements.analysis.real_trajectory_simulation import (
    RealTrajectoryPath,
    RealTrajectorySimulator,
    RealTrajectoryStep,
)
from enhancements.analysis.sensitivity_analysis import (
    SensitivityMetrics,
    run_random_trajectories,
    run_top_biased_trajectories,
)
from enhancements.analysis.trajectory_simulation import (
    CandidateScore,
    TrajectoryPath,
    TrajectorySimulator,
    TrajectoryStep,
    compare_trajectories,
    format_trajectory_table,
)

__all__ = [
    # Simulated trajectory simulation
    'TrajectorySimulator',
    'TrajectoryPath',
    'TrajectoryStep',
    'CandidateScore',
    'compare_trajectories',
    'format_trajectory_table',
    # Real trajectory simulation (actual model re-ranking)
    'RealTrajectorySimulator',
    'RealTrajectoryPath',
    'RealTrajectoryStep',
    # Sensitivity analysis
    'SensitivityMetrics',
    'run_random_trajectories',
    'run_top_biased_trajectories',
    # Diversity metrics
    'DiversitySnapshot',
    'TrajectoryDiversity',
    'DiversityAnalysisResult',
    'analyze_diversity',
    'compute_gini_coefficient',
    # Counterfactual analysis
    'CounterfactualAnalyzer',
    'AblationResult',
    'RankingSnapshot',
    'compute_kendall_tau',
]
