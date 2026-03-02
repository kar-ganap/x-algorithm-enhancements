"""Experiment configuration for alternative loss function testing.

Defines the hyperparameter grid and experiment configurations for
systematic comparison of loss functions.
"""

import itertools
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from enhancements.reward_modeling.alternative_losses import (
    LossConfig,
    LossType,
    StakeholderType,
)


@dataclass
class ExperimentConfig:
    """Master configuration for loss function experiments."""

    # Experiment identification
    experiment_name: str = "alternative_loss_comparison"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Data configuration (scaled up for comprehensive experiments)
    n_training_samples: int = 50000  # 10x more data for reliable gradients
    n_content_items: int = 2000      # 4x more content for robust ranking comparisons
    n_users: int = 100
    random_seed: int = 42

    # Training configuration (shared baseline)
    base_learning_rate: float = 0.01
    base_num_epochs: int = 30  # 30 epochs sufficient (92%+ accuracy, marginal gain beyond)
    base_batch_size: int = 64
    base_l2_weight: float = 0.001

    # Evaluation
    run_causal_tests: bool = True
    run_all_8_tests: bool = True
    save_intermediate: bool = True


# =============================================================================
# Hyperparameter Grids
# =============================================================================

# Finer hyperparameter grids for comprehensive search
MARGIN_VALUES = [0.05, 0.1, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
CALIBRATION_WEIGHTS = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
CONSTRAINT_WEIGHTS = [0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
RERANK_ALPHAS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]

# Stakeholders to test
STAKEHOLDERS = [StakeholderType.USER, StakeholderType.PLATFORM, StakeholderType.SOCIETY]


def generate_all_configs(base_config: ExperimentConfig) -> list[LossConfig]:
    """Generate all experiment configurations.

    Creates a cross-product of:
    - Loss types × hyperparameters × stakeholders

    Returns:
        List of LossConfig objects to run
    """
    configs = []

    # Baseline: Standard Bradley-Terry
    for stakeholder in STAKEHOLDERS:
        configs.append(LossConfig(
            loss_type=LossType.BRADLEY_TERRY,
            stakeholder=stakeholder,
            learning_rate=base_config.base_learning_rate,
            num_epochs=base_config.base_num_epochs,
            batch_size=base_config.base_batch_size,
            l2_weight=base_config.base_l2_weight,
        ))

    # A1: Margin-BT
    for margin, stakeholder in itertools.product(MARGIN_VALUES, STAKEHOLDERS):
        configs.append(LossConfig(
            loss_type=LossType.MARGIN_BT,
            stakeholder=stakeholder,
            learning_rate=base_config.base_learning_rate,
            num_epochs=base_config.base_num_epochs,
            batch_size=base_config.base_batch_size,
            l2_weight=base_config.base_l2_weight,
            margin=margin,
        ))

    # A2: Calibrated-BT
    for cal_weight, stakeholder in itertools.product(CALIBRATION_WEIGHTS, STAKEHOLDERS):
        configs.append(LossConfig(
            loss_type=LossType.CALIBRATED_BT,
            stakeholder=stakeholder,
            learning_rate=base_config.base_learning_rate,
            num_epochs=base_config.base_num_epochs,
            batch_size=base_config.base_batch_size,
            l2_weight=base_config.base_l2_weight,
            calibration_weight=cal_weight,
        ))

    # C1: Constrained-BT
    for constraint_weight, stakeholder in itertools.product(CONSTRAINT_WEIGHTS, STAKEHOLDERS):
        configs.append(LossConfig(
            loss_type=LossType.CONSTRAINED_BT,
            stakeholder=stakeholder,
            learning_rate=base_config.base_learning_rate,
            num_epochs=base_config.base_num_epochs,
            batch_size=base_config.base_batch_size,
            l2_weight=base_config.base_l2_weight,
            constraint_weight=constraint_weight,
        ))

    return configs


def generate_quick_test_configs(base_config: ExperimentConfig) -> list[LossConfig]:
    """Generate a small set of configs for quick sanity testing.

    Returns one config per loss type with middle-of-range hyperparameters.
    """
    configs = []

    # One of each loss type with Platform stakeholder
    stakeholder = StakeholderType.PLATFORM

    # Baseline
    configs.append(LossConfig(
        loss_type=LossType.BRADLEY_TERRY,
        stakeholder=stakeholder,
        learning_rate=base_config.base_learning_rate,
        num_epochs=10,  # Quick test - few epochs
        batch_size=base_config.base_batch_size,
        l2_weight=base_config.base_l2_weight,
    ))

    # Margin-BT
    configs.append(LossConfig(
        loss_type=LossType.MARGIN_BT,
        stakeholder=stakeholder,
        learning_rate=base_config.base_learning_rate,
        num_epochs=10,
        batch_size=base_config.base_batch_size,
        l2_weight=base_config.base_l2_weight,
        margin=0.5,
    ))

    # Calibrated-BT
    configs.append(LossConfig(
        loss_type=LossType.CALIBRATED_BT,
        stakeholder=stakeholder,
        learning_rate=base_config.base_learning_rate,
        num_epochs=10,
        batch_size=base_config.base_batch_size,
        l2_weight=base_config.base_l2_weight,
        calibration_weight=1.0,
    ))

    # Constrained-BT
    configs.append(LossConfig(
        loss_type=LossType.CONSTRAINED_BT,
        stakeholder=stakeholder,
        learning_rate=base_config.base_learning_rate,
        num_epochs=10,
        batch_size=base_config.base_batch_size,
        l2_weight=base_config.base_l2_weight,
        constraint_weight=10.0,
    ))

    return configs


def get_config_name(config: LossConfig) -> str:
    """Generate a unique name for a config."""
    name = f"{config.loss_type.value}_{config.stakeholder.value}"

    if config.loss_type == LossType.MARGIN_BT:
        name += f"_m{config.margin}"
    elif config.loss_type == LossType.CALIBRATED_BT:
        name += f"_cal{config.calibration_weight}"
    elif config.loss_type == LossType.CONSTRAINED_BT:
        name += f"_cw{config.constraint_weight}"

    return name


def get_rerank_configs() -> list[dict[str, Any]]:
    """Get post-hoc reranking configurations.

    These don't require training, just evaluation with different alphas.
    """
    configs = []

    for alpha in RERANK_ALPHAS:
        for stakeholder in STAKEHOLDERS:
            configs.append({
                "alpha": alpha,
                "stakeholder": stakeholder,
                "name": f"posthoc_{stakeholder.value}_a{alpha}",
            })

    return configs


# =============================================================================
# Summary Statistics
# =============================================================================

def count_experiments() -> dict[str, int]:
    """Count total experiments in the grid."""
    counts = {
        "baseline": len(STAKEHOLDERS),
        "margin_bt": len(MARGIN_VALUES) * len(STAKEHOLDERS),
        "calibrated_bt": len(CALIBRATION_WEIGHTS) * len(STAKEHOLDERS),
        "constrained_bt": len(CONSTRAINT_WEIGHTS) * len(STAKEHOLDERS),
        "posthoc": len(RERANK_ALPHAS) * len(STAKEHOLDERS),
    }
    counts["total_training"] = sum(v for k, v in counts.items() if k != "posthoc")
    counts["total_evaluation"] = counts["posthoc"]
    counts["total"] = counts["total_training"] + counts["total_evaluation"]

    return counts


if __name__ == "__main__":
    # Print experiment summary
    counts = count_experiments()

    print("=" * 60)
    print("EXPERIMENT GRID SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (Bradley-Terry): {counts['baseline']}")
    print(f"Margin-BT: {counts['margin_bt']} ({len(MARGIN_VALUES)} margins × {len(STAKEHOLDERS)} stakeholders)")
    print(f"Calibrated-BT: {counts['calibrated_bt']} ({len(CALIBRATION_WEIGHTS)} weights × {len(STAKEHOLDERS)} stakeholders)")
    print(f"Constrained-BT: {counts['constrained_bt']} ({len(CONSTRAINT_WEIGHTS)} weights × {len(STAKEHOLDERS)} stakeholders)")
    print(f"Post-Hoc Reranking: {counts['posthoc']} ({len(RERANK_ALPHAS)} alphas × {len(STAKEHOLDERS)} stakeholders)")
    print("-" * 60)
    print(f"Total training experiments: {counts['total_training']}")
    print(f"Total evaluation-only: {counts['total_evaluation']}")
    print(f"TOTAL: {counts['total']}")
    print("=" * 60)

    # Estimate time
    est_time_per_experiment = 3  # minutes
    total_time = counts['total_training'] * est_time_per_experiment
    print(f"\nEstimated training time: ~{total_time} minutes ({total_time / 60:.1f} hours)")
