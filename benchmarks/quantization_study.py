#!/usr/bin/env python3
"""Phase 4/4b: Quantization Comparative Study.

This script runs a comprehensive comparison of different quantization
configurations to find the best tradeoff between compression and accuracy.

Usage:
    uv run python benchmarks/quantization_study.py [--quick] [--extended]

Options:
    --quick     Run with fewer batches for faster results (for debugging)
    --extended  Include Phase 4b extended configs (mixed precision, per-group, etc.)

Output:
    - Comparison table with all configurations
    - Winner selection with rationale
    - JSON results saved to results/f2_phase4/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root and phoenix to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

from phoenix.grok import TransformerConfig
from phoenix.recsys_model import HashConfig, PhoenixModelConfig

from enhancements.optimization.quantization import (
    BitWidth,
    EXTENDED_STUDY_CONFIGS,
    Granularity,
    MixedPrecisionConfig,
    QuantizationConfig,
    QuantizationStudy,
    STUDY_CONFIGS,
    StudyConfig,
    Symmetry,
    format_results_table,
    select_winner,
)


def create_model_config(size: str = "small") -> PhoenixModelConfig:
    """Create model configuration for study.

    Args:
        size: "small" for quick tests, "medium" for realistic benchmark

    Returns:
        PhoenixModelConfig
    """
    if size == "small":
        return PhoenixModelConfig(
            emb_size=128,
            num_actions=19,
            history_seq_len=16,
            candidate_seq_len=8,
            hash_config=HashConfig(
                num_user_hashes=2,
                num_item_hashes=2,
                num_author_hashes=2,
            ),
            model=TransformerConfig(
                emb_size=128,
                key_size=64,
                num_q_heads=4,
                num_kv_heads=2,
                num_layers=4,
                widening_factor=2.0,
                attn_output_multiplier=1.0,
            ),
        )
    else:  # medium
        return PhoenixModelConfig(
            emb_size=256,
            num_actions=19,
            history_seq_len=32,
            candidate_seq_len=16,
            hash_config=HashConfig(
                num_user_hashes=2,
                num_item_hashes=2,
                num_author_hashes=2,
            ),
            model=TransformerConfig(
                emb_size=256,
                key_size=64,
                num_q_heads=8,
                num_kv_heads=4,
                num_layers=8,
                widening_factor=2.0,
                attn_output_multiplier=1.0,
            ),
        )


def run_study(quick: bool = False, extended: bool = False) -> dict:
    """Run the quantization comparative study.

    Args:
        quick: If True, use fewer batches for faster results
        extended: If True, include Phase 4b extended configs

    Returns:
        Dict with results and winner info
    """
    print("=" * 70)
    if extended:
        print("Phase 4b: Extended Quantization Comparative Study")
    else:
        print("Phase 4: Quantization Comparative Study")
    print("=" * 70)
    print()

    # Configure study
    if quick:
        model_config = create_model_config("small")
        study_config = StudyConfig(
            num_eval_batches=10,
            num_warmup_runs=2,
            num_timing_runs=5,
        )
        if extended:
            # Quick extended: baseline + one mixed precision + one per-group
            configs = [
                QuantizationConfig(bit_width=BitWidth.FP16, name="fp16_baseline"),
                QuantizationConfig(
                    bit_width=BitWidth.INT8,
                    granularity=Granularity.PER_CHANNEL,
                    name="int8_channel_sym",
                ),
                QuantizationConfig(
                    use_mixed_precision=True,
                    mixed_precision=MixedPrecisionConfig(
                        ffn_bit_width=BitWidth.INT4,
                        attention_bit_width=BitWidth.INT8,
                    ),
                    name="mixed_int4_ffn_int8_attn",
                ),
                QuantizationConfig(
                    bit_width=BitWidth.INT4,
                    granularity=Granularity.PER_GROUP,
                    group_size=128,
                    name="int4_pergroup128_sym",
                ),
            ]
        else:
            configs = [
                QuantizationConfig(bit_width=BitWidth.FP16, name="fp16_baseline"),
                QuantizationConfig(
                    bit_width=BitWidth.INT8,
                    granularity=Granularity.PER_TENSOR,
                    name="int8_tensor_sym",
                ),
                QuantizationConfig(
                    bit_width=BitWidth.INT8,
                    granularity=Granularity.PER_CHANNEL,
                    name="int8_channel_sym",
                ),
            ]
    else:
        model_config = create_model_config("small")
        study_config = StudyConfig(
            num_eval_batches=50,
            num_warmup_runs=5,
            num_timing_runs=20,
        )
        if extended:
            # Full extended study: base configs + extended configs
            configs = list(STUDY_CONFIGS) + list(EXTENDED_STUDY_CONFIGS)
        else:
            configs = list(STUDY_CONFIGS)

    print(f"Model: emb_size={model_config.emb_size}, "
          f"layers={model_config.model.num_layers}")
    print(f"Configs to test: {len(configs)}")
    print()

    # Run study
    study = QuantizationStudy(model_config, study_config)
    results = study.run(configs)

    # Print results table
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print()
    print(format_results_table(results))
    print()

    # Select winner
    print("=" * 70)
    print("Winner Selection")
    print("=" * 70)
    print()

    winner, selection_details = select_winner(results)

    if winner:
        print(f"Winner: {winner.config_name}")
        print()
        print("Winner Metrics:")
        print(f"  Kendall's tau:     {winner.kendall_tau:.3f}")
        print(f"  Top-3 preserved:   {winner.top3_preserved_rate:.1%}")
        print(f"  Memory reduction:  {winner.memory_reduction_ratio:.1%}")
        print(f"  Latency ratio:     {winner.latency_ratio:.2f}x")
        print()
        print(f"Selection Score: {selection_details['winner_score']:.3f}")
        print()

        print("All Scores (passing configs):")
        for name, score in selection_details['all_scores']:
            marker = " <-- WINNER" if name == winner.config_name else ""
            print(f"  {name}: {score:.3f}{marker}")
    else:
        print("No winner - all configurations failed gates!")
        print(f"Details: {selection_details.get('error', 'Unknown error')}")

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "model_config": {
            "emb_size": model_config.emb_size,
            "num_layers": model_config.model.num_layers,
            "history_len": model_config.history_seq_len,
            "candidate_len": model_config.candidate_seq_len,
        },
        "study_config": {
            "num_eval_batches": study_config.num_eval_batches,
            "num_timing_runs": study_config.num_timing_runs,
            "min_top3_preserved": study_config.min_top3_preserved,
            "min_memory_reduction": study_config.min_memory_reduction,
            "max_latency_ratio": study_config.max_latency_ratio,
        },
        "results": [
            {
                "config_name": r.config_name,
                "kendall_tau": r.kendall_tau,
                "top3_preserved_rate": r.top3_preserved_rate,
                "max_score_diff": r.max_score_diff,
                "mean_score_diff": r.mean_score_diff,
                "memory_reduction_ratio": r.memory_reduction_ratio,
                "latency_p50_ms": r.latency_p50_ms,
                "latency_ratio": r.latency_ratio,
                "passes_all_gates": r.passes_all_gates,
            }
            for r in results
        ],
        "winner": {
            "config_name": winner.config_name if winner else None,
            "score": selection_details.get("winner_score"),
        },
        "selection_details": selection_details,
    }

    return output


def save_results(output: dict, results_dir: str = "results/f2_phase4"):
    """Save results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(results_dir, f"study_results_{timestamp}.json")

    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to: {filepath}")

    # Also save as latest
    latest_path = os.path.join(results_dir, "study_results_latest.json")
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Latest results: {latest_path}")


def main():
    parser = argparse.ArgumentParser(description="Quantization Comparative Study")
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick version with fewer batches"
    )
    parser.add_argument(
        "--extended", action="store_true",
        help="Include Phase 4b extended configs (mixed precision, per-group, etc.)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save results to file"
    )
    args = parser.parse_args()

    output = run_study(quick=args.quick, extended=args.extended)

    if not args.no_save:
        results_dir = "results/f2_phase4b" if args.extended else "results/f2_phase4"
        save_results(output, results_dir=results_dir)

    print()
    print("=" * 70)
    print("Study Complete")
    print("=" * 70)

    # Return exit code based on winner
    if output["winner"]["config_name"]:
        return 0
    else:
        print("WARNING: No configuration passed all gates!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
