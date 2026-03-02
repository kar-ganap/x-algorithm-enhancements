#!/usr/bin/env python3
"""Run verification suite on trained synthetic model.

Usage:
    uv run python scripts/verify_synthetic.py [--model PATH]
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))

from enhancements.data.synthetic_adapter import SyntheticTwitterPhoenixAdapter
from enhancements.verification.suite import (
    VerificationConfig,
    run_verification_suite,
)
from phoenix.runners import ModelRunner, RecsysInferenceRunner


def main():
    parser = argparse.ArgumentParser(description="Run verification suite")
    parser.add_argument(
        "--model", type=str, default="models/synthetic_twitter/best_model.pkl",
        help="Path to trained model"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/synthetic_twitter",
        help="Path to synthetic data"
    )
    parser.add_argument(
        "--output", type=str, default="results/synthetic_verification.json",
        help="Output path for results"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick verification with smaller samples"
    )
    parser.add_argument(
        "--skip-embedding", action="store_true",
        help="Skip embedding probe tests"
    )
    parser.add_argument(
        "--skip-behavioral", action="store_true",
        help="Skip behavioral tests"
    )
    parser.add_argument(
        "--skip-action", action="store_true",
        help="Skip action differentiation tests"
    )
    parser.add_argument(
        "--skip-counterfactual", action="store_true",
        help="Skip counterfactual tests"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA VERIFICATION SUITE")
    print("=" * 70)
    print()

    # Load model
    print(f"Loading model from {args.model}...")
    with open(args.model, "rb") as f:
        checkpoint = pickle.load(f)

    model_config = checkpoint["model_config"]
    params = {
        "model": checkpoint["model"],
        "embeddings": checkpoint["embeddings"],
    }
    # Load classifier if available (for archetype-based action prediction)
    if "classifier" in checkpoint:
        params["classifier"] = checkpoint["classifier"]
        print("  Classifier: loaded")
    # Load action predictor if available (fallback)
    if "action_predictor" in checkpoint:
        params["action_predictor"] = checkpoint["action_predictor"]
        print("  Action predictor: loaded")
    print(f"  Model: emb_size={model_config.emb_size}, layers={model_config.model.num_layers}")
    print()

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    data_path = Path(args.data_dir)
    with open(data_path / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    with open(data_path / "splits.pkl", "rb") as f:
        splits = pickle.load(f)
    print(f"  Users: {dataset.num_users}")
    print(f"  Posts: {dataset.num_posts}")
    print()

    # Create adapter
    print("Creating adapter...")
    adapter = SyntheticTwitterPhoenixAdapter(dataset, model_config)
    adapter.set_splits(splits["train"], splits["val"], splits["test"])
    adapter.set_embedding_params(params["embeddings"])
    print()

    # Create runner
    print("Creating inference runner...")
    model_runner = ModelRunner(model=model_config)
    runner = RecsysInferenceRunner(runner=model_runner, name="verification")
    runner.initialize()
    print()

    # Configure verification
    if args.quick:
        config = VerificationConfig(
            user_sample_size=100,
            post_sample_size=200,
            behavioral_samples=20,
            action_samples=30,
            block_tests=20,
            flip_tests=10,
        )
    else:
        config = VerificationConfig()

    # Run verification
    print("Running verification suite...")
    print()

    results = run_verification_suite(
        adapter, dataset, runner, params,
        config=config,
        skip_embedding=args.skip_embedding,
        skip_behavioral=args.skip_behavioral,
        skip_action=args.skip_action,
        skip_counterfactual=args.skip_counterfactual,
    )

    # Print report
    print()
    print(results.report())

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results.to_dict(), f, indent=2)
    print(f"Results saved to: {output_path}")

    return 0 if results.all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
