#!/usr/bin/env python3
"""Hyperparameter search for cls_weight.

Tests different cls_weight values and compares all verification metrics.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "phoenix"))


def run_training(cls_weight: float, epochs: int = 10, patience: int = 5) -> str:
    """Run training with given cls_weight and return model path."""
    model_path = f"models/synthetic_twitter/model_cls{cls_weight}.pkl"

    cmd = [
        "uv", "run", "python", "scripts/train_synthetic.py",
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--cls-weight", str(cls_weight),
        "--output", model_path,
    ]

    print(f"\n{'='*60}")
    print(f"Training with cls_weight={cls_weight}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Training failed: {result.stderr}")
        return None

    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return model_path


def run_verification(model_path: str) -> dict:
    """Run verification and return results."""
    cmd = [
        "uv", "run", "python", "scripts/verify_synthetic.py",
        "--model", model_path,
        "--output", "/dev/null",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse key metrics from output
    output = result.stdout + result.stderr
    metrics = {}

    # Extract metrics using simple parsing
    for line in output.split("\n"):
        if "User archetype clustering:" in line:
            try:
                metrics["user_clustering"] = float(line.split(":")[1].split()[0])
            except:
                pass
        elif "Tests passed:" in line and "ACTION" not in output[output.find(line)-100:output.find(line)]:
            # This is behavioral tests
            pass
        elif "Lurker repost ratio:" in line:
            try:
                metrics["lurker_repost_ratio"] = float(line.split(":")[1].strip())
            except:
                pass
        elif "Power user repost ratio:" in line:
            try:
                metrics["power_user_repost_ratio"] = float(line.split(":")[1].strip())
            except:
                pass
        elif "power_vs_lurker_repost:" in line:
            try:
                val = line.split(":")[1].strip().split("x")[0]
                metrics["power_vs_lurker"] = float(val)
            except:
                pass
        elif "Block effect rate:" in line:
            try:
                metrics["block_effect"] = float(line.split(":")[1].strip().replace("%", ""))
            except:
                pass

    # Count passed tests
    metrics["action_tests_passed"] = output.count("[PASS]")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Tune cls_weight")
    parser.add_argument("--weights", type=str, default="1.0,2.0,3.0,5.0",
                        help="Comma-separated cls_weight values to try")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")]

    results = {}

    for cls_weight in weights:
        model_path = run_training(cls_weight, args.epochs, args.patience)
        if model_path:
            metrics = run_verification(model_path)
            results[cls_weight] = metrics
            print(f"\ncls_weight={cls_weight}: {metrics}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'cls_weight':<12} {'user_clust':<12} {'lurker_rt':<12} {'power_rt':<12} {'ratio':<12} {'tests':<8}")
    print("-"*80)

    for w, m in sorted(results.items()):
        print(f"{w:<12.1f} {m.get('user_clustering', 0):<12.3f} "
              f"{m.get('lurker_repost_ratio', 0):<12.3f} "
              f"{m.get('power_user_repost_ratio', 0):<12.3f} "
              f"{m.get('power_vs_lurker', 0):<12.2f}x "
              f"{m.get('action_tests_passed', 0):<8}")

    # Save results
    with open("results/cls_weight_tuning.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/cls_weight_tuning.json")


if __name__ == "__main__":
    main()
