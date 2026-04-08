#!/bin/bash
# Run all MovieLens experiments on a given dataset directory.
# Usage: bash scripts/experiments/run_all_movielens.sh data/ml-1m
#        bash scripts/experiments/run_all_movielens.sh data/ml-100k

set -e

DATA_DIR=${1:-data/ml-1m}
echo "============================================"
echo "Running all experiments on: $DATA_DIR"
echo "============================================"

echo ""
echo ">>> Foundation verification"
uv run python scripts/experiments/verify_movielens_foundation.py --data "$DATA_DIR"

echo ""
echo ">>> Labels-not-loss (all groups)"
uv run python scripts/experiments/run_movielens_labels_not_loss.py --data "$DATA_DIR" --all

echo ""
echo ">>> LOSO + data budget"
uv run python scripts/experiments/run_movielens_loso.py --data "$DATA_DIR" --all

echo ""
echo ">>> Scalarization baseline"
uv run python scripts/experiments/run_scalarization_baseline.py --data "$DATA_DIR" --dataset movielens

echo ""
echo ">>> Goodhart"
uv run python scripts/experiments/run_movielens_goodhart.py --data "$DATA_DIR"

echo ""
echo "============================================"
echo "All experiments complete for $DATA_DIR"
echo "Results in results/$(basename $DATA_DIR)_*.json"
echo "============================================"
