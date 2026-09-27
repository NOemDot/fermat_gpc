#!/usr/bin/env bash
# Reproduce the central results: baseline counts, RL metrics, comparison table.
set -euo pipefail

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

mkdir -p data

# Baseline at the two reference degrees
python scripts/baseline.py --d 13  --out data/baseline.csv
python scripts/baseline.py --d 199 --out data/baseline.csv

# RL training at the two reference degrees
python scripts/train_rl.py --d 13  --epochs 15 --batch 512 --seed 0 --out data/rl_metrics.csv
python scripts/train_rl.py --d 199 --epochs 15 --batch 512 --seed 0 --out data/rl_metrics.csv

# Full comparison across the Jumagulov range
python scripts/compare_methods.py --degrees 13 25 51 101 199

echo "Done. See data/comparison.csv"