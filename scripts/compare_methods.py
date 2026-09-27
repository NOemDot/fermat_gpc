"""Run baseline and RL at a series of odd degrees and compare wall-clock time."""
import argparse, subprocess, sys, csv, json
from pathlib import Path

DEGREES_DEFAULT = [13, 25, 51, 101, 199]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrees", type=int, nargs="+", default=DEGREES_DEFAULT)
    ap.add_argument("--rl-epochs", type=int, default=15)
    ap.add_argument("--rl-batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    py = sys.executable
    here = Path(__file__).resolve().parent
    for d in args.degrees:
        run([py, str(here / "baseline.py"), "--d", str(d),
             "--out", "data/baseline.csv"])
        run([py, str(here / "train_rl.py"), "--d", str(d),
             "--epochs", str(args.rl_epochs), "--batch", str(args.rl_batch),
             "--seed", str(args.seed), "--out", "data/rl_metrics.csv"])

    # Merge
    import pandas as pd
    base = pd.read_csv("data/baseline.csv").drop_duplicates("d", keep="last")
    rl = pd.read_csv("data/rl_metrics.csv").drop_duplicates("d", keep="last")
    merged = base.merge(rl, on="d", how="outer")
    merged["time_ratio"] = merged["seconds_y"] / merged["seconds_x"]
    merged.to_csv("data/comparison.csv", index=False)
    print("\nMerged comparison saved to data/comparison.csv")
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()