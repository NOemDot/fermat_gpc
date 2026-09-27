"""Train the RL agent at degree d and record metrics."""
import argparse, csv, json, time, sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fermat_gpc.rl.environment import FermatCertificateBandit
from fermat_gpc.rl.policy import FermatPolicyNet
from fermat_gpc.rl.train import train_cem, theoretical_ceiling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="data/rl_metrics.csv")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = FermatCertificateBandit(d=args.d, seed=args.seed)
    policy = FermatPolicyNet(num_units=env.num_units, num_pairings=env.num_pairings)

    # Ceiling requires enumeration; for d > 100 we fall back to an estimate.
    if args.d <= 100:
        ceil = theoretical_ceiling(args.d)
    else:
        ceil = {"n_total": None, "n_full": None, "n_weak": None, "R_max": None}

    t0 = time.time()
    hist = train_cem(env, policy, epochs=args.epochs, batch_size=args.batch)
    dt = time.time() - t0

    row = {
        "d": args.d, "epochs": args.epochs, "batch": args.batch,
        "seconds": round(dt, 2),
        "final_mean_reward": hist["mean_reward"][-1],
        "final_full_rate": hist["full_rate"][-1],
        "final_loss": hist["loss"][-1],
        "R_max": ceil["R_max"],
        "reward_frac_of_ceiling": (
            hist["mean_reward"][-1] / ceil["R_max"] if ceil["R_max"] else None
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()