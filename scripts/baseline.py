"""Streaming baseline: classify all middle-Hodge tuples at degree d."""
import argparse, csv, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fermat_gpc import iter_sorted_tuples, galois_units, classify_tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--out", type=str, default="data/baseline.csv")
    args = ap.parse_args()

    units = galois_units(args.d)
    n_total = n_full = n_weak = 0
    t0 = time.time()
    for tup in iter_sorted_tuples(args.d):
        n_total += 1
        cls = classify_tuple(tup, args.d, units)
        n_full += int(cls["full_pair"])
        n_weak += int(cls["weak"])
    dt = time.time() - t0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    row = {
        "d": args.d,
        "total": n_total,
        "full": n_full,
        "weak": n_weak,
        "full_pct": 100 * n_full / n_total,
        "weak_pct": 100 * n_weak / n_total,
        "uncertified": n_total - n_weak,
        "seconds": round(dt, 3),
    }
    write_header = not Path(args.out).exists()
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(row)


if __name__ == "__main__":
    main()