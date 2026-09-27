"""Run the PSLQ search on uncertified middle-Hodge tuples at degree d.

Reports (a) the number of tuples for which an integer relation exists in the
log-Gamma basis, and (b) the coefficient bound implied by the working precision.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fermat_gpc.enumeration import iter_sorted_tuples, galois_units
from fermat_gpc.certificates import classify_tuple
from fermat_gpc.pslq_analysis import search_relation
from fermat_gpc.io_utils import set_seeds, timer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--dps", type=int, default=500,
                    help="Working precision in decimal digits")
    ap.add_argument("--maxcoeff", type=int, default=10**30,
                    help="PSLQ coefficient bound")
    ap.add_argument("--max-tuples", type=int, default=None,
                    help="Optional cap on the number of uncertified tuples to test")
    ap.add_argument("--out", type=str, default="data/pslq_results.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seeds(args.seed)
    units = galois_units(args.d)

    tested = 0
    found = 0
    uncertified = 0
    t0 = time.time()

    for tup in iter_sorted_tuples(args.d):
        cls = classify_tuple(tup, args.d, units)
        if cls["weak"]:
            continue
        uncertified += 1
        if args.max_tuples and tested >= args.max_tuples:
            continue
        rel = search_relation(tup, args.d, dps=args.dps, maxcoeff=args.maxcoeff)
        tested += 1
        if rel is not None:
            found += 1

    elapsed = time.time() - t0
    # Coefficient bound: standard rule of thumb is 10^(dps / dim) for a
    # relation of dimension dim searched at dps digits. Here dim = 9
    # (six log-Gamma values plus log pi, log 2, log d).
    bound_exponent = args.dps // 9

    row = {
        "d": args.d,
        "dps": args.dps,
        "maxcoeff": args.maxcoeff,
        "uncertified_total": uncertified,
        "tested": tested,
        "relations_found": found,
        "coefficient_bound_exp": bound_exponent,
        "seconds": round(elapsed, 3),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(row)


if __name__ == "__main__":
    main()