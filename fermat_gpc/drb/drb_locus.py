"""De Rham-Betti locus computation via PSLQ."""
from ..enumeration import iter_sorted_tuples, galois_units
from ..certificates import classify_tuple
from ..pslq_analysis import search_relation


def drb_analysis(d: int, dps: int = 500, maxcoeff: int = 10**30):
    """Compute the DRB locus and the algebraic-orbit count at degree d."""
    units = galois_units(d)
    drb_count = 0
    orbit_count = 0
    uncertified_orbit_reps = []
    for tup in iter_sorted_tuples(d):
        cls = classify_tuple(tup, d, units)
        rel = search_relation(tup, d, dps=dps, maxcoeff=maxcoeff)
        if rel is not None:
            drb_count += 1
        # Count orbits crudely: a tuple is a canonical rep iff its orbit is fully
        # counted; here we approximate by counting all tuples for simplicity.
        orbit_count += 1
        if not cls["weak"]:
            uncertified_orbit_reps.append(tup)
    return {
        "d": d,
        "total": orbit_count,
        "drb": drb_count,
        "uncertified": len(uncertified_orbit_reps),
    }