"""PSLQ relation search on the Fermat log-period algebra."""
import mpmath as mp
from typing import Optional
from .periods import log_period_basis


def search_relation(tup: tuple, d: int, dps: int = 500,
                    maxcoeff: int = 10**30) -> Optional[tuple]:
    """Return an integer relation among the log-period basis, or None."""
    mp.mp.dps = dps
    vec = log_period_basis(tup, d, dps)
    rel = mp.pslq(vec, maxcoeff=maxcoeff, maxsteps=10**6)
    return tuple(rel) if rel is not None else None