"""Streaming enumeration of middle-Hodge tuples and Galois orbits.

The middle-Hodge condition for a Fermat fourfold X_d^4 is:

    a_i in {1, ..., d-1},  sum(a_i) = 3d

for k = 6 indices. We never materialise the full tuple list: all functions
return iterators, so memory is O(k) regardless of d.
"""
from math import gcd
from typing import Iterator, Tuple, Optional


def galois_units(d: int) -> list[int]:
    """Elements of (Z/dZ)^*, i.e. t in [1, d-1] with gcd(t, d) = 1."""
    return [t for t in range(1, d) if gcd(t, d) == 1]


def iter_sorted_tuples(d: int, k: int = 6) -> Iterator[Tuple[int, ...]]:
    """Yield sorted tuples (a_1 <= ... <= a_k), 1 <= a_i <= d-1, sum = (k/2)*d."""
    target = (k // 2) * d

    def rec(prefix: tuple, cur_sum: int, depth: int):
        if depth == k:
            if cur_sum == target:
                yield prefix
            return
        lo = prefix[-1] if prefix else 1
        hi = min(d - 1, target - cur_sum - (k - depth - 1))
        for v in range(lo, hi + 1):
            rem = k - depth - 1
            if cur_sum + v + rem > target:
                break
            if cur_sum + v + rem * (d - 1) < target:
                continue
            yield from rec(prefix + (v,), cur_sum + v, depth + 1)

    yield from rec((), 0, 0)


def count_tuples(d: int, k: int = 6) -> int:
    """Exact count of middle-Hodge tuples at degree d."""
    # Closed-form via the same recursion but counting only.
    from functools import lru_cache

    target = (k // 2) * d

    @lru_cache(maxsize=None)
    def f(min_val: int, cur_sum: int, depth: int) -> int:
        if depth == k:
            return 1 if cur_sum == target else 0
        total = 0
        hi = min(d - 1, target - cur_sum - (k - depth - 1))
        for v in range(min_val, hi + 1):
            rem = k - depth - 1
            if cur_sum + v + rem > target:
                break
            if cur_sum + v + rem * (d - 1) < target:
                continue
            total += f(v, cur_sum + v, depth + 1)
        return total

    return f(1, 0, 0)


def canonical_orbit_representative(tup: tuple, d: int,
                                    units: Optional[list[int]] = None) -> tuple:
    """Lexicographically smallest tuple in the Galois orbit of tup."""
    if units is None:
        units = galois_units(d)
    best = tup
    for t in units[1:]:
        twisted = tuple(sorted((t * a) % d for a in tup))
        if twisted < best:
            best = twisted
    return best


def is_canonical(tup: tuple, d: int, units=None) -> bool:
    return canonical_orbit_representative(tup, d, units) == tup


def iter_orbit_representatives(d: int, k: int = 6) -> Iterator[Tuple[int, ...]]:
    """Yield one representative per Galois orbit, in lex order."""
    units = galois_units(d)
    for tup in iter_sorted_tuples(d, k):
        if canonical_orbit_representative(tup, d, units) == tup:
            yield tup