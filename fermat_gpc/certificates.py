"""O(k) certificate tests for full-pair and weak decomposability.

Full-pair: under some twist t, the twisted multiset pairs as {x, d-x}.
Weak:      under some twist t, there exists at least one pair summing to d.
           (The complementary four indices then sum to 0 mod d automatically,
            since the total sum is 3d ≡ 0 mod d.)

Both tests are O(k) per twist using a Counter, not O(k!).
"""
from collections import Counter
from math import gcd
from typing import Optional


def has_full_pair_decomposition(twisted, d: int) -> bool:
    """True iff the multiset can be partitioned into disjoint pairs each summing to d."""
    counts = Counter(twisted)
    # If d is odd, no element pairs with itself.
    # If d is even, x = d/2 pairs with itself; counts[d/2] must be even.
    for x, c in counts.items():
        y = (d - x) % d
        if x == y:
            if c % 2 != 0:
                return False
        else:
            if counts.get(y, 0) != c:
                return False
    return True


def has_zero_sum_pair(twisted, d: int) -> bool:
    """True iff some pair of distinct indices sums to d mod d."""
    counts = Counter(twisted)
    for x, c in counts.items():
        y = (d - x) % d
        if x == y:
            if c >= 2:
                return True
        else:
            if counts.get(y, 0) > 0:
                return True
    return False


def classify_tuple(tup: tuple, d: int, units: Optional[list[int]] = None) -> dict:
    """Return {'full_pair': bool, 'weak': bool} over all Galois twists."""
    if units is None:
        units = [t for t in range(1, d) if gcd(t, d) == 1]
    full = weak = False
    for t in units:
        twisted = tuple((t * a) % d for a in tup)
        if not full and has_full_pair_decomposition(twisted, d):
            full = True
        if not weak and has_zero_sum_pair(twisted, d):
            weak = True
        if full and weak:
            break
    return {"full_pair": full, "weak": weak}