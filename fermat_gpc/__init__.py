"""Fermat fourfold GPC/Hodge toolkit."""
__version__ = "0.1.0"

from .enumeration import (
    iter_sorted_tuples,
    galois_units,
    canonical_orbit_representative,
    count_tuples,
)
from .certificates import classify_tuple, has_full_pair_decomposition, has_zero_sum_pair

__all__ = [
    "iter_sorted_tuples",
    "galois_units",
    "canonical_orbit_representative",
    "count_tuples",
    "classify_tuple",
    "has_full_pair_decomposition",
    "has_zero_sum_pair",
]