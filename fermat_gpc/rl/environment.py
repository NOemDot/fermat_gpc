"""One-step bandit environment for certificate discovery.

For scalability to d up to 199, states are sampled on demand via rejection
sampling rather than enumerated. Rejection sampling is efficient because the
mean of 6 uniform draws in [1, d-1]^6 is exactly 3d - 0.5 · 6/d ≈ 3d - small,
so the acceptance probability is Θ(1/d).
"""
import numpy as np
from math import gcd
from typing import Optional


class FermatCertificateBandit:
    def __init__(self, d: int, seed: int = 0):
        self.d = d
        self.units = [t for t in range(1, d) if gcd(t, d) == 1]
        self.num_units = len(self.units)
        self.pairings = self._all_matchings(6)
        self.num_pairings = len(self.pairings)
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _all_matchings(n: int) -> list:
        items = list(range(n))
        out = []

        def rec(rem, acc):
            if not rem:
                out.append(tuple(sorted(tuple(sorted(p)) for p in acc)))
                return
            first = rem[0]
            for i in range(1, len(rem)):
                second = rem[i]
                rest = [x for x in rem if x != first and x != second]
                rec(rest, acc + [(first, second)])

        rec(items, [])
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return uniq

    def sample_random_tuple(self) -> tuple:
        d = self.d
        target = 3 * d
        # Rejection: for d large this takes O(d) tries on average.
        # For d > 50 we use a smarter direct sampler via composition.
        if d <= 50:
            while True:
                raw = self.rng.integers(1, d, size=6)
                if raw.sum() == target:
                    return tuple(sorted(int(x) for x in raw))
        else:
            return self._sample_composition(target)
    
    def _sample_composition(self, target: int) -> tuple:
        """Direct sampling of a sorted 6-tuple in [1, d-1] summing to target."""
        d = self.d
        while True:
            # Sample a random composition of target into 6 parts in [1, d-1]
            # using stars-and-bars rejection.
            # Efficient for our parameter regime.
            cuts = sorted(self.rng.choice(target - 1, size=5, replace=False))
            parts = np.diff([0] + list(cuts) + [target])
            if np.all(parts >= 1) and np.all(parts <= d - 1):
                return tuple(sorted(int(p) for p in parts))

    def sample_state(self) -> np.ndarray:
        tup = self.sample_random_tuple()
        return np.asarray(tup, dtype=np.float32) / float(self.d)

    def evaluate(self, tup: tuple, twist_idx: int, pairing_idx: int) -> tuple:
        t = self.units[twist_idx % self.num_units]
        pairing = self.pairings[pairing_idx % self.num_pairings]
        twisted = [(t * a) % self.d for a in tup]
        valid = sum(1 for i, j in pairing if (twisted[i] + twisted[j]) % self.d == 0)
        if valid == 3:
            reward = 1.0
        elif valid == 1:
            reward = 0.5
        else:
            reward = 0.0
        info = {
            "tuple": tup,
            "twist": t,
            "twisted": tuple(twisted),
            "valid_pairs": valid,
            "full_certificate": valid == 3,
            "weak_certificate": valid == 1,
        }
        return reward, info