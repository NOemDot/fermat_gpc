"""Gamma-product periods and log-period algebra."""
import mpmath as mp


def fermat_period(tup: tuple, d: int, dps: int = 500) -> mp.mpf:
    """P_a = (2*pi*i)^4 / (4 d^4) * prod Gamma(a_i / d)."""
    mp.mp.dps = dps
    prefactor = (2 * mp.pi * mp.j) ** 4 / (4 * mp.mpf(d) ** 4)
    product = mp.mpf(1)
    for a in tup:
        product *= mp.gamma(mp.mpf(a) / d)
    return prefactor * product


def log_period_basis(tup: tuple, d: int, dps: int = 500) -> list:
    """Return the log-Gamma basis L(a) = {log Gamma(a_i/d)} ∪ {log pi, log 2, log d}."""
    mp.mp.dps = dps
    basis = [mp.log(mp.gamma(mp.mpf(a) / d)) for a in tup]
    basis.append(mp.log(mp.pi))
    basis.append(mp.log(mp.mpf(2)))
    basis.append(mp.log(mp.mpf(d)))
    return basis