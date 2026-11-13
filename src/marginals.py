"""Discretization of marginal laws and convex-order verification.

Implements the conditional-average discretization of Appendix A.1 and the
discrete convex-order check used throughout the paper (mu1 <=_c mu2 iff
discrete call prices of mu2 dominate those of mu1 at every strike in the
union of the two supports).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def discretize_lognormal(v: float, n_support: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Conditional-average discretization of a lognormal law with E[X] = 1.

    log X ~ N(-v/2, v). Following Appendix A.1, both dates use the percentile
    cutoffs u_j = sin^2(pi*j / (2*n_support)), j = 0,...,n_support. Masses are
    p_j = u_j - u_{j-1} and support points are the conditional average of X
    over (u_{j-1}, u_j]:

        x_j = [Phi(Phi^-1(u_j) - sqrt(v)) - Phi(Phi^-1(u_{j-1}) - sqrt(v))] / p_j

    This preserves E[X] = 1 exactly (up to floating-point round-off) because
    integrating the lognormal density in conditional-average form is the
    Phi(z - sqrt(v)) identity for E[X * 1{X <= x}].

    Parameters
    ----------
    v : total log-variance (T * sigma^2)
    n_support : number of discretization points

    Returns
    -------
    (p, x): probability masses and support points, each of length n_support
    """
    if v <= 0:
        raise ValueError("v must be positive")
    j = np.arange(0, n_support + 1)
    u = np.sin(np.pi * j / (2 * n_support)) ** 2
    u[0], u[-1] = 0.0, 1.0
    p = np.diff(u)
    # Avoid Phi^-1(0) = -inf and Phi^-1(1) = +inf by computing the end terms
    # of the F(z - sqrt(v)) telescoping sum directly from u (not through z).
    z = norm.ppf(u)
    tail = norm.cdf(z - np.sqrt(v))  # tail[0] = Phi(-inf - ...) = 0, tail[-1] = Phi(+inf) = 1
    tail[0] = 0.0
    tail[-1] = 1.0
    x = np.diff(tail) / p
    assert np.all(p > 0)
    assert abs(p.sum() - 1.0) < 1e-10
    assert abs(np.dot(p, x) - 1.0) < 1e-6
    return p, x


def discrete_call_price(p: np.ndarray, x: np.ndarray, strikes: np.ndarray) -> np.ndarray:
    """Vector of discrete call prices E[(X-K)^+] at each strike K."""
    payoff = np.maximum(x[None, :] - strikes[:, None], 0.0)
    return payoff @ p


def check_convex_order(p1: np.ndarray, x1: np.ndarray, p2: np.ndarray, x2: np.ndarray,
                        tol: float = 1e-9) -> bool:
    """Check mu1 <=_c mu2 by comparing discrete call prices at every strike in
    the union of the two supports (Sec. 2 / Appendix A.1 convex-order check).
    """
    strikes = np.union1d(x1, x2)
    c1 = discrete_call_price(p1, x1, strikes)
    c2 = discrete_call_price(p2, x2, strikes)
    return bool(np.all(c2 - c1 >= -tol))


def reference_independent_gaussian_price(v1: float, v2: float) -> float:
    """E[(X2 - X1)^+] under independent lognormal legs sharing E[Xi] = 1,
    log Xi ~ N(-vi/2, vi): closed form 2*Phi(sqrt(w)/2) - 1 with w = v2 - v1
    (Sec. 5.1.1 reference price under independent Gaussian log returns).
    """
    w = v2 - v1
    if w < 0:
        raise ValueError("v2 must exceed v1 for this closed form")
    return 2.0 * norm.cdf(np.sqrt(w) / 2.0) - 1.0
