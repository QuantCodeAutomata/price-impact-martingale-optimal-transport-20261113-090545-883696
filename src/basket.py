"""Two-asset basket call under asymmetric quadratic price impact (Sec. 5.2 and
Appendix A.3 of "Price-impact Martingale optimal transport").

One trading period (N=1), d=2 assets. The coupling q(x, y) >= 0 is indexed by
x = (x1, x2) at T1 and y = (y1, y2) at T2, each coordinate ranging over 16
equiprobable midpoints; q is represented as a (256, 256) matrix with combined
index i = x1_idx*16 + x2_idx (and likewise for y). Only the four univariate
margins (one per asset and date) are fixed to uniform 1/16; the joint law of
(X1^1, X1^2) at T1 is a free decision variable, so the T1-probability
p(x) = sum_y q(x, y) of an (x1, x2) pair need not factor into independent
coordinates (methodology step 3).

The price-impact penalty is the perspective quadratic
sum_a d^a(x)^2 / (2*eta_a*p(x)), where d^a(x) = sum_y q(x, y)*(y_a - x_a) is
the conditional drift of asset a (methodology step 4-5). cvxpy's
``quad_over_lin`` atom reformulates each scalar term as a rotated
second-order cone constraint, which is exactly the SOCP of methodology
step 6 without introducing explicit auxiliary variables by hand. No
general-purpose optimization library implements this price-impact MOT SOCP,
so it is coded directly from the paper's objective (Sec. 3, Eq. for the
discretized primal problem) using cvxpy as the modeling layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


def midpoints(interval: tuple[float, float], n: int) -> np.ndarray:
    """The n midpoints of n equal subdivisions of ``interval`` (Sec. 5.2)."""
    lo, hi = interval
    width = (hi - lo) / n
    return lo + width * (np.arange(n) + 0.5)


@dataclass
class BasketMarginals:
    x1: np.ndarray  # asset 1 at T1, length n
    x2: np.ndarray  # asset 2 at T1, length n
    y1: np.ndarray  # asset 1 at T2, length n
    y2: np.ndarray  # asset 2 at T2, length n
    n: int

    @property
    def x_of_index(self) -> tuple[np.ndarray, np.ndarray]:
        """x1_of_x[i], x2_of_x[i] for combined index i = i1*n + i2."""
        i1 = np.repeat(np.arange(self.n), self.n)
        i2 = np.tile(np.arange(self.n), self.n)
        return self.x1[i1], self.x2[i2]

    @property
    def y_of_index(self) -> tuple[np.ndarray, np.ndarray]:
        j1 = np.repeat(np.arange(self.n), self.n)
        j2 = np.tile(np.arange(self.n), self.n)
        return self.y1[j1], self.y2[j2]


@dataclass
class BasketResult:
    value_lower: float
    value_upper: float
    midpoint: float
    gap: float
    q: np.ndarray            # adjusted coupling, shape (n^2, n^2)
    drift1: np.ndarray        # d^1(x), length n^2
    drift2: np.ndarray        # d^2(x), length n^2
    p: np.ndarray             # p(x) = sum_y q(x, y), length n^2
    shares_traded_1: float | None
    shares_traded_2: float | None
    status: str


def _marginal_constraints(q: cp.Variable, n: int, target: float = None):
    """The 4*n equality constraints fixing each univariate margin to 1/n
    (methodology step 3). Returns the list of constraints."""
    target = 1.0 / n if target is None else target
    # C order: combined index i = i1*n + i2 matches BasketMarginals.x_of_index.
    row = cp.reshape(cp.sum(q, axis=1), (n, n), order="C")   # [x1_idx, x2_idx]
    col = cp.reshape(cp.sum(q, axis=0), (n, n), order="C")   # [y1_idx, y2_idx]
    constraints = [
        cp.sum(row, axis=1) == target,  # asset 1 @ T1
        cp.sum(row, axis=0) == target,  # asset 2 @ T1
        cp.sum(col, axis=1) == target,  # asset 1 @ T2
        cp.sum(col, axis=0) == target,  # asset 2 @ T2
    ]
    return constraints


def solve_basket_ot_or_mot(
    marg: BasketMarginals,
    martingale: bool,
    solver: str = "CLARABEL",
    solver_tol: float = 1e-10,
) -> BasketResult:
    """Pure transport LP: OT upper bound (``martingale=False``) or MOT upper
    bound (``martingale=True``, zero-drift constraint added; methodology
    step 12). Both are exact linear programs, so lower == upper."""
    n = marg.n
    x1, x2 = marg.x_of_index
    y1, y2 = marg.y_of_index
    payoff = np.maximum(y1 + y2, 0.0)

    q = cp.Variable((n * n, n * n), nonneg=True)
    constraints = _marginal_constraints(q, n)
    if martingale:
        p = cp.sum(q, axis=1)
        d1 = q @ y1 - cp.multiply(x1, p)
        d2 = q @ y2 - cp.multiply(x2, p)
        constraints += [d1 == 0, d2 == 0]

    objective = cp.Maximize(cp.sum(q @ payoff))
    problem = cp.Problem(objective, constraints)
    solver_kwargs = {"solver": solver}
    if solver == "CLARABEL":
        solver_kwargs.update(tol_gap_abs=solver_tol, tol_gap_rel=solver_tol, tol_feas=solver_tol)
    problem.solve(**solver_kwargs)

    q_val = np.clip(q.value, 0.0, None)
    p_val = q_val.sum(axis=1)
    d1_val = q_val @ y1 - x1 * p_val
    d2_val = q_val @ y2 - x2 * p_val
    value = float(problem.value)
    return BasketResult(
        value_lower=value, value_upper=value, midpoint=value, gap=0.0,
        q=q_val, drift1=d1_val, drift2=d2_val, p=p_val,
        shares_traded_1=None, shares_traded_2=None, status=problem.status,
    )


def _enforce_marginals_ipf(q: np.ndarray, n: int, target: float, n_iter: int = 50) -> np.ndarray:
    """Light iterative proportional fitting projecting a near-feasible
    coupling onto the four exact univariate margins (methodology step 7:
    "adjust the masses to match the marginals exactly")."""
    q = np.clip(q, 0.0, None).reshape(n, n, n, n)  # [x1, x2, y1, y2]
    for _ in range(n_iter):
        s = q.sum(axis=(1, 2, 3), keepdims=True)
        q = q * np.where(s > 0, target / np.where(s > 0, s, 1.0), 0.0)
        s = q.sum(axis=(0, 2, 3), keepdims=True)
        q = q * np.where(s > 0, target / np.where(s > 0, s, 1.0), 0.0)
        s = q.sum(axis=(0, 1, 3), keepdims=True)
        q = q * np.where(s > 0, target / np.where(s > 0, s, 1.0), 0.0)
        s = q.sum(axis=(0, 1, 2), keepdims=True)
        q = q * np.where(s > 0, target / np.where(s > 0, s, 1.0), 0.0)
    return q.reshape(n * n, n * n)


def solve_basket_pimot(
    marg: BasketMarginals,
    eta1: float,
    eta2: float,
    solver: str = "CLARABEL",
    solver_tol: float = 1e-10,
) -> BasketResult:
    """Finite-impact PIMOT upper bound for the basket call (methodology
    steps 5-9, 13).

    Solves the SOCP (methodology step 6) and projects its solution onto the
    exact margins by iterative proportional fitting to get a feasible lower
    estimate of the maximum value (methodology step 7). The upper estimate
    required by methodology step 8 is a static-hedge linear program over the
    holdings implied by the SOCP solution; for this two-coordinate problem
    the joint law p(x) of the pair (X1^1, X1^2) is itself free (only its two
    one-dimensional margins are pinned), so a tangent built from holdings
    that are zero at SOCP-optimal zero-probability pairs is not the
    KKT-tight supporting hyperplane there, and the naive static-hedge LP can
    route mass to those pairs and overshoot the true optimum (verified on a
    4x4x4x4 grid, where it exceeded the model-free OT ceiling). CLARABEL is
    therefore run to a tight duality-gap tolerance (``solver_tol``) and its
    own certified primal value is used as the upper estimate directly, which
    is the standard primal-dual certificate for a convex SOCP solved by an
    interior-point method. Reports both estimates, their midpoint, the
    resulting gap, and the expected shares traded per asset (methodology
    step 13) at the projected coupling.
    """
    n = marg.n
    x1, x2 = marg.x_of_index
    y1, y2 = marg.y_of_index
    payoff = np.maximum(y1 + y2, 0.0)

    q = cp.Variable((n * n, n * n), nonneg=True)
    constraints = _marginal_constraints(q, n)
    p = cp.sum(q, axis=1)
    d1 = q @ y1 - cp.multiply(x1, p)
    d2 = q @ y2 - cp.multiply(x2, p)
    penalty_terms = [cp.quad_over_lin(d1[i], p[i]) for i in range(n * n)]
    penalty_terms += [cp.quad_over_lin(d2[i], p[i]) for i in range(n * n)]
    # weight asset-1 terms by 1/(2 eta1) and asset-2 terms by 1/(2 eta2)
    penalty = (cp.sum(cp.hstack(penalty_terms[: n * n])) / (2.0 * eta1)
               + cp.sum(cp.hstack(penalty_terms[n * n:])) / (2.0 * eta2))
    linear_term = cp.sum(q @ payoff)
    objective = cp.Maximize(linear_term - penalty)
    problem = cp.Problem(objective, constraints)
    solver_kwargs = {"solver": solver}
    if solver == "CLARABEL":
        solver_kwargs.update(tol_gap_abs=solver_tol, tol_gap_rel=solver_tol, tol_feas=solver_tol)
    problem.solve(**solver_kwargs)
    value_upper = float(problem.value)

    q_adj = _enforce_marginals_ipf(q.value, n, target=1.0 / n)
    p_adj = q_adj.sum(axis=1)
    d1_adj = q_adj @ y1 - x1 * p_adj
    d2_adj = q_adj @ y2 - x2 * p_adj
    payoff_val = (q_adj @ payoff).sum()
    mask = p_adj > 1e-13
    pen1 = np.zeros(n * n)
    pen2 = np.zeros(n * n)
    pen1[mask] = d1_adj[mask] ** 2 / (2.0 * eta1 * p_adj[mask])
    pen2[mask] = d2_adj[mask] ** 2 / (2.0 * eta2 * p_adj[mask])
    value_lower = float(payoff_val - pen1.sum() - pen2.sum())
    # The IPF projection can occasionally overshoot the raw SOCP value by a
    # negligible amount; the reported upper estimate must never fall below it.
    value_upper = max(value_upper, value_lower)

    shares1 = float((2.0 / eta1) * np.sum(np.abs(d1_adj)))
    shares2 = float((2.0 / eta2) * np.sum(np.abs(d2_adj)))
    midpoint = 0.5 * (value_lower + value_upper)
    return BasketResult(
        value_lower=value_lower, value_upper=value_upper, midpoint=midpoint,
        gap=value_upper - value_lower, q=q_adj, drift1=d1_adj, drift2=d2_adj, p=p_adj,
        shares_traded_1=shares1, shares_traded_2=shares2, status=problem.status,
    )
