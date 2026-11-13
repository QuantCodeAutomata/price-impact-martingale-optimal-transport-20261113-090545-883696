"""Discretized price-impact martingale optimal transport (PIMOT) for a single
pair of marginals (Sec. 3 and Appendix A.1 of the paper).

The discretized primal problem (Appendix A.1, Eq. for the forward-start call)
is the concave quadratic program

    max_{gamma >= 0}  sum_ij gamma_ij * payoff(x1_i, x2_j)
                       - (1/(2*eta)) * sum_i p1_i * ((gamma @ x2)_i / p1_i - x1_i)^2
    s.t.  sum_j gamma_ij = p1_i  for all i   (row/marginal-1 constraint)
          sum_i gamma_ij = p2_j  for all j   (column/marginal-2 constraint)

The MOT upper bound adds the martingale constraint (gamma @ x2)_i = p1_i * x1_i
for all i (zero drift). The OT upper bound drops the quadratic penalty and the
martingale constraint, leaving a pure transport linear program.

For finite eta, the conditional drift b_i = (gamma @ x2)_i / p1_i - x1_i gives
the optimal signed holding h_i = b_i / eta (paper's h_eta(x) = b_{gamma_eta}(x)/eta)
and the expected number of shares traded per claim
    V_eta = (2/eta) * sum_i |b_i| * p1_i.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass
class PimotResult:
    value: float            # primal objective value (the price bound, raw units)
    dual_value: float        # dual objective value from the marginal constraints (upper certificate)
    gamma: np.ndarray         # optimal coupling, shape (n1, n2)
    drift: np.ndarray         # b_i = (gamma @ x2)_i / p1_i - x1_i, length n1
    holding: np.ndarray       # h_i = b_i / eta (zeros if eta is None), length n1
    shares_traded: float      # V_eta = (2/eta) * sum_i |b_i| * p1_i (nan if eta is None)
    status: str


def payoff_forward_start_call(x1: np.ndarray, x2: np.ndarray, k: float) -> np.ndarray:
    """Payoff matrix Phi_k(x1_i, x2_j) = (x2_j - k*x1_i)^+, shape (n1, n2)."""
    return np.maximum(x2[None, :] - k * x1[:, None], 0.0)


def project_unit_mean(p: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Re-project masses onto {sum(p)=1, dot(p,x)=1} after dropping near-zero
    support points (not a paper step: a calibrated law already satisfies both
    constraints exactly as hard equalities in the calibration LP, Appendix
    A.2; discarding near-zero nodes and renormalizing the remaining mass can
    leave a floating-point residual of order 1e-6 in the mean, which makes the
    hard martingale equality of the MOT bound infeasible in exact arithmetic).

    Minimal L2 correction solving min ||p' - p||^2 s.t. A p' = b with
    A = [[1,...,1], x], b = [1, 1] (closed-form 2x2 Lagrange-multiplier
    projection); the correction is of the same tiny order as the residual it
    removes, so masses stay nonnegative without re-masking.
    """
    a_rows = np.vstack([np.ones_like(x), x])
    gram = a_rows @ a_rows.T
    residual = a_rows @ p - np.array([1.0, 1.0])
    lagrange = np.linalg.solve(gram, residual)
    p_adj = p - a_rows.T @ lagrange
    assert np.all(p_adj > -1e-9), "projection produced a negative mass"
    return np.clip(p_adj, 0.0, None)


def solve_pimot(
    p1: np.ndarray,
    x1: np.ndarray,
    p2: np.ndarray,
    x2: np.ndarray,
    payoff_matrix: np.ndarray,
    eta: float | None,
    martingale: bool = False,
    solver: str = "CLARABEL",
    solver_tol: float = 1e-10,
) -> PimotResult:
    """Solve the discretized PIMOT / MOT / OT problem.

    Parameters
    ----------
    p1, x1 : masses and support points of the first marginal (T1 law)
    p2, x2 : masses and support points of the second marginal (T2 law)
    payoff_matrix : Phi(x1_i, x2_j), shape (len(x1), len(x2))
    eta : effective impact coefficient. ``None`` means no quadratic penalty
        (the OT relaxation, unless ``martingale=True`` gives the MOT bound).
    martingale : if True, add the zero-drift (martingale) constraint; this is
        the MOT upper bound when eta is None, or has no further effect when a
        finite eta is given together with martingale=True (not used in this
        paper, kept for completeness).
    """
    n1, n2 = payoff_matrix.shape
    assert p1.shape == (n1,) and x1.shape == (n1,)
    assert p2.shape == (n2,) and x2.shape == (n2,)

    gamma = cp.Variable((n1, n2), nonneg=True)
    row_sum = cp.sum(gamma, axis=1)
    col_sum = cp.sum(gamma, axis=0)
    row_constraint = row_sum == p1
    col_constraint = col_sum == p2
    constraints = [row_constraint, col_constraint]

    linear_term = cp.sum(cp.multiply(gamma, payoff_matrix))

    drift_constraint = None
    if eta is not None:
        d = gamma @ x2 - cp.multiply(p1, x1)  # d_i = p1_i * b_i
        penalty = cp.sum(cp.multiply(cp.square(d), 1.0 / p1)) / (2.0 * eta)
        objective = cp.Maximize(linear_term - penalty)
    else:
        objective = cp.Maximize(linear_term)

    if martingale:
        d_eq = gamma @ x2 - cp.multiply(p1, x1)
        drift_constraint = d_eq == 0
        constraints.append(drift_constraint)

    problem = cp.Problem(objective, constraints)
    solver_kwargs = {"solver": solver}
    if solver == "CLARABEL":
        solver_kwargs.update(tol_gap_abs=solver_tol, tol_gap_rel=solver_tol,
                              tol_feas=solver_tol)
    problem.solve(**solver_kwargs)

    gamma_val = gamma.value
    gamma_val = np.clip(gamma_val, 0.0, None)

    b = (gamma_val @ x2) / p1 - x1
    if eta is not None:
        holding = b / eta
        shares_traded = float((2.0 / eta) * np.sum(np.abs(b) * p1))
    else:
        holding = np.zeros_like(b)
        shares_traded = float("nan")

    if eta is None:
        # Pure transport LP (OT, or MOT when martingale=True): the row/column
        # dual multipliers are static positions at every support point, and
        # strong LP duality gives an exact upper certificate.
        dual_value = float(np.dot(row_constraint.dual_value, p1)
                            + np.dot(col_constraint.dual_value, p2))
    else:
        dual_value = static_hedge_upper_bound(p1, p2, x1, x2, payoff_matrix, eta, holding,
                                               solver=solver, solver_tol=solver_tol)

    return PimotResult(
        value=float(problem.value),
        dual_value=dual_value,
        gamma=gamma_val,
        drift=b,
        holding=holding,
        shares_traded=shares_traded,
        status=problem.status,
    )


def static_hedge_upper_bound(
    p1: np.ndarray,
    p2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    payoff_matrix: np.ndarray,
    eta: float,
    holding: np.ndarray,
    solver: str = "CLARABEL",
    solver_tol: float = 1e-10,
) -> float:
    """Upper estimate of the PIMOT value for fixed holdings (methodology step 8).

    By the Fenchel-Young inequality applied to f(t) = t^2/(2*eta),
        -t^2/(2*eta) <= -t*h + (eta/2)*h^2   for every t, with equality at t = h,
    so for ANY feasible coupling gamma' sharing the same two marginals,

        Obj(gamma') <= sum_ij gamma'_ij*(payoff_ij - h_i*(x2_j - x1_i))
                        + (eta/2)*sum_i h_i^2*p1_i.

    The right-hand side is linear in gamma' once h is fixed, i.e. a static
    super-hedge "at every pair of support points" (x1_i, x2_j); its supremum
    over feasible gamma' is a transport linear program. Equality holds when
    gamma' is the optimal coupling itself (t_i = h_i there), so this LP value
    is a true upper bound that coincides with the primal optimum up to solver
    tolerance.
    """
    n1, n2 = payoff_matrix.shape
    residual = payoff_matrix - holding[:, None] * (x2[None, :] - x1[:, None])
    gamma2 = cp.Variable((n1, n2), nonneg=True)
    constraints = [cp.sum(gamma2, axis=1) == p1, cp.sum(gamma2, axis=0) == p2]
    objective = cp.Maximize(cp.sum(cp.multiply(gamma2, residual)))
    problem = cp.Problem(objective, constraints)
    solver_kwargs = {"solver": solver}
    if solver == "CLARABEL":
        solver_kwargs.update(tol_gap_abs=solver_tol, tol_gap_rel=solver_tol,
                              tol_feas=solver_tol)
    problem.solve(**solver_kwargs)
    constant = 0.5 * eta * np.sum(holding ** 2 * p1)
    return float(problem.value) + constant
