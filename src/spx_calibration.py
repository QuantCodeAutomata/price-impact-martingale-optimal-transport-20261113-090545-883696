"""Calibration of the two SPX-implied marginal laws used in the forward-start
call experiment (Sec. 5.1.2 and Appendix A.2 of "Price-impact Martingale
optimal transport").

Two steps, in the order given by Appendix A.2:

1. Put-call parity fit: for each maturity, regress (call - put) on strike over
   the parity strike window to recover the forward price F and discount
   factor D (``fit_forward_discount``).
2. Marginal fit: assign masses to a common support grid for both maturities by
   minimizing the average scaled pricing error, then, among near-minimizers,
   the one with the smallest sum of second moments, subject to each law having
   unit mean and the pair satisfying discrete convex order
   (``fit_marginal_masses``).

Custom implementation of the paper's calibration LPs (no portfolio/calibration
library implements this two-stage, convex-order-constrained option fit); cvxpy
is used as the paper's implementation notes specify.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class ParityFit:
    forward: float
    discount: float
    rmse: float
    n_points: int


def fit_forward_discount(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    spot: float,
    moneyness_low: float,
    moneyness_high: float,
) -> ParityFit:
    """Forward price F and discount factor D from put-call parity.

    Matches calls and puts at common strikes in
    [``moneyness_low`` * spot, ``moneyness_high`` * spot] and regresses
    (call - put) on strike K by ordinary least squares:
    call - put = D*(F - K) = -D*K + D*F. The fitted slope is -D and the
    intercept is D*F, so the line crosses zero at F = intercept / (-slope)
    (Appendix A.2, put-call parity fit).
    """
    strike_lo, strike_hi = moneyness_low * spot, moneyness_high * spot
    c = calls[(calls["strike"] >= strike_lo) & (calls["strike"] <= strike_hi)]
    p = puts[(puts["strike"] >= strike_lo) & (puts["strike"] <= strike_hi)]
    merged = pd.merge(c[["strike", "price"]], p[["strike", "price"]], on="strike",
                       suffixes=("_call", "_put"))
    if len(merged) < 2:
        raise ValueError("Not enough matched call/put strikes for the parity fit")
    diff = (merged["price_call"] - merged["price_put"]).to_numpy()
    strikes = merged["strike"].to_numpy()
    slope, intercept = np.polyfit(strikes, diff, 1)
    discount = -slope
    forward = intercept / discount
    fitted = intercept + slope * strikes
    rmse = float(np.sqrt(np.mean((fitted - diff) ** 2)))
    return ParityFit(forward=float(forward), discount=float(discount), rmse=rmse,
                      n_points=len(merged))


def build_calibration_grid(regular: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """The common 188-node support grid for both marginals (methodology step 4):
    181 regular nodes from 0.55 to 1.45 in steps of 0.005, plus 7 extra nodes."""
    grid = np.sort(np.unique(np.concatenate([regular, extra])))
    assert len(grid) == len(regular) + len(extra), "grid nodes must be distinct"
    return grid


def _normalized_quotes(
    option_chain: pd.DataFrame,
    forward: float,
    discount: float,
    side: str,
    moneyness_low: float,
    moneyness_high: float,
) -> pd.DataFrame:
    """Select puts with K < F and calls with K >= F, K/F in
    [``moneyness_low``, ``moneyness_high``], and normalize strikes by F and
    prices by D*F (methodology steps 3-4: Xi = S_Ti/Fi has unit mean)."""
    chain = option_chain.copy()
    chain["moneyness"] = chain["strike"] / forward
    if side == "put":
        chain = chain[chain["strike"] < forward]
    else:
        chain = chain[chain["strike"] >= forward]
    chain = chain[(chain["moneyness"] >= moneyness_low) & (chain["moneyness"] <= moneyness_high)]
    chain = chain[chain["price"] > 0]
    chain["normalized_price"] = chain["price"] / (discount * forward)
    return chain


def _call_price_from_put(row_price: float, moneyness: float) -> float:
    """Normalized call price implied by a normalized put price via parity:
    call - put = 1 - k (in units where F=1, D cancels out)."""
    return row_price + max(1.0 - moneyness, 0.0)


def select_calibration_quotes(
    calls: pd.DataFrame,
    puts: pd.DataFrame,
    forward: float,
    discount: float,
    moneyness_low: float,
    moneyness_high: float,
    min_open_interest: int,
) -> pd.DataFrame:
    """Normalized call-equivalent quotes used to fit one marginal: puts with
    K < F and calls with K >= F, restricted to K/F in
    [``moneyness_low``, ``moneyness_high``] and open interest >=
    ``min_open_interest`` (data_requirements)."""
    put_leg = _normalized_quotes(puts, forward, discount, "put", moneyness_low, moneyness_high)
    put_leg = put_leg[put_leg["open_interest"] >= min_open_interest]
    put_leg["normalized_call_price"] = [
        _call_price_from_put(pr, m) for pr, m in zip(put_leg["normalized_price"], put_leg["moneyness"])
    ]
    call_leg = _normalized_quotes(calls, forward, discount, "call", moneyness_low, moneyness_high)
    call_leg = call_leg[call_leg["open_interest"] >= min_open_interest]
    call_leg["normalized_call_price"] = call_leg["normalized_price"]
    combined = pd.concat([
        put_leg[["moneyness", "normalized_call_price"]],
        call_leg[["moneyness", "normalized_call_price"]],
    ], ignore_index=True)
    return combined.drop_duplicates(subset="moneyness").sort_values("moneyness").reset_index(drop=True)


def fit_marginal_masses(
    grid: np.ndarray,
    quotes1: pd.DataFrame,
    quotes2: pd.DataFrame,
    error_floor: float,
    second_stage_tol: float,
    solver: str = "CLARABEL",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Two-stage calibration LP (Appendix A.2, methodology steps 4-5).

    Stage 1 minimizes the average scaled pricing error of both maturities
    jointly, subject to: masses nonnegative, unit mass, unit mean, and
    discrete convex order mu1 <=_c mu2 (the call-price-difference nonnegativity
    condition at every grid node). Stage 2 minimizes the sum of second moments
    among fits within ``second_stage_tol`` of the stage-1 minimum.

    Returns (p1, p2, diagnostics) with diagnostics holding the achieved RMSE
    (in index points, undoing the D*F normalization is the caller's job) and
    solver statuses.
    """
    n = len(grid)
    p1 = cp.Variable(n, nonneg=True)
    p2 = cp.Variable(n, nonneg=True)

    base_constraints = [
        cp.sum(p1) == 1, cp.sum(p2) == 1,
        p1 @ grid == 1.0, p2 @ grid == 1.0,
    ]
    # Discrete convex order: at every node x_r, the call-price difference
    # sum_j (x_j - x_r)^+ (p2_j - p1_j) must be nonnegative (methodology step 4).
    payoff_at_nodes = np.maximum(grid[None, :] - grid[:, None], 0.0)  # (r, j)
    convex_order_constraint = payoff_at_nodes @ (p2 - p1) >= 0

    def _pricing_errors(quotes: pd.DataFrame, p: cp.Variable) -> list[cp.Expression]:
        errors = []
        for moneyness, market_price in zip(quotes["moneyness"], quotes["normalized_call_price"]):
            payoff = np.maximum(grid - moneyness, 0.0)
            fitted = payoff @ p
            denom = error_floor
            errors.append(cp.abs(fitted - market_price) / denom)
        return errors

    errors1 = _pricing_errors(quotes1, p1)
    errors2 = _pricing_errors(quotes2, p2)
    all_errors = errors1 + errors2
    avg_error = cp.sum(all_errors) / len(all_errors)

    stage1 = cp.Problem(cp.Minimize(avg_error), base_constraints + [convex_order_constraint])
    stage1.solve(solver=solver)
    if stage1.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage-1 calibration LP failed: {stage1.status}")
    min_error = float(stage1.value)

    second_moment = p1 @ (grid ** 2) + p2 @ (grid ** 2)
    stage2_constraints = base_constraints + [
        convex_order_constraint,
        avg_error <= min_error + second_stage_tol,
    ]
    stage2 = cp.Problem(cp.Minimize(second_moment), stage2_constraints)
    stage2.solve(solver=solver)
    if stage2.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Stage-2 calibration LP failed: {stage2.status}")

    p1_val = np.clip(p1.value, 0.0, None)
    p2_val = np.clip(p2.value, 0.0, None)
    p1_val /= p1_val.sum()
    p2_val /= p2_val.sum()

    diagnostics = {
        "stage1_avg_scaled_error": min_error,
        "stage2_avg_scaled_error": float(avg_error.value),
        "stage2_second_moment": float(stage2.value),
        "stage1_status": stage1.status,
        "stage2_status": stage2.status,
        "n_quotes_1": len(quotes1),
        "n_quotes_2": len(quotes2),
        "n_support_1": int(np.sum(p1_val > 1e-10)),
        "n_support_2": int(np.sum(p2_val > 1e-10)),
    }
    return p1_val, p2_val, diagnostics
