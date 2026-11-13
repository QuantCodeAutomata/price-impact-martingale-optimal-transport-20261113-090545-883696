"""Unit tests for src/spx_calibration.py on synthetic option quotes.

Checks the two steps of Appendix A.2 independently of any live market call:
the put-call parity fit recovers a known forward/discount pair, and the
marginal-fit LP returns masses with unit mean and discrete convex order.
"""

import numpy as np
import pandas as pd
import pytest

from src.spx_calibration import (
    build_calibration_grid,
    fit_forward_discount,
    fit_marginal_masses,
    select_calibration_quotes,
)


def _synthetic_chain(forward: float, discount: float, strikes: np.ndarray,
                      sigma: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Black-76 calls and puts at ``strikes`` for a known (F, D), exactly
    consistent with put-call parity, so the parity fit has a known answer."""
    from scipy.stats import norm

    tau = 1.0  # a nominal time-to-maturity, cancels out of the parity relation
    d1 = (np.log(forward / strikes) + 0.5 * sigma ** 2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    call = discount * (forward * norm.cdf(d1) - strikes * norm.cdf(d2))
    put = call - discount * (forward - strikes)  # put-call parity, exact by construction
    calls = pd.DataFrame({"strike": strikes, "price": call, "open_interest": 500})
    puts = pd.DataFrame({"strike": strikes, "price": put, "open_interest": 500})
    return calls, puts


def test_fit_forward_discount_recovers_known_forward_and_discount():
    forward, discount = 100.0, 0.99
    strikes = np.linspace(85.0, 115.0, 31)
    calls, puts = _synthetic_chain(forward, discount, strikes, sigma=0.2)

    fit = fit_forward_discount(calls, puts, spot=forward, moneyness_low=0.85, moneyness_high=1.15)

    assert fit.forward == pytest.approx(forward, rel=1e-6)
    assert fit.discount == pytest.approx(discount, rel=1e-6)
    assert fit.rmse < 1e-6  # quotes are exactly parity-consistent by construction
    # Floating-point rounding of the moneyness bound may drop the outermost strike.
    assert fit.n_points >= len(strikes) - 1


def test_fit_forward_discount_requires_matched_strikes():
    calls = pd.DataFrame({"strike": [100.0], "price": [5.0], "open_interest": [10]})
    puts = pd.DataFrame({"strike": [90.0], "price": [2.0], "open_interest": [10]})
    with pytest.raises(ValueError):
        fit_forward_discount(calls, puts, spot=100.0, moneyness_low=0.85, moneyness_high=1.15)


def test_build_calibration_grid_matches_paper_node_count():
    regular = np.round(np.arange(0.55, 1.45 + 1e-9, 0.005), 3)
    extra = np.array([0.0, 0.2, 0.35, 0.45, 1.6, 1.8, 2.0])
    grid = build_calibration_grid(regular, extra)
    assert len(grid) == 181 + 7  # Appendix A.2: 188 support nodes
    assert np.all(np.diff(grid) > 0)


def test_fit_marginal_masses_satisfies_mean_and_convex_order():
    """A small synthetic chain calibrated on a coarse grid: the fitted masses
    must sum to one, have unit mean, and satisfy discrete convex order
    mu1 <=_c mu2 (methodology step 4), the property the paper requires rather
    than any specific fitted level."""
    grid = build_calibration_grid(
        np.round(np.arange(0.70, 1.30 + 1e-9, 0.05), 3), np.array([0.0, 2.0])
    )
    forward, discount = 100.0, 0.99
    strikes = forward * grid[(grid >= 0.75) & (grid <= 1.25)]
    v1, v2 = 0.02, 0.03  # maturity-1 narrower than maturity-2, consistent with mu1 <=_c mu2
    calls1, puts1 = _synthetic_chain(forward, discount, strikes, sigma=np.sqrt(v1))
    calls2, puts2 = _synthetic_chain(forward, discount, strikes, sigma=np.sqrt(v2))

    quotes1 = select_calibration_quotes(calls1, puts1, forward, discount, 0.70, 1.30, min_open_interest=1)
    quotes2 = select_calibration_quotes(calls2, puts2, forward, discount, 0.70, 1.30, min_open_interest=1)

    p1, p2, diagnostics = fit_marginal_masses(grid, quotes1, quotes2, error_floor=0.01,
                                               second_stage_tol=1e-6)

    assert p1.sum() == pytest.approx(1.0, abs=1e-8)
    assert p2.sum() == pytest.approx(1.0, abs=1e-8)
    assert np.all(p1 >= -1e-9) and np.all(p2 >= -1e-9)
    assert grid @ p1 == pytest.approx(1.0, abs=1e-4)
    assert grid @ p2 == pytest.approx(1.0, abs=1e-4)

    # Discrete convex order: call-price difference nonnegative at every node.
    payoff_at_nodes = np.maximum(grid[None, :] - grid[:, None], 0.0)
    call_diff = payoff_at_nodes @ (p2 - p1)
    assert np.all(call_diff >= -1e-6)

    assert diagnostics["stage1_status"] in ("optimal", "optimal_inaccurate")
    assert diagnostics["stage2_status"] in ("optimal", "optimal_inaccurate")
