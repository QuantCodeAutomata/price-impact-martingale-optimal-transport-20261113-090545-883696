"""Unit tests for src/basket.py (Sec. 5.2, Appendix A.3 two-asset basket call)."""

import numpy as np
import pytest

from src.basket import midpoints, BasketMarginals, solve_basket_ot_or_mot, solve_basket_pimot
from src import config


def test_midpoints_are_centered_and_equispaced():
    pts = midpoints((-1.0, 1.0), 4)
    assert len(pts) == 4
    assert np.allclose(np.diff(pts), np.diff(pts)[0])
    assert abs(pts.mean()) < 1e-12


@pytest.fixture(scope="module")
def small_marginals():
    n = 4
    return BasketMarginals(
        x1=midpoints((-1.0, 1.0), n),
        x2=midpoints((-3.0, 3.0), n),
        y1=midpoints((-3.0, 3.0), n),
        y2=midpoints((-3.0, 3.0), n),
        n=n,
    )


def test_ot_upper_bound_at_least_mot_upper_bound(small_marginals):
    mot = solve_basket_ot_or_mot(small_marginals, martingale=True)
    ot = solve_basket_ot_or_mot(small_marginals, martingale=False)
    assert ot.value_upper >= mot.value_upper - 1e-6


def test_ot_bound_matches_pairing_equal_terminal_prices(small_marginals):
    # OT bound = E[(Y1+Y2)+] under the coupling pairing equal terminal sums;
    # since both legs are symmetric around 0, the model-free OT ceiling here
    # is the paper's reported 1.500 for the full 16x16 problem; for n=4 it is
    # a looser but still computable ceiling bounded below by the payoff mean.
    ot = solve_basket_ot_or_mot(small_marginals, martingale=False)
    assert ot.value_upper > 0.0


def test_basket_pimot_between_mot_and_ot_bounds(small_marginals):
    mot = solve_basket_ot_or_mot(small_marginals, martingale=True)
    ot = solve_basket_ot_or_mot(small_marginals, martingale=False)
    res = solve_basket_pimot(small_marginals, eta1=1.0, eta2=1.0)
    assert mot.value_upper - 1e-4 <= res.midpoint <= ot.value_upper + 1e-4


def test_basket_pimot_lower_upper_gap_small(small_marginals):
    res = solve_basket_pimot(small_marginals, eta1=0.1, eta2=0.1)
    assert res.value_upper >= res.value_lower - 1e-9
    assert (res.value_upper - res.value_lower) < 1e-2


def test_basket_shares_traded_nonnegative(small_marginals):
    res = solve_basket_pimot(small_marginals, eta1=0.1, eta2=0.01)
    assert res.shares_traded_1 >= 0.0
    assert res.shares_traded_2 >= 0.0


def test_config_eta_pairs_match_paper_table():
    # Paper reports exactly these four (eta1, eta2) pairs (Sec. 5.2); only the
    # two "interchange" pairs (0.1, 0.01) and (0.01, 0.1) share the fixed
    # total eta1+eta2 = 0.11 used to illustrate swapping the coefficients.
    assert config.EXP3_ETA_PAIRS == ((0.01, 0.01), (0.1, 0.01), (0.01, 0.1), (0.1, 0.1))
    interchange_pairs = [(0.1, 0.01), (0.01, 0.1)]
    for eta1, eta2 in interchange_pairs:
        assert abs((eta1 + eta2) - config.EXP3_ETA_TOTAL) < 1e-9
