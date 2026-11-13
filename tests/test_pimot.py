"""Unit tests for src/pimot.py: OT/MOT limits, monotonicity in eta, no look-ahead."""

import numpy as np
import pytest

from src.marginals import discretize_lognormal
from src.pimot import solve_pimot, payoff_forward_start_call, project_unit_mean


@pytest.fixture(scope="module")
def toy_marginals():
    v1 = 0.5 * 0.195 ** 2
    v2 = 1.0 * 0.175 ** 2
    p1, x1 = discretize_lognormal(v1, 32)
    p2, x2 = discretize_lognormal(v2, 32)
    return p1, x1, p2, x2


def test_ot_upper_bound_dominates_mot_upper_bound(toy_marginals):
    p1, x1, p2, x2 = toy_marginals
    payoff = payoff_forward_start_call(x1, x2, k=1.0)
    mot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=True)
    ot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=False)
    # Dropping the martingale constraint only enlarges the feasible set.
    assert ot.value >= mot.value - 1e-8


def test_pimot_bound_increases_monotonically_with_eta(toy_marginals):
    p1, x1, p2, x2 = toy_marginals
    payoff = payoff_forward_start_call(x1, x2, k=1.0)
    etas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
    values = [solve_pimot(p1, x1, p2, x2, payoff, eta=e).value for e in etas]
    assert np.all(np.diff(values) >= -1e-9)


def test_pimot_value_stays_between_mot_and_ot(toy_marginals):
    p1, x1, p2, x2 = toy_marginals
    payoff = payoff_forward_start_call(x1, x2, k=1.0)
    mot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=True)
    ot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=False)
    for eta in (1e-4, 1.0, 100.0):
        res = solve_pimot(p1, x1, p2, x2, payoff, eta=eta)
        assert mot.value - 1e-6 <= res.value <= ot.value + 1e-6


def test_shares_traded_decreases_as_eta_grows(toy_marginals):
    p1, x1, p2, x2 = toy_marginals
    payoff = payoff_forward_start_call(x1, x2, k=1.0)
    etas = [1e-3, 1e-1, 10.0]
    shares = [solve_pimot(p1, x1, p2, x2, payoff, eta=e).shares_traded for e in etas]
    assert shares[0] >= shares[1] >= shares[2]
    assert all(s >= 0 for s in shares)


def test_lower_and_upper_estimates_agree_within_tolerance(toy_marginals):
    p1, x1, p2, x2 = toy_marginals
    payoff = payoff_forward_start_call(x1, x2, k=1.0)
    res = solve_pimot(p1, x1, p2, x2, payoff, eta=1.0)
    gap_bp = abs(res.dual_value - res.value) * 1e4
    assert gap_bp < 1e-3  # looser than the paper's 2.2e-7 bp since n_support=32 here


def test_project_unit_mean_restores_hard_constraints():
    p = np.array([0.3, 0.3, 0.4]) + 1e-6
    x = np.array([0.8, 1.0, 1.2])
    p_adj = project_unit_mean(p, x)
    assert abs(p_adj.sum() - 1.0) < 1e-10
    assert abs(np.dot(p_adj, x) - 1.0) < 1e-10
    assert np.all(p_adj >= 0)
