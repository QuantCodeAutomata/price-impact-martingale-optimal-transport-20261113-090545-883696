"""Unit tests for src/marginals.py (Appendix A.1 discretization)."""

import numpy as np
import pytest

from src.marginals import (
    discretize_lognormal,
    discrete_call_price,
    check_convex_order,
    reference_independent_gaussian_price,
)


def test_discretize_lognormal_preserves_mean_and_mass():
    p, x = discretize_lognormal(v=0.02, n_support=64)
    assert p.shape == x.shape == (64,)
    assert np.all(p > 0)
    assert abs(p.sum() - 1.0) < 1e-9
    assert abs(np.dot(p, x) - 1.0) < 1e-6
    assert np.all(x > 0)  # lognormal support is positive


def test_discretize_lognormal_rejects_nonpositive_variance():
    with pytest.raises(ValueError):
        discretize_lognormal(v=0.0)


def test_discrete_call_price_decreasing_in_strike():
    p, x = discretize_lognormal(v=0.03, n_support=32)
    strikes = np.array([0.5, 1.0, 1.5])
    prices = discrete_call_price(p, x, strikes)
    assert np.all(np.diff(prices) <= 0)
    assert prices[0] >= 1.0 - strikes[0]  # intrinsic-value lower bound at E[X]=1


def test_convex_order_holds_for_paper_parameters():
    v1 = 0.5 * 0.195 ** 2
    v2 = 1.0 * 0.175 ** 2
    p1, x1 = discretize_lognormal(v1, 256)
    p2, x2 = discretize_lognormal(v2, 256)
    assert check_convex_order(p1, x1, p2, x2)


def test_convex_order_fails_when_marginals_swapped():
    v1 = 0.5 * 0.195 ** 2
    v2 = 1.0 * 0.175 ** 2
    p1, x1 = discretize_lognormal(v1, 128)
    p2, x2 = discretize_lognormal(v2, 128)
    # mu2 <=_c mu1 should not hold since v2 > v1 (mu1 has smaller variance)
    assert not check_convex_order(p2, x2, p1, x1)


def test_reference_independent_gaussian_price_matches_paper_value():
    v1 = 0.5 * 0.195 ** 2
    v2 = 1.0 * 0.175 ** 2
    price_bp = reference_independent_gaussian_price(v1, v2) * 1e4
    assert abs(price_bp - 429.70) < 0.5


def test_reference_price_rejects_decreasing_variance():
    with pytest.raises(ValueError):
        reference_independent_gaussian_price(0.05, 0.01)
