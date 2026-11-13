"""Experiment 2: forward-start call pricing with SPX-calibrated marginals
across order sizes and strike ratios (Sec. 5.1.2 and Appendix A.2 of
"Price-impact Martingale optimal transport").

Run: python -m exp.exp2
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from data.data_loader import load_spx_chains
from src.spx_calibration import (
    fit_forward_discount,
    build_calibration_grid,
    select_calibration_quotes,
    fit_marginal_masses,
)
from src.pimot import solve_pimot, payoff_forward_start_call, project_unit_mean

RESULTS_DIR = "results"


def _atm_spot_estimate(calls: pd.DataFrame, puts: pd.DataFrame) -> float:
    """Data-driven spot estimate: the strike where quoted call and put prices
    are closest (put-call parity holds approximately at the money), used only
    to center the parity-fit strike window before the forward/discount fit."""
    merged = pd.merge(calls[["strike", "price"]], puts[["strike", "price"]],
                       on="strike", suffixes=("_call", "_put"))
    merged["abs_diff"] = (merged["price_call"] - merged["price_put"]).abs()
    return float(merged.loc[merged["abs_diff"].idxmin(), "strike"])


def _rmse_index_points(quotes: pd.DataFrame, grid: np.ndarray, p: np.ndarray,
                        forward: float, discount: float) -> float:
    """RMSE of the fitted call prices against quoted mid prices, in raw index
    points (undoing the D*F normalization of Appendix A.2)."""
    errors = []
    for moneyness, market_norm_price in zip(quotes["moneyness"], quotes["normalized_call_price"]):
        payoff = np.maximum(grid - moneyness, 0.0)
        fitted_norm = float(payoff @ p)
        errors.append((fitted_norm - market_norm_price) * discount * forward)
    return float(np.sqrt(np.mean(np.square(errors))))


def calibrate(use_cache: bool = True) -> dict:
    chains = load_spx_chains(
        (config.EXP2_EXPIRY_1, config.EXP2_EXPIRY_2),
        config.EXP2_STRIKE_FETCH_LOW, config.EXP2_STRIKE_FETCH_HIGH,
        use_cache=use_cache,
    )
    grid = build_calibration_grid(config.EXP2_GRID_REGULAR, config.EXP2_GRID_EXTRA)

    parity, quotes = {}, {}
    for expiry in (config.EXP2_EXPIRY_1, config.EXP2_EXPIRY_2):
        calls, puts = chains[expiry]["calls"], chains[expiry]["puts"]
        calls = calls[calls["open_interest"] >= config.EXP2_MIN_OPEN_INTEREST].reset_index(drop=True)
        puts = puts[puts["open_interest"] >= config.EXP2_MIN_OPEN_INTEREST].reset_index(drop=True)
        spot_guess = _atm_spot_estimate(calls, puts)
        fit = fit_forward_discount(calls, puts, spot_guess,
                                    config.EXP2_PARITY_MONEYNESS_LOW,
                                    config.EXP2_PARITY_MONEYNESS_HIGH)
        parity[expiry] = fit
        quotes[expiry] = select_calibration_quotes(
            calls, puts, fit.forward, fit.discount,
            config.EXP2_CALIBRATION_MONEYNESS_LOW, config.EXP2_CALIBRATION_MONEYNESS_HIGH,
            config.EXP2_MIN_OPEN_INTEREST,
        )

    p1, p2, diagnostics = fit_marginal_masses(
        grid, quotes[config.EXP2_EXPIRY_1], quotes[config.EXP2_EXPIRY_2],
        config.EXP2_MIN_SCALED_ERROR_FLOOR, config.EXP2_SECOND_STAGE_TOLERANCE,
    )

    rmse1 = _rmse_index_points(quotes[config.EXP2_EXPIRY_1], grid, p1,
                                parity[config.EXP2_EXPIRY_1].forward, parity[config.EXP2_EXPIRY_1].discount)
    rmse2 = _rmse_index_points(quotes[config.EXP2_EXPIRY_2], grid, p2,
                                parity[config.EXP2_EXPIRY_2].forward, parity[config.EXP2_EXPIRY_2].discount)

    mask1 = p1 > config.EXP2_SUPPORT_MASS_TOL
    mask2 = p2 > config.EXP2_SUPPORT_MASS_TOL
    x1, x2 = grid[mask1], grid[mask2]
    p1_support = project_unit_mean(p1[mask1] / p1[mask1].sum(), x1)
    p2_support = project_unit_mean(p2[mask2] / p2[mask2].sum(), x2)

    return {
        "grid": grid, "x1": x1, "p1": p1_support, "x2": x2, "p2": p2_support,
        "parity": parity, "diagnostics": diagnostics,
        "rmse_1_index_points": rmse1, "rmse_2_index_points": rmse2,
        "n_support_1": int(mask1.sum()), "n_support_2": int(mask2.sum()),
    }


def price_bounds(x1: np.ndarray, p1: np.ndarray, x2: np.ndarray, p2: np.ndarray) -> dict:
    """Seller's upper and buyer's lower bound per claim, at every order size
    and strike ratio, plus the order-independent MOT/OT reference bounds
    (methodology steps 6-10)."""
    rows = []
    mot_ot = {}
    for k in config.EXP2_STRIKE_RATIOS:
        payoff = payoff_forward_start_call(x1, x2, k)
        floor = max(1.0 - k, 0.0)
        mot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=True,
                           solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
        ot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=False,
                          solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
        mot_ot[k] = {
            "mot_bound_bp": (mot.value - floor) * 1e4,
            "ot_bound_bp": (ot.value - floor) * 1e4,
        }
        for n in config.EXP2_ORDER_SIZES:
            eta = n * config.EXP2_ETA0
            seller = solve_pimot(p1, x1, p2, x2, payoff, eta=eta, martingale=False,
                                  solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
            buyer = solve_pimot(p1, x1, p2, x2, -payoff, eta=eta, martingale=False,
                                 solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
            u_bp = (seller.value - floor) * 1e4
            l_bp = (-buyer.value - floor) * 1e4
            rows.append({
                "k": k, "n": n, "eta": eta,
                "seller_upper_bound_bp": u_bp,
                "buyer_lower_bound_bp": l_bp,
                "mot_bound_bp": mot_ot[k]["mot_bound_bp"],
                "ot_bound_bp": mot_ot[k]["ot_bound_bp"],
                "shares_traded_per_claim": seller.shares_traded,
                "shares_traded_whole_order": n * seller.shares_traded,
                "seller_status": seller.status,
                "buyer_status": buyer.status,
            })
    return {"rows": rows, "mot_ot": mot_ot}


def run(use_cache: bool = True) -> dict:
    calib = calibrate(use_cache=use_cache)
    bounds = price_bounds(calib["x1"], calib["p1"], calib["x2"], calib["p2"])
    table = pd.DataFrame(bounds["rows"])
    table.to_csv(f"{RESULTS_DIR}/exp2_bounds_table.csv", index=False)

    summary = {
        "forward_1": calib["parity"][config.EXP2_EXPIRY_1].forward,
        "discount_1": calib["parity"][config.EXP2_EXPIRY_1].discount,
        "forward_2": calib["parity"][config.EXP2_EXPIRY_2].forward,
        "discount_2": calib["parity"][config.EXP2_EXPIRY_2].discount,
        "parity_rmse_1_index_points": calib["parity"][config.EXP2_EXPIRY_1].rmse,
        "parity_rmse_2_index_points": calib["parity"][config.EXP2_EXPIRY_2].rmse,
        "calibration_rmse_1_index_points": calib["rmse_1_index_points"],
        "calibration_rmse_2_index_points": calib["rmse_2_index_points"],
        "n_support_1": calib["n_support_1"],
        "n_support_2": calib["n_support_2"],
        "diagnostics": calib["diagnostics"],
        "rows": bounds["rows"],
    }
    with open(f"{RESULTS_DIR}/exp2_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_interval_widening(table)
    return summary


def _plot_interval_widening(table: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    for n in sorted(table["n"].unique()):
        sub = table[table["n"] == n].sort_values("k")
        ax.plot(sub["k"], sub["seller_upper_bound_bp"], "o-", label=f"seller U, n={n:g}")
        ax.plot(sub["k"], sub["buyer_lower_bound_bp"], "s--", label=f"buyer L, n={n:g}")
    ax.set_xlabel("strike ratio k")
    ax.set_ylabel("normalized price bound (bp)")
    ax.set_title("Price bounds vs. strike ratio and order size")
    ax.legend(fontsize=6)

    ax = axes[1]
    k1 = table[table["k"] == 1.0].sort_values("n")
    ax.semilogx(k1["n"], k1["seller_upper_bound_bp"], "o-", label="seller upper bound U_n(1.00)")
    ax.semilogx(k1["n"], k1["buyer_lower_bound_bp"], "s-", label="buyer lower bound L_n(1.00)")
    ax.axhline(k1["mot_bound_bp"].iloc[0], color="tab:green", linestyle="--", label="MOT bound")
    ax.axhline(k1["ot_bound_bp"].iloc[0], color="tab:red", linestyle="--", label="OT bound")
    ax.set_xlabel("order size n (units of the claim)")
    ax.set_ylabel("normalized price bound (bp)")
    ax.set_title("Interval widening with order size, k = 1.00")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp2_interval_widening.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for row in summary["rows"]:
        print(row)
