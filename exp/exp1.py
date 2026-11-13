"""Experiment 1: single-asset forward-start call under quadratic price impact
with lognormal marginals (Sec. 5.1.1 and Appendix A.1 of "Price-impact
Martingale optimal transport").

Run: python -m exp.exp1
"""

from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.marginals import (
    discretize_lognormal,
    check_convex_order,
    reference_independent_gaussian_price,
)
from src.pimot import solve_pimot, payoff_forward_start_call

RESULTS_DIR = "results"


def run(n_support: int = config.EXP1_N_SUPPORT) -> dict:
    p1, x1 = discretize_lognormal(config.EXP1_V1, n_support)
    p2, x2 = discretize_lognormal(config.EXP1_V2, n_support)

    convex_order_holds = check_convex_order(p1, x1, p2, x2)
    assert convex_order_holds, "mu1 <=_c mu2 must hold for the forward-start marginals"

    payoff = payoff_forward_start_call(x1, x2, config.EXP1_K)
    ref_price_bp = reference_independent_gaussian_price(config.EXP1_V1, config.EXP1_V2) * 1e4

    mot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=True,
                       solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
    ot = solve_pimot(p1, x1, p2, x2, payoff, eta=None, martingale=False,
                      solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)

    rows = []
    holding_curves = {}
    max_gap_bp = 0.0
    t0 = time.time()
    for eta in config.EXP1_ETA_GRID:
        res = solve_pimot(p1, x1, p2, x2, payoff, eta=float(eta), martingale=False,
                           solver=config.QP_SOLVER, solver_tol=config.QP_SOLVER_TOL)
        gap_bp = abs(res.dual_value - res.value) * 1e4
        max_gap_bp = max(max_gap_bp, gap_bp)
        rows.append({
            "eta": float(eta),
            "upper_bound_bp": res.value * 1e4,
            "upper_estimate_bp": res.dual_value * 1e4,
            "lower_upper_gap_bp": gap_bp,
            "shares_traded_per_claim": res.shares_traded,
            "status": res.status,
        })
        if any(np.isclose(eta, e) for e in config.EXP1_ETA_PLOT):
            holding_curves[float(eta)] = res.holding.copy()
    runtime_s = time.time() - t0

    summary = {
        "n_support": n_support,
        "convex_order_holds": convex_order_holds,
        "reference_independent_gaussian_price_bp": ref_price_bp,
        "mot_upper_bound_bp": mot.value * 1e4,
        "ot_upper_bound_bp": ot.value * 1e4,
        "max_lower_upper_gap_bp": max_gap_bp,
        "paper_tolerance_bp": config.EXP1_UPPER_LOWER_TOL_BP,
        "eta_grid_runtime_seconds": runtime_s,
        "rows": rows,
    }

    with open(f"{RESULTS_DIR}/exp1_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_bound_and_volume(rows, mot.value * 1e4, ot.value * 1e4)
    _plot_holding_curves(x1, holding_curves)

    return summary


def _plot_bound_and_volume(rows: list[dict], mot_bp: float, ot_bp: float) -> None:
    etas = [r["eta"] for r in rows]
    bounds = [r["upper_bound_bp"] for r in rows]
    shares = [r["shares_traded_per_claim"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.semilogx(etas, bounds, "o-", label=r"PIMOT upper bound $P_\eta(\Phi_1)$")
    ax.axhline(mot_bp, color="tab:green", linestyle="--", label=f"MOT bound ({mot_bp:.2f} bp)")
    ax.axhline(ot_bp, color="tab:red", linestyle="--", label=f"OT bound ({ot_bp:.2f} bp)")
    ax.set_xlabel(r"effective impact coefficient $\eta$")
    ax.set_ylabel("price bound (basis points)")
    ax.set_title("Forward-start call: PIMOT upper bound vs. $\\eta$")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.loglog(etas, shares, "o-", color="tab:purple")
    ax.set_xlabel(r"effective impact coefficient $\eta$")
    ax.set_ylabel(r"expected shares traded per claim $V_\eta$")
    ax.set_title("Expected trading volume vs. $\\eta$")

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_bound_and_volume.png", dpi=150)
    plt.close(fig)


def _plot_holding_curves(x1: np.ndarray, holding_curves: dict[float, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for eta, holding in sorted(holding_curves.items()):
        ax.plot(x1, holding, label=rf"$\eta={eta:g}$")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"optimal holding $h_\eta(x_1)$")
    ax.set_title("Optimal holding function at selected impact levels")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp1_holding_function.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = run()
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    for row in summary["rows"]:
        print(row)
