"""Experiment 3: two-asset basket call under asymmetric quadratic price impact
(Sec. 5.2 and Appendix A.3 of "Price-impact Martingale optimal transport").

Run: python -m exp.exp3
"""

from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.basket import midpoints, BasketMarginals, solve_basket_ot_or_mot, solve_basket_pimot

RESULTS_DIR = "results"


def _build_marginals() -> BasketMarginals:
    n = config.EXP3_N_MIDPOINTS
    return BasketMarginals(
        x1=midpoints(config.EXP3_ASSET1_INTERVAL_T1, n),
        x2=midpoints(config.EXP3_ASSET2_INTERVAL_T1, n),
        y1=midpoints(config.EXP3_ASSET1_INTERVAL_T2, n),
        y2=midpoints(config.EXP3_ASSET2_INTERVAL_T2, n),
        n=n,
    )


def run() -> dict:
    marg = _build_marginals()

    mot = solve_basket_ot_or_mot(marg, martingale=True, solver=config.QP_SOLVER,
                                  solver_tol=config.QP_SOLVER_TOL)
    ot = solve_basket_ot_or_mot(marg, martingale=False, solver=config.QP_SOLVER,
                                 solver_tol=config.QP_SOLVER_TOL)

    common_rows = []
    for eta in config.EXP3_ETA_COMMON_GRID:
        res = solve_basket_pimot(marg, float(eta), float(eta), solver=config.QP_SOLVER,
                                  solver_tol=config.QP_SOLVER_TOL)
        common_rows.append({
            "eta": float(eta),
            "value_lower": res.value_lower,
            "value_upper": res.value_upper,
            "midpoint": res.midpoint,
            "gap": res.gap,
            "shares_traded_1": res.shares_traded_1,
            "shares_traded_2": res.shares_traded_2,
            "status": res.status,
        })

    pair_rows = []
    for eta1, eta2 in config.EXP3_ETA_PAIRS:
        # Only the (0.1, 0.01) and (0.01, 0.1) pairs share the fixed total
        # eta1+eta2=0.11 (Fig. 3 caption); (0.01, 0.01) and (0.1, 0.1) are the
        # matching points on the common-eta curve, included for comparison.
        res = solve_basket_pimot(marg, eta1, eta2, solver=config.QP_SOLVER,
                                  solver_tol=config.QP_SOLVER_TOL)
        pair_rows.append({
            "eta1": eta1, "eta2": eta2,
            "value_lower": res.value_lower,
            "value_upper": res.value_upper,
            "midpoint": res.midpoint,
            "gap": res.gap,
            "shares_traded_1": res.shares_traded_1,
            "shares_traded_2": res.shares_traded_2,
            "status": res.status,
        })

    max_gap = max(r["gap"] for r in common_rows + pair_rows)
    summary = {
        "mot_upper_bound": mot.value_upper,
        "ot_upper_bound": ot.value_upper,
        "mot_status": mot.status,
        "ot_status": ot.status,
        "max_lower_upper_gap": max_gap,
        "paper_tolerance": config.EXP3_UPPER_LOWER_TOL,
        "common_eta_rows": common_rows,
        "asymmetric_pair_rows": pair_rows,
    }
    with open(f"{RESULTS_DIR}/exp3_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    _plot_common_eta(common_rows, mot.value_upper, ot.value_upper)
    _plot_asymmetric_pairs(pair_rows)
    return summary


def _plot_common_eta(rows: list[dict], mot_bound: float, ot_bound: float) -> None:
    etas = [r["eta"] for r in rows]
    bounds = [r["midpoint"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.semilogx(etas, bounds, "o-", label=r"PIMOT upper bound ($\eta_1=\eta_2=\eta$)")
    ax.axhline(mot_bound, color="tab:green", linestyle="--", label=f"MOT bound ({mot_bound:.4f})")
    ax.axhline(ot_bound, color="tab:red", linestyle="--", label=f"OT bound ({ot_bound:.4f})")
    ax.set_xlabel(r"common effective impact coefficient $\eta$")
    ax.set_ylabel("basket call upper bound")
    ax.set_title("Two-asset basket call: convergence from MOT to OT")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_common_eta.png", dpi=150)
    plt.close(fig)


def _plot_asymmetric_pairs(rows: list[dict]) -> None:
    labels = [f"({r['eta1']:g}, {r['eta2']:g})" for r in rows]
    bounds = [r["midpoint"] for r in rows]
    shares1 = [r["shares_traded_1"] for r in rows]
    shares2 = [r["shares_traded_2"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.bar(labels, bounds, color="tab:blue")
    ax.set_xlabel(r"$(\eta_1, \eta_2)$, fixed total")
    ax.set_ylabel("basket call upper bound")
    ax.set_title("Effect of interchanging impact coefficients")

    ax = axes[1]
    width = 0.35
    x_pos = np.arange(len(labels))
    ax.bar(x_pos - width / 2, shares1, width, label="asset 1 shares traded")
    ax.bar(x_pos + width / 2, shares2, width, label="asset 2 shares traded")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"$(\eta_1, \eta_2)$, fixed total")
    ax.set_ylabel("expected shares traded per claim")
    ax.set_title("Trading volume by asset")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{RESULTS_DIR}/exp3_asymmetric_pairs.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    summary = run()
    print(json.dumps(
        {k: v for k, v in summary.items() if k not in ("common_eta_rows", "asymmetric_pair_rows")},
        indent=2))
    for row in summary["common_eta_rows"]:
        print(row)
    for row in summary["asymmetric_pair_rows"]:
        print(row)
