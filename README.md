# Price-impact Martingale Optimal Transport

Reproduction of three numerical experiments from *Price-impact Martingale
optimal transport*: seller/buyer price bounds for forward-start and basket
options under quadratic price impact, computed via the dual formulation of
the price-impact martingale optimal transport (PIMOT) problem.

## What is implemented

- **Experiment 1** (Sec. 5.1.1): single-asset forward-start call on two
  discretized lognormal marginals. Solves the PIMOT quadratic program over a
  13-point grid of effective impact coefficients η, showing the seller's upper
  bound interpolate between the martingale optimal transport (MOT) and optimal
  transport (OT) limits, and plots the optimal holding function.
- **Experiment 2** (Sec. 5.1.2, Appendix A.2): the same forward-start call with
  marginals calibrated to an SPX option chain for two maturities, across order
  sizes n ∈ {0.1, 1, 10} and strike ratios k ∈ {0.90, 0.95, 1.00, 1.05, 1.10}.
- **Experiment 3** (Sec. 5.2, Appendix A.3): a two-asset basket call under
  asymmetric per-asset impact coefficients, showing how interchanging the
  coefficients between the two assets changes the seller's upper bound.

Each experiment solves the discretized convex program described in the
paper's Appendix A (a quadratic program for the single-asset case, a
second-order cone program for the two-asset case) via `cvxpy`, computes a
static-hedge upper estimate to bound the duality gap, and reports the
resulting price bounds and expected trading volumes.

## Repository layout

```
src/            shared library: marginal discretization, the PIMOT QP/SOCP/LP
                solvers, SPX calibration, and src/config.py (all parameters)
exp/            exp1.py, exp2.py, exp3.py — one script per experiment
data/           data_loader.py — Massive REST option-chain snapshots (exp. 2)
tests/          pytest unit tests of the shared src/ components
results/        RESULTS.md and every metric/plot produced by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Experiment 2 downloads a live SPX option chain snapshot through the Massive
REST API. Copy `.env.example` to `.env` and set `MASSIVE_API_KEY` (get a key at
https://massive.com/); experiments 1 and 3 need no API key.

## Running

```bash
python -m pytest                      # unit tests of src/
python -m exp.exp1                    # Sec. 5.1.1: lognormal marginals
python -m exp.exp2                    # Sec. 5.1.2: SPX-calibrated marginals (needs MASSIVE_API_KEY)
python -m exp.exp3                    # Sec. 5.2: two-asset basket call
```

or `bash reproduce.sh`, which runs the tests and all three experiments in
order. Every run overwrites the files in `results/`; see `results/RESULTS.md`
for the comparison against the paper's reported values and the recorded
deviations.

## Parameters

All paper-derived parameters (lognormal volatilities, discretization grids,
η grids, SPX strike windows, solver settings) live as module-level variables
in `src/config.py`, each documented with its meaning, unit and the paper
section it comes from.
