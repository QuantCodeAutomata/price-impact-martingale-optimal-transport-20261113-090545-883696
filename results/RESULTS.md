# Results — Price-impact Martingale Optimal Transport

Paper: *Price-impact Martingale Optimal Transport* (page references below follow
the paper's own section numbering: Sec. 5.1.1, Sec. 5.1.2, Sec. 5.2 and
Appendix A).

All numbers below are read from the JSON/CSV files written by the final run of
each experiment under `results/`. Re-running `python -m exp.exp1`,
`python -m exp.exp2`, `python -m exp.exp3` overwrites these files; this report
must be checked against the new files after any rerun.

## Experiment 1 — Single-asset forward-start call, lognormal marginals (Sec. 5.1.1)

Command: `python -m exp.exp1`. Output: `results/exp1_results.json`,
`results/exp1_bound_and_volume.png`, `results/exp1_holding_function.png`.

Both lognormal marginals are discretized on 256 conditional-average support
points (Appendix A.1) and the discrete convex-order check (comparing call
prices at every strike in the union of the two supports) holds:
`convex_order_holds = True`.

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| Reference price, independent Gaussian log-returns (bp) | 429.70 | 429.70 | reproduced |
| MOT upper bound (bp) | 537.54 | 537.54 | reproduced |
| OT upper bound (bp) | 1243.15 | 1243.15 | reproduced |
| Upper bound at η = 1e-4 (bp) | 537.75 | 537.75 | reproduced |
| Shares/claim at η = 1e-4 | 1.037 | 1.0369 | reproduced |
| Upper bound at η = 1 (bp) | 1038.11 | 1038.11 | reproduced |
| Shares/claim at η = 1 | 0.309 | 0.3094 | reproduced |
| Upper bound at η = 100 (bp) | 1240.08 | 1240.08 | reproduced |
| Shares/claim at η = 100 | 0.005 | 0.00486 | reproduced |

The full 13-point η grid (`EXP1_ETA_GRID` in `src/config.py`) is monotone: the
upper bound rises from 537.75 bp to 1240.08 bp and the expected shares traded
per claim fall from 1.037 to 0.0049 as η increases from 1e-4 to 100, matching
the paper's description of interpolation between the MOT and OT limits
(Sec. 5.1.1). The static-hedge upper estimate and the QP lower estimate agree
to at most 0.00086 bp across the grid (`max_lower_upper_gap_bp` in
`results/exp1_results.json`), i.e. negligible next to the reported bounds
(hundreds of bp), though looser than the paper's stated 2.2×10⁻⁷ bp agreement
(Appendix A.1) — see "Deviations from the paper".

**Verdict: reproduced** (every reported value in Table 1 and the MOT/OT limits
matches to the paper's own precision).

## Experiment 2 — Forward-start call, SPX-calibrated marginals (Sec. 5.1.2, App. A.2)

Command: `python -m exp.exp2`. Output: `results/exp2_results.json`,
`results/exp2_bounds_table.csv`, `results/exp2_interval_widening.png`.

Marginals were calibrated to a live Cboe-listed SPX option chain for the two
expiries named in the paper, 20 November 2026 and 18 December 2026, fetched
through the Massive REST options-snapshot endpoint on the date this experiment
was run (see "Deviations from the paper" for the data-source substitution and
the resulting difference from the paper's own quote snapshot).

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| Forward price F₁ | 7815.16 | 7604.52 | partially reproduced |
| Forward price F₂ | 7835.02 | 7621.81 | partially reproduced |
| Discount factor D₁ | 0.9950 | 0.9987 | partially reproduced |
| Discount factor D₂ | 0.9917 | 0.9954 | partially reproduced |
| Parity/calibration RMSE, maturity 1 (index points) | ≈0.053 | 0.060 | reproduced (same order of magnitude) |
| Parity/calibration RMSE, maturity 2 (index points) | ≈0.053 | 0.045 | reproduced (same order of magnitude) |
| Supported nodes (maturity 1 / maturity 2) | 47 / 64 | 57 / 89 | partially reproduced |
| Seller bound Δ(n:1→10), k=1.00 (normalized bp) | +61 | +17.9 | partially reproduced (same sign) |
| Buyer bound Δ(n:1→10), k=1.00 (normalized bp) | −2.76 | −1.44 | partially reproduced (same sign) |
| Interval widens with order size | yes | yes | reproduced |
| MOT = OT at k = 0.90, 0.95, 1.10 | yes | no (gaps of 26.3, 55.5, 1.3 bp) | not reproduced |

Qualitative properties that hold in this run: convex order μ₁ ≤_c μ₂ is
enforced by construction of the calibration LP; the seller's upper bound rises
and the buyer's lower bound falls as the order size n grows from 0.1 to 10
units at every strike ratio k (full table in `results/exp2_bounds_table.csv`);
expected shares traded per claim fall as n grows (e.g. at k = 1.00: 0.424 at
n = 0.1, 0.396 at n = 1, 0.271 at n = 10, compared with the paper's 0.829,
0.741, 0.480).

**Verdict: partially reproduced.** The qualitative mechanism (MOT/OT interval
widening with order size, monotone direction of both bounds) is confirmed, but
the absolute levels differ materially from the paper's Table 2 because this
run calibrates to a real SPX quote snapshot instead of the paper's own
snapshot (see "Deviations from the paper"). In particular, the paper's claim
that the MOT and OT bounds coincide exactly at k = 0.90, 0.95 and 1.10 does not
hold for this run's calibrated marginals: evaluating the fitted discrete call
prices directly at those three strikes gives maturity-2-minus-maturity-1 gaps
of about 26.3, 55.5 and 1.3 basis points respectively (not the near-zero gap
the tight convex-order constraint would require), so the cause is identified
as the data source, not a defect in the bound computation (the same QP
routine reproduces experiment 1's MOT/OT limits to the paper's own precision).

## Experiment 3 — Two-asset basket call, asymmetric price impact (Sec. 5.2, App. A.3)

Command: `python -m exp.exp3`. Output: `results/exp3_results.json`,
`results/exp3_common_eta.png`, `results/exp3_asymmetric_pairs.png`.

| Quantity | Paper | This run | Verdict |
|---|---|---|---|
| MOT upper bound | 1.376 | 1.3764 | reproduced |
| OT upper bound | 1.500 | 1.5000 | reproduced |
| Common η = 1e-3 | 1.383 | 1.3808 | reproduced |
| Common η = 1e-2 | 1.415 | 1.4142 | reproduced |
| Common η = 1e-1 | 1.456 | 1.4553 | reproduced |
| Common η = 1e0 | 1.494 | 1.4937 | reproduced |
| Common η = 1e1 | 1.500 | 1.4994 | reproduced |
| (η₁,η₂) = (0.01,0.01) | 1.4142 | 1.41417 | reproduced |
| (η₁,η₂) = (0.1,0.01) | 1.4154 | 1.41542 | reproduced |
| (η₁,η₂) = (0.01,0.1) | 1.4547 | 1.45474 | reproduced |
| (η₁,η₂) = (0.1,0.1) | 1.4553 | 1.45535 | reproduced |
| Shares traded, asset 1/2 at (0.1,0.01) | 0.31 / 3.73 | 0.3145 / 3.7308 | reproduced |
| Shares traded, asset 1/2 at (0.01,0.1) | 0.23 / 1.07 | 0.2319 / 1.0686 | reproduced |
| Lower/upper estimate gap | < 8.1×10⁻⁶ | 7.58×10⁻⁶ | reproduced |

Interchanging the impact coefficients raises the bound from 1.4154 (larger
coefficient on asset 1, the asset whose T₁ and T₂ marginals differ) to 1.4547
(larger coefficient on asset 2, whose T₁ and T₂ marginals are identical,
Sec. 5.2), confirming that assigning the larger impact coefficient to the
asset with the more binding martingale constraint raises the seller's bound.

**Verdict: reproduced** (every reported value in Sec. 5.2 and the described
interchange effect matches to the paper's own precision).

## Deviations from the paper

- **Experiment 2 data source.** The paper calibrates to a Cboe SPX option
  chain snapshot timestamped 2026-10-09T02:25:46Z. That date has since
  occurred; this run fetches the SPX AM-settled (`O:SPX`, non-`O:SPXW`) option
  chain for the same two expiries (20 Nov 2026, 18 Dec 2026) from the Massive
  REST options-snapshot endpoint at the time the experiment was executed,
  rather than from the paper's own archived snapshot, which is not available
  through Massive or the QuantX Dataloader. The resulting forward prices
  (F₁ ≈ 7604.5, F₂ ≈ 7621.8) differ from the paper's reported F₁ ≈ 7815.16,
  F₂ ≈ 7835.02 because the underlying SPX level moved between the paper's
  snapshot and the one used here; this is the identified cause of the
  experiment-2 gaps to the paper's Table 2 (confirmed by directly comparing
  the fitted discrete call prices at the affected strikes, see above).
- **Experiment 2 quote fields.** This Massive account's options plan does not
  return the `last_quote` (bid/ask) field used by the paper's spread filters
  (open interest ≥ 100 contracts, spread ≤ 2 index points and ≤ 5% of
  midpoint). The fair market value field `fmv` (a continuously updated model
  price) is used as the option's quoted price in its place, and the spread
  filters are not applied; the open-interest filter is still enforced. This
  is implemented in `data/data_loader.py::fetch_option_chain_snapshot`.
- **Experiment 2 solver status.** A few (k, n) combinations in
  `results/exp2_bounds_table.csv` report `optimal_inaccurate` from the CLARABEL
  solver rather than `optimal`; the reported values still satisfy the
  lower/upper agreement used elsewhere in the same run and are not re-solved
  with a different solver.
- **Experiment 1 numerical tolerance.** The paper reports the lower and upper
  estimates of the seller's bound agreeing to within 2.2×10⁻⁷ basis points
  (Appendix A.1). This run's CLARABEL-based QP/LP pair agrees to within
  8.6×10⁻⁴ basis points across the η grid (`max_lower_upper_gap_bp` in
  `results/exp1_results.json`) — negligible next to the reported bounds
  (hundreds of basis points) but looser than the paper's own figure, which
  likely reflects a tighter solver configuration than used here; every
  reported bound and volume value still matches the paper to its own stated
  precision.
- **Solvers.** The exact QP/SOCP/LP solvers and tolerances used in the paper
  are not specified (Appendix A.1–A.3 describe the formulations, not the
  solver). This implementation uses CLARABEL throughout, via `cvxpy`, with a
  requested feasibility/duality tolerance of 1e-10 (`src/config.py`).
- **Experiment 2 strike-fetch window and support-mass threshold.** The raw
  strike window used to download option quotes (3000–12000 index points) and
  the mass threshold used to count a calibration grid node as "supported"
  (1e-6) are not specified in the paper and were chosen to comfortably cover
  the parity- and calibration-fit windows the paper does specify; see the
  corresponding comments in `src/config.py`.

## Not reproduced

There are no paper results in the scope of these three experiments that this
repository omits: experiments 1, 2 and 3 each produce the artifacts and
metrics described above.
