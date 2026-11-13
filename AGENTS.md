# Repository guide for coding agents

This repository reproduces three numerical experiments from *Price-impact
Martingale optimal transport*: dual price bounds for a seller/buyer of a
forward-start or basket option under quadratic price impact, computed from
the discretized convex programs of the paper's Appendix A.

## Layout

```
src/        shared library — no experiment-specific code here
  config.py          every paper-derived parameter (module-level, documented)
  marginals.py        conditional-average lognormal discretization (App. A.1)
  pimot.py             the single-asset PIMOT QP, MOT/OT limits, static-hedge
                        upper estimate, expected trading volume (App. A.1)
  spx_calibration.py   put-call parity fit and SPX marginal-calibration LP
                        (Sec. 5.1.2, App. A.2)
  basket.py            the two-asset basket SOCP under asymmetric impact
                        (Sec. 5.2, App. A.3)
exp/        one script per experiment (exp1.py, exp2.py, exp3.py), calling
            only the public functions in src/
data/       data_loader.py — Massive REST option-chain snapshots for exp2;
            experiments 1 and 3 need no market data
tests/      pytest unit tests of src/ components (synthetic/small fixtures,
            including a synthetic option chain for spx_calibration.py)
results/    RESULTS.md plus every JSON/CSV/PNG written by the experiments
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -m "not market_data"   # unit tests (no live calls)
python -m exp.exp1                      # Sec. 5.1.1
python -m exp.exp2                      # Sec. 5.1.2 (needs MASSIVE_API_KEY in .env)
python -m exp.exp3                      # Sec. 5.2
bash reproduce.sh                       # tests + all three experiments
```

No test in this repository is marked `market_data`: experiment 2's live
Massive call is exercised only by running `exp.exp2` itself, not by the
pytest suite.

## Conventions

- Every paper-derived constant lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment giving its meaning, unit and the
  paper section it comes from; "not specified in the paper" marks a free
  numerical choice.
- Market data is loaded only through `data/data_loader.py` (Massive REST
  option-chain snapshots); no other module calls an external data API.
  `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` (see `.env.example`) and never appear in code, logs or results.
- Running an experiment script overwrites its files under `results/`; the
  numbers and verdicts in `results/RESULTS.md` are written by hand from
  those files and must be re-checked after a rerun, not computed live by
  `RESULTS.md` itself.
- Experiments run as `python -m exp.<name>` and tests as `python -m pytest`
  from the repository root; no script edits `sys.path`.
