"""Parameters for every experiment, taken from the paper "Price-impact Martingale
optimal transport" (Section 5 and Appendix A). Each variable states its meaning,
unit and source; "not specified in the paper" marks a free numerical choice.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Experiment 1: single-asset forward-start call, lognormal marginals (Sec. 5.1.1)
# ---------------------------------------------------------------------------
EXP1_X0 = 1.0                 # initial spot, normalized to 1 (Sec. 5.1)
EXP1_T1 = 0.5                 # first fixing date, years (Sec. 5.1.1)
EXP1_T2 = 1.0                 # maturity, years (Sec. 5.1.1)
EXP1_K = 1.0                  # forward-start strike ratio k (Sec. 5.1.1, "k = 1")
EXP1_SIGMA1 = 0.195            # annualized volatility to T1 (Sec. 5.1.1)
EXP1_SIGMA2 = 0.175            # annualized volatility to T2 (Sec. 5.1.1)
EXP1_V1 = EXP1_T1 * EXP1_SIGMA1 ** 2   # v1 = T1*sigma1^2 (Sec. 5.1.1)
EXP1_V2 = EXP1_T2 * EXP1_SIGMA2 ** 2   # v2 = T2*sigma2^2 (Sec. 5.1.1)
EXP1_N_SUPPORT = 256            # support points per marginal (Appendix A.1)
# Thirteen effective impact coefficients at which (5.1) is solved (Sec. 5.1.1, Table 1)
EXP1_ETA_GRID = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2,
                           1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0])
EXP1_ETA_PLOT = (0.01, 0.1, 1.0)   # holding-function impact levels plotted (Sec. 5.1.1)
EXP1_UPPER_LOWER_TOL_BP = 2.2e-7   # paper's reported lower/upper agreement, basis points (Appendix A.1)

# ---------------------------------------------------------------------------
# Experiment 2: forward-start call, SPX-calibrated marginals (Sec. 5.1.2, App. A.2)
# ---------------------------------------------------------------------------
EXP2_SNAPSHOT_TIMESTAMP = "2026-10-09T02:25:46Z"   # Cboe quote snapshot time (Sec. 5.1.2)
EXP2_EXPIRY_1 = "2026-11-20"   # first SPX expiry (Sec. 5.1.2)
EXP2_EXPIRY_2 = "2026-12-18"   # second SPX expiry (Sec. 5.1.2)
EXP2_PARITY_MONEYNESS_LOW = 0.85    # parity-fit strike window, fraction of spot (Appendix A.2)
EXP2_PARITY_MONEYNESS_HIGH = 1.15   # parity-fit strike window, fraction of spot (Appendix A.2)
EXP2_CALIBRATION_MONEYNESS_LOW = 0.70    # marginal-fit strike window, K/F (Appendix A.2)
EXP2_CALIBRATION_MONEYNESS_HIGH = 1.30   # marginal-fit strike window, K/F (Appendix A.2)
EXP2_MIN_OPEN_INTEREST = 100     # minimum open interest filter, contracts (data_requirements)
EXP2_MAX_SPREAD_POINTS = 2.0     # maximum bid-ask spread, index points (data_requirements)
EXP2_MAX_SPREAD_FRACTION = 0.05  # maximum bid-ask spread, fraction of midpoint (data_requirements)
EXP2_MIN_SCALED_ERROR_FLOOR = 0.025   # floor on half-spread used to scale pricing errors, index points (Appendix A.2)
EXP2_SECOND_STAGE_TOLERANCE = 1e-8    # tolerance for "within of the minimum" in the two-stage fit (Appendix A.2)
# Support grid for the marginal fit: 181 regular nodes plus 7 extra nodes (methodology)
EXP2_GRID_REGULAR = np.round(np.arange(0.55, 1.45 + 1e-9, 0.005), 3)
EXP2_GRID_EXTRA = np.array([0.0, 0.2, 0.35, 0.45, 1.6, 1.8, 2.0])
EXP2_ETA0 = 0.01                  # impact coefficient for one claim, eta0 = kappa1+kappa2 (Sec. 5.1.2)
EXP2_ORDER_SIZES = (0.1, 1.0, 10.0)        # order sizes n, in units of the claim (Sec. 5.1.2)
EXP2_STRIKE_RATIOS = (0.90, 0.95, 1.00, 1.05, 1.10)   # strike ratios k (Sec. 5.1.2)
# Raw strike-fetch window from the option chain snapshot, wide enough to cover
# both the parity-fit window (0.85-1.15 of spot) and the calibration window
# (0.70-1.30 of forward) around the live SPX level (not specified in the
# paper; sized from a smoke test of the option-chain snapshot).
EXP2_STRIKE_FETCH_LOW = 3000.0
EXP2_STRIKE_FETCH_HIGH = 12000.0
EXP2_SUPPORT_MASS_TOL = 1e-6       # mass threshold to keep a calibration grid node as "supported" (not specified in the paper)

# ---------------------------------------------------------------------------
# Experiment 3: two-asset basket call under asymmetric impact (Sec. 5.2, App. A.3)
# ---------------------------------------------------------------------------
EXP3_N_MIDPOINTS = 16              # midpoints per marginal (Sec. 5.2)
EXP3_ASSET1_INTERVAL_T1 = (-1.0, 1.0)   # asset 1 support interval at T1 (Sec. 5.2)
EXP3_ASSET1_INTERVAL_T2 = (-3.0, 3.0)   # asset 1 support interval at T2 (Sec. 5.2)
EXP3_ASSET2_INTERVAL_T1 = (-3.0, 3.0)   # asset 2 support interval at T1 (Sec. 5.2)
EXP3_ASSET2_INTERVAL_T2 = (-3.0, 3.0)   # asset 2 support interval at T2 (Sec. 5.2)
EXP3_ETA_COMMON_GRID = np.array([1e-3, 1e-2, 1e-1, 1e0, 1e1])   # common eta1=eta2 grid (Sec. 5.2)
EXP3_ETA_TOTAL = 0.11               # fixed total eta1+eta2 for the (0.1,0.01)/(0.01,0.1) interchange pair (Fig. 3 caption)
# Four (eta1, eta2) pairs shown in Fig. 3's right panel: the middle two share
# the fixed total 0.11 (interchange study), the outer two match the common-eta
# curve at eta=0.01 and eta=0.1 (Sec. 5.2)
EXP3_ETA_PAIRS = ((0.01, 0.01), (0.1, 0.01), (0.01, 0.1), (0.1, 0.1))
EXP3_UPPER_LOWER_TOL = 8.1e-6       # paper's reported lower/upper agreement, payoff units (methodology)

# ---------------------------------------------------------------------------
# Solver configuration (not specified in the paper; chosen for numerical accuracy)
# ---------------------------------------------------------------------------
QP_SOLVER = "CLARABEL"   # interior-point conic solver used for the forward-start QP and the basket SOCP
QP_SOLVER_TOL = 1e-10    # solver feasibility/duality tolerance requested from cvxpy
LP_SOLVER = "CLARABEL"   # solver used for the static-hedge upper-bound linear programs
RANDOM_SEED = 0          # seed for any tie-breaking numerical routine (not specified in the paper)
