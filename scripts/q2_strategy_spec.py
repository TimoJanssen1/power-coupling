"""
Q2 mean-reversion strategy — specification for the DEPRECATED hourly-marks
backtest (``run_q2_backtest.py``).

HONESTY NOTES (July 2026 revision):
  * "Pre-registered" was the intent, but the git history cannot verify it —
    all code and results landed together in the repo's initial commit.
  * This spec parameterises the abandoned z-score entry/exit strategy, NOT
    the honest hold-to-delivery backtest (whose parameters are declared at
    the top of ``run_q2_honest_backtest.py``).
  * The spread "s = ID3 − continuous" is, per the series identification in
    FINDINGS.md, actually DK1 day-ahead − Belgian day-ahead. Hourly entries
    and exits on day-ahead auction prices are not executable; that is why
    this backtest is deprecated.
"""

from __future__ import annotations

# ----- Strategy parameters (locked) ------------------------------------------
ENTRY_Z       = 2.0
EXIT_Z        = 0.5
MAX_HOLD      = 24      # hours
STOP_LOSS_Z   = 4.0
TRAIN_MONTHS  = 12
TEST_MONTHS   = 3

# ----- Cost scenarios --------------------------------------------------------
# Each scenario specifies a round-trip cost function.
COST_SCENARIOS = {
    "optimistic": {
        # Exchange member, no slippage (theoretical floor).
        "exchange_fee_eur_per_mwh_per_side": 0.03,
        "impact_pct_of_sigma": 0.0,
    },
    "realistic": {
        # Mid-tier intraday participant. Half-spread + slippage per side
        # ~3 €/MWh on the continuous leg (the auction leg has zero exec risk
        # because it clears at one price). Round-trip ≈ 6 €/MWh at σ ≈ 30.
        "exchange_fee_eur_per_mwh_per_side": 0.03,
        "impact_pct_of_sigma": 0.10,   # 10% × σ × 2 sides ≈ 6 €/MWh
    },
    "pessimistic": {
        # Retail-sized order against a thin orderbook. Round-trip ≈ 12 €/MWh.
        "exchange_fee_eur_per_mwh_per_side": 0.03,
        "impact_pct_of_sigma": 0.20,
    },
}

# ----- Bootstrap inference ---------------------------------------------------
N_BOOTSTRAP = 2000

# ----- Random seed -----------------------------------------------------------
SEED = 20260507
