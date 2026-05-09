"""
Q2 mean-reversion strategy — PRE-REGISTERED SPECIFICATION.

This file is committed to git BEFORE the backtest is run. Any subsequent
tuning, sensitivity analysis, or robustness check is reported separately;
the headline Sharpe is whatever this exact configuration produces.

Strategy:
  Trade the spread  s = ID3 − continuous  (the leg with the longest history).
  Use a rolling z-score over the training window:
      z_t = (s_t − mean(train)) / std(train)
  Entry:  enter at |z| > ENTRY_Z, direction = −sign(z)
          (we trade the auction side; VECM α-directionality says it's the
          mover, not the continuous market)
  Exit:   |z| < EXIT_Z   OR   hold ≥ MAX_HOLD   OR   |z| > STOP_LOSS_Z
          against the position
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
