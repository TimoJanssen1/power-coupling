# Deprecated: do not quote these numbers

These are the artifacts of the first Q2 backtest (`scripts/run_q2_backtest.py`).
They are kept for the audit trail and deprecated for two independent reasons:

1. It marks the index itself as hourly-tradeable. The strategy enters and exits
   the spread at hourly observations with a z-score rule ("4.2-hour mean hold",
   Sharpe 5.54/4.69/3.65 across cost scenarios). Each leg actually settles at
   one auction price per delivery hour, so there is nothing to exit into. The
   replacement (`scripts/run_q2_honest_backtest.py`, artifacts in
   `results/q2_backtest_honest/`) makes one decision per delivery day and holds
   to delivery.

2. The series identification behind the original Q2 framing was wrong. The
   spread traded here is not "ID3 auction minus continuous VWAP"; the SMARD
   filter codes used return the Danish DK1 and Belgian day-ahead prices. See
   the revision history in [FINDINGS.md](../../FINDINGS.md).

Read the Sharpe ratios in `tables/summary.csv` only as a worked example of how
marking an untradeable series at hourly frequency manufactures performance.
