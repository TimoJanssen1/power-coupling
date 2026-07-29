# DEPRECATED — do not quote these numbers

These are the artifacts of the first Q2 backtest (`scripts/run_q2_backtest.py`).
They are kept for the audit trail and deprecated for two independent reasons:

1. **It marks the index itself as hourly-tradeable.** The strategy enters and
   exits the spread at hourly observations with a z-score rule ("4.2-hour mean
   hold", Sharpe 5.54/4.69/3.65 across cost scenarios). Its successor's own
   docstring calls this fictitious: each leg settles at one auction price per
   delivery hour — there is nothing to exit into. The honest replacement
   (`scripts/run_q2_honest_backtest.py`, artifacts in
   `results/q2_backtest_honest/`) makes one decision per delivery day and holds
   to delivery.

2. **The series identification behind the whole Q2 framing was wrong.** The
   spread traded here is not "ID3 auction minus continuous VWAP" — the SMARD
   filter codes used actually return the Danish DK1 day-ahead price and the
   Belgian day-ahead price. See "Revision notes (July 2026)" in
   [FINDINGS.md](../../FINDINGS.md).

The inflated Sharpe ratios in `tables/summary.csv` should be read only as a
worked example of how marking an untradeable series at hourly frequency
manufactures performance.
