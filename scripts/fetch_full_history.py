"""
Pull the full 2018-10 → today panel for the paper.

Run with:
    .venv/bin/python scripts/fetch_full_history.py

First run is slow (~30–60 min). All results cached as parquet — subsequent
calls are seconds.

We start from 2018-10-01 because that's when the German bidding zone (DE_LU)
went live. Pre-Oct-2018 data lives under DE_AT_LU and changes the population
under study; the spec's hypothesis is about DE post-coupling.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from power_microstructure.data import EntsoeFetcher, SmardFetcher  # noqa: E402
from power_microstructure.runconfig import resolve_end_date  # noqa: E402

START = "2018-10-01"
END = resolve_end_date()  # pinned to 2026-05-04 unless --end/PM_END says otherwise

print(f"window: {START} → {END}")

# ENTSO-E -------------------------------------------------------------------
print("\n[ENTSO-E] forecast_error  (wind+solar forecast vs actual, hourly)")
t0 = time.perf_counter()
ef = EntsoeFetcher()
errors = ef.forecast_error(start=START, end=END)
print(f"  shape={errors.shape}  elapsed={time.perf_counter()-t0:.1f}s")

print("\n[ENTSO-E] day-ahead prices")
t0 = time.perf_counter()
da = ef.day_ahead_prices(start=START, end=END)
print(f"  shape={getattr(da, 'shape', len(da))}  elapsed={time.perf_counter()-t0:.1f}s")

print("\n[ENTSO-E] actual load (regime control)")
t0 = time.perf_counter()
load = ef.actual_load(start=START, end=END)
print(f"  shape={getattr(load, 'shape', len(load))}  elapsed={time.perf_counter()-t0:.1f}s")

print("\n[ENTSO-E] renewable_share (interaction variable)")
t0 = time.perf_counter()
rshare = ef.renewable_share(start=START, end=END)
print(f"  shape={rshare.shape}  elapsed={time.perf_counter()-t0:.1f}s")

# SMARD ---------------------------------------------------------------------
# NOTE (July 2026): the SMARD "Marktpreis" filters are zonal DAY-AHEAD prices.
# The panel's legacy column names map to: da_price = DE/LU DA, id3_price =
# Danish DK1 DA, id_continuous = Belgian DA, id1_price = empty (invalid filter).
print("\n[SMARD] full hourly day-ahead price panel (DE/LU, DK1, BE; legacy column names)")
t0 = time.perf_counter()
sf = SmardFetcher()
panel = sf.price_panel(start=START, end=END)
print(f"  shape={panel.shape}  elapsed={time.perf_counter()-t0:.1f}s")
print("  non-null counts:")
for c in panel.columns:
    print(f"    {c:18}  n={int(panel[c].notna().sum())}")

print("\n[SMARD] quarter-hourly 'Anrainer DE/LU' price series (legacy name: QH intraday index)")
t0 = time.perf_counter()
qh = sf.intraday_continuous_index_qh(start=START, end=END)
print(f"  shape={qh.shape}  elapsed={time.perf_counter()-t0:.1f}s")

print("\n[SMARD] solar (PV) generation")
t0 = time.perf_counter()
solar = sf.solar_generation(start=START, end=END)
print(f"  shape={solar.shape}  elapsed={time.perf_counter()-t0:.1f}s")

print("\nDone — all series cached under ~/.cache/power_microstructure/")
