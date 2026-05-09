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

import os
import sys
import time
from pathlib import Path

import pandas as pd

os.environ.setdefault("ENTSOE_API_KEY", os.environ.get("ENTOSE_API_KEY", ""))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from power_microstructure.data import EntsoeFetcher, SmardFetcher  # noqa: E402

START = "2018-10-01"
END = (pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

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
print("\n[SMARD] full hourly price panel (DA, ID1, ID3, continuous)")
t0 = time.perf_counter()
sf = SmardFetcher()
panel = sf.price_panel(start=START, end=END)
print(f"  shape={panel.shape}  elapsed={time.perf_counter()-t0:.1f}s")
print("  non-null counts:")
for c in panel.columns:
    print(f"    {c:18}  n={int(panel[c].notna().sum())}")

print("\n[SMARD] quarter-hourly intraday continuous index")
t0 = time.perf_counter()
qh = sf.intraday_continuous_index_qh(start=START, end=END)
print(f"  shape={qh.shape}  elapsed={time.perf_counter()-t0:.1f}s")

print("\n[SMARD] solar generation (capacity-share proxy)")
t0 = time.perf_counter()
solar = sf.solar_generation(start=START, end=END)
print(f"  shape={solar.shape}  elapsed={time.perf_counter()-t0:.1f}s")

print("\nDone — all series cached under ~/.cache/power_microstructure/")
