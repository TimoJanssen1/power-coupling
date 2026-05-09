"""
Q3 / battery operator simulation.

Models a 100 MW / 200 MWh utility-scale BESS participating in the German DA
+ intraday markets over the full 2018-10 → today sample. Runs four
strategies:

    naive               cheapest-2h charge / richest-2h discharge per day
    da_lp               LP optimised against DA prices (baseline)
    da_plus_tilt        DA + Q1 forecast-error tilt (the "smart" strategy)
    perfect_foresight   LP against realised intraday prices (ceiling)

Outputs revenue per year, revenue per cycle, charge/discharge profile by
hour-of-day, and uplift attribution.

Usage:
    .venv/bin/python scripts/run_q3_battery.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from power_microstructure.data import EntsoeFetcher, SmardFetcher  # noqa: E402
from power_microstructure.strategy.battery import (  # noqa: E402
    BatterySpec,
    run_all_strategies,
)

OUT = ROOT / "results" / "q3_battery"
OUT.mkdir(parents=True, exist_ok=True)
(FIG := OUT / "figures").mkdir(exist_ok=True)
(TAB := OUT / "tables").mkdir(exist_ok=True)

START = "2018-10-01"
END = (pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10,
})


def _save(fig, name):
    p = FIG / f"{name}.png"
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

print(f"Sample window: {START} → {END}")
print("Loading cached data …")
sf = SmardFetcher()
ef = EntsoeFetcher()

panel = sf.price_panel(start=START, end=END)
panel.index = pd.to_datetime(panel.index, utc=True)
da = panel["da_price"].dropna()
idc = panel["id_continuous"].dropna()

errors = ef.forecast_error(start=START, end=END)
errors.index = pd.to_datetime(errors.index, utc=True) if errors.index.tz is None \
    else errors.index.tz_convert("UTC")
fe = errors["total_error"]

# Align hourly grid
common = da.index.intersection(idc.index).intersection(fe.index)
da, idc, fe = da.loc[common], idc.loc[common], fe.loc[common]
print(f"  hourly grid: n={len(common):,}  start={common[0]}  end={common[-1]}")

# Battery
spec = BatterySpec(
    power_mw=100.0,
    energy_mwh=200.0,
    eta_round_trip=0.85,
    wear_eur_per_mwh=2.0,
    max_cycles_per_day=1.5,
    soc_init=100.0,
)
print(f"Battery: {spec.power_mw} MW / {spec.energy_mwh} MWh, "
      f"η_round_trip={spec.eta_round_trip:.2f}, wear={spec.wear_eur_per_mwh} €/MWh")

# ---------------------------------------------------------------------------
# Run all strategies
# ---------------------------------------------------------------------------

print("\nRunning all strategies (this is the slow step — daily LP × ~2 700 days × 3 strategies) …")
results = run_all_strategies(da, idc, fe, spec)

# ---------------------------------------------------------------------------
# Headline summary
# ---------------------------------------------------------------------------

n_years = (common[-1] - common[0]).total_seconds() / (365.25 * 24 * 3600)

rows = []
for name, r in results.items():
    rev = r.revenue_eur
    cycles = r.n_full_cycles
    rows.append({
        "strategy": name,
        "revenue_eur_total": round(rev, 0),
        "revenue_eur_per_year": round(rev / n_years, 0),
        "revenue_eur_per_mw_per_year": round(rev / n_years / spec.power_mw, 0),
        "n_full_cycles": round(cycles, 1),
        "rev_per_cycle_eur": round(r.revenue_per_full_cycle, 1),
        "rev_per_mwh_throughput": round(r.revenue_per_mwh_throughput, 2),
    })
summary = pd.DataFrame(rows).set_index("strategy")
summary.to_csv(TAB / "headline_summary.csv")
print("\n=== Headline summary ===")
print(summary.to_string())

# Uplift attribution (vs naive)
naive_rev = float(summary.loc["naive", "revenue_eur_per_year"])
uplift = (summary["revenue_eur_per_year"] - naive_rev).round(0)
uplift_pct = (uplift / naive_rev * 100).round(1)
attribution = pd.DataFrame({
    "annual_rev_eur": summary["revenue_eur_per_year"],
    "uplift_vs_naive_eur": uplift,
    "uplift_pct": uplift_pct,
    "rev_per_cycle_eur": summary["rev_per_cycle_eur"],
})
attribution.to_csv(TAB / "uplift_attribution.csv")
print("\n=== Uplift attribution (vs naive baseline) ===")
print(attribution.to_string())

# ---------------------------------------------------------------------------
# Annual revenue chart
# ---------------------------------------------------------------------------

annual_rows = []
for name, r in results.items():
    s = r.schedule
    rev_h = (s["discharge_mwh"] - s["charge_mwh"]) * s["settlement_price"] - \
            spec.wear_eur_per_mwh * s["discharge_mwh"]
    yr = rev_h.groupby(rev_h.index.year).sum()
    for y, v in yr.items():
        annual_rows.append({"strategy": name, "year": int(y), "revenue_eur": float(v)})
annual = pd.DataFrame(annual_rows)
annual.to_csv(TAB / "annual_revenue.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4.5))
years = sorted(annual["year"].unique())
strategies = ["naive", "da_lp", "da_plus_tilt", "perfect_foresight"]
colors = {"naive": "C7", "da_lp": "C0", "da_plus_tilt": "C2", "perfect_foresight": "C3"}
x = np.arange(len(years))
width = 0.20
for i, strat in enumerate(strategies):
    sub = annual[annual["strategy"] == strat].set_index("year")
    vals = [sub.loc[y, "revenue_eur"] / 1e6 if y in sub.index else 0 for y in years]
    ax.bar(x + (i - 1.5) * width, vals, width=width, label=strat, color=colors[strat], alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_ylabel("Annual revenue  (M EUR)")
ax.set_title(f"BESS annual revenue by strategy  —  {spec.power_mw:.0f} MW / {spec.energy_mwh:.0f} MWh")
ax.legend(loc="upper left")
_save(fig, "q3_annual_revenue")

# ---------------------------------------------------------------------------
# Hour-of-day charge/discharge profile (vs the duck curve)
# ---------------------------------------------------------------------------

hod_rows = []
for name, r in results.items():
    s = r.schedule.copy()
    s["hour"] = s.index.hour
    by_hr = s.groupby("hour")[["charge_mwh", "discharge_mwh"]].mean()
    by_hr["net_dispatch_mwh"] = by_hr["discharge_mwh"] - by_hr["charge_mwh"]
    by_hr["strategy"] = name
    hod_rows.append(by_hr.reset_index())
hod = pd.concat(hod_rows)
hod.to_csv(TAB / "hour_of_day_profile.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4.5))
for strat in strategies:
    sub = hod[hod["strategy"] == strat].set_index("hour")
    ax.plot(sub.index, sub["net_dispatch_mwh"], marker="o", lw=1.6,
            color=colors[strat], label=strat)
ax.axhline(0, color="black", lw=0.5)
ax.axvspan(10, 14, color="orange", alpha=0.10, label="solar peak")
ax.axvspan(17, 21, color="purple", alpha=0.10, label="evening ramp")
ax.set_xticks(np.arange(24))
ax.set_xlabel("hour of day (CET)"); ax.set_ylabel("avg net dispatch (MWh; +discharge / −charge)")
ax.set_title("Q3: BESS dispatch profile by hour-of-day — strategies and the duck curve")
ax.legend(loc="upper right")
_save(fig, "q3_hour_of_day_dispatch")

# ---------------------------------------------------------------------------
# Cumulative revenue chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4.5))
for strat in strategies:
    s = results[strat].schedule
    rev_h = (s["discharge_mwh"] - s["charge_mwh"]) * s["settlement_price"] - \
            spec.wear_eur_per_mwh * s["discharge_mwh"]
    cum = rev_h.cumsum() / 1e6
    ax.plot(cum.index, cum.values, lw=1.5, color=colors[strat],
            label=f"{strat}  (€{results[strat].revenue_eur/1e6:.1f}M total)")
ax.set_ylabel("Cumulative revenue (M EUR)")
ax.set_title(f"BESS cumulative revenue  —  {spec.power_mw:.0f} MW / {spec.energy_mwh:.0f} MWh")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
_save(fig, "q3_cumulative_revenue")

# ---------------------------------------------------------------------------
# Per-cycle revenue distribution
# ---------------------------------------------------------------------------

cycle_rows = []
for name, r in results.items():
    s = r.schedule.copy()
    s["day"] = s.index.date
    daily = s.groupby("day").apply(lambda d: float(
        (d["discharge_mwh"] - d["charge_mwh"]).mul(d["settlement_price"]).sum()
        - spec.wear_eur_per_mwh * d["discharge_mwh"].sum()
    ), include_groups=False)
    daily_throughput = s.groupby("day")["discharge_mwh"].sum()
    daily_per_cycle = daily / (daily_throughput / spec.energy_mwh).replace(0, np.nan)
    for d, val in daily_per_cycle.dropna().items():
        cycle_rows.append({"strategy": name, "day": d, "rev_per_cycle_eur": float(val)})
cycle_df = pd.DataFrame(cycle_rows)

fig, ax = plt.subplots(figsize=(11, 4.5))
for strat in strategies:
    vals = cycle_df[cycle_df["strategy"] == strat]["rev_per_cycle_eur"].clip(-2000, 8000)
    ax.hist(vals, bins=80, color=colors[strat], alpha=0.45, label=strat)
ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("Daily revenue per full cycle  (€/cycle, clipped to ±€8 k)")
ax.set_title("Q3: distribution of daily revenue-per-cycle across strategies")
ax.legend()
_save(fig, "q3_revenue_per_cycle_distribution")

# ---------------------------------------------------------------------------
# Persist consolidated summary
# ---------------------------------------------------------------------------

with open(OUT / "summary.json", "w") as f:
    json.dump({
        "spec": {
            "power_mw": spec.power_mw, "energy_mwh": spec.energy_mwh,
            "eta_round_trip": spec.eta_round_trip,
            "wear_eur_per_mwh": spec.wear_eur_per_mwh,
            "max_cycles_per_day": spec.max_cycles_per_day,
        },
        "window": {"start": str(common[0].date()), "end": str(common[-1].date()), "years": round(n_years, 2)},
        "headline": summary.to_dict(orient="index"),
        "uplift_vs_naive": attribution.to_dict(orient="index"),
    }, f, indent=2, default=str)

print(f"\nWrote {OUT.relative_to(ROOT)}/summary.json + figures + tables.")
