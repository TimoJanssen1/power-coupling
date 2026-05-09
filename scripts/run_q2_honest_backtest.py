"""
Q2 honest backtest — auction-commitment-time decision, hold to delivery.

The previous version (`run_q2_backtest.py`) treated `ID3 - continuous_VWAP`
as a tradeable hourly time series, which it is *not*: ID3 clears at 22:00
day t-1 (single auction price, no exits) and continuous_VWAP is a
post-delivery settlement statistic. The "4.2-hour mean hold" the inflated
backtest reports is fictitious.

This version simulates what a real desk could actually do:

  At 22:00 day t-1, observe:
      ID3[t, h]  for all 24 delivery hours of day t (just-cleared auction)
      Trailing-90-day mean of (ID3 - continuous_VWAP) per delivery hour h
      from prior delivery days (only)

  Trade rule (one trade per delivery day):
      For each delivery hour h, signal = trailing_mean_spread[h] (lagged).
      Pick h* = argmax_h |signal[h]|
      if |signal[h*]| > ENTRY_THRESHOLD (€/MWh):
          direction = sign(signal[h*])  # ride the trailing bias
          commit at the auction (no exit possible)
          hold through to physical delivery (~24 hours wall clock)
          PnL = direction * (ID3[t, h*] - continuous_VWAP[t, h*])
          costs apply on the continuous leg only (auction has no slippage)

The signal uses ONLY past observations of the trade target, with strict
lag (yesterday and earlier). No leg of the trade is observed before the
decision. This is the cleanest, most defensible specification.

Sharpe annualised by sqrt(252) on daily PnL.
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

from power_microstructure.data import SmardFetcher  # noqa: E402

OUT = ROOT / "results" / "q2_backtest_honest"
OUT.mkdir(parents=True, exist_ok=True)
(FIG := OUT / "figures").mkdir(exist_ok=True)
(TAB := OUT / "tables").mkdir(exist_ok=True)

START = "2018-10-01"
END = (pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

# ---- Pre-registered spec ----------------------------------------------------
ENTRY_THRESHOLD = 5.0     # |trailing_mean_spread| in EUR/MWh required to enter
TRAINING_DAYS = 90        # rolling window of past delivery days for the signal
N_BOOTSTRAP = 2000

COST_SCENARIOS = {
    "optimistic":  {"continuous_round_trip_eur_per_mwh": 0.06},   # exchange fees only
    "realistic":   {"continuous_round_trip_eur_per_mwh": 6.0},    # bid-ask + slippage on continuous leg
    "pessimistic": {"continuous_round_trip_eur_per_mwh": 12.0},   # retail-sized in thin orderbook
}

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
# Load data
# ---------------------------------------------------------------------------

print(f"Sample: {START} → {END}")
sf = SmardFetcher()
panel = sf.price_panel(start=START, end=END)
panel.index = pd.to_datetime(panel.index, utc=True)
df = panel[["da_price", "id3_price", "id_continuous"]].dropna().copy()
df["delivery_day"] = df.index.tz_convert("Europe/Berlin").date
df["delivery_hour"] = df.index.tz_convert("Europe/Berlin").hour
df["realised_spread"] = df["id3_price"] - df["id_continuous"]   # observable post-delivery
print(f"  hourly grid: n={len(df):,}  delivery-days={df['delivery_day'].nunique():,}")

# ---------------------------------------------------------------------------
# Per delivery hour, compute strictly-lagged trailing mean of realised spread.
# This is observable at auction time (only uses days strictly before t).
# ---------------------------------------------------------------------------


def _trailing_signal(group: pd.DataFrame, training_days: int) -> pd.DataFrame:
    g = group.sort_values("delivery_day").reset_index(drop=True)
    s = g["realised_spread"].astype(float)
    g["signal"] = s.shift(1).rolling(training_days, min_periods=30).mean()
    g["signal_std"] = s.shift(1).rolling(training_days, min_periods=30).std()
    return g


# Reduce to one row per (delivery_day, delivery_hour)
df_one = (
    df.reset_index()
      .drop_duplicates(subset=["delivery_day", "delivery_hour"], keep="first")
      .sort_values(["delivery_hour", "delivery_day"])
)

sig_blocks = []
for hour, grp in df_one.groupby("delivery_hour"):
    sig_blocks.append(_trailing_signal(grp, TRAINING_DAYS))
sig_panel = pd.concat(sig_blocks).sort_values(["delivery_day", "delivery_hour"]).reset_index(drop=True)

# ---------------------------------------------------------------------------
# One trade per delivery day — pick the hour with strongest |signal|
# ---------------------------------------------------------------------------

sig_panel["abs_signal"] = sig_panel["signal"].abs()


def _pick_top_per_day(g):
    g = g.dropna(subset=["abs_signal"])
    if g.empty:
        return None
    return g.loc[g["abs_signal"].idxmax()]


picks = (
    sig_panel.dropna(subset=["abs_signal"])
            .groupby("delivery_day", as_index=False, group_keys=False)
            .apply(_pick_top_per_day, include_groups=False)
            .dropna(subset=["abs_signal"])
            .reset_index(drop=True)
)
picks["direction"] = np.sign(picks["signal"])  # ride the trailing bias
picks = picks[picks["abs_signal"] > ENTRY_THRESHOLD].reset_index(drop=True)
print(f"  candidate trades (|trailing_mean| > €{ENTRY_THRESHOLD}/MWh): {len(picks):,}  "
      f"({len(picks) / max(sig_panel['delivery_day'].nunique(), 1):.1%} of delivery days)")

# Gross PnL per trade = direction * (ID3 - continuous_VWAP) for that delivery hour
picks["gross_pnl_eur_per_mwh"] = picks["direction"] * picks["realised_spread"]
picks["delivery_day_ts"] = pd.to_datetime(picks["delivery_day"]).dt.tz_localize("UTC")

# ---------------------------------------------------------------------------
# Apply three cost scenarios
# ---------------------------------------------------------------------------


def _summarise(pnl_daily: pd.Series, name: str, n_bootstrap: int):
    pnl_daily = pnl_daily.dropna()
    if len(pnl_daily) == 0 or pnl_daily.std() < 1e-12:
        return {
            "scenario": name,
            "sharpe_annualised": 0.0,
            "sortino": 0.0,
            "max_drawdown_eur": 0.0,
            "win_rate": 0.0,
            "n_trades": 0,
            "total_pnl_eur_per_mwh": 0.0,
            "avg_pnl_per_trade_eur_per_mwh": 0.0,
            "bootstrap_p_sharpe_pos": 1.0,
        }, np.zeros(n_bootstrap)

    sharpe = float((pnl_daily.mean() / pnl_daily.std()) * np.sqrt(252))
    downside = pnl_daily[pnl_daily < 0]
    sortino = (
        float((pnl_daily.mean() / downside.std()) * np.sqrt(252))
        if len(downside) and downside.std() > 1e-12
        else float("inf")
    )
    cum = pnl_daily.cumsum()
    roll_max = cum.cummax()
    dd = (cum - roll_max).min()
    win_rate = float((pnl_daily > 0).mean())

    # Block bootstrap p-value on Sharpe (block length sqrt(T))
    arr = pnl_daily.values
    T = len(arr)
    block_len = max(1, int(np.sqrt(T)))
    rng = np.random.default_rng(20260507)
    boot_sharpes = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        n_blocks = T // block_len + 1
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        bs = np.concatenate([arr[s : s + block_len] for s in starts])[:T]
        s = bs.std()
        boot_sharpes[b] = (bs.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0
    p_value = float(np.mean(boot_sharpes <= 0))

    return {
        "scenario": name,
        "sharpe_annualised": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_eur": round(float(dd), 1),
        "win_rate": round(win_rate, 3),
        "n_trades": int((pnl_daily != 0).sum()),
        "total_pnl_eur_per_mwh": round(float(pnl_daily.sum()), 1),
        "avg_pnl_per_trade_eur_per_mwh": round(
            float(pnl_daily[pnl_daily != 0].mean()) if (pnl_daily != 0).any() else 0.0, 3
        ),
        "bootstrap_p_sharpe_pos": round(p_value, 4),
    }, boot_sharpes


# Index PnL by delivery day for daily aggregation (tz-aware UTC midnight)
all_days = sorted(pd.to_datetime(sig_panel["delivery_day"].unique()))
day_index = pd.DatetimeIndex(all_days).tz_localize("UTC")

results = {}
boot_dists = {}
for name, params in COST_SCENARIOS.items():
    cost = float(params["continuous_round_trip_eur_per_mwh"])
    net_pnl_per_trade = picks["gross_pnl_eur_per_mwh"] - cost
    pnl_daily = (
        pd.Series(net_pnl_per_trade.values, index=picks["delivery_day_ts"])
        .reindex(day_index, fill_value=0.0)
    )
    summary, boots = _summarise(pnl_daily, name, n_bootstrap=N_BOOTSTRAP)
    results[name] = {"summary": summary, "pnl": pnl_daily}
    boot_dists[name] = boots

summary_df = pd.DataFrame([r["summary"] for r in results.values()]).set_index("scenario")
summary_df.to_csv(TAB / "summary.csv")
print("\n=== Honest backtest — summary ===")
print(summary_df.to_string())

# ---------------------------------------------------------------------------
# Cumulative PnL chart
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4.5))
colors = {"optimistic": "C2", "realistic": "C0", "pessimistic": "C3"}
for name, r in results.items():
    cum = r["pnl"].cumsum()
    ax.plot(cum.index, cum.values, color=colors[name], lw=1.6,
            label=f"{name}  (Sharpe {r['summary']['sharpe_annualised']:.2f}, "
                  f"p={r['summary']['bootstrap_p_sharpe_pos']:.3f})")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Cumulative PnL (EUR/MWh, unit position)")
ax.set_title(f"Q2 honest backtest — one trade/delivery day, hold-to-delivery, |signal|>€{ENTRY_THRESHOLD}/MWh")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
_save(fig, "q2_honest_cumulative_pnl")

# ---------------------------------------------------------------------------
# Yearly PnL attribution
# ---------------------------------------------------------------------------

annual_rows = []
for name, r in results.items():
    by_year = r["pnl"].groupby(r["pnl"].index.year).sum()
    for y, v in by_year.items():
        annual_rows.append({"scenario": name, "year": int(y), "pnl_eur_per_mwh": round(float(v), 1)})
annual = pd.DataFrame(annual_rows)
annual.to_csv(TAB / "pnl_by_year.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4))
years = sorted(annual["year"].unique())
x = np.arange(len(years))
width = 0.27
for i, scen in enumerate(("optimistic", "realistic", "pessimistic")):
    sub = annual[annual["scenario"] == scen].set_index("year")
    vals = [float(sub.loc[y, "pnl_eur_per_mwh"]) if y in sub.index else 0.0 for y in years]
    ax.bar(x + (i - 1) * width, vals, width=width, color=colors[scen], label=scen, alpha=0.85)
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(years)
ax.set_ylabel("Annual PnL (EUR/MWh, unit position)")
ax.set_title("Q2 honest backtest — PnL attribution by year")
ax.legend()
_save(fig, "q2_honest_pnl_by_year")

# ---------------------------------------------------------------------------
# Bootstrap distribution (realistic scenario)
# ---------------------------------------------------------------------------

real_summary = results["realistic"]["summary"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(boot_dists["realistic"], bins=50, color="C0", alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, label="Sharpe = 0")
ax.axvline(real_summary["sharpe_annualised"], color="C3", lw=2,
           label=f"Realised Sharpe = {real_summary['sharpe_annualised']:.2f}")
ax.set_xlabel("Bootstrap Sharpe (annualised)")
ax.set_ylabel("Frequency")
ax.set_title(f"Bootstrap Sharpe — honest backtest, realistic costs  "
             f"(p={real_summary['bootstrap_p_sharpe_pos']:.3f})")
ax.legend()
_save(fig, "q2_honest_bootstrap_sharpe")

# ---------------------------------------------------------------------------
# Persist consolidated summary
# ---------------------------------------------------------------------------

with open(OUT / "summary.json", "w") as f:
    json.dump({
        "spec": {
            "ENTRY_THRESHOLD_eur_per_mwh": ENTRY_THRESHOLD,
            "TRAINING_DAYS": TRAINING_DAYS,
            "N_BOOTSTRAP": N_BOOTSTRAP,
            "signal": "trailing mean of realised (ID3 - continuous_VWAP) per delivery hour",
            "decision_horizon": "auction commitment (22:00 day t-1)",
            "holding": "to physical delivery, ~24h wall clock",
            "execution_costs_apply_to": "continuous leg only",
        },
        "data": {
            "n_delivery_days": int(sig_panel["delivery_day"].nunique()),
            "n_candidate_trades": int(len(picks)),
        },
        "scenarios": {name: r["summary"] for name, r in results.items()},
    }, f, indent=2, default=str)

print(f"\nWrote {OUT.relative_to(ROOT)}/")
