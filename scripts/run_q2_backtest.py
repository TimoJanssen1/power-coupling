"""
Q2 walk-forward mean-reversion backtest.

Loads the cached SMARD price panel, constructs the (ID3 − continuous) spread,
and runs the pre-registered strategy from ``q2_strategy_spec.py`` under three
cost scenarios. Outputs PnL series, trade tables, fold diagnostics and a
combined summary to ``results/q2_backtest/``.

Usage:
    .venv/bin/python scripts/run_q2_backtest.py
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
from power_microstructure.strategy.backtest import WalkForwardBacktest  # noqa: E402
from scripts.q2_strategy_spec import (  # noqa: E402
    COST_SCENARIOS,
    ENTRY_Z,
    EXIT_Z,
    MAX_HOLD,
    N_BOOTSTRAP,
    SEED,
    STOP_LOSS_Z,
    TEST_MONTHS,
    TRAIN_MONTHS,
)

OUT = ROOT / "results" / "q2_backtest"
OUT.mkdir(parents=True, exist_ok=True)
(FIG := OUT / "figures").mkdir(exist_ok=True)
(TAB := OUT / "tables").mkdir(exist_ok=True)

START = "2018-10-01"
END = (pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 140,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def _save_fig(fig, name):
    p = FIG / f"{name}.png"
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


# -------------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------------

print(f"Sample window: {START} → {END}")
print("Loading cached SMARD price panel …")
sf = SmardFetcher()
panel = sf.price_panel(start=START, end=END)
panel.index = pd.to_datetime(panel.index, utc=True)
print(f"  panel shape={panel.shape}")

spread = (panel["id3_price"] - panel["id_continuous"]).dropna()
spread.name = "id3_minus_continuous"
print(
    f"  ID3 − Continuous: n={len(spread):,}  "
    f"mean={float(spread.mean()):+.2f}  std={float(spread.std()):.2f} EUR/MWh"
)

# -------------------------------------------------------------------------
# Run three cost scenarios on the same pre-registered strategy
# -------------------------------------------------------------------------

results = {}
for scenario_name, costs in COST_SCENARIOS.items():
    print(f"\n[{scenario_name}]  fee={costs['exchange_fee_eur_per_mwh_per_side']}  "
          f"impact_pct_of_sigma={costs['impact_pct_of_sigma']}")
    bt = WalkForwardBacktest(
        spread=spread,
        train_months=TRAIN_MONTHS,
        test_months=TEST_MONTHS,
        n_bootstrap=N_BOOTSTRAP,
        volume_mwh=1.0,
        exchange_fee=costs["exchange_fee_eur_per_mwh_per_side"],
        impact_pct_of_sigma=costs["impact_pct_of_sigma"],
    )
    res = bt.run(
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        max_hold=MAX_HOLD,
        stop_loss_z=STOP_LOSS_Z,
    )
    results[scenario_name] = res
    s = res.summary()
    print(s.to_string())
    res.trades.to_csv(TAB / f"trades_{scenario_name}.csv", index=False)

# -------------------------------------------------------------------------
# Cross-scenario summary table
# -------------------------------------------------------------------------

summary_rows = []
for scen, res in results.items():
    summary_rows.append({
        "scenario": scen,
        "sharpe_annualised": round(res.sharpe, 3),
        "sortino": round(res.sortino, 3),
        "max_drawdown_pct": round(100 * res.max_drawdown, 2),
        "win_rate_pct": round(100 * res.win_rate, 2),
        "n_trades": int(res.n_trades),
        "avg_hold_h": round(res.avg_hold_hours, 1),
        "total_pnl_eur_per_mwh": round(res.total_pnl, 2),
        "avg_pnl_per_trade_eur_per_mwh": round(res.avg_pnl_per_trade, 3),
        "bootstrap_p_sharpe_pos": round(res.bootstrap_pvalue, 4),
    })
summary = pd.DataFrame(summary_rows).set_index("scenario")
summary.to_csv(TAB / "summary.csv")
print("\n=== Cross-scenario summary ===")
print(summary.to_string())

# -------------------------------------------------------------------------
# Cumulative PnL plot
# -------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4.5))
colors = {"optimistic": "C2", "realistic": "C0", "pessimistic": "C3"}
for scen, res in results.items():
    ax.plot(
        res.cumulative_pnl.index,
        res.cumulative_pnl.values,
        color=colors[scen],
        lw=1.6,
        label=f"{scen}  (Sharpe {res.sharpe:.2f}, p={res.bootstrap_pvalue:.3f})",
    )
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Cumulative PnL (EUR/MWh, unit position)")
ax.set_title("Q2 walk-forward backtest — pre-registered spec, three cost scenarios")
ax.legend(loc="upper left")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
_save_fig(fig, "q2_cumulative_pnl")

# -------------------------------------------------------------------------
# Fold-by-fold consistency
# -------------------------------------------------------------------------

fold_rows = []
for scen, res in results.items():
    for (t_start, t_end), s in zip(res.fold_dates, res.fold_sharpes):
        fold_rows.append({
            "scenario": scen,
            "fold_start": str(pd.Timestamp(t_start).date()),
            "fold_end": str(pd.Timestamp(t_end).date()),
            "fold_sharpe": round(float(s), 3),
        })
fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(TAB / "fold_sharpes.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4.5))
for scen, res in results.items():
    fold_mids = [
        pd.Timestamp(t_start).normalize() + (pd.Timestamp(t_end) - pd.Timestamp(t_start)) / 2
        for t_start, t_end in res.fold_dates
    ]
    ax.plot(fold_mids, res.fold_sharpes, marker="o", color=colors[scen], lw=1.0,
            label=f"{scen}  (mean fold Sharpe {np.mean(res.fold_sharpes):.2f})")
ax.axhline(0, color="black", lw=0.5)
ax.set_ylabel("Fold Sharpe (annualised)")
ax.set_title("Q2 fold-by-fold Sharpe consistency")
ax.legend(loc="upper right")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
_save_fig(fig, "q2_fold_sharpes")

# -------------------------------------------------------------------------
# PnL attribution by year
# -------------------------------------------------------------------------

pnl_by_year_rows = []
for scen, res in results.items():
    yr = res.pnl.groupby(res.pnl.index.year).sum()
    for y, v in yr.items():
        pnl_by_year_rows.append({"scenario": scen, "year": int(y), "pnl_eur_per_mwh": round(float(v), 2)})
pnl_year_df = pd.DataFrame(pnl_by_year_rows)
pnl_year_df.to_csv(TAB / "pnl_by_year.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 4))
years = sorted(pnl_year_df["year"].unique())
x = np.arange(len(years))
width = 0.27
for i, scen in enumerate(("optimistic", "realistic", "pessimistic")):
    sub = pnl_year_df[pnl_year_df["scenario"] == scen].set_index("year")
    vals = [float(sub.loc[y, "pnl_eur_per_mwh"]) if y in sub.index else 0.0 for y in years]
    ax.bar(x + (i - 1) * width, vals, width=width, color=colors[scen], label=scen, alpha=0.85)
ax.axhline(0, color="black", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.set_ylabel("Annual PnL (EUR/MWh, unit position)")
ax.set_title("Q2 PnL attribution by year")
ax.legend()
_save_fig(fig, "q2_pnl_by_year")

# -------------------------------------------------------------------------
# Bootstrap distribution of Sharpe — realistic scenario
# -------------------------------------------------------------------------

real = results["realistic"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(real.bootstrap_sharpe_dist, bins=50, color="C0", alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, label="Sharpe = 0")
ax.axvline(real.sharpe, color="C3", lw=2, label=f"Realised Sharpe = {real.sharpe:.2f}")
ax.set_xlabel("Bootstrap Sharpe (annualised)")
ax.set_ylabel("Frequency")
ax.set_title(f"Block bootstrap distribution of Sharpe (realistic costs, p = {real.bootstrap_pvalue:.3f})")
ax.legend()
_save_fig(fig, "q2_bootstrap_sharpe_realistic")

# -------------------------------------------------------------------------
# Persist consolidated summary as JSON
# -------------------------------------------------------------------------

with open(OUT / "summary.json", "w") as f:
    json.dump(
        {
            "spec": {
                "ENTRY_Z": ENTRY_Z, "EXIT_Z": EXIT_Z,
                "MAX_HOLD": MAX_HOLD, "STOP_LOSS_Z": STOP_LOSS_Z,
                "TRAIN_MONTHS": TRAIN_MONTHS, "TEST_MONTHS": TEST_MONTHS,
                "N_BOOTSTRAP": N_BOOTSTRAP, "SEED": SEED,
            },
            "spread": {
                "name": "id3_minus_continuous",
                "n_obs": int(len(spread)),
                "mean": float(spread.mean()),
                "std": float(spread.std()),
                "start": str(spread.index[0].date()),
                "end": str(spread.index[-1].date()),
            },
            "scenarios": {
                scen: {
                    "sharpe": float(res.sharpe),
                    "sortino": float(res.sortino),
                    "max_drawdown": float(res.max_drawdown),
                    "win_rate": float(res.win_rate),
                    "n_trades": int(res.n_trades),
                    "avg_hold_h": float(res.avg_hold_hours),
                    "total_pnl": float(res.total_pnl),
                    "bootstrap_p_sharpe_pos": float(res.bootstrap_pvalue),
                    "fold_sharpes": [float(s) for s in res.fold_sharpes],
                }
                for scen, res in results.items()
            },
        },
        f,
        indent=2,
        default=str,
    )

print(f"\nWrote {OUT.relative_to(ROOT)}/summary.json and figures/tables.")
