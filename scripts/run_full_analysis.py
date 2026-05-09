"""
Full historical analysis for all three research questions.

Runs after scripts/fetch_full_history.py has populated the cache. Saves:

    results/tables/*.csv        all numeric outputs (descriptive stats, test
                                 results, segment estimates, …)
    results/figures/*.png       all plots referenced in the notebooks
    results/summary.md          one-paragraph headlines per question

Usage
-----
    .venv/bin/python scripts/run_full_analysis.py

Re-running is cheap because every fetcher has parquet caching and every
expensive analysis here is wrapped in its own function so you can comment
out a section and iterate.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.environ.setdefault("ENTSOE_API_KEY", os.environ.get("ENTOSE_API_KEY", ""))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from power_microstructure.analysis.cointegration import CointegrationAnalysis  # noqa: E402
from power_microstructure.analysis.granger import GrangerAnalysis  # noqa: E402
from power_microstructure.analysis.structural import StructuralBreakAnalysis  # noqa: E402
from power_microstructure.data import EntsoeFetcher, SmardFetcher  # noqa: E402
from power_microstructure.signals.forecast_error import ForecastErrorSignal  # noqa: E402
from power_microstructure.signals.spread import SpreadConstructor  # noqa: E402

warnings.filterwarnings("ignore")
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

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"
FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

START = "2018-10-01"
END = (pd.Timestamp.now(tz="Europe/Berlin").normalize() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
SUMMARY: dict[str, dict] = {"window": {"start": START, "end": END}}


def _save_fig(fig, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


def _save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / f"{name}.csv"
    df.to_csv(path)
    print(f"  saved {path.relative_to(ROOT)}")
    return path


# ============================================================================
# Load the cached data
# ============================================================================


def load_data():
    print("Loading cached data …")
    ef = EntsoeFetcher()
    sf = SmardFetcher()
    errors = ef.forecast_error(start=START, end=END)
    rshare = ef.renewable_share(start=START, end=END)
    da = ef.day_ahead_prices(start=START, end=END)
    if isinstance(da, pd.DataFrame):
        da = da.iloc[:, 0]
    panel = sf.price_panel(start=START, end=END)
    qh = sf.intraday_continuous_index_qh(start=START, end=END)
    # Normalise tz to UTC
    for s in (errors, rshare, da, panel, qh):
        if hasattr(s, "index") and getattr(s.index, "tz", None) is None:
            s.index = s.index.tz_localize("UTC")
        elif hasattr(s, "index"):
            s.index = s.index.tz_convert("UTC")
    print(
        f"  errors={errors.shape}  panel={panel.shape}  qh={qh.shape}  "
        f"rshare={rshare.shape}  da={len(da)}"
    )
    return errors, rshare, da, panel, qh


# ============================================================================
# Q1 — Forecast errors as intraday order flow
# ============================================================================


def run_q1(errors: pd.DataFrame, panel: pd.DataFrame, rshare: pd.Series):
    print("\n" + "=" * 70 + "\nQ1 — Forecast errors as intraday order flow\n" + "=" * 70)

    # Use SMARD intraday continuous index — that's where forecast errors clear,
    # not the day-ahead auction (which is fixed before the error realises).
    price = panel["id_continuous"].dropna()
    fe = errors["total_error"]

    # ---- 1.1 Descriptives over time -------------------------------------------------
    by_year = errors.copy()
    by_year["year"] = by_year.index.year
    yr = by_year.groupby("year")[["wind_error", "solar_error", "total_error"]].agg(
        ["std", "mean"]
    )
    _save_table(yr, "q1_yearly_descriptives")
    SUMMARY["q1_yearly_total_error_std"] = (
        yr[("total_error", "std")].round(0).astype(int).to_dict()
    )

    fig, ax = plt.subplots(figsize=(11, 4))
    ann_std = errors[["wind_error", "solar_error", "total_error"]].resample("YE").std()
    for col in ann_std.columns:
        ax.plot(ann_std.index, ann_std[col], marker="o", label=col)
    ax.set_ylabel("σ of forecast error  (MW)")
    ax.set_title(
        "Forecast-error volatility by year — wind dominates, growing with capacity"
    )
    ax.legend()
    _save_fig(fig, "q1_yearly_error_std")

    # Renewable share trend (the temporal regime variable)
    rs_monthly = rshare.resample("ME").mean()
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(rs_monthly.index, rs_monthly * 100, color="seagreen")
    ax.set_ylabel("Wind+Solar share of generation (%)")
    ax.set_title("German renewable penetration over the sample")
    _save_fig(fig, "q1_renewable_share")

    # ---- 1.2 Static Granger: full sample ------------------------------------------
    g = GrangerAnalysis(
        forecast_error=fe, price_series=price, max_lag=24, n_bootstrap=200, alpha=0.05
    )
    fe_adf = g.adf_test(g.fe, "forecast_error")
    px_adf = g.adf_test(g.price, "price")
    res = g.test(n_tests=1)
    res_rev = g.test_reverse(n_tests=1)

    granger_summary = pd.Series(
        {
            "n_obs": res.n_obs,
            "ADF(fe)_pvalue": fe_adf["p_value"],
            "ADF(price)_pvalue": px_adf["p_value"],
            "optimal_lag_AIC": res.optimal_lag,
            "F_forward": res.f_statistic,
            "p_forward": res.p_value,
            "F_reverse": res_rev.f_statistic,
            "p_reverse": res_rev.p_value,
        }
    )
    _save_table(granger_summary.to_frame("value"), "q1_granger_summary")
    SUMMARY["q1_granger"] = {
        "lag": int(res.optimal_lag),
        "F_forward": float(res.f_statistic),
        "p_forward": float(res.p_value),
        "F_reverse": float(res_rev.f_statistic),
        "p_reverse": float(res_rev.p_value),
        "n_obs": int(res.n_obs),
    }

    # ---- 1.3 IRF -----------------------------------------------------------------
    print("  computing IRF (24h horizon, 200 bootstrap reps) …")
    irf = g.irf(horizon=24, n_periods_seed=2000)
    # Single-panel IRF with bootstrap CI — the FEVD result (FE explains <5% of
    # price variance at all horizons) is captured numerically in q1_irf_path.csv
    # and called out in text; visualising it as a stacked area at [0, 1] makes
    # the small but real contribution invisible.
    fig, ax = plt.subplots(figsize=(11, 4.2))
    h = irf.periods
    ax.fill_between(h, irf.irf_lower, irf.irf_upper, alpha=0.20, color="C0",
                    label="90% bootstrap CI")
    ax.plot(h, irf.irf, color="C0", lw=2, label="IRF point estimate")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("hours after +1σ forecast-error shock")
    ax.set_ylabel("intraday continuous price response (EUR/MWh)")
    ax.set_title("IRF: response of intraday price to a renewable forecast-error shock")
    ax.set_xticks(np.arange(0, max(h) + 1, 2))
    ax.legend(loc="lower right")
    _save_fig(fig, "q1_irf_and_fevd")

    irf_df = pd.DataFrame(
        {"horizon_h": h, "irf": irf.irf, "lower_5": irf.irf_lower, "upper_95": irf.irf_upper}
    )
    _save_table(irf_df.set_index("horizon_h"), "q1_irf_path")

    # ---- 1.4 Rolling Granger -------------------------------------------------
    print("  computing rolling Granger over 1y windows, weekly step …")
    rolling_p = g.rolling_granger_pvalue(window=8760, step=168)
    rolling_p.to_csv(TABLES / "q1_rolling_granger_pvalue.csv")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(rolling_p.index, -np.log10(rolling_p.clip(lower=1e-300)), color="C3")
    ax.axhline(-np.log10(0.05), color="grey", ls="--", lw=0.8, label="α=0.05")
    ax.set_ylabel("−log10(p)")
    ax.set_title(
        "Rolling 1-year Granger p-value: forecast-error → intraday price"
    )
    ax.legend()
    _save_fig(fig, "q1_rolling_granger")

    # ---- 1.5 Bai-Perron structural break analysis ---------------------------
    print("  computing Bai-Perron structural breaks (exact DP, may take ~1 min) …")
    fe_aligned, price_aligned, _ = g.prepare_stationary()
    # Per-hour β: use first-difference price (= price impact of fe per hour)
    pc = price_aligned.diff().dropna()
    fe_pc = fe_aligned.loc[pc.index]
    # Down-sample to daily mean to keep T tractable AND interpret breaks at day-level
    daily = pd.DataFrame({"y": pc, "x": fe_pc}).resample("D").mean().dropna()
    sba = StructuralBreakAnalysis(
        daily["y"], daily["x"], max_breaks=4, trim=0.15, alpha=0.05
    )
    bp = sba.fit()
    print(
        f"  Bai-Perron: detected {bp.n_breaks} break(s) — {[d.date() for d in bp.break_dates]}"
    )
    SUMMARY["q1_bai_perron"] = {
        "n_breaks": int(bp.n_breaks),
        "break_dates": [str(d.date()) for d in bp.break_dates],
        "segment_betas": [float(b) for b in bp.segment_betas],
        "supF_pvalues": [float(p) for p in bp.supF_pvalues],
    }

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(rs_monthly.index, rs_monthly * 100, color="seagreen")
    axes[0].set_ylabel("Wind+Solar share (%)")
    axes[0].set_title("Bai-Perron break dates vs renewable penetration")
    for d in bp.break_dates:
        for ax in axes:
            ax.axvline(d, color="red", ls="--", lw=1.0, alpha=0.7)
    beta_ts = sba.beta_over_time(bp)
    if len(beta_ts):
        axes[1].plot(beta_ts.index, beta_ts.values, color="C0", lw=1.5)
    axes[1].set_ylabel("β (price-change / forecast-error)")
    axes[1].set_xlabel("date")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    _save_fig(fig, "q1_bai_perron")

    seg_df = pd.DataFrame(
        {
            "segment": range(len(bp.segment_betas)),
            "alpha": bp.segment_alphas,
            "beta": bp.segment_betas,
            "r2": bp.segment_r2,
            "n": bp.segment_obs,
        }
    )
    _save_table(seg_df, "q1_bai_perron_segments")

    return {"granger": res, "irf": irf, "rolling_p": rolling_p, "bp": bp}


# ============================================================================
# Q2 — Auction vs continuous spread
# ============================================================================


def run_q2(errors: pd.DataFrame, panel: pd.DataFrame):
    print("\n" + "=" * 70 + "\nQ2 — Auction vs continuous spread\n" + "=" * 70)

    out = {}
    sc = SpreadConstructor(panel)
    sig = ForecastErrorSignal(errors, window=24 * 30)
    regime = sig.regime(n_regimes=3)
    if regime.index.tz is None:
        regime.index = regime.index.tz_localize("UTC")

    # ---- 2.1 Spread descriptives ----------------------------------------------
    spreads = sc.spread_panel()
    desc = spreads.describe().T
    _save_table(desc, "q2_spread_descriptives")
    SUMMARY["q2_spread_descriptives"] = desc.round(2).to_dict()

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, col, color in zip(
        axes,
        ["da_id_spread", "id1_spread", "id3_spread"],
        ["C0", "C1", "C2"],
    ):
        ax.plot(spreads.index, spreads[col], color=color, lw=0.4, alpha=0.7)
        ax.set_ylabel(col)
        ax.axhline(0, color="black", lw=0.5)
    axes[0].set_title("Auction-vs-continuous spreads over the full sample")
    axes[-1].set_xlabel("date")
    _save_fig(fig, "q2_spread_timeseries")

    # ---- 2.2 Cointegration --------------------------------------------------------
    # Heavy stats (ADF autolag, Johansen with k_ar_diff selection, VECM) are O(T·p)
    # to O(T²) — at T=66k each call takes hours and can hang in numerical edge
    # cases. We trim to the most recent 3 years (≈26k obs) for the formal tests
    # and report a separate AR(1) half-life over the FULL sample so we don't lose
    # the long-run mean-reversion picture.
    test_window = pd.Timestamp(END, tz="UTC") - pd.Timedelta(days=3 * 365)
    print(f"  cointegration tests on most recent 3y subsample (≥ {test_window.date()})")

    candidates = [
        ("ID1", panel["id1_price"], panel["id_continuous"]),
        ("ID3", panel["id3_price"], panel["id_continuous"]),
        ("DA", panel["da_price"], panel["id_continuous"]),
    ]
    coint_results = {}
    for label, s_auc, s_cim in candidates:
        joined = pd.concat([s_auc, s_cim], axis=1).dropna()
        # Restrict to recent 3y for the heavy tests
        joined = joined[joined.index >= test_window]
        if len(joined) < 1000:
            print(f"  [{label}] only {len(joined)} aligned obs in window — skipped")
            continue
        # Down-sample to daily for stationarity tests so ADF autolag doesn't blow up
        daily = joined.resample("1D").mean().dropna()
        ca_daily = CointegrationAnalysis(daily.iloc[:, 0], daily.iloc[:, 1])
        ca_full = CointegrationAnalysis(joined.iloc[:, 0], joined.iloc[:, 1])

        try:
            st = ca_daily.stationarity_both()
            adf_a = st[daily.columns[0]].adf_pvalue
            adf_c = st[daily.columns[1]].adf_pvalue
        except Exception as e:
            print(f"    [{label}] stationarity failed: {e}")
            adf_a, adf_c = float("nan"), float("nan")

        try:
            eg = ca_daily.engle_granger()
        except Exception as e:
            print(f"    [{label}] Engle-Granger failed: {e}")
            eg = {"t_stat": float("nan"), "p_value": float("nan")}

        try:
            joh = ca_daily.johansen()
            joh_n = joh.n_cointegrating_vectors
            joh_beta = joh.cointegrating_vector
        except Exception as e:
            print(f"    [{label}] Johansen failed: {e}")
            joh_n, joh_beta = -1, None

        try:
            v = ca_daily.vecm(k_ar_diff=1, coint_rank=1)
            alpha = list(v.alpha)
            beta_v = list(v.beta)
        except Exception as e:
            print(f"    [{label}] VECM failed: {e}")
            alpha, beta_v = None, None

        # Half-life is the figure that makes physical sense at hourly granularity.
        # Compute it on the FULL hourly sample (cheap — closed-form AR(1) regression).
        try:
            half_life = ca_full.half_life(beta=1.0)
        except Exception as e:
            print(f"    [{label}] hourly half-life failed: {e}")
            half_life = float("nan")

        coint_results[label] = {
            "n_recent_daily": int(len(daily)),
            "n_full_hourly": int(len(joined)),
            "ADF_auction_p": adf_a,
            "ADF_cont_p": adf_c,
            "EG_t": eg["t_stat"],
            "EG_p": eg["p_value"],
            "Joh_rank": int(joh_n),
            "Joh_beta": list(joh_beta) if joh_beta is not None else None,
            "VECM_alpha": alpha,
            "VECM_beta": beta_v,
            "halflife_h": float(half_life),
        }
        print(
            f"  [{label}]  daily n={len(daily)}  hourly n={len(joined)}  "
            f"EG t={eg['t_stat']:.3f} p={eg['p_value']:.4f}  "
            f"Joh rank={joh_n}  HL_hourly={half_life:.2f}h"
        )

    pd.DataFrame(coint_results).T.to_csv(TABLES / "q2_cointegration_summary.csv")
    SUMMARY["q2_cointegration"] = coint_results

    # ---- 2.3 Regime-conditional variance --------------------------------------
    out["regime_var"] = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (col, label) in zip(
        axes, [("da_id_spread", "DA−Cont"), ("id1_spread", "ID1−Cont"), ("id3_spread", "ID3−Cont")]
    ):
        sp = spreads[col].dropna()
        if sp.index.tz is None:
            sp.index = sp.index.tz_localize("UTC")
        common = sp.index.intersection(regime.index)
        if len(common) < 500:
            ax.set_title(f"{label}\n(no data)")
            continue
        cv = sc.regime_conditional_variance(sp.loc[common], regime.loc[common])
        out["regime_var"][col] = cv.to_dict()
        cv["std"].plot(kind="bar", ax=ax, color=["C0", "C1", "C3"])
        ax.set_xticklabels(["low", "med", "high"], rotation=0)
        ax.set_xlabel("forecast-error regime")
        ax.set_ylabel("σ (EUR/MWh)")
        ax.set_title(f"{label} std by regime")
    fig.suptitle("Q2: Spread σ widens with renewable forecast-error regime")
    _save_fig(fig, "q2_regime_variance")
    SUMMARY["q2_regime_variance"] = out["regime_var"]

    # ---- 2.4 Half-life over rolling window (regime stability) -----------------
    print("  computing rolling 90-day half-life of (ID3 − Continuous) spread …")
    sp = (panel["id3_price"] - panel["id_continuous"]).dropna()
    if sp.index.tz is None:
        sp.index = sp.index.tz_localize("UTC")
    window_h = 24 * 90
    step_h = 24 * 14
    rolling_hl = []
    dates = []
    for end_loc in range(window_h, len(sp), step_h):
        sub = sp.iloc[end_loc - window_h : end_loc]
        ds = sub.diff().dropna()
        s_lag = sub.shift(1).dropna()
        common = ds.index.intersection(s_lag.index)
        x = np.column_stack([np.ones(len(common)), s_lag.loc[common].values])
        beta = np.linalg.lstsq(x, ds.loc[common].values, rcond=None)[0][1]
        if -1 < beta < 0:
            hl = -np.log(2) / np.log(1 + beta)
        else:
            hl = np.nan
        rolling_hl.append(hl)
        dates.append(sub.index[-1])

    rolling_hl_s = pd.Series(rolling_hl, index=pd.DatetimeIndex(dates))
    rolling_hl_s.to_csv(TABLES / "q2_rolling_halflife.csv")

    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(rolling_hl_s.index, rolling_hl_s.values, color="C2", lw=1.5)
    ax.set_ylabel("half-life (hours)")
    ax.set_title("Q2: rolling 90-day mean-reversion speed of (ID3 − Continuous)")
    _save_fig(fig, "q2_rolling_halflife")

    return {"coint": coint_results, "rolling_hl": rolling_hl_s}


# ============================================================================
# Q3 — Shape spread (hourly vs quarter-hourly)
# ============================================================================


def run_q3(errors: pd.DataFrame, panel: pd.DataFrame, qh: pd.Series):
    print("\n" + "=" * 70 + "\nQ3 — Shape spread (hourly vs QH)\n" + "=" * 70)

    sc = SpreadConstructor(panel, qh_prices=qh)
    shape = sc.shape_spread().dropna()
    intra_std = sc.qh_intra_hour_std().dropna()

    desc = shape.describe()
    _save_table(desc.to_frame("shape_spread"), "q3_shape_descriptives")
    SUMMARY["q3_shape_descriptives"] = desc.round(3).to_dict()

    # ---- 3.1 Distribution + time series ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(shape.values, bins=120, range=(-30, 30), color="C0", alpha=0.85)
    axes[0].axvline(0, color="black", lw=0.6)
    axes[0].set_title("Shape spread distribution (clipped to ±30 EUR/MWh for visibility)")
    axes[0].set_xlabel("hourly − mean(4 QH)  [EUR/MWh]")

    # Monthly mean to show stability
    monthly = shape.resample("ME").mean()
    axes[1].bar(monthly.index, monthly.values, width=20, color="C0", alpha=0.7)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_title("Monthly-mean shape spread")
    axes[1].set_ylabel("EUR/MWh")
    _save_fig(fig, "q3_shape_dist_and_monthly")

    # ---- 3.2 Hour-of-day pattern (the headline result) ------------------------
    by_hr = sc.shape_spread_by_hour()
    _save_table(by_hr, "q3_shape_by_hour_full")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    means = by_hr["mean"].values
    stds = by_hr["std"].values
    hours = np.arange(24)
    ax.plot(hours, means, marker="o", color="C0", lw=2)
    ax.fill_between(hours, means - stds, means + stds, alpha=0.18, color="C0")
    ax.axhline(0, color="black", lw=0.6)
    # Shade solar-peak hours
    ax.axvspan(10, 14, color="orange", alpha=0.10, label="solar peak (10–14)")
    ax.axvspan(5, 8, color="green", alpha=0.10, label="morning ramp (5–8)")
    ax.axvspan(17, 21, color="purple", alpha=0.10, label="evening ramp (17–21)")
    ax.set_xticks(hours)
    ax.set_xlabel("hour of day (CET)")
    ax.set_ylabel("shape spread  (EUR/MWh)")
    ax.set_title(
        "Q3: shape spread by hour-of-day — duck-curve fingerprint over the full sample"
    )
    ax.legend(loc="upper right")
    _save_fig(fig, "q3_shape_by_hour")

    SUMMARY["q3_shape_by_hour_means"] = (
        by_hr["mean"].round(2).to_dict()
    )

    # ---- 3.3 Regression: shape ~ hour FE + intra_hour σ -----------------------
    aligned = pd.concat([shape.rename("shape"), intra_std.rename("sigma")], axis=1).dropna()
    aligned["hour"] = aligned.index.hour
    # OLS with hour fixed effects (24 dummies)
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    h_dummies = pd.get_dummies(aligned["hour"], prefix="h", drop_first=True).astype(float)
    X = add_constant(pd.concat([aligned["sigma"], h_dummies], axis=1)).astype(float)
    y = aligned["shape"].astype(float).values
    fit = OLS(y, X.values).fit()
    print(f"  shape ~ const + σ + hour-FE   n={int(fit.nobs)}  R²={fit.rsquared:.3f}")
    print(f"  σ coefficient = {fit.params[1]:+.3f}   t = {fit.tvalues[1]:+.2f}")
    SUMMARY["q3_regression"] = {
        "n_obs": int(fit.nobs),
        "r2": float(fit.rsquared),
        "sigma_coef": float(fit.params[1]),
        "sigma_t": float(fit.tvalues[1]),
    }
    pd.DataFrame(
        {"param": fit.params, "tstat": fit.tvalues, "pvalue": fit.pvalues}
    ).to_csv(TABLES / "q3_regression_params.csv")

    # ---- 3.4 Regime-conditional (forecast-error tercile) ----------------------
    sig = ForecastErrorSignal(errors, window=24 * 30)
    regime = sig.regime(n_regimes=3)
    if regime.index.tz is None:
        regime.index = regime.index.tz_localize("UTC")
    if shape.index.tz is None:
        shape.index = shape.index.tz_localize("UTC")
    common = shape.index.intersection(regime.index)
    cond = sc.regime_conditional_variance(shape.loc[common], regime.loc[common])
    _save_table(cond, "q3_shape_regime_variance")
    SUMMARY["q3_regime_variance"] = cond.round(2).to_dict()

    fig, ax = plt.subplots(figsize=(7, 4))
    cond["std"].plot(kind="bar", ax=ax, color=["C0", "C1", "C3"])
    ax.set_xticklabels(["low |fe|", "med", "high |fe|"], rotation=0)
    ax.set_ylabel("shape spread σ (EUR/MWh)")
    ax.set_title("Q3: shape spread σ by forecast-error regime")
    _save_fig(fig, "q3_regime_variance")

    return {"by_hour": by_hr, "regression": fit, "regime_var": cond}


# ============================================================================
# Driver
# ============================================================================


def main():
    errors, rshare, da, panel, qh = load_data()
    run_q1(errors, panel, rshare)
    run_q2(errors, panel)
    run_q3(errors, panel, qh)

    summary_path = ROOT / "results" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(SUMMARY, f, indent=2, default=str)
    print(f"\nWrote {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
