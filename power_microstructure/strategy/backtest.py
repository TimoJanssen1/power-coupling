"""
Walk-forward backtest for intraday spread strategies.

Design principles
-----------------
- Walk-forward ONLY.  No look-ahead bias: signals, thresholds, and model parameters
  are estimated on the training window, then applied to the out-of-sample test window.
- Transaction costs are explicit:
    - EPEX exchange fee: ~€0.03/MWh (midpoint of published €0.02–€0.04 range)
    - Market impact: estimated as 0.1 × σ_bid_ask (conservative linear model)
    - Round-trip cost = 2 × (exchange_fee + impact)
- Negative prices are handled correctly (they are not outliers; a spread can be
  profitable or a loss in either direction).
- Performance metrics: Sharpe ratio, max drawdown, win rate, and bootstrap p-value
  on Sharpe to confirm it is not sampling luck.

Strategy logic (Q2 spread mean-reversion)
------------------------------------------
    Signal = z-score of spread over rolling training window
    Enter long spread when z < −threshold (spread abnormally low)
    Enter short spread when z > +threshold (spread abnormally high)
    Exit when |z| < exit_threshold or after max_holding periods
    Position size = 1 MWh (unit position; scale independently)

Usage
-----
>>> from power_microstructure.strategy import WalkForwardBacktest
>>> bt = WalkForwardBacktest(spread, train_months=12, test_months=3)
>>> result = bt.run(entry_z=2.0, exit_z=0.5, max_hold=24)
>>> print(result.sharpe, result.bootstrap_pvalue)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd


# EPEX exchange fee EUR/MWh (midpoint of 0.02–0.04 range per EPEX public tariff sheet)
_EXCHANGE_FEE = 0.03
_MARKET_IMPACT_FACTOR = 0.10  # 10% of rolling bid-ask proxy (spread std / sqrt(vol))


@dataclass
class BacktestResult:
    """Aggregated walk-forward backtest performance."""

    pnl: pd.Series              # hourly PnL series (EUR/MWh, unit position)
    cumulative_pnl: pd.Series
    trades: pd.DataFrame        # one row per closed trade

    # Summary statistics
    sharpe: float               # annualised Sharpe (252 * 24 hours)
    sortino: float
    max_drawdown: float         # peak-to-trough as fraction
    win_rate: float
    avg_hold_hours: float
    n_trades: int
    total_pnl: float
    avg_pnl_per_trade: float

    # Inference
    bootstrap_pvalue: float     # H0: Sharpe ≤ 0 (one-sided, percentile bootstrap)
    bootstrap_sharpe_dist: np.ndarray  # for plotting

    # Walk-forward fold results
    fold_sharpes: list[float] = field(default_factory=list)
    fold_dates: list[tuple] = field(default_factory=list)

    def summary(self) -> pd.Series:
        return pd.Series({
            "Sharpe (annualised)": round(self.sharpe, 3),
            "Sortino": round(self.sortino, 3),
            "Max Drawdown": f"{self.max_drawdown:.1%}",
            "Win Rate": f"{self.win_rate:.1%}",
            "N Trades": self.n_trades,
            "Total PnL (EUR/MWh)": round(self.total_pnl, 2),
            "Avg PnL / Trade": round(self.avg_pnl_per_trade, 4),
            "Avg Hold (hours)": round(self.avg_hold_hours, 1),
            "Bootstrap p-value (Sharpe>0)": round(self.bootstrap_pvalue, 3),
        })


class WalkForwardBacktest:
    """
    Walk-forward backtest engine for mean-reversion spread strategies.

    Parameters
    ----------
    spread : pd.Series
        Hourly spread series (EUR/MWh). Negative values are valid.
    train_months : int
        Length of each training window (months). Default 12.
    test_months : int
        Length of each test window (months). Default 3.
    n_bootstrap : int
        Number of bootstrap samples for Sharpe p-value. Default 1000.
    volume_mwh : float
        Trade size in MWh. Used for cost scaling only; PnL is expressed per MWh.
    """

    def __init__(
        self,
        spread: pd.Series,
        train_months: int = 12,
        test_months: int = 3,
        n_bootstrap: int = 1000,
        volume_mwh: float = 1.0,
        exchange_fee: float | None = None,
        impact_pct_of_sigma: float | None = None,
    ) -> None:
        self.spread = spread.dropna()
        self.train_months = train_months
        self.test_months = test_months
        self.n_bootstrap = n_bootstrap
        self.volume_mwh = volume_mwh
        # Configurable cost components. None ⇒ fall back to module defaults.
        self._fee = _EXCHANGE_FEE if exchange_fee is None else float(exchange_fee)
        self._impact = (
            _MARKET_IMPACT_FACTOR
            if impact_pct_of_sigma is None
            else float(impact_pct_of_sigma)
        )

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        max_hold: int = 48,
        stop_loss_z: float = 4.0,
    ) -> BacktestResult:
        """
        Execute walk-forward backtest.

        Parameters
        ----------
        entry_z : float
            Z-score threshold to open a position. |z| > entry_z → enter.
        exit_z : float
            Z-score threshold to close a position. |z| < exit_z → exit.
        max_hold : int
            Maximum number of hours to hold before forced exit.
        stop_loss_z : float
            Stop-loss: exit if z moves further against you beyond this level.
        """
        folds = self._build_folds()
        all_pnl: list[pd.Series] = []
        all_trades: list[pd.DataFrame] = []
        fold_sharpes: list[float] = []
        fold_dates: list[tuple] = []

        for train_slice, test_slice in folds:
            train_spread = self.spread.loc[train_slice]
            test_spread = self.spread.loc[test_slice]

            roll_mean = train_spread.mean()
            roll_std = train_spread.std()
            if roll_std < 1e-6:
                continue

            pnl_series, trades_df = self._simulate(
                test_spread=test_spread,
                roll_mean=roll_mean,
                roll_std=roll_std,
                entry_z=entry_z,
                exit_z=exit_z,
                max_hold=max_hold,
                stop_loss_z=stop_loss_z,
            )
            all_pnl.append(pnl_series)
            if not trades_df.empty:
                all_trades.append(trades_df)

            fold_sharpe = self._sharpe(pnl_series)
            fold_sharpes.append(fold_sharpe)
            fold_dates.append((test_spread.index[0], test_spread.index[-1]))

        if not all_pnl:
            raise RuntimeError("No folds produced any PnL — check data coverage.")

        pnl = pd.concat(all_pnl).sort_index()
        trades = pd.concat(all_trades).reset_index(drop=True) if all_trades else pd.DataFrame()

        cum_pnl = pnl.cumsum()
        sharpe = self._sharpe(pnl)
        sortino = self._sortino(pnl)
        max_dd = self._max_drawdown(cum_pnl)
        boot_pvalue, boot_dist = self._bootstrap_sharpe_pvalue(pnl)

        return BacktestResult(
            pnl=pnl,
            cumulative_pnl=cum_pnl,
            trades=trades,
            sharpe=sharpe,
            sortino=sortino,
            max_drawdown=max_dd,
            win_rate=float((trades["pnl"] > 0).mean()) if not trades.empty else 0.0,
            avg_hold_hours=float(trades["hold_hours"].mean()) if not trades.empty else 0.0,
            n_trades=len(trades),
            total_pnl=float(pnl.sum()),
            avg_pnl_per_trade=float(trades["pnl"].mean()) if not trades.empty else 0.0,
            bootstrap_pvalue=boot_pvalue,
            bootstrap_sharpe_dist=boot_dist,
            fold_sharpes=fold_sharpes,
            fold_dates=fold_dates,
        )

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _simulate(
        self,
        test_spread: pd.Series,
        roll_mean: float,
        roll_std: float,
        entry_z: float,
        exit_z: float,
        max_hold: int,
        stop_loss_z: float,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """
        Simulate trades on the test spread using training-window mean/std.

        Position conventions:
            +1 = long spread (expect spread to rise to mean)
            -1 = short spread (expect spread to fall to mean)
        """
        z_scores = (test_spread - roll_mean) / roll_std
        n = len(test_spread)
        prices = test_spread.values
        zs = z_scores.values
        idx = test_spread.index

        pnl_arr = np.zeros(n)
        trades: list[dict] = []

        position = 0       # 0 = flat, +1 = long, -1 = short
        entry_price = 0.0
        entry_idx = 0
        entry_time = None

        for t in range(n):
            if position == 0:
                # Check entry
                if zs[t] < -entry_z:
                    position = 1
                    entry_price = prices[t]
                    entry_idx = t
                    entry_time = idx[t]
                elif zs[t] > entry_z:
                    position = -1
                    entry_price = prices[t]
                    entry_idx = t
                    entry_time = idx[t]
            else:
                hold_bars = t - entry_idx
                z_now = zs[t]
                exit_condition = (
                    abs(z_now) < exit_z
                    or hold_bars >= max_hold
                    or (position == 1 and z_now < -stop_loss_z)
                    or (position == -1 and z_now > stop_loss_z)
                )
                if exit_condition:
                    exit_price = prices[t]
                    gross_pnl = position * (exit_price - entry_price)
                    cost = self._transaction_cost(roll_std)
                    net_pnl = gross_pnl - cost
                    pnl_arr[t] = net_pnl
                    trades.append({
                        "entry_time": entry_time,
                        "exit_time": idx[t],
                        "direction": position,
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_price),
                        "gross_pnl": float(gross_pnl),
                        "cost": float(cost),
                        "pnl": float(net_pnl),
                        "hold_hours": int(hold_bars),
                        "entry_z": float(zs[entry_idx]),
                        "exit_z": float(z_now),
                    })
                    position = 0
                    entry_price = 0.0

        # Force close any open position at end of window
        if position != 0:
            exit_price = prices[-1]
            gross_pnl = position * (exit_price - entry_price)
            cost = self._transaction_cost(roll_std)
            pnl_arr[-1] += gross_pnl - cost
            trades.append({
                "entry_time": entry_time,
                "exit_time": idx[-1],
                "direction": position,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_pnl": float(gross_pnl),
                "cost": float(cost),
                "pnl": float(gross_pnl - cost),
                "hold_hours": int(n - 1 - entry_idx),
                "entry_z": float(zs[entry_idx]),
                "exit_z": float(zs[-1]),
                "forced_exit": True,
            })

        pnl_series = pd.Series(pnl_arr, index=idx, name="pnl")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return pnl_series, trades_df

    # ------------------------------------------------------------------
    # Transaction costs
    # ------------------------------------------------------------------

    def _transaction_cost(self, spread_std: float) -> float:
        """
        Round-trip transaction cost per MWh.

        Components:
          - EPEX exchange fee: configurable, default €0.03/MWh × 2 (entry + exit)
          - Market impact: configurable fraction × spread_std as half-spread proxy
        """
        exchange = 2 * self._fee
        impact = 2 * self._impact * spread_std
        return (exchange + impact) * self.volume_mwh

    # ------------------------------------------------------------------
    # Walk-forward folds
    # ------------------------------------------------------------------

    def _build_folds(self) -> list[tuple[slice, slice]]:
        """
        Build (train_slice, test_slice) pairs for walk-forward validation.

        Training windows are anchored: each test window starts right after the
        previous one, but training always goes back to the beginning
        (expanding window).  Alternatively, rolling windows can be used by
        adjusting train_start below.
        """
        total_start = self.spread.index[0]
        total_end = self.spread.index[-1]
        train_delta = pd.DateOffset(months=self.train_months)
        test_delta = pd.DateOffset(months=self.test_months)

        folds = []
        test_start = total_start + train_delta
        while test_start < total_end:
            test_end = min(test_start + test_delta, total_end)
            train_slice = slice(total_start, test_start - pd.Timedelta(hours=1))
            test_slice = slice(test_start, test_end)
            if len(self.spread.loc[train_slice]) > 100 and len(self.spread.loc[test_slice]) > 10:
                folds.append((train_slice, test_slice))
            test_start = test_end + pd.Timedelta(hours=1)

        return folds

    # ------------------------------------------------------------------
    # Performance metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _sharpe(pnl: pd.Series, periods_per_year: int = 252) -> float:
        """
        Annualised Sharpe ratio.

        Aggregates hourly PnL to **daily** before computing mean/std, then
        annualises by √252. Naïvely annualising hourly PnL by √8760 inflates
        the Sharpe by ≈√(8760/252) ≈ 6× because most hours have zero PnL
        (the strategy is in-market a few hours per week, not continuously).
        """
        if isinstance(pnl.index, pd.DatetimeIndex):
            daily = pnl.resample("1D").sum()
        else:
            daily = pnl  # fallback: assume already daily-aggregated
        if daily.std() < 1e-12:
            return 0.0
        return float((daily.mean() / daily.std()) * np.sqrt(periods_per_year))

    @staticmethod
    def _sortino(pnl: pd.Series, periods_per_year: int = 252) -> float:
        """Annualised Sortino ratio (downside deviation denominator) on daily PnL."""
        if isinstance(pnl.index, pd.DatetimeIndex):
            daily = pnl.resample("1D").sum()
        else:
            daily = pnl
        downside = daily[daily < 0]
        if len(downside) == 0 or downside.std() < 1e-12:
            return float("inf")
        return float((daily.mean() / downside.std()) * np.sqrt(periods_per_year))

    @staticmethod
    def _max_drawdown(cumulative_pnl: pd.Series) -> float:
        """Maximum peak-to-trough drawdown as a fraction of peak cumulative PnL."""
        roll_max = cumulative_pnl.cummax()
        drawdown = (cumulative_pnl - roll_max)
        peak = roll_max.replace(0, np.nan)
        rel_dd = (drawdown / peak.abs()).replace([np.inf, -np.inf], np.nan).dropna()
        return float(rel_dd.min()) if len(rel_dd) > 0 else 0.0

    def _bootstrap_sharpe_pvalue(
        self, pnl: pd.Series
    ) -> tuple[float, np.ndarray]:
        """
        Bootstrap p-value for H0: Sharpe ≤ 0 (one-sided), on daily PnL.

        Block bootstrap preserves autocorrelation structure. Block length
        = √T_daily. Sharpes are annualised by √252.
        """
        if isinstance(pnl.index, pd.DatetimeIndex):
            daily = pnl.resample("1D").sum()
        else:
            daily = pnl
        arr = daily.values
        T = len(arr)
        block_len = max(1, int(np.sqrt(T)))
        rng = np.random.default_rng(42)

        boot_sharpes = np.empty(self.n_bootstrap)
        for b in range(self.n_bootstrap):
            n_blocks = T // block_len + 1
            starts = rng.integers(0, T - block_len + 1, size=n_blocks)
            boot_sample = np.concatenate([arr[s: s + block_len] for s in starts])[:T]
            s = boot_sample.std()
            boot_sharpes[b] = float(boot_sample.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0

        pvalue = float(np.mean(boot_sharpes <= 0))
        return pvalue, boot_sharpes

    # ------------------------------------------------------------------
    # Sensitivity analysis
    # ------------------------------------------------------------------

    def sensitivity(
        self,
        entry_z_range: list[float] | None = None,
        exit_z_range: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Grid search over entry/exit z-score thresholds.

        Returns DataFrame of Sharpe ratios — useful for showing robustness
        is not confined to a single parameter choice.
        """
        entry_z_range = entry_z_range or [1.0, 1.5, 2.0, 2.5, 3.0]
        exit_z_range = exit_z_range or [0.25, 0.5, 0.75, 1.0]

        rows = []
        for ez in entry_z_range:
            for xz in exit_z_range:
                if xz >= ez:
                    continue
                try:
                    result = self.run(entry_z=ez, exit_z=xz)
                    rows.append({
                        "entry_z": ez,
                        "exit_z": xz,
                        "sharpe": result.sharpe,
                        "n_trades": result.n_trades,
                        "max_drawdown": result.max_drawdown,
                        "bootstrap_pvalue": result.bootstrap_pvalue,
                    })
                except Exception:
                    pass

        return pd.DataFrame(rows).sort_values("sharpe", ascending=False)
