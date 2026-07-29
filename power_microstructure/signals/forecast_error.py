"""
Renewable forecast error signal construction.

Forecast error = actual − forecast (signed, MWh).

  Positive error → more wind/solar than expected → bearish price pressure
  Negative error → less wind/solar than expected → bullish price pressure

Q1's driver variable. (v1 framed it as intraday order flow; post-revision it
is related to next-day zonal day-ahead prices — see FINDINGS.md.)

Key outputs
-----------
ForecastErrorSignal.build()          — raw hourly error series (wind, solar, total)
ForecastErrorSignal.standardised()  — z-scored over rolling 30-day window
ForecastErrorSignal.cumulative()    — cumulative intraday error (reset at 00:00)
ForecastErrorSignal.uncertainty()   — rolling forecast variance (proxy for uncertainty)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class ForecastErrorSignal:
    """
    Constructs and transforms renewable forecast error signals.

    Parameters
    ----------
    errors : pd.DataFrame
        DataFrame with columns [wind_error, solar_error, total_error].
        Index must be a UTC-aware DatetimeIndex at hourly frequency.
    window : int
        Rolling window (hours) for standardisation and uncertainty estimates.
        Default 30 days = 720 hours.
    """

    def __init__(self, errors: pd.DataFrame, window: int = 720) -> None:
        self._validate(errors)
        self.errors = errors.copy()
        self.window = window

    # ------------------------------------------------------------------
    # Core signal outputs
    # ------------------------------------------------------------------

    def build(self) -> pd.DataFrame:
        """Return raw hourly forecast errors with additional derived columns."""
        df = self.errors.copy()
        df["abs_total_error"] = df["total_error"].abs()
        df["error_direction"] = np.sign(df["total_error"])
        # Separate surprise magnitude by type
        wind_scale = df["wind_error"].abs().rolling(self.window, min_periods=168).mean()
        df["wind_pct_surprise"] = df["wind_error"] / (wind_scale + 1e-6)
        solar_scale = df["solar_error"].abs().rolling(self.window, min_periods=168).mean()
        df["solar_pct_surprise"] = df["solar_error"] / (solar_scale + 1e-6)
        return df

    def standardised(self) -> pd.DataFrame:
        """Z-score each error column over a rolling window."""
        df = self.errors.copy()
        for col in ["wind_error", "solar_error", "total_error"]:
            roll_mean = df[col].rolling(self.window, min_periods=168).mean()
            roll_std = df[col].rolling(self.window, min_periods=168).std()
            df[f"{col}_z"] = (df[col] - roll_mean) / (roll_std + 1e-9)
        return df[[c for c in df.columns if c.endswith("_z")]]

    def cumulative(self) -> pd.Series:
        """
        Intraday cumulative error, reset at midnight.

        Captures the net imbalance signal that has accumulated within the
        delivery day.
        """
        df = self.errors.copy()
        df["date"] = df.index.normalize()
        return df.groupby("date")["total_error"].cumsum().rename("cumulative_error")

    def uncertainty(self) -> pd.Series:
        """
        Rolling forecast uncertainty: variance of total_error over the window.

        High uncertainty periods are expected to drive wider cross-zonal spreads.
        Used as a conditioning variable in Q2 spread regressions.
        """
        return (
            self.errors["total_error"]
            .rolling(self.window, min_periods=168)
            .var()
            .rename("forecast_uncertainty")
        )

    def regime(self, n_regimes: int = 3) -> pd.Series:
        """
        Discretise absolute forecast error into uncertainty regimes.

        0 = low,  1 = medium,  2 = high (based on rolling terciles).
        Used as a conditioning variable for regime-conditional variance analysis.
        """
        abs_err = self.errors["total_error"].abs()
        rolling_terciles = abs_err.rolling(self.window, min_periods=168).quantile
        low = rolling_terciles(1 / n_regimes)
        high = rolling_terciles((n_regimes - 1) / n_regimes)
        regime = pd.Series(1, index=abs_err.index, dtype=int, name="error_regime")
        regime[abs_err <= low] = 0
        regime[abs_err >= high] = 2
        return regime

    def by_hour_of_day(self) -> pd.DataFrame:
        """Average absolute error by hour of day — shows intraday structure."""
        df = self.errors.copy()
        df["hour"] = df.index.hour
        return df.groupby("hour")[["wind_error", "solar_error", "total_error"]].agg(
            ["mean", "std", "count"]
        )

    def renewable_penetration_quartile(self, renewable_share: pd.Series) -> pd.Series:
        """
        Assign each observation to a renewable penetration quartile.

        Allows interaction analysis: does the forecast error → price link
        strengthen at higher penetration levels?
        """
        shared_idx = self.errors.index.intersection(renewable_share.index)
        quantile_bins = pd.qcut(renewable_share.loc[shared_idx], q=4, labels=[0, 1, 2, 3])
        return quantile_bins.rename("penetration_quartile").astype(int)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        required = {"wind_error", "solar_error", "total_error"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Index must be DatetimeIndex")
