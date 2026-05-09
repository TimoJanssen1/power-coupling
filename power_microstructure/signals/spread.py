"""
Spread construction and decomposition.

Three spreads — one per research question:

  Q2  Auction/continuous spread  = ID auction price − intraday continuous VWAP
      (for the same delivery hour/period)
  Q3  Shape spread               = hourly block price − mean of four QH prices
      (within the same delivery hour)

Negative prices are preserved — they are a structural feature of the German market
during high-renewable/low-demand periods and must not be winsorized away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SpreadConstructor:
    """
    Builds and decomposes intraday spread series.

    Parameters
    ----------
    price_panel : pd.DataFrame
        Must contain columns: da_price, id1_price, id3_price, id_continuous.
        UTC-aware DatetimeIndex, hourly frequency.
    qh_prices : pd.Series, optional
        Quarter-hourly intraday continuous prices for shape spread (Q3).
        UTC-aware DatetimeIndex, 15-min frequency.
    """

    def __init__(self, price_panel: pd.DataFrame, qh_prices: pd.Series | None = None) -> None:
        self._validate_panel(price_panel)
        self.panel = price_panel.copy()
        self.qh_prices = qh_prices.copy() if qh_prices is not None else None

    # ------------------------------------------------------------------
    # Q2 — Auction/continuous spread
    # ------------------------------------------------------------------

    def id1_spread(self) -> pd.Series:
        """
        ID1 auction price minus intraday continuous VWAP for the same delivery hour.

        ID1 clears at 15:00 CET for next-day delivery, so the spread is observable
        from ~15:00 onwards when both series are available.

        Positive spread  → auction cleared higher than post-auction continuous.
        Negative spread  → market sold off after auction (unexpected renewable surplus).
        """
        spread = self.panel["id1_price"] - self.panel["id_continuous"]
        return spread.rename("id1_spread")

    def id3_spread(self) -> pd.Series:
        """
        ID3 auction price minus intraday continuous VWAP for the same delivery hour.

        ID3 clears at 22:00 CET, so it captures overnight information.
        """
        spread = self.panel["id3_price"] - self.panel["id_continuous"]
        return spread.rename("id3_spread")

    def da_id_spread(self) -> pd.Series:
        """Day-ahead price minus intraday continuous VWAP."""
        spread = self.panel["da_price"] - self.panel["id_continuous"]
        return spread.rename("da_id_spread")

    def spread_panel(self) -> pd.DataFrame:
        """All spread series aligned to the price panel index."""
        return pd.DataFrame({
            "da_id_spread": self.da_id_spread(),
            "id1_spread": self.id1_spread(),
            "id3_spread": self.id3_spread(),
        })

    # ------------------------------------------------------------------
    # Q3 — Shape spread (hourly vs. quarter-hourly)
    # ------------------------------------------------------------------

    def shape_spread(self) -> pd.Series:
        """
        Shape spread = hourly block price − mean(4 × 15-min prices within that hour).

        A positive shape spread means the hourly block traded richer than the QH sum —
        market participants are paying a premium for the simplicity of an hourly block
        or for insurance against intra-hour renewable variability.

        Requires qh_prices to be provided at construction.
        """
        if self.qh_prices is None:
            raise ValueError("qh_prices must be provided for shape spread calculation.")

        # Resample QH to hourly mean (equal-weight within hour)
        qh_hourly_mean = self.qh_prices.resample("1h").mean()

        shared_idx = self.panel.index.intersection(qh_hourly_mean.index)
        hourly_price = self.panel["id_continuous"].loc[shared_idx]
        qh_mean = qh_hourly_mean.loc[shared_idx]

        spread = hourly_price - qh_mean
        return spread.rename("shape_spread")

    def shape_spread_by_hour(self) -> pd.DataFrame:
        """Shape spread decomposed by hour of day — captures intra-day structure."""
        ss = self.shape_spread()
        df = ss.to_frame()
        df["hour"] = df.index.hour
        return df.groupby("hour")["shape_spread"].agg(["mean", "std", "count", _median])

    def qh_intra_hour_std(self) -> pd.Series:
        """
        Intra-hour standard deviation of QH prices.

        This is the realized within-hour price volatility — the direct measure of
        intra-hour renewable variability that the shape spread is supposed to price.
        """
        if self.qh_prices is None:
            raise ValueError("qh_prices must be provided.")
        return self.qh_prices.resample("1h").std().rename("qh_intra_hour_std")

    # ------------------------------------------------------------------
    # Spread decomposition utilities
    # ------------------------------------------------------------------

    def half_life(self, spread: pd.Series) -> float:
        """
        Mean-reversion half-life from AR(1) regression on spread levels.

        Model: ΔS_t = α + β·S_{t-1} + ε_t
        Half-life = −ln(2) / ln(1 + β)

        Returns half-life in hours. Negative = diverging (non-mean-reverting).
        """
        s = spread.dropna()
        ds = s.diff().dropna()
        s_lag = s.shift(1).dropna()
        common = ds.index.intersection(s_lag.index)

        X = np.column_stack([np.ones(len(common)), s_lag.loc[common].values])
        y = ds.loc[common].values
        beta = np.linalg.lstsq(X, y, rcond=None)[0][1]

        if beta >= 0:
            return float("inf")  # unit root / diverging
        return float(-np.log(2) / np.log(1 + beta))

    def rolling_spread_stats(self, spread: pd.Series, window: int = 168) -> pd.DataFrame:
        """
        Rolling mean, std, and z-score of spread over a given window (hours).

        The z-score is the primary trading signal: enter when |z| > threshold.
        """
        roll = spread.rolling(window, min_periods=24)
        mean = roll.mean()
        std = roll.std()
        z = (spread - mean) / (std + 1e-9)
        return pd.DataFrame({
            "spread": spread,
            "roll_mean": mean,
            "roll_std": std,
            "z_score": z,
        })

    def negative_price_flag(self) -> pd.Series:
        """
        Binary flag: 1 when any auction price is negative.

        Germany has frequent negative intraday prices during high renewable / low demand.
        These are NOT outliers — they are structural. Flag them for conditional analysis.
        """
        neg = (
            (self.panel["id1_price"] < 0) |
            (self.panel["id3_price"] < 0) |
            (self.panel["da_price"] < 0)
        )
        return neg.rename("negative_price_flag").astype(int)

    def regime_conditional_variance(
        self,
        spread: pd.Series,
        regime: pd.Series,
    ) -> pd.DataFrame:
        """
        Spread variance conditioned on a discrete regime variable.

        Parameters
        ----------
        spread  : pd.Series
        regime  : pd.Series of integers (e.g. 0=low, 1=med, 2=high uncertainty)

        Returns DataFrame indexed by regime with spread variance and observation count.
        """
        df = pd.DataFrame({"spread": spread, "regime": regime}).dropna()
        return df.groupby("regime")["spread"].agg(["var", "std", "mean", "count"])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_panel(df: pd.DataFrame) -> None:
        required = {"da_price", "id1_price", "id3_price", "id_continuous"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


def _median(x: pd.Series) -> float:
    return float(x.median())
