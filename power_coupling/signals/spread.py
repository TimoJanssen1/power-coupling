"""
Spread construction and decomposition.

Series identity (July 2026 revision): the SMARD price panel this module
consumes carries legacy column names from v1's misidentification of the SMARD
filter codes. What the spreads actually are:

  "da_id_spread"  = DE/LU day-ahead − Belgian day-ahead      (cross-zonal DA spread)
  "id3_spread"    = Danish DK1 day-ahead − Belgian day-ahead (cross-zonal DA spread)
  "id1_spread"    = all-NaN (SMARD filter 251 does not exist)
  "shape_spread"  = Belgian DA (hourly) − hourly mean of the "Anrainer DE/LU"
                    quarter-hourly series. Before the 15-min MTU go-live
                    (Oct 2025) the QH leg is an hourly price replicated 4×, so
                    the "shape" component is zero and the spread is purely
                    cross-zonal.

None of these is an auction-vs-continuous or hourly-vs-QH microstructure spread;
see FINDINGS.md "Revision notes" for how this changes the interpretation.

Negative prices are preserved: they are a structural feature of European power
markets during high-renewable/low-demand periods and must not be winsorized away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SpreadConstructor:
    """
    Builds and decomposes cross-zonal price spread series.

    Parameters
    ----------
    price_panel : pd.DataFrame
        Must contain columns: da_price, id1_price, id3_price, id_continuous.
        UTC-aware DatetimeIndex, hourly frequency.
    qh_prices : pd.Series, optional
        Quarter-hourly price series for the shape spread (Q3): in this repo
        the "Anrainer DE/LU" day-ahead series (legacy name id_continuous_qh).
        UTC-aware DatetimeIndex, 15-min frequency.
    """

    def __init__(self, price_panel: pd.DataFrame, qh_prices: pd.Series | None = None) -> None:
        self._validate_panel(price_panel)
        self.panel = price_panel.copy()
        self.qh_prices = qh_prices.copy() if qh_prices is not None else None

    # ------------------------------------------------------------------
    # Q2: cross-zonal DA spreads (legacy "auction/continuous" names)
    # ------------------------------------------------------------------

    def id1_spread(self) -> pd.Series:
        """
        All-NaN by construction: the "id1_price" column is empty over the whole
        sample because SMARD filter 251 does not exist (the fetcher's weekly
        requests all 404). Kept only so the spread-panel schema is stable.

        (The real EPEX ID1 is a volume-weighted index of continuous trades in
        the last hour before delivery: an ex-post statistic, not an auction
        price, and not available from SMARD.)
        """
        spread = self.panel["id1_price"] - self.panel["id_continuous"]
        return spread.rename("id1_spread")

    def id3_spread(self) -> pd.Series:
        """
        Danish DK1 day-ahead price minus Belgian day-ahead price for the same
        delivery hour (legacy name "id3_spread"). A cross-zonal spread between
        two zones coupled to DE/LU in the single day-ahead auction; both legs
        clear simultaneously at ~12:00 D-1.
        """
        spread = self.panel["id3_price"] - self.panel["id_continuous"]
        return spread.rename("id3_spread")

    def da_id_spread(self) -> pd.Series:
        """DE/LU day-ahead price minus Belgian day-ahead price (cross-zonal spread)."""
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
    # Q3: shape spread (hourly vs. quarter-hourly)
    # ------------------------------------------------------------------

    def shape_spread(self) -> pd.Series:
        """
        "Shape spread" = hourly price − mean(4 × 15-min prices within that hour).

        With this repo's data the hourly leg is the Belgian DA price and the QH
        leg is SMARD's "Anrainer DE/LU" neighbouring-zone series, so this is a
        cross-zonal spread, not an hourly-vs-QH product spread. Before Oct 2025
        the QH leg carries no intra-hour variation at all (hourly values
        replicated 4×). See FINDINGS.md "Revision notes".

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
        """Shape spread stats by hour of day."""
        ss = self.shape_spread()
        df = ss.to_frame()
        df["hour"] = df.index.hour
        return df.groupby("hour")["shape_spread"].agg(["mean", "std", "count", "median"])

    def qh_intra_hour_std(self) -> pd.Series:
        """
        Intra-hour standard deviation of the QH price series.

        With this repo's data the QH series is hourly-replicated before the
        15-minute MTU go-live (Oct 2025), so this statistic is identically
        zero for ~91% of the sample and only becomes informative afterwards.
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

        Returns half-life in hours (inf if non-mean-reverting).
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

        Day-ahead auctions clear negative during high renewable / low demand;
        structural, not outliers. Flagged for conditional analysis.
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
        Spread variance conditioned on a discrete regime variable
        (regime: integer series, e.g. 0=low, 1=med, 2=high uncertainty).
        Returns a DataFrame indexed by regime.
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
