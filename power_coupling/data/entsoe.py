"""
ENTSO-E Transparency Platform client wrapper.

Caches raw responses as Parquet in ~/.cache/power_coupling/entsoe/
so repeated runs don't burn API quota.  Cache key = (query_name, area, start, end).

Usage
-----
>>> from power_coupling.data import EntsoeFetcher
>>> fetcher = EntsoeFetcher(api_key="YOUR_KEY")
>>> fe = fetcher.forecast_error("DE_LU", "2022-01-01", "2024-01-01")
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(
    os.environ.get("PM_CACHE_DIR", Path.home() / ".cache" / "power_coupling" / "entsoe")
)
_AREA = "DE_LU"  # Germany-Luxembourg bidding zone (post-2018 market coupling)


class EntsoeFetcher:
    """Thin wrapper around entsoe-py with Parquet caching and error handling."""

    def __init__(self, api_key: str | None = None, cache: bool = True) -> None:
        key = api_key or os.environ.get("ENTSOE_API_KEY")
        if not key:
            raise ValueError(
                "Provide api_key= or set ENTSOE_API_KEY env var. "
                "Register at https://transparency.entsoe.eu/"
            )
        self._client = EntsoePandasClient(api_key=key)
        self._cache = cache
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wind_solar_forecast(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Day-ahead generation forecast for wind (on+offshore) and solar (MWh)."""
        return self._cached("wind_solar_forecast", area, start, end,
                            self._client.query_wind_and_solar_forecast, area,
                            start=self._ts(start), end=self._ts(end), psr_type=None)

    def actual_generation_per_type(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.DataFrame:
        """Actual generation by production type (MWh).

        ENTSO-E's ``query_generation`` returns ~17 columns × 35k rows per year.
        For multi-year windows the ENTSO-E gateway routinely returns 504s on
        the year-sized requests, so we chunk in 60-day blocks ourselves and
        cache the *combined* result. Each block has its own retry path inside
        entsoe-py's decorator.
        """
        cache_path = self._cache_path("actual_generation", area, start, end)
        if self._cache and cache_path.exists():
            return pd.read_parquet(cache_path)

        s_ts = self._ts(start)
        e_ts = self._ts(end)
        chunks: list[pd.DataFrame] = []
        cursor = s_ts
        while cursor < e_ts:
            block_end = min(cursor + pd.Timedelta(days=60), e_ts)
            try:
                df = self._client.query_generation(area, start=cursor, end=block_end)
                if df is not None and not df.empty:
                    chunks.append(df)
                logger.info(
                    "actual_generation chunk %s → %s  (%s rows)",
                    cursor.date(), block_end.date(),
                    "0" if df is None or df.empty else len(df),
                )
            except Exception as exc:
                logger.warning(
                    "actual_generation chunk %s → %s failed: %s",
                    cursor.date(), block_end.date(), exc,
                )
            cursor = block_end

        if not chunks:
            return pd.DataFrame()

        combined = pd.concat(chunks).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        if self._cache:
            saveable = combined.copy()
            saveable.columns = [str(c) for c in saveable.columns]
            saveable.to_parquet(cache_path)
        return combined

    def day_ahead_prices(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.Series:
        """Day-ahead market clearing prices (EUR/MWh)."""
        return self._cached("da_prices", area, start, end,
                            self._client.query_day_ahead_prices, area,
                            start=self._ts(start), end=self._ts(end))

    def load_forecast(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.Series:
        """Day-ahead total load forecast (MW)."""
        return self._cached("load_forecast", area, start, end,
                            self._client.query_load_forecast, area,
                            start=self._ts(start), end=self._ts(end))

    def actual_load(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.Series:
        """Actual total load (MW)."""
        return self._cached("actual_load", area, start, end,
                            self._client.query_load, area,
                            start=self._ts(start), end=self._ts(end))

    def cross_border_flows(
        self, area_from: str, area_to: str, start: str = "2022-01-01", end: str = "2024-01-01"
    ) -> pd.Series:
        """Physical cross-border flows between two areas (MWh)."""
        tag = f"{area_from}_{area_to}"
        return self._cached("flows", tag, start, end,
                            self._client.query_crossborder_flows, area_from, area_to,
                            start=self._ts(start), end=self._ts(end))

    # ------------------------------------------------------------------
    # Derived / convenience
    # ------------------------------------------------------------------

    def forecast_error(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.DataFrame:
        """
        Signed renewable forecast error: actual − forecast (MWh).

        Positive  → more renewable generation than expected (bearish price pressure).
        Negative  → less renewable generation than expected (bullish price pressure).

        Returns DataFrame with columns: wind_error, solar_error, total_error.
        Resampled to 60-minute intervals to align forecast (hourly) with actuals (15-min).
        """
        forecast = self.wind_solar_forecast(area, start, end)
        actual = self.actual_generation_per_type(area, start, end)

        forecast = self._resample_hourly(forecast)
        actual = self._resample_hourly(actual)

        wind_cols_fc = [c for c in forecast.columns if "Wind" in str(c)]
        solar_cols_fc = [c for c in forecast.columns if "Solar" in str(c)]

        wind_fc = (
            forecast[wind_cols_fc].sum(axis=1)
            if wind_cols_fc
            else pd.Series(0, index=forecast.index)
        )
        solar_fc = (
            forecast[solar_cols_fc].sum(axis=1)
            if solar_cols_fc
            else pd.Series(0, index=forecast.index)
        )

        wind_cols_ac = [c for c in actual.columns if "Wind" in str(c)]
        solar_cols_ac = [c for c in actual.columns if "Solar" in str(c)]

        wind_ac = (
            actual[wind_cols_ac].sum(axis=1)
            if wind_cols_ac
            else pd.Series(0, index=actual.index)
        )
        solar_ac = (
            actual[solar_cols_ac].sum(axis=1)
            if solar_cols_ac
            else pd.Series(0, index=actual.index)
        )

        idx = wind_fc.index.intersection(wind_ac.index)
        wind_error = wind_ac.loc[idx] - wind_fc.loc[idx]
        solar_error = solar_ac.loc[idx] - solar_fc.loc[idx]

        return pd.DataFrame({
            "wind_error": wind_error,
            "solar_error": solar_error,
            "total_error": wind_error + solar_error,
        })

    def renewable_share(
        self, area: str = _AREA, start: str = "2018-01-01", end: str = "2024-01-01"
    ) -> pd.Series:
        """Hourly renewable share = (wind + solar) / total generation."""
        actual = self._resample_hourly(self.actual_generation_per_type(area, start, end))
        wind_cols = [c for c in actual.columns if "Wind" in str(c)]
        solar_cols = [c for c in actual.columns if "Solar" in str(c)]
        renewable = actual[wind_cols + solar_cols].sum(axis=1)
        total = actual.sum(axis=1).replace(0, float("nan"))
        return (renewable / total).rename("renewable_share")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ts(date_str: str) -> pd.Timestamp:
        return pd.Timestamp(date_str, tz="Europe/Berlin")

    @staticmethod
    def _resample_hourly(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        if isinstance(df, pd.Series):
            return df.resample("1h").mean()
        return df.resample("1h").mean()

    def _cache_path(self, *parts: str) -> Path:
        key = hashlib.md5("_".join(parts).encode()).hexdigest()[:12]
        name = "_".join(p.replace("/", "-") for p in parts[:3])
        return _CACHE_ROOT / f"{name}_{key}.parquet"

    def _cached(
        self, query_name: str, area: str, start_key: str, end_key: str, fn, *args, **kwargs
    ):
        path = self._cache_path(query_name, area, start_key, end_key)
        if self._cache and path.exists():
            logger.debug("Cache hit: %s", path.name)
            cached = pd.read_parquet(path)
            # Series-returning queries (prices, load) are stored as one-column
            # frames (parquet has no Series); squeeze back so cache hits return
            # the same type as live API calls.
            if cached.shape[1] == 1:
                return cached.iloc[:, 0]
            return cached
        try:
            result = fn(*args, **kwargs)
        except NoMatchingDataError:
            logger.warning("No data returned for %s %s %s–%s", query_name, area, start_key, end_key)
            return pd.DataFrame()
        if self._cache and result is not None and not getattr(result, "empty", True):
            to_save = result.to_frame() if isinstance(result, pd.Series) else result
            # parquet requires string column names
            to_save = to_save.copy()
            to_save.columns = [str(c) for c in to_save.columns]
            to_save.to_parquet(path)
        return result
