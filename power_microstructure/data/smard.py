"""
SMARD (Bundesnetzagentur) data fetcher.

SMARD provides free, no-key-required access to German electricity market data including:
  - Intraday continuous index prices (ID1, ID3, intraday volume-weighted)
  - Day-ahead auction prices and volumes
  - Actual generation / consumption by energy source
  - Quarter-hourly and hourly granularity

API documentation: https://www.smard.de/en/downloadcenter/download-market-data/

Data is cached as Parquet; set PM_CACHE_DIR env var to override location.

Usage
-----
>>> from power_microstructure.data import SmardFetcher
>>> sf = SmardFetcher()
>>> prices = sf.intraday_continuous_index("2022-01-01", "2024-01-01")
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Literal
import os
import hashlib

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(os.environ.get("PM_CACHE_DIR", Path.home() / ".cache" / "power_microstructure" / "smard"))

# SMARD filter/region codes
_SMARD_BASE = "https://www.smard.de/app/chart_data"

# SMARD data index codes (from their API)
_FILTER_CODES = {
    # Market prices
    "da_price": 4169,           # Day-ahead price (EPEX DE-LU)
    "intraday_index": 4996,     # Intraday continuous mean (hourly)
    "intraday_index_qh": 5078,  # Intraday continuous mean (15-min)
    "id1_price": 251,           # ID1 auction price
    "id3_price": 252,           # ID3 auction price
    # Generation (actual)
    "wind_onshore": 1225,
    "wind_offshore": 1224,
    "solar": 4067,
    "biomass": 4066,
    "hydro": 1226,
    "other_renewable": 1228,
    "nuclear": 1223,
    "lignite": 1221,
    "hard_coal": 1222,
    "gas": 4071,
    "pumped_hydro": 4359,
    # Load
    "load_actual": 410,
    "load_forecast": 4359,
    # Cross-border
    "export": 1227,
    "import": 4996,
}

_REGION = "DE"
_RESOLUTION_MAP = {
    "hourly": "hour",
    "qh": "quarterhour",
    "daily": "day",
    "weekly": "week",
}


class SmardFetcher:
    """Fetches SMARD data via their JSON chart API with Parquet caching."""

    def __init__(self, cache: bool = True, throttle: float = 0.5) -> None:
        self._cache = cache
        self._throttle = throttle  # seconds between API calls to be polite
        _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "power-microstructure-research/0.1"})

    # ------------------------------------------------------------------
    # Public API — price series
    # ------------------------------------------------------------------

    def day_ahead_prices(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """EPEX DE-LU day-ahead auction clearing prices (EUR/MWh, hourly)."""
        return self._fetch_series("da_price", start, end, "hourly").rename("da_price")

    def intraday_continuous_index(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """Intraday continuous volume-weighted mean price (EUR/MWh, hourly)."""
        return self._fetch_series("intraday_index", start, end, "hourly").rename("id_continuous")

    def intraday_continuous_index_qh(self, start: str = "2022-01-01", end: str = "2024-01-01") -> pd.Series:
        """Intraday continuous volume-weighted mean price (EUR/MWh, quarter-hourly)."""
        return self._fetch_series("intraday_index_qh", start, end, "qh").rename("id_continuous_qh")

    def id1_prices(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """EPEX ID1 intraday auction clearing prices (EUR/MWh, hourly)."""
        return self._fetch_series("id1_price", start, end, "hourly").rename("id1_price")

    def id3_prices(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """EPEX ID3 intraday auction clearing prices (EUR/MWh, hourly)."""
        return self._fetch_series("id3_price", start, end, "hourly").rename("id3_price")

    # ------------------------------------------------------------------
    # Public API — generation & load
    # ------------------------------------------------------------------

    def wind_generation(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.DataFrame:
        """Actual wind generation: onshore + offshore (MWh, hourly)."""
        on = self._fetch_series("wind_onshore", start, end, "hourly").rename("wind_onshore")
        off = self._fetch_series("wind_offshore", start, end, "hourly").rename("wind_offshore")
        return pd.concat([on, off], axis=1)

    def solar_generation(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """Actual solar (photovoltaic) generation (MWh, hourly)."""
        return self._fetch_series("solar", start, end, "hourly").rename("solar")

    def actual_load(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.Series:
        """Actual total load (MW, hourly)."""
        return self._fetch_series("load_actual", start, end, "hourly").rename("load_actual")

    # ------------------------------------------------------------------
    # Derived: full price panel for spread construction
    # ------------------------------------------------------------------

    def price_panel(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.DataFrame:
        """
        Aligned panel of DA, ID1, ID3, and continuous intraday prices.

        Columns: da_price, id1_price, id3_price, id_continuous
        Index: UTC-aware DatetimeIndex at hourly frequency.

        Missing values (e.g. ID3 not available for early dates) are left as NaN;
        callers should decide how to handle them rather than silently filling.
        """
        da = self.day_ahead_prices(start, end)
        id1 = self.id1_prices(start, end)
        id3 = self.id3_prices(start, end)
        cim = self.intraday_continuous_index(start, end)
        panel = pd.concat([da, id1, id3, cim], axis=1)
        panel.index = pd.to_datetime(panel.index, utc=True)
        return panel

    def renewable_generation_panel(self, start: str = "2018-01-01", end: str = "2024-01-01") -> pd.DataFrame:
        """Wind (on+off) and solar actual generation aligned to hourly UTC index."""
        wind = self.wind_generation(start, end)
        solar = self.solar_generation(start, end)
        df = pd.concat([wind, solar], axis=1)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_series(self, series_key: str, start: str, end: str, resolution: str) -> pd.Series:
        cache_path = self._cache_path(series_key, start, end, resolution)
        if self._cache and cache_path.exists():
            logger.debug("Cache hit: %s", cache_path.name)
            return pd.read_parquet(cache_path).iloc[:, 0]

        filter_code = _FILTER_CODES[series_key]
        res = _RESOLUTION_MAP[resolution]

        # Ask SMARD for the list of valid weekly anchor timestamps. Anchors
        # are at the start-of-week boundary in *local* terms (typically
        # Sunday 22:00 UTC = Monday 00:00 Berlin, shifted by DST).
        index_url = f"{_SMARD_BASE}/{filter_code}/{_REGION}/index_{res}.json"
        idx_resp = self._session.get(index_url, timeout=30)
        idx_resp.raise_for_status()
        all_ts = idx_resp.json().get("timestamps", [])

        s_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        # Include the week BEFORE start (so the start day is covered) and
        # all weeks up to end.
        prior = max((t for t in all_ts if t <= s_ms), default=None)
        weeks = [t for t in all_ts if s_ms <= t <= e_ms + 7 * 24 * 3600 * 1000]
        if prior is not None and prior not in weeks:
            weeks.insert(0, prior)

        chunks: list[pd.Series] = []
        for ts in weeks:
            try:
                chunk = self._fetch_week(filter_code, ts, res)
                if chunk is not None:
                    chunks.append(chunk)
            except Exception as exc:
                logger.warning("Failed to fetch %s @ %s: %s", series_key, ts, exc)
            time.sleep(self._throttle)

        if not chunks:
            return pd.Series(dtype=float, name=series_key)

        result = pd.concat(chunks).sort_index()
        result = result[~result.index.duplicated(keep="first")]
        # Filter to requested window (tz-aware)
        s_aware = pd.Timestamp(start, tz="UTC")
        e_aware = pd.Timestamp(end, tz="UTC")
        result = result.loc[(result.index >= s_aware) & (result.index <= e_aware)]

        df = result.to_frame()
        if self._cache:
            df.to_parquet(cache_path)

        return result

    def _fetch_week(self, filter_code: int, timestamp_ms: int, resolution: str) -> pd.Series | None:
        url = f"{_SMARD_BASE}/{filter_code}/{_REGION}/{filter_code}_{_REGION}_{resolution}_{timestamp_ms}.json"
        resp = self._session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        series_data = data.get("series", [])
        if not series_data:
            return None
        # Each entry is [timestamp_ms, value_or_null]
        records = [(pd.Timestamp(ts, unit="ms", tz="UTC"), v) for ts, v in series_data if v is not None]
        if not records:
            return None
        idx, vals = zip(*records)
        return pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=float)

    def _cache_path(self, *parts: str) -> Path:
        key = hashlib.md5("_".join(parts).encode()).hexdigest()[:12]
        name = "_".join(str(p).replace("/", "-") for p in parts[:3])
        return _CACHE_ROOT / f"{name}_{key}.parquet"

    # ------------------------------------------------------------------
    # Fallback: direct CSV download from SMARD download center
    # ------------------------------------------------------------------

    def download_csv(
        self,
        module_ids: list[int],
        start: str,
        end: str,
        resolution: Literal["quarterhour", "hour", "day"] = "hour",
        region: str = "DE",
    ) -> pd.DataFrame:
        """
        Download data via SMARD's CSV download endpoint.

        Useful when the chart JSON API lacks a specific series.
        """
        s_ts = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
        e_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
        url = (
            "https://www.smard.de/app/downloadcenter/download_market_data/?"
            f"request_form[moduleIds][]={'&request_form[moduleIds][]='.join(str(m) for m in module_ids)}"
            f"&request_form[region]={region}"
            f"&request_form[timestamp_from]={s_ts}"
            f"&request_form[timestamp_to]={e_ts}"
            f"&request_form[type]={resolution}"
            f"&request_form[language]=en"
        )
        resp = self._session.get(url, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(
            io.StringIO(resp.text),
            sep=";",
            decimal=",",
            parse_dates=["Date from", "Date to"],
            dayfirst=True,
        )
        df = df.set_index("Date from")
        df.index = pd.to_datetime(df.index, utc=True)
        return df
