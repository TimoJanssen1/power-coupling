"""
Shared fixtures for the power_microstructure test suite.

All fixtures use synthetic data so tests run without API keys or network access.
The synthetic series are designed to have known statistical properties that allow
us to test that our analysis modules produce the correct output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_index():
    """Two years of hourly UTC timestamps starting 2022-01-01."""
    return pd.date_range("2022-01-01", periods=8760 * 2, freq="1h", tz="UTC")


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def forecast_errors(hourly_index, rng):
    """
    Synthetic forecast error DataFrame with known properties.

    wind_error ~ N(0, 500^2), solar_error ~ N(0, 200^2) (plausible MWh scale).
    """
    n = len(hourly_index)
    wind = rng.normal(0, 500, size=n)
    solar = rng.normal(0, 200, size=n)
    return pd.DataFrame(
        {"wind_error": wind, "solar_error": solar, "total_error": wind + solar},
        index=hourly_index,
    )


@pytest.fixture
def price_panel(hourly_index, rng):
    """
    Synthetic price panel: DA, ID1, ID3, continuous (EUR/MWh).

    DA and ID1 are cointegrated (share a common trend), with an I(0) spread.
    Some negative prices are included to test that they are handled correctly.
    """
    n = len(hourly_index)
    common_trend = np.cumsum(rng.normal(0, 0.5, size=n))
    da = 50 + common_trend + rng.normal(0, 3, size=n)
    id1 = da + rng.normal(0, 2, size=n)      # spread ~ N(0, 4)
    id3 = da + rng.normal(0, 2.5, size=n)
    cim = da + rng.normal(0, 1.5, size=n)

    # Inject some negative prices (structurally realistic for Germany)
    neg_mask = rng.random(n) < 0.03
    da[neg_mask] -= 80
    id1[neg_mask] -= 80

    return pd.DataFrame(
        {"da_price": da, "id1_price": id1, "id3_price": id3, "id_continuous": cim},
        index=hourly_index,
    )


@pytest.fixture
def qh_index():
    """One year of quarter-hourly UTC timestamps."""
    return pd.date_range("2022-01-01", periods=4 * 8760, freq="15min", tz="UTC")


@pytest.fixture
def qh_prices(qh_index, rng):
    """Synthetic quarter-hourly prices."""
    n = len(qh_index)
    return pd.Series(50 + rng.normal(0, 3, size=n), index=qh_index, name="id_continuous_qh")


@pytest.fixture
def mean_reverting_spread(hourly_index, rng):
    """
    An AR(1) mean-reverting spread with half-life ~50 hours.

    phi = exp(-ln(2)/50) ≈ 0.9863
    """
    phi = np.exp(-np.log(2) / 50)
    n = len(hourly_index)
    s = np.zeros(n)
    s[0] = 0.0
    noise = rng.normal(0, 1, size=n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + noise[t]
    return pd.Series(s, index=hourly_index, name="spread")


@pytest.fixture
def granger_price_series(hourly_index, rng, forecast_errors):
    """
    Price series that IS Granger-caused by forecast_error with a 2-hour lag.

    price_change_t = 0.005 * fe_{t-2} + noise
    """
    fe = forecast_errors["total_error"].values
    n = len(hourly_index)
    price_change = np.zeros(n)
    for t in range(2, n):
        price_change[t] = 0.005 * fe[t - 2] + rng.normal(0, 2)
    return pd.Series(price_change, index=hourly_index, name="price_change")
