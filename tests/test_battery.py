"""Tests for power_microstructure.strategy.battery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_microstructure.strategy import (
    BatterySpec,
    strategy_alt_series_lp,
    strategy_da_only_lp,
    strategy_naive,
)


def _soc_is_physical(schedule: pd.DataFrame, spec: BatterySpec) -> bool:
    """Replay the schedule chronologically and check SoC never goes negative
    or above capacity (energy cannot be discharged before it is charged)."""
    soc = spec.soc_init
    for _, row in schedule.iterrows():
        soc = soc + spec.eta_charge * row["charge_mwh"] - row["discharge_mwh"] / spec.eta_discharge
        if soc < -1e-9 or soc > spec.energy_mwh + 1e-9:
            return False
    return True


@pytest.fixture
def spec_empty_start():
    return BatterySpec(soc_init=0.0)


@pytest.fixture
def two_days_hourly():
    return pd.date_range("2023-06-01", periods=48, freq="1h", tz="UTC")


class TestNaiveChronology:
    def test_no_discharge_before_charge(self, spec_empty_start, two_days_hourly):
        """Richest hours FIRST in the day, cheapest hours later, empty battery:
        the naive strategy must not discharge energy it has not yet stored."""
        prices = np.full(48, 50.0)
        for day0 in (0, 24):
            prices[day0 + 0] = 200.0   # richest
            prices[day0 + 1] = 190.0   # second richest
            prices[day0 + 20] = 1.0    # cheapest
            prices[day0 + 21] = 2.0    # second cheapest
        da = pd.Series(prices, index=two_days_hourly)
        res = strategy_naive(da, da, spec_empty_start)
        s = res.schedule

        # Day 1, hours 0-1 (the expensive hours): battery starts empty, so
        # discharging there would be physically impossible.
        assert s["discharge_mwh"].iloc[0] == 0.0
        assert s["discharge_mwh"].iloc[1] == 0.0
        # It still charges in the cheap hours.
        assert s["charge_mwh"].iloc[20] > 0.0
        assert _soc_is_physical(s, spec_empty_start)

    def test_discharge_after_charge_allowed(self, spec_empty_start, two_days_hourly):
        """Cheapest hours before richest hours: the classic arbitrage day works."""
        prices = np.full(48, 50.0)
        for day0 in (0, 24):
            prices[day0 + 2] = 1.0     # cheapest
            prices[day0 + 3] = 2.0
            prices[day0 + 19] = 200.0  # richest
            prices[day0 + 20] = 190.0
        da = pd.Series(prices, index=two_days_hourly)
        res = strategy_naive(da, da, spec_empty_start)
        s = res.schedule
        assert s["discharge_mwh"].iloc[19] > 0.0
        assert _soc_is_physical(s, spec_empty_start)
        assert res.revenue_eur > 0.0

    def test_soc_physical_on_random_prices(self, spec_empty_start):
        idx = pd.date_range("2023-06-01", periods=24 * 10, freq="1h", tz="UTC")
        rng = np.random.default_rng(3)
        da = pd.Series(50 + 30 * rng.standard_normal(len(idx)), index=idx)
        res = strategy_naive(da, da, spec_empty_start)
        assert _soc_is_physical(res.schedule, spec_empty_start)

    def test_soc_column_matches_replay(self, spec_empty_start, two_days_hourly):
        """The reported soc_mwh column must equal the replayed SoC path."""
        rng = np.random.default_rng(5)
        da = pd.Series(50 + 30 * rng.standard_normal(48), index=two_days_hourly)
        res = strategy_naive(da, da, spec_empty_start)
        soc = spec_empty_start.soc_init
        for _, row in res.schedule.iterrows():
            soc = (soc + spec_empty_start.eta_charge * row["charge_mwh"]
                   - row["discharge_mwh"] / spec_empty_start.eta_discharge)
            assert abs(row["soc_mwh"] - soc) < 1e-9


class TestSettlementConvention:
    def test_naive_settles_at_da(self, two_days_hourly):
        rng = np.random.default_rng(11)
        da = pd.Series(50 + 20 * rng.standard_normal(48), index=two_days_hourly)
        other = da + 10.0
        res = strategy_naive(da, other, BatterySpec())
        pd.testing.assert_series_equal(
            res.schedule["settlement_price"], res.schedule["price_da"],
            check_names=False,
        )

    def test_da_lp_settles_at_da(self, two_days_hourly):
        rng = np.random.default_rng(12)
        da = pd.Series(50 + 20 * rng.standard_normal(48), index=two_days_hourly)
        other = da + 10.0
        res = strategy_da_only_lp(da, other, BatterySpec())
        pd.testing.assert_series_equal(
            res.schedule["settlement_price"], res.schedule["price_da"],
            check_names=False,
        )

    def test_alt_series_lp_settles_on_alt_series(self, two_days_hourly):
        rng = np.random.default_rng(13)
        da = pd.Series(50 + 20 * rng.standard_normal(48), index=two_days_hourly)
        other = pd.Series(50 + 20 * rng.standard_normal(48), index=two_days_hourly)
        res = strategy_alt_series_lp(da, other, BatterySpec())
        np.testing.assert_allclose(
            res.schedule["settlement_price"].values,
            other.loc[res.schedule.index].values,
        )
