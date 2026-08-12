"""Tests for power_coupling.strategy.backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_coupling.strategy import BacktestResult, WalkForwardBacktest


@pytest.fixture
def stationary_spread(hourly_index, rng):
    """Fast-mean-reverting spread: HL ~8 hours (aggressive for testing)."""
    phi = np.exp(-np.log(2) / 8)
    n = len(hourly_index)
    s = np.zeros(n)
    noise = rng.normal(0, 3, size=n)
    for t in range(1, n):
        s[t] = phi * s[t - 1] + noise[t]
    return pd.Series(s, index=hourly_index, name="spread")


@pytest.fixture
def random_walk_spread(hourly_index, rng):
    """Non-mean-reverting (random walk) spread; should not show consistent profit."""
    return pd.Series(
        np.cumsum(rng.normal(0, 1, size=len(hourly_index))), index=hourly_index, name="spread"
    )


class TestWalkForwardBacktest:
    def test_run_returns_backtest_result(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        assert isinstance(result, BacktestResult)

    def test_pnl_series_length_matches_oos_data(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        # PnL should cover the out-of-sample period (not the entire series)
        oos_start = stationary_spread.index[0] + pd.DateOffset(months=6)
        assert result.pnl.index[0] >= oos_start

    def test_cumulative_pnl_is_cumsum_of_pnl(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        pd.testing.assert_series_equal(
            result.cumulative_pnl,
            result.pnl.cumsum(),
            check_names=False,
        )

    def test_trades_dataframe_has_required_columns(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        if result.n_trades > 0:
            required = {
                "entry_time", "exit_time", "direction", "entry_price",
                "exit_price", "pnl", "hold_hours",
            }
            assert required.issubset(set(result.trades.columns))

    def test_direction_values_are_plus_minus_one(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        if result.n_trades > 0:
            assert set(result.trades["direction"].unique()).issubset({-1, 1})

    def test_sharpe_is_finite(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run()
        assert np.isfinite(result.sharpe)

    def test_bootstrap_pvalue_in_unit_interval(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=200)
        result = bt.run(entry_z=1.5, exit_z=0.3)
        assert 0.0 <= result.bootstrap_pvalue <= 1.0

    def test_max_drawdown_is_non_positive(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run()
        assert result.max_drawdown <= 0.0

    def test_transaction_cost_deducted(self, stationary_spread):
        """Net PnL per trade should be less than gross PnL."""
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run()
        if result.n_trades > 0:
            assert (result.trades["pnl"] <= result.trades["gross_pnl"]).all(), (
                "Net PnL must be ≤ gross PnL after costs."
            )

    def test_fold_structure_is_non_overlapping(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        folds = bt._build_folds()
        assert len(folds) > 0
        for i in range(1, len(folds)):
            _, prev_test = folds[i - 1]
            _, curr_test = folds[i]
            prev_end = stationary_spread.loc[prev_test].index[-1]
            curr_start = stationary_spread.loc[curr_test].index[0]
            assert curr_start > prev_end, "Test folds should not overlap."

    def test_sensitivity_returns_dataframe(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=50)
        df = bt.sensitivity(entry_z_range=[1.5, 2.0], exit_z_range=[0.3, 0.5])
        assert isinstance(df, pd.DataFrame)
        assert "sharpe" in df.columns

    def test_summary_has_all_keys(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run()
        summary = result.summary()
        assert "Sharpe (annualised)" in summary.index
        assert "Bootstrap p-value (Sharpe>0)" in summary.index

    def test_win_rate_in_unit_interval(self, stationary_spread):
        bt = WalkForwardBacktest(stationary_spread, train_months=6, test_months=3, n_bootstrap=100)
        result = bt.run()
        assert 0.0 <= result.win_rate <= 1.0

    def test_no_negative_price_handling_crashes(self, price_panel):
        """Spread constructed from panel with negative prices should not crash."""
        from power_coupling.signals import SpreadConstructor
        sc = SpreadConstructor(price_panel)
        spread = sc.id1_spread().dropna()
        bt = WalkForwardBacktest(spread, train_months=6, test_months=3, n_bootstrap=50)
        result = bt.run(entry_z=1.0, exit_z=0.3)
        assert np.isfinite(result.sharpe)
