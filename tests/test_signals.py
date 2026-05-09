"""Tests for power_microstructure.signals modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_microstructure.signals import ForecastErrorSignal, SpreadConstructor


class TestForecastErrorSignal:
    def test_build_returns_expected_columns(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors)
        df = sig.build()
        assert "abs_total_error" in df.columns
        assert "error_direction" in df.columns
        assert (df["abs_total_error"] >= 0).all()

    def test_standardised_z_scores_near_zero_mean(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors, window=720)
        z = sig.standardised()
        # After the window burn-in, z-scores should have ~zero mean
        tail = z["total_error_z"].dropna().iloc[720:]
        assert abs(tail.mean()) < 0.2, f"Z-score mean too far from 0: {tail.mean()}"

    def test_cumulative_reset_daily(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors)
        cum = sig.cumulative()
        # First observation of each day should equal that hour's error
        daily_firsts = cum.groupby(cum.index.normalize()).first()
        daily_errors = forecast_errors["total_error"].groupby(forecast_errors.index.normalize()).first()
        common = daily_firsts.index.intersection(daily_errors.index)
        pd.testing.assert_series_equal(
            daily_firsts.loc[common].reset_index(drop=True),
            daily_errors.loc[common].reset_index(drop=True),
            check_names=False,
        )

    def test_uncertainty_is_positive(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors, window=168)
        unc = sig.uncertainty().dropna()
        assert (unc >= 0).all(), "Uncertainty (variance) must be non-negative."

    def test_regime_values_in_range(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors)
        regime = sig.regime(n_regimes=3).dropna()
        assert set(regime.unique()).issubset({0, 1, 2})

    def test_validation_raises_on_missing_columns(self, hourly_index):
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]}, index=hourly_index[:3])
        with pytest.raises(ValueError, match="Missing required columns"):
            ForecastErrorSignal(bad_df)

    def test_by_hour_of_day_has_24_entries(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors)
        hod = sig.by_hour_of_day()
        assert len(hod) == 24

    def test_direction_sign_consistency(self, forecast_errors):
        sig = ForecastErrorSignal(forecast_errors)
        df = sig.build()
        # error_direction should match sign of total_error
        nonzero = df[df["total_error"] != 0]
        assert (np.sign(nonzero["total_error"]) == nonzero["error_direction"]).all()


class TestSpreadConstructor:
    def test_id1_spread_shape_matches_panel(self, price_panel):
        sc = SpreadConstructor(price_panel)
        spread = sc.id1_spread()
        assert len(spread) == len(price_panel)
        assert spread.name == "id1_spread"

    def test_spread_panel_columns(self, price_panel):
        sc = SpreadConstructor(price_panel)
        panel = sc.spread_panel()
        assert set(panel.columns) == {"da_id_spread", "id1_spread", "id3_spread"}

    def test_shape_spread_requires_qh(self, price_panel):
        sc = SpreadConstructor(price_panel)
        with pytest.raises(ValueError, match="qh_prices must be provided"):
            sc.shape_spread()

    def test_shape_spread_length(self, price_panel, qh_prices):
        # Align indices: only the overlapping hourly index matters
        sc = SpreadConstructor(price_panel.iloc[:8760], qh_prices)
        ss = sc.shape_spread()
        assert len(ss) > 0
        assert ss.name == "shape_spread"

    def test_negative_price_flag_detects_negatives(self, price_panel):
        sc = SpreadConstructor(price_panel)
        flag = sc.negative_price_flag()
        # We injected ~3% negative prices in the fixture
        assert flag.sum() > 0, "Should detect at least some negative prices."
        assert flag.isin([0, 1]).all()

    def test_half_life_is_finite_for_mean_reverting(self, mean_reverting_spread):
        sc = SpreadConstructor(
            pd.DataFrame({
                "da_price": mean_reverting_spread,
                "id1_price": mean_reverting_spread + 0.1,
                "id3_price": mean_reverting_spread + 0.2,
                "id_continuous": mean_reverting_spread,
            })
        )
        hl = sc.half_life(mean_reverting_spread)
        assert np.isfinite(hl), "Half-life should be finite for mean-reverting spread."
        # Should be in a reasonable range (10–200 hours for our fixture with HL~50)
        assert 5 < hl < 500, f"Unexpected half-life: {hl}"

    def test_regime_conditional_variance_has_expected_regimes(self, price_panel, forecast_errors):
        sc = SpreadConstructor(price_panel)
        spread = sc.id1_spread()
        from power_microstructure.signals import ForecastErrorSignal
        sig = ForecastErrorSignal(forecast_errors)
        regime = sig.regime().reindex(spread.index)
        result = sc.regime_conditional_variance(spread, regime)
        assert len(result) <= 3  # at most 3 regimes

    def test_validation_raises_on_missing_columns(self, hourly_index):
        bad_df = pd.DataFrame({"da_price": [1.0]}, index=hourly_index[:1])
        with pytest.raises(ValueError, match="Missing required columns"):
            SpreadConstructor(bad_df)
