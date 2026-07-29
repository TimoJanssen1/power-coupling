"""Tests for the IRF residual bootstrap in analysis.granger."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from power_microstructure.analysis import GrangerAnalysis


@pytest.fixture
def var_pair():
    """
    Synthetic bivariate system with a known lagged effect:

        fe_t    = white noise
        price_t = 0.5 * price_{t-1} + 0.3 * fe_{t-1} + noise

    Small n and few bootstrap reps so the test stays fast.
    """
    rng = np.random.default_rng(7)
    n = 600
    fe = rng.normal(0, 1.0, size=n)
    price = np.zeros(n)
    noise = rng.normal(0, 0.5, size=n)
    for t in range(1, n):
        price[t] = 0.5 * price[t - 1] + 0.3 * fe[t - 1] + noise[t]
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.Series(fe, index=idx, name="fe"), pd.Series(price, index=idx, name="price")


class TestIRFResidualBootstrap:
    def test_bands_finite_and_bracket_point_estimate(self, var_pair):
        fe, price = var_pair
        ga = GrangerAnalysis(fe, price, max_lag=3, n_bootstrap=40)
        res = ga.irf(horizon=8, seed=123)

        assert np.isfinite(res.irf).all()
        assert np.isfinite(res.irf_lower).all()
        assert np.isfinite(res.irf_upper).all()
        # Percentile bands from re-estimated IRFs should bracket the point IRF
        assert (res.irf_lower <= res.irf + 1e-12).all(), (
            f"lower band exceeds point IRF: {res.irf_lower - res.irf}"
        )
        assert (res.irf_upper >= res.irf - 1e-12).all(), (
            f"upper band below point IRF: {res.irf - res.irf_upper}"
        )
        # Bands must have positive width (a degenerate bootstrap would collapse
        # onto the point estimate — the old silent-fallback failure mode)
        assert (res.irf_upper - res.irf_lower).max() > 1e-6

    def test_known_effect_detected_at_lag_one(self, var_pair):
        fe, price = var_pair
        ga = GrangerAnalysis(fe, price, max_lag=3, n_bootstrap=40)
        res = ga.irf(horizon=8, seed=123)
        # By construction price responds positively to fe with a one-period lag,
        # and the effect is strong enough that the 90% band excludes zero there.
        assert res.irf[1] > 0
        assert res.irf_lower[1] > 0

    def test_different_seed_gives_different_bands(self, var_pair):
        fe, price = var_pair
        ga = GrangerAnalysis(fe, price, max_lag=3, n_bootstrap=25)
        res_a = ga.irf(horizon=6, seed=1)
        res_b = ga.irf(horizon=6, seed=2)
        # Same point estimate (deterministic fit) …
        np.testing.assert_allclose(res_a.irf, res_b.irf)
        # … but resampling-dependent bands
        assert not np.allclose(res_a.irf_lower, res_b.irf_lower)
        assert not np.allclose(res_a.irf_upper, res_b.irf_upper)

    def test_same_seed_reproducible(self, var_pair):
        fe, price = var_pair
        ga = GrangerAnalysis(fe, price, max_lag=3, n_bootstrap=25)
        res_a = ga.irf(horizon=6, seed=99)
        res_b = ga.irf(horizon=6, seed=99)
        np.testing.assert_allclose(res_a.irf_lower, res_b.irf_lower)
        np.testing.assert_allclose(res_a.irf_upper, res_b.irf_upper)
