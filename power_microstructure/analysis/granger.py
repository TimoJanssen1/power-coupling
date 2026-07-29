"""
Granger causality analysis: renewable forecast errors → power prices.

(In this repo the price leg is the Belgian day-ahead series carried under the
legacy column name "id_continuous" — see FINDINGS.md "Revision notes".)

Methodology
-----------
1. Lag selection via AIC (Akaike Information Criterion) over a candidate range.
2. VAR(p) estimation on [forecast_error, price_change] pairs.
3. Granger F-test with Bonferroni correction for multiple delivery periods.
4. Impulse Response Functions (IRF) with bootstrap confidence bands.
5. Variance decomposition to quantify the fraction of price variance
   attributable to forecast error shocks.

All series are tested for stationarity (ADF) before VAR estimation.
If the level series is non-stationary, first differences are used and
this is flagged in the results.

Reference
---------
Granger, C.W.J. (1969). "Investigating Causal Relations by Econometric Models
and Cross-spectral Methods." Econometrica 37(3): 424–438.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

logger = logging.getLogger(__name__)


@dataclass
class GrangerResult:
    """Results for a single Granger causality test."""

    cause: str
    effect: str
    optimal_lag: int
    f_statistic: float
    p_value: float
    p_value_bonferroni: float
    n_obs: int
    aic_values: dict[int, float]
    reject_null: bool  # H0: cause does NOT Granger-cause effect


@dataclass
class IRFResult:
    """Impulse Response Function results."""

    periods: np.ndarray           # horizon (hours)
    irf: np.ndarray               # IRF point estimates (periods,)
    irf_lower: np.ndarray         # bootstrap 5th percentile
    irf_upper: np.ndarray         # bootstrap 95th percentile
    cause: str
    effect: str
    variance_decomposition: pd.DataFrame


class GrangerAnalysis:
    """
    Granger causality tests and IRF for renewable forecast error → price dynamics.

    Parameters
    ----------
    forecast_error : pd.Series
        Hourly total renewable forecast error (MWh).
    price_series : pd.Series
        Hourly price or price change (EUR/MWh).
    max_lag : int
        Maximum lag order to consider in AIC selection. Default 24 (one day).
    n_bootstrap : int
        Number of bootstrap replications for IRF confidence bands.
    alpha : float
        Significance level for hypothesis tests.
    """

    def __init__(
        self,
        forecast_error: pd.Series,
        price_series: pd.Series,
        max_lag: int = 24,
        n_bootstrap: int = 500,
        alpha: float = 0.05,
    ) -> None:
        self.max_lag = max_lag
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha

        aligned = pd.concat(
            [forecast_error.rename("fe"), price_series.rename("price")], axis=1
        ).dropna()
        self.fe = aligned["fe"]
        self.price = aligned["price"]
        self.n_obs = len(aligned)

        self._stationarity_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Stationarity pre-checks
    # ------------------------------------------------------------------

    def adf_test(self, series: pd.Series, name: str | None = None) -> dict:
        """ADF test. Returns dict with stat, p-value, lags, and stationarity verdict."""
        name = name or series.name or "series"
        if name in self._stationarity_cache:
            return self._stationarity_cache[name]
        adf_stat, p_val, lags_used, n_obs, critical_vals, _ = adfuller(
            series.dropna(), autolag="AIC"
        )
        result = {
            "name": name,
            "adf_stat": adf_stat,
            "p_value": p_val,
            "lags_used": lags_used,
            "n_obs": n_obs,
            "critical_values": critical_vals,
            "stationary": p_val < self.alpha,
        }
        self._stationarity_cache[name] = result
        return result

    def prepare_stationary(self) -> tuple[pd.Series, pd.Series, bool]:
        """
        Return (fe, price, differenced) where differenced=True if first differences
        were needed to achieve stationarity.
        """
        fe_adf = self.adf_test(self.fe, "forecast_error")
        price_adf = self.adf_test(self.price, "price")

        if fe_adf["stationary"] and price_adf["stationary"]:
            return self.fe, self.price, False

        # Use first differences if level is non-stationary
        fe_diff = self.fe.diff().dropna()
        price_diff = self.price.diff().dropna()
        common = fe_diff.index.intersection(price_diff.index)
        return fe_diff.loc[common], price_diff.loc[common], True

    # ------------------------------------------------------------------
    # AIC lag selection
    # ------------------------------------------------------------------

    def aic_lag_selection(self, fe: pd.Series, price: pd.Series) -> tuple[int, dict[int, float]]:
        """Select optimal VAR lag by AIC over range [1, max_lag]."""
        data = pd.concat([fe, price], axis=1).dropna().values
        aic_values: dict[int, float] = {}
        for lag in range(1, self.max_lag + 1):
            try:
                var_model = VAR(data)
                result = var_model.fit(lag, ic=None, trend="c")
                aic_values[lag] = result.aic
            except Exception:
                pass
        if not aic_values:
            raise RuntimeError("VAR estimation failed for all lag orders.")
        optimal_lag = min(aic_values, key=aic_values.get)
        return optimal_lag, aic_values

    # ------------------------------------------------------------------
    # Granger causality test
    # ------------------------------------------------------------------

    def test(self, n_tests: int = 1) -> GrangerResult:
        """
        Test: forecast_error Granger-causes price.

        Parameters
        ----------
        n_tests : int
            Total number of Granger tests being run (for Bonferroni correction).
            Set to the number of delivery periods tested if running multiple.
        """
        fe, price, differenced = self.prepare_stationary()
        optimal_lag, aic_values = self.aic_lag_selection(fe, price)

        data = pd.concat([fe.rename("fe"), price.rename("price")], axis=1).dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_results = grangercausalitytests(
                data[["price", "fe"]], maxlag=optimal_lag, verbose=False
            )

        # Extract F-stat from the optimal lag result
        f_stat, p_val, df_denom, df_num = gc_results[optimal_lag][0]["ssr_ftest"]
        p_bonferroni = min(p_val * n_tests, 1.0)

        return GrangerResult(
            cause="forecast_error",
            effect="price",
            optimal_lag=optimal_lag,
            f_statistic=float(f_stat),
            p_value=float(p_val),
            p_value_bonferroni=float(p_bonferroni),
            n_obs=len(data),
            aic_values=aic_values,
            reject_null=p_bonferroni < self.alpha,
        )

    def test_reverse(self, n_tests: int = 1) -> GrangerResult:
        """Test: price Granger-causes forecast_error (should be rejected — sanity check)."""
        fe, price, _ = self.prepare_stationary()
        optimal_lag, aic_values = self.aic_lag_selection(fe, price)

        data = pd.concat([fe.rename("fe"), price.rename("price")], axis=1).dropna()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gc_results = grangercausalitytests(
                data[["fe", "price"]], maxlag=optimal_lag, verbose=False
            )

        f_stat, p_val, _, _ = gc_results[optimal_lag][0]["ssr_ftest"]
        p_bonferroni = min(p_val * n_tests, 1.0)

        return GrangerResult(
            cause="price",
            effect="forecast_error",
            optimal_lag=optimal_lag,
            f_statistic=float(f_stat),
            p_value=float(p_val),
            p_value_bonferroni=float(p_bonferroni),
            n_obs=len(data),
            aic_values=aic_values,
            reject_null=p_bonferroni < self.alpha,
        )

    # ------------------------------------------------------------------
    # Impulse Response Functions
    # ------------------------------------------------------------------

    def irf(self, horizon: int = 24, seed: int = 42) -> IRFResult:
        """
        Compute IRF: response of price to a one-standard-deviation shock in forecast_error.

        Confidence bands come from a residual bootstrap (percentile method,
        5th/95th): estimated residuals are resampled i.i.d. with replacement,
        a synthetic sample of the SAME length as the data is rebuilt through
        the fitted VAR recursion (initialised with the actual first p
        observations), the VAR is re-estimated on it, and the orthogonalised
        IRF recomputed. Replications that fail to re-estimate are dropped and
        counted; if more than 5% fail, a RuntimeError is raised (no silent
        fallback to the point estimate).

        Parameters
        ----------
        horizon : int
            Number of periods (hours) ahead to trace the impulse response.
        seed : int
            Seed for the bootstrap resampling RNG.
        """
        fe, price, _ = self.prepare_stationary()
        optimal_lag, _ = self.aic_lag_selection(fe, price)
        data = pd.concat([fe, price], axis=1).dropna().values

        var_model = VAR(data)
        var_result = var_model.fit(optimal_lag, trend="c")

        # Point IRF (Cholesky-orthogonalized: fe first, price second)
        irf_obj = var_result.irf(horizon)
        # Shape: (horizon+1, 2, 2) — [period, variable_idx, shock_idx]
        # We want: response of price (idx 1) to shock in fe (idx 0)
        irf_point = irf_obj.orth_irfs[:, 1, 0]

        # Variance decomposition. statsmodels' FEVD .decomp shape varies by
        # version: it is either (neqs, periods, neqs) or (periods, neqs, neqs),
        # and "periods" is sometimes `horizon`, sometimes `horizon + 1`. Pick
        # the axes that put neqs (=2) on the equation/shock dimensions.
        fevd = var_result.fevd(horizon)
        decomp = np.asarray(fevd.decomp)
        price_decomp: np.ndarray
        if decomp.ndim == 3 and decomp.shape[0] == 2 and decomp.shape[2] == 2:
            # (neqs, periods, neqs) — price equation index = 1
            price_decomp = decomp[1, :, :]
        elif decomp.ndim == 3 and decomp.shape[1] == 2 and decomp.shape[2] == 2:
            # (periods, neqs, neqs)
            price_decomp = decomp[:, 1, :]
        else:
            price_decomp = np.full((horizon, 2), np.nan)
        vd_df = pd.DataFrame(
            price_decomp,
            columns=["from_fe", "from_price"],
            index=range(price_decomp.shape[0]),
        )

        # Residual bootstrap at the actual sample length.
        rng = np.random.default_rng(seed)
        T, k = data.shape
        p = optimal_lag
        residuals = var_result.resid - var_result.resid.mean(axis=0)  # centre
        n_resid = len(residuals)                                      # = T - p
        coefs = var_result.coefs                                      # (p, k, k)
        intercept = var_result.params[0]                              # (k,)

        # Simulate all replications jointly: Y has shape (B, T, k). The first p
        # observations are the actual data (common starting condition); the
        # recursion then applies the fitted VAR with resampled residuals.
        B = self.n_bootstrap
        Y = np.empty((B, T, k))
        Y[:, :p, :] = data[:p]
        draw = rng.integers(0, n_resid, size=(B, T - p))
        E = residuals[draw]  # (B, T-p, k)
        for t in range(p, T):
            acc = intercept + E[:, t - p, :]
            for i in range(1, p + 1):
                acc = acc + Y[:, t - i, :] @ coefs[i - 1].T
            Y[:, t, :] = acc

        boot_irfs = np.full((B, horizon + 1), np.nan)
        n_failed = 0
        for b in range(B):
            try:
                boot_var = VAR(Y[b]).fit(p, trend="c")
                boot_irfs[b] = boot_var.irf(horizon).orth_irfs[:, 1, 0]
            except Exception as exc:
                n_failed += 1
                logger.warning("IRF bootstrap replication %d failed: %s", b, exc)

        if n_failed > 0.05 * B:
            raise RuntimeError(
                f"IRF residual bootstrap: {n_failed}/{B} replications failed to "
                "re-estimate — confidence bands would be unreliable."
            )
        if n_failed:
            logger.warning(
                "IRF bootstrap: %d/%d replications failed and were dropped from the bands.",
                n_failed, B,
            )

        return IRFResult(
            periods=np.arange(horizon + 1),
            irf=irf_point,
            irf_lower=np.nanpercentile(boot_irfs, 5, axis=0),
            irf_upper=np.nanpercentile(boot_irfs, 95, axis=0),
            cause="forecast_error",
            effect="price",
            variance_decomposition=vd_df,
        )

    # ------------------------------------------------------------------
    # Rolling Granger (temporal stability)
    # ------------------------------------------------------------------

    def rolling_granger_pvalue(
        self,
        window: int = 8760,  # one year
        step: int = 168,     # step weekly
    ) -> pd.Series:
        """
        Rolling Granger p-value over time.

        Used to assess whether the forecast_error → price relationship has
        strengthened as German renewable penetration increased over the sample.
        Low p-values over time → relationship is structural, not spurious.
        """
        fe, price, _ = self.prepare_stationary()
        data = pd.concat([fe.rename("fe"), price.rename("price")], axis=1).dropna()
        dates = data.index[window::step]
        p_values: list[float] = []

        for end_loc in range(window, len(data), step):
            window_data = data.iloc[end_loc - window: end_loc]
            try:
                lag = min(self.max_lag, len(window_data) // 10)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gc = grangercausalitytests(
                        window_data[["price", "fe"]], maxlag=lag, verbose=False
                    )
                p_values.append(gc[lag][0]["ssr_ftest"][1])
            except Exception:
                p_values.append(float("nan"))

        return pd.Series(p_values, index=dates[: len(p_values)], name="rolling_granger_pvalue")

    # ------------------------------------------------------------------
    # Multi-period sweep (Bonferroni-corrected)
    # ------------------------------------------------------------------

    def test_multiple_periods(
        self,
        price_by_period: dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        Run Granger test for each delivery period in price_by_period.

        Applies Bonferroni correction across all tests.

        Parameters
        ----------
        price_by_period : dict mapping label → price series
            e.g. {"hour_10": series_10, "hour_11": series_11, ...}

        Returns
        -------
        DataFrame with one row per period, sorted by p-value.
        """
        n = len(price_by_period)
        rows = []
        for label, price_s in price_by_period.items():
            try:
                analyzer = GrangerAnalysis(
                    self.fe, price_s, self.max_lag, self.n_bootstrap, self.alpha
                )
                result = analyzer.test(n_tests=n)
                rows.append({
                    "period": label,
                    "optimal_lag": result.optimal_lag,
                    "f_statistic": result.f_statistic,
                    "p_value": result.p_value,
                    "p_value_bonferroni": result.p_value_bonferroni,
                    "reject_null": result.reject_null,
                    "n_obs": result.n_obs,
                })
            except Exception as e:
                rows.append({"period": label, "error": str(e)})

        return pd.DataFrame(rows).sort_values("p_value_bonferroni")
