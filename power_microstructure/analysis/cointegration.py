"""
Cointegration and error correction analysis for intraday spread pairs.

Methodology
-----------
1. Unit root pre-tests: ADF + KPSS (both, as recommended by Kwiatkowski et al. 1992).
   ADF: H0 = unit root.  KPSS: H0 = stationary.
   Conclusion is drawn from the joint outcome:
     ADF reject + KPSS not-reject → stationary
     ADF not-reject + KPSS reject → unit root
     Both reject or neither → inconclusive (proceed with caution)

2. Johansen cointegration test (trace and max-eigenvalue statistics).
   Determines whether price pairs share a long-run equilibrium.

3. Vector Error Correction Model (VECM) estimation.
   Gives the speed of adjustment (alpha coefficients) — how quickly each
   series corrects toward the long-run equilibrium after a shock.

4. Half-life estimation via OLS on AR(1) spread:
   ΔS_t = β·S_{t-1} + ε_t  →  HL = −ln(2)/ln(1+β)

5. Spread stationarity test conditional on regime (high/low uncertainty).

References
----------
Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors..."
  Econometrica 59(6): 1551–1580.
Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and Error Correction."
  Econometrica 55(2): 251–276.
Kwiatkowski, D. et al. (1992). "Testing the null hypothesis of stationarity..."
  Journal of Econometrics 54(1–3): 159–178.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM


@dataclass
class StationarityResult:
    name: str
    adf_stat: float
    adf_pvalue: float
    kpss_stat: float
    kpss_pvalue: float
    adf_stationary: bool    # reject H0: unit root
    kpss_stationary: bool   # fail to reject H0: stationary
    conclusion: str         # "stationary", "unit_root", "inconclusive"
    adf_critical: dict
    kpss_critical: dict


@dataclass
class JohansenResult:
    n_cointegrating_vectors: int   # number of cointegration relations (at alpha)
    trace_stats: np.ndarray
    trace_crit_95: np.ndarray
    max_eig_stats: np.ndarray
    max_eig_crit_95: np.ndarray
    cointegrating_vector: np.ndarray   # normalized to first element = 1
    eigenvalues: np.ndarray


@dataclass
class VECMResult:
    alpha: np.ndarray              # adjustment speeds (n_variables,)
    beta: np.ndarray               # cointegrating vector
    half_life_hours: float
    aic: float
    bic: float
    n_obs: int
    alpha_tstat: np.ndarray
    alpha_pvalue: np.ndarray


class CointegrationAnalysis:
    """
    Full cointegration workflow for an intraday spread pair.

    Parameters
    ----------
    series1, series2 : pd.Series
        The two price series to test for cointegration (e.g. ID1 auction and
        intraday continuous prices for the same delivery hour).
        Must be UTC-aware DatetimeIndex, hourly frequency.
    alpha : float
        Significance level. Default 0.05.
    """

    def __init__(self, series1: pd.Series, series2: pd.Series, alpha: float = 0.05) -> None:
        self.alpha = alpha
        aligned = pd.concat([series1, series2], axis=1).dropna()
        self.s1 = aligned.iloc[:, 0]
        self.s2 = aligned.iloc[:, 1]
        self.n = len(aligned)
        self._data = aligned.values

    # ------------------------------------------------------------------
    # Step 1: Stationarity
    # ------------------------------------------------------------------

    def stationarity(self, series: pd.Series, name: str | None = None) -> StationarityResult:
        """Joint ADF + KPSS stationarity test."""
        name = name or str(series.name)
        s = series.dropna()

        adf_stat, adf_p, _, _, adf_crit, _ = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, _, kpss_crit = kpss(s, regression="c", nlags="auto")

        adf_stat_ok = adf_p < self.alpha
        kpss_stat_ok = kpss_p > self.alpha  # fail to reject stationarity null

        if adf_stat_ok and kpss_stat_ok:
            conclusion = "stationary"
        elif (not adf_stat_ok) and (not kpss_stat_ok):
            conclusion = "unit_root"
        else:
            conclusion = "inconclusive"

        return StationarityResult(
            name=name,
            adf_stat=float(adf_stat),
            adf_pvalue=float(adf_p),
            kpss_stat=float(kpss_stat),
            kpss_pvalue=float(kpss_p),
            adf_stationary=adf_stat_ok,
            kpss_stationary=kpss_stat_ok,
            conclusion=conclusion,
            adf_critical=adf_crit,
            kpss_critical=kpss_crit,
        )

    def stationarity_both(self) -> dict[str, StationarityResult]:
        return {
            self.s1.name: self.stationarity(self.s1, str(self.s1.name)),
            self.s2.name: self.stationarity(self.s2, str(self.s2.name)),
        }

    # ------------------------------------------------------------------
    # Step 2: Engle-Granger residual-based cointegration test
    # ------------------------------------------------------------------

    def engle_granger(self) -> dict:
        """
        Engle-Granger two-step cointegration test.
        Simple but less powerful than Johansen for n>2 series.
        """
        t_stat, p_value, crit_values = coint(self.s1.values, self.s2.values)
        return {
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "critical_values": crit_values,
            "cointegrated": p_value < self.alpha,
        }

    # ------------------------------------------------------------------
    # Step 3: Johansen cointegration test
    # ------------------------------------------------------------------

    def johansen(self, det_order: int = 0, k_ar_diff: int = 1) -> JohansenResult:
        """
        Johansen trace and max-eigenvalue cointegration test.

        Parameters
        ----------
        det_order : int
            -1 = no constant, 0 = restricted constant, 1 = unrestricted constant.
        k_ar_diff : int
            Number of lagged difference terms in the VAR.
        """
        result = coint_johansen(self._data, det_order=det_order, k_ar_diff=k_ar_diff)

        # Count cointegrating vectors at 95% critical value
        trace_stats = result.lr1
        trace_crit = result.cvt[:, 1]  # 95% column
        n_coint = int(np.sum(trace_stats > trace_crit))

        # Normalize cointegrating vector (first element = 1)
        beta = result.evec[:, 0]
        if beta[0] != 0:
            beta = beta / beta[0]

        return JohansenResult(
            n_cointegrating_vectors=n_coint,
            trace_stats=result.lr1,
            trace_crit_95=result.cvt[:, 1],
            max_eig_stats=result.lr2,
            max_eig_crit_95=result.cvm[:, 1],
            cointegrating_vector=beta,
            eigenvalues=result.eig,
        )

    # ------------------------------------------------------------------
    # Step 4: VECM estimation
    # ------------------------------------------------------------------

    def vecm(self, k_ar_diff: int = 1, coint_rank: int = 1) -> VECMResult:
        """
        Estimate Vector Error Correction Model.

        The alpha (adjustment speed) coefficients tell you:
        - How fast does ID1 auction price correct toward continuous?
        - How fast does continuous correct toward ID1 auction?

        A larger |alpha| = faster mean-reversion.
        """
        data = pd.DataFrame(self._data, columns=[str(self.s1.name), str(self.s2.name)])
        model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic="co")
        fit = model.fit()

        # Extract adjustment speeds and their t-stats
        alpha = fit.alpha.flatten()
        # VECM result object structure varies by statsmodels version
        try:
            se_alpha = fit.stderr_alpha.flatten()
            t_alpha = alpha / (se_alpha + 1e-12)
            from scipy import stats as sp_stats
            p_alpha = 2 * (1 - sp_stats.t.cdf(np.abs(t_alpha), df=fit.nobs - fit.k_exog))
        except AttributeError:
            t_alpha = np.full_like(alpha, float("nan"))
            p_alpha = np.full_like(alpha, float("nan"))

        # Half-life from spread AR(1)
        spread = self.s1.values - fit.beta[0, 0] * self.s2.values
        spread_series = pd.Series(spread, index=self.s1.index).dropna()
        half_life = self._half_life_ar1(spread_series)

        # Information criteria are not exposed on VECMResults in newer statsmodels.
        aic = float(getattr(fit, "aic", float("nan")))
        bic = float(getattr(fit, "bic", float("nan")))

        return VECMResult(
            alpha=alpha,
            beta=fit.beta.flatten(),
            half_life_hours=half_life,
            aic=aic,
            bic=bic,
            n_obs=fit.nobs,
            alpha_tstat=t_alpha,
            alpha_pvalue=p_alpha,
        )

    # ------------------------------------------------------------------
    # Step 5: Spread stationarity
    # ------------------------------------------------------------------

    def spread_stationarity(self, beta: float | None = None) -> StationarityResult:
        """
        Test stationarity of the linear spread: s1 − β·s2.

        If beta is None, it is estimated from OLS (s1 ~ s2).
        """
        if beta is None:
            from statsmodels.tools import add_constant as sm_add_const
            from statsmodels.regression.linear_model import OLS
            X = sm_add_const(self.s2.values)
            fit = OLS(self.s1.values, X).fit()
            beta = float(fit.params[1])

        spread = self.s1.values - beta * self.s2.values
        spread_series = pd.Series(spread, index=self.s1.index, name=f"spread_beta{beta:.3f}")
        return self.stationarity(spread_series)

    # ------------------------------------------------------------------
    # Half-life
    # ------------------------------------------------------------------

    def half_life(self, beta: float | None = None) -> float:
        """
        Mean-reversion half-life of the spread in hours.

        Model: ΔS_t = γ·S_{t-1} + ε_t,  HL = −ln(2)/ln(1+γ)
        """
        if beta is None:
            from statsmodels.tools import add_constant as sm_add_const
            from statsmodels.regression.linear_model import OLS
            X = sm_add_const(self.s2.values)
            fit = OLS(self.s1.values, X).fit()
            beta = float(fit.params[1])

        spread = pd.Series(self.s1.values - beta * self.s2.values, index=self.s1.index).dropna()
        return self._half_life_ar1(spread)

    # ------------------------------------------------------------------
    # Regime-conditional cointegration strength
    # ------------------------------------------------------------------

    def conditional_half_life(self, regime: pd.Series) -> dict[int, float]:
        """
        Compute half-life of spread conditional on regime.

        Returns {regime_value: half_life_hours}.
        """
        spread = self.s1 - self.s2  # unit cointegrating vector; refine if needed
        df = pd.DataFrame({"spread": spread, "regime": regime}).dropna()
        results: dict[int, float] = {}
        for r in sorted(df["regime"].unique()):
            sub = df.loc[df["regime"] == r, "spread"]
            if len(sub) < 50:
                continue
            results[int(r)] = self._half_life_ar1(sub)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _half_life_ar1(spread: pd.Series) -> float:
        ds = spread.diff().dropna()
        s_lag = spread.shift(1).dropna()
        common = ds.index.intersection(s_lag.index)
        X = np.column_stack([np.ones(len(common)), s_lag.loc[common].values])
        y = ds.loc[common].values
        gamma = np.linalg.lstsq(X, y, rcond=None)[0][1]
        if gamma >= 0:
            return float("inf")
        # gamma <= -1 means anti-persistent/oscillating or white noise;
        # log(1+gamma) would be log(0) or log(negative) — treat as HL=0.
        if gamma <= -1:
            return 0.0
        return float(-np.log(2) / np.log(1 + gamma))
