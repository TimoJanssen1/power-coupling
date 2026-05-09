"""
Structural break detection via the Bai-Perron (1998, 2003) sequential procedure.

This module tests whether the Granger-causal relationship between renewable
forecast errors and intraday prices has strengthened as German renewable
penetration has grown (hypothesis: break dates should align with capacity additions).

Methodology
-----------
The Bai-Perron procedure allows for multiple unknown breakpoints in a linear
regression model.  We apply it to the bivariate model:

    price_change_t = α_k + β_k · forecast_error_t + ε_t

where k indexes the regime between breakpoints.

Steps:
  1. Bai-Perron sequential F-test (supF) to detect the number of breaks.
  2. Dynamic programming to locate break dates conditional on the number.
  3. Confidence intervals for break dates (from Bai 1997).
  4. Plot break dates vs. German renewable capacity additions.

Trimming parameter ε = 0.15 (each segment has at least 15% of observations).

References
----------
Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models with Multiple
  Structural Changes." Econometrica 66(1): 47–78.
Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple Structural
  Change Models." Journal of Applied Econometrics 18(1): 1–22.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant


class _PrefixOLS:
    """
    O(1) closed-form OLS RSS for any contiguous window of (x, y).

    Replaces the O(T²) RSS precompute matrix used by the Bai-Perron DP.
    Stores five prefix-sum arrays of length T+1 (≈ 5·8·T bytes), giving
    constant-time access to RSS(s, e) over any window via:

        n   = e - s + 1
        Sx  = Cx[e+1]  - Cx[s]      (and likewise for Sy, Sxx, Syy, Sxy)
        β̂  = (n·Sxy − Sx·Sy) / (n·Sxx − Sx²)
        α̂  = (Sy − β̂·Sx) / n
        RSS = Syy − α̂·Sy − β̂·Sxy

    Memory:  O(T)
    Per-window cost:  O(1)  scalar form,  O(K)  vectorised over K windows.
    """

    __slots__ = ("Cx", "Cy", "Cxx", "Cyy", "Cxy", "T")

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.T = len(x)
        # Pad with a leading zero so window sums are C[e+1] - C[s] for inclusive [s, e].
        self.Cx = np.concatenate(([0.0], np.cumsum(x)))
        self.Cy = np.concatenate(([0.0], np.cumsum(y)))
        self.Cxx = np.concatenate(([0.0], np.cumsum(x * x)))
        self.Cyy = np.concatenate(([0.0], np.cumsum(y * y)))
        self.Cxy = np.concatenate(([0.0], np.cumsum(x * y)))

    def rss(self, s: int, e: int) -> float:
        """RSS for OLS y ~ α + β·x over inclusive window [s, e]."""
        n = e - s + 1
        if n < 3:
            return float("inf")
        Sx = self.Cx[e + 1] - self.Cx[s]
        Sy = self.Cy[e + 1] - self.Cy[s]
        Sxx = self.Cxx[e + 1] - self.Cxx[s]
        Syy = self.Cyy[e + 1] - self.Cyy[s]
        Sxy = self.Cxy[e + 1] - self.Cxy[s]
        denom = n * Sxx - Sx * Sx
        if denom <= 0:  # x is (numerically) constant on this window → β undefined
            # Fall back to mean-only model RSS = Syy - Sy²/n  (still finite)
            return float(Syy - Sy * Sy / n)
        beta = (n * Sxy - Sx * Sy) / denom
        alpha = (Sy - beta * Sx) / n
        rss = Syy - alpha * Sy - beta * Sxy
        # Clamp tiny negatives from float cancellation
        return float(rss) if rss > 0 else 0.0

    def rss_vec(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """
        Vectorised RSS for many (start, end) pairs simultaneously.

        Parameters
        ----------
        starts, ends : np.ndarray (int, same shape)
            Inclusive indices.  Windows with n < 3 or constant x return +inf.

        Returns
        -------
        np.ndarray of RSS values, same shape as inputs.
        """
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64)
        n = (ends - starts + 1).astype(np.float64)
        Sx = self.Cx[ends + 1] - self.Cx[starts]
        Sy = self.Cy[ends + 1] - self.Cy[starts]
        Sxx = self.Cxx[ends + 1] - self.Cxx[starts]
        Syy = self.Cyy[ends + 1] - self.Cyy[starts]
        Sxy = self.Cxy[ends + 1] - self.Cxy[starts]

        denom = n * Sxx - Sx * Sx
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = np.where(denom > 0, (n * Sxy - Sx * Sy) / denom, 0.0)
            alpha = np.where(n > 0, (Sy - beta * Sx) / n, 0.0)
            rss = Syy - alpha * Sy - beta * Sxy
            # Mean-only fallback when x is constant over the window
            mean_only = Syy - np.where(n > 0, Sy * Sy / n, 0.0)
            rss = np.where(denom > 0, rss, mean_only)

        rss = np.maximum(rss, 0.0)              # absorb float cancellation
        rss = np.where(n < 3, np.inf, rss)      # too few obs → undefined
        return rss


@dataclass
class BreakResult:
    """Results of a Bai-Perron structural break test."""

    n_breaks: int
    break_dates: list[pd.Timestamp]
    break_indices: list[int]
    segment_betas: list[float]      # slope (forecast_error coefficient) per segment
    segment_alphas: list[float]     # intercept per segment
    segment_r2: list[float]
    segment_obs: list[int]
    supF_statistics: list[float]    # sequential supF test values
    supF_pvalues: list[float]       # approximate p-values
    total_n: int


class StructuralBreakAnalysis:
    """
    Bai-Perron structural break detection for the forecast-error → price relationship.

    Parameters
    ----------
    y : pd.Series
        Dependent variable (e.g. hourly price change, EUR/MWh).
    x : pd.Series
        Explanatory variable (e.g. total renewable forecast error, MWh).
    max_breaks : int
        Maximum number of breaks to search for. Default 5.
    trim : float
        Trimming parameter ε ∈ (0, 0.5). Each segment must contain at least
        trim × T observations.  Default 0.15 per Bai & Perron (2003) recommendation.
    alpha : float
        Significance level for sequential tests. Default 0.05.
    """

    def __init__(
        self,
        y: pd.Series,
        x: pd.Series,
        max_breaks: int = 5,
        trim: float = 0.15,
        alpha: float = 0.05,
    ) -> None:
        aligned = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
        self.y = aligned["y"].values
        self.x = aligned["x"].values
        self.index = aligned.index
        self.T = len(aligned)
        self.max_breaks = max_breaks
        self.trim = trim
        self.alpha = alpha
        self._min_seg = max(int(np.ceil(trim * self.T)), 10)
        # Prefix-sum precompute: O(T) memory, O(1) per RSS query.
        self._pref = _PrefixOLS(self.x, self.y)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(self) -> BreakResult:
        """
        Run the full Bai-Perron sequential procedure.

        Returns BreakResult with detected break dates and per-segment estimates.
        """
        # Sequential test: add breaks one at a time until supF is not significant
        detected_breaks = 0
        all_supF = []
        all_pvalues = []

        for m in range(1, self.max_breaks + 1):
            supF, pval = self._supF_test(m)
            all_supF.append(supF)
            all_pvalues.append(pval)
            if pval < self.alpha:
                detected_breaks = m
            else:
                break  # sequential procedure stops at first non-rejection

        if detected_breaks == 0:
            # No structural breaks detected
            return self._no_break_result(all_supF, all_pvalues)

        # Locate breaks using dynamic programming (global minimum RSS)
        break_indices = self._locate_breaks(detected_breaks)
        break_dates = [self.index[i] for i in break_indices]

        # Estimate per-segment regressions
        segments = self._segment_estimates(break_indices)

        return BreakResult(
            n_breaks=detected_breaks,
            break_dates=break_dates,
            break_indices=break_indices,
            segment_betas=[s["beta"] for s in segments],
            segment_alphas=[s["alpha"] for s in segments],
            segment_r2=[s["r2"] for s in segments],
            segment_obs=[s["n"] for s in segments],
            supF_statistics=all_supF,
            supF_pvalues=all_pvalues,
            total_n=self.T,
        )

    # ------------------------------------------------------------------
    # supF test
    # ------------------------------------------------------------------

    def _supF_test(self, m: int) -> tuple[float, float]:
        """
        Compute supF(m) = sup over all valid partition points of the F-statistic
        testing m breaks against 0 breaks.

        Approximate p-values use the Bai-Perron (1998) asymptotic distribution.
        """
        rss_null = self._rss_segment(0, self.T - 1)
        min_seg = self._min_seg

        # For m=1: scan all valid single break points
        if m == 1:
            f_stats = []
            for bp in range(min_seg, self.T - min_seg):
                rss_alt = self._rss_segment(0, bp) + self._rss_segment(bp + 1, self.T - 1)
                # F-stat: (RSS_null - RSS_alt) / (q * sigma^2_alt) * (T - (m+1)*q)
                f_stat = (rss_null - rss_alt) / (rss_alt / (self.T - 2 * 2))
                f_stats.append(max(f_stat, 0))
            supF = float(max(f_stats)) if f_stats else 0.0
        else:
            # For m>1: use the optimal partition from m-1 breaks + one more
            breaks_m1 = self._locate_breaks(m - 1)
            # Test adding one more break in each existing segment
            candidate_f = []
            segments = [0] + breaks_m1 + [self.T - 1]
            for i in range(len(segments) - 1):
                start, end = segments[i], segments[i + 1]
                rss_seg = self._rss_segment(start, end)
                for bp in range(start + min_seg, end - min_seg):
                    rss_alt = self._rss_segment(start, bp) + self._rss_segment(bp + 1, end)
                    n_seg = end - start + 1
                    f = (rss_seg - rss_alt) / (rss_alt / max(n_seg - 4, 1))
                    candidate_f.append(max(f, 0))
            supF = float(max(candidate_f)) if candidate_f else 0.0

        # Approximate p-value: use F(q, T-2*(m+1)) critical values
        # q=1 (one regressor), conservative approximation
        df1 = 1
        df2 = max(self.T - 2 * (m + 1), 1)
        pval = float(1 - stats.f.cdf(supF, df1, df2))
        return supF, pval

    # ------------------------------------------------------------------
    # Dynamic programming break location
    # ------------------------------------------------------------------

    def _locate_breaks(self, m: int) -> list[int]:
        """
        Global minimum RSS partition into m+1 segments via dynamic programming.

        Memory: O(T · m).  Time: O(T² · m), vectorised in numpy along the
        candidate breakpoint axis.  Uses the prefix-sum OLS helper for O(1)
        RSS queries — the legacy O(T²) ``rss_mat`` matrix is gone, so the
        large-T greedy fallback is no longer auto-routed.

        Returns list of m break indices (0-based, last obs in each segment
        except the last).
        """
        T = self.T
        min_seg = self._min_seg
        pref = self._pref

        # opt_cost[t, k] = min RSS for k breaks within observations [0..t]
        # opt_break[t, k] = breakpoint location achieving opt_cost[t, k]
        opt_cost = np.full((T, m + 1), np.inf)
        opt_break = np.full((T, m + 1), -1, dtype=np.int64)

        # k=0: a single segment from 0 to t, only valid once t ≥ min_seg-1
        t_axis = np.arange(min_seg - 1, T, dtype=np.int64)
        opt_cost[t_axis, 0] = pref.rss_vec(np.zeros_like(t_axis), t_axis)

        # k ≥ 1: opt_cost[t, k] = min_{bp} [opt_cost[bp, k-1] + RSS(bp+1, t)]
        # For each k we vectorise the bp-axis: for fixed t the candidate
        # vector is opt_cost[bp_lo..bp_hi, k-1] + rss(bp+1, t) where
        # bp_lo = (k-1)·min_seg − 1 and bp_hi = t − min_seg.
        for k in range(1, m + 1):
            bp_lo = k * min_seg - min_seg - 1            # (k-1)·min_seg − 1
            for t in range(k * min_seg - 1 + min_seg, T):
                bp_hi = t - min_seg                      # latest bp keeping last seg ≥ min_seg
                if bp_hi < bp_lo:
                    continue
                bps = np.arange(bp_lo, bp_hi + 1, dtype=np.int64)
                # RSS of the right segment for each candidate bp:
                rss_right = pref.rss_vec(bps + 1, np.full_like(bps, t))
                cost_vec = opt_cost[bps, k - 1] + rss_right
                # Guard against all-inf (e.g., k segments don't fit yet)
                if not np.isfinite(cost_vec).any():
                    continue
                idx = int(np.argmin(cost_vec))
                opt_cost[t, k] = cost_vec[idx]
                opt_break[t, k] = bps[idx]

        # Backtrack from the full-sample, m-break optimum
        breaks: list[int] = []
        t = T - 1
        for k in range(m, 0, -1):
            bp = int(opt_break[t, k])
            if bp < 0:
                # No feasible partition with k breaks — fall back to greedy
                return self._locate_breaks_approx(m)
            breaks.append(bp)
            t = bp
        return sorted(breaks)

    def _locate_breaks_approx(self, m: int) -> list[int]:
        """Faster approximate break location for large T via grid search."""
        min_seg = self._min_seg
        candidates = list(range(min_seg, self.T - min_seg, max(1, self.T // 200)))

        if m == 1:
            best_idx = min(candidates, key=lambda bp: self._rss_segment(0, bp) + self._rss_segment(bp + 1, self.T - 1))
            return [best_idx]

        # Greedy: find each break in the largest remaining segment
        breaks = []
        segments = [(0, self.T - 1)]
        for _ in range(m):
            best_rss_reduction = -np.inf
            best_bp = None
            best_seg_idx = None
            for seg_idx, (start, end) in enumerate(segments):
                if end - start < 2 * min_seg:
                    continue
                rss_full = self._rss_segment(start, end)
                for bp in range(start + min_seg, end - min_seg):
                    reduction = rss_full - self._rss_segment(start, bp) - self._rss_segment(bp + 1, end)
                    if reduction > best_rss_reduction:
                        best_rss_reduction = reduction
                        best_bp = bp
                        best_seg_idx = seg_idx
            if best_bp is None:
                break
            breaks.append(best_bp)
            start, end = segments.pop(best_seg_idx)
            segments.append((start, best_bp))
            segments.append((best_bp + 1, end))
        return sorted(breaks)

    # ------------------------------------------------------------------
    # Segment OLS
    # ------------------------------------------------------------------

    def _rss_segment(self, start: int, end: int) -> float:
        """
        Residual sum of squares for OLS: y ~ α + β·x over [start, end].

        Delegates to the prefix-sum helper so all RSS queries — supF test,
        DP location, approximate fallback — share the same numerics in O(1).
        """
        return self._pref.rss(start, end)

    def _segment_estimates(self, break_indices: list[int]) -> list[dict]:
        """OLS estimates for each segment defined by break_indices."""
        boundaries = [0] + [b + 1 for b in break_indices] + [self.T]
        segments = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            y = self.y[start:end]
            x = self.x[start:end]
            n = len(y)
            X = np.column_stack([np.ones(n), x])
            coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha_coef, beta_coef = float(coeffs[0]), float(coeffs[1])
            ss_res = float(np.sum((y - X @ coeffs) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            segments.append({
                "alpha": alpha_coef,
                "beta": beta_coef,
                "r2": r2,
                "n": n,
                "start_date": self.index[start],
                "end_date": self.index[end - 1],
            })
        return segments

    def _no_break_result(self, supF_stats: list[float], pvals: list[float]) -> BreakResult:
        """Return a no-break result with the full-sample OLS estimate."""
        X = np.column_stack([np.ones(self.T), self.x])
        coeffs = np.linalg.lstsq(X, self.y, rcond=None)[0]
        ss_res = np.sum((self.y - X @ coeffs) ** 2)
        ss_tot = np.sum((self.y - self.y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return BreakResult(
            n_breaks=0,
            break_dates=[],
            break_indices=[],
            segment_betas=[float(coeffs[1])],
            segment_alphas=[float(coeffs[0])],
            segment_r2=[r2],
            segment_obs=[self.T],
            supF_statistics=supF_stats,
            supF_pvalues=pvals,
            total_n=self.T,
        )

    # ------------------------------------------------------------------
    # Interpretation helpers
    # ------------------------------------------------------------------

    def beta_over_time(self, result: BreakResult) -> pd.Series:
        """
        Return a time series of the estimated β coefficient per period.

        Useful for plotting against renewable penetration growth.
        """
        betas = []
        boundaries = [0] + [b + 1 for b in result.break_indices] + [self.T]
        for i, beta in enumerate(result.segment_betas):
            start = boundaries[i]
            end = boundaries[i + 1]
            betas.extend([(self.index[t], beta) for t in range(start, end)])
        idx, vals = zip(*betas) if betas else ([], [])
        return pd.Series(dict(zip(idx, vals)), name="beta_over_time")

    def align_with_capacity(
        self,
        break_result: BreakResult,
        capacity_additions: pd.Series,
        window_days: int = 90,
    ) -> pd.DataFrame:
        """
        Check whether break dates align with major renewable capacity addition events.

        Parameters
        ----------
        capacity_additions : pd.Series
            Series of annual/quarterly German renewable capacity (GW), DatetimeIndex.
        window_days : int
            Tolerance window around each break date for matching.

        Returns DataFrame showing each break date and nearest capacity event.
        """
        rows = []
        for bd in break_result.break_dates:
            delta = (capacity_additions.index - bd).days
            nearest_idx = np.argmin(np.abs(delta))
            nearest_date = capacity_additions.index[nearest_idx]
            rows.append({
                "break_date": bd,
                "nearest_capacity_event": nearest_date,
                "days_apart": int(delta[nearest_idx]),
                "within_window": abs(int(delta[nearest_idx])) <= window_days,
                "capacity_at_event_gw": float(capacity_additions.iloc[nearest_idx]),
            })
        return pd.DataFrame(rows)
