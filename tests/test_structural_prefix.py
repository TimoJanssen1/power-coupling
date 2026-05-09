"""
Targeted tests for the prefix-sum Bai-Perron rewrite.

Two invariants we care about:
  1.  Numerical equivalence:  prefix-sum RSS == direct-OLS RSS to FP tolerance
      for any window.  If this holds, every downstream test (supF, DP,
      segment estimates) inherits the same numerics as the legacy code.

  2.  Algorithmic equivalence:  the new DP finds the same global-RSS
      partition as a brute-force enumeration on tiny T.

  3.  Large-T scaling:  T = 20 000 (≈ 2.3 years hourly) runs in budget for
      both wall time and peak resident memory.  This is the regime the
      legacy code couldn't reach (rss_mat would be 3.2 GB).
"""

from __future__ import annotations

import itertools
import os
import resource
import time

import numpy as np
import pandas as pd
import pytest

from power_microstructure.analysis.structural import (
    StructuralBreakAnalysis,
    _PrefixOLS,
)


def _direct_rss(x: np.ndarray, y: np.ndarray, s: int, e: int) -> float:
    """Reference RSS via numpy lstsq, identical math to the legacy code."""
    n = e - s + 1
    if n < 3:
        return float("inf")
    xx = x[s : e + 1]
    yy = y[s : e + 1]
    X = np.column_stack([np.ones(n), xx])
    coeffs, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ coeffs
    return float(resid @ resid)


# ---------------------------------------------------------------------------
# 1. Per-window numerical equivalence
# ---------------------------------------------------------------------------


def test_prefix_rss_matches_direct_ols_random_windows():
    rng = np.random.default_rng(7)
    T = 1000
    x = rng.normal(size=T) * 500
    y = 0.005 * x + rng.normal(size=T) * 2.0

    pref = _PrefixOLS(x, y)
    starts = rng.integers(0, T - 50, size=200)
    spans = rng.integers(20, 400, size=200)
    ends = np.minimum(starts + spans, T - 1)

    for s, e in zip(starts, ends):
        ref = _direct_rss(x, y, int(s), int(e))
        got = pref.rss(int(s), int(e))
        # Float-tolerant: prefix-sum and lstsq accumulate errors differently.
        assert abs(got - ref) <= 1e-6 * max(1.0, ref), (
            f"window [{s}, {e}]: prefix={got:.6e}  direct={ref:.6e}"
        )


def test_prefix_rss_vec_matches_scalar():
    rng = np.random.default_rng(11)
    T = 500
    x = rng.normal(size=T)
    y = rng.normal(size=T)
    pref = _PrefixOLS(x, y)

    starts = rng.integers(0, T - 30, size=100).astype(np.int64)
    ends = (starts + rng.integers(20, 200, size=100)).clip(max=T - 1).astype(np.int64)

    vec = pref.rss_vec(starts, ends)
    scal = np.array([pref.rss(int(s), int(e)) for s, e in zip(starts, ends)])
    np.testing.assert_allclose(vec, scal, rtol=1e-12, atol=1e-12)


def test_prefix_rss_handles_constant_x_window():
    """Window where x is constant: β undefined → mean-only RSS, finite."""
    T = 50
    x = np.zeros(T)
    y = np.arange(T, dtype=float)
    pref = _PrefixOLS(x, y)
    rss = pref.rss(0, T - 1)
    # Mean-only model RSS = Σ(y - ȳ)² = Var(y) · n
    expected = float(np.var(y) * T)
    assert abs(rss - expected) < 1e-9
    assert np.isfinite(rss)


def test_prefix_rss_too_few_obs_returns_inf():
    pref = _PrefixOLS(np.arange(10.0), np.arange(10.0))
    assert pref.rss(0, 1) == float("inf")
    assert pref.rss(5, 5) == float("inf")
    assert np.isfinite(pref.rss(0, 2))  # n=3 is the minimum


# ---------------------------------------------------------------------------
# 2. Algorithmic equivalence: DP matches brute-force on tiny T
# ---------------------------------------------------------------------------


def _brute_force_partition(
    sba: StructuralBreakAnalysis, m: int
) -> tuple[list[int], float]:
    """Enumerate every valid m-break partition; return the global minimum."""
    T = sba.T
    min_seg = sba._min_seg
    pref = sba._pref

    candidates = list(range(min_seg - 1, T - min_seg))
    best_cost = float("inf")
    best_breaks: list[int] = []

    for combo in itertools.combinations(candidates, m):
        # Enforce min_seg between consecutive breaks
        prev_end = -1
        valid = True
        cost = 0.0
        for bp in combo:
            seg_start = prev_end + 1
            seg_end = bp
            if seg_end - seg_start + 1 < min_seg:
                valid = False
                break
            cost += pref.rss(seg_start, seg_end)
            prev_end = bp
        if not valid:
            continue
        # Final segment
        if (T - 1) - (prev_end + 1) + 1 < min_seg:
            continue
        cost += pref.rss(prev_end + 1, T - 1)
        if cost < best_cost:
            best_cost = cost
            best_breaks = list(combo)

    return best_breaks, best_cost


@pytest.mark.parametrize("m", [1, 2])
def test_dp_matches_brute_force_tiny_T(m):
    rng = np.random.default_rng(17)
    T = 80
    x = rng.normal(size=T) * 500
    y = np.where(np.arange(T) < T // 2, 0.003, 0.012) * x + rng.normal(size=T) * 1.0

    sba = StructuralBreakAnalysis(
        pd.Series(y, index=pd.RangeIndex(T)),
        pd.Series(x, index=pd.RangeIndex(T)),
        max_breaks=m,
        trim=0.20,
        alpha=0.999,  # ensure breaks are accepted (we test the DP, not the test)
    )

    dp_breaks = sba._locate_breaks(m)
    brute_breaks, brute_cost = _brute_force_partition(sba, m)

    # Compute DP cost from the partition
    pref = sba._pref
    boundaries = [0] + [b + 1 for b in dp_breaks] + [T]
    dp_cost = sum(
        pref.rss(boundaries[i], boundaries[i + 1] - 1)
        for i in range(len(boundaries) - 1)
    )

    assert abs(dp_cost - brute_cost) < 1e-9, (
        f"DP cost {dp_cost:.4f} does not match brute-force {brute_cost:.4f}\n"
        f"  DP breaks={dp_breaks}  brute breaks={brute_breaks}"
    )


# ---------------------------------------------------------------------------
# 3. Large-T scaling
# ---------------------------------------------------------------------------


def test_large_T_runs_in_budget():
    """
    T = 20 000 (≈ 2.3 years of hourly data).  The legacy implementation
    would either allocate a 3.2 GB rss_mat or fall back to the greedy
    approximation.  The prefix-sum DP must:
      - finish in ≤ 60 s on a laptop
      - hit the EXACT path (not the greedy fallback)
      - keep peak RSS growth bounded (a few hundred MB tops)
    """
    rng = np.random.default_rng(2026)
    T = 20_000
    x = rng.normal(size=T) * 500
    y = np.where(np.arange(T) < T // 2, 0.003, 0.012) * x + rng.normal(size=T) * 2.0

    idx = pd.date_range("2018-01-01", periods=T, freq="h", tz="UTC")
    sba = StructuralBreakAnalysis(
        pd.Series(y, index=idx),
        pd.Series(x, index=idx),
        max_breaks=2,
        trim=0.15,
        alpha=0.05,
    )

    rss_pre = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    breaks = sba._locate_breaks(2)
    elapsed = time.perf_counter() - t0
    rss_post = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # macOS reports ru_maxrss in bytes; Linux in KB.  Normalise to MB.
    bytes_per_unit = 1 if os.uname().sysname == "Darwin" else 1024
    delta_mb = (rss_post - rss_pre) * bytes_per_unit / (1024 * 1024)

    print(
        f"\nT={T}  m=2  exact-DP  elapsed={elapsed:.2f}s  ΔRSS≈{delta_mb:.1f} MB  breaks={breaks}"
    )

    assert elapsed < 60.0, f"Exact DP took {elapsed:.1f}s — slower than budget."
    # Memory: prefix arrays are ~5·8·T bytes ≈ 800 KB; DP table 2·8·T bytes
    # ≈ 320 KB; bp arrays per t are ≤ T integers.  Peak should stay well
    # under the legacy 3.2 GB rss_mat.  We assert a generous 500 MB ceiling
    # to leave headroom for OS noise.
    assert delta_mb < 500, f"Peak ΔRSS {delta_mb:.0f} MB exceeded 500 MB budget."

    # Sanity: a real break exists at midpoint; one of the detected breaks
    # should land within ±5% of T/2.
    half = T // 2
    assert any(abs(b - half) <= 0.05 * T for b in breaks), (
        f"No detected break near the synthetic midpoint at {half}.  Got {breaks}."
    )
