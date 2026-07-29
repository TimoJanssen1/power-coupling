"""
Battery energy-storage system (BESS) dispatch simulator.

Models a utility-scale BESS (~100 MW / 200 MWh, 2-hour duration) doing energy
arbitrage in the German day-ahead auction. Solves the dispatch problem as a
daily linear program over a 24-hour horizon, parameterised by:

  • the price vector the LP optimises against
  • the price vector cash flows settle at
  • the cycling, efficiency and wear constraints
  • the boundary state-of-charge between days

Settlement convention (July 2026 revision): strategies that schedule against
the DE/LU day-ahead price also SETTLE at the DE/LU day-ahead price — the
schedule is committed in the DA auction and clears there. v1 settled all
strategies at the series then believed to be the German intraday continuous
index; that series is in fact the Belgian day-ahead price (see FINDINGS.md
"Revision notes"), so v1's revenue was a DE-scheduled/BE-settled hybrid.

Four strategies share the same simulator with different inputs:

    naive          Hour-of-day heuristic (charge the day's cheapest 2h,
                   discharge the richest 2h), processed chronologically so
                   energy cannot be discharged before it has been charged.
    da_lp          LP against DE/LU DA prices, settled at DE/LU DA — what a
                   baseline DA-auction arbitrage operator does. DA prices are
                   known before gate closure, so this is implementable.
    da_plus_tilt   LP against DA prices tilted by the Q1 forecast-error impact,
                   settled at DA. The tilt uses realised forecast errors (not
                   knowable at gate closure), so it is an UPPER BOUND on what
                   such a signal could add — and it adds ~nothing.
    alt_series_lp  LP optimised AND settled on the alternative price series
                   passed to it. With this repo's data that series is the
                   Belgian DA price (mislabelled "intraday" in v1) — retained
                   as a cross-zonal reference point, not a foresight ceiling.

All values are EUR. Energy is MWh. Power is MW. Time is hourly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# ---------------------------------------------------------------------------
# Battery spec
# ---------------------------------------------------------------------------


@dataclass
class BatterySpec:
    """Physical and operational parameters of the BESS."""

    power_mw: float = 100.0          # rated charge/discharge power
    energy_mwh: float = 200.0        # usable energy capacity
    eta_round_trip: float = 0.85     # round-trip efficiency
    wear_eur_per_mwh: float = 2.0    # variable degradation cost per MWh discharged
    max_cycles_per_day: float = 1.5  # warranty-driven cycle cap (× energy_mwh / day)
    soc_init: float = 100.0          # initial state of charge (MWh)

    @property
    def eta_charge(self) -> float:
        return float(np.sqrt(self.eta_round_trip))

    @property
    def eta_discharge(self) -> float:
        return float(np.sqrt(self.eta_round_trip))


# ---------------------------------------------------------------------------
# LP-based 24-hour dispatch
# ---------------------------------------------------------------------------


def _solve_day_lp(
    prices: np.ndarray,
    spec: BatterySpec,
    soc_start: float,
    soc_end_target: float | None = None,
) -> dict[str, np.ndarray] | None:
    """
    Solve the daily dispatch LP for a 24-hour price vector.

    Variables (in order, length 3·H):
        c_t  ∈ [0, P]   charge MWh in hour t  (energy delivered to grid → battery)
        d_t  ∈ [0, P]   discharge MWh in hour t (energy delivered to grid)
        s_t  ∈ [0, E]   state of charge at end of hour t

    Objective (maximise revenue, so linprog minimises negative):
        Σ_t  p_t·d_t  −  p_t·c_t  −  wear·d_t

    SoC dynamics:
        s_t = s_{t-1} + η_c · c_t − d_t / η_d

    Cycle cap:
        Σ_t d_t  ≤  max_cycles_per_day · E

    Optional soft endpoint:
        s_{H-1} = soc_end_target   (energy-neutral day if requested)
    """
    H = len(prices)
    P = spec.power_mw
    E = spec.energy_mwh
    eta_c = spec.eta_charge
    eta_d = spec.eta_discharge
    wear = spec.wear_eur_per_mwh

    nvar = 3 * H  # [c_0..c_{H-1}, d_0..d_{H-1}, s_0..s_{H-1}]
    idx_c = lambda t: t                  # noqa: E731
    idx_d = lambda t: H + t              # noqa: E731
    idx_s = lambda t: 2 * H + t          # noqa: E731

    # Objective: maximise Σ p·d − p·c − wear·d  ⇒ minimise -obj
    c_obj = np.zeros(nvar)
    for t in range(H):
        c_obj[idx_c(t)] = prices[t]            # cost of charging
        c_obj[idx_d(t)] = -prices[t] + wear    # negative revenue + wear cost

    # Bounds
    bounds = (
        [(0.0, P)] * H            # c
        + [(0.0, P)] * H          # d
        + [(0.0, E)] * H          # s
    )

    # Equality constraints: SoC dynamics
    A_eq = np.zeros((H, nvar))
    b_eq = np.zeros(H)
    for t in range(H):
        # s_t − η_c·c_t + d_t/η_d − s_{t-1} = 0
        A_eq[t, idx_s(t)] = 1.0
        A_eq[t, idx_c(t)] = -eta_c
        A_eq[t, idx_d(t)] = 1.0 / eta_d
        if t == 0:
            b_eq[t] = soc_start
        else:
            A_eq[t, idx_s(t - 1)] = -1.0

    if soc_end_target is not None:
        # Append an extra equality s_{H-1} = soc_end_target
        extra = np.zeros((1, nvar))
        extra[0, idx_s(H - 1)] = 1.0
        A_eq = np.vstack([A_eq, extra])
        b_eq = np.concatenate([b_eq, [soc_end_target]])

    # Inequality constraint: cycle cap on total discharge
    A_ub = np.zeros((1, nvar))
    for t in range(H):
        A_ub[0, idx_d(t)] = 1.0
    b_ub = np.array([spec.max_cycles_per_day * E])

    res = linprog(
        c=c_obj,
        A_eq=A_eq, b_eq=b_eq,
        A_ub=A_ub, b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return None

    x = res.x
    return {
        "charge": x[:H],
        "discharge": x[H : 2 * H],
        "soc": x[2 * H : 3 * H],
        "objective_eur": -res.fun,  # we minimised −obj
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    schedule: pd.DataFrame   # columns: charge_mwh, discharge_mwh, soc_mwh, price_da, price_id
    spec: BatterySpec
    name: str

    @property
    def revenue_eur(self) -> float:
        s = self.schedule
        # Cash flows clear at the "settlement_price" column, which each
        # strategy sets to the market its schedule is actually committed in
        # (DA-scheduled strategies settle at DA; alt_series_lp settles on the
        # series it optimises).
        rev = (s["discharge_mwh"] * s["settlement_price"]).sum()
        cost = (s["charge_mwh"] * s["settlement_price"]).sum()
        wear = self.spec.wear_eur_per_mwh * s["discharge_mwh"].sum()
        return float(rev - cost - wear)

    @property
    def total_throughput_mwh(self) -> float:
        return float(self.schedule["discharge_mwh"].sum())

    @property
    def n_full_cycles(self) -> float:
        return self.total_throughput_mwh / self.spec.energy_mwh

    @property
    def revenue_per_mwh_throughput(self) -> float:
        t = self.total_throughput_mwh
        return self.revenue_eur / t if t > 0 else 0.0

    @property
    def revenue_per_full_cycle(self) -> float:
        n = self.n_full_cycles
        return self.revenue_eur / n if n > 0 else 0.0


# -- Naive heuristic ---------------------------------------------------------


def strategy_naive(
    da_prices: pd.Series,
    id_prices: pd.Series,
    spec: BatterySpec,
) -> DispatchResult:
    """
    Charge during the cheapest 2 hours of each day, discharge during the most
    expensive 2 hours of the same day. No optimisation, no foresight beyond
    the calendar day. Settles at the DA price it schedules against.

    Hours are processed CHRONOLOGICALLY: a "richest" hour that falls before
    the day's "cheapest" hours can only discharge energy already in storage
    at that time — energy cannot be discharged before it has been charged.
    (v1 charged all cheap hours first regardless of their position in the day,
    which allowed physically impossible discharge-before-charge schedules.)

    ``id_prices`` is recorded on the schedule for reference only.
    """
    da_prices = da_prices.dropna()
    id_prices = id_prices.reindex(da_prices.index).interpolate()

    rows = []
    soc = spec.soc_init
    for _day, day_prices in da_prices.groupby(da_prices.index.date):
        if len(day_prices) < 24:
            continue
        order = day_prices.argsort().values
        cheapest = set(order[:2])
        richest = set(order[-2:])

        charge = np.zeros(len(day_prices))
        discharge = np.zeros(len(day_prices))
        soc_path = np.zeros(len(day_prices))

        # Walk the day in time order; charge/discharge subject to the SoC
        # actually available at that hour.
        for t in range(len(day_prices)):
            if t in cheapest:
                free = spec.energy_mwh - soc
                if free > 0:
                    qty = min(spec.power_mw * 1.0, free / spec.eta_charge)
                    charge[t] = qty
                    soc += spec.eta_charge * qty
            elif t in richest:
                avail = soc * spec.eta_discharge
                if avail > 0:
                    qty = min(spec.power_mw * 1.0, avail)
                    discharge[t] = qty
                    soc -= qty / spec.eta_discharge
            soc_path[t] = soc

        for t, ts in enumerate(day_prices.index):
            rows.append({
                "ts": ts,
                "charge_mwh": charge[t],
                "discharge_mwh": discharge[t],
                "soc_mwh": soc_path[t],
                "price_da": float(day_prices.iloc[t]),
                "price_id": float(id_prices.loc[ts]) if ts in id_prices.index else float("nan"),
                # Schedule is committed in the DA auction → settles at DA.
                "settlement_price": float(day_prices.iloc[t]),
            })

    df = pd.DataFrame(rows).set_index("ts")
    return DispatchResult(schedule=df, spec=spec, name="naive_heuristic")


# -- LP-based strategies ---------------------------------------------------


def _strategy_lp(
    objective_prices: pd.Series,    # what the LP optimises against
    settlement_prices: pd.Series,   # what cash flows actually clear at
    spec: BatterySpec,
    name: str,
) -> DispatchResult:
    """Solve a daily LP using objective_prices, settle revenue at settlement_prices."""
    objective_prices = objective_prices.dropna()
    settlement_prices = settlement_prices.reindex(objective_prices.index).interpolate()

    rows = []
    soc = spec.soc_init
    for day, day_prices in objective_prices.groupby(objective_prices.index.date):
        if len(day_prices) < 24:
            continue
        sol = _solve_day_lp(day_prices.values, spec, soc_start=soc)
        if sol is None:
            continue
        for t, ts in enumerate(day_prices.index):
            rows.append({
                "ts": ts,
                "charge_mwh": float(sol["charge"][t]),
                "discharge_mwh": float(sol["discharge"][t]),
                "soc_mwh": float(sol["soc"][t]),
                "price_da": float(day_prices.iloc[t]),
                "price_id": float(settlement_prices.loc[ts]),
                "settlement_price": float(settlement_prices.loc[ts]),
            })
        soc = float(sol["soc"][-1])

    df = pd.DataFrame(rows).set_index("ts")
    return DispatchResult(schedule=df, spec=spec, name=name)


def strategy_da_only_lp(
    da_prices: pd.Series, id_prices: pd.Series, spec: BatterySpec,
) -> DispatchResult:
    """
    LP optimised against DE/LU DA prices and SETTLED at DE/LU DA prices.

    DA auction results are published before the schedule must be firm, so
    optimising "against DA" involves no foresight — this is the implementable
    baseline of a DA-auction arbitrage battery. ``id_prices`` is unused for
    cash flows (kept in the signature for a stable strategy interface).
    """
    return _strategy_lp(
        objective_prices=da_prices,
        settlement_prices=da_prices,
        spec=spec,
        name="da_lp",
    )


def strategy_alt_series_lp(
    da_prices: pd.Series, id_prices: pd.Series, spec: BatterySpec,
) -> DispatchResult:
    """
    LP optimised AND settled on the alternative price series (``id_prices``).

    In this repo's data that series is the BELGIAN day-ahead price (v1
    mislabelled it as the German intraday continuous index and treated this
    run as a "perfect foresight ceiling"). It is retained as a cross-zonal
    reference: the same battery arbitraging a neighbouring zone's DA curve —
    NOT a foresight ceiling for the German battery.
    """
    return _strategy_lp(
        objective_prices=id_prices,
        settlement_prices=id_prices,
        spec=spec,
        name="alt_series_lp",
    )


def strategy_da_plus_tilt(
    da_prices: pd.Series,
    id_prices: pd.Series,
    forecast_error: pd.Series,
    spec: BatterySpec,
    beta: float = -1.4e-4,           # Q1 average β (€/MWh per MWh of fe)
    tilt_horizon_hours: int = 13,    # apply fe tilt over horizons 5–13 hours out
) -> DispatchResult:
    """
    DA optimisation with a Q1-style forecast-error tilt, settled at DA.

    For each hour t the DA price at horizons t+5..t+13 is shifted by β·fe_t
    before the daily LP is solved; cash flows still settle at the actual DA
    price (the schedule is committed in the DA auction).

    LOOK-AHEAD CAVEAT: fe is the REALISED forecast error, which is not
    knowable at DA gate closure. The tilt therefore upper-bounds what a
    real-time forecast-error signal could contribute to the DA schedule —
    and empirically it contributes ~nothing (see Q3 results).
    """
    da_prices = da_prices.dropna()
    id_prices = id_prices.reindex(da_prices.index).interpolate()
    fe = forecast_error.reindex(da_prices.index).fillna(0.0)

    # Build tilted-DA prices
    tilt = pd.Series(0.0, index=da_prices.index)
    for hours in range(5, tilt_horizon_hours + 1):
        tilt = tilt.add(beta * fe.shift(hours), fill_value=0.0)
    tilted_prices = da_prices + tilt

    return _strategy_lp(
        objective_prices=tilted_prices,
        settlement_prices=da_prices,
        spec=spec,
        name="da_plus_tilt",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_all_strategies(
    da_prices: pd.Series,
    id_prices: pd.Series,
    forecast_error: pd.Series,
    spec: BatterySpec,
) -> dict[str, DispatchResult]:
    """Convenience: run every strategy and return them keyed by name."""
    return {
        "naive": strategy_naive(da_prices, id_prices, spec),
        "da_lp": strategy_da_only_lp(da_prices, id_prices, spec),
        "da_plus_tilt": strategy_da_plus_tilt(da_prices, id_prices, forecast_error, spec),
        "alt_series_lp": strategy_alt_series_lp(da_prices, id_prices, spec),
    }
