# Findings

The longer version of what's in the README. Three questions, the data behind them, the methods, and the bits that ran the wrong way against my prior.

> **Read this first (July 2026).** The first version of this repo believed it was studying the
> German intraday market. It was not. The SMARD "intraday" price series I fetched are actually
> day-ahead auction prices of neighbouring bidding zones — the full story, with evidence, is in
> [Revision notes](#revision-notes-july-2026). Every section below has been rewritten around what
> the data actually is, and every number regenerated from the corrected pipeline. The original
> claims are preserved in the revision notes, not silently deleted.

## Contents

- [Why this market](#why-this-market)
- [Data — what it actually is](#data--what-it-actually-is)
- [Q1. Forecast errors and next-day prices](#q1-forecast-errors-and-next-day-prices)
- [Q2. Zonal day-ahead spreads and the backtest](#q2-zonal-day-ahead-spreads-and-the-backtest)
- [Q3. The "shape spread" and battery dispatch](#q3-the-shape-spread-and-battery-dispatch)
- [What you'd actually do with this](#what-youd-actually-do-with-this)
- [Where this falls over](#where-this-falls-over)
- [Revision notes (July 2026)](#revision-notes-july-2026)
- [If I come back to this](#if-i-come-back-to-this)

## Why this market

German power for delivery hour *h* tomorrow clears in a sequence of venues: the day-ahead (DA)
auction at 12:00 day-1 (the European coupled auction, SDAC — the reference price), then the
intraday auctions (since June 2024: IDA1 at 15:00 day-1, IDA2 at 22:00 day-1, IDA3 at 10:00 on
the delivery day, all quarter-hourly), then continuous intraday trading up to five minutes before
delivery. The EPEX **ID1** and **ID3** figures you see quoted are *not* auction prices — they are
volume-weighted indices of the continuous trades in the last one and three hours before delivery,
i.e. ex-post statistics of the continuous market. Renewables come in different from forecast, and
that imbalance has to clear across these venues in that order. Same physical megawatt-hour,
several sequential prices: that's what makes spread questions worth asking.

That was the plan. What this repo's free data actually supports is a related but different set of
questions about **zonal day-ahead prices under European market coupling** — see the next section.

## Data — what it actually is

Two free public sources. The ENTSO-E series are what they claim to be. The SMARD price series are
not what v1 of this repo believed, and the column names in the code are kept for cache and schema
continuity, so here is the translation table you need to read everything else:

| Legacy name in code/results | SMARD filter | What it actually is |
|---|---|---|
| `da_price` | 4169 "Marktpreis: Deutschland/Luxemburg" | DE/LU day-ahead auction price (correct as believed) |
| `id3_price` | 252 "Marktpreis: Dänemark 1" | **Danish DK1 day-ahead price** — not the EPEX ID3 index |
| `id_continuous` | 4996 "Marktpreis: Belgien" | **Belgian day-ahead price** — not a continuous-trading VWAP |
| `id_continuous_qh` | 5078 "Marktpreis: Anrainer DE/LU" | Neighbouring-zone day-ahead series at QH resolution; **hourly values replicated 4× before Oct 2025** (the SDAC 15-minute go-live) |
| `id1_price` | 251 | Not a valid SMARD filter; empty over the whole sample |

SMARD's "Marktpreis" modules publish zonal day-ahead prices only. EPEX's intraday indices are not
in them. How the misidentification was caught and verified is in the
[revision notes](#revision-notes-july-2026); the practical consequence is that every "auction vs
continuous" and "hourly vs quarter-hourly" framing in v1 was actually a **cross-zonal day-ahead**
comparison.

| Source | Series | Why |
|---|---|---|
| ENTSO-E | DA wind & solar generation forecast (per technology, hourly) | The forecast the DA market clears against. Subtract from actuals to get the imbalance volume. |
| ENTSO-E | Actual generation per type (hourly) | The realised side of the forecast error. |
| ENTSO-E | DA clearing prices, actual load | Price anchor and confounder/regime variable. |
| SMARD | DE/LU, DK1, Belgian day-ahead prices (hourly) | The three zonal price legs (legacy names above). |
| SMARD | "Anrainer DE/LU" neighbouring-zone price (15-min) | The QH leg of Q3 (informative only from Oct 2025). |

Window is 2018-10-01 to 2026-05-04, about 66,500 hourly observations. October 2018 is when the
German bidding zone became `DE_LU` on its own; before that it was `DE_AT_LU`, which is a different
population. Negative prices are deliberately not cleaned away — they are the structural signature
of renewables saturation, in every zone here.

One limit to keep up front: there is still no order-book data anywhere in this repo. True
microstructure (spreads, depth, order flow) is invisible in free data; EPEX sells it. What the
free data supports is auction-price econometrics, and that is now what the repo claims to do.

## Q1. Forecast errors and next-day prices

When wind comes in higher or lower than the DA forecast, somebody has to trade the gap. The
original hypothesis was that the price response should grow as Germany's renewable share grew.

**Series caveat first.** The "price" leg here is the Belgian day-ahead price (legacy name
`id_continuous`). A day-ahead price for delivery hour *t* is fixed at ~12:45 the day before, so
the forecast error realised *at* hour *t* cannot move it through any trading channel. What the
VAR actually measures is how German renewable forecast errors propagate into *subsequent* daily
auction prices of a tightly coupled zone — through weather-error persistence and shared drivers —
not intraday clearing of imbalances. The v1 write-up claimed the latter; that claim is retired.

The chain of tests: ADF and KPSS jointly (opposite nulls; a series must convince both), Granger
causality in both directions at AIC-selected lag, an IRF with residual-bootstrap bands, the FEVD,
and Bai-Perron sequential break detection on the daily impact coefficient.

Forward Granger rejects very strongly: F = 11.05, p ≈ 2×10⁻⁴² at AIC-selected lag 24 (the search
ceiling — worth knowing it hit the boundary), n = 66,526. Reverse is also significant (F = 8.31,
p ≈ 1.5×10⁻²⁹). At this sample size, bidirectional rejection is what you'd expect when both series
share common drivers like weather regimes, not evidence of prices causing weather. So Granger is
necessary but not sufficient.

The IRF:

![IRF](results/figures/q1_irf_and_fevd.png)

A +1σ forecast-error shock is followed by a small positive price response over hours 1–4 (peak
+0.92 €/MWh at h = 3), then a clear negative trough of −1.26 €/MWh around hour 11, decaying to
about −0.7 by hour 24. **With the corrected residual bootstrap (July 2026), the 90% band excludes
zero at hours 1–4 and 6–24** — at the trough the band is [−1.49, −1.08]. v1 reported the opposite
("CI never excludes zero"); that was an artifact of a mislabelled bootstrap that simulated
length-2,000 series instead of the full 66k sample, inflating the bands roughly √33 ≈ 5.8×. The
IRF is statistically significant. It remains economically small: the FEVD attributes at most
**1.1%** of price forecast-error variance to forecast-error innovations at any horizon up to 24h
(now exported to `results/tables/q1_fevd.csv`; v1 said "<5%" without shipping the number).

Bai-Perron is where the original hypothesis broke. I expected β (the daily price-impact
coefficient) to drift up over time as renewables grew. Instead the test finds two clean breaks:

![Bai-Perron breaks vs renewable share](results/figures/q1_bai_perron.png)

| Date | supF p (heuristic, see below) | β jump | Where this lands |
|---|---:|---|---|
| 2021-07-27 | 1.5×10⁻² | β goes from −7.9×10⁻⁵ to −4.3×10⁻⁴ (5.5× bigger) | TTF gas going vertical (€25 → €100 by October) |
| 2022-12-03 | 8.8×10⁻⁵ | β returns to −1.4×10⁻⁴ (~1.8× the pre-crisis level) | Gas back to ~€120 from August's €330 peak |

A third break is rejected (p = 0.22). Honesty note on those p-values: they evaluate the
sup-statistic against a pointwise F(2,·) distribution, which overstates significance for a
supremum over thousands of candidate dates (the proper Bai-Perron critical values are larger and
not implemented here). The break *dates* and the segment β's are exact-DP estimates and don't
depend on that approximation.

Those dates are gas-crisis bookends, not renewable-capacity events. Renewable share grew steadily
across the sample with no step changes; gas had two regime shifts at exactly the dates the test
picks out. The story isn't "renewables grew so β grew", it's "the price impact of an MWh of
imbalance scales with the cost of the gas plant covering it" — and that logic carries over
unchanged to the Belgian price leg, since the marginal unit in the coupled north-west European
system in those years was overwhelmingly gas. The model-risk implication survives intact: a
static β fitted on 2018–2021 will be silently wrong by ~5× inside a gas-price regime shift.
(Caveat that was true in v1 and still is: gas prices are never actually pulled or regressed —
the gas attribution is an interpretation of the break dates.)

Forecast-error volatility itself does grow steadily with the fleet:

![Forecast-error volatility growing](results/figures/q1_yearly_error_std.png)

So imbalance *volume* grows with renewable capacity, but the €/MWh price of an imbalance moves
with the merit-order regime. Two effects, two trends, both real, often confused.

Rolling 1-year Granger −log₁₀(p) confirms the relationship has been persistently significant
since mid-2020:

![Rolling Granger p-value](results/figures/q1_rolling_granger.png)

## Q2. Zonal day-ahead spreads and the backtest

What v1 called "auction vs continuous" is actually two cross-zonal day-ahead spreads:

* `da_id_spread` = DE/LU DA − Belgian DA (mean −1.28, σ 25.1 €/MWh)
* `id3_spread`  = DK1 DA − Belgian DA (mean −8.52, σ 33.7 €/MWh)

Both legs of each spread clear **simultaneously** in the single European day-ahead auction and
publish together at ~12:45 day-1. There is no sequencing, no 22:00 decision point, and no
"price-discovery venue" question in these data — the v1 headline "the auction is the price taker"
rested entirely on the series mislabelling and is retired. What the data *do* support is a
question about **market coupling**: how often are coupled zones' prices literally identical, and
how fast do zonal spreads close when congestion opens them?

The coupling signature is strong. Computed from the panel: DE/LU and DK1 clear at the *same price
to the cent* in 59% of all hours (including the 936.28 €/MWh Dunkelflaute spike of 2024-12-12,
which both zones printed identically); DE/LU and Belgium in 31% of hours. The median of both
spreads is ~0. When the spread is not zero, an interconnector is congested and the zones decouple.

![Zonal day-ahead spreads](results/figures/q2_spread_timeseries.png)

The formal tests. **Disclosure that v1 buried:** ADF/KPSS, Engle-Granger, Johansen and the VECM
below run on **daily means over the most recent 3 years** (n = 1,096) — full-sample hourly ADF
with autolag is computationally infeasible. Only the AR(1) half-life uses the full hourly sample.
VECM α loadings are therefore in per-day units.

| | DE/LU vs BE | DK1 vs BE |
|---|---:|---:|
| ADF p (leg 1, daily) | 4.2×10⁻⁴ | 3.8×10⁻⁵ |
| ADF p (BE leg, daily) | 1.4×10⁻³ | 1.4×10⁻³ |
| Engle-Granger t | −7.07 | −5.12 |
| Johansen rank | 2 (= joint stationarity, see below) | 2 |
| Spread half-life (full hourly sample) | **3.5 h** | **5.5 h** |
| VECM α, leg 1 (per day) | −0.96 | −0.71 |
| VECM α, BE leg (per day) | −0.55 | −0.26 |

Two corrections to how v1 read this table:

1. **Johansen rank 2 in a bivariate system is not cointegration** — it means Π is full rank and
   both series are jointly stationary. That is consistent with the ADF rows above: at daily
   frequency over 2023–2026, *both* price legs reject a unit root. v1 quoted "rank 2" as
   cointegration support, which was wrong by the repo's own decision rule. On this window the
   honest statement is: the levels themselves are (near-)stationary at daily frequency, so a
   stationary spread is unremarkable, and the Engle-Granger t-stats are better read as strong
   mean reversion than as a unit-root-sharing result.
2. The VECM was nonetheless fit at rank 1. With rank 2 indicated, that makes its α's descriptive
   statistics, not estimates of a supported model. Taken as description: when the spread opens,
   the DE/DK1 leg does 1.8–2.8× more of the closing than the Belgian leg, in per-day units. Read
   it as relative anchoring of coupled zonal prices, not as venue price discovery.

Mean-reversion speed itself is stable across the sample, with seasonal modulation and no secular
trend — congestion episodes are transient everywhere in the window:

![Rolling 90-day half-life](results/figures/q2_rolling_halflife.png)

Spread variance widens with the German renewable forecast-error regime in both pairs (σ in
€/MWh; terciles of |fe|):

| Forecast-error regime | DE/LU−BE σ | DK1−BE σ |
|---|---:|---:|
| Low \|fe\| | 23.0 | 31.0 |
| Med \|fe\| | 24.0 | 32.9 |
| High \|fe\| | 27.8 | 36.8 |

Variance ratios ≈1.2× high-vs-low in both pairs — modest, monotone, and physically sensible:
big German renewable surprises are exactly when interconnectors congest.

### The backtest

**What is actually traded.** The spread is DK1 DA − Belgian DA. Signal: for each delivery day *t*,
the trailing-90-day mean of the realised spread per delivery hour, lagged one delivery day — its
newest input published ~12:45 on day *t−2*, so the signal exists almost a full day before the
day-*t* auction clears. Nothing is peeked at; the strictly-lagged construction from v1 survives
unchanged and is the sound part of the design. Pick the hour with the largest |signal|; trade only
if it exceeds €5/MWh; direction = sign of the signal; one position per delivery day; PnL =
direction × (DK1 − BE) for that hour, minus costs.

**Executability caveat, up front.** This is a *paper* spread. You cannot capture a zonal
day-ahead price difference by submitting spot orders in two zones — monetising it is what
cross-zonal transmission rights (FTRs / interconnector capacity) are for, and those are not
modelled. The cost scenarios below (€0.06 / €6 / €12 per MWh round-trip) are placeholders on one
leg, not a model of FTR auctions. Read the table as a measurement of zonal-spread persistence,
not a deployable strategy. Also retired: v1 called the spec "pre-registered", but all code and
results landed in the repo's initial commit, so the claim isn't verifiable from history — and the
committed spec file actually parameterised an earlier, abandoned strategy.

Walk-forward across the full sample, stationary block bootstrap for the Sharpe p-value (block
length √T preserves the autocorrelation of daily PnL). Sharpe annualised by √252 (equity
convention, kept for v1 comparability); since power clears every calendar day, the √365 figure is
reported alongside (≈20% larger).

![Cumulative PnL](results/q2_backtest_honest/figures/q2_honest_cumulative_pnl.png)

| Scenario | Sharpe (√252) | Sharpe (√365) | Bootstrap p | 2021–22 share of lifetime PnL |
|---|---:|---:|---:|---:|
| Optimistic (fees only) | 5.95 | 7.16 | < 0.001 | 49% |
| Realistic (€6/MWh round-trip) | **3.89** | 4.68 | < 0.001 | 59% |
| Pessimistic (€12/MWh) | 1.80 | 2.17 | 0.025 | **95%** |

(v1 said "75% of lifetime PnL from 2021–22" for all three rows; that was an editorial round
number. The per-scenario shares above are computed from `results/q2_backtest_honest/tables/`.)

The €5/MWh entry gate barely binds — 92.3% of delivery days trade — so this is closer to an
always-on position than a selective one. Year by year under pessimistic costs (€/MWh on a unit
position):

```
2018Q4   +1 559    2022    +8 291    ← gas-crisis peak
2019     −2 041    2023    +2 289
2020       −128    2024      −814    ← compressed
2021     +4 272    2025    −1 115
                   2026*     +870      (*to 2026-05-04)
```

Lost money in 4 of 9 calendar years under pessimistic costs; 95% of lifetime pessimistic PnL sat
in 2021–2022. Same regime story as Q1's Bai-Perron breaks, viewed from the spread side: the gas
crisis blew out zonal spreads (congestion became violent and directional), and trailing means
harvested it. This is a regime harvester on congestion persistence, not a permanent alpha.

## Q3. The "shape spread" and battery dispatch

v1's framing — hourly block vs four 15-minute QH products for the same delivery hour, "the duck
curve as a tradeable spread" — does not survive the series identification. What
`shape_spread = hourly − mean(4 QH)` actually computes here is the **Belgian hourly DA price
minus the hourly mean of SMARD's "Anrainer DE/LU" neighbouring-zone series**. Two consequences:

1. It is a *cross-zonal* spread, not an hourly-vs-QH granularity premium.
2. Before October 2025 (the SDAC 15-minute go-live) the QH leg is an hourly price replicated
   four times — verified in the cache: intra-hour σ is exactly zero for 100% of pre-Oct-2025
   hours and nonzero for every hour after. So for ~91% of the sample the "shape" component is
   zero by construction, and the profile below is purely a zonal price difference.

The hour-of-day pattern is still striking, and still real — it's just a different fact than v1
claimed:

![Shape spread by hour of day](results/figures/q3_shape_by_hour.png)

+€9.8/MWh at hour 21, −€3.9/MWh at hour 12, stable across 7.5 years: Belgium clears rich against
the DE/LU-neighbour average in the evening and cheap at the solar peak. That is a duck-curve
*difference* between coupled zones (different solar shares, different flexibility), not the price
of intra-hour shape.

The regression, now with honest standard errors:

```
shape_spread ~ const + intra_hour_σ + 23 hour FE
n = 56 569       R² = 0.054
intra_hour_σ coefficient: −0.91   (t = −9.7, HAC/Newey-West, 24 lags)
```

Two big caveats. First, v1 reported t = −30.8 from plain OLS on 56k serially correlated hourly
observations; HAC standard errors cut that to −9.7. Still significant, but a third of the size —
the OLS number was overstated and unflagged. Second, because σ is identically zero before
Oct 2025, the coefficient is identified entirely by the ~5,200 post-go-live hours: it says that
since quarter-hourly day-ahead pricing exists, hours with more intra-hour dispersion in the
neighbouring-zone QH prices see the Belgian hourly price cheap relative to the QH mean. Direction
consistent with a shape/uncertainty premium, but it's seven months of data wearing a
7.5-year regression's n.

Conditioning on the German forecast-error regime tells a compatible story (σ of the spread widens
21.1 → 24.6 €/MWh from low to high |fe| tercile, mean drifts 4.9 → 2.0), but after the reframing
this reads as congestion sensitivity again, not intra-hour ramp pricing.

### Battery dispatch sim

The battery LP survives the revision better than the framing around it did, because its engine
never depended on the series identity. Specs: 100 MW / 200 MWh (2-hour duration), 85% round-trip
efficiency, €2/MWh wear, 1.5 cycles/day. Daily LP over 24 hours, 72 variables/day,
`scipy.optimize.linprog`/HiGHS, ~2,700 days.

Two v1 bugs fixed in July 2026 change the numbers materially:

* **Settlement convention.** v1 "optimised on DA, settled at intraday" — but the "intraday"
  series was the Belgian DA price, so v1's headline €4.71M/yr was a DE-scheduled, BE-settled
  hybrid with no economic meaning. DA strategies now settle at the DE/LU DA price they schedule
  against.
* **Naive chronology.** v1's naive baseline could discharge energy before charging it (it
  processed cheap hours first regardless of clock order). It now walks the day chronologically.

| Strategy | Annual revenue | €/MW/year | Uplift vs naive |
|---|---:|---:|---:|
| Naive (cheapest-2h charge / richest-2h discharge, chronological) | €3.91 M | €39.1 k | baseline |
| DA-LP (optimise and settle on DE/LU DA) | **€5.64 M** | **€56.4 k** | +44.5% |
| DA + Q1 forecast-error tilt | €5.64 M | €56.4 k | +44.3% (adds nothing) |
| Belgian-DA LP (legacy "perfect foresight" — cross-zonal reference) | €5.51 M | €55.1 k | +41.2% |

Reading notes. DA-LP optimises against the day's realised DA prices; since hour selection in the
auction strictly precedes price publication, treat €56.4k/MW/yr as the *ceiling* of DA-auction
arbitrage for this spec — desks get close to it with price-curve bids because the DA shape is
highly forecastable, but it is a benchmark, not a P&L. It sits at the top of the €40–60k/MW/yr
range quoted for German BESS arbitrage in 2024–25. The Q1 tilt adds nothing even though it uses
*realised* forecast errors (information a real bidder wouldn't have) — an upper bound on the
signal's value, and it's ~zero, consistent with the FEVD: DA prices already embed the renewable
forecast, and the residual error is too small to move a 24-hour schedule. The old
"perfect-foresight ceiling" row is retired; optimising and settling on the Belgian curve is just
the same battery in a different zone, and its similarity to DA-LP (€55.1k vs €56.4k) says the two
zones offer near-identical arbitrage-relevant daily shape.

This is energy arbitrage only — no FCR/aFRR stacking, which is another 30–50% of real BESS
revenue. Floor on the strategy set, ceiling per strategy.

## What you'd actually do with this

Post-revision, the honest inventory:

* **The Bai-Perron result is the most useful piece and survives untouched.** Any model with a
  hardcoded renewable-impact coefficient was silently wrong by ~5× during the gas crisis. If you
  run price-impact models, condition them on the marginal-fuel regime.
* **The zonal-spread work is a coupling study, not a trade.** The persistent DK1−BE and DE−BE
  biases and their 3.5–5.5h half-lives are the sort of statistics an FTR / transmission-rights
  desk cares about; the backtest shows the biases were harvestable on paper mostly inside the
  2021–22 congestion blowout. Turning that into a real position needs FTR data this repo doesn't
  have.
* **The battery LP is the directly reusable output.** €56k/MW/yr DA-arbitrage ceiling at 2-hour
  duration, ~+45% over a naive heuristic, forecast-error tilt worthless. The engine is honest
  (physics constraints, settlement = schedule market) and easy to point at any hourly price
  vector.
* Q1's IRF/FEVD says German renewable forecast errors are real but second-order for next-day
  auction prices — a useful negative for anyone hoping to trade DA on public forecast data alone.

## Where this falls over

Things that affect the headline numbers:

1. **The series misidentification is the headline limitation of v1** — it's fixed in the
   documentation and framing, but the underlying panel is still three zonal DA prices wearing
   legacy intraday names. No intraday price appears anywhere in this repo.
2. Bidirectional Granger at this sample size implies confounding. Common drivers (load forecast
   errors, weather regimes) bleed into both series. Adding load FE as exogenous in the VAR is
   still the standard fix; still not done.
3. The formal cointegration-style tests run on daily means of a 3-year subsample (n=1,096), not
   the full hourly history; and on that window both legs are individually stationary, so the
   spread tests are mean-reversion measurements, not cointegration evidence. Half-life is on the
   full hourly sample via closed-form AR(1).
4. Bai-Perron supF p-values use a pointwise F(2,·) reference — anti-conservative for a supremum;
   the proper Bai-Perron critical values are not implemented. Dates and β's are exact.
5. Cholesky ordering in the IRF is a choice (forecast error first). A sign-restriction SVAR would
   make it defensible; the +bump at h1–4 may be partly an ordering artifact.

Things that affect the Q2 backtest:

6. Not executable as specified — needs cross-zonal transmission rights; costs are placeholders.
7. Regime concentration: pessimistic-cost Sharpe 1.80 (p = 0.025) with 95% of PnL from 2021–22
   and losses in 4 of 9 years. The €5 entry gate is near-non-binding (92% of days trade).
8. Sharpe uses √252 by convention; √365 is ~20% larger. Both reported, one headline.

Things that affect the Q3/battery numbers:

9. The σ regression's regressor is degenerate before Oct 2025; the effect is identified on ~7
   months of data.
10. DA-LP is a with-hindsight benchmark within each day (hour selection precedes price
    publication in a real auction); no bid-ask or auction-impact costs anywhere in the sim.
11. Energy arbitrage only; no ancillary revenue stack.

And generalisability: one sample of one corner of Europe, 2018–2026, containing one once-a-decade
gas crisis that dominates several results.

## Revision notes (July 2026)

A full pass over this repo before internship applications turned up one large error and several
methodological bugs. Everything below is fixed in code, all artifacts under `results/` are
regenerated from the corrected pipeline (sample window pinned to 2018-10-01 → 2026-05-04, data
from the local parquet cache fetched in May 2026), and the old headline numbers are recorded here
rather than silently overwritten. Tests went from 56 to 67; `ruff` is clean.

**1. The SMARD price series were misidentified — the central revision.**
v1 mapped SMARD filter codes 4996 → "intraday continuous VWAP", 252 → "ID3 auction", 251 → "ID1",
5078 → "QH intraday index". Per the SMARD OpenAPI specification
([bundesAPI/smard-api](https://github.com/bundesAPI/smard-api)) these are actually: 4996 =
"Marktpreis: Belgien" (Belgian DA), 252 = "Marktpreis: Dänemark 1" (DK1 DA), 5078 = "Marktpreis:
Anrainer DE/LU" (neighbouring-zone DA), and 251 does not exist (hence the all-NaN "ID1" column).
Verified against the data itself, not just the spec: the "ID3" series equals the DE/LU DA price
to the cent in 59% of all hours — including the 936.28 €/MWh Dunkelflaute print of 2024-12-12 —
which is a market-coupling signature no continuous-trading VWAP could produce; the QH series has
exactly zero intra-hour variation until precisely 2025-10-01, the SDAC 15-minute MTU go-live
([EPEX press release](https://www.epexspot.com/en/news/market-coupling-steering-committee-confirms-go-live-15-minute-mtu-sdac-trading-day-30));
and the real EPEX ID1/ID3 are ex-post VWAP indices of continuous trades in the last 1h/3h before
delivery, not auctions at all ([EPEX indices](https://www.epexspot.com/en/indices)). The actual
intraday auctions are the IDAs, launched 14 June 2024
([ENTSO-E](https://www.entsoe.eu/network_codes/cacm/implementation/ida/)). Legacy column and
method names are kept (cache/schema continuity) and documented in
`power_microstructure/data/smard.py`. A related shift in the generation filter table (e.g.
"solar" was actually fetching wind-onshore: filter 4067 vs 4068) is fixed; those series were
fetched but never used in any analysis.

**2. Q1 IRF bootstrap was fake; the conclusion flips.** The "residual bootstrap" computed
resampled residuals, never used them, simulated Gaussian length-2,000 series regardless of the
66k sample, and silently fell back to the point estimate on failure. It is now a real residual
bootstrap at full sample length (loud failure if >5% of replications die). Old: "90% band never
excludes zero". New: **the band excludes zero at hours 1–4 and 6–24** (trough −1.26, band
[−1.49, −1.08]). The FEVD is now exported (`q1_fevd.csv`): max 1.1% — the "economically small"
half of the old conclusion stands; the "statistically inconclusive" half was a bug.

**3. Q3 regression now uses HAC standard errors.** t on the σ coefficient: **−30.8 (plain OLS) →
−9.7 (Newey-West, 24 lags)**. Coefficient unchanged at −0.91.

**4. Q2 reframed; Johansen misreading fixed.** "Auction vs continuous price discovery" became
"zonal DA spreads under market coupling" (see item 1). Johansen "rank 2" was quoted as
cointegration support; in a bivariate system it means both series are stationary — consistent
with the shipped ADF p-values on the daily test window. Disclosed that formal tests run on
daily-resampled 3-year data (n=1,096) and that VECM α's are per-day. The old headline "the
auction is the price taker (α ratio 1.8–2.8×)" is retired; the α asymmetry survives only as a
descriptive statement about which zonal price adjusts faster.

**5. Backtest honesty.** "Pre-registered" wording dropped (unverifiable: code and results landed
in one initial commit, and the committed spec parameterised a different, abandoned strategy).
"75% of lifetime PnL from 2021–22" replaced by per-scenario shares 49% / 59% / 95%
(optimistic/realistic/pessimistic). `n_trades` renamed `n_traded_days` (what it actually counts).
√365 Sharpe reported next to √252. The earlier backtest that marked the spread at hourly
frequency (Sharpe up to 5.5) is quarantined in `results/deprecated_q2_backtest_hourly_marks/`
with its own DEPRECATED.md; its numbers should be read only as a worked example of manufacturing
performance by marking an untradeable series.

**6. Battery sim corrections.** DA strategies settled at the (mislabelled) Belgian series; they
now settle at the DE/LU DA price they schedule in — headline DA-LP revenue moves **€4.71M →
€5.64M/yr (€47k → €56.4k/MW/yr)**. The naive baseline could discharge before charging
(cheapest-hours-first processing); it is now chronological — €3.92M → €3.91M/yr. The
"perfect-foresight" strategy is renamed `alt_series_lp` (it was never a German foresight ceiling,
just the Belgian curve): €5.52M → €5.51M/yr. The tilt strategy is flagged as using realised
forecast errors (upper bound), and still adds ~nothing.

**7. Bai-Perron supF fix.** The F-statistic now uses q=2 restrictions per break (intercept and
slope both shift; v1 used q=1) and the p-value carries an explicit anti-conservativeness caveat.
Break p-values move 3.7×10⁻³ → 1.5×10⁻² and 1.5×10⁻⁵ → 8.8×10⁻⁵; dates and β's unchanged.

**8. Reproducibility.** All five runners previously used END = "today − 3 days", so every run was
date-dependent; END is now pinned to 2026-05-04 (`power_microstructure/runconfig.py`), with
`--end YYYY-MM-DD` / `--end live` opt-outs. The `ENTOSE_API_KEY` typo-fallback is gone; ENTSO-E
cache hits now return the same type (Series) as live calls; `seaborn`/`tqdm` removed from
dependencies (never imported). Figures regenerated with titles matching the corrected series
identities. Notebook outputs predate this revision and are next in line for a re-render.

## If I come back to this

Roughly in priority order:

* Get the actual German intraday data — the EPEX ID1/ID3 indices (or better, order-book data)
  are paid; ENTSO-E publishes some intraday prices per zone worth evaluating first. Then the
  original Q2/Q3 questions become answerable for real.
* Add load forecast error as exogenous in the VAR (closes the bidirectional-Granger objection).
* Interact the Q1 β with TTF gas prices directly instead of arguing from break dates.
* Implement proper Bai-Perron critical values for the supF sequence.
* Re-run the Q3 σ regression as the post-Oct-2025 QH sample grows; it's the one place the repo
  now has a genuine granularity signal, seven months old.
* Pull FTR auction results to price the executability of the zonal-spread "trade".
* Threshold VECM for asymmetric mean reversion in the zonal spreads.
* Re-render the notebooks against the regenerated artifacts.
