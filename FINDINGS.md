# Findings

The long version of the README: three questions, the data behind them, the methods, and the
results, including the ones that ran against my prior. The [revision history](#revision-history-july-2026)
at the end records what changed in July 2026 and why.

## Contents

- [Why this market](#why-this-market)
- [Data](#data)
- [Q1. Forecast errors and next-day prices](#q1-forecast-errors-and-next-day-prices)
- [Q2. Zonal day-ahead spreads and the backtest](#q2-zonal-day-ahead-spreads-and-the-backtest)
- [Q3. Zonal profile spread and battery dispatch](#q3-zonal-profile-spread-and-battery-dispatch)
- [What you'd do with this](#what-youd-do-with-this)
- [Where this falls over](#where-this-falls-over)
- [Revision history (July 2026)](#revision-history-july-2026)
- [If I come back to this](#if-i-come-back-to-this)

## Why this market

German power for delivery hour *h* tomorrow clears in a sequence of venues: the coupled European
day-ahead auction (SDAC) at 12:00 the day before, three quarter-hourly intraday auctions since
June 2024 (IDA1 at 15:00 and IDA2 at 22:00 on the day before, IDA3 at 10:00 on the delivery day),
and continuous intraday trading up to five minutes before delivery. Renewables come in different
from forecast, and the gap has to clear somewhere in that sequence. Same physical megawatt-hour,
several sequential prices. That is what makes spread questions worth asking.

The free data this repo runs on supports a related set of questions about zonal day-ahead prices
under European market coupling: how renewable forecast errors propagate into next-day auction
prices, how tightly coupling ties neighbouring zones together, and what the daily price profile
is worth to storage.

## Data

Two free sources, ENTSO-E and SMARD. The SMARD price columns keep their historical names in code
and results for cache and schema continuity, so this table is required reading:

| Column name | SMARD filter | Series |
|---|---|---|
| `da_price` | 4169 "Marktpreis: Deutschland/Luxemburg" | DE/LU day-ahead auction price |
| `id3_price` | 252 "Marktpreis: Dänemark 1" | Danish DK1 day-ahead price |
| `id_continuous` | 4996 "Marktpreis: Belgien" | Belgian day-ahead price |
| `id_continuous_qh` | 5078 "Marktpreis: Anrainer DE/LU" | Neighbouring-zone day-ahead price at 15-minute resolution; hourly values replicated 4x before Oct 2025 (the SDAC 15-minute go-live) |
| `id1_price` | 251 | Not a valid SMARD filter; empty |

SMARD's Marktpreis modules publish zonal day-ahead prices only. EPEX's ID1 and ID3 figures are
volume-weighted averages of continuous trades in the last one and three hours before delivery,
not auctions, and not available for free. Nothing in this repo is an intraday price. How the
column names came to disagree with their contents is in the
[revision history](#revision-history-july-2026).

From ENTSO-E: day-ahead wind and solar generation forecasts per technology, actual generation
per type, day-ahead prices, and actual load, all hourly. Forecast error is actual minus forecast.

The window is 2018-10-01 to 2026-05-04, about 66,500 hourly observations. It starts when the
German bidding zone became DE/LU on its own; before that it was DE-AT-LU, a different population.
Negative prices stay in the sample. They are the structural signature of renewables saturation,
in every zone here.

One limit up front: there is no order-book data anywhere in this repo. Spreads, depth and order
flow are paid data. What free data supports is auction-price econometrics, and that is what this
repo does.

## Q1. Forecast errors and next-day prices

When wind comes in above or below the day-ahead forecast, someone has to trade the gap. I
expected the price response to grow with Germany's renewable build-out.

A timing point first. The price leg here is the Belgian day-ahead price. A day-ahead price for
hour *t* is fixed around 12:45 the day before, so the forecast error realised at hour *t* cannot
move it through any trading channel. What the VAR measures is how German forecast errors
propagate into subsequent daily auction prices of a tightly coupled zone, through weather
persistence and shared drivers.

The chain of tests: ADF and KPSS jointly (opposite nulls; a series has to convince both), Granger
causality in both directions at AIC-selected lag, an IRF with residual-bootstrap bands, the FEVD,
and Bai-Perron break detection on the daily impact coefficient.

Granger rejects strongly in the forward direction: F = 11.05, p on the order of 1e-42, at lag 24
(the AIC search ceiling; it chose the boundary), n = 66,526. The reverse direction also rejects
(F = 8.31). At this sample size, rejection in both directions is what shared drivers like weather
regimes produce, and prices do not cause weather. I treat Granger here as necessary but not
sufficient.

The impulse response:

![IRF](results/figures/q1_irf_and_fevd.png)

A one-sigma forecast-error shock is followed by a small positive response over hours 1 to 4 (peak
+0.92 EUR/MWh at hour 3), then a trough of -1.26 EUR/MWh around hour 11, decaying to about -0.7
by hour 24. The 90% band comes from a residual bootstrap: resample the estimated residuals,
rebuild the series through the fitted VAR recursion at full sample length, re-estimate, repeat
200 times. It excludes zero at hours 1 to 4 and 6 to 24; at the trough the band is
[-1.49, -1.08]. The FEVD attributes at most 1.1% of price forecast-error variance to
forecast-error innovations at any horizon up to 24 hours (`results/tables/q1_fevd.csv`). The
shocks are orthogonalised by Cholesky with the forecast error ordered first. That is defensible
(weather is exogenous to the market within the hour) but it is an assumption, and the positive
bump at hours 1 to 4 may partly be an ordering artifact.

Bai-Perron is where my prior broke. I expected the daily impact coefficient to drift up as
renewables grew. The test finds two breaks instead:

![Bai-Perron breaks vs renewable share](results/figures/q1_bai_perron.png)

| Date | supF p (heuristic, see below) | Coefficient move | What was happening |
|---|---:|---|---|
| 2021-07-27 | 1.5e-2 | -7.9e-5 to -4.3e-4 (5.5x) | TTF gas going from 25 to 100 EUR by October |
| 2022-12-03 | 8.8e-5 | back to -1.4e-4 (about 1.8x pre-crisis) | gas back near 120 from the 330 peak |

A third break is rejected (p = 0.22). The p-values compare the sup-statistic against a pointwise
F(2, .) distribution, which overstates significance for a supremum over thousands of candidate
dates; the proper Bai-Perron critical values are larger and not implemented here. The break dates
and segment coefficients come from exact dynamic programming and do not depend on that
approximation.

Those dates are gas-crisis bookends. Renewable share grew steadily across the sample with no
step changes, and gas had two regime shifts at the dates the test picks. My reading: the price
impact of a MWh of imbalance scales with the cost of the gas plant covering it, and that carries
over to the Belgian price leg because the marginal unit in coupled north-west Europe in those
years was mostly gas. Gas prices are never pulled or regressed here, so the attribution rests on
the dates. The practical point survives either way: a static impact coefficient fitted on
2018-2021 is wrong by about 5x inside a gas regime shift.

Forecast-error volatility itself does grow with the fleet:

![Forecast-error volatility growing](results/figures/q1_yearly_error_std.png)

Imbalance volume grows with renewable capacity; the price of a MWh of imbalance moves with the
merit-order regime. Two trends, easy to confuse.

Rolling one-year Granger tests stay significant from mid-2020 on:

![Rolling Granger p-value](results/figures/q1_rolling_granger.png)

## Q2. Zonal day-ahead spreads and the backtest

Two cross-zonal day-ahead spreads:

* `da_id_spread` = DE/LU minus Belgium (mean -1.28, sigma 25.1 EUR/MWh)
* `id3_spread` = DK1 minus Belgium (mean -8.52, sigma 33.7 EUR/MWh)

Both legs of each spread clear simultaneously in the single coupled auction and publish together
around 12:45 the day before delivery. The question the data supports is about market coupling:
how often are coupled zones' prices literally identical, and how fast do spreads close when
congestion opens them?

The coupling signature is strong. DE/LU and DK1 clear at the same price to the cent in 59% of
all hours, including the 936.28 EUR/MWh Dunkelflaute hour of 2024-12-12, which both zones printed
identically. DE/LU and Belgium match in 31% of hours. The median of both spreads is about zero.
A nonzero spread means an interconnector is congested and the zones have decoupled.

![Zonal day-ahead spreads](results/figures/q2_spread_timeseries.png)

The formal tests run on daily means over the most recent three years (n = 1,096); full-sample
hourly ADF with automatic lag selection is not computationally feasible. Only the AR(1)
half-life uses the full hourly sample. VECM adjustment coefficients are in per-day units.

| | DE/LU vs BE | DK1 vs BE |
|---|---:|---:|
| ADF p (leg 1, daily) | 4.2e-4 | 3.8e-5 |
| ADF p (BE leg, daily) | 1.4e-3 | 1.4e-3 |
| Engle-Granger t | -7.07 | -5.12 |
| Johansen rank | 2 | 2 |
| Spread half-life (full hourly sample) | 3.5 h | 5.5 h |
| VECM alpha, leg 1 (per day) | -0.96 | -0.71 |
| VECM alpha, BE leg (per day) | -0.55 | -0.26 |

Two things about reading this table. Johansen rank 2 in a bivariate system means the matrix is
full rank and both series are jointly stationary; it is not evidence of cointegration. That is
consistent with the ADF rows: at daily frequency on this window, both price legs individually
reject a unit root. So the spread tests here measure mean reversion, not shared stochastic
trends. And since rank 2 is indicated, the rank-1 VECM is descriptive rather than a supported
model. Taken as description, when a spread opens the DE or DK1 leg does 1.8 to 2.8 times more of
the closing than the Belgian leg.

Mean-reversion speed is stable across the sample, with seasonal wiggle and no trend. Congestion
episodes are transient throughout:

![Rolling 90-day half-life](results/figures/q2_rolling_halflife.png)

Spread dispersion widens with the German forecast-error regime in both pairs (sigma in EUR/MWh,
terciles of absolute forecast error):

| Forecast-error regime | DE/LU-BE | DK1-BE |
|---|---:|---:|
| Low | 23.0 | 31.0 |
| Medium | 24.0 | 32.9 |
| High | 27.8 | 36.8 |

About a 1.2x ratio high versus low in both pairs. Modest, monotone, and physically sensible:
big German renewable surprises are when interconnectors congest.

### The backtest

What is traded: the DK1 minus Belgium spread. The signal for delivery day *t* is the
trailing-90-day mean of the realised spread per delivery hour, lagged one delivery day. Its
newest input publishes around 12:45 on day *t-2*, so the signal exists almost a full day before
the day-*t* auction clears; nothing is peeked at. Take the hour with the largest absolute
signal, trade only if it exceeds 5 EUR/MWh, direction equals the sign, one position per delivery
day, PnL equals direction times the realised spread minus costs. Cost scenarios are 0.06, 6 and
12 EUR/MWh round trip. The Sharpe p-value comes from a stationary block bootstrap on daily PnL
(expected block length sqrt(T), 2,000 resamples, one-sided test against Sharpe of zero). Sharpe
is annualised by sqrt(252), with the sqrt(365) figure alongside since power clears every
calendar day.

Executability, up front: this is a paper spread. You cannot capture a zonal day-ahead price
difference by placing spot orders in two zones. Monetising it is what cross-zonal transmission
rights are for, and those are not modelled; the cost scenarios are placeholders on one leg. Read
the table as a measurement of congestion persistence, not a deployable strategy.

![Cumulative PnL](results/q2_backtest_honest/figures/q2_honest_cumulative_pnl.png)

| Scenario | Sharpe (sqrt 252) | Sharpe (sqrt 365) | Bootstrap p | 2021-22 share of PnL |
|---|---:|---:|---:|---:|
| Optimistic (fees only) | 5.95 | 7.16 | < 0.001 | 49% |
| Realistic (6 EUR/MWh round trip) | 3.89 | 4.68 | < 0.001 | 59% |
| Pessimistic (12 EUR/MWh) | 1.80 | 2.17 | 0.025 | 95% |

The 5 EUR/MWh entry gate barely binds: 92.3% of delivery days trade, so this is closer to an
always-on position than a selective one. Year by year under pessimistic costs (EUR/MWh on a unit
position):

```
2018Q4   +1 559    2022    +8 291    <- gas-crisis peak
2019     -2 041    2023    +2 289
2020       -128    2024      -814    <- compressed
2021     +4 272    2025    -1 115
                   2026*     +870      (*to 2026-05-04)
```

Losses in 4 of 9 calendar years under pessimistic costs, and 95% of pessimistic lifetime PnL sat
in 2021-2022. This is the same regime story as Q1's breaks, viewed from the spread side: the gas
crisis blew out zonal spreads, congestion became violent and directional, and trailing means
harvested it. A regime harvester on congestion persistence.

## Q3. Zonal profile spread and battery dispatch

The spread here, `shape_spread` = Belgian hourly day-ahead price minus the hourly mean of the
neighbouring-zone quarter-hourly series, is a cross-zonal quantity. Two facts shape everything
in this section:

1. The two legs belong to different zones, with different solar shares and flexibility.
2. Before October 2025 the quarter-hourly leg is the hourly price replicated four times, so its
   intra-hour dispersion is exactly zero; real quarter-hourly variation exists only from the
   SDAC 15-minute go-live onward, about 5,200 hours of the sample.

The hour-of-day profile is striking and stable:

![Shape spread by hour of day](results/figures/q3_shape_by_hour.png)

Plus 9.8 EUR/MWh at hour 21, minus 3.9 at hour 12, across the whole sample: Belgium clears rich
against the DE/LU-neighbour average in the evening ramp and cheap at the solar peak. A duck-curve
difference between coupled zones.

The regression:

```
shape_spread ~ const + intra_hour_sigma + 23 hour fixed effects
n = 56,569      R2 = 0.054
intra_hour_sigma coefficient: -0.91   (t = -9.7, Newey-West with 24 lags)
```

Because the regressor is identically zero before October 2025, the coefficient is identified
entirely by the post-go-live hours. It says: in hours with more intra-hour dispersion in the
neighbouring-zone quarter-hourly prices, the Belgian hourly price trades cheap relative to the
quarter-hourly mean. The direction is consistent with a shape premium, and it is seven months of
data wearing a 7.5-year regression's n. The HAC window of 24 lags covers one delivery day of
hourly serial correlation.

Conditioning on the German forecast-error regime: the spread's sigma widens from 21.1 to 24.6
EUR/MWh between low and high error terciles. After the cross-zonal reading of the spread, this
is congestion sensitivity again.

### Battery dispatch

Specs: 100 MW / 200 MWh (two-hour duration), 85% round-trip efficiency, 2 EUR/MWh wear cost,
1.5 cycles per day. One linear program per day: 72 variables (charge, discharge and state of
charge per hour), state-of-charge recursion with separate charge and discharge efficiencies,
power and energy bounds, a daily cycle cap, solved with HiGHS through scipy. About 2,700 days.

| Strategy | Annual revenue | EUR/MW/year | vs naive |
|---|---:|---:|---:|
| Naive (cheapest 2h charge, richest 2h discharge, chronological) | 3.91 M | 39.1 k | baseline |
| DA-LP (optimise and settle on DE/LU day-ahead) | 5.64 M | 56.4 k | +44.5% |
| DA-LP + forecast-error tilt | 5.64 M | 56.4 k | +44.3% |
| Belgian-DA LP (same battery, other zone) | 5.51 M | 55.1 k | +41.2% |

Reading the table. DA-LP optimises against the day's realised day-ahead prices, and hour
selection precedes price publication in a real auction, so 56.4k EUR/MW/yr is the ceiling of
day-ahead arbitrage for this spec rather than a P&L; desks get close to it with price-curve bids
because the day-ahead shape is highly forecastable. It sits at the top of the 40 to 60k EUR/MW/yr
range quoted for German battery arbitrage in 2024-25. The tilt strategy uses realised forecast
errors, information a real bidder would not have, and still adds nothing. That is consistent
with the FEVD: day-ahead prices already embed the renewable forecast, and the residual error is
too small to move a 24-hour schedule. The Belgian row is the same battery pointed at the other
zone's curve, and its closeness to DA-LP says the two zones offer near-identical daily shape.
Energy arbitrage only; frequency-response and reserve stacking, worth another 30 to 50% in
practice, is out of scope.

## What you'd do with this

* The Bai-Perron result is the most useful piece. Any model with a hardcoded renewable-impact
  coefficient was silently wrong by about 5x during the gas crisis. Condition price-impact
  models on the marginal-fuel regime.
* The zonal-spread work is a coupling study. The persistent DK1-BE and DE-BE biases and their
  3.5 to 5.5 hour half-lives are the statistics a transmission-rights desk cares about, and the
  backtest shows the paper harvest concentrated in the 2021-22 congestion blowout.
* The battery LP is the directly reusable output: honest physics, settlement in the market each
  strategy schedules in, easy to point at any hourly price vector.
* Q1 is a useful negative: German renewable forecast errors are real but second-order for
  next-day auction prices, so public forecast data alone is not a day-ahead trading edge.

## Where this falls over

Things that affect the headline numbers:

1. The panel is three zonal day-ahead prices wearing historical intraday names. No intraday
   price appears anywhere in the repo, so the original intraday questions stay open.
2. Bidirectional Granger at this sample size implies confounding. Load forecast errors and
   weather regimes drive both series; adding load as an exogenous VAR term is the standard fix
   and is not done.
3. The formal spread tests run on daily means of a three-year window (n = 1,096), where both
   legs are individually stationary, so they measure mean reversion rather than cointegration.
   The half-life comes from a closed-form AR(1) on the full hourly sample.
4. Bai-Perron supF p-values use a pointwise F(2, .) reference, anti-conservative for a supremum.
   Dates and coefficients are exact.
5. The IRF's Cholesky ordering is a choice. A sign-restriction SVAR would firm it up, and the
   positive bump at hours 1 to 4 may be partly an ordering artifact.

Things that affect the Q2 backtest:

6. Not executable as specified. Capturing zonal spreads needs transmission rights; the costs
   are placeholders.
7. Regime concentration: pessimistic-cost Sharpe 1.80 (p = 0.025) with 95% of PnL from 2021-22
   and losses in 4 of 9 years. The entry gate is near non-binding.
8. Sharpe uses sqrt(252) by convention; sqrt(365) runs about 20% higher. Both are reported.

Things that affect the Q3 and battery numbers:

9. The dispersion regressor is degenerate before October 2025; the effect is identified on
   about seven months of data.
10. DA-LP is a within-day hindsight benchmark, and there are no bid-ask or auction-impact costs
    in the simulation.
11. Energy arbitrage only, no ancillary revenue stack.

And generalisability: one sample of one corner of Europe, 2018-2026, containing one
once-a-decade gas crisis that dominates several results.

## Revision history (July 2026)

A full pass over this repo before internship applications found one large error and a set of
methodological bugs. The short version is that the methods aged better than my data labels did.
Everything below is fixed in code, every artifact under `results/` is regenerated from the
corrected pipeline (window pinned to 2018-10-01 through 2026-05-04, data from the local parquet
cache), and the old headline numbers are recorded here instead of being overwritten. Tests went
from 56 to 67.

**1. The SMARD price series were misidentified.** The original version mapped SMARD filter 4996
to "intraday continuous VWAP", 252 to "ID3 auction", 251 to "ID1" and 5078 to "quarter-hourly
intraday index". Per the SMARD OpenAPI specification
([bundesAPI/smard-api](https://github.com/bundesAPI/smard-api)) they are actually: 4996 Belgian
day-ahead, 252 Danish DK1 day-ahead, 5078 neighbouring-zone day-ahead, and 251 does not exist
(hence an all-NaN "ID1" column that got explained away instead of investigated). Verified
against the data, not just the spec: the "ID3" series equals the DE/LU day-ahead price to the
cent in 59% of hours, including the 936.28 EUR/MWh Dunkelflaute print of 2024-12-12, which no
continuous-trading VWAP could produce; the quarter-hourly series has exactly zero intra-hour
variation until precisely 2025-10-01, the SDAC 15-minute go-live
([EPEX](https://www.epexspot.com/en/news/market-coupling-steering-committee-confirms-go-live-15-minute-mtu-sdac-trading-day-30));
and the real EPEX ID1/ID3 are ex-post indices of continuous trades, not auctions
([EPEX indices](https://www.epexspot.com/en/indices)). The actual intraday auctions are the
IDAs, launched June 2024 ([ENTSO-E](https://www.entsoe.eu/network_codes/cacm/implementation/ida/)).
The lesson is not subtle: check what a filter code returns before building three research
questions on it. A related off-by-one in the generation filter table (the "solar" filter fetched
onshore wind) is also fixed; those series were fetched but never used in any analysis.

**2. The IRF bootstrap was not a bootstrap.** It computed resampled residuals, never used them,
simulated Gaussian series of length 2,000 regardless of the 66,000-observation sample, and fell
back to the point estimate silently on failure. It is now a real residual bootstrap at full
sample length that fails loudly. The old conclusion ("90% band never excludes zero") flips: the
band excludes zero at almost all horizons. The economic half of the old conclusion stands, since
the FEVD share is at most 1.1% (now actually exported; the old text said "under 5%" without
shipping the number).

**3. Q3 standard errors.** The dispersion regression previously reported t = -30.8 from plain
OLS on 56,000 serially correlated hourly observations. With Newey-West (24 lags) it is -9.7,
same coefficient.

**4. Q2 reframed and a Johansen misreading fixed.** "Auction versus continuous price discovery"
became "zonal day-ahead spreads under market coupling" (see item 1). Johansen rank 2 had been
quoted as support for cointegration; in a bivariate system it means both series are stationary,
which matches the ADF results on the daily window. The old headline "the auction is the price
taker" is retired; what survives is a descriptive statement about which zonal price adjusts
faster. Also disclosed: the formal tests run on a daily-resampled three-year window, and the
VECM coefficients are per-day.

**5. Backtest honesty.** The "pre-registered" label is dropped: code and results landed in a
single initial commit, so the claim is not verifiable, and the committed spec file described an
earlier abandoned strategy anyway. "75% of lifetime PnL from 2021-22" was an editorial round
number; the per-scenario shares are 49%, 59% and 95%. `n_trades` became `n_traded_days`, which
is what it counts. An earlier backtest that marked the spread at hourly frequency lives on in
`results/deprecated_q2_backtest_hourly_marks/` with its own DEPRECATED.md, as a worked example
of manufacturing performance by marking an untradeable series.

**6. Battery corrections.** Day-ahead strategies used to settle at the mislabelled Belgian
series, a schedule-in-one-country-settle-in-another hybrid; they now settle at the DE/LU price
they schedule against, which moves the headline from 4.71M to 5.64M EUR/yr. The naive baseline
could discharge energy before charging it; it now walks the day in clock order (3.92M to 3.91M).
The "perfect foresight" strategy was never a foresight ceiling, just the Belgian curve, and is
renamed accordingly.

**7. Bai-Perron degrees of freedom.** The supF statistic now uses two restrictions per break
(intercept and slope both shift; previously one), and the p-values carry an explicit
anti-conservativeness caveat. Break p-values move from 3.7e-3 to 1.5e-2 and from 1.5e-5 to
8.8e-5; dates and coefficients are unchanged.

**8. Reproducibility.** All runners used "today minus 3 days" as the sample end, so every run
was date-dependent; the end date is now pinned in `power_coupling/runconfig.py` with `--end`
overrides. An environment-variable typo fallback is gone, ENTSO-E cache hits return the same
type as live calls, and two never-imported dependencies are removed.

## If I come back to this

Roughly in priority order:

* Get actual German intraday data. The EPEX ID1/ID3 indices and order-book data are paid;
  ENTSO-E publishes some intraday prices per zone worth evaluating first. Then the original
  intraday questions become answerable.
* Add load forecast error as an exogenous VAR term (closes the bidirectional-Granger objection).
* Interact the Q1 coefficient with TTF gas prices directly instead of arguing from break dates.
* Implement proper Bai-Perron critical values for the supF sequence.
* Re-run the Q3 regression as the post-Oct-2025 quarter-hourly sample grows; it is the one
  place the repo has a genuine granularity signal, and it is seven months old.
* Pull transmission-rights auction results to price the executability of the zonal spread.
* Threshold VECM for asymmetric mean reversion in the spreads.
