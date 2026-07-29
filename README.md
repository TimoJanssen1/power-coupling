# power-microstructure

Applied econometrics on 7.6 years of free public power-market data: European day-ahead
market coupling, congestion spreads, and battery dispatch. Three questions, each taken from
mechanism to method to a number with an error bar.

The project began as a study of the German intraday market; a July 2026 revision established
that its SMARD price series are in fact day-ahead prices of neighbouring bidding zones, and
every claim was relabelled, corrected or retired accordingly — see
[FINDINGS.md, Revision notes](FINDINGS.md#revision-notes-july-2026). The repository name
predates the revision.

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-67%20passing-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

![Zonal spread by hour of day](results/figures/q3_shape_by_hour.png)

The Belgian day-ahead price runs ~€10/MWh rich against the average of DE/LU's neighbouring zones
at the evening ramp and ~€4/MWh cheap at the solar peak — every day, for seven and a half years.
v1 read this figure as "the duck curve as a tradeable hourly-vs-quarter-hourly spread"; it is
actually a cross-zonal day-ahead profile difference. Still a clean, stable structural fingerprint
recovered from public data — of a different mechanism than advertised.

## Findings, short version

| Q | Original hypothesis | Where it landed |
|---|---|---|
| Q1 | Renewable forecast errors drive prices, and the link grows with renewable build-out. | The link exists and — after fixing a broken bootstrap — is statistically significant (IRF trough −1.26 €/MWh at h=11, 90% band [−1.49, −1.08]) but economically small: FEVD ≤ 1.1% of price variance. Structural breaks land on the gas-crisis bookends (2021-07, 2022-12), not on capacity events: **β tracks gas, not build-out.** |
| Q2 | Auction and continuous prices cointegrate; the auction discovers price. | The two "venues" turned out to be **DK1 and Belgian day-ahead prices** — so this became a market-coupling study. Zones clear identically to the cent in 59% (DE–DK1) / 31% (DE–BE) of hours; congestion spreads mean-revert with 3.5–5.5 h half-lives. A strictly-lagged paper backtest on the DK1−BE spread: Sharpe 5.95/3.89/1.80 by cost scenario (p ≤ 0.025), with 49%/59%/95% of PnL from 2021–22 — a gas-crisis congestion harvest, and not executable without transmission rights. Numbers in [FINDINGS.md](FINDINGS.md#q2-zonal-day-ahead-spreads-and-the-backtest). |
| Q3 | Hourly vs four 15-min QH prices should price intra-hour ramp risk. | The QH leg carries **zero intra-hour information before Oct 2025** (hourly values replicated 4×), so the celebrated profile above is cross-zonal, not a shape premium. The σ regression survives at −0.91 with HAC t = −9.7 (was −30.8 under plain OLS) but is identified on ~7 months of true QH data. The battery LP is the durable output: DA-arbitrage ceiling **€56.4k/MW/yr** at 100 MW/200 MWh, +45% over a naive heuristic, and the Q1 forecast-error tilt adds nothing. |

Deep version of all this — including what was wrong and how it was caught — in
[FINDINGS.md](FINDINGS.md).

## Running it

```bash
# 1. clone + install
git clone https://github.com/TimoJanssen1/power-microstructure.git
cd power-microstructure
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. set your free ENTSO-E API key (the platform takes 1 to 3 working days)
export ENTSOE_API_KEY="your-key-here"

# 3. fetch data (~45 min first time, instant after; parquet cached)
python scripts/fetch_full_history.py

# 4. run the three econometric analyses (~3 min from cache)
python scripts/run_full_analysis.py

# 5. run the Q2 walk-forward backtest (~10 s)
python scripts/run_q2_honest_backtest.py

# 6. run the BESS dispatch simulation (~1 min, ~2 700 daily LPs)
python scripts/run_q3_battery.py

# 7. open the notebooks
jupyter lab notebooks/
```

Every figure and number in the write-ups regenerates from these commands. The sample end date is
pinned to 2026-05-04 (the shipped-artifact window) so runs are reproducible from cache; pass
`--end live` for a fresh window. SMARD needs no key; ENTSO-E does.

## What's where

The package is layered. Each layer depends only on the one above it. Notebooks read from
`results/`, never from raw data.

```
power_microstructure/
├── data/                I/O layer: raw market data → tidy DataFrames
│   ├── entsoe.py        ENTSO-E API wrapper (forecasts, generation, DA, load)
│   └── smard.py         SMARD wrapper — zonal day-ahead prices. Column names are
│                        legacy ("id3", "id_continuous", …); the series-identification
│                        table in the module docstring is required reading.
├── signals/             Feature layer: tidy DataFrames → research signals
│   ├── forecast_error.py    Q1: forecast errors and derivatives
│   └── spread.py            Q2/Q3: zonal spreads (legacy names documented in-module)
├── analysis/            Inference layer: signals → econometric tests
│   ├── granger.py           VAR / Granger / IRF (real residual bootstrap) / FEVD
│   ├── cointegration.py     ADF + KPSS / Engle-Granger / Johansen / VECM
│   └── structural.py        Bai-Perron with custom prefix-sum DP
├── strategy/            Application layer: signals → PnL
│   ├── backtest.py          Walk-forward, transaction costs, block-bootstrap p-value
│   └── battery.py           BESS dispatch LP (naive + 3 LP strategies)
└── runconfig.py         Pinned sample end date (reproducibility)

scripts/                 Runners: fetch, full analysis, honest Q2 backtest, Q3 battery
                         (run_q2_backtest.py is the deprecated hourly-marks version)
tests/                   67 passing tests on synthetic data, no API keys needed
notebooks/               Three EDA notebooks (outputs predate the July 2026 revision)
results/                 figures/, tables/, summary.json per question;
                         deprecated_q2_backtest_hourly_marks/ is quarantined, see its DEPRECATED.md
```

A few specifics worth pointing at:

* Every API call is sha-keyed on its parameters and stored as parquet under
  `~/.cache/power_microstructure/`. First fetch takes ~45 minutes; subsequent runs are seconds.
* ENTSO-E returns 504s on multi-year `query_generation` calls.
  `EntsoeFetcher.actual_generation_per_type` chunks into 60-day blocks and recovers from gateway
  timeouts.
* Bai-Perron at T > ~5 000 needs more memory than a laptop has if you precompute the RSS matrix.
  [`structural.py`](power_microstructure/analysis/structural.py) replaces the matrix with
  closed-form OLS over prefix sums (O(T·m) memory, exact global optimum).
  Brute-force-equivalence-tested on small T; scaling-tested at T = 20 000.
* [`battery.py`](power_microstructure/strategy/battery.py) is a real linear program. 72 variables
  per day, SoC dynamics with separate charge/discharge efficiencies, daily cycle cap, scipy/HiGHS
  solver. Same engine, different price vectors — and since July 2026, settlement happens in the
  market each strategy actually schedules in.

## Tests

```bash
pytest tests/
# 67 passed in ~100s
```

Synthetic data with known statistical properties. No API keys, no network. Each econometric tool
is checked against a series with the property it should detect (Granger on a series constructed
to be Granger-caused; cointegration on a pair sharing a stochastic trend; Bai-Perron DP against
brute-force enumeration; the IRF bootstrap against a known lagged effect; battery schedules
replayed for physical feasibility).

## Other reading

[FINDINGS.md](FINDINGS.md) is the longer write-up: the three questions, what the data turned out
to be, methods, results, limitations, and the July 2026 revision notes.

## License

MIT, see [LICENSE](LICENSE).

## Where the data comes from

| Source | Auth |
|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Free API key. Email `transparency@entsoe.eu`, subject *"Restful API access"*, body = your registered email. Token shows up at My Account Settings → Web Api Security Token after they reply, which takes 1 to 3 working days because they appear to do this by hand. |
| [SMARD (Bundesnetzagentur)](https://www.smard.de/en) | None, open HTTP. Filter-code semantics per the [bundesAPI/smard-api](https://github.com/bundesAPI/smard-api) OpenAPI spec — check them against the data; this repo's v1 didn't and paid for it. |

The ENTSO-E client is [`entsoe-py`](https://github.com/EnergieID/entsoe-py). Methodological prior
art is [Florian Ziel](https://dsee.wiwi.uni-due.de/team/florian-ziel/) and the
[House of Energy Markets and Finance](https://www.hemf.wiwi.uni-due.de/) at Duisburg-Essen.
