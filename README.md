# power-coupling

Econometrics on 7.6 years of free public power-market data (2018-2026): European day-ahead
market coupling, congestion spreads, and battery storage. Three questions, each taken from
market mechanism to method to a number with an error bar.

![Zonal spread by hour of day](results/figures/q3_shape_by_hour.png)

The Belgian day-ahead price runs about 10 EUR/MWh above the average of DE/LU's neighbouring
zones at the evening ramp and about 4 EUR/MWh below it at the solar peak, on a stable hourly
profile across the whole sample.

## Findings, short version

| Question | Result |
|---|---|
| Do renewable forecast errors move prices, and does the effect grow with renewable build-out? | A one-sigma forecast-error shock moves the price by about 1 EUR/MWh (IRF trough of -1.26 EUR/MWh at hour 11, 90% band [-1.49, -1.08]) and explains at most 1.1% of price variance at any horizon (FEVD). Bai-Perron breaks land on the gas-crisis bookends (Jul 2021, Dec 2022) with the impact coefficient rising about 5x between them; no break lands on a capacity milestone. The impact tracks the marginal fuel, not build-out. |
| How tightly does market coupling tie neighbouring zones together, and do congestion spreads mean-revert? | DE/LU and DK1 clear at identical prices to the cent in 59% of hours, DE/LU and Belgium in 31%. When spreads open they close with half-lives of 3.5 to 5.5 hours. A strictly lagged paper backtest on the DK1-Belgium spread reaches Sharpe 5.95 / 3.89 / 1.80 across three cost scenarios (bootstrap p <= 0.025), with 49% / 59% / 95% of lifetime PnL earned in 2021-22. Capturing it for real would require transmission rights. |
| What is intra-hour shape worth, and what does storage earn on the daily profile? | German quarter-hourly day-ahead prices exist since the 15-minute SDAC go-live (Oct 2025); before that the quarter-hourly series replicates the hourly price. On the true quarter-hourly sample, intra-hour dispersion prices at -0.91 EUR/MWh per unit of within-hour standard deviation (HAC t = -9.7). A 100 MW / 200 MWh battery earns 56.4k EUR/MW/yr on day-ahead arbitrage, 45% above a naive heuristic, and tilting its dispatch with the forecast-error signal adds nothing. |

The full write-up, with methods and limitations, is [FINDINGS.md](FINDINGS.md).

## Running it

```bash
# 1. clone + install
git clone https://github.com/TimoJanssen1/power-coupling.git
cd power-coupling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. set your free ENTSO-E API key (approval takes 1 to 3 working days)
export ENTSOE_API_KEY="your-key-here"

# 3. fetch data (~45 min first time, instant after; parquet cached)
python scripts/fetch_full_history.py

# 4. run the three analyses (~3 min from cache)
python scripts/run_full_analysis.py

# 5. run the Q2 walk-forward backtest (~10 s)
python scripts/run_q2_honest_backtest.py

# 6. run the battery dispatch simulation (~1 min, ~2,700 daily LPs)
python scripts/run_q3_battery.py

# 7. open the notebooks
jupyter lab notebooks/
```

Every figure and number in the write-ups regenerates from these commands. The sample end date
is pinned to 2026-05-04 so runs reproduce the shipped artifacts from cache; pass `--end live`
for a fresh window. SMARD needs no key; ENTSO-E does.

## What's where

Each layer depends only on the one above it. Notebooks read from `results/`, never from raw
data.

```
power_coupling/
├── data/                I/O layer: raw market data -> tidy DataFrames
│   ├── entsoe.py        ENTSO-E API wrapper (forecasts, generation, DA, load)
│   └── smard.py         SMARD wrapper: zonal day-ahead prices. Column names are
│                        historical ("id3", "id_continuous", ...); the table in the
│                        module docstring maps each one to what the series is.
├── signals/             Feature layer: tidy DataFrames -> research signals
│   ├── forecast_error.py    Q1: forecast errors and derivatives
│   └── spread.py            Q2/Q3: zonal spreads
├── analysis/            Inference layer: signals -> econometric tests
│   ├── granger.py           VAR / Granger / IRF (residual bootstrap) / FEVD
│   ├── cointegration.py     ADF + KPSS / Engle-Granger / Johansen / VECM
│   └── structural.py        Bai-Perron with a prefix-sum DP
├── strategy/            Application layer: signals -> PnL
│   ├── backtest.py          Walk-forward, transaction costs, block-bootstrap p-value
│   └── battery.py           Battery dispatch LP (naive + 3 LP strategies)
└── runconfig.py         Pinned sample end date (reproducibility)

scripts/                 Runners: fetch, full analysis, Q2 backtest, Q3 battery
tests/                   67 tests on synthetic data, no API keys needed
notebooks/               Three EDA notebooks
results/                 figures/, tables/, summary.json per question;
                         deprecated_q2_backtest_hourly_marks/ holds a retired backtest
                         design, explained in its DEPRECATED.md
```

Some implementation details:

* Every API call is sha-keyed on its parameters and stored as parquet under
  `~/.cache/power_coupling/`. First fetch takes ~45 minutes; afterwards runs are seconds.
* ENTSO-E returns 504s on multi-year generation queries, so the fetcher chunks them into
  60-day blocks and retries through gateway timeouts.
* Bai-Perron at T > ~5,000 needs more memory than a laptop has if you precompute the full RSS
  matrix. [`structural.py`](power_coupling/analysis/structural.py) replaces the matrix with
  closed-form OLS over prefix sums (O(T·m) memory, exact global optimum), tested against
  brute-force enumeration on small T and scale-tested at T = 20,000.
* [`battery.py`](power_coupling/strategy/battery.py) is a real linear program: 72 variables
  per day, state-of-charge dynamics with separate charge and discharge efficiencies, a daily
  cycle cap, scipy/HiGHS solver. The four strategies share the engine and differ in which
  price vector they schedule and settle against.

## Tests

```bash
pytest tests/
# 67 passed in ~100s
```

Synthetic data with known statistical properties, no network. Each tool is checked against a
series with the property it should detect: Granger on a series built to be Granger-caused,
cointegration on a pair sharing a stochastic trend, Bai-Perron against brute-force enumeration,
the IRF bootstrap against a known lagged effect, battery schedules replayed for physical
feasibility.

## Where the data comes from

| Source | Auth |
|---|---|
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) | Free API key. Email `transparency@entsoe.eu` with subject "Restful API access" and your registered email in the body. The token appears under My Account Settings after they reply, which takes 1 to 3 working days because they appear to do this by hand. |
| [SMARD (Bundesnetzagentur)](https://www.smard.de/en) | None, open HTTP. Filter-code semantics per the [bundesAPI/smard-api](https://github.com/bundesAPI/smard-api) OpenAPI spec. Verify codes against the data they return before trusting them. |

The ENTSO-E client is [`entsoe-py`](https://github.com/EnergieID/entsoe-py). Methodological
prior art: [Florian Ziel](https://dsee.wiwi.uni-due.de/team/florian-ziel/) and the
[House of Energy Markets and Finance](https://www.hemf.wiwi.uni-due.de/) at Duisburg-Essen.

## License

MIT, see [LICENSE](LICENSE).
