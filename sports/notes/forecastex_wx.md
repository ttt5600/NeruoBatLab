# ForecastEx (IBKR) weather: E1/E2 consensus-reject port, DEV backtest (2026-10-28)

## Verdict
**The rule clears the p-value bar on DEV, but this is not a confirmed edge.**
- **Numbers:** 159 bets, +20.9u, ROI +13.2%, p .0004 (bar .025). The conservative pricing variant (T2) also clears: +14.4u, ROI +9.1%, p .001.
- **Not resolvable by our own pre-registration.** n = 159 is below the pre-registered floor of 500, so it gets reported as "not resolvable". The profit sits in a thin market priced only from **last trades, not quotes**: mean cost was 0.88, against about 0.97 for the same kind of pick on Kalshi.
- **Probably a different edge from Kalshi E1.** Weather Underground's max sits 1-2F below the NWS value on about 2/3 of days.
- **Ask the user: YES to a sealed test** (Aug 15 to Oct 26, under the new settlement wording). It is a free, one-shot test. **Do not open an IBKR account yet.** The account matters only afterwards: the clean next step is a forward paper test on live bid/ask, and only an IBKR login can supply those quotes.

## What was tested (pre-registered in `scripts/forecastex_wx.py`)
- **Pre-registration timing:** the docstring was frozen (sha256 `8a6edbb5…`, 06:10 UTC) *before* any price file was pulled.
- **Rule:** frozen E1. Same 5 Open-Meteo lead-1 models, read at each Weather Underground (WU) airport, with a 3F margin.
  - The contract is "high EXCEEDS X" (from the CFTC filing).
  - When all models put the high at ≤ X-3, buy NO.
  - When all models put the high at ≥ X+4, buy YES. This is E1's low-tail NO in a different form.
- **Scope:** 24 cities, daily highs only.
- **Split by event date:**
  - DEV: 2026-03-01 to 08-14 (70%).
  - SEALED: 08-15 onward. `pull` never writes a sealed contract to disk.
- **Pricing:** the bought side's trade price from the daily file dated D-1, only if that side traded that day, plus a $0.01/contract fee.
  - T1 uses the last trade.
  - T2 uses the day's *highest* trade, the worst price any buyer paid.
  - Two trials, so the bar is .05/2.
- **Granularity limit:** the daily files have no bid/ask and no timestamps.
  - The trade-date boundary is UNVERIFIED. It is probably Central time, so the D-1 last trade comes before 22:00 local (stale, no lookahead) or at most about 2 h after it in the East.
  - A last trade is not a fill we could have had, which biases T1 upward. T2 bounds that bias, and it still passes.

## DEV results

| | T1 last trade | T2 day-high |
|---|---|---|
| Bets / station-days | 159 / 101 | 159 / 101 |
| Losses (rate, Wilson 95%) | 6 (3.8%, 1.7-8.0%) | 6 (3.8%) |
| Break-even loss rate | 12.5% | 11.1% |
| Units, ROI | +20.92, +13.2% | +14.41, +9.1% |
| Bootstrap 95% CI | [+8.5, +37.6] | [+6.0, +22.3] |
| p | .0004 | .0010 |
| Halves (H1 / H2) | +2.13/31 bets, +18.79/128 | +1.99/31, +12.43/128 |
| NO / YES side | +14.78/120, +6.14/39 | +13.53/120, +0.88/39 |

**Reasons not to trust the result yet:**
- **D-1 trading barely existed before July.** Contracts that traded the day before, by month:

  | Month | Contracts traded D-1 |
  |---|---|
  | Mar | 321 |
  | Apr | 30 |
  | May | 0 |
  | Jun | 0 |
  | Jul | 1,544 |
  | Aug (half month) | 1,332 |

  So 128 of the 159 bets fall in Jul-Aug 14. Effectively this is one month of evidence.
- **The market is thin.** Median D-1 size on a picked contract is 26 pairs.
- **The profit comes from cheap prices.**
  - Bets priced below 0.80 made +10.0u from 27 bets with 4 losses. Some of these are bizarre, such as MDW 7/31 ">77" YES at 0.21 when the models said 82+.
  - Bets priced 0.80-0.98 made +10.9u from 132 bets with 2 losses.
  - Whether those prices were executable is exactly what daily files cannot show.
- **The profit is concentrated.** MDW gave +7.3u from 10 bets.

## Independence from Kalshi E1/E2
- **W1, settlement source:** 2,878 station-days had both venues settled at 18 shared airports.
  - On 64% of them, Kalshi's NWS/TWC value falls outside the WU interval implied by ForecastEx outcomes. Kalshi was higher on 1,826 of those days and lower on 18.
  - On days where WU is pinned exactly (a 6-airport sample), Kalshi minus WU was: 0F on 32%, +1F on 55%, +2F on 12%, +3F on about 1%.
  - So WU settles systematically lower than NWS, consistent with an hourly-observation max against the CLI's continuous max. That shifts the outcomes, which helps NO-above picks and hurts YES picks.
  - Caveat: this assumes "exceed" means strictly greater. If ForecastEx actually settles at ≥, the offset shrinks by 1F. The filing text says "exceed".
- **W2, overlap of picks:**
  - Of 62 ForecastEx pick station-days at shared airports, 46 also had a Kalshi E1/E2 pick, because both use the same forecasts. Pick days are therefore highly correlated.
  - Of the 5 ForecastEx losing days, 2 were also Kalshi losing days.
  - Partly independent, on very small n.

## Next steps (needs the user)
1. **Sealed test, "yes FX1":**
   - Write `sealed --confirm` against the unchanged docstring.
   - Pull files from 08-15 onward and score T1 once.
   - Pass if p < .05 and units > 0, and report n against the 500 floor.
   - The sealed window has more D-1 trading and the new Daily-Observations wording, so it is the relevant regime.
2. **Only if the sealed test passes:** open an IBKR Pro account with ForecastEx permission (CA eligibility UNVERIFIED) and run a forward **paper** logger on live bid/ask at 22:00. That is the first test that prices at a quote.

Files:
- `scripts/forecastex_wx.py`
- `data/forecastex/uh_*.csv` (DEV only, 17 MB, gitignored)
- `data/forecastex_wx.parquet`
- Ledger: 2 rows in `lab/ledger.jsonl`, dataset `forecastex_wx`. They are marked "pass" on p and do not record the under-power flag.

## ForecastEx sealed pre-registration (FX1, written 2026-10-28, before any sealed file was downloaded)
The user said "yes ForecastEx" on 2026-10-28. The full text is the docstring of `scripts/forecastex_wx_sealed.py` (sha256 at writing `6c0e0ac9…`). Summary:
- **Code:** the exact frozen DEV path. The new script imports `scripts/forecastex_wx.py` unchanged (and aborts if its sha256 is not `8a6edbb5…`). It re-points only DATA, OUT, FIRST_EVENT and SPLIT to the sealed window. Rule, margin, models and fee are untouched.
- **Window:** contracts for events 2026-08-15 to 2026-10-26. Price files dated 08-14 to 10-27 are downloaded, and only in-window UH rows are kept. Settled only: a contract counts only with a final YES settlement of exactly 0 or 1. Unsettled contracts and non-monotone station-days are dropped and counted.
- **New settlement wording (WU Daily Observations), decided before seeing data:**
  - The exchange's own settlement_price is the truth. No outcome is re-derived from WU or NWS.
  - The rule applies unchanged: "exceed" means strictly greater, with no WU-vs-NWS offset.
  - Everything is pooled into one test, with no split or exclusion by wording era. The effective date is unverified, so the by-month split is descriptive only.
  - A schema gate checks tickers and columns only: at least 20 cities and 60 event dates. If it fails, the run stops with "schema_abort" and no P&L is computed.
  - A "settlement_anomaly" flag is raised if more than 5% of station-days are non-monotone or more than 5% of contracts are unsettled. It is descriptive and does not change the verdict.
- **Primary (T1):** the last D-1 trade plus $0.01. One-sided station-day bootstrap (5,000 resamples, seed 0), with p < .05 and units > 0. It also needs the **loss-rate check**: the Wilson 95% upper bound on the loss rate must be below the break-even loss rate (1 − mean cost). DEV only reported this check; it is now a gate. On DEV it reads 8.0% against 12.5%.
- **Secondary (T2):** the day-high trade plus $0.01, scored the same way. If T1 passes and T2 fails, the verdict gets the suffix "-FRAGILE".
- **Verdict ladder:**
  - FAIL: T1 has no bets, p ≥ .05, units ≤ 0, or the loss check fails.
  - PASS-UNDERPOWERED: T1 passes everything but has fewer than 500 bets.
  - PASS: T1 passes everything with at least 500 bets.
  - A plain PASS below 500 bets is impossible. No verdict authorises real money; even a PASS only buys the forward paper test on live IBKR bid/ask.
- **Expected n:** about 200-400, so PASS-UNDERPOWERED is the best realistic label.
- **One-shot guard:**
  - The run needs `--confirm-lockbox`.
  - It aborts if any of these exist: `results/forecastex_wx_sealed.json`, a lockbox entry `forecastex_wx_FX1`, or a non-empty `data/forecastex_sealed/`.
  - It writes `{"status":"started"}` before any download.
  - It records sha256 hashes of every sealed CSV, the built parquet and both scripts.
- **DEV reproduction (`devcheck`, 2026-10-28):** the full rebuild from `data/forecastex/` goes through the same picks/score/verdict code.
  - T1: 159 bets, +20.92u, p .0004.
  - T2: 159 bets, +14.41u, p .0010.
  - The pick set and P&L are identical to the committed parquet: MATCH (`results/forecastex_wx_devcheck.json`).
  - Under the sealed ladder, DEV itself would read PASS-UNDERPOWERED.
- **Run (one shot):** `../.venv_quant/bin/python scripts/forecastex_wx_sealed.py sealed --confirm-lockbox`
