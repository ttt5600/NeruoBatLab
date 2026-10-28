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
