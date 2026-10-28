# Q007: ForecastEx breadth scan (EXPLORATORY, 2026-10-28)

This is a hypothesis generator, not a trial. No orders were placed and nothing was added to the ledger.
- **Data:** read-only public data only.
  - ForecastEx: `forecastex.com/api/download?type=prices` daily files and `/api/products`.
  - Kalshi: public `historical/markets` and hourly candlesticks.
- **Window:** daily files dated 2024-08-01..2026-08-14. Only contracts **expiring before 2026-08-15** were kept. Nothing at or after the FX1 sealed window was read or stored, and no UH contract with an event date of 08-15 or later was kept.
- **Code:** scratch only, not committed: `~/.claude/jobs/588c8b26/tmp/code/` (`pull.py`, `products.py`, `build.py`, `analyze.py`, `kmarkets.py`, `match.py`, `kcandles.py`, `compare.py`). Outputs are in the same `tmp/` (`calib.json`, `compare.out`, `xvenue_paired.parquet`).
- **Granularity limits** (same as E5): the daily files give OHLC of trades, `pair_quantity`, OI and settlement. They have **no bid/ask and no trade timestamps**, so every "price" below is a last trade, not a fill.

## 1. Catalog (dev window)

- **Scope:** 968 products in the live catalog: Elections 602, Economic 186, Environmental 93, Financial 61, Government 18, other 8.
- **Settled sample:** about 196k settled contracts. 533k YES rows were kept (traded days plus settlement days).
- **Volume units:** `pair_quantity` is in pairs. It is very large on promoted products, which IBKR/ForecastEx incentive programs probably inflate. Use it as a relative measure only.

Expiry range is for settled contracts expiring before 2026-08-15. `pairs` sums `pair_quantity` over traded days.

| family | example codes | settled contracts / events | expiries | pairs traded | median pairs per traded contract | still listed? |
|---|---|---|---|---|---|---|
| Elections | PRESH/PREST (2024 pres), MNYCG, GRNJ/GRVA, primaries | 472 / 164 | 2025-01..2026-08 | 6.3e8 (mostly PRESH/PREST) | 330 | live = 2026-28 races, unsettled |
| Daily high, current (UH*, 24 cities) | UHLAX | 45,181 / 4,422 | 2026-01..08 | 1.5e8 | 876 | yes |
| Daily high/low, legacy (DH*/DL*) | DHLAX | 13,720 / 1,667 | 2025-10..2026-02 | 1.5e8 | 8,742 | delisted |
| Daily low (UL*) | ULATL | 31,466 / 2,976 | 2026-02..08 | 2.6e6 | 60 | yes |
| Sports game winner | NFLGW, NBAGW, CBMGW | 1,324 / 663 | 2025-11..2026-03 | 2.5e8 | 1.6e6 | **delisted** |
| Financial markets | FES, FNQ, IN* indices, FX pairs | 56,775 / 3,465 | 2025-04..2026-08 | 4.6e7 | 529 | yes |
| Crypto | CF*/CB* | 12,748 / 873 | 2025-11..2026-08 | 1.7e7 | 300 | yes |
| Fed | FF, FFDEC, FEDRC | 198 / 27 | 2024-09..2026-07 | 8.3e6 | 862 | yes |
| US inflation | CPIY, CPIC, MMCPI, PCEY, PPIY, CPCEY | 1,191 / 114 | 2024-08..2026-08 | 7.6e5 | 320 | yes |
| US labour | PREMP, UNR, IJC, JOLT | 1,528 / 171 | 2024-08..2026-08 | 7.8e5 | 283 | yes |
| Other econ | GDP, housing, retail, oil/gas, regional CPI, gas prices | 15,943 / 1,736 | 2024-08..2026-08 | 3.9e6 | 190 | yes |
| Climate / environment | GT, GTM, ACD, GSL, emissions | 13,178 / 1,485 | 2024-10..2026-08 | 5.8e5 | 100 | yes |
| Monthly city temp/rain | OA*/TR* | 217 / 27 | 2025-10..2026-08 | 2.6e4 | 440 | yes |
| Other (delisted misc.) | commodities, FX legacy, pope, etc. | 1,717 / 222 | 2024-09..2026-07 | 4.3e7 | 285 | mixed |

**Liquidity reality check:** the median day of trading on an econ contract is 20-50 pairs, i.e. $20-50 of notional.

## 2. Calibration by family (last trade at the expiry date minus h days, h in {1, 7, 30}; staleness of 7 days or less; event-clustered bootstrap)

**Definitions:**
- `slope`: OLS of outcome on price. Above 1 means favourites are underpriced and longshots overpriced, i.e. a favourite-longshot bias (FLB).
- `fav`: buy whichever side trades at 0.80 or more, at last trade + $0.01.
- `fav_worst`: the same bet at the day's *worst* trade price (the E5 T2 analogue).

| family, h | n / events | slope [95% CI] | fav ROI [CI] | fav_worst | YES longshot overpriced (1-20c) |
|---|---|---|---|---|---|
| ALL ex-weather h1 | 17,770 / 3,373 | 1.076 [1.067, 1.085] | +2.5% [2.0, 2.9] | +1.8% | +3.9c |
| ALL ex-weather h7 | 5,011 / 1,249 | 1.071 [1.054, 1.088] | +2.2% [1.4, 3.0] | +1.9% | +3.6c |
| wx daily high h1 (same-day) | 35,183 / 4,228 | 1.082 | +3.2% [3.0, 3.4] | +1.7% | +5.1c |
| wx daily low h1 | 15,086 / 2,354 | 1.089 | +3.4% | +2.7% | +4.4c |
| fin markets h1 (FES/FNQ/indices) | 10,751 / 1,851 | 1.092 | +3.2% [2.6, 3.7] | +2.3% | +4.6c |
| crypto h1 | 221 / 52 | 1.247 | +9.5% [8.3, 10.9] | +9.1% | +9.7c |
| fed h1 / h30 | 79 / 26; 88 / 25 | 1.06 / 1.12 | +1.9% / +3.6% | +1.5% / +3.3% | +3.5c / +5.9c |
| US inflation h7 | 368 / 86 | 1.087 [1.04, 1.13] | +3.1% [0.9, 5.2] | +3.0% | +6.4c |
| US labour h7 | 727 / 141 | 1.115 [1.06, 1.16] | +3.4% [1.2, 5.4] | +2.9% | +4.6c |
| IJC (claims) h30 | 217 / 48 | 1.24 [1.14, 1.35] | +7.2% [3.3, 11.0] | +7.0% | +7.9c |
| sports game-winner h7 | 92 / 47 | 1.08 | +3.0% [2.0, 4.2] | +1.9% | +4.1c |
| other econ h7 | 2,748 / 705 | 1.05 | +1.1% [-0.1, 2.3] | +0.9% | +2.7c |
| climate/env h30 | 171 / 54 | 1.13 | +4.9% [3.0, 7.6] | +4.9% | +5.0c |

**Read:**
- **There is an FLB on every family.** The slope is 1.05-1.12 and favourites earn +1.5-3.5% at last trade.
- **It is the same size as Kalshi's FLB at the mid**, which is +2.3-4.5% in `smallmarket_calibration.md`. On Kalshi that bias nets to about 0 after the spread.
- **The costs have not been seen.** Without FX quotes, "+2% at last trade" says nothing about what you would pay to buy.
- **About 40 cells were examined**, so expect several CI-excludes-0 cells by chance.

## 3. ForecastEx vs Kalshi, same event and same strike (econ)

**Matching:** contracts were matched on release date and strike. The match was verified by the settled outcome agreeing with Kalshi's:

| ForecastEx | Kalshi | outcome agreement |
|---|---|---|
| CPIY | KXCPIYOY | 142/142 |
| CPIC | KXCPICOREYOY | 114/114 |
| MMCPI | KXCPI | 30/30 |
| PREMP | KXPAYROLLS | 41/41 |
| UNR | KXU3 | 104/104 |
| IJC | KXJOBLESSCLAIMS (`>=` vs FX "exceed") | 248/252; the 4 disagreeing rows were dropped |
| FF | KXFED, with the FX strike = Kalshi strike - 0.125 | 115/115 (the -0.125 / 0 offsets gave 86% / 0 matches) |

**Paired sample:** each FX trade day at lags of 1-30 days, against the Kalshi hourly book at 23:00 UTC that day, two-sided with a spread of 20c or less. That gives **751 pairs over 92 release events** (2024-08 to 2026-08). The median Kalshi spread is 3c and the median FX day is 30 pairs.

- **There is no level gap.** Mean FX minus Kalshi mid is +0.4c [-0.4, +1.1].
- **FX is a compressed copy of Kalshi:**

  | Kalshi mid | FX minus Kalshi mid |
  |---|---|
  | ≤ 0.30 | +2.4c |
  | 0.7-0.9 | -2.2c |
  | > 0.9 | -1.4c |

  This is the E5 pattern (FX favourites cheaper), at about 2-3c against weather's about 9c.
- **Kalshi is the better forecaster.** On the same days the Brier score is 0.0657 for the Kalshi mid and 0.0727 for the FX last trade. The difference CI is [+0.002, +0.012]. The FX last trade sits outside Kalshi's book on 60-75% of days; part of that is the unknown FX trade time.
- **The symmetric cross-venue rule is noise.** It takes the cheap side on FX whenever the FX last trade is outside Kalshi's book by t or more:

  | t | bets | ROI [CI] |
  |---|---|---|
  | 0 | 106 | -0.8% [-15, +13] |
  | 3c | 69 | +10% [-12, +31] |
  | 5c | 45 | +9% [-18, +32] |
  | 10c | 17 | +34% [-5, +72] |

  Every CI straddles 0. The longshot-side legs lose.
- **The Kalshi-anchored favourite leg is the one strong-looking cell.**
  - Rule: Kalshi mid ≥ 0.85 on a side, FX last trade for that side at least 2c below the Kalshi mid, first signal per contract.
  - Result: **64 bets / 57 events, 0 losses, mean FX cost 0.868, ROI +16.2% [13.5, 18.9] at FX last trade + 1c.**
  - The same bets bought at the Kalshi ask make +4.7%.
  - At the Kalshi mid about 5 losses were expected, and 0 occurred. With strikes from one release correlated, the effective n is about 57 or less.
  - Without the 2c filter: 91 bets, 0 losses, +12.0%.
  - By product: IJC 18, CPIY 16, FF 11, CPIC 9, UNR 8, PREMP 2.
  - Signals per quarter: 4 / 4 / 7 / 9 / 12 / 1 / 8 / 17 / 2, so **about 40-50 a year**.
  - Median FX day size 46 pairs; median lag 15 days.

## Ranked hypotheses

**H1 (best; proposed as Q008): Kalshi-anchored favourites on ForecastEx econ contracts.**
- **Mechanism:** this is E5 generalised. ForecastEx is a thin, IBKR-retail venue, so its prices are a compressed copy of Kalshi's. A side Kalshi prices at 0.85 or more trades several cents cheaper on FX, and the Kalshi FLB adds about +4% more on top.
- **Effect size:** +16% dev at last trade. Any real fill is worse. A realistic guess, if the FX ask sits near the Kalshi mid, is +3-7%.
- **Capacity:** about $25-50 per signal-day at observed volume, and about 40-50 signals a year. That is tiny: roughly $100-300 a year of edge even if it is real.
- **Sample needed:** break-even loss rate is about 1 minus cost, roughly 10%. To get the Wilson upper bound on the loss rate below the break-even rate with a +5% true edge, about 100 bets are needed, which is **about 2 years forward**. A forward test is only worth running cheaply, bolted onto the existing FX1-fwd logger.
- **Pre-registration sketch:**
  - **Universe:** FX CPIY, CPIC, MMCPI, PREMP, UNR, IJC and FF, matched to the Kalshi series above using the frozen strike map (FF offset +0.125) and release date.
  - **Daily snapshot at 15:00 CT,** for every matched pair with 1-30 days to release:
    - Kalshi bid/ask.
    - The FX side's ask and size, from `scripts/venues/forecastex_client.py` public quotes.
  - **Signal:** Kalshi mid ≥ 0.85 for the side, and the FX ask for that side ≤ Kalshi mid - 0.02, with at least 10 contracts at the ask.
  - **Bet:** paper-buy 1 unit at FX ask + $0.01. Only the first signal per contract counts.
  - **Metric:** ROI per $ at the ask, with an event-clustered bootstrap.
  - **Pass** (all of these): at least 100 bets and 40 events; p < .05 one-sided; ROI > 0; Wilson 95% upper bound on the loss rate < 1 − mean cost.
  - **Kill:** 3 or more losses in the first 30 bets. Kill also if fewer than 25% of Kalshi-favoured days have an FX ask that meets the filter, which would mean the dev gap was a last-trade artifact.
  - **Secondary, descriptive:** the median FX ask minus FX last trade. This calibrates how much of every last-trade backtest here is real.

**H2: a venue-wide favourite-longshot bias on ForecastEx at the quote.**
- **Mechanism:** the same retail FLB as Kalshi and Polymarket.
- **Effect:** +1.5-3.5% at last trade and +1-2% at the worst trade of the day. It is probably zero net, as on Kalshi.
- **What to do:** no trial. Have the H1 logger record the FX bid/ask on **every** contract priced 0.80 or more. After about 1 month this gives the cost to cross, and if the median spread is 2c or less, propose a dev test.

**H3: the claims (IJC) FLB at 30 days (slope 1.24, fav +7% [3, 11]).**
- **Mechanism:** a weekly release with strikes listed far out; retail anchors on the last print.
- **Caveats:** 48 events; it is one of about 40 cells; and the staleness filter allows a week-old trade.
- **What to do:** fold IJC into H1 (it is already its largest product). No separate trial.

## Do not pursue

- **Crypto (CF*/CB*) and financial-market dailies (FES/FNQ/indices).** These show +3-9.5% fav ROI and slopes of 1.09-1.25. Their underlying is continuous, so a morning last trade compared with a 16:00 settlement is a stale-price artifact the daily files cannot rule out. Kalshi and CEX have the same products with better books.
- **Sports (NFLGW/NBAGW/CBMGW).** Listed only Nov 2025-Mar 2026, then delisted. Reported volumes are incentive-inflated. Pinnacle-vs-soft-book work (Q-soccer) dominates anyway.
- **Elections.** The settled dev sample is the 2024 presidential plus small primaries, about 23-95 events. The live catalog is 2026-28 races that settle long after any useful horizon. IBKR's incentive coupon, which pays interest on positions, also distorts long-dated prices.
- **Climate / long-dated environmental contracts** (GT, ACD, sea level, emissions). The h30 FLB is +4.9%, but on 54 events, with multi-month capital lock-up and the incentive coupon confounding it. Too few events a year to ever test.
- **The symmetric "FX outside the Kalshi book" arbitrage.** The CI straddles 0 at every threshold, and the longshot legs lose.
- **Weather lows (UL*).** The FLB matches the highs, but Kalshi lows (X2) already failed dev, and E5/FX1 covers the high-temperature mechanism.

## Caveats that apply to everything above

- **Last trade is not an executable price.** The daily files have no timestamps.
- **The ForecastEx trade-day boundary is unverified.**
- **The Kalshi snapshot time (23:00 UTC) does not match the FX trade time.**
- **The FLB statistics use an unweighted contract mix,** with strikes of the same event correlated. The CIs resample whole events, but they do not cluster by release date across products (for example, CPIY and CPIC on the same day).
