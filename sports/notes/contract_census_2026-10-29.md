# Q010 contract census across venues (2026-10-29)

Read-only public data. No orders, no keys, no account calls, no outcome backtests. Every number below was read
from the census CSV or the raw API caches; nothing is estimated except where a row says "extrapolated".

- Code: `scripts/contract_census.py` (`fetch [kalshi|pm|fx]`, `build`, `settled <series...>`).
- Raw caches (gitignored): `data/census/raw_2026-10-29/`. The raw files are `kalshi_series.json`, `kalshi_events_*.json` (82 pages),
  `pm_markets_*.json` (164 pages), `pm_bbo_sample.json` (1,758 BBOs), `fx_products.json` (963), `fx_prices_20261028.csv`
  and `kalshi_settled_event_counts.json`.
- Table: `data/census/census_2026-10-29.csv`. One row per family (6,466 rows), with columns `reference_forecast` and `fund_status`
  hand-annotated for the families assessed here. Everything else is marked "not assessed".

## How each column was measured (read before quoting the CSV)

| venue | family key | markets_open | vol24h_contracts | open_interest | spread |
|---|---|---|---|---|---|
| Kalshi | `series_ticker` | active markets in open events | sum of `volume_24h_fp` | sum of `open_interest_fp` | median (yes_ask - yes_bid) over two-sided markets |
| Polymarket US | first two slug tokens (e.g. `astatc-nfl`, `rotc-t`) | open markets | **not available** (the public gateway has no 24h field) | **extrapolated**: mean `openInterest` of <= 4 sampled BBOs x markets_open | median from list `bestBid/AskQuote` |
| ForecastEx | `product_id` | YES contracts listed in the 2026-10-28 price file | `pair_quantity` summed for 2026-10-28 (pairs, one day) | catalog `open_interest` (reads **0** for MORT1/IJC despite trades: unreliable) | none (no bid/ask in public data) |

Fees: Kalshi gives `fee_type` per series. Most are `quadratic` (taker 0.07·p(1-p), maker 0). Some are `quadratic_with_maker_fees`
(e.g. KXFEDDECISION, KXFED, KXCPI*, KXLLM1, KXINXY) and some are `x0`, i.e. fee-free (KXBTCY, KXGDPYEAR).
Polymarket US: every market's `feeCoefficient` = 0.0695, maker 0.
ForecastEx: $0.01/contract (notes/venue_expansion.md).

## 1. Census summary

| venue | families with open contracts | open markets | 24h volume | open interest |
|---|---|---|---|---|
| Kalshi | 5,021 (of 15,501 series in /series) | 167,407 in 16,228 events | 114,236,352 contracts | 1,398,758,000 (rounded) |
| Polymarket US | 482 | 81,666 | n/a | ~294M (extrapolated) |
| ForecastEx | 963 products | 12,113 listed YES contracts | 644,615 pairs (10-28) | 10.7M (catalog) |

Kalshi by category (families / open markets / 24h volume / OI):

| category | fam | markets | 24h vol | OI |
|---|---|---|---|---|
| Sports | 1,404 | 101,337 | 72,287,448 | 584,990,303 |
| Elections | 878 | 25,940 | 31,221,473 | 628,781,510 |
| Economics | 450 | 5,505 | 2,459,430 | 22,395,755 |
| Crypto | 80 | 4,048 | 2,338,032 | 29,899,776 |
| Climate and Weather | 204 | 1,733 | 1,670,058 | 2,604,915 |
| Entertainment | 341 | 9,808 | 1,304,863 | 32,815,957 |
| Politics | 730 | 3,053 | 1,230,794 | 43,106,985 |
| Financials | 533 | 10,444 | 566,855 | 20,184,021 |
| Science and Technology | 158 | 1,180 | 486,176 | 27,419,546 |
| Mentions | 158 | 2,378 | 445,929 | 882,805 |
| Commodities | 47 | 1,540 | 186,774 | 5,213,651 |
| Companies | 36 | 439 | 38,521 | 462,212 |

Kalshi series `frequency` among families with open events: one_off 2,302, custom 1,677, annual 553, monthly 193,
daily 148, weekly 87, hourly 25, fifteen_min 23, quarterly 13.

Polymarket US is 87% sports by market count (71,308 of 81,666). Its non-sports families are thin next to Kalshi's. The biggest
are `cpc-btc` (BTC one-touch, 582 markets), the election families, `tac-oscars`, `ccrc-sptfy` and `aimrc-company1`. Its weather (`tc-temp`)
is already covered by F4.

ForecastEx's top products by 10-28 pairs: FFDEC 205,728; MLAXG 79,325; FES 50,680; UHLAX 49,123; HORC 43,353; SENM 34,000;
IJC 30,030; FNQ 14,078; YXHBT 13,886; UHLGA 13,216; MORT1 12,300.

What I excluded as dead or covered (FUND_BOOK 3d/3a), with nothing in the census that changes the family:
- Sports on every venue. The prices are made off sharp books (M2, S1-S3, PMUS-vs-Pinnacle).
- KXHIGH* (live E1/E2/E4) and UH* (live E5).
- Lows: KXLOWT*, UL*.
- KXRAIN (R1 sealed FAIL) and KXAAAGASD (N6).
- CPI ladders (N2) and econ consensus-reject (claims/U3/payrolls/GDP/PCE, IJC).
- Mentions (M3, Q003), including the 131 KXEARNINGSMENTION* families.
- FX fin/crypto dailies (FES/FNQ, Q007).
- Elections (L18/M1 cross-venue already covered; settle after any useful horizon).

## 2. Top 15 families (excluding dead)

Score = R x L x C:
- **R** = how much better than the market the public reference plausibly is.
  - 3: the settlement variable is partly *observed* before close, or there is a published official forecast.
  - 2: a public pricing model exists that pros also use.
- **L** = Kalshi 24h volume: 3 is >= 100k, 2 is 10k-100k, 1 is < 10k.
- **C** = cadence: 3 is daily/hourly, 2 is weekly, 1 is a handful of events a year.

Ties are broken by 24h volume. "settled" = settled events from `/events?status=settled`. That count is a lower bound: older events may have
moved to Kalshi's historical tier, and KXBTCD hit my 60-page cap.

| # | group (CSV `fund_status` tag) | families | 24h vol / OI (lead family) | spread | settled | reference | R·L·C | why / caveat |
|---|---|---|---|---|---|---|---|---|
| 1 | G-CRYPTO | KXBTCD, KXETHD, KXBTC, KXETH | KXBTCD 1,076,097 / 1,126,159; KXETHD 105,190 | .01 / .02 | KXBTCD >= 12,000 | Deribit option IV + CF Benchmarks BRTI spot (settlement = 60-s BRTI avg) | 2·3·3=18 | Most capacity of any non-sports family; prior_research #6 says the mispricing is at listing and in path/tail strikes. **Competitive**: social_edges says HFT-dominated; L19 K-FAV fav-maker already died on these series (adverse selection) |
| 2 | G-RT | KXRT, KXMC; PM rotc-t | KXRT 348,192 / 2,348,307 (217 mkts, 18 events) | .01 | KXRT 85, KXMC 19 | live Tomatometer + review count; settlement value is partly observed for days before close | 3·3·2=18 | Biggest entertainment family; best "partial observation" structure. MarketWatch-profiled RT scraper *lost money* (social_edges). LIP programs active on KXRT |
| 3 | G-TRUTH | KXTRUTHSOCIAL, KXTRUTHSOCIALD | 61,484 / 111,880 | .01 | 37 weekly + 28 daily | running post count (settlement source Roll Call); count to date is public mid-window | 3·2·3=18 | A count process with most of the window observed before close |
| 4 | G-APPROVAL | KXAPRPOTUS, KXTRUMPAPPROVE, KXTRUMPVH, KXGENERICBALLOTVOTEHUB | 36,922 + 13,837 | .01 | 261 + 58 | RCP/VoteHub average is a deterministic function of listed polls; poll release cadence known | 3·2·3=18 | Settlement variable is recomputable; question is whether the price lags new polls / dropping old ones |
| 5 | G-GAS | KXAAAGASW, KXAAAGASM | 281,106 / 307,760; 190,795 / 507,778 | .01 / .02 | KXAAAGASW 142 | AAA daily (partial obs) + RBOB futures pass-through (wholesale leads retail) | 2·3·2=12 | **Dead-adjacent** N6 (daily, random-walk band gave 0 bets). RBOB lead over 7-30 days is a different reference and horizon; needs PM's view on "materially different" |
| 6 | G-COMMOD | KXWTI, KXWTIW, KXBRENTW, KXGOLDW, KXNATGASW | KXWTI 31,862 / 471,536 | .02 | KXWTI 867 | the settling futures contract itself + CME option IV | 2·2·3=12 | Same shape as crypto with less HFT talk. Risk: same "ladders only quote near-money" outcome as N6/N2 |
| 7 | G-YOUTUBE | KXYTVIEWSW | 27,153 / 500,930 (248 mkts, 16 events) | .03 | 158 | YouTube Charts daily views; settles on max daily views "at any point" in the week, partly observed | 3·2·2=12 | Max-of-week with days already printed = locked YES/NO states |
| 8 | G-CHARTS | KXALBUMEQUIV, KXBBSINGLE, KXTOPSONG, KXTOPALBUM; PM ccrc-bilbrd | 22,719 + 18,944 + 3,848 + 1,186 | .01-.06 | 86 / 3 / 176 / n.a. | HitsDailyDouble first-week projections; Spotify/Kworb daily streams; settlement Luminate/Billboard | 3·2·2=12 | Industry projections published mid-week before Billboard; classic "public better forecast" |
| 9 | G-PORTWATCH | KXHORMUZWEEKLY, KXPANAMAWEEKLY, KXSUEZWEEKLY, KXBABELMANDEBWEEKLY | 21,558 + 17,978 + 8,014 + 14,442 | .01 | 26 / 11 / 13 / n.a. | IMF PortWatch daily transit calls (settlement source itself; lagged) | 3·2·2=12 | Settlement data series is public daily. History is short (Hormuz from 2026-04-28) |
| 10 | G-ERCOT | KXTXERCOTPEAKD, KXMISOINDW, KXERCOTNORTHW, KXPJMDOMW | 5,898 / 3,130; hubs ~2.9-3.2k each | n.a. / .05-.15 | 62 / 1 / n.a. | ERCOT 7-day hourly load forecast; ISO day-ahead prices | 3·1·3=9 | Official grid-operator forecast = closest analogue to NWS. **Thin** (KXTXERCOTPEAKD median YES-bid size 0) |
| 11 | G-FED | KXFEDDECISION, KXFED; FX FFDEC, FF; PM rdc-usfed | 383,508 / 2,131,143; FX FFDEC 205,728 pairs | .02 (KXFED .35) | 29 | CME FedWatch (fed funds futures) | 2·3·1=6 | Deepest reference in finance, but 8 meetings/yr and pros arb it. Q008 already covers FX-vs-Kalshi |
| 12 | G-TSA | KXTSAW, KXTSAD | 2,066 / 5,171; 1,487 | .035 / .07 | KXTSAW 276 | TSA daily checkpoint numbers (weekly avg partly observed) | 3·1·2=6 | N7 was only *deprioritized* ("ladder structure"), never tested; thin |
| 13 | G-MORT | FX MORT1, MORT2 | 12,300 + 2,700 pairs (10-28) | n.a. | n.a. | Mortgage News Daily daily index leads Freddie Mac PMMS (Thursday) | 3·1·2=6 | FX only (Kalshi KXFM30YMTG: 0 settled, not open). $0.01 fee bites at high prices |
| 14 | G-NETFLIX | KXNETFLIXRANKSHOW (+runner-up, global); PM ccrc-ntflx | 5,454 / 8,756 | .01 | 94 | FlixPatrol daily Netflix top-10 (partial obs of the week) | 2·1·2=4 | Weekly chart predictable from dailies; thin |
| 15 | G-EIA | KXEIACRUDEW | 1,001 / 996 | .03 | 9 | API Tuesday report + analyst consensus | 2·1·2=4 | Thin; settlement-source metadata says "Federal Reserve Bank of New York" (see odd things) |

Just missed the list: KXSPACEXCOUNT (61,621 24h; annual, launch manifests); KXRAINWKND (16,697; dead-adjacent R1);
KXBWAYATTENDANCEW (5,012; 1 settled); PM `rainc-rain` daily rain (any METAR rain code, a different settlement from KXRAIN,
OI ~242 extrapolated, too thin).

## 3. Pre-registration stubs for the top 5 (QUESTIONS.md style; criteria are PROPOSALS for /prereg, nothing has run)

Power numbers are from `scripts/edge_math.py power`:
- `--roi 0.02 --cost 0.93` -> 871 bets.
- `--roi 0.03 --cost 0.90` -> 574.
- `--roi 0.05 --cost 0.5` -> 2,467.
- `--roi 0.03 --cost 0.5` -> 6,863.

These counts assume independent bets. Strikes of one event are correlated, so the event count is what binds.

### Q012 [proposed] Kalshi hourly crypto above/below vs Deribit-implied probability (G-CRYPTO)
question:   At T-60 min and T-15 min before each KXBTCD/KXETHD hourly close, is the Kalshi ask on far strikes (|z| >= 1.5 under
            Deribit-IV lognormal) richer than the IV-implied probability by more than fee + spread?
mechanism:  Retail lottery demand on far OTM strikes; the risk-neutral IV probability is public and continuous. The other side is
            HFT makers who already hedge on Deribit/Binance, so any gap may be their fee + risk margin.
baseline:   buy-nothing; and L19 K-FAV (DEAD, adverse selection on KXBTCD favourites; notes/kalshi_fav_maker.md).
change:     a single rule: buy NO (taker) when Kalshi YES bid - IV prob >= 3c, the strike is far OTM, and the spread is <= 2c.
metric:     ROI per $ at the ask + fee, clustered by hour-event, on a date-split DEV.
success:    proposal: ROI LCB > 0 at p < .05 / #trials on DEV, both halves positive.
kill:       proposal: DEV ROI <= 0, or < 300 qualifying events in DEV.
power:      at ~+2% on 93c NO -> 871 bets. KXBTCD has >= 12,000 settled events, so the bet count is there. The binding constraint
            is whether historical Deribit IV at matching timestamps is free (prior_research: "check how far back").
data:       Kalshi candlesticks (hourly bid/ask) for KXBTCD/KXETHD; Deribit public `get_volatility_index_data`/option
            book history; CF BRTI for settlement checks.

### Q013 [proposed] Rotten Tomatoes / Metacritic threshold markets vs the live score (G-RT)
question:   Given the live Tomatometer and review count at decision time (T-72h, T-24h to close), are KXRT strikes that the
            score already sits > k reviews' worth away from priced below their beta-binomial probability?
mechanism:  The settlement variable is mostly observed. Each new review moves the score by ~1/N, so the remaining uncertainty is
            small and computable. Retail prices the headline number, not the review-count arithmetic. The other side includes
            known scrapers (MarketWatch case lost money) and LIP market makers.
baseline:   market price at the same timestamp, and a naive "score stays where it is" rule.
change:     beta-binomial (or empirical drift by review-count bucket) probability that the final score crosses the strike.
metric:     ROI per $ at the ask + fee, clustered by film. Brier of the model vs the mid.
success:    proposal: DEV ROI LCB > 0 at p < .05 / #trials, >= 2 of 3 time-halves positive.
kill:       proposal: model Brier not better than the mid on DEV, or < 60 films with a usable score history.
power:      85 KXRT + 19 KXMC settled events. If ~5 bettable strikes per film at 90c with +3% -> 574 bets needs ~115
            films. **Underpowered today**; it becomes testable as a forward logger. The film rate is not measured here: 18 events
            are open, closing 2026-11-02..12-28.
data:       Kalshi KXRT settled markets + hourly candles. RT review history per film (needs a scrape of review timestamps:
            check ToS before building; never bypass blocks).

### Q014 [proposed] Truth Social post-count markets vs the running count (G-TRUTH)
question:   With d days of the window elapsed and the running count known, are KXTRUTHSOCIAL/KXTRUTHSOCIALD brackets priced
            off a negative-binomial forecast of the remaining posts?
mechanism:  A count process with heavy day-to-day overdispersion. Retail anchors on round numbers. Partial observation means the
            remaining variance is computable from history.
baseline:   market mid at the same time; Poisson model (to show whether overdispersion matters).
change:     negative-binomial model of the remaining count, fit on strictly earlier weeks.
metric:     ROI per $ at the ask + fee clustered by week; Brier vs mid.
success:    proposal: DEV ROI LCB > 0 at p < .05 / #trials.
kill:       proposal: < 40 settled weekly events in DEV after the split, or Brier no better than the mid.
power:      37 weekly + 28 daily settled. With ~5 bettable brackets each, roughly 300 correlated bets -> **underpowered for a
            2-3% edge** (574-871 needed). Run as descriptive DEV plus forward logger. A sealed test needs about another 6 months.
data:       Kalshi settled markets + candles; a public archive of Trump's Truth Social posts with timestamps
            (settlement source is Roll Call/Factba.se; confirm which archive matches its counts before use).

### Q015 [proposed] RCP / VoteHub approval-average markets vs the recomputed average (G-APPROVAL)
question:   Can the RCP average at settlement time be recomputed from the polls on the RCP page plus the known release schedule,
            and is the Kalshi bracket price slower than that recomputation after a poll enters or drops out?
mechanism:  The settlement value is a deterministic average of listed polls. The only uncertainty is which polls are released or
            aged out before the snapshot. Retail trades the trend, not the arithmetic.
baseline:   market mid at the decision time; naive "average unchanged" forecast.
change:     recomputed average + scheduled-poll distribution (Rasmussen/Morning Consult cadence) -> bracket probabilities.
metric:     ROI per $ at the ask + fee, clustered by settlement day; Brier vs the mid.
success:    proposal: DEV ROI LCB > 0 at p < .05 / #trials.
kill:       proposal: recomputation misses the settled RCP value on > 10% of DEV days (reference not good enough), or ROI <= 0.
power:      KXAPRPOTUS 261 + KXTRUMPAPPROVE 58 settled events (the KXAPRPOTUS strike dates start 2022-12-23). With ~3
            bettable brackets each -> ~950 correlated bets, borderline for +2% at 93c (871).
data:       Kalshi settled markets + candles; RCP poll table history (Wayback snapshots), VoteHub data.

### Q016 [proposed] AAA weekly/monthly gas vs RBOB pass-through (G-GAS; dead-adjacent to N6, PM decides if it is "new")
question:   Does an RBOB-futures pass-through model (wholesale leads retail by ~1-2 weeks) price KXAAAGASW/KXAAAGASM rungs better
            than the Kalshi mid at the weekly/monthly horizon, where N6's random-walk band produced 0 bets at the daily horizon?
mechanism:  Retail gas prices follow RBOB with a documented lag ("rockets and feathers"). A random walk ignores this; the market may too.
baseline:   N6 random-walk band (scripts/kalshi_gas.py); market mid.
change:     forecast = AAA(t) + beta * (lagged RBOB change), fit on strictly earlier data, at the same decision time.
metric:     Brier vs the mid on DEV; ROI per $ at the ask + fee, clustered by event.
success:    proposal: Brier improvement CI excludes 0 AND ROI LCB > 0 at p < .05 / #trials.
kill:       proposal: Brier not better than the mid, or < 300 qualifying bets (the N6 0-bet failure mode).
power:      KXAAAGASW 142 settled weekly events (from 2023-10-09) with 21 rungs open now. Bettable rungs per event unknown
            until the quoted-range check; run that check FIRST (it is what killed N6).
data:       Kalshi settled markets + candles; AAA daily national average (rebuild from Kalshi expiration values as N6 did);
            RBOB front-month settles (EIA/CME public).

## 4. Odd things found

1. **KXEIACRUDEW names the wrong settlement source.** The series' `settlement_sources` is "Federal Reserve Bank of New York", but the rules
   text is about EIA commercial crude inventories. Read the contract terms PDF before trading it.
2. **ForecastEx catalog `open_interest` is 0 for MORT1 and IJC**, yet they traded 12,300 and 30,030 pairs on 10-28. Don't
   use the catalog OI.
3. **The ForecastEx `/api/products` endpoint is paged (121 pages).** The first call returns only 8 products. My first fetch got 8 and
   needed a fix. Q007's 968 vs today's 963 is consistent.
4. **The Polymarket US public gateway has no 24h volume field.** OI comes only per market via `/bbo` (`openInterest`,
   lifetime `sharesTraded`), so PM OI here is a 4-market-per-family extrapolation.
5. **Polymarket slug families are noisy.** Per-game slugs make 6,256 "families" on the full prefix. I used the first two
   tokens, which gives 482.
6. **Several Kalshi settled-event lists return empty `strike_date`** (KXRT, KXYTVIEWSW, KXTOPSONG...). KXTSAW's 276 settled
   events all report strike_date 2022-12-25. Use market close times, not `strike_date`, for any time split.
7. **Settled counts are lower bounds.** KXBTCD hit my 60-page cap (12,000). `/events?status=settled` may omit events in
   Kalshi's historical tier.
8. **QUESTIONS.md and FUND_BOOK disagree on Q003.** QUESTIONS.md says Q003 is "DEAD: sealed FAIL 10-29", but FUND_BOOK 3c still lists
   Q003/L41 as "SEALED ASK". FUND_BOOK needs the update. I did not edit either file.
9. **15,501 Kalshi series exist, but only 5,021 have an open event.** Two-thirds of the series catalogue is dormant.
