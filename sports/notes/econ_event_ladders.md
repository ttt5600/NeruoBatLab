# econ_event_ladders -- Kalshi '<asset> after economic events' ladders (series watch 2026-11-15 lead)

Agent: econ_event_ladders (research subagent, 2026-11-15). Dev only, no orders, no git. Sibling `state_gas` covers AAA gas.
Scripts: `scripts/econ_event_census.py` (read-only census) -> `results/econ_event_ladders/census_raw.json`.
Analogy: OT1 (notes/options_implied_tails.md): Deribit smile vs Kalshi daily crypto tails = LEAD (+4.60% [+1.07, +7.62],
387 DEV bets, p .0058 > bar); index arm died on a stale (30-min-old) input; makers adverse-selected.

## Question
Do the new Kalshi KX*EVENT ladders (asset level shortly after a scheduled macro release) misprice far tails against a
free forecaster, enough for an E4/OT1-style taker fade (buy the far-from-tail side at ask + fee)?

## Sealing decision (written BEFORE any outcome was read)
Every one of these 16 series first listed markets 2026-11-05..11-11; the first events closed 2026-11-10. So all settled
events (11-10..11-15) are in the post-sealed / pre-forward gap of every other vault. Decision: they are CENSUS ONLY --
I read listing metadata, quotes, volume; I do NOT read `result`, settlement values or post-release prices, and no trial
is ever scored on them. There is no DEV window for these families (no history exists), so any trial is FORWARD-only.

## Census (2026-11-15, Sunday; census_raw.json, outcome-free fields only)
| family | events (all) | mkts | rungs/event | spacing | settle ref + time | vol to date | live book (nearest event) |
|---|---|---|---|---|---|---|---|
| GOLD | 26 | 414 | 13-17 | $50 | Pyth 1OZGOLD 1-min candle close, release+1h (or +30m auctions) | 734 | 7/17 two-sided, med spread 4c |
| SILVER | 26 | 414 | 13-17 | $1 | Pyth SILVER 1-min close | 2,058 | 8/17 two-sided, 4c |
| COPPER | 26 | 414 | 13-17 | $0.10 | Pyth CU 1-min close | 373 | 9/17 two-sided, 4c |
| BTC | 10 | 170 | 17 | $500 | CF BRTI 60-s avg before 9:30 ET (release+1h) | 10,051 | 0 two-sided (asks .91-.99, no bids) |
| ETH | 9 | 153 | 17 | $25 | CF ETHUSD_RTI 60-s avg | 2,194 | 0 two-sided |
| SOL | 10 | 170 | 17 | $1 | CF SOLUSD_RTI 60-s avg | 3,473 | 0 two-sided |
| EURUSD | 14 | 246 | 13-21 | .0025 | Pyth FX point at release+1h | 20,762 | 11 "two-sided" at 2c/99c (placeholder) |
| GBPUSD | 16 | 288 | 13-21 | .0025 | Pyth FX | 3,422 | 0 |
| USDJPY | 17 | 309 | 13-21 | .25 | Pyth FX | 13,872 | 15 at 2c/99c placeholder, 1 at 9/81 |
| AUDUSD | 16 | 288 | 13-21 | .002 | Pyth FX | 6,506 | 0 |
| USDCAD | 14 | 246 | 13-21 | .0025 | Pyth FX | 25,540 | 18 at 2c/99c placeholder |
| USDMXN | 15 | 267 | 13-21 | .10 | Pyth FX | 1,267 | 0 |
| 2Y/5Y/10Y/30Y yield | 18-20 each | 766-868 each | 51 | 1 bp | Pyth security yield at 8:45/13:15/16:30 ET (release+15m) | 3,229-11,195 | 0 (60c/100c placeholders) |
Fees: all quadratic, multiplier 1 (taker 0.07 p(1-p)). Event cadence: metals ~23/month (CPI, PPI, claims, auctions...),
FX 11-17/month (incl. non-US releases: UK GDP 3am, BOJ 7:50pm, RBA 8:30pm, Canada CPI), crypto ~8/month, rates
~15/month x 4 tenors (near-perfectly correlated with each other -> one cluster per release).
Settled so far: 1-5 events per family (2-17 finalized markets for crypto, 43-55 metals/FX, 217 per rates tenor).

## Free forecaster sources (checked 2026-11-15, outcome-free)
- Crypto: Deribit public API (live + history host), as OT1. BTC/ETH only (SOL options on Deribit too thin to fit a smile).
  Settlement is CF BRTI 60-s average; Deribit index is a near proxy (stated).
- Settlement oracle Pyth: Hermes now returns HTTP 401 for both historical (/v2/updates/price/{ts}) and LATEST prices
  (only /v2/price_feeds metadata is open) -> the settlement source itself is NOT free (a Pyth API key might be; untried).
  benchmarks.pyth.network TradingView shim: 404.
- FX / metals / Treasuries: Yahoo (yfinance) 1h back to 2024-01/06, 5m 60 d, 1m 7 d: EURUSD=X, JPY=X, MXN=X ..., GC=F,
  SI=F, HG=F (futures, not Pyth spot index), ^FVX/^TNX/^TYX (CBOE yield indices; no 2Y index -- ZT=F futures price
  only). Dukascopy minute FX: HTTP 429 on first try. No free options-implied distribution for any of these.
  => any FX/metals/rates forecaster is a MODEL (realised vol of event windows), not a market, and its spot input is a
  Yahoo proxy of the Pyth reference (futures basis for metals; yield index vs Pyth security yield for rates).

## Pre-registration (written 2026-11-15 BEFORE any Kalshi outcome of these families was read; none ever will be for
## events closing before 2026-11-16, see Sealing decision)
No DEV/sealed window exists (first event closed 2026-11-10), so the only honest test is FORWARD. Nothing below can be
dev-tested; the bar arithmetic is written so a forward null is read correctly.
Decision time T = release + 5 min, where release = close - offset (offset from the event subtitle: '1 hour after' /
9:30 claims/PPI/CPI/retail = 60 min; '30 minutes after' = 30; rates 8:45/13:15/16:30 = 15 min; unknown -> skip).
Rationale: post-release the jump is known, the remaining window is plain diffusion, and a far rung that did not reprice
after the jump is the E4-style stale quote this family scored S4 on.
Quote: Kalshi GET /markets for the event at T (live bid/ask, size at best). Spread filter ask - bid <= 0.06.
Inputs strictly older than the quote, and at most 2 min old (OT3 lesson); else no decision.
- EV1 CRYPTO_TAKER (BTC, ETH): p from Deribit smile exactly as OT1 (first expiry after close, sqrt-time standardised
  moneyness, no extrapolation), F = Deribit index at the last option trade before T, tau_K = close - T.
- EV2 MACRO_MODEL_TAKER (gold, silver, copper, 6 FX, 4 rates; a MODEL, not a market): CENTRE = the Kalshi ladder's own
  median at T (strike where the interpolated mid crosses 0.50; needs >= 1 two-sided rung on each side, else skip) -- the
  Yahoo proxies (futures for metals, CBOE yield index for rates) do not match the Pyth reference LEVEL, so EV2 judges the
  tails' WIDTH only. WIDTH: sd of the log move (bp for rates) over close - T = max(same-day realised sd of Yahoo 1m
  returns in [release, T], 60-day same-ET-clock sd of 1m returns, both strictly before T) x sqrt(minutes left) x 1.25
  (no VRP in a realised model -> inflate); Student-t(4) unit-variance shape. Proxies with no 1m data (2Y) -> skip.
Rule (both): tail side p_T <= 0.05; buy the other side F at ask if (1 - p_T) - ask - 0.07 ask (1 - ask) >= 0.03 and
ask in [0.50, 0.99]; 1 contract; no maker arm (maker adverse selection repeated in OT2/OT4/F2).
Cluster = release (all assets of the same release share one cluster: rates tenors, FX pairs, metals move together).
Bar: family kalshi_data_ladders at registration (49 trials -> 51, p < ~1e-3). min_n 150 bets.
Kill (forward): ROI CI upper < 0 after 150 bets, or < 30 bets after 3 months (market too thin to matter).
Power: like OT1, +2% ROI needs ~1,700 independent bets; releases are ~25/month in total, so even at 5 bets/release
a +5% edge takes ~6-8 months to clear 1e-3. A forward run is a liquidity/fill study first.

## Liquidity on the settled events (census only; scripts/econ_event_liquidity.py -> results/econ_event_ladders/
## liquidity_pre_release.json; last hourly candle ending <= close - 60 min, i.e. PRE-release; no result/post-release read)
- 1,349 settled markets (64 events, 11-10..11-13 weekdays). Every market has candles, but only 274 have a spread <= 6c,
  and those are almost all one maker's 97/99 (or 96/99) quote on the deep in-the-money side of the rates ladders
  (41 of 51 rungs of KX10YRRATEEVENT-12NOV261315 alone). Far out-of-the-money rungs: yes bid 0 -> no NO offer at all.
- Rule-eligible quotes (tight, tail-side mid <= .10, F ask in [.50, .96] -- .96 is the max ask that can clear a 3c edge
  after fee): **1 of 1,349** (KX30YRRATEEVENT-12NOV261315-T540, .95/.96). Crypto BTC/ETH/SOL: 0 two-sided pre-release.
  Metals: 2-7% of rungs two-sided, none tight (Sunday 11-15 live books show 4c spreads near the money only).
- Pre-release traded volume, all 16 families, 64 events: 32,990 contracts; median spread of quoted rungs 98-99c.

## Decision
- EV1/EV2 are written but NOT registered in edgelab: with ~0 eligible quotes per event, a forward trial would take
  years to reach min_n 150, and registering would only raise the kalshi_data_ladders bar (49 -> 51) for every other
  trial. Register them only if a re-census shows >= ~5 eligible quotes per release.
- No family is worth a forward logger NOW. Cheapest useful step (proposal for the lead, not done): a read-only quote
  snapshot of these 16 series at release + 5 min (a few GETs per event), re-census in 4-6 weeks. If makers arrive, EV1
  (BTC/ETH vs Deribit, the only free market-implied source) goes first; EV2 is a model and ranks second.
- Bounded claim: the books are empty; that is not evidence of no mispricing. Post-release quotes were not examined.

## Learnings
### Kalshi KX*EVENT macro-event ladders (Nov 2026 launch) have no tradable tails yet
16 series (gold/silver/copper, BTC/ETH/SOL, 6 FX, 2/5/10/30Y yields), first close 2026-11-10. Pre-release quotes on
1,349 settled markets: 1 rule-eligible far-tail quote; median quoted spread 98-99c; 32,990 contracts traded over 64
events. Source: results/econ_event_ladders/liquidity_pre_release.json. Re-census before building any logger.

### Pyth Hermes price history now needs an API key
hermes.pyth.network /v2/updates/price/{ts} and /latest return HTTP 401 (2026-11-15); only /v2/price_feeds metadata is
open; benchmarks.pyth.network shim 404. Kalshi settles FX/metals/rates event ladders on Pyth, so the settlement source
cannot be replayed for free; the Yahoo 1m/5m/1h proxies (futures, CBOE yield indices) do not match its level.

### New Kalshi families: check that a book exists before building a forecaster
The series-watch score (10/10 for metals) rated source, ladder and cadence, but the books were placeholders. One
pre-release candle pull (~1,350 GETs, ~7 min) answered go/no-go before any model work.
