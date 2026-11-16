# quake_monthly -- Kalshi KXBIGGESTQUAKEMONTH (biggest quake in a calendar month) vs USGS climatology

Agent topic `quake_monthly`, started 2026-11-16 (research subagent of the quant-trader lead). Scratch:
`sports/results/quake_monthly/` (scripts in `scripts/`). Related dead work: `notes/e1x_quake.md` (daily KXBIGGESTQUAKE,
dead: tail YES ask premium only lifted after a quake happens -> maker adverse selection; bid side at climatology).

## Census (2026-11-16, `scripts/census.py` -> `census_raw.json`; `scripts/book.py` -> `book_snapshot.json`)
- Series created 2026-11-10 (market created 01:59Z, open 04:59Z). ONE event ever: KXBIGGESTQUAKEMONTH-30NOV26.
  0 closed, 0 settled markets -> **no settled history; Kalshi-price dev is impossible**.
- 9 rungs, strike_type greater_or_equal, M 6.6..8.2 step 0.2. fee_type quadratic (taker 0.07 p(1-p); maker 0 assumed
  as on other quadratic series).
- Rules: max USGS-reported magnitude worldwide 2026-11-01 00:00 .. 11-30 23:59:59 UTC; determining value = USGS
  magnitude displayed 2026-12-01 10:00 ET, Mw preferred else primary type. Can close early once USGS finalized.
  (So late-month preliminary magnitudes that get revised after 12-01 15:00Z settle at the preliminary value.)
- Volume to date (contracts): 8.2 10, 8.0 428, 7.8 79, 7.6 4, 7.4 3, 7.2 61, 7.0 54, 6.8 53, 6.6 0. ~30 trades total,
  sizes 1-140. Takers mostly YES on the tail rungs (8.0: all 7 trades YES at .12-.14).
- Who quotes: one ladder-shaped MM -- 100-lot pairs 1c apart on both sides of every rung (e.g. 8.0 NO bids .87x100/.88x100,
  YES bid .02x200) placed 11-10, plus small 5-lot top-of-book quotes. Market-object yes_bid/ask fields show
  updated_time 11-10 04:59 (stale); use /orderbook. Book at pull (yes bid / yes ask = 1 - best NO bid):
  6.6 .70/.78, 6.8 ?/.62, 7.0 .40/.47, 7.2 ?/.35, 7.4 .17?/.26, 7.6 .10/.19, 7.8 .06/.15, 8.0 .03/.10, 8.2 .02/.09.
  Spread 7-9c everywhere -- same shape as the daily series (wide, ask-side lottery premium).
- USGS Nov max so far (pull 2026-11-16, `scripts/usgs.py` -> `usgs_m6.parquet`): M6.4 Solomon Is. 11-09; no M>=6.6.

## Disclosed before the prereg below
I have seen the live 11-16 book above and the November-to-date USGS max (6.4). I have NOT computed any model
probability yet. Nothing below can be judged on this book except as forward trial event #1.

## PRE-REGISTERED (2026-11-16, before any model number was computed)
Model (frozen, `scripts/model.py`): p(K,t) = 1 if the observed USGS max since month start >= K, else the empirical
share of all T-day windows (T = month end - t) starting each UTC day 2000-01-01 .. t-30d whose catalog max >= K
(USGS ComCat M>=6, current values). Empirical windows absorb aftershock clustering (a window is hit once).
ETAS-lite guard: skip the month-decision if an M >= 7.5 occurred in the 30 d before t.
Decision times: daily 15:00 UTC from listing to month end - 1 d. Quotes from /orderbook at the decision time
(YES ask = 1 - best NO bid, NO ask = 1 - best YES bid; need size >= 1).
- Q1 TAKER (near-certain side): for each rung and side S in {YES, NO} with p_S >= 0.85, buy 1 contract iff
  p_S - ask_S - 0.07 ask_S(1-ask_S) >= 0.03 (E4 margin). Hold to settlement.
- Q2 MAKER: on the near-certain side (p_S >= 0.85), post a bid at q = best bid_S + 0.01 (if < ask_S) iff p_S - q >= 0.03;
  filled iff a trade prints at taker_side opposite and price crossing q before the next decision; report fill rate
  and loss rate filled vs unfilled (adverse-selection check, the e1x_quake killer).
Unit of inference = month (all rungs and days of a month are one cluster). PASS (to propose real money) requires
>= 12 settled months, ROI per $ > 0 with month-clustered bootstrap one-sided p < .05/2 (k=2), exact binomial loss
rate CI upper bound below the mean breakeven; plus filled-vs-unfilled loss rate gap < 10 pts for Q2.
KILL: first 6 months ROI <= 0 on both, or any month with a filled near-certain loss in Q2 while unfilled were all wins
in >= 2 of the first 6 months (adverse selection), or the series stops listing.
Power note: one event/month; ~12 clusters/yr. Underpowered by construction; this is a slow forward logger, not a dev test.

## Results

### R1 -- first forward decision 2026-11-16 15:00 UTC (model `scripts/model.py 2026-11-16T15:00:00Z`; book `book_snapshot.json` pulled ~14:35Z)
T = 14.4 d left, Nov max so far 6.4, guard not triggered, 9773 windows.
| K | p_model YES | book YES bid / ask | p_NO | NO ask | QM1 edge NO |
|---|---|---|---|---|---|
| 6.6 | .703 | .70 / .78 | .297 | .30 | -.018 |
| 7.0 | .413 | .40 / .47 | .587 | .60 | -.030 |
| 7.4 | .212 | .17 / .26 | .788 | .83 | -.051 |
| 7.6 | .146 | .10 / .19 | .854 | .90 | -.052 |
| 7.8 | .080 | .06 / .15 | .920 | .94 | -.024 |
| 8.0 | .039 | .03 / .10 | .961 | .97 | -.011 |
| 8.2 | .022 | .02 / .09 | .978 | .98 | -.004 |
- **QM1 fires 0.** Same structure as the daily series: the YES **bid** sits at or just below climatology on every rung
  (within 0-5c); the lottery premium is on the YES ask (8.0 ask .10 vs .039). The near-certain side (NO) is never cheap.
- QM2 would post today: NO bids q = .92 (8.2), .91 (8.0), .86 (7.8), .82 (7.6) (p_NO - q = .058/.051/.060/.034).
- Model stability (same T, 7-yr blocks, `model.py` output): 7.0 rung .474/.467/.425/.404/.388 (1990-97 .. 2018-25);
  8.0 .029/.023/.058/.041/.030. Drifts down ~8 pts on 7.0 over 35 y; the 2000 window start is a mild YES-side choice.
- One difference from daily e1x: 8.0 YES was lifted 7 times at .12-.14 on quiet days 11-10..11-12 (428 ct, no M>=7
  anywhere) -- lottery buyers DO take on quiet days here, so maker fills need not be pure adverse selection. One event.
- What-if (`scripts/decay.py`, descriptive, NOT a result): if no M>=6.6 arrives AND the 11-16 book stays frozen, QM1
  starts firing 11-24 (7.4 NO +.06) and on 4-5 rungs by 11-27. So the forward question is: **does the MM re-price for
  time decay?** Market-object fields say updated 11-10, but trades printed through 11-16 -- unknown.

## Verdict
LEAD, forward-only, low priority. No settled history (series born 2026-11-10, 1 event); 1 cluster/month -> 12 months
minimum for any test; family bar p < 8.06e-04 (62 trials) is out of reach for ~12 clusters unless the late-month
stale-book effect is large. Registered qm1_quake_month_taker-c9d0f295, qm2_quake_month_maker-071f02b8 (edgelab; dataset
kalshi_quake_monthly added to edgelab/vault.yaml, forward only, rule unwired).

## Logger proposal (NOT installed -- lead/user decision)
Read-only, daily 15:00 UTC (+ 23:00 UTC on the month's last 5 days, descriptive):
1. kget /events?series_ticker=KXBIGGESTQUAKEMONTH (open) -> each market /orderbook (depth 10) + /markets/trades since last run.
2. USGS fdsnws M>=6 since month start (max, magType, status) + last 30 d (guard).
3. model.p_model(t) -> QM1 picks (first fire per ticker-side), QM2 posted bids; append JSONL
   ~/.claude/quant-sports/data/quake_monthly/{snap,picks}.jsonl. QM2 fill = later trade crossing q (taker side opposite).
4. Settle from result/expiration_value after the 1st; log USGS value at 1st 15:00Z vs later revisions.
~12 GETs/day via kx_throttle; can ride inside an existing daily logger instead of a new launchd job.

## Learnings
### KXBIGGESTQUAKEMONTH bid sits at climatology, premium on the ask (same as daily)
11-16 15:00Z, 14.4 d left: model p vs YES bid within 0-5c on all 9 rungs; YES ask 4-7c above (8.0: p .039, bid .03,
ask .10). Taker on the near-certain NO side fires 0. Source: notes/quake_monthly.md R1. Matters: the e1x_quake mechanism
(lottery premium only on the ask) holds across horizons on Kalshi quake ladders.
### Empirical T-day window frequency is the simple aftershock-proof quake model
Counting windows with >= 1 event (not Poisson on raw counts) makes clustering a non-issue; stable across 7-yr blocks
(M7.0 in 14.4 d: .39-.47). Source: results/quake_monthly/scripts/model.py.
### Monthly Kalshi ladders with one event per month cannot be dev-tested at listing
KXBIGGESTQUAKEMONTH born 2026-11-10 with 0 settled events; at 1 cluster/month the kalshi_data_ladders bar (p < 8e-4)
is out of reach for > 1 year. Check first whether the MM re-prices for time decay before spending more on it.
