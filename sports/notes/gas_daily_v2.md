# gas_daily_v2 -- KXAAAGASD daily national gas: can a sharper nowcast beat the far rungs? (agent gas_daily_v2, 2026-11-21)

Brief: lead, after winners_transfer P2 (KXAAAGASD tail pool $488,558/30d paid by losing <=15c buyers, largest non-weather
pool; results/winners_transfer/tailpool_result.json). Prior tests: N6 random-walk band 0 bets (scripts/kalshi_gas.py,
lab/ledger.jsonl l.83); Q016 T1 weekly KXAAAGASW RBOB pass-through Brier +0.035 [-0.017,+0.084] n.d. vs mid
(notes/q016_commodities.md); state_gas GS1 parked until 26DEC14 (notes/state_gas.md). Lesson carried in: count_nowcast --
the book reprices off the FASTEST public page, not the named settlement source.
No git, no orders, no launchd. Scratch: results/gas_daily_v2/.

## Plan
1. Settlement DGP: contract text, timing (open/close/AAA post/settle), precision; audit >= 50 Kalshi expiration_values vs
   AAA's own published numbers (homepage current/yesterday/week/month-ago rows; Wayback captures).
2. Leading signals: AAA print time vs book; RBOB (RB=F, EIA NYH/GC spot); EIA weekly; GasBuddy live state/national page
   (robots + terms); AAA state averages (same print time as national -> not a lead). Model daily change c_D.
3. Book: implied median error by hour through D-1 vs the nowcast (dev only).
4. Prereg rule before any return is computed; edgelab family kalshi_data_ladders; dev only; date-clustered CI; top-5.

## Log
### 1. Settlement + quote DGP (read 2026-11-21; results/gas_daily_v2/fetch_markets.py, build_aaa.py, markets.parquet)
- Series KXAAAGASD: fee_type **quadratic**, fee_multiplier 1 (GET /series/KXAAAGASD). Rungs "strictly greater than X",
  X every $0.005 (239/240 events), median 17 rungs (10-37). Rule text: "If average regular gas prices for United States
  are strictly greater than $X on <D> according to AAA". Terms AAAGAS.pdf (assets.kalshi.com/contract_terms/AAAGAS.pdf):
  underlying = AAA average regular price for <date>; revisions after expiration ignored; last trading 11:59 PM ET the
  day BEFORE <date>; if no data, last available day's value. (The 2023 certification had last trading 9:55 AM ET ON
  <date>, i.e. after AAA's overnight print; the 2023 AAAGASD-*-US events were single-rung with 0-330 contracts.)
- Timing (240 KX events 2026-03-26..2026-11-20): open 08:00 ET on D-1 (154 events; rest 08:50-11:15 ET); close 23:59 ET
  D-1 (239); AAA posts D's value overnight (~03:15-03:40 ET per state_gas census); Kalshi settles 05:00-13:51 ET on D
  (median hour 8). So there is NO post-print trading window: the book forecasts a print ~3.5 h after it closes.
- Kalshi expiration_value = AAA print at AAA's display precision: 3 decimals until 2026-08-04, 4 decimals from 2026-08-05
  (122 / 101 / 16 / 1 events with 3 / 4 / 2 / 1 significant decimals). 0/4,288 settled rungs disagree with
  result == (value > strike). 28 ties value == strike all settled NO (strictly greater) -- ties are only likely in the
  3-decimal era.
- AAA(D) daily change c (cents, 239 changes): mean -0.12, sd 2.15, median |c| 1.20, 90% |c| 3.2, max +9.2;
  lag-1 autocorrelation 0.774, lag-2 0.507 (build_aaa.py). The series is a smooth trend, not a random walk.
- AAA homepage (curl 2026-11-21 05:30Z, robots.txt allows `*` with Crawl-delay 10, no terms page restricting use):
  current 11/20 $3.6898, yesterday $3.7212, week-ago $3.8724 = Kalshi 26NOV20/26NOV19/26NOV13 exactly.
- GasBuddy /usa shows LIVE state averages with today's change (fetched once by hand 2026-11-21 05:3xZ), but its Terms of
  Use s.22 forbid "scrape or download ... fuel price or other data ... (including automated scripts, spiders, robots,
  crawlers...)" and "create a database by systematically downloading". So the fastest public page is off-limits to an
  automated nowcast and has no free history.
- Wayback (web.archive.org) is unreachable via local DNS (resolves to 31.169.123.46, TLS fails); pinned 207.241.237.3
  answers 429. Background retry job: results/gas_daily_v2/wayback_fetch.py. Common Crawl index: 504.

### 2. Nowcast model exploration (DEV AAA values only, no prices; results/gas_daily_v2/explore_model.py)
Expanding-window OOS on 114 dev days (2026-04-10..08-31 after a 30-day burn-in), forecast of c_D at 08:00 ET D-1
(AAA through D-1, RB=F settles through D-2):
| model | RMSE c | MAE c | |err| q99 |
|---|---|---|---|
| random walk (c=0) | 1.783 | 1.386 | 4.88 |
| persistence c_D = c_{D-1} | 1.254 | 0.927 | 4.35 |
| AR(2) | 1.143 | 0.808 | 4.18 |
| AR(3) | 1.085 | 0.765 | 4.10 |
| AR(2) + RB 5d, 5-10d | 1.116 | 0.883 | 2.83 |
RBOB adds little at a 1-day horizon (pass-through is a 1-2 week effect and already in the AR terms). A t4 predictive
(scale = sd of the last 60 in-sample residuals) is calibrated in the tails on dev OOS: PIT < .03 in 0.9%, > .97 in 2.6%
of 114 days. Best free nowcast = +-1.1 c RMSE -> a rung 3 c from the forecast is a ~2.5-sd tail.

## PRE-REGISTRATION G2 (written 2026-11-21 ~05:55Z BEFORE any KXAAAGASD candle, quote or trade was fetched or read)
Seen before writing: settled expiration values (outcomes) and market metadata only; model fits on dev AAA values.
- Split by event day D (outcome-free, fixed now): DEV 2026-03-26..2026-08-31 (159 events); SEALED 2026-09-01..2026-11-20
  (81 events, candles NOT fetched; opened only on the user's yes); FORWARD 2026-11-21+.
- Decision time t = 10:00 America/New_York on D-1. Quote = last hourly candle with end_period_ts in (t-6h, t];
  yes_bid / yes_ask = candle close (`close` on /historical = dollars, `close_dollars` live). No candle -> no bet.
- Inputs known at t: AAA values for days <= D-1 (AAA posts overnight; Kalshi's own settle time is irrelevant to when
  AAA's page showed it), RB=F daily settles dated <= D-2.
- Model M: OLS c_D ~ 1 + c_{D-1} + c_{D-2} + drb5 + drb5_10 (cents; drb5 = RB(D-2) - RB 5 sessions earlier, drb5_10 =
  the 5 sessions before that), expanding window over strictly-past days, >= 30 rows. AAA(D) = AAA(D-1) + yhat/100 +
  s T4/sqrt(2), s = sd of the last 60 in-sample residuals. P(YES at X) = P(AAA(D) >= X + h), h = 0.0005 for D <=
  2026-08-04 (3-decimal display), 0.00005 after (4-decimal).
- Rule G2 (taker, 1 contract per bet, every qualifying rung, cluster = D): fee(a) = ceil_cent(0.07 a (1-a)).
  NO arm: p <= 0.03 and yes_bid >= 0.01 -> a = 1 - yes_bid; bet if (1 - p) - a - fee(a) >= 0.02.
  YES arm: p >= 0.97 and yes_ask <= 0.99 -> a = yes_ask; bet if p - a - fee(a) >= 0.02.
- Placebos (same quote, thresholds, fee): P-RW random walk AAA(D-1) + s_rw T4/sqrt2, s_rw = sd of last 60 changes;
  P-MOM naive momentum AAA(D-1) + c_{D-1} + s_m T4/sqrt2, s_m = RMS of last 60 persistence errors; P-BASE model-free
  favourite: buy the favourite side of every rung whose favourite ask is in [0.90, 0.99] (NO if yes_bid <= 0.10, YES
  if yes_ask >= 0.90).
- PASS (dev, all): n >= 100; net ROI > 0 with date-cluster bootstrap 95% CI lower bound > 0 and one-sided p < the
  kalshi_data_ladders family bar at registration; ROI > each placebo's ROI; ROI > 0 after dropping the top-profit date
  (edgelab BAR_VERSION 2). KILL: any fails. A pass earns only a sealed ASK, planned at dev ROI x 0.66 (STRATEGY_v2 amd 3).
- Descriptive (not trials, declared now): D1 book timing = implied-median error (rung mids interpolated at 0.5) by hour
  of D-1 vs realized, against the model's MAE; D2 Benter stacking: logit(outcome) ~ logit(mid) + logit(p_M) on rungs
  with mid in [.03,.97] at t, date-cluster CI on the p_M coefficient; D3 availability: share of p <= .03 rungs with any
  YES bid at t (N6 failure mode); D4 top-5 dates behind pooled PnL. Maker variant: NOT tested (no fill data).
- Traps: epoch via // pd.Timedelta(seconds=1); candle `close` vs `close_dollars`; candles with no bid (yes_bid 0/None)
  treated as no bid; too-good (ROI > 10% on > 200 bets) -> inspect 20 rows.
- Registered BEFORE any candle fetch: edgelab **g2_gas_daily_far-54b11381** (family kalshi_data_ladders, 73 trials
  all-time, bar p < 6.85e-04); spec edgelab/specs/g2_gas_daily_far.yaml, rule edgelab/rules/gas_daily.py:g2_far,
  dataset vault kalshi_gas_daily (loader added to edgelab/loaders.py, splits in edgelab/vault.yaml).
- Clarification added before scoring (still no candle read): placebos are scored on the same events as G2 (events where
  model M is defined, i.e. >= 30 strictly-past training rows), so the comparison is like-for-like.

### Other candidates (STRATEGY_v2 amendment 5: settlement/quote mechanics, reformulation, negate a default)
- O2 settlement tie: "strictly greater" + 3-decimal display made value == strike possible (28 ties, all NO). Since
  2026-08-05 AAA shows 4 decimals -> tie probability ~10x smaller. Not a forward edge; noted for any 3-dec backtest.
- O2 post-print window: the 2023 terms let trading run to 9:55 AM ET on D (after AAA's ~03:30 print) -- a lock-in window.
  Current terms close 23:59 ET D-1, so none exists now (verified on all 239 KX events' close_time).
- O6 reformulation, cross-series consistency: national = weighted mean of the 50 state averages, and the 50
  KXAAAGASCD* state change books (logged 2/day by state_gas_books since 11-15) imply a national change; compare with the
  national book at the same minute. Forward-only (6 dates of state books); proposal below.
- O8 negate "taker far-rung NO": P-BASE (model-free favourite) is exactly that negation's control -- if it wins as much as
  G2, the money is a favourite-longshot shape, not nowcast information.

## RESULT G2 (dev scored once, 2026-11-21; sealed candles never fetched)
Frame: results/gas_daily_v2/build_frame.py -> data/gas_daily_v2/gd_frame.parquet (2,813 dev rungs, 159 events; model M
defined on 2,242 rungs / 126 events 2026-04-28..08-31; quote at T on 95% of rungs, candle age 0 s median).
edgelab evaluate g2_gas_daily_far-54b11381 --split dev -> **fail, state dead**: n **65** (< min 100) over 29 dates, 5 losses,
net ROI **+5.43% [-2.88%, +11.45%]**, p .081 vs bar 6.85e-4; halves +4.7% / +6.1%; top week 2026-W30 = 70% of profit
(ROI ex top week +2.4%, ex top date +3.3%); CLV +2.1c. Reasons: a_p_below_bar, d_min_n.
Placebos (results/gas_daily_v2/analyze.py -> analyze.log; same 126 events, same quote/fee/thresholds):
| rule | n | dates | losses | mean cost | ROI [95% date CI] |
|---|---|---|---|---|---|
| G2 model M | 65 | 29 | 5 | .876 | +5.4% [-2.9, +11.5] |
| P-RW random walk | 55 | 14 | 17 | .789 | -12.4% [-49.3, +10.0] |
| P-MOM naive momentum | 61 | 17 | 8 | .854 | +1.7% [-14.6, +13.6] |
| P-BASE model-free favourite .90-.99 | 820 | 125 | 66 | .979 | **-6.1% [-9.6, -3.1]** |
G2 beats every placebo on the point estimate, but no CI excludes 0 except P-BASE's (negative). Verdict: **dead
(too few buyable far rungs + p far above bar)**.
Top-5 dates (D4, g2_dev_bets.parquet): 07-22 +1.51 u (10 bets), 06-03 +0.55, 07-23 +0.45, 06-16 +0.39, 07-14 +0.31;
worst 07-09 -1.42 (2 losses), 07-28 -0.50, 07-15 -0.14. Total +3.09 u on 65 u staked: one date = 49% of profit.
Rows inspected (all 5 losses + all 30 bets at price <= .90): 30 of 65 bets were rungs the book priced at 10-66c that the
model put at <= 3% -- big disagreements during the July price spike (3.80 -> 4.12, Jul 9-28). The model was wrong on
07-09 (AAA 3.846 vs model mean 3.797, +4.9c surprise) and 07-15 (+2.8c), right on 07-22 (NO at .39 on 4.065, AAA 4.060).
ROI on price <= .90 bets +8.2% (30), on price > .90 bets +1.8% (35). This is rule PA's zone: a coin flip at turning
points, not a stale-tail harvest.
D3 availability (N6 failure mode again): 579 rungs had p <= .03; 540 had a quote, but only **177 had any YES bid**
(median YES bid 0.00) -> the far NO costs $1.00 on 70% of them. 599 rungs had p >= .97, 185 with YES ask <= .99.
Calibration on those rungs: realized YES 2.59% vs mean model p 0.92% on p <= .03 rungs (model low tail too thin in a
spike month); NO 0.67% vs 0.91% on p >= .97 rungs.
D2 (two-sided rungs, mid in [.03,.97] at T; 1,056 rungs / 125 dates): Brier mid .1244 vs model .1562, diff +0.0318
[+0.0111, +0.0531] -> the BOOK is better calibrated than the nowcast at 10:00 ET. Benter stacking: coef logit(mid) 1.007,
coef logit(p_M) **0.271 [0.136, 0.419]** (date bootstrap, 1,000 resamples) -> the AR nowcast carries some information the
10:00 book has not absorbed, near the money (not in the far tails, where G2 trades).
D2 robustness (results/gas_daily_v2/d2_robust.py -> d2_robust.log; refit blend on one part of dev, score the other):
fit first half of dev dates -> test second half: blend Brier .1229 vs mid .1188 (WORSE); fit H2 -> test H1: .1182 vs
.1284; leave-one-month-out pooled .1155 vs .1244 but Apr/Aug worse, gain mostly May (.1164 vs .1457); coefficient on
p_M ranges +0.06..+0.40. Forward in time (the only honest direction) the blend did not beat the mid -> no G3 trial.

### 3. Settlement audit (>= 50 required; results/gas_daily_v2/wayback_fetch.py, audit.py -> audit.log, audit_rows.parquet)
260 Wayback raw captures of gasprices.aaa.com (1/day, 2026-03-01..11-20; fetched via pinned IP 207.241.237.3 after the
429 cleared, <= 1 req / 6 s) + today's live page. Each capture gives Current / Yesterday / Week-ago / Month-ago rows.
- 903 capture-rows hit a Kalshi KX date; **all 240 KX settlement dates audited**. The "Current" row on the capture day
  equals Kalshi's expiration_value exactly on **235/235 rows (198 dates)**.
- 35 rows (23 dates) differ by <= $0.0005: every one is a date before 2026-08-05 that AAA later re-displays at 4
  decimals (e.g. 08-04: Kalshi 4.089, AAA's later "Yesterday" 4.0892); all 35 equal the 4-decimal value rounded to 3.
  So Kalshi settles on the first-published display value (3 dp before 08-05, 4 dp after); later redisplays never moved
  a settlement.
- AAA print time: latest capture still showing the previous day 03:16 ET, earliest showing the new day 03:30 ET (29
  vs 232 captures) -> AAA posts ~03:16-03:30 ET, 3.3-3.5 h after the Kalshi close.

### 4. Book timing vs the fastest page (D1; results/gas_daily_v2/d1_timing.py -> d1_timing.log, d1_timing.parquet)
Implied median of AAA(D) from rung mids, abs error vs the print (cents), 125 dev events with model M:
| hour ET on D-1 | 09 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|
| book MAE | .725 | .716 | .579 | .485 | .442 | .390 | .394 | .345 | .342 |
| model M MAE | .956 | .986 | .974 | .969 | .969 | .969 | .940 | .969 | .969 |
Book 10h minus 23h error (paired): +0.358c [+0.238, +0.494]; at 10h model minus book: +0.271c [+0.138, +0.393] (n 125).
Reading: the book is already sharper than the best free nowcast when it is 2 h old, and it halves its error again
during D-1 with no new AAA or futures settle in between -- it is absorbing a live intraday source (GasBuddy's crowd
average / OPIS rack-to-retail feeds). count_nowcast's lesson holds: the book reprices off the fastest page, and that
page (GasBuddy) is ToS-blocked for scraping.

### 5. Where the $488k tail pool comes from (POST HOC, descriptive, not a trial; inline script 2026-11-21)
Same rungs as P-BASE (favourite ask .90-.99 at 10:00 ET, 126 dev events): the favourite wins 91.95% at mean ask .969
(-5.1% at ask before fee; -6.1% after, P-BASE); the cheap side wins 4.31% at mean ask .112 (cost .124 with fee):
ROI **-65.2% [-78.8, -48.8]** (n 673, 121 dates; ex top-5 dates -77.3%). The far rungs carry ~8c quotes around a roughly
fair mid; longshot TAKERS fund the pool and the quoting maker collects it. A taker on either side pays the spread; the
pool is a maker's pool, and earning it needs a centre sharper than the book's (which, per D1, means GasBuddy-grade live
data) plus a filled-vs-unfilled check against adverse selection (E5/RD2).

## VERDICT (2026-11-21): DEAD for taker far rungs; no sealed ask; no forward logger proposed
- G2 dev: n 65 < 100, ROI +5.4% [-2.9, +11.5], p .081 vs bar 6.85e-4 (edgelab g2_gas_daily_far-54b11381 -> dead).
- The sharper free nowcast (AR2 + RB=F, RMSE 1.1c vs 1.8c random walk) is still LESS sharp than the book: Brier .1562
  vs .1244 at 10:00 ET, implied-median MAE .97c vs .72c at 10h and .34c at 23h.
- Far rungs are not stale: 70% of model-certain rungs have no YES bid to sell into (median bid 0.00); where they do,
  the disagreement is a turning-point coin flip (July spike).
- Sealed (2026-09-01..11-20, 81 events): candles never fetched; nothing to open.
- Not proposed: a GasBuddy logger (ToS s.22 forbids automated download); a maker variant (no fill data, and STRATEGY_v2
  "stop doing" maker without filled-vs-unfilled). Optional, low priority: if state_gas GS1 is scored at its 26DEC14
  freeze, add the national book to that logger so a cross-series (50 states -> national) consistency check exists.

## Learnings
### KXAAAGASD settles exactly on AAA's first-published display value
All 240 KX settlement dates (2026-03-26..11-20) audited against 261 Wayback/live AAA captures: Current row = Kalshi on
235/235; the only differences are 3-dp-era values later redisplayed at 4 dp (rounding, <= $0.0005). AAA posts 03:16-03:30
ET, 3.5 h after the 23:59 ET close. Source: results/gas_daily_v2/audit.log. No post-print window exists (the 2023 terms
had one, to 9:55 AM ET on D).
### AAA daily gas changes are strongly autocorrelated, so random-walk bands are the wrong null
Daily change lag-1 autocorrelation .774, sd 2.15c; random-walk RMSE 1.78c vs AR(2)+RBOB 1.12c (dev OOS n 114).
Source: results/gas_daily_v2/build_aaa.py, explore_model.py. N6's 3-sd random-walk band was ~6c wide: 0 bets.
### The KXAAAGASD book beats any free nowcast and sharpens through the trading day
Implied-median MAE .72c at 10:00 ET -> .34c at 23:00 ET (paired diff +0.36c [+0.24,+0.49]) while the AR2+RBOB nowcast
stays at ~.97c; Brier mid .1244 vs model .1562 [+0.011,+0.053]. Source: results/gas_daily_v2/d1_timing.log, analyze.log.
The book reads a live crowd/rack source (GasBuddy) whose ToS forbid scraping.
### A Benter stacking CI above zero can vanish forward in time
D2 stacking coefficient on the nowcast 0.271 [0.136, 0.419] over 125 dates, but fit on the first half of dates the
blend lost to the mid on the second half (.1229 vs .1188). Source: results/gas_daily_v2/d2_robust.log. Check the
forward-in-time split before calling a stacking coefficient a lead.
### A large tail pool can be a maker's pool that takers cannot reach
KXAAAGASD's $488k/30d paid by losing <= 15c buyers comes from longshot takers paying ~11c for ~4% events (-65%
[-79,-49], n 673) while favourite takers also lose (-6.1% [-9.6,-3.1] after fees, n 820). Source: analyze.log, section 5.
Pool size bounds a maker's take, not a taker's.
