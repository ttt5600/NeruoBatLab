# ai_nowcast -- Kalshi / PM US AI series: free settlement source nowcast vs stale far rungs (agent ai_nowcast, 2026-11-20)

Brief: STRATEGY_v2 #14 (G-TOKENS, info_gap_census_2026-11-07 stub C). NO git, NO orders, NO launchd. Scratch: results/ai_nowcast/.

## 1. Census (live read-only GETs 2026-11-21 ~04:30Z; results/ai_nowcast/census_kalshi.json, series_all.json)
16,768 Kalshi series scanned; AI-related Science & Technology series with events listed in census_kalshi.json. Settlement source
families (settled events = distinct event tickers with finalized markets, from census_kalshi.json):
- **OpenRouter weekly** (openrouter.ai/rankings): KXTOKENUSE weekly total tokens (10 settled events, 2026-09-14..11-16; 57k-148k
  contracts/event, /tmp/tu.json); market-share series KXANTHSHARE/KXGOOGSHARE/KXDEEPSHARE/KXOPENSHARE (10 each), KXTOPUSAGEAI (10),
  KXOPENSOURCESHAREOR weekly (4), KXZAISHARE/KXMETASHARE/KXMISTRALSHARE (1-3), KXTOKENUSEM monthly (2), KXTOP3AID/KXTOKENUSED daily (0-2).
- **Vercel AI Gateway leaderboards**: KXOPENSOURCESHARE (42 settled), 8 lab spend/request share series (29 each), 3 more (11).
- **LMArena** (arena.ai, style control removed): KXLLM1 (12), KXCHINAAI (12), KXTECHRANKLISTAICODE (12), KXTOPMODEL (6), KXARENASCORE (3).
- **Artificial Analysis** arenas: KXVIDEOAI/KXIMAGEAI/KXSPEECHAI/KXOPENINTAI (10 each). Ramp AI index monthly (1 each).
(volume field in census_kalshi.json is 0 because the API field is `volume_fp`; per-event volume for KXTOKENUSE read separately.)

## 2. OpenRouter data gate (KXTOKENUSE) -- PASSED
Settlement source "OpenRouter - AI Model Rankings #top-models"; rule: value for 7 UTC days Mon..Sun, "as observed at 10:00 AM ET"
the next Monday; market closes Sun 23:59 ET (04:59Z Mon); series fee_type quadratic (maker 0), fee_multiplier 1.
Undocumented free JSON endpoints found in the page's JS chunks (results/ai_nowcast/ dl in /tmp/or_dl):
`/api/frontend/v1/rankings/model-rankings-chart[?byok=all]`, `/author-market-share`, `/open-closed`; weekly bars keyed by Monday
(UTC), 53 weeks back to 2025-11-17, current week = running partial.
Reconstruction (fetched 2026-11-21T04:28Z): byok=all chart (= author-market-share sum) week 2026-11-09 = 211.28T vs settled
KXTOKENUSE-26NOV16 expiration_value 211.281198783451; 11-02 209.48 vs 209.5175; 10-26 193.58 vs "194T"; 10-19 188.01 vs "188T";
10-12 163.83 vs "164T"; 10-05 171.84 vs "172T"; 09-28 165.85 vs "166T". The DEFAULT chart (no byok param) is ~17-20% lower
(11-09: 171.21) -> it EXCLUDES BYOK; settlements so far used the BYOK-inclusive total.

**Live definitional hazard (not a nowcast edge):** current week 2026-11-16: byok=all 254.8T (Tencent BYOK spike 101.9T vs 20.4T
last week), default chart 154.0T. KXTOKENUSE-26NOV23-T238 traded .79-.80 on 11-17, then .02-.05 from 2026-11-21T04:03Z, and Kalshi
listed new low strikes T137..T200 at 04:10Z (/tmp/tr238.json, tu.json). The book now prices the BYOK-EXCLUSIVE number. Either
Kalshi/OpenRouter changed what "#top-models" shows, or traders expect exclusion. Which number settles NOV23 is a rules question;
do not trade it. It does mean the settlement definition itself is unstable (R1 risk).

## 3. Pre-registration (written 2026-11-21 ~04:45Z, BEFORE any historical Kalshi price or return is computed)
Question: at a decision time T inside the settlement week, does a pace nowcast from the partially observed week, read only from
data published before T, find Kalshi KXTOKENUSE far rungs priced off it, profitably after ask + fee?
Data: Wayback captures of `model-rankings-chart` (default view; 80 captures 2026-08-01..11-19) give the partial current-week bar AS
PUBLISHED at the capture timestamp (no lookahead: capture ts <= T). Kalshi hourly candlesticks give yes bid/ask at T.
Nowcast: N = partial_bar x 7 / d, d = days of the week covered by the bar at T (data lag to be MEASURED from captures and fixed
before prices: d = (T - week_start - lag) in days, lag estimated as the median from captures). Error model: s(d) = sd of
log(final_week / N) over all captured weeks at the same d bucket (d in [2,3),[3,4),[4,5),[5,7)); weeks are the cluster.
Because settled values used the BYOK-inclusive total, the nowcast is scaled by the BYOK ratio of the previous complete week
(byok=all / default, read from the same capture's prior bar is impossible -- default only -- so use ratio from the CURRENT
byok=all endpoint for the prior week; this is a mild lookahead for the ratio only; flagged).
Rule R-TOK1: decision times T = each capture with d >= 2.0 (Wed 00Z+) and T < market close; for each listed strike K (">K" YES),
p = Phi((ln(N*ratio) - ln K)/s(d)). Buy YES at yes_ask if p - ask - fee >= 0.05; buy NO at (1 - yes_bid) if (1-p) - no_ask - fee
>= 0.05. Far-rung filter: bought side's model prob >= 0.80. Fee = ceil_cent(0.07*10*q*(1-q))/10 per contract (10-lot), q = price paid.
Placeholder filter: skip rungs whose candle has yes_bid == 0 or yes_ask >= 0.99, or spread > 0.10. One trade per (week, strike,
side) -- the first qualifying T.
Metric: ROI per $ staked, week-cluster bootstrap 95% CI. Brier of p vs Kalshi mid on the same rows.
Placebo P0 (base-rate shape): same rule but N := previous week's settled value (no current-week information), same s(d) and
filters. Edge must come from the nowcast: R-TOK1 ROI - P0 ROI > 0.
Split: dev = weeks with Monday <= 2026-10-12, hold = weeks 10-19..11-09 (descriptive). Settled weeks available: 10 (09-07..11-09).
Success (dev): impossible to pass formally -- min_n 20 settled weeks not met. Pre-set outcome: if ROI CI lower bound > 0 AND
beats P0 on dev+hold combined -> forward paper logger (R-TOK1) with forward kill; if ROI <= 0 or <= P0 -> dead (priced).
Kill: ROI point <= 0, or R-TOK1 <= P0, or < 5 qualifying weeks (then forward logger decides alone).

## 4. Log
(see below)

### 4a. Results so far (2026-11-21 ~05:00Z)
- Wayback panel: 71 parseable captures of the default chart 2026-08-01..11-19 (results/ai_nowcast/wayback/, wayback_panel_raw.csv;
  files are gzip). Data is near-real-time: Monday 03:00Z captures already show 1.1-3.0T for the new week (rows 8, 41, 48, 65).
  Up to the 11-19T21:07Z capture the default chart had NO `significantByokSeries` field and its completed weeks equal the settled
  values (e.g. capture 11-17: prev week 211.280 vs settled 211.2812) -> historically default == BYOK-inclusive == settlement, so the
  prereg BYOK ratio is 1 for every historical capture (deviation from s3 text, decided before prices).
- Pace nowcast error log(final/N) by days observed d (nowcast_acc.py -> nowcast_panel.csv, 69 captures, all weeks 07-27..11-09):
  d[2,3) n13 mean -.031 sd .060 | d[3,4) n8 -.009 sd .039 | d[4,5) n11 -.033 sd .036 | d[5,6) n8 -.030 sd .015 | d[6,7) n9 -.012
  sd .009. Pace OVERestimates by ~3% (weekend usage lower); within-week sd 1.5-4% from day 3.
- Week-boundary: Monday 03:00Z captures show the prior week already equal to the final within 0.01% (11-09 cap prev 209.5173 vs
  settled 209.5175; 09-21 cap 128.895 vs final 128.888) while the market closes 04:59Z (EST) / 03:59Z (EDT).
- **R-TOK1 prereg result** (backtest_tok1.py prereg -> tok1_trades_prereg.csv): rule all n 12 trades / 4 weeks, ROI +22.7%
  [-9.7, +45.7] (week bootstrap) -> not distinguishable from 0; dev +42.0% [+7.0,+54.6] on 2 weeks (CI from 2 clusters is not
  meaningful); hold +5.0% [-21.9,+18.5]. Placebo P0 (prior week's value) all n 15 / 6 weeks +1.1% [-42.0,+39.3]. Pre-set kill
  clause "< 5 qualifying weeks" triggered -> no dev verdict; forward logger decides. Exploratory variants (NOT prereg): s=RMSE
  +32.1% [-14.8,+54.6] n9/3wk; bias-corrected +23.2% [-14.8,+35.8] n8/3wk. Losses (3/12) are all YES rungs just above the pace
  nowcast late in the week (T210 11-08, T215 11-12, T218 11-14) = the +3% pace bias; wins include T168/T170 bought at .38/.36 at
  d=5.1 with nowcast 174.9 (final 171.84).
- Coverage limit: Wayback has 0-5 captures per week inside the market window; most of the market's life is unobserved.

### 3b. Pre-registration R-TOK2 (written BEFORE looking at any end-of-week price): post-period, pre-close window
Question: after the 7 UTC days end (Mon 00:00Z) and before close (Mon 03:59Z EDT / 04:59Z EST), the full-week value is already on
the free endpoint. Are rungs still offered at prices that ignore it?
Value V = final week total (byok=all endpoint as fetched 2026-11-21; proxy for what Mon 01:00Z showed -- justified by the
Mon-03:00Z captures above matching final within 0.01%; flagged as a mild proxy lookahead).
Decision T = close of each hourly candle ending in [Mon 01:00Z, close - 0h]; first qualifying hour per rung.
Rule: buy YES at yes_ask if K <= V x 0.995 and ask + fee <= 0.97; buy NO at 1 - yes_bid if K >= V x 1.005 and that + fee <= 0.97.
Fee 10-lot ceil_cent as R-TOK1. Placeholder filter: yes_bid > 0 and yes_ask < 0.99 (a real two-sided quote); report how many rungs
fail it. Metric ROI per $ staked, week-cluster bootstrap. Placebo: same thresholds with V := previous week's final (no info).
Success: n >= 20 trades over >= 5 weeks, ROI CI lower bound > 0 and > placebo -> forward logger at the week boundary.
Kill: < 5 rungs qualifying across all weeks (the book clears itself before 01:00Z) or ROI <= placebo.
Also report descriptively: the yes_bid/yes_ask of every rung at the Mon 00:00Z and 01:00Z candles vs its result.

### 4b. R-TOK2 result + R-TOK1 calibration (2026-11-21 ~05:10Z)
- **R-TOK2 DEAD at its kill rule** (backtest_tok2.py -> tok2_boundary_quotes.csv): 420 hourly candles in [Mon 00:00Z, close] over
  141 settled rungs, only 12 two-sided; 0 rungs qualify. At the Mon 00:00Z/01:00Z candles every rung is already .96-1.00 / 0-.01
  on the correct side (e.g. week 11-02 V 209.48: T207 .99/1.00 won, T210 0/.01 lost). The book reprices to the free endpoint before
  the week even ends. edgelab: tok2_openrouter_week_boundary retired.
- **R-TOK1 Brier (prereg metric)** (brier_tok1.py, 115 capture x rung rows with two-sided quotes, 9 weeks): Kalshi mid .0674,
  prereg pace nowcast .1278, bias-corrected (LOWO mean) .0786. Brier(mid) - Brier(prereg) = -.0604 [-.1396, -.0092] (week
  bootstrap) -> the MARKET is better calibrated; bias-corrected -.0112 [-.0729, +.0307] not distinguishable. The R-TOK1 +22.7%
  on 12 trades is noise on top of a model that loses to the price.
- Live evidence the book is bot-watched: after OpenRouter's default view switched to BYOK-excluded (between the 11-19T21:07Z
  capture, no BYOK field, and 11-21T03:55Z cachedAt with `significantByokSeries`), T238 went .80 -> .05 within minutes
  (first trade 04:03Z) and Kalshi listed T137..T200 at 04:10Z.

## 5. Forward plan (spec; no launchd installed)
Script: scripts/tokenuse_logger.py (one read-only snapshot per run -> results/ai_nowcast/fwd/snapshots.jsonl; first run
2026-11-21T04:45Z OK, 24 open rungs). Logs default AND byok=all weekly totals + every open rung's bid/ask/size.
Cadence if the lead installs it: hourly. Offline scoring = R-TOK1 bias-corrected variant frozen now (mu, s by d bucket from
nowcast_panel.csv, LOWO not needed forward), taker at logged ask, one trade per (week, rung, side), fee 10-lot ceil.
Forward kill (pre-set): after 20 settled weeks, ROI CI lower bound <= 0 or Brier(model) >= Brier(mid) -> dead. At ~3 trades/week
this is ~5 months for n ~60, below the SIZING_RULE 100-trade bar -> recommendation: LOW priority; run only because it is free
and it also records which definition (BYOK in/out) Kalshi settles NOV23 and later weeks on (KXTOKENUSE-26NOV23 expiration_value
vs default 7-day vs byok=all 7-day).

## 6. Other AI series (not tested; data gate notes)
- LMArena (KXLLM1 12 settled, KXCHINAAI 12, KXTOPMODEL 6, KXARENASCORE 3; PM US 20 "aimrc" rank questions at fee .095): arena.ai
  page returns 403 to scripts, but HF dataset lmarena-ai/leaderboard-dataset is free with daily commits (last 2026-11-21T04:02Z)
  and full history: text/full parquet 1,301,969 rows, 259 publish dates 2023-05-08..2026-11-19, columns model/org/rating/CI/
  votes/rank/category/leaderboard_publish_date (text = no style control; text_style_control separate). R1 passes. Mechanism is
  rank persistence vs model-release shocks (event-driven), not a count nowcast; <= 12 settled events per series -> R4 = 0.
  Stub only: P(#1 org unchanged over k days) from 259 publish dates vs book price at T-k; forward-log only.
- OpenRouter share series (KXANTHSHARE/KXGOOGSHARE/KXDEEPSHARE/KXOPENSHARE 10 settled each, KXTOPUSAGEAI 10): same endpoint
  family (author-market-share, byok param); same bot-watched book is likely. Not tested.
- Vercel AI Gateway shares (KXOPENSOURCESHARE 42 settled; 8 lab series 29 each): most settled events of any AI family; source
  vercel.com/ai-gateway/leaderboards (not probed for an intra-period JSON). Next candidate if anyone continues.

## 7. Verdict
KXTOKENUSE: R1 source free and exact (data gate passed); the free source is near-real-time, and the book tracks it (beats our
nowcast on Brier, CI excludes 0; clears completely by the week boundary). No stale far rungs found. Mechanism = "watched number",
not "unwatched tail" -> priced. R-TOK2 dead; R-TOK1 not distinguishable and calibration-dominated -> optional free forward logger
only. Rubric revision: R3 (neglect) for KXTOKENUSE should be 0, not 2 (STRATEGY_v2 row #14 scored 6).

## Learnings
### OpenRouter weekly token totals are reproducible from a free JSON endpoint
`/api/frontend/v1/rankings/model-rankings-chart?byok=all` weekly bar = Kalshi KXTOKENUSE expiration_value (11-09 week 211.28 vs
211.2812; 7 weeks match to rounding). Source: results/ai_nowcast/final_byok_all_20261121.json, kx_tokenuse_markets_20261121.json.
### OpenRouter's default rankings chart switched to BYOK-excluded around 2026-11-20/21
Default chart 11-09 week 171.21 vs byok=all 211.28; captures up to 11-19T21:07Z had no BYOK field and equalled settlement. The
settlement definition of KXTOKENUSE is therefore unstable; T238 repriced .80 -> .05 within minutes. Source: wayback_panel_raw.csv,
/tmp/tr238.json trades.
### Wayback captures of an undocumented JSON endpoint give a lookahead-free partial-period panel
71 captures of model-rankings-chart (gzip bodies via /web/<ts>id_/) gave the partial week as published; pace nowcast sd 1.5-4%
from day 3 with a +3% upward bias (weekend dip). Source: nowcast_panel.csv.
### A near-real-time free source makes a book efficient, not stale
KXTOKENUSE rungs were .96-1.00 / 0-.01 at Mon 00:00Z (tok2_boundary_quotes.csv), and Kalshi mid beat the pace nowcast on Brier
.0674 vs .1278 (diff -.060 [-.140,-.009]). The nowcast edge needs a source that is free but SLOW or hard to read; an hourly JSON
the crowd's bots poll leaves nothing.
### LMArena's leaderboard is free on Hugging Face despite the arena.ai 403
lmarena-ai/leaderboard-dataset text/full: 1.3M rows, 259 publish dates since 2023-05-08, daily commits. Source: HF API, /tmp/lma.
