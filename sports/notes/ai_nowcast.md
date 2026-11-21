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
