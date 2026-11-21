# vercel_share -- Kalshi Vercel AI Gateway share series + OpenRouter share series (agent vercel_share, 2026-11-21)

Brief (lead): KXOPENSOURCESHARE (42 settled) + 8 Vercel AI Gateway lab-share series (~29 settled each) + OpenRouter share
series (KXANTHSHARE/GOOG/DEEP/OPENSHARE, KXTOPUSAGEAI; 10 settled each). Per series: rules, settlement source + timing, is the
source free and observable intra-period (find the data endpoint behind vercel.com/ai-gateway/leaderboards; robots/terms),
loser-paid pool per 30d (tailpool.py), spreads, fee type. Rebuild >= 20 settled values exactly (Wayback for point-in-time).
Then: how early does the share lock in; how stale are far rungs vs the intra-period share at fixed decision times; time the
book vs the fastest page. Pre-register a rule BEFORE returns, register in edgelab, dev with date-clustered CIs, top-5.
NO git, NO orders, NO launchd. Scratch: results/vercel_share/.

Do-not-redo inherited: ai_nowcast (KXTOKENUSE TOK1/TOK2 dead: free near-real-time JSON -> bot-watched book; market Brier beat
pace nowcast), count_nowcast (book prices off the FASTEST page, not the named source), FUND_BOOK 3d (N6, MEDIA, Q014...).

## 1. Census + rules (read-only GETs 2026-11-21 ~05:50Z; results/vercel_share/series_meta.json, mk/<SERIES>.json)
All 17 series: fee_type quadratic, fee_multiplier 1 (maker free). Rules text identical across every event of each series
(checked by de-numbering rules_primary/secondary; one version per series).
- **Vercel lab series** (KXANTHVSPEND, KXANTHVREQ, KXOPENVREQ, KXOPENVSPEND, KXMOONVSPEND, KXGOOGVREQ, KXDEEPVREQ: 32 events,
  29 settled; KXGOOGVSPEND/KXZAIVSPEND/KXZAIVREQ: 14 events, 11 settled, from 10-28): "<Lab> Vercel share of AI Gateway
  <spend|requests> across the top labs for <day D> above K%", value rounded to 1 dp, "At 10:00 AM ET on D+1, Kalshi will use
  the chart's updated value for D" (vercel.com/ai-gateway/leaderboards/labs). An event every OTHER day; open ~4-5 days ahead;
  **close 23:59 ET on D** (03:59Z/04:59Z D+1). 9 rungs (19 on two Nov events), strike_type greater.
- **KXOPENSOURCESHARE** (47 events, 42 settled; every other day until 10-31, daily since): Open Weights % of "Open vs. Closed
  Token Volume" for day D; ticker date = D+1; **close 10:00 ET on D+1** (14:00Z/15:00Z) = 14-15 h AFTER the UTC day ends.
- **KXOSSPEND** (12 events, 9 settled, from 11-02): Open Weights % of Spend, models leaderboard grouped by weights, close 23:59 ET D.
- **OpenRouter share** (KXANTHSHARE/GOOG/DEEP/OPENSHARE 10 settled each; KXTOPUSAGEAI 10): weekly, "10:00AM ET Monday updated
  market share values ... rounded to one decimal"; the Market Share chart counts **text REQUESTS** by author (page methodology:
  "The Market Share chart shows each model author's share of text requests"), not tokens. Close Sun 23:59 ET (some 10:00 ET Mon).
Volume per settled event (sum volume_fp): KXOPENSOURCESHARE 22k-76k contracts (most ~25-65k); lab series 1k-73k (Nov: 1-15k);
KXOSSPEND mostly 0 (one 26k); KXGOOGVSPEND 0-9k; KXANTHSHARE 52k-334k/week.

## 2. Settlement source (R1) -- PASSED for Vercel and OpenRouter share
- vercel.com/robots.txt: `Disallow: /api/` (pages allowed). BUT Vercel's docs (vercel.com/docs/ai-gateway/leaderboards) publish an
  **official export endpoint** `GET https://vercel.com/api/ai/leaderboard-export?dataset=models|labs|apps|providers&modality=..
  &from=YYYY-MM-DD&to=..&format=json|csv`, "cached for 24 hours", earliest 2025-10-01, data **CC BY 4.0** ("free to use ... including
  commercially"), page text "public JSON export without authentication". Data "aggregated daily" in UTC days.
- The open-weights grouping is not in the export; the page's own JS calls the undocumented
  `/api/ai/v5/gateway-model-leaderboard?groupBy=weightClass&from=..&to=..` (Cache-Control max-age=0, live). Same CC-BY data;
  under /api/ (robots) -> low-rate research reads only; flag for the lead.
- **Both endpoints serve the CURRENT partial UTC day live.** At 05:57Z 11-21 the cached export (Age 5493 s) and a cache-missed
  export (from= param) gave different 11-21 values (anthropic spend 32.63 vs 32.47; deepseek requests 18.20 vs 17.16); the 11-20
  values were identical in both (complete day). The leaderboard PAGE (ISR, X-Nextjs-Stale-Time 300) shows completed days only
  (page 11-20 values == export 11-20 values; no 11-21).
- **Rebuild** (rebuild.py -> rebuild.csv): 224/285 numeric settled values exact (round 1 dp); 284/285 within 0.4 pts; by month Sep
  21/42, Oct 88/125, **Nov 115/118**. Misses are almost all page > Kalshi by 0.1-0.4 on Sep-Oct days, several labs on the same day
  together (09-26: openai +0.2, google +0.2, deepseek +0.3) -> a later restatement of the denominator, not a definition change.
  One large miss: KXOPENSOURCESHARE-26OCT19 (day 10-18) Kalshi 62.9 vs today's 67.7 -> day values CAN be revised after settlement.
- OpenRouter share (rebuild_or.py -> rebuild_or.csv): `/api/frontend/v1/rankings/author-market-share?metric=requests&modality=text`
  reproduces 30 of the 34 settled values on weeks the endpoint has; misses are 0.06-0.08 (GOOG 13.68 vs 13.6, 15.16 vs 15.1, 17.86
  vs 17.8; DEEP 25.34 vs 25.4); 6 events use non-Monday "week of" labels absent from the endpoint. The token-based default
  (`author-market-share` with no params) does NOT match (GOOG 3.4 vs 15.1). `byok=all` returns identical request shares ->
  **the 11-19..21 BYOK default switch does not affect the share series** (it affects token totals, KXTOKENUSE). KXTOPUSAGEAI: top
  single model by weekly tokens matches 9/9 resolvable weeks (DeepSeek 8, OpenAI 1); 26SEP28 top was a stealth model, no YES.
- Live check 2026-11-21T06:01Z (live/opens_*.json): KXOPENSOURCESHARE-26NOV21 (day 11-20, value known: 65.743) quoted T64.4
  .99/1.00, T66.4 0/.04, T67+ 0/.01-.04 -> the post-period window is priced, as in TOK2. NOV22 (day 11-21, 6 h of data, partial
  71.65): T67 .37/.96, T72.5 .16/.32 -> wide.

## 3. Source-only statistics (source_panel.py -> source_stats.csv, panel.parquet; no prices)
Daily shares are NOT slow: sd of the 1-day change, sealed window 10-26..11-20: KXOPENSOURCESHARE 4.0 pts, KXOSSPEND 6.5,
KXANTHVSPEND 4.8, KXMOONVSPEND 4.8, KXOPENVSPEND 3.0, KXDEEPVREQ 2.1, KXOPENVREQ 1.7, KXANTHVREQ 0.9, KXGOOGVREQ 0.9; lag-1
autocorrelation 0.13-0.67 (dev window 0.28-0.94). Ladder spacing ~1.5-2 pts. Big level shifts vs the fit window (open token share
mean 20.3% Oct-25..Sep-26 -> 66% in Sep-Nov 2026; Anthropic spend 69.8 -> 39) -> any model must use a short trailing window.
Day-of-week range of means 1.4-9.4 pts. Reading: across days the share does not lock in; within a day it can (forward logger).
No intraday history exists: Wayback has 0 captures of either API endpoint and the 433 page captures since 09-01 show completed
days only (cdx_leaderboards.json); the v5 endpoint rejects sub-day `from` and ignores granularity params.

## PRE-REGISTRATION VS1 + VS2 (written 2026-11-21 ~06:20Z, BEFORE any Kalshi candle/price history is fetched)
Seen before writing: rules, event list with settled values (for the rebuild), source data, ONE live book snapshot (section 2).
Split by value day D: **dev 2026-09-11..2026-10-25** (20 open-share events + 17 x 7 lab events); **sealed 2026-10-26..2026-11-20**
(candles NOT fetched; user yes only; plan at dev ROI x 0.66 per STRATEGY_v2 amendment 3); forward 2026-11-21+.
Common: fee(a) = ceil_cent(0.07 a (1-a)) for 1 contract (quadratic, multiplier 1); quote = last hourly candle with end_period_ts
in (T-6h, T] (`close_dollars`, or `close` on /historical); a real price is required (YES: 0.01 <= yes_ask <= 0.99; NO: 0.01 <=
yes_bid <= 0.99, NO price 1 - yes_bid); one bet per rung; stake = a + fee; cluster = value day D (labs move together);
day-cluster bootstrap 10k; YES iff round(V_D,1) > K, modelled as V_D > K + 0.05. Source values for days <= D-1 are taken from
the 2026-11-21 fetch (revised data; Sep-Oct mean revision +0.08, one 4.8 outlier) -> mild lookahead on inputs only, flagged.
**VS1 (pre-period far rungs, taker).** T = 05:00Z on day D (D-1 complete; the 04:25Z cached export already had the final 11-20).
Model M: x = logit(V/100); OLS x_t = a + b x_{t-1} on the trailing 60 days ending D-1 (>= 30 obs; b clipped to [0,1]); p_yes =
(k + 0.5)/(n + 1), k = # in-sample residuals e with 100*sigmoid(a + b x_{D-1} + e) > K + 0.05. Bet YES if p_yes >= 0.85 and
p_yes - a - fee(a) >= 0.05; NO if p_yes <= 0.15 and (1 - p_yes) - (1 - yes_bid) - fee >= 0.05.
Placebos (same filters, thresholds, quotes): **P-persist** = AR on lag 2 (x_t on x_{t-2}) applied to x_{D-2} = the last value Kalshi
itself settled for the every-other-day lab series (what a Kalshi-only trader sees); **P-clim (base-rate shape)** = p_yes from the
trailing 60-day empirical distribution of levels V_t (no recency).
Success (dev): n >= 30 bets on >= 10 days, net ROI 95% LCB > 0, ROI(M) > ROI(P-persist) and > ROI(P-clim), edgelab dev gates
pass. Kill: any of these fails. Report top-5 contributing days/series. Secondary (descriptive): Brier(M) vs Brier(Kalshi mid) on
two-sided rungs (spread <= .10) at T, day-cluster CI; mean (mid - p) on far rungs.
**VS2 (post-period, taker).** T = each hourly candle end in [D+1 01:00Z, close]; V = source value of day D (2026-11-21 fetch;
proxy for what the live endpoint showed after 00:00Z D+1 -- forward logger measures the drift). YES if K <= V - 0.3, NO if
K >= V + 0.3, and a + fee(a) <= 0.97; first qualifying candle per rung. Placebo: V := V_{D-1}.
Success: n >= 20 on >= 5 days, ROI LCB > 0, ROI > placebo. Kill: < 5 qualifying rungs, or ROI <= placebo, or LCB <= 0.
Descriptive: per rung, hours after D+1 00:00Z until the outcome side's quote is >= .95 (book clearing time).

## 4. Log
- 06:2xZ edgelab: vault `kalshi_vercel_share` (dev 09-11..10-25, sealed 10-26..11-20 unfetched, forward 11-21+), loader
  `kalshi_vercel_share`, rule edgelab/rules/vercel_share.py; registered BEFORE any candle: **vs1_vercel_far_taker-9bc1ed8f**
  (74 trials, bar p < 6.76e-04) and **vs2_vercel_post_taker-8acc4032** (75 trials, bar 6.67e-04).
- Wrong turn: the first candle build ran beside tailpool.py; Kalshi /historical answered 429 and my fetcher returned [] after 6
  tries and CACHED it -> would have silently dropped markets (= fewer bets, biased to recent markets on /series). Killed it
  (1 market cached), made 429 retry up to 20x with backoff and never cache a failed fetch; reran alone; tailpool rerun after.
- OpenRouter share series have no point-in-time partial-week history: Wayback has 2 captures of author-market-share (wrong
  params: modality=all&traffic=batch, or/cdx_author_share.json); the endpoint is weekly only (period/interval/granularity params
  ignored). -> forward-log only; 10 settled events per series (R4 ~4/month).
- **Fastest page timing (wayback_labs.py -> wayback/labs_panel.csv; only 4 raw /labs captures since 09-20).** The capture at
  **2026-10-21 00:43Z already showed day 10-20 complete** (anthropic spend 30.43 vs settled 30.4; 43 min after the UTC day ended,
  3 h 16 m before the lab market closed at 03:59Z); captures at 10-28 22:13Z, 11-12 20:29Z, 11-14 16:45Z show through D-1 only
  (completed days). Revisions: day 10-18 values moved between settlement (10:00 ET 10-19) and the 10-21 capture (OpenAI requests
  19.1 -> 19.40 -> now 19.45; DEEP 14.7 -> 14.92; GOOG 13.6 -> 13.83; open-weights tokens 62.9 -> 67.7 now); November days have
  wayback == settled == now. So: day D is final-ish on the free page within an hour of 00:00Z D+1; occasional post-settlement
  restatements of a few tenths (one 4.8).
- **OpenRouter KXTOPUSAGEAI live definitional hazard (2026-11-21 ~06:20Z, live/or_books_*.json):** week of 11-16, byok=all model
  chart top = tencent/hy4-flash-20261116 96.77T vs deepseek v4.1-flash 49.16T; default (BYOK-excluded since 11-19..21) top =
  DeepSeek 49.19T. Book: DEEP .86/.99, TENC 0/.14 -> the book prices the BYOK-EXCLUDED chart (same as KXTOKENUSE). Settled history
  cannot distinguish (both views agree on every settled week; 09-07 OpenAI 18.18T vs Tencent 16.82T byok=all). Rules question, not
  a trade. KXTOPUSAGEAI lock-in from ai_nowcast's 71 Wayback captures is NOT computable cleanly (captures were BYOK-inclusive, the
  current default is not; or/topusage_lockin.csv mixes definitions -- do not use).
- OpenRouter share books track the free endpoint: KXDEEPSHARE-26NOV23 (week 11-16) every rung >= 28.5 at 0/.01 while the partial
  week text-request share was 25.18% (or/authreq_text_*.json, both byok variants 25.18/25.19).

## 5. RESULTS (dev scored once, 2026-11-21 ~06:52Z; sealed 10-26..11-20 never fetched)
Frame: build_frame.py -> data/vercel_share/vs_frame.parquet (1,230 dev rungs: KXOPENSOURCESHARE 180 + 7 lab series 148-151 each;
KXGOOGVSPEND/ZAI*/OSSPEND start after the dev window), vs_candles.json (1,230/1,230 with hourly candles, 0 failed fetches).
**edgelab vs1_vercel_far_taker-9bc1ed8f evaluate dev -> DEAD (5a p >= bar):** n 85 bets / 15 days, net ROI **+2.86% [-8.78,
+15.56]**, p .293 vs bar 6.67e-04 (75 trials); halves +0.7% / +5.0%; top day 2026-10-08 = 70% of profit, ROI ex top day +0.93%;
CLV mean +0.085. Prereg kill (LCB <= 0) triggered.
**edgelab vs2_vercel_post_taker-8acc4032 evaluate dev -> DEAD (5a, 5c, 5d):** n 10 < 20, 2 days, ROI +11.42% [-100, +13.31].
score.py (score_dev.log, score_dev.json, bets_*.csv; day-cluster bootstrap 10k):
| arm | n | days | losses | mean price | ROI [95% CI] |
|---|---|---|---|---|---|
| VS1 M (x_{D-1}) | 85 | 15 | 17 | .761 | +2.86% [-9.09, +15.12] |
| VS1 P-persist (x_{D-2}) | 58 | 15 | 20 | .749 | -14.45% [-30.85, +5.58] |
| VS1 P-clim (60-d levels) | 133 | 16 | 48 | .708 | -11.83% [-28.04, +5.24] |
| VS2 V (day-D value) | 10 | 2 | 2 | .706 | +11.42% [-100, +13.31] |
| VS2 P-V1 (day D-1 value) | 141 | 19 | 132 | .354 | -82.58% [-100, -55.54] |
- VS1 beats both placebos on the point estimate (by ~14-17 pts) but its own CI spans 0 -> not distinguishable from 0. The D-1
  value carries real information vs D-2 (Brier on 158 two-sided rungs, spread <= .10, 17 days: model .0612, P-persist .0854,
  P-clim .1700; model - persist -.0242 [-.0472, +.0026] n.d.), but **the Kalshi mid is better calibrated than the model**: mid
  .0482, model - mid +.0130 [-.0235, +.0582] n.d. Far rungs (model p >= .85 / <= .15, n 120): model p_side .958, mid_side .921,
  realised .983 -> the certain side is slightly cheap at MID, but the bets the rule could actually take were wide-spread rungs at
  mean ask .761 with 80% realised (68/85) -- model overconfident exactly where it disagrees with the book (rule PA again).
- Top-5 VS1 days by pnl: 10-08 +1.32, 10-20 +1.11, 10-22 +1.00, 10-18 +0.99, 10-06 +0.77 (total +1.89 on 85 bets); worst 10-24
  -2.16, 09-22 -2.07. By series: ANTHVREQ +2.15, DEEPVREQ +1.75, OPENVREQ +1.04, OPENVSPEND +0.68, GOOGVREQ +0.25, OPENSOURCESHARE
  -0.12, MOONVSPEND -1.07, ANTHVSPEND -2.79 (6 losses of 13).
- VS2's 10 bets are all KXOPENSOURCESHARE: 8 wins on day 09-20 (series' first week, event volume 9.6k) and 2 losses on day 10-18 =
  the revised day: today's data says 67.7, Kalshi settled 62.9; the book priced T64.1 YES at .07 because the page then showed ~62.9.
  My "V" (current data) was the stale number, not the book -> revision trap.
- **Spreads at T1 (05:00Z):** two-sided share lab .967 / open share .914, but median spread on two-sided rungs **.56 (lab)** and
  **.23 (open share)** -> R5 = 0.
- **Book clearing after the day ends (VS2 descriptive, vs2_clearing.csv, vs2_lab_quotes.log):** open share (14 h window): 89.7% of
  rungs reach >= .95 on the outcome side, 52.6% within 1 h, median 1 h. Lab (4 h window): only 38.2% reach .95, but the outcome-side
  buy price at the first candle >= D+1 01:00Z is 1.00 (no offer) on 83.8% of lab rungs and .97-.99 on 16.0%; <= .90 on 0.2%. The
  "uncleared" lab rungs are EMPTY books, not stale offers. Open share: <= .96 offers on 4.0% of rungs (6 rungs >= 1 pt from V).
Verdict: **VS1 dead (not distinguishable; market better calibrated; R5 spreads .23-.56); VS2 dead (book clears within ~1 h or
offers nothing).** No sealed ask. Mechanism = "watched number on a live free endpoint" (as TOK1/TOK2), not an unwatched tail.
- **Tail pools (tailpool.py, 30 d to 2026-11-21, tailpool_30d.json):** loser-paid <= 15c per 30 d: KXOPENSOURCESHARE $20,156 (full
  pool $64,044; tail risk $6,956); KXANTHVSPEND $4,166; KXOPENVREQ $2,731; KXANTHVREQ $2,637; KXMOONVSPEND $2,523; KXOPENVSPEND
  $2,405; KXDEEPVREQ $2,379; KXGOOGVREQ $1,934; KXOSSPEND $810; KXZAIVSPEND $395; KXZAIVREQ $237; KXGOOGVSPEND $165. OpenRouter:
  KXDEEPSHARE $21,619, KXGOOGSHARE $18,942 (tail risk $114,145 = cheap side won big), KXOPENSHARE $13,596, KXANTHSHARE $8,574,
  KXTOPUSAGEAI $133. All Vercel lab series except GOOGVSPEND/ZAI* clear the $500 amendment-1 bar.
- **OpenRouter boundary-settlement risk:** KXGOOGSHARE-26OCT12 "above 17.8" settled NO at 17.8; today's endpoint says 17.858 (would
  round to 17.9 = YES); that rung traded 183,200 contracts, last .22. All 3 GOOG mismatches are Kalshi 0.06-0.08 BELOW today's
  endpoint (15.1 vs 15.156, 13.6 vs 13.679, 17.8 vs 17.858) -> either late-data revisions after the Monday 10:00 ET snapshot or a
  pre-final snapshot. Rungs within 0.1 of the live value carry settlement risk the endpoint cannot resolve (for source_shift).
- **Forward poller caveat (found 07:05Z):** the documented export is CDN-cached per URL ("cached for 24 hours"): my same-day
  polls with fixed from/to were HITs (Age 3,511 s) and froze the 11-21 lab partial at 32.47 while a nonce-param MISS gave 31.85.
  The v5 weightClass endpoint (max-age=0) was live in every snapshot (11-21 open tokens 71.58 -> 71.47 -> 71.58 over 06:05-06:55Z).
  poll.py now adds a nonce. Poller stopped at ~07:05Z (8 snapshots; loops stay off) -- intraday lock-in remains unmeasured.

## 6. Forward logger (proposal; NOT installed -- no launchd)
results/vercel_share/poll.py = one read-only snapshot per run -> fwd/snapshots.jsonl: Vercel labs export + weightClass for the
last 3 UTC days (cache-busted by explicit from/to) + every open Kalshi market of the 12 Vercel series (bid/ask/size/last/vol/OI).
Ran in-session every 10 min from 2026-11-21T06:05Z (fwd/poll.log). Analyzer: fwd_lockin.py (|partial - final| by hours into
the UTC day; post-midnight drift of the completed day). Proposed cadence if the lead installs it: every 15 min (2 Vercel GETs +
12 Kalshi /markets GETs per run). What it answers that history cannot: (a) how fast the intraday share converges (lock-in curve),
(b) whether the book's far rungs track the live partial share during day D, (c) post-midnight drift of day D before 10:00 ET D+1.

## Learnings
### Vercel publishes its AI Gateway leaderboard data as an official CC-BY JSON export
`https://vercel.com/api/ai/leaderboard-export?dataset=labs|models&from=YYYY-MM-DD&to=..` (documented at
vercel.com/docs/ai-gateway/leaderboards, CC BY 4.0, back to 2025-10-01) reproduces the Kalshi Vercel lab-share settlements: 224/285
exact over all settled values, Nov 115/118. Open-weights share needs the page-JS endpoint
`/api/ai/v5/gateway-model-leaderboard?groupBy=weightClass`. Source: results/vercel_share/rebuild.csv, rebuild.py.
### Free live sources include the current partial UTC day; the book clears about an hour after the day ends
Both Vercel endpoints serve the running partial day (cache-missed fetches differ minute to minute). The /labs page showed the
completed day 43 min after 00:00Z (Wayback 2026-10-21 00:43Z). KXOPENSOURCESHARE rungs reached >= .95 on the outcome side
within 1 h on 52.6% of rungs (median 1 h). On lab rungs after the day ends, the outcome side has no offer 83.8% of the time.
Source: results/vercel_share/vs2_clearing.csv, vs2_lab_quotes.log, wayback/labs_panel.csv.
### Daily AI usage shares are not slow-moving
The sd of the 1-day change is 0.9-6.5 pts in the 10-26..11-20 window, lag-1 autocorrelation 0.13-0.67, and level shifts are large
(open-weights token share 20% -> 66% within a year). A D-1 persistence model beat a D-2 one (Brier .0612 vs .0854) but lost to the
Kalshi mid (.0482). Source: results/vercel_share/source_stats.csv, score_dev.log.
### VS1 far-rung rule on Vercel share ladders is dead: CI spans zero, market better calibrated, spreads .23-.56
n 85 over 15 days, ROI +2.86% [-8.78, +15.56], p .29 (edgelab vs1_vercel_far_taker-9bc1ed8f). Placebos scored -14.5% / -11.8%.
70% of profit came from one day. Median two-sided spread at 05:00Z was .56 (lab) and .23 (open share). Source: score_dev.log.
### Current source data is not a point-in-time proxy on revised days
Day 10-18 open-weights share settled 62.9 but reads 67.7 today. The book priced T64.1 YES at .07 (correct). A post-period rule
keyed to today's 67.7 lost both of its bets that day. Lab values for Sep-Oct were restated +0.08 on average after settlement.
Source: rebuild.csv, bets_VS2_V.csv.
### OpenRouter market-share series settle on TEXT REQUESTS by author, not tokens
`author-market-share?metric=requests&modality=text` matches 30/34 settled ANTH/GOOG/DEEP/OPENSHARE values; the token default gives
GOOG 3.4 vs Kalshi 15.1. byok=all leaves request shares unchanged, so the BYOK switch hits token series only.
Source: results/vercel_share/rebuild_or.csv.
### KXTOPUSAGEAI carries the BYOK definitional hazard for the week of 11-16
byok=all top model = Tencent hy4-flash 96.77T; default (BYOK-excluded) = DeepSeek 49.19T. The book has DeepSeek at .86/.99 and
Tencent at 0/.14, so it prices the BYOK-excluded chart, as KXTOKENUSE does. Treat this as a rules question, not a trade.
Source: results/vercel_share/live/or_books_*.json, or/models_*.json.
### Shared Kalshi rate limits can silently empty a candle cache
/historical candlesticks cost 50 of a 600-token budget. A fetcher that returned [] after 6 straight 429s and cached it would have
dropped markets without a trace. Retry on x-ratelimit-reset and never cache a failed fetch. Source: build_frame.py.
