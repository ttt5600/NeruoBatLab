# news_lag -- does Kalshi reprice slowly after official, exact-time NFL/NBA availability news?

Agent topic `news_lag`, started 2026-11-15 09:28Z (research subagent of the quant-trader lead; AGENT_PROTOCOL rules:
read-only, no orders, no git). Code: scripts/news_lag.py (dev), scripts/news_lag_logger.py (forward).
Scratch: results/news_lag/.

## Prior (read, not redone)
Q011 NBA injury-report PDF lag DEAD (notes/q011_nba_injury_lag.md): ~half the move happens BEFORE the PDF; post-report
drift +0.34c < half-spread 0.55c + fee 1.75c -> T1 net CLV -1.6c. Section 3d: GDELT news -> price leads news by ~4 h.
So this study only targets (a) large-impact events (starting QB) whose (b) official release has a fixed time.

## PRE-REGISTRATION (written 2026-11-15 ~09:45Z, BEFORE any Kalshi price for this question was fetched or joined)

### News events (NFL, dev)
- Official release: NFL game-day inactive lists, submitted and published at kickoff - 90 min. t0 := KO - 90 min
  (KO = nflverse schedule gameday+gametime, America/New_York -> UTC). The forward logger measures the real
  publication time on 2026-11-15; the dev uses the rule time.
- Source tables (nflverse releases, downloaded 2026-11-15 to results/news_lag/raw/): injuries_<season>.parquet
  (final weekly game-status designation `report_status`), snap_counts_<season>.parquet.
- Event filter E-QB (frozen): a QB with report_status in {Questionable, Doubtful} for week w, who was his team's
  STARTER: in his team's most recent game of the same season before week w in which he appeared in snap_counts, he
  had offense_pct >= 0.70 and the most offense snaps among that team's QBs. Week-1 rows (no prior game) dropped.
- Resolution (known at t0 from the inactive list): PLAYED = offense_snaps > 0 in week w snap counts; NOT PLAYED
  otherwise. Proxy risk: an active QB who does not play would be mislabeled as inactive (lookahead); the rows where
  this could happen are listed and counted, not silently used.
- Benefiting team: NOT PLAYED -> opponent; PLAYED -> the QB's team.
- Trials (k = 2, one-sided bar p < 0.05/2 = 0.025):
  - T1: all E-QB events.
  - T2: "against the designation" subset: Questionable QB NOT PLAYED, or Doubtful QB PLAYED.
- Dedup: one bet per game. Both teams with events pointing to the same team -> one bet; opposite -> drop (counted).

### Prices and trade rule (frozen)
- Kalshi KXNFLGAME per-team YES markets; 1-min candlesticks (historical endpoint, then series endpoint), read-only
  via scripts/kx_throttle.py kget. mid = (bid+ask)/2 of last candle with end <= t and 0 < bid <= ask < 1.
- Decision td = t0 + 3 min (= KO - 87 min). Entry = first candle with end in [td, td + 10 min], end < KO, two-sided,
  ask - bid <= 0.10; else no bet (counted). Taker: buy YES of the benefiting team's market at its YES ask.
- Fee 0.07 p (1-p), unrounded. PRIMARY metric: net CLV = mid_KO - ask - fee (mid_KO = mid at KO).
  Secondary: ROI = (settle - ask - fee)/ask. Descriptive: gross drift mid_KO - mid(td); move shares.
- Statistic: mean net CLV per bet, bootstrap over games 10,000 resamples numpy default_rng(0), 95% percentile CI,
  p = share of resampled means <= 0.
- PASS (dev) = mean > 0 AND p < 0.025 AND both chronological halves positive. A dev pass earns only the holdout
  (needs the lead's / user's yes).
- KILL: T1 mean net CLV <= 0, or no trial passes -> dead; the holdout is not priced.
- Too-good guard: mean net CLV > +3c = bug until rows are inspected (KO time, side mapping, candle alignment).

### Descriptive lag questions (no pass/fail, reported with CIs)
- D1 (all games, news-agnostic): mean |mid(t+10m) - mid(t)| for the window starting at t = KO-91 min vs placebo
  windows starting at KO-151, KO-121, KO-61. A real inactive-list shock shows as a larger KO-91 window.
- D2 (E-QB events, benefit direction): mid path at KO-24h, -6h, -3h, -2h, -95m, -91m, -85m, -80m, -60m, -30m, KO;
  share of the (KO-24h -> KO) move that happened before t0 (KO-90); for events with a >= 3c move after t0, the
  minutes until half of the post-t0 move was done (1-min resolution; tick trades if time allows).

### Split
- DEV = 2025 season (REG+POST). HOLDOUT = 2026 season weeks 1-10: event table built (news side only) but NOT priced.

### NBA
- Q011 already covers the injury-report PDF to tip-10 min (dead). New angle only forward: late scratches and
  starting lineups in the last 30 min before tip (no free historical timestamped source found yet -> forward logger
  records news time + Kalshi book; no dev trial).

## Plan
1. Event table from nflverse (news side only), count + power. 2. Prices for DEV games only. 3. Score once.
4. Forward logger for 2026-11-15 (NFL inactives ~16:30Z for 1 pm ET games; NBA tonight) in results/news_lag/forward/.

## Log
- 09:28Z started; inbox row written.
- 09:31Z forward logger launched (nohup; results/news_lag/forward/logger.out): scripts/news_lag_logger.py
  --until 2026-11-16T08:00Z. Kalshi: one /markets?series_ticker call per league per poll (5 s hot / 120 s cold) via
  kx_throttle. News: ESPN summary injuries+boxscore, ESPN core per-game roster (404 until posted -> first-200 time is a
  candidate inactive-list timestamp), ESPN news feed, NBA 15-min injury PDFs. NFL London game NE-DET KO 14:30Z
  (inactives ~13:00Z), 8 games at 18:00Z (inactives ~16:30Z), 3 at ~21:05-21:25Z, SNF PIT@CIN 01:20Z; NBA 9 games
  18:00Z-02:00Z.
- 09:50Z edgelab: news_t1_nfl_qb_inactive-0c76768a, news_t2_nfl_qb_against_designation-e63cb28f (family kalshi_news
  now 6 trials all-time -> edgelab bar p < 8.33e-3, stricter than my k=2 bar 0.025; the STRICTER edgelab bar governs).
- 09:55Z event table (news side only, no prices): results/news_lag/events.parquet. 58 Q/D QB rows -> kept 27
  (drops: 18 no previous same-season game incl. week 1, 13 not starter). DEV 2025: 19 events, all Questionable
  (14 did NOT play, 5 played); 0 Doubtful. HOLDOUT 2026: 8 (5 not played). T2 dev = 14. Played (snaps) vs
  nflverse starting QB: 0 mismatches.
- POWER (written before prices): n = 19 dev bets. If per-bet net-CLV SD ~5c, SE ~1.1c; only a true edge > ~3c could
  clear p < 0.0083. T1/T2 will almost surely be "not distinguishable" unless the effect is huge; the informative
  parts are D1 (all ~285 dev games, news-agnostic KO-90 window) and D2 (move timing on the 19 events).
- 10:20Z prices: 558 KXNFLGAME tickers, 1-min candles KO-26h..KO+1min (results/news_lag/candles_dev.parquet, 779,599
  rows, median 1,468 candles/ticker); 279 of 285 dev games matched to a Kalshi event (market_map_dev.parquet;
  unmatched: 2025_18_CAR_TB, 2025_18_SEA_SF, 2025_21_NE_DEN, 2025_22_SEA_NE). KO check: 2025_01_DAL_PHI ko
  1757031600 equals kalshi_early_nfl_markets ko_ts.

## DEV RESULT (2026-11-15, scored once; results/news_lag/dev_results.json, bets_dev.csv, paths_dev.csv, d1_dev.csv) -- KILL
| trial | bets | net CLV $/contract [95% game CI] | p | halves | gross drift td->KO [CI] | ROI [CI] |
|---|---|---|---|---|---|---|
| T1 all starting-QB Q events | 18 | -0.0152 [-0.0213, -0.0085] | 1.00 | -0.0186 / -0.0118 | +0.0047 [-0.0017, +0.0117] | +14% [-36, +69] |
| T2 against designation | 14 | -0.0169 [-0.0232, -0.0102] | 1.00 | -0.0193 / -0.0145 | +0.0025 [-0.0043, +0.0093] | +32% [-23, +93] |
- 1 event no bet (2025_20_SF_SEA: playoff game unmatched). Mean quoted spread at entry 1.0c; the fee at ~50c is 1.75c,
  so the taker starts ~2.2c behind and the drift after the inactive list (+0.47c, n.d.) does not cover it.
  ROI is positive but its CI spans -36%..+69% on 18 bets: luck-shaped, CLV is the metric (pre-registered).
- D1 (279 games, home-team market, news-agnostic): mean |mid move| over 10 min starting KO-91 = 0.131c vs placebo
  windows KO-151 0.075c, KO-121 0.097c, KO-61 0.118c (KO-31 0.194c). KO-91 minus placebo mean +0.034c
  [-0.010, +0.079], p = 0.064: not distinguishable. Per-minute mean |move| at KO-88/-89 is 0.029/0.022c vs
  0.013-0.030c in the surrounding half hour: NO burst at the inactive release. The whole pre-game hour moves ~0.1c per
  10 min on average; moves grow only in the last 30 min (KO-5: 0.09c/min).
- D2 (18 events, benefit direction, mid relative to KO-91m): -24h -0.58c, -6h +0.08, -2h -0.03, -85m -0.08, -60m +0.19,
  -30m +0.44, KO +0.44c. The total -24h -> KO move is only ~1c: for these QBs the outcome was priced more than a day
  before (e.g. Purdy, Murray, McCarthy flat at their price all day). 57% of that small move came before KO-90.
  Inspected all 18 rows (paths_dev.csv): mids are flat across KO-95 .. KO-80 on 16/18; no row is too good.
- Verdict: KILL per the pre-registered rule (T1 mean net CLV <= 0; 0/2 pass). Holdout (2026 wk1-10, 8 events) never
  priced. edgelab news_t1-0c76768a and news_t2-e63cb28f retired (-> dead) 2026-11-15 with this file as the reason.
  Not yet critic-reviewed (a kill on a pre-registered rule; the lead may run /finding).
- Mechanism: "Questionable" QBs who sit are reported out by team/insider news on Friday-Saturday; the official
  KO-90 inactive list is a formality for them. The only possible surprise class is a true game-time decision announced
  at KO-90, which did not appear in 2025 dev (0 Doubtful QBs; Q QBs who played moved the price +0.0..+4c, mostly in the
  last hour, e.g. Herbert 2025_14 +4c between KO-80 and KO-30, AFTER the list -> the market drifted, not jumped).
- NBA: no free timestamped historical lineup source; cdn.nba.com liveData returns 403 without browser headers; Q011's
  cached report data (data/q011/) is not present in this worktree, so no new NBA dev. Forward log only.

## Forward (2026-11-15) -- running, descriptive only (no trial, dev is dead)
- Process: `pgrep -f news_lag_logger` (started 09:31Z, exits 2026-11-16 08:00Z on its own). Output
  results/news_lag/forward/{kalshi,news}_YYYYMMDD.jsonl, stdout logger.out. Verified in a 60 s forced-hot test:
  Kalshi poll every ~5.5 s per league (11 polls/min/league = 0.37 req/s total through kx_throttle), ESPN injuries,
  roster, news rows written.
- Read it after the games: `../.venv_quant/bin/python scripts/news_lag.py forward` -> results/news_lag/forward_events.csv
  (per ESPN injury/roster/boxscore change: minutes to KO, Kalshi mids of both team markets at -2, 0, +1, +3, +10 min).
  Questions it answers: (1) when does ESPN's game roster first return 200 / DNP set appear relative to KO-90 (the real
  inactive-list timestamp on a free feed)? (2) does any Kalshi mid move >= 2c within 10 min of an NFL inactive or an NBA
  late scratch (boxscore DNP/starter change, PDF status change)? If no move >= ask+fee shows up on 11-15, close the idea.
- Proposal (not done): if the lead wants more forward data, add the logger to launchd for NFL Sundays; it is read-only.

## Learnings

### The NFL inactive list carries no price news for Kalshi KXNFLGAME
Mean |10-min mid move| at KO-91 is 0.131c vs 0.097c average in placebo windows, difference +0.034c [-0.010,+0.079]
over 279 2025 games (results/news_lag/dev_results.json D1). There is no repricing burst to be fast on.

### Questionable starting QBs who sit are priced a day early
On 18 2025 events the whole KO-24h -> KO move in the benefit direction was ~1c, 57% of it before the inactive list
(results/news_lag/paths_dev.csv). Insider/team news front-runs official NFL designations even more than Q011 found for NBA PDFs.

### Taker news-lag on Kalshi needs > 2.2c of post-news drift
Quoted spread 1c + fee 1.75c at ~50c; post-list drift was +0.47c [-0.17,+1.17], net CLV -1.52c [-2.13,-0.85]
(news_t1, n = 18). Same shape as Q011 (drift +0.34c, net -1.6c): small real drift, far below taker cost.

### nflverse injuries + snap counts make a clean QB availability event table
injuries_<season> report_status + snap_counts (pfr id via roster_weekly) reproduce the nflverse starting QB with 0
mismatches on 27 events (results/news_lag/events.parquet); reuse for any QB-news study.
