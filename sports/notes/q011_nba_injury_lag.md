# Q011 (NBA): does Kalshi react slowly to official NBA injury-report status changes?

Status: PRE-REGISTERED 2026-10-29, written BEFORE any Kalshi price was fetched for this question and before any
outcome join. Code: scripts/q011_reports.py (news), scripts/q011_nba_injury_lag.py (events, prices, dev).
Ledger dataset: `q011_nba_injury_lag`.

## Question and mechanism
Our only working sports mechanism is TIMING (totals model beats openers, not closes). If Kalshi's NBA game markets
take minutes to absorb an official injury-report status change, a rule that buys the benefiting side within a few
minutes of the report's PUBLICATION should earn closing-line value (CLV) to tip-off.
Prior evidence against (section 3d): rumor/GDELT spikes -> Kalshi, "price moves BEFORE news (median 4 h)";
NFL/NBA in-game repricing within ~1 min; L2 sides losses were "underdogs priced off rest/injury news".
So the null is expected; the test is cheap because the reports are public with timestamps.

## News data
- Source: https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HHAM.pdf (hourly, 2025-10-21 ..
  2025-12) and ..._YYYY-MM-DD_HH_MMAM.pdf (every 15 min, 2026-01 .. now). Verified by probing 2026-10-29.
- TIMESTAMP TRAP found before any analysis: in the hourly era the URL label is NOT the publication time. Example
  `Injury-Report_2025-11-10_01PM.pdf`: PDF header "11/10/25 01:30 PM", HTTP Last-Modified 18:30:04 GMT (= 13:30 ET).
  15-min era example `_2026-01-15_05_15PM`: header 05:15 PM, Last-Modified 22:15:06 GMT (= 17:15 ET).
- Publication time `pub` = max(HTTP Last-Modified, PDF header time). Using the later of the two is conservative
  (a re-upload can only push Last-Modified later). Reports where they differ by > 10 min are counted and reported.
- Fetched: game days only (ESPN schedule, regular + play-in + playoffs), labels 07:00 .. 23:45 ET.

## Event definition (frozen)
- Status category per (game, team, player) in a report: OUT = {Out, Doubtful}; Q = {Questionable};
  IN = {Probable, Available} or not listed while that team's report is submitted. Level IN=0, Q=1, OUT=2.
  Teams marked NOT YET SUBMITTED in either report -> no event for that team in that pair.
- An EVENT = a level change for a player between two CONSECUTIVE fetched reports R_prev -> R_cur that both list
  the game, with pub(R_cur) - pub(R_prev) <= 20 min (15-min era) or <= 65 min (hourly era).
  Event time t0 = pub(R_cur).
- Window: tip - 10 h <= t0 <= tip - 10 min, tip = ESPN scheduled start_utc.
- Impact player: mean ESPN box minutes over his last 10 games PLAYED (minutes > 0, 2024-25 onward) strictly
  before the game's date >= 25.0, with >= 5 such games. Name join: report "Last,First" -> normalized
  first+last (ascii, alphanumerics) == normalized ESPN name; ambiguous/unmatched names are dropped and counted.
- Direction: level up (less available) -> benefiting team = OPPONENT; level down -> benefiting team = player's team.
  If one game has events pointing both ways in the same report pair they are still separate events.

## Decision, prices, metric (frozen)
- Prices: Kalshi 1-min candlesticks (public API, /historical/markets/{t}/candlesticks then /series/... fallback),
  cached in data/q011/. mid(t) = (bid+ask)/2 of the last candle with end <= t and 0 < bid <= ask < 1.
- Decision time td = t0 + L, L = 3 min (primary); sensitivity L = 2 and 5 min (descriptive only).
- Entry = the first candle with end >= td and end < tip with a two-sided quote, ask - bid <= 0.10, and end <= td + 10 min
  (otherwise no bet; counted). Taker at ASK of the side bought (YES ask; for NO the NO ask = 1 - yes_bid). 1 contract, hold.
- Dedup: events in the same game with the same t0 that buy the same market side collapse into ONE bet (two players
  ruled out in one report = one bet). Events with the same t0 pointing opposite ways in one game: both dropped (counted).
- Fee: 0.07 * p * (1 - p) per contract, unrounded (sensitivity: ceil to the cent).
- PRIMARY metric per bet: net CLV = mid_tip(side) - entry_price - fee(entry_price), mid_tip = mid at tip.
  SECONDARY: ROI = (settle - price - fee) / price. Descriptive: gross drift mid_tip - mid(td); pre-news move
  mid(t0) - mid(t0 - 60 min) and mid(t0) - mid(t0 - 15 min) in the benefit direction; share of the
  (t0-60 -> tip) move that happened before t0.

## Trials (k = 4, bar one-sided p < 0.05/4 = 0.0125)
- T1 KXNBAGAME: buy the benefiting team's YES, all events.
- T2 KXNBAGAME: same, only players with mean minutes >= 30 (stars).
- T3 KXNBASPREAD: among the game's spread markets with a two-sided quote at td, the one whose mid is closest to 0.50;
  buy YES if its team is the benefiting team, else NO.
- T4 KXNBATOTAL: level up (player less available) -> UNDER = NO; level down -> OVER = YES; market with mid closest to
  0.50 at td.
- Statistic: mean net CLV per contract; game-clustered bootstrap, 10,000 resamples, numpy default_rng(0);
  p = share of resampled means <= 0. 95% percentile CI.
- PASS (dev) = mean > 0 AND p < 0.0125 AND both chronological halves of dev (split at the median event date)
  positive. A dev pass only earns a sealed run, which needs the lead's / user's yes.

## Split
- Distinct ET game dates carrying >= 1 qualifying event (news-side only; computed before prices), sorted:
  first 70% = DEV, last 30% = SEALED. Sealed prices are NOT fetched in this run.

## Kill
- If T1 dev mean net CLV <= 0, or no trial passes, Q011-NBA goes to section 3d and the sealed half is never run.
- Too-good guard: any dev mean net CLV > +3c is treated as a bug until rows are inspected (pub time vs label,
  candle alignment, tip time, side mapping).

## Coverage and power (filled from the event table BEFORE any Kalshi price was fetched; 2026-10-29)
- Reports: 12,070 labels on 223 ESPN game days (2025-10-21 .. 2026-10-29), 11,963 HTTP 200, 0 parse errors.
  Format switch found on 2025-12-22 (that day has 2 hourly + 15-min files); first build mislabeled 12-22..12-31
  as hourly (all 403) and was fixed before events were frozen. Missing days: 2026-02-15 (All-Star, 0/68),
  partial 2025-12-22, 2026-04-30 (66/68), 2026-10-29 (today, in progress).
- Publication time: Last-Modified - PDF header time median 5 s, p99 15 s, max 87 s over 11,963 reports; 0 differ by
  > 10 min. Hourly-era labels HH are published at HH:30 (header and Last-Modified agree), so labels were never used.
  Live check 2026-10-29: _07_45PM first returned HTTP 200 at 23:45:30 UTC (Last-Modified 23:45:27);
  _08_00PM first 200 at 00:00:05 (Last-Modified 00:00:04). Poll every 10 s, so Last-Modified ~ public availability.
- Events: 3,787 raw in-window status changes; 117 unmatched names (rookies/two-ways), 0 ambiguous; 1,574 impact
  events. Transitions: Q->IN 721, Q->OUT 600, IN->Q 150, IN->OUT 79, OUT->IN 13, OUT->Q 11. Flip-backs within
  35 min (parse-glitch check) 5 of 3,670 raw.
- Split: 200 qualifying dates; DEV = dates <= 2026-03-21 (1,189 events, 644 games, 633 star events;
  396 hourly-era + 793 15-min-era). SEALED = 2026-03-22 .. 2026-10-29 (385 events, 204 games) -- not priced here.
- Hurdle: entry at ask pays the half-spread (~0.5-1c) plus fee (1.75c at 50c), so the price must keep drifting our
  way by ~2.5c on average after t0 + 3 min just to break even.
- Power: ~600 games; if per-bet net-CLV SD is ~5c (unknown until priced), the game-clustered SE is ~0.2-0.3c, so a
  true +1c is detectable at p < .0125 with power near 1; a FAIL here is a real negative, not low power.

## DEV RESULT (2026-10-30, scored once; ledger dataset q011_nba_injury_lag, 4 rows) -- FAIL, KILLED
Source: data/q011/dev_results.json, data/q011/bets_T*.csv, data/q011/dev.log. Net CLV in $/contract (1.0 = $1).
| trial | bets (games) | net CLV [95% game-cluster CI] | p | halves | gross drift entry->tip [CI] | ROI [CI] |
|---|---|---|---|---|---|---|
| T1 game, all | 884 (605) | -0.0161 [-0.0176, -0.0146] | 1.00 | -0.0167 / -0.0156 | +0.0034 [+0.0019, +0.0049] | -1.8% [-10.6, +7.7] |
| T2 game, stars >= 30 min | 549 (419) | -0.0164 [-0.0184, -0.0144] | 1.00 | -0.0151 / -0.0173 | +0.0034 [+0.0013, +0.0053] | -1.6% [-12.1, +9.2] |
| T3 spread ~50c | 875 (603) | -0.0206 [-0.0226, -0.0185] | 1.00 | -0.0201 / -0.0209 | +0.0044 [+0.0025, +0.0065] | -4.8% [-11.4, +2.1] |
| T4 total ~50c | 876 (603) | -0.0214 [-0.0236, -0.0191] | 1.00 | -0.0204 / -0.0220 | +0.0037 [+0.0015, +0.0060] | -0.6% [-7.4, +6.2] |
- Latency sensitivity (T1): L=2 min -0.0157 [-0.0172, -0.0142]; L=5 min -0.0163 [-0.0178, -0.0148]. Cent-ceil fee T1 -0.0204.
- Mechanism: Kalshi DOES drift a little after the report (T1 mid path relative to mid at t0: -60 min -0.0076,
  -15 min -0.0045, +5 min +0.0041, +15 min +0.0051, +60 min +0.0061, tip +0.0073), and the post-entry drift is
  significantly > 0 (+0.34c), but the taker pays half-spread 0.55c + fee ~1.75c, so net CLV is about -1.6c.
- News already in the price before the report: about half of the t0-60 -> tip move happened BEFORE publication
  (T1 51%, T2 57%, T3 46%, T4 46%). Mid moved >= 2c in the benefit direction in the hour before the report on
  18% of T1 bets (T2 25%), >= 2c against on 2%. Consistent with team reporters / shootaround news front-running the PDF.
- By transition (T1, descriptive): Q->IN 399 bets -0.0180; Q->OUT 327 -0.0168; IN->Q 107 -0.0108 (largest
  post-report move +1.5c); IN->OUT 42 -0.0065; OUT->* 9. None positive.
- Too-good check not triggered (all negative). Publication-time check passed (above). Dev price data:
  data/q011/candles_dev.parquet (1-min candles, median spacing 60 s, median quoted spread 1c).
- Verdict: KILL per the pre-registered rule (T1 dev net CLV <= 0; 0/4 pass). Sealed half (2026-03-22 .. 10-29)
  never priced; do not run it. Open lead (NOT tested, would need its own prereg): the +0.3-0.4c post-report drift
  is real, so a MAKER entry (no spread, lower fee) is the only version that could clear costs; adverse selection
  on fills is the obvious risk (L19 fav-maker died that way).
- Deviations / oddities: (1) ladder candles fetched by bracket expansion around p = 0.5 (Kalshi 429s, shared with
  another session's pull) -- equivalent to the pre-registered "closest to 0.50" choice for monotone ladders.
  (2) While checking the candle schema I printed one quote of one dev market (KXNBAGAME-26FEB12PORUTA-POR,
  0.71/0.72, ~3-6 h before close) before scoring; no outcome was joined. (3) 188 T1 event rows dropped as
  same-report conflicts (teammate Q->IN and Q->OUT in one report), as pre-registered. (4) Ledger `sha` is the HEAD
  commit; the q011 scripts were uncommitted at run time.
