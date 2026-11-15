# L22 -- Kalshi politics favourites 60-85c, days before close (agent pol_favourites, started 2026-11-15)

Question: Le (arXiv 2602.19520 v2, Table 4) finds Kalshi Politics markets UNDERconfident at 2d-1mo (recalibration slope
1.83; raw 0.75 resolves 0.886; 6,609 markets, no returns, no per-horizon CIs). Burgi-Deng-Whelan find politics n.s.
Does buying the favourite at 60-85c, 7 days before scheduled close, at the real ask + taker fee, earn a positive return on
markets settled AFTER Le's sample (2026-07-01..2026-11-15, incl. the 11-03 midterms)? Does the underconfidence (slope > 1)
replicate on that sample?
Not a redo: LX2 (notes/lx2_tail_no_carry.md) tested only >= 90c at >= 90 d (politics -0.2%, crosses 0). pm_timing found
Kalshi favourites ~0 after fees with no time pattern (small-market sample, all categories) -- a weak negative prior.

Code: scripts/pol_favourites.py (meta / prices / score). Data: data/pol_favourites/. Results: results/pol_favourites/.

## Log
- 2026-11-15: started; inbox row. Kalshi /historical/cutoff = 2026-09-16 (markets settled before that live under
  /historical). First metadata pull = /events?status=settled&min_close_ts=2026-06-01 (nested markets), categories
  Politics / Elections / World.
- WRONG TURN: /events ignores min_close_ts (a filter at 2027-01-15 still returned 200 settled events), so the first
  pager was walking every settled event ever; killed. /events of a series whose markets moved to /historical return
  NO nested markets. /historical/markets also ignores min_close_ts. Fix: per series (5,196 Politics/Elections/World
  series from results/pm_literature_scout/series.json): /markets?status=settled&min_close_ts (honoured) +
  /historical/markets paged in full, filtered at ingest on metadata only (binary, close >= 2025-01-01, open <= close-3d).

## PRE-REGISTRATION (written 2026-11-15 ~12:45Z, before any candle price or market result was loaded or inspected)
Universe: Kalshi series with category Politics or Elections (primary); World reported separately, not in the pass test.
Binary markets, result yes/no, metadata from the per-series pull above.
Windows (by settlement_ts, UTC):
  PRIMARY = 2026-07-01 .. 2026-11-15 (after Le's sample; includes the 11-03 midterms; one-shot, no tuning).
  PAPER-ERA = 2025-01-01 .. 2026-06-30 (overlaps Le's sample: an in-sample replication, reported separately; it is not
  a dev set for the primary -- the rule below is fixed from the paper, not fitted on any data).
Anchor S (primary) = expected_expiration_time (scheduled at listing, LX2 precedent). close_time is NOT the primary anchor
because an early close is the event happening (lookahead: KXNEXTROMANIAPM expected 2045, closed 2026-10-20).
Sensitivity anchor (reported, not a pass criterion): S' = close_time, flagged lookahead-prone.
Entry t = S - 7 d (trial T7, primary); secondary T14 = S - 14 d, T3 = S - 3 d.
Eligible at t: open_time <= t < close_time (still trading at t; knowable at t).
Quote = last DAILY candle (period 1440) whose end_period_ts is in (t - 36 h, t]; need two-sided: yes_bid >= 0.01,
yes_ask <= 0.99, yes_ask > yes_bid. No spread cap (the spread is the real cost).
Favourite = side whose mid ((bid+ask)/2 for YES; 1 - that for NO) > 0.5. Bet if the favourite's ASK is in [0.60, 0.85]:
YES ask = yes_ask; NO ask = 1 - yes_bid. Buy 1 contract at that ask + taker fee = series fee_multiplier * 0.07 * p(1-p)
(exact, primary; 1-ct ceil-to-cent as sensitivity). Hold to settlement. Side fixed by the snapshot (later flips do not
matter). Every market in a multi-candidate event is a separate bet (NO on a 15c candidate = 85c favourite).
P&L = won - price - fee; ROI = sum P&L / sum cost.
Inference: 95% CI by bootstrap over EVENTS (event_ticker = race), 10,000 resamples. Sensitivity clusters: series, and
"election day" = UTC date of close_time (correlated election-night outcomes).
SUCCESS (T7, PRIMARY window, Politics+Elections): ROI event-CI lower > 0 AND >= 150 distinct events AND no single
election day (close_time UTC date) > 30% of total P&L (when total > 0) AND edgelab family bar (new family
kalshi_politics, 3 trials T7/T14/T3 registered before scoring -> bar p < 0.05/3 per trial).
KILL: T7 fails any part -> L22 dead as a taker strategy at this horizon. A T14/T3-only or category-only pass = lead.
If T7 has < 150 events: "underpowered", not a pass and not a kill of the calibration claim.
Calibration (descriptive, both windows): at T7, all eligible two-sided markets, logistic regression of result==yes on
logit(YES mid), mid clipped to [0.02, 0.98]; slope > 1 = underconfident (Le: 1.83). 95% CI by event bootstrap.
Also win rate vs mean price of the favourite bets. "Replicates" = slope CI lower > 1.
edgelab (registered 12:3xZ, before any price): pol_t7_fav_taker-9d327451 (primary), pol_t14_fav_taker-86c51c7a,
pol_t3_fav_taker-84fe66ea; family kalshi_politics (3 trials -> bar p < .0167). Dataset kalshi_politics_fav, dev split =
2026-07-01..11-15 (the primary window; one shot), forward 11-16+. Rule edgelab/rules/politics_fav.py (bets_frame shared
with the script so both paths select the same bets).

## Log (cont.)
- Rate: single-thread pull ran 200 series in 16 min (latency-bound, ~0.4 req/s). Tried 6 threads (all through
  kx_throttle.slot, so the cap held), lead asked not to crowd the shared budget (live bots) -> 2 threads, and the
  universe narrowed BEFORE any outcome on metadata only: Politics+Elections series with KX tickers (4,602 of 5,059;
  the 457 legacy non-KX tickers are the 2024-cycle naming). World is no longer fetched (was report-only) -- deviation
  noted: no World breakdown.
- Metadata (no prices/outcomes): 24,936 binary markets close >= 2025-01-01 (data/pol_favourites/meta.parquet; 17,290
  from /historical). Eligible (open at t; `prices` log): PRIMARY T7 2,356 market-rows / 465 events, T14 1,771 / 502,
  T3 2,564 / 516, C7 (close anchor) 11,249 / 2,327; PAPER T7 4,498 / 1,119. expected_expiration - close_time in PRIMARY:
  median 71 d, p75 337 d -- many election markets (KXHOUSERACE etc.) carry a far expected_expiration, so the ex-ante
  anchor drops most of election night; the close anchor keeps it (but is lookahead-prone for early closes).
- Candles: 24,074 requests at ~1.3 req/s under the shared budget = ~5 h, so staged by priority: PRIMARY T-anchored
  (3,443 req) first, then PAPER T, then C7 (STAGE env in `prices`; all cached in data/pol_favourites/candles.jsonl).

## RESULT -- PRIMARY window (settled 2026-07-01..11-15), scored 2026-11-15 ~15:15Z
Source: `scripts/pol_favourites.py score` -> results/pol_favourites/summary.json, bets.csv, score_primary.log.
Checks before outcomes: quote end <= t for all 7,863 priced rows (traps.check_no_lookahead non-strict: 26 rows have
the candle ending exactly at t; check_epoch_seconds ok); quote lag median 10 h, max 35.7 h.
Bug caught on scoring: 109 metadata markets settled `scalar` (partial value, e.g. KXHORMUZMAX 0.25/0.50, KXGOVCANOMD).
The first pass dropped them (script) or settled them at 0 (edgelab result_equals_side). Fixed in the script by fetching
settlement_value_dollars (data/pol_favourites/scalar_values.json): YES pays the value, NO pays 1 - value.
| trial | n / events / days | win / mean ask | ROI [95% event CI] | day-cluster CI | 1-ct fee |
|---|---|---|---|---|---|
| T7 (primary) | 337 / 154 / 47 | 75.5% / 74.6c | -0.56% [-6.28, +5.07] | [-7.10, +5.46] | -1.26% |
| T14 | 273 / 149 / 61 | 68.5% / 74.2c | -9.26% [-16.50, -2.37] | [-14.98, -2.99] | -9.95% |
| T3 | 506 / 236 / 89 | 74.9% / 75.1c | -1.93% [-6.53, +2.53] | [-6.76, +2.78] | -2.64% |
| C7 close anchor (PARTIAL: only rows whose candles the T stage fetched) | 399 / 187 / 56 | 74.3% / 74.5c | -1.98% [-7.57, +3.62] | | |
Total P&L is negative in every trial, so the 30%-per-day concentration rule does not apply.
edgelab dev (registered before pricing): pol_t7_fav_taker-9d327451 FAIL n 337 -3.00% [-9.99, +3.36] p .81 (bar .0167)
-> dead; pol_t14 FAIL -9.26% [-16.05, -1.97] p .99 -> dead; pol_t3 FAIL n 506 -3.74% [-8.99, +0.96] p .94 -> dead.
(edgelab's T7/T3 ROIs are lower than the script's because it settles the 7 / 10 scalar rows at 0. T14 has none and
matches exactly. The script is authoritative; both fail.)
VERDICT: T7 FAILS the prereg (the CI crosses 0; 154 events clears the 150 floor) -> L22 DEAD as a taker strategy at 3-14 d.
Politics T7 +2.06% [-4.06, +7.85] (122 events), Elections -13.2% [-29.8, +3.6] (30 events): both cross 0.
Calibration (T7, all two-sided P+E markets, at mid): slope 1.23 [1.08, 1.42] (1,194 markets / 307 events) -> mild
underconfidence REPLICATES (CI lower > 1) but is far below Le's 1.83. Close anchor C7 (partial): 1.12 [0.98, 1.29].
Where it lives (T7, favourite side by mid bin, inline check on priced.parquet): mid .60-.70 won 63.1% vs ask 68.9c;
.70-.80 won 81.8% vs ask 78.4c; .80-.90 won 90.5% vs ask 88.0c; .90-.95 won 95.1% vs ask 95.1c. Mean spread is 6-13c
below .80, so the ask sits ~3-6c above mid and eats the underconfidence. In the bet band the favourite won 75.5% at
a 74.6c ask: priced right at the ask. The .80-.90 mid bin looks positive POST HOC (untested, one window).

## RESULT -- PAPER-ERA window (settled 2025-01-01..2026-06-30; overlaps Le's sample), script only
Source: same command after the PAPER T candle stage (priced 22,361 of 42,911 rows; results/pol_favourites/score_all.log).
| trial | n / events / days | win / mean ask | ROI [95% event CI] | day-cluster CI |
|---|---|---|---|---|
| T7 | 547 / 249 / 141 | 68.4% / 73.9c | -9.09% [-18.09, -1.11] | [-18.38, -0.57] |
| T14 | 585 / 286 / 149 | 69.4% / 74.1c | -7.93% [-16.58, -0.16] | [-16.83, +0.18] |
| T3 | 691 / 350 / 193 | 72.2% / 74.3c | -4.56% [-9.31, -0.15] | [-9.36, -0.08] |
T7 Politics -13.8% [-27.4, -1.8] (116 events), Elections -4.2% [-13.6, +4.7] (133 events).
Calibration T7 (mid): slope 1.04 [0.93, 1.15] over 1,892 markets / 672 events -> does NOT replicate Le's 1.83 in his own
period with a market-level, fixed-horizon snapshot. Le's estimate is trade-weighted (4.9M trades), which weights
heavily traded, late, near-certain markets; a per-market snapshot at S-7d is what a trader can actually buy.
Favourites in the 60-85c band LOST money at the ask in the paper era too (win 68.4% vs 73.9c paid).

## Not done (sensitivity, deprioritised under the shared rate budget)
C7 close-anchored rows were only priced where the T stages overlapped (PRIMARY C7 399 bets). A full C7 pull is
~15k more requests (~3 h). Given T7 fails in both windows and the band wins at or below its ask, it would not change the verdict.
No critic subagent was run (negative result; the lead can run /finding if it will be quoted).

## Proposal (no action taken)
L22 dead: move it to FUND_BOOK 3d ("Kalshi politics favourites 60-85c at 3-14 d, taker: PRIMARY T7 -0.6% [-6.3,+5.1],
paper era -9.1% [-18.1,-1.1]; slope 1.23 / 1.04"). Do not redo as a taker. The only untested remnant is maker-side
(rest a bid at mid on .70-.90 favourites), which pm_timing's adverse-selection result (unfilled favourites all won)
argues against.

## Learnings
### Kalshi politics favourites at 60-85c earn nothing at the ask, 3-14 days out
PRIMARY (settled 2026-07..11-15) T7 -0.56% [-6.28, +5.07] over 337 bets / 154 events. The favourite won 75.5% at a
74.6c ask (results/pol_favourites/summary.json). Le's underconfidence finding does not survive as a trade.
### The paper-era replication loses money too
Settled 2025-01..2026-06, T7 -9.09% [-18.09, -1.11] over 249 events; calibration slope 1.04 [0.93, 1.15] vs Le's 1.83.
A per-market fixed-horizon snapshot is what a trader buys; trade-weighted calibration is not that.
### Underconfidence at mid is eaten by the half-spread
PRIMARY T7 slope 1.23 [1.08, 1.42] at mid, but politics spreads run 6-13c below 80c, so the ask sits 3-6c above mid,
which is about the size of the bias. Always price calibration claims at the ask before calling them an edge.
### Kalshi binaries can settle `scalar`
109 of 24,936 politics markets settled at a partial value (e.g. KXHORMUZMAX 0.25/0.50). result_equals_side scores them
as losses, so fetch settlement_value_dollars (data/pol_favourites/scalar_values.json) before scoring.
### Kalshi API filters that are silently ignored
/events ignores min_close_ts. /historical/markets ignores min_close_ts. /events for series moved to historical returns
no nested markets. Page per series: /markets?status=settled&min_close_ts plus /historical/markets.
### expected_expiration_time is far from close for most politics markets
In PRIMARY the gap was a median 71 d and p75 337 d (open-ended "next PM" markets run to 2045), so an ex-ante S-7d anchor keeps only ~20% of markets.
Election-night markets mostly fall outside it.
