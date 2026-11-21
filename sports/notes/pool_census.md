# pool_census -- where is the loser-paid money on Kalshi? (agent pool_census, started 2026-11-21T05:56Z)

Brief (lead): extend winners_transfer P2 TAILPOOL (results/winners_transfer/tailpool.py) from 35 series to EVERY Kalshi
series with markets settled in the last 90 days; add the mirror pool (losers who paid >= 85c); rank by pool per 30 days;
top 40 with pool, events/30d, median spread, fee type, category, TESTED/UNTESTED (cited); top-5 contributing events per
top pool. Then DGP + fastest public page + "can a free nowcast beat the book" for the top 3 UNTESTED recurring pools.
Measurement / map only: no return tests, no edgelab trial, no orders, no git.

## Pre-registration (written before any pool is computed)
Definitions are P2's, unchanged (notes/winners_transfer.md "P2 TAILPOOL"):
- universe: Kalshi markets with status settled and close_time in the 90 days before the run; result in {yes, no};
  volume > 0. Scalar / void markets are excluded (same as tailpool.py).
- decision window per market: final 24 h before close if (close - open) <= 3 d, else final 7 d.
- per trade (GET /markets/trades): loser price p_l = yes_price if result = no else no_price.
- TAIL pool = sum(count x p_l) over trades with p_l <= 0.15 (cheap-side buyers who lost).
- MIRROR pool = sum(count x p_l) over trades with p_l >= 0.85 (expensive-side buyers who lost). Identity check: this
  equals tailpool.py's TAIL RISK (sum count x (1 - p_w), p_w <= .15) because yes_price + no_price = 1 on every trade;
  I assert that on the tape.
- per 30 days = 90-day sum x 30/90 (series whose first settled market is younger than 90 d get scaled by their own span,
  reported with a flag).
Search strategy (branch and bound, so the top 40 is exact without pulling every trade on Kalshi):
- Stage 1: enumerate all settled markets (close in last 90 d) via GET /markets (1000 per page); per series: markets,
  events, contracts traded (volume_fp), and an upper bound on the pools: TAIL <= 0.15 x contracts, MIRROR <= contracts.
- Stage 2: compute exact pools (trade tape) series by series in descending order of the bound; stop when the next
  series' TAIL bound is below the current 40th-largest measured TAIL pool (and likewise for MIRROR within budget).
  If the trade tape for a series is too large for the shared rate limit (kx_throttle, ~3 req/s shared with the live bots),
  fall back to hourly candlesticks (volume x loser price per hour) and FLAG the row as approximate.
Pre-set reading rules (set now, before seeing outcomes):
- "recurring" = top-1 event share of the series' TAIL pool < 50% AND events/30d >= 4 (rubric R4 floor).
  "one-event pool" = top-1 event share >= 50% (reported, not ranked as a recurring opportunity).
- STRATEGY_v2 amendment 1 threshold: < $500/30d TAIL -> forward-log only.
- TESTED = series (or same-mechanism family) appears in FUND_BOOK 3a-3d, LEARNINGS_2026-11, EDGES_AND_AGENTS or a notes
  file with a test; cite the line. UNTESTED otherwise.
- Median spread = median (yes_ask - yes_bid) over currently open markets of the series with both sides quoted
  (live snapshot), or "n/a" if no open markets.

## Plan
1. Stage 1 enumeration -> results/pool_census/settled_markets.parquet, series_bounds.csv
2. Stage 2 exact pools -> results/pool_census/pools.json (+ per-event contributions)
3. Spread + fee + category from series_watch snapshot (~/.claude/quant-sports/state/series_watch/snap_2026-11-20.json.gz)
   + live GET /markets for open events.
4. TESTED/UNTESTED by grep over notes/.
5. Top-3 UNTESTED recurring: rules text, source, timing, fastest public page, nowcast plausibility.

## Log
- 05:56Z started row in inbox; notes created.
- 05:58Z API facts (read-only probes): GET /historical/cutoff -> market_settled_ts and trades_created_ts 2026-09-21T00:00Z;
  live /markets still returns KXHIGHNY markets closing from 2026-09-13 (42 rows for 08-23..09-20), i.e. live retention
  is ~69 days, not 90. CONSEQUENCE for P2: its 90-day rows (KXTSAW, AI share, Netflix, CB, rain-month ...) got ~10 weekly
  events instead of ~13 and divided by 90 -> those per-30d pools are UNDERSTATED by roughly 90/69 (~1.3x); its 30-day rows
  (weather, KXAAAGASD, KXTSAD) are unaffected. (tailpool_result.json: KXTSAW 90 d -> 10 events.)
  => DESIGN CHANGE (before computing any pool): window = markets closing in the last 60 days (all after the cutoff),
  per-30d = sum x 0.5. Stage 1 still enumerates 90 d to locate the retention edge.
- Batch candles GET /markets/candlesticks: max 100 markets and max 10,000 requested candles per call (hourly: 24 h window
  -> 100 markets/call; 7 d window -> 59 markets/call; 1-min impossible at scale). Exact trade tapes for the sports
  series are far beyond the shared ~3 req/s budget, so Stage 2 = HOURLY-CANDLE approximation for every series:
  per hour h in the decision window, loser price L_h = mean_dollars (if result no) or 1 - mean_dollars (if yes);
  TAIL += volume_h x L_h if L_h <= .15; MIRROR += volume_h x L_h if L_h >= .85.
  VALIDATION (pre-set): recompute P2's 35 series (exact tape) with the candle method on the SAME window definition;
  acceptable if candle/exact TAIL ratio lies in [0.7, 1.3] for >= 80% of series with exact TAIL >= $1k/30d. If not, try
  the range-interpolated variant (share of the hour's [low, high] below .15) and use it only if it meets the same bar;
  otherwise rank with the candle numbers but label them "order-of-magnitude".
  (Not exact for the same 30/90-day windows P2 used: I use P2's own `days` per series and P2's run time is not stored,
  so windows differ by ~1 day; compare as ratios.)
- 06:00-06:10Z THROUGHPUT forced a second design change (before any pool beyond P2's validation series was computed):
  kx_throttle is shared with gas_daily_v2, source_shift and the loggers; I got ~0.5-1 call/s. A 60-day global
  enumeration is ~6,000 pages (~100k settled non-MVE markets per day; only ~29% traded). Final design:
  (a) WEEK SAMPLE: global enumeration of markets closing 2026-11-14T00:00Z..11-21T00:00Z (Sat-Fri, 7 full UTC days;
      settled_90d.jsonl, 769 pages, 1,565 series traded, 221,540 traded markets, 8.58e9 contracts). Pools x 30/7.
      Branch-and-bound by 0.15 x weekly contracts (run_week.py, K = 40); heavy series event-sampled (top-10 events by
      volume + seeded random rest, Horvitz-Thompson weight), flagged `sampled`.
  (b) NON-SPORTS PASS 2: same week, all non-Sports categories down to bound >= $20k/30d (so nowcastable series below the
      sports-dominated top 40 are still mapped).
  (c) LOW-FREQUENCY PASS: series with no traded settlement in the week (monthly, one-off, seasonal; NB the week is an
      international break, so EPL/LaLiga/Serie A are absent from it) -> per-series 60-day enumeration in descending
      lifetime volume (series_all.json from GET /series?include_volume=true), candles when 0.15 x V60 / 2 >= bound.
  Resolution: markets living <= 3 h (crypto 15-min / hourly) get 1-min candles over their life; others hourly.
  KXMVE* combos excluded: 1,000 MVE markets closed in 3 s with 0 traded in a probe (RFQ parlays, not a nowcast target).
- Validation so far (validate_p2 partial, hourly vs exact tape, P2 windows): TAIL ratio KXHIGHNY 1.09, CHI 1.08, MIA
  1.06, AUS 1.17, DEN 0.95, KXTSAW 0.84; FULL ratio 1.000 on CHI/AUS (=> mean_dollars is the volume-weighted price).
  MIRROR is noisier with plain means (CHI 0.60, DEN 0.56); the rng variant is closer (CHI 0.87, DEN 0.76).
- 06:25Z VALIDATION P2 (results/pool_census/validate_p2.json; hourly candles vs P2 exact tape, same window rule):
  TAIL ratio candle/exact within [0.7, 1.3] for 19/19 complete series with exact TAIL >= $1k (range 0.79 KXEOWEEK ..
  1.17 KXTSAD; weather 0.92-1.17) -> pre-set bar MET, mean variant used for TAIL. KXNFLFG incomplete (validate_p2
  died on a 429 after 374/1192 markets) -> no in-game verdict from P2. MIRROR with plain means is biased low
  (LAX 0.40, PHIL 0.41, TSAD 0.45); the rng variant is 0.61-1.13 -> MIRROR reported with the rng variant, labelled
  approximate (decided after seeing this validation; secondary metric).
  1-min candles vs exact tape on 2 KXBTC15M markets (validate_extra.log): TAIL 28,964 vs 35,362 (0.82), 71,005 vs
  80,664 (0.88); FULL equal to <0.01% -> short-lived crypto pools are understated ~15%.
  429s seen on the shared IP (validate_p2 crash) -> I now run at most 2 processes.
- 06:35Z LEAD ADDITION (from gas_daily_v2: KXAAAGASD's $488k pool is collected by the quoting MM; far rungs ~8c wide,
  177/579 low-prob rungs had a YES bid): add per row (i) median far-rung spread, (ii) share of far rungs two-sided at
  decision time, (iii) TAKER-REACHABLE pool; rank on (iii), keep raw pool as a column. Implementation (set before
  computing any of it):
  * decision time = first candle of the P2 window; far rung = cheap side buyable at <= .15 (yes_ask <= .15 or
    yes_bid >= .85); two-sided = yes_bid > 0 and yes_ask < 1; far spread = ask - bid on two-sided far rungs.
    (candle_pool now stores a0/b0/far0/two_sided0; run_week restarted from scratch with K = 80, partial first run
    archived in results/pool_census/old_run1/.)
  * taker-reachable = a losing buyer whose order was RESTING (maker): on the tape, taker_side == winning side
    (the taker sold the loser side into a resting bid). phi = reachable $ / pool $ on an exact-tape sample of each
    series' top-tail markets (tape_phi.py, <= 5 pages per market, <= 14 per series, newest-first; truncation flagged);
    reach_tail30 = phi_tail x candle TAIL; reach_mirror30 = phi_mirror x MIRROR(rng).
- 06:50Z IN-GAME FINDING + FIX: hourly candles hid the whole in-game tail: KXNBAGAME-26NOV13NYKIND-NYK exact tape TAIL
  $150,564 vs hourly $0 (the loser goes .4 -> .01 inside one hour; the hourly mean stays > .15). New resolution:
  Sports series with life > 3 h = hourly up to the hour boundary <= close - 4 h, then 1-min to close; markets living
  <= 3 h = 1-min; everything else hourly. Validation vs exact tape (validate_hybrid.json; n markets): TAIL ratio
  KXBTC15M 1.05 (3), KXETH15M 0.77 (1), KXBTCD 0.94 (24), KXNBAGAME 1.10 (2), KXNCAAMBGAME 0.97 (4),
  KXATPCHALLENGERMATCH 0.88 (4), KXNCAAMBTOTAL 1.00 (18), KXNBAPTS 1.08 (39); FULL 1.000 everywhere. MIRROR at 1-min
  with plain means 0.89-1.08 (NBAGAME n.d., $2.8k exact) -> MIRROR = plain mean on 1-min markets, rng on hourly ones.
  Second run archived (old_run2/); run_week restarted (K = 80) with the hybrid resolution + far-rung window stats
  (per candle close: far = yes_ask <= .15 or yes_bid >= .85; two-sided share; far-spread histogram).
- 06:40Z budget changes (documented, not outcome-driven): pass 1 series #1-28 CAP 30 calls / TOPN 10; #29-34 CAP 8 /
  TOPN 3; #35+ CAP 8 / TOPN 3 and a bound floor MINB $5M (series with week bound < $5M are unmeasured in pass 1; across
  measured series the tail/bound ratio was 0.01-0.30, so an unmeasured series' tail is very likely < $1.5M/30d).
  Series flagged lower_bound when the call cap left no random rest sample (e.g. KXNBAPTS). 429s hit tape_phi once
  (KXDOGE15M batch; retried by the loop).
