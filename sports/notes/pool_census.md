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

## DGP write-ups: top UNTESTED recurring pools (ranked on taker-reachable TAIL, week sample x 30/7)
Ranks 1-3 of the UNTESTED recurring pools are all tennis (one data-generating process), so write-up A covers them; B and
C are the next distinct untested families. Numbers: results/pool_census/pool_table.csv, phi_week.json, top_events.json;
rules text: results/pool_census/rules_sample.txt (read-only GET /markets).

### A. Tennis match winners: KXATPCHALLENGERMATCH (#2 overall, reach $16.7M/30d = phi .506 x TAIL $33.1M), KXITFMATCH
(#9, $6.1M = .341 x $17.9M), KXITFWMATCH (#12, $5.2M = .321 x $16.2M); siblings KXATPMATCH $4.8M (ATP Finals week,
top1 16%), KXWTACHALLENGERMATCH $1.3M, KXATPSETWINNER $0.5M, KXWTAMATCH $0.5M.
- Recurring: 1,436 / 2,837 / 2,293 events per 30 d; top-1 event 3.8% / 3.8% / 3.0% of TAIL (top-5: top_events.json).
  Off-season risk: Challenger/ITF calendars thin out in December.
- Rules: "If <player> wins the <A vs B> professional tennis match in <tournament, round> after a ball has been played ->
  Yes"; no ball played -> fair price (ATP Ch) / $0.50 (ITF); ITF: retirement after the start -> retiring player No;
  postponed -> stays open <= 2 weeks; "closes and expires after a winner is declared", settlement timer 60 s. Named
  sources: ATP (atptour.com) for Challengers; ESPN / ITF / Fox / Flashscore for ITF.
- Timing: markets open ~13-15 h before close (median life 14.3 h, ATP Ch, 4-day sample), so P2's decision window is the
  whole life and the pool accrues IN-GAME as the trailing player's price falls through 15c.
- Who sees the outcome first: the chair umpire's scoring device. ATP/Challenger official data = Tennis Data Innovations,
  distributed to sportsbooks as a fast feed (IMG Arena) plus Sportradar's umpire-chair secondary feed (2022 deal);
  ITF World Tennis Tour = Infront Bettor 2025-29 (handheld umpire devices, data for betting only via Infront).
  Fastest FREE pages: ATP Challenger TV streams (free), ATP live scores, ITF Live Scores app, Flashscore / tennis.com
  (Sportradar widgets) -- all downstream of the paid fast feed; one vendor quotes 2-10 s delay on Challenger/ITF feeds.
- Book quality: far rungs two-sided in 85-87% of far candles, median far spread 1c (pool_table.csv) -- a tightly made
  book. Blind cheap-side seller net (TAIL - MIRROR) was NEGATIVE: ATP Ch -$5.0M, ITF -$0.75M, ITFW -$1.55M per 30 d
  (comebacks paid out more than longshot losers paid in).
- Verdict: a free nowcast does NOT plausibly beat this book (rubric R6 = 0: feed subscribers see points first; the pool
  is the in-play reversal premium, not stale tails). The only testable angle is SHARPLINE pre-match (Kalshi vs Pinnacle
  tennis, ~6,500 events/30 d across the three series = power in days), R2 = 1 at best.

### B. UFC fight winners: KXUFCFIGHT (#21, reach $2.0M = .523 x TAIL $3.86M; 51 fights/30 d; top-1 fight 46.5% =
borderline recurring; MIRROR $13.6M in the week = upsets). KXUFCMOV (MOV) is TESTED (U1 dead, EA l.426).
- Rules: fighter wins the fight "originally scheduled for <date>"; draw / no contest -> both $0.50; revisions after
  expiration ignored (DQ/overturn after close does not count); starts within 48 h -> stays open, else fair price.
- Fastest page: the live broadcast; in-fight the book reprices round by round off it (live-watchers = R6 0).
- One genuine nowcast gap: after the final horn of a fight that goes to the judges, the result is fixed but unannounced
  for ~1-3 min. Free inputs then: round-by-round official stats (UFC.com live stats / UFCStats), media scorecards
  (MMADecisions), the broadcast's own stat graphics. Plausibility LOW-MODERATE: the crowd watches the same fight, the
  window is minutes, and only fights that reach the judges qualify; worth a descriptive look only (price path between
  final horn and announcement on decision fights vs a strikes-landed judge model).

### C. eSoccer / eBasketball: KXESOCCERGAME (#60, reach $0.75M = .695 x TAIL $1.07M; 13,637 events/30 d -- the most
events of any Kalshi series; top-1 24%), KXEBASKETBALLGAME (#76, $0.35M = .542 x $0.65M; 2,803 events/30 d).
Estimates are noisy (event-sampled: 18 of 3,182 eSoccer events measured, Horvitz-Thompson x212).
- Rules: result after "90 minutes plus stoppage" of in-game time (no ET/penalties); postponed > 2 weeks -> $0.33.
  Source: DRAFTED.GG (B2B esports content: Valhalla Cup (men) / Valkyrie Cup (women), eFootball 2x4-minute halves,
  24/7 streams, ~6,000 + 3,000 matches/month; results page drafted.gg/en/valhalla-cup/results; public YouTube channel
  @valhallacup1; team icons served from oddin.gg = an esports odds/data B2B vendor).
- Fastest page: the B2B stream/data feed to betting operators; the public YouTube stream lags it by seconds.
- Book quality: far rungs two-sided in only 58% of far candles, median far spread 2c (thinner than tennis/UFC).
  Blind cheap-side seller net +$1.06M (eSoccer) and +$0.50M (eBasketball) per 30 d -- the only untested family here
  where longshot buyers overpaid in the week.
- Verdict: in-play nowcast NO (paid feed first). Pre-match there is a free, huge, fast-growing results history
  (thousands of matches per month between a small player pool) -> a player-rating model is MODEL-class (0/7 sealed
  elsewhere) but this book looks provider-made and thin; cheapest probe = descriptive calibration of pre-match mids vs
  outcomes on 1-2 weeks of settled events (R4 power is enormous).
Other untested recurring pools (no write-up): cricket ODI/T20/WT20 $0.57-0.72M each (Cricbuzz/Cricinfo text lags TV;
NR rule priced, rules_quirks l.128); FX 15-min on Pyth (KXUSDCAD/GBPUSD/AUDUSD15M $0.36-0.52M TAIL, phi .48-.67) and
KXINX15M ($0.34M, phi .56); Treasury-yield 15-min (KX2Y/5Y/10Y/30YRRATE15M $0.09-0.16M) -- all settle on a public
real-time price (Pyth / S&P), so they are latency games against market makers reading the same feed, not nowcasts.

## RESULT (2026-11-21 ~07:00Z) -- where the loser-paid money is on Kalshi
Coverage: 391 series measured (325 in the week sample 2026-11-14..11-20 x 30/7; 66 low-frequency/one-off series over
the last 60 d x 0.5). Calls ~2,700 candle/tape GETs through kx_throttle. Sum of measured TAIL = $659.1M per 30 d.
Ranking key = TAKER-REACHABLE TAIL = phi_tail x TAIL (phi = share of the losers' <= 15c $ that sat as RESTING orders,
from exact-tape samples: tape_phi.py; '~' = no tape sample, median phi used). phi is an UPPER bound on what an outside
taker gets: it counts every resting loser order, including ones market makers/bots hit first.
TAIL - MIRROR = what a blind seller of every cheap side would have netted at trade prices (P2: MIRROR == TAIL RISK).

Family totals (pool_table.csv; reach uses median phi where missing): other sports TAIL $256.8M (reach $100.6M, 69 series);
crypto 15-min $242.1M ($94.2M, 10); tennis $78.6M ($35.1M, 7); commodity/FX/rate 15-min $28.4M ($16.5M, 13); other crypto
$34.5M ($12.8M, 17); elections (one-offs) $10.5M; entertainment $2.5M; WEATHER (43 series) $1.49M ($0.56M);
economics $1.07M; financials $0.96M; mentions $0.84M; politics $0.61M.
Position of our books: KXAAAGASD rank 95 by TAIL ($349k/30d, phi .452, far rungs two-sided only 15% of far candles),
KXHIGHNY 125 ($137k, phi .40), KXHIGHLAX 135 ($103k), KXHIGHMIA 155 ($61k); KXGOOGSHARE 192 ($25.8k); KXTSAD 276 ($5.7k).
Cross-checks vs P2 exact tape: KXHIGHDEN week $27.3k vs P2 30-d $26.3k; KXGOOGSHARE $25.8k vs P2 $18.7k x 90/69 = ~$24k.

median measured phi_tail = 0.484 (used, flagged '~', where a series has no tape sample)

| # | series | cat | reach TAIL $/30d | phi | raw TAIL | MIRROR | ev/30d | top1 | recurring | far 2-sided | far spr | fee | TESTED? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | KXBTC15M | Crypto | 73.11M | 0.366 | 199.74M | 123.83M | 2846 | 0.007 | Y | 0.75 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 2 | KXATPCHALLENGERMATCH | Sports | 16.73M | 0.506 | 33.06M | 38.06M | 1436 | 0.038 | Y | 0.85 | 0.01 | q | UNTESTED: only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74) |
| 3 | KXNBAGAME | Sports | 15.09M | 0.421 | 35.85M | 43.93M | 223 | 0.067 | Y | 0.76 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 4 | KXNCAAFGAME | Sports | 12.62M | 0.406 | 31.09M | 34.13M | 553 | 0.093 | Y | 0.74 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 5 | KXNCAAMBGAME | Sports | 12.61M | 0.476 | 26.50M | 13.02M | 1406 | 0.153 | Y | 0.69 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 6 | KXBTCD | Crypto | 12.13M | 0.369 | 32.87M | 43.06M | 703 | 0.116 | Y | 0.42 | 0.01 | q | TESTED: Q012 hourly crypto DEAD (EA l.317); OT1 daily lead/weakened (LRN l.268-276); range ext dead (LRN l.235) |
| 7 | KXGOLD15M | Commodit | 9.54M | 0.573 | 16.65M | 11.10M | 1916 | 0.013 | Y | 0.82 | 0.0 | q | SIBLING-DEAD: Q016 commodities DEAD (EDGES_AND_AGENTS l.319); STRATEGY_v2 l.103 |
| 8 | KXETH15M | Crypto | 7.17M | 0.546 | 13.13M | 9.56M | 2846 | 0.014 | Y | 0.75 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 9 | KXITFMATCH | Sports | 6.10M | 0.341 | 17.89M | 18.63M | 2837 | 0.038 | Y | 0.86 | 0.01 | q | UNTESTED: only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74) |
| 10 | KXUEFANLGAME | Sports | 6.05M | 0.511 | 11.85M | 7.12M | 146 | 0.176 | Y (intl window only (x30/7 overstates)) | 0.8 | 0.01 | q | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 11 | KXNFLGAME | Sports | 5.45M | 0.163 | 33.42M | 7.14M | 60 | 0.191 | Y | 0.67 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 12 | KXITFWMATCH | Sports | 5.18M | 0.321 | 16.15M | 17.70M | 2293 | 0.03 | Y | 0.87 | 0.01 | q | UNTESTED: only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74) |
| 13 | KXATPMATCH | Sports | 4.77M | 0.68 | 7.02M | 0k | 56 | 0.163 | Y | 0.91 | 0.01 | q+maker | UNTESTED: only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74) |
| 14 | KXXRP15M | Crypto | 4.65M | 0.502 | 9.26M | 4.68M | 2846 | 0.02 | Y | 0.76 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 15 | KXMLBGAME | Sports | 3.89M~ 60d | ~ | 8.03M | 6.21M | 64 | 0.046 | Y (season ended) | 0.76 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 16 | KXSOL15M | Crypto | 3.24M | 0.514 | 6.30M | 4.22M | 2846 | 0.029 | Y | 0.76 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 17 | KXNCAAWBGAME | Sports | 3.03M | 0.388 | 7.81M | 12.23M | 1337 | 0.325 | Y | 0.74 | 0.02 | q | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 18 | KXWTI15M | Commodit | 2.96M | 0.663 | 4.47M | 2.27M | 2126 | 0.014 | Y | 0.8 | 0.0 | q | SIBLING-DEAD: Q016 commodities DEAD (EDGES_AND_AGENTS l.319); STRATEGY_v2 l.103 |
| 19 | KXCONCACAFNLGAME | Sports | 2.53M | 0.496 | 5.10M | 5.48M | 99 | 0.311 | Y (intl window only (x30/7 overstates)) | 0.84 | 0.01 | q | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 20 | KXINTLFRIENDLYGAME | Sports | 2.43M | 0.473 | 5.13M | 3.38M | 111 | 0.183 | Y (intl window only (x30/7 overstates)) | 0.78 | 0.01 | q | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 21 | KXUFCFIGHT | Sports | 2.02M | 0.523 | 3.86M | 13.58M | 51 | 0.465 | Y | 0.89 | 0.01 | q | UNTESTED: winner itself untested; U1 MOV used it as input -> DEAD (EA l.426) |
| 22 | KXNBATOTAL | Sports | 1.82M | 0.588 | 3.10M | 1.90M | 223 | 0.137 | Y | 0.78 | 0.01 | q+maker | TESTED: Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334) |
| 23 | KXNCAAFSPREAD | Sports | 1.67M | 0.54 | 3.09M | 1.72M | 553 | 0.189 | Y | 0.7 | 0.01 | q+maker | TESTED: Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334) |
| 24 | KXUCLGAME | Sports | 1.61M~ 60d | ~ | 3.33M | 3.83M | 27 | 0.107 | Y | 0.82 | 0.01 | q+maker | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 25 | KXZEC15M | Crypto | 1.60M | 0.568 | 2.82M | 957k | 2846 | 0.125 | Y | 0.76 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 26 | KXNHLGAME | Sports | 1.60M | 0.522 | 3.06M | 4.58M | 210 | 0.087 | Y | 0.75 | 0.01 | q+maker | TESTED: in-game N0/M4/CBB-IG (FB l.231-238), longshots G2/G3 (FB l.331), xvenue M2 (FB l.346), openers (LRN l.155/334) |
| 27 | KXNBASPREAD | Sports | 1.58M | 0.428 | 3.69M | 2.27M | 223 | 0.178 | Y | 0.7 | 0.01 | q+maker | TESTED: Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334) |
| 28 | KXSILVER15M | Commodit | 1.38M | 0.517 | 2.67M | 2.48M | 1916 | 0.046 | Y | 0.81 | 0.0 | q | SIBLING-DEAD: Q016 commodities DEAD (EDGES_AND_AGENTS l.319); STRATEGY_v2 l.103 |
| 29 | KXWTACHALLENGERMATCH | Sports | 1.35M | 0.509 | 2.65M | 248k | 137 | 0.131 | Y | 0.81 | 0.01 | q | UNTESTED: only first-listing staleness (kalshi_first_listing l.66-70) + rules flag (rules_quirks l.74) |
| 30 | KXNCAAMBSPREAD | Sports | 1.30M | 0.422 | 3.08M | 1.02M | 1371 | 0.26 | Y | 0.64 | 0.02 | q+maker | TESTED: Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334) |
| 31 | KXDOGE15M | Crypto | 1.25M | 0.347 | 3.60M | 2.52M | 2846 | 0.024 | Y | 0.77 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 32 | KXHYPE15M | Crypto | 1.23M | 0.436 | 2.82M | 2.10M | 2846 | 0.158 | Y | 0.73 | 0.01 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 33 | KXNEAR15M | Crypto | 1.20M | 0.489 | 2.45M | 697k | 2846 | 0.097 | Y | 0.76 | 0.0 | q | SIBLING-DEAD: STRATEGY_v2 l.103 KX*15M excluded (Q012/Q016/OT1-ext); latency-arb version untested, low prior (forum_strategy_scout l.43/80) |
| 34 | KXNCAAMBTOTAL | Sports | 1.14M | 0.498 | 2.29M | 1.24M | 1367 | 0.21 | Y | 0.62 | 0.02 | q+maker | TESTED: Q004 / NBA live totals / early NFL totals (FB l.165, l.249, l.338); openers (LRN l.334) |
| 35 | KXUFCMOV | Sports | 1.13M | 0.285 | 3.97M | 2.53M | 56 | 0.418 | Y | 0.91 | 0.01 | q | TESTED: U1 UFC MOV DEAD (EA l.426) |
| 36 | KXUEFANLTOTAL | Sports | 1.12M | 0.657 | 1.71M | 1.19M | 146 | 0.169 | Y (intl window only (x30/7 overstates)) | 0.86 | 0.01 | q | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 37 | KXEPLGAME | Sports | 1.12M~ 60d | ~ | 2.31M | 1.54M | 25 | 0.084 | Y | 0.74 | 0.01 | q+maker | TESTED: soccer 1X2 S1-S3 (FB l.333), SOC-IG (FB l.239), vs Pinnacle (LRN l.149) |
| 38 | KXMLBHR | Sports | 1.05M~ 60d | ~ | 2.18M | 1.30M | 64 | 0.12 | Y (season ended) | 0.6 | 0.01 | q | TESTED: MLB ML/totals/bullpen rows (FB 3d 'Sports models'); season ended (60d window) |
| 39 | KXNFLTD | Sports | 1.04M LB | 0.143 | 7.24M | 6.30M | 60 | 0.251 | Y | 0.8 | 0.01 | q | TESTED: props: Kalshi vs DK gap 0c (FB l.295); NFLUP DEAD (EA l.431-435); prop ladders paper |
| 40 | CONTROLH | Election | 1.02M~ 60d | ~ | 2.10M | 0k | 0 | 1.0 | N | 0.99 | 0.0 | q | ONE-OFF: election one-off (election_needle_2026.md) |

Top-5 TAIL events (event, $/30d-scaled contribution, share of the series' measured TAIL):
- KXBTC15M: KXBTC15M-26NOV191845 1.46M (0.007); KXBTC15M-26NOV141130 1.26M (0.006); KXBTC15M-26NOV191930 1.20M (0.006); KXBTC15M-26NOV201800 1.18M (0.006); KXBTC15M-26NOV201545 1.06M (0.005)
- KXATPCHALLENGERMATCH: KXATPCHALLENGERMATCH-26NOV14BASALT 797k (0.038); KXATPCHALLENGERMATCH-26NOV15MATSUN 469k (0.022); KXATPCHALLENGERMATCH-26NOV14MARVIR 454k (0.022); KXATPCHALLENGERMATCH-26NOV14ZHUKUM 362k (0.017); KXATPCHALLENGERMATCH-26NOV16HOLSUR 317k (0.015)
- KXNBAGAME: KXNBAGAME-26NOV19NYKMIN 2.40M (0.067); KXNBAGAME-26NOV19LACSAS 2.40M (0.067); KXNBAGAME-26NOV13GSWSAS 1.89M (0.053); KXNBAGAME-26NOV17DETSAS 1.70M (0.048); KXNBAGAME-26NOV13UTAHOU 1.67M (0.047)
- KXNCAAFGAME: KXNCAAFGAME-26NOV13ILLUCLA 2.88M (0.093); KXNCAAFGAME-26NOV13HOUCOLO 1.64M (0.053); KXNCAAFGAME-26NOV14USCIND 1.58M (0.051); KXNCAAFGAME-26NOV17EMUWMU 1.57M (0.051); KXNCAAFGAME-26NOV14UTAHARIZ 1.13M (0.036)
- KXNCAAMBGAME: KXNCAAMBGAME-26NOV19HOUKU 3.25M (0.153); KXNCAAMBGAME-26NOV19FLAWVU 1.09M (0.051); KXNCAAMBGAME-26NOV17MSUTENN 980k (0.046); KXNCAAMBGAME-26NOV18CSFUCLA 977k (0.046); KXNCAAMBGAME-26NOV17FLAND 944k (0.044)
- KXBTCD: KXBTCD-26NOV2017 1.08M (0.116); KXBTCD-26NOV1517 727k (0.078); KXBTCD-26NOV1417 526k (0.056); KXBTCD-26NOV1917 466k (0.05); KXBTCD-26NOV1717 462k (0.05)
- KXGOLD15M: KXGOLD15M-26NOV192230 221k (0.013); KXGOLD15M-26NOV191615 213k (0.013); KXGOLD15M-26NOV190115 165k (0.01); KXGOLD15M-26NOV160030 160k (0.01); KXGOLD15M-26NOV161130 131k (0.008)
- KXETH15M: KXETH15M-26NOV142315 182k (0.014); KXETH15M-26NOV151815 118k (0.009); KXETH15M-26NOV201730 113k (0.009); KXETH15M-26NOV150215 102k (0.008); KXETH15M-26NOV151015 92k (0.007)
- KXITFMATCH: KXITFMATCH-26NOV17SURARC 220k (0.038); KXITFMATCH-26NOV20ROCROD 214k (0.037); KXITFMATCH-26NOV14MARDEA 205k (0.035); KXITFMATCH-26NOV16YAMOKI 134k (0.023); KXITFMATCH-26NOV13JADZIN 116k (0.02)
- KXUEFANLGAME: KXUEFANLGAME-26NOV15ESPENG 2.09M (0.176); KXUEFANLGAME-26NOV15FRATUR 1.10M (0.093); KXUEFANLGAME-26NOV14NORWAL 901k (0.076); KXUEFANLGAME-26NOV16GRESRB 778k (0.066); KXUEFANLGAME-26NOV17WALPOR 705k (0.06)
- KXNFLGAME: KXNFLGAME-26NOV15PITCIN 6.37M (0.191); KXNFLGAME-26NOV15SFDAL 6.06M (0.181); KXNFLGAME-26NOV16LACBAL 5.11M (0.153); KXNFLGAME-26NOV19INDHOU 3.85M (0.115); KXNFLGAME-26NOV15MINGB 3.31M (0.099)
- KXITFWMATCH: KXITFWMATCH-26NOV14OKTSAK 197k (0.03); KXITFWMATCH-26NOV19METFER 184k (0.028); KXITFWMATCH-26NOV17PONMEY 179k (0.028); KXITFWMATCH-26NOV17GADYUA 169k (0.026); KXITFWMATCH-26NOV19KOIGAD 150k (0.023)
- KXATPMATCH: KXATPMATCH-26NOV17ZVESHE 1.14M (0.163); KXATPMATCH-26NOV17ZVEFIL 868k (0.124); KXATPMATCH-26NOV17SHEJOD 784k (0.112); KXATPMATCH-26NOV17ALCCOB 559k (0.08); KXATPMATCH-26NOV16MEDCOB 557k (0.079)
- KXXRP15M: KXXRP15M-26NOV181100 109k (0.02); KXXRP15M-26NOV141830 97k (0.018); KXXRP15M-26NOV190245 77k (0.014); KXXRP15M-26NOV151245 75k (0.014); KXXRP15M-26NOV132230 73k (0.014)
- KXMLBGAME: KXMLBGAME-26SEP291700CWSHOU 277k (0.046); KXMLBGAME-26OCT142000MILLAD 268k (0.044); KXMLBGAME-26OCT282000MILTB 222k (0.037); KXMLBGAME-26OCT072200MILSD 222k (0.037); KXMLBGAME-26OCT062130MILSD 210k (0.035)
- KXSOL15M: KXSOL15M-26NOV151115 106k (0.029); KXSOL15M-26NOV141745 69k (0.019); KXSOL15M-26NOV181615 53k (0.014); KXSOL15M-26NOV191530 51k (0.014); KXSOL15M-26NOV160130 51k (0.014)
- KXNCAAWBGAME: KXNCAAWBGAME-26NOV18USUBYU 283k (0.325); KXNCAAWBGAME-26NOV13NWWVU 179k (0.206); KXNCAAWBGAME-26NOV15ORSTARIZ 121k (0.139); KXNCAAWBGAME-26NOV19BGSUPSU 88k (0.101); KXNCAAWBGAME-26NOV18UMBCPITT 73k (0.084)
- KXWTI15M: KXWTI15M-26NOV201445 62k (0.014); KXWTI15M-26NOV132200 47k (0.011); KXWTI15M-26NOV161500 38k (0.009); KXWTI15M-26NOV181315 38k (0.008); KXWTI15M-26NOV201830 37k (0.008)
- KXCONCACAFNLGAME: KXCONCACAFNLGAME-26NOV14JAMMEX 1.58M (0.311); KXCONCACAFNLGAME-26NOV14HTIUSA 785k (0.154); KXCONCACAFNLGAME-26NOV17MEXJAM 656k (0.129); KXCONCACAFNLGAME-26NOV13GTMCAN 422k (0.083); KXCONCACAFNLGAME-26NOV16CANGTM 336k (0.066)
- KXINTLFRIENDLYGAME: KXINTLFRIENDLYGAME-26NOV13ARGURU 854k (0.183); KXINTLFRIENDLYGAME-26NOV13COLCHI 440k (0.094); KXINTLFRIENDLYGAME-26NOV17JPNPAR 376k (0.081); KXINTLFRIENDLYGAME-26NOV14JPNBRA 355k (0.076); KXINTLFRIENDLYGAME-26NOV17ARGCRI 315k (0.067)
- KXUFCFIGHT: KXUFCFIGHT-26NOV14GANHOK 1.79M (0.465); KXUFCFIGHT-26NOV14HASGUR 790k (0.205); KXUFCFIGHT-26NOV14BORBEL 270k (0.07); KXUFCFIGHT-26NOV14COSGOR 219k (0.057); KXUFCFIGHT-26NOV14THORAD 181k (0.047)
- KXNBATOTAL: KXNBATOTAL-26NOV17CHALAL 153k (0.137); KXNBATOTAL-26NOV16PHXDEN 112k (0.101); KXNBATOTAL-26NOV13OKCLAC 97k (0.087); KXNBATOTAL-26NOV13NYKIND 85k (0.076); KXNBATOTAL-26NOV15OKCSAC 81k (0.072)
- KXNCAAFSPREAD: KXNCAAFSPREAD-26NOV13HOUCOLO 352k (0.189); KXNCAAFSPREAD-26NOV14UTAHARIZ 264k (0.142); KXNCAAFSPREAD-26NOV13ILLUCLA 206k (0.11); KXNCAAFSPREAD-26NOV14TEXLSU 183k (0.098); KXNCAAFSPREAD-26NOV14USCIND 133k (0.071)
- KXUCLGAME: KXUCLGAME-26NOV04AEKRMA 358k (0.107); KXUCLGAME-26OCT14ROMRMA 245k (0.073); KXUCLGAME-26OCT14MCIPSG 165k (0.05); KXUCLGAME-26NOV03MUNROM 164k (0.049); KXUCLGAME-26NOV03BARAVL 158k (0.047)
- KXZEC15M: KXZEC15M-26NOV140930 46k (0.125); KXZEC15M-26NOV191715 29k (0.079); KXZEC15M-26NOV141600 21k (0.057); KXZEC15M-26NOV171200 18k (0.05); KXZEC15M-26NOV141830 18k (0.049)
- KXNHLGAME: KXNHLGAME-26NOV15NSHSJ 267k (0.087); KXNHLGAME-26NOV18COLUTA 264k (0.086); KXNHLGAME-26NOV16VANUTA 240k (0.078); KXNHLGAME-26NOV15VANVGK 212k (0.069); KXNHLGAME-26NOV17LABOS 130k (0.043)
- KXNBASPREAD: KXNBASPREAD-26NOV17CHALAL 241k (0.178); KXNBASPREAD-26NOV15OKCSAC 137k (0.101); KXNBASPREAD-26NOV18UTALAL 122k (0.09); KXNBASPREAD-26NOV16PHIBKN 116k (0.085); KXNBASPREAD-26NOV17DETSAS 109k (0.081)
- KXSILVER15M: KXSILVER15M-26NOV181700 24k (0.046); KXSILVER15M-26NOV201600 23k (0.044); KXSILVER15M-26NOV160045 19k (0.035); KXSILVER15M-26NOV201345 17k (0.033); KXSILVER15M-26NOV152215 17k (0.032)
- KXWTACHALLENGERMATCH: KXWTACHALLENGERMATCH-26NOV20MARLAZ 284k (0.131); KXWTACHALLENGERMATCH-26NOV19LABLAZ 249k (0.115); KXWTACHALLENGERMATCH-26NOV17PAPSAM 213k (0.098); KXWTACHALLENGERMATCH-26NOV19ZOLBUR 188k (0.086); KXWTACHALLENGERMATCH-26NOV14ANDMAN 134k (0.062)
- KXNCAAMBSPREAD: KXNCAAMBSPREAD-26NOV19HOUKU 260k (0.26); KXNCAAMBSPREAD-26NOV17FLAND 105k (0.105); KXNCAAMBSPREAD-26NOV18HOUFLA 104k (0.104); KXNCAAMBSPREAD-26NOV19FLAWVU 90k (0.09); KXNCAAMBSPREAD-26NOV18CONNARIZ 75k (0.075)
- KXDOGE15M: KXDOGE15M-26NOV141845 50k (0.024); KXDOGE15M-26NOV141745 43k (0.02); KXDOGE15M-26NOV160900 39k (0.018); KXDOGE15M-26NOV141945 37k (0.018); KXDOGE15M-26NOV180245 33k (0.015)
- KXHYPE15M: KXHYPE15M-26NOV151515 53k (0.158); KXHYPE15M-26NOV141030 24k (0.069); KXHYPE15M-26NOV171515 21k (0.061); KXHYPE15M-26NOV141730 14k (0.042); KXHYPE15M-26NOV151730 13k (0.038)
- KXNEAR15M: KXNEAR15M-26NOV142215 28k (0.097); KXNEAR15M-26NOV142200 19k (0.065); KXNEAR15M-26NOV150100 18k (0.061); KXNEAR15M-26NOV151030 15k (0.052); KXNEAR15M-26NOV201815 13k (0.045)
- KXNCAAMBTOTAL: KXNCAAMBTOTAL-26NOV19HOUKU 159k (0.21); KXNCAAMBTOTAL-26NOV19FLAWVU 103k (0.136); KXNCAAMBTOTAL-26NOV17FLAND 90k (0.119); KXNCAAMBTOTAL-26NOV18HOUFLA 80k (0.106); KXNCAAMBTOTAL-26NOV19NDAUB 74k (0.098)
- KXUFCMOV: KXUFCMOV-26NOV14GANHOK 1.66M (0.418); KXUFCMOV-26NOV14MEDHOL 549k (0.138); KXUFCMOV-26NOV14HARNUN 481k (0.121); KXUFCMOV-26NOV14BORBEL 294k (0.074); KXUFCMOV-26NOV14HASGUR 248k (0.063)
- KXUEFANLTOTAL: KXUEFANLTOTAL-26NOV16GERNED 161k (0.169); KXUEFANLTOTAL-26NOV17DENNOR 156k (0.165); KXUEFANLTOTAL-26NOV14KOSISR 98k (0.103); KXUEFANLTOTAL-26NOV15ESPENG 84k (0.089); KXUEFANLTOTAL-26NOV16KAZSVK 62k (0.065)
- KXEPLGAME: KXEPLGAME-26NOV07NFOMCI 194k (0.084); KXEPLGAME-26NOV07ARSHUL 175k (0.076); KXEPLGAME-26OCT17BRELFC 143k (0.062); KXEPLGAME-26OCT18NFOARS 140k (0.061); KXEPLGAME-26OCT17MCIIPS 123k (0.053)
- KXMLBHR: KXMLBHR-26OCT142000MILLAD 97k (0.12); KXMLBHR-26OCT232000TBMIL 92k (0.114); KXMLBHR-26OCT152000MILLAD 78k (0.096); KXMLBHR-26OCT061800LADATL 73k (0.09); KXMLBHR-26OCT272000MILTB 71k (0.087)
- KXNFLTD: KXNFLTD-26NOV16LACBAL 1.82M (0.251); KXNFLTD-26NOV15PITCIN 1.19M (0.165); KXNFLTD-26NOV19INDHOU 1.11M (0.154); KXNFLTD-26NOV15SFDAL 636k (0.088); KXNFLTD-26NOV15MINGB 594k (0.082)
- CONTROLH: CONTROLH-2026 2.10M (1.0)

### Caveats that change decisions
- ONE week (11-14..11-20) for 325 series: an international break (EPL/LaLiga/Serie A absent -> measured from 60 d
  instead; UEFA NL / CONCACAF NL / friendlies inflated by x30/7) and a mid-season NFL/NCAAF week.
- Heavy series are event-sampled (Horvitz-Thompson): KXESOCCERGAME 18 of 3,182 events (x212) is the noisiest row;
  `lower_bound` rows (KXNFLTD, KXNBAPTS, KXNBAREB, KXNFLRECYDS/RSHYDS) are under-counted.
- phi comes from <= 15 newest-first tape pages on each series' top-TAIL markets (truncated on liquid markets), so it is a
  rough share; KXNFLGAME .163 / KXNFLTD .143 are the least reliable. MIRROR is approximate (rng on hourly markets).
- Unmeasured: series with week bound < $5M (sports) or < $20k (non-sports), KXMVE* combos, and low-frequency series
  outside the top-36 sports / top-200 non-sports by lifetime volume.

## Learnings
### Kalshi's loser-paid money sits in sports and 15-minute crypto, not in nowcastable data series
Of $659M/30d TAIL measured across 391 series, sports took $335M (tennis $78.6M of it), crypto 15-min $242M; all 43
weather series together $1.49M (0.2%), economics $1.07M. E4's books rank 125-155 by TAIL. A data-nowcast edge will
always be small in dollars; the dollars are in in-play sports and 15-min crypto, where a paid fast feed or a latency
bot sees the outcome first. Source: results/pool_census/pool_table.csv, build_table.py.
### A big loser-paid pool can be pure reversal premium: tennis blind sellers lost money
ATP Challenger TAIL $33.1M vs MIRROR $38.1M per 30 d (net -$5.0M); ITF -$0.75M, ITFW -$1.55M; NBA game -$8.1M; KXBTCD
-$10.2M. Positive nets: KXBTC15M +$75.9M, KXNCAAMBGAME +$13.5M, KXNFLGAME +$26.3M (week). Always report TAIL - MIRROR
next to TAIL before calling a pool harvestable. Source: results/pool_census/pool_table.csv.
### Hourly candles hide in-play tails; resolve the price path at the speed the market moves
KXNBAGAME-26NOV13NYKIND-NYK exact tape TAIL $150,564 vs hourly-candle $0; with 1-min candles over the last 4 h $160,900
(1.07). Hourly is fine for daily weather/gas (19/19 P2 series within 0.79-1.17). Source: validate_extra.log,
validate_hybrid.json, validate_p2.json.
### Kalshi's live REST API keeps only ~69 days; P2's 90-day pools are ~1.3x understated
GET /historical/cutoff = 2026-09-21; live /markets returned KXHIGHNY markets only from 09-13. P2 rows run with 90 d got
~10 weekly events and divided by 90. Use a window inside live retention or add /historical. Source: notes/pool_census.md log.
### Taker-reachable share of tail pools is ~0.3-0.7 even where far rungs are one-sided
phi_tail (resting-loser share of TAIL $) ranged .04-.92 over 94 series (median .484); KXAAAGASD .45 although only 15% of
its far candles were two-sided -- the resting cheap bids appear briefly and someone (often the MM) hits them. phi is an
upper bound for an outsider. Source: results/pool_census/phi_week.json, phi_week2.json.
