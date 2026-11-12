# LX1 -- Kalshi logical-dependency arbitrage scanner (agent lx1_logical_arb, started 2026-11-12)

Source lead: FUND_BOOK 3c; ~/.claude/seed-runner/research_notes/opps_2026-11-12/02_prediction_markets.md (data).
Literature: Saguillo et al. arXiv 2508.03474 (13 / 46,360 strict cross-market arbs on Polymarket);
Cheng et al. arXiv 2605.00864 (290 NBA episodes, median ~101 bp, ~15 shares).
Prior in-repo: scripts/arb_scan.py (cross-venue + YES dutch on a few futures series; negRisk overround DEAD per
EDGES_AND_AGENTS line ~198). This scanner is Kalshi-internal, all open markets, executable asks + depth.

## Question
Do logically dependent Kalshi contracts trade at executable prices that lock in a risk-free profit after taker fees,
often and large enough to be worth hedged execution?

## Pre-registered (written 2026-11-12 BEFORE any price was inspected)
- **Violation** = a set of legs, each bought at the best ask (walking the book), whose MINIMUM payoff over all
  logically possible outcomes exceeds total cost + Kalshi taker fees (0.07*p*(1-p) per contract, ceil to the cent per
  leg order; series fee_multiplier used when the series reports one, never lower than... the reported value)
  by **>= 2c per contract-set, at a size of >= 10 contracts executable on EVERY leg**.
- **Kill**: fewer than **5 distinct violations in 7 days** of scanning (distinct = different leg set; the same leg set
  seen again in a later scan counts once). Also kill if all violations are in one degenerate class (e.g. settled-but-
  not-closed markets) -- those get inspected by hand first.
- **Success (worth proposing hedged execution to the user)**: >= 5 distinct violations in 7 days AND median persistence
  long enough to hit both legs (>= 1 min on candles/re-scan).
- Not a pass on its own: YES-dutch on a mutually exclusive set (needs the set to be EXHAUSTIVE, which Kalshi's flag
  does not guarantee) -- reported separately as `yes_dutch_unverified`, never counted toward the bar unless the rules
  text shows exhaustiveness.

## Constraint types (deterministic, no LLM)
1. Threshold ladder within an event, strike_type greater/greater_or_equal: P(X>k) non-increasing in k.
   Arb: buy YES(k_lo) + NO(k_hi), k_lo < k_hi; min payoff $1.
   less/less_or_equal: mirror (YES(k_hi) + NO(k_lo)).
2. Mutually exclusive set (event.mutually_exclusive): buy NO on any subset of m legs -> min payoff m-1.
   YES on all legs -> payoff 1 only if exhaustive (unverified, reported separately).
3. Date ladders ("by date X"): same series, same proposition across events with different deadlines -> cumulative.
   Arb: YES(later) + NO(earlier) when P(by later) < P(by earlier). Detected from ticker/title structure; inspected.
4. KXFED (rate above k at meeting) vs KXFEDDECISION (cut/hold/hike): if mapping derivable.
5. Seat totals vs races: no open election ladders expected after 2026-11-03; checked, reported if any.

## Plan
probe all open markets (mve_filter=exclude) -> screen on top-of-book (sizes in /markets) -> orderbook depth for
candidates -> log to results/lx1_logical_arb/violations.jsonl -> candles for persistence.

## Log
- 2026-11-12: 182,322 open non-MVE markets in 60 s (results/lx1_logical_arb/probe.py). strike_type counts:
  greater 64,146; structured 46,333; custom 42,878; greater_or_equal 19,559; between 3,398; less 690.
- Scanner: scripts/lx1_arb_scan.py (scan | persist | report). Fees: GET /series -> fee_multiplier (16,261 series:
  16,066 quadratic x1, 18 x0.5, 14 x0). Fed: current upper bound from FRED DFEDTARU = 4.00 (2026-11-12).
- Scan 1 (17:29Z) INVALID: parser bug. The subtitle rule read "$2,769.99 or above" as 769.99 (thousands comma), so KXETHD
  strike ladders looked like +98c arbs (400 "violations"). Quarantined in results/lx1_logical_arb/bugged_scan1/. Fix:
  strip commas/$; numeric strike_type markets go only through the floor/cap_strike family.
- Offline re-screen of the saved snapshot (results/lx1_logical_arb/offline_check.py) exposed 3 more false-positive
  classes, all metadata, none real:
  (a) KXSTARSHIPSPACE-26 "exactly k" buckets are tagged strike_type `less` with floor == cap -> require a one-sided
      strike and no "exactly" in title/rules;
  (b) KXEARNINGSMENTION* "India (2+ times)" vs "Club Wingstop (3+ times)" share a generic title -> group key includes the
      digit-stripped sub-title;
  (c) USCLIMATE-2025 (<=4909.9 Mt by 2025) vs USCLIMATE-2030 (<=3317.5 Mt by 2030) vary on two dimensions -> a pair's
      rules_primary must differ in exactly ONE number.
  After the fixes the stale snapshot screens only 5 thin top-of-book gaps (<= 2.2c, sizes 2-50): KXU3MAX 9/10, KXFUSION
  2030/2035/2040, KXTRUMPADMINLEAVE Burgum Jul-27 vs Jan-28, KXDECLAREPRESFIRSTD Harris Nov-01 vs Nov-07.
  Date-ladder caveat: "after Issuance".
- More bugs found and fixed before trusting anything: (d) size search broke at n=1, where per-leg cent-rounded fees eat
  the edge, so a violation showed max size 0 -> scan the whole size grid; (e) 1-min candles exist only when a quote
  changes, so intersecting timestamps gave 21 minutes for 72 h -> forward-fill onto a 1-min grid; (f) the candle chunk
  loop never ended with a float `end`; (g) a 4xx (window before the market existed) was retried as if transient and drew
  429s -> 4xx is final, plus a client throttle of <= 4 req/s, because the live bots share this IP.
- edgelab: NOT registered. Its trials are bet-ROI rows with bet-based kill kinds, and this study's kill is a count of
  scan opportunities. If the user approves hedged execution, register the executed sets as a forward trial then
  (price_model taker, kill losses_in_first, since any loss on a "risk-free" set means a rules mismatch).

## Results 2026-11-12 (day 0 of 7; NOT a verdict)
- Valid scans: 17:36Z, 20:15Z, 20:18Z (results/lx1_logical_arb/scans.jsonl; the 20:18Z scan ran back-to-back with
  20:15Z, so it is not an independent time point, and its summary shows violations 0 because of a dropped counter, since
  fixed; violations.jsonl has its row). `report` -> 1 distinct set, seen 3x, max 51 sets.
  Scratch snapshots (probe_markets.json 180k markets, series.json) were deleted after use; the scanner refetches live. Each scan covers
  ~182-186k open non-MVE markets, ~21.2-21.5k events, ~1.42-1.60M strike-ladder pairs, 4,255 subtitle pairs, 163
  date-ladder pairs, ~11.2k exclusive events, 8 Fed equivalences.
- Counted violations: ONE distinct leg set, seen in both scans: BUY NO KXTRUMPADMINLEAVE-27JUL01-DBUR @0.70 (x50) + BUY
  YES KXTRUMPADMINLEAVE-28JAN01-DBUR @0.25 (x200), i.e. Burgum leaves before Jul 1 2027 vs before Jan 1 2028.
  Net +2.1c/set at 10; max 51 sets at >= 2c (net 2.08c) -> ~$1.06 profit locking ~$49 until as late as 2028-01-01
  (violations.jsonl, 20:15Z row). Small, and capital is tied up 14 months.
- Zero hits: Fed KXFED vs KXFEDDECISION (Dec), strike ladders (on 1.4-1.6M pairs the only top-of-book gap is KXU3MAX
  9/10 at +0.2c), and excl_no (no exclusive event has YES bids summing above 1 + fees).
- yes_dutch: ~190 events "pay" 10-85c at top of book (KXSTATE51, Oscars, primaries, ...). These are named-candidate
  fields, NOT exhaustive (if an unlisted outcome wins, every leg loses). Not counted; moved to yes_dutch_unverified.jsonl.
- 7-day backward replay (history.csv; date/subtitle/Fed families only, 4,428 constraints over 1,894 markets, 1-min
  candles forward-filled, top of book, 1 contract, so an UPPER bound with no depth): 5 distinct leg sets were ever >= 2c:
  Burgum (1,186 of 8,202 min, 3 runs, median 491 min, max 686), KXFUSION 2035/2030 (473 min, median run 53), Blanche
  (TBLA) Jul-27/Jan-28 (138 min, median 6), KXFUSION 2040/2030 (114 min, median 6), KXMANAGERSOUT MJAI Jan/Jun-27
  (60 min, 2 runs of ~41). Persistence is minutes to hours in thin long-dated markets, so a 2-leg hedge is feasible.
  Most of these fail the 10-contract depth leg (e.g. KXFUSION-30 NO top size 2-6).
- Read so far: the pre-registered bar (>= 5 distinct at >= 10 contracts in 7 days) is at 1/5 after day 0. The history
  upper bound (5 at top of book, mostly < 10 deep) suggests it will likely miss. Judge on 2026-11-19.
- Kalshi vs Polymarket US: not done today. Prior work marks same-event cross-venue spreads DEAD, and cross-venue logical
  pairs need rules-text mapping (the LLM step). Next step only if the Kalshi-internal bar passes.

## Proposed 7-day scan schedule (NOT installed; launchd is the lead's/user's call)
- `cd sports && ../.venv_quant/bin/python scripts/lx1_arb_scan.py scan` every 2 h (12/day, 84 scans). A scan takes about
  5-9 min (~290 list pages + book walks at <= 4 req/s), so 2 h leaves headroom and keeps the shared IP quiet.
- Once a day: `persist --hours 48` (run lengths for the logged sets) and `report` (distinct sets vs the bar).
- Day 7 (2026-11-19): judge the pre-registered rule from `report` (distinct violation keys over 11-12..11-19) plus
  `history --hours 168` as the top-of-book upper bound. Kill if < 5 distinct sets.
- launchd cannot read ~/Desktop (TCC). The agent would have to run a copy outside Desktop, as the other quant agents do.

## Learnings
### Kalshi strike metadata lies: "exactly k" buckets are tagged strike_type less
KXSTARSHIPSPACE-26 "exactly k" markets have strike_type `less` with floor_strike == cap_strike. A naive ladder scanner
reads them as "below k" and finds fake 58c arbs. Require a one-sided strike and no "exactly" in the rules
(scripts/lx1_arb_scan.py).
### A ladder pair must differ in exactly one number in rules_primary
USCLIMATE-2025 vs -2030 vary both the threshold and the year, and KXEARNINGSMENTION shares a generic title across
unrelated words. Grouping on title alone produced fake arbs. Compare the rules text with digits masked, and require one
differing number.
### Kalshi 1-min candles exist only when a quote changes
Intersecting candle timestamps across legs gave 21 minutes for 72 h; forward-filling gave 8,194 minutes. Persistence
studies must forward-fill. A start_ts before market creation returns 400 (not transient; don't retry).
### Kalshi-internal logical arbs exist but are tiny and long-dated
Day 0: 1 counted set (Burgum leaves Jul-27 vs Jan-28: +2.08c x 51 sets, ~$1 locked to 2028). The 7-day top-of-book
replay found 5 sets, all thin political/fusion/manager date ladders lasting minutes to hours. The liquid families
(strike ladders, Fed, exclusive NO) showed zero (results/lx1_logical_arb/history.csv, scans.jsonl).
### YES-dutch on Kalshi "mutually exclusive" events is not an arb
About 190 events sum to well under $1 at the ask because they list named candidates, not every outcome. The
mutually_exclusive flag does not mean exhaustive. If events were issued at different times and the act happened between the
  issuances, nesting could break (the later event would usually not list the person then).
