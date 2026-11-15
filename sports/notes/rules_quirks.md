# rules_quirks -- Kalshi settlement-definition quirks (agent started 2026-11-15)

## Question
Do any open Kalshi series settle on a definition (rules_primary/secondary: source, OT inclusion, cutoffs, timezone,
>= vs >, station) that differs from the dominant reference definition (sportsbooks / official stats) in a way that
changes the outcome in a non-trivial share of events -- and do Kalshi prices reflect the Kalshi definition or the
reference definition?

## Pre-registration (written before any outcome/price data is examined)
Step 1 (census, descriptive): classify every open series' rules text. FLAG = definition differs from the reference
default AND the difference plausibly flips >= 2% of events. No p-values here.
Step 2 (per flag, quantification): (a) share of historical events where the Kalshi definition and the reference
definition give different outcomes; (b) pricing test: compare Kalshi pre-game implied prob p_K with the
reference-implied prob p_ref and with the definition-adjusted prob p_adj. Statistic: mean(p_K - p_adj) vs
mean(p_K - p_ref) on matched markets; "Kalshi prices the book definition" if |mean(p_K - p_ref)| < |mean(p_K - p_adj)|
with the gap CI excluding 0 (event-clustered bootstrap, 2000 resamples).
Trade rule (only if (b) says Kalshi prices the reference definition): written below as an addendum BEFORE scoring;
taker at ask + Kalshi fee; success = ROI CI lower bound > 0 and p below the edgelab family bar; kill = ROI <= 0 or
n < 100 settled.

## Plan / data
- Kalshi GET /series, /events?status=open&with_nested_markets=true (kx_throttle.kget). Cache under results/rules_quirks/.
- Classifier: regex on rules_primary + rules_secondary per series (sample market per series).

## Early flag from the cached 10-29 census (rules text only, no prices looked at) -- 2026-11-15
- **KXPGATOP5/10/20 (+ KXPGAR{1,2,3}TOP{5,10}, LPGA/DPW twins if present): "finishes in the top 10 (including ties)"
  -> every tied player gets the full $1.** Sportsbooks settle top-N finishes with DEAD-HEAT rules (stake split so total
  payout = N). So the Kalshi contract is worth MORE than the book contract whenever there are ties at the cutoff; the
  field-wide sum of fair Kalshi YES values = E[# players at position <= N incl. ties] > N. A trader who copies book /
  DataGolf-style top-N odds underprices Kalshi YES.
- Round-leader (R1/R2/R3LEAD) and 3-ball markets pay $1/N on ties = dead heat = same as books (no flag).
- NHL totals: OT counts, shootout = 1 goal to winner = book convention (no flag). NFL/NBA totals/spreads: "score"
  with no OT exclusion = final score incl. OT = book convention (no flag). EPL: 90'+stoppage = book convention.
  NFL ML tie -> 0.50 = 2-way push (no flag).

## Pre-registration addendum G (golf top-N ties) -- written BEFORE any golf price or settlement count is pulled
Data: all settled KXPGATOP5/KXPGATOP10/KXPGATOP20 events (live + /historical endpoints). Per market: result, and hourly
candle yes_ask/yes_bid at T_ref = last hourly candle ending <= 11:00 UTC on the tournament's first-round day (first
round day = Thursday of the event week unless metadata says otherwise; markets with no ask at T_ref are dropped).
(G-a) Definition gap: per event, R = # markets settled YES; gap = R - N. Report mean R/N with event bootstrap CI.
(G-b) Pricing: S_mid = sum of mids at T_ref per event (only events where >= 90% of markets have a quote).
  "Kalshi prices the BOOK definition" if mean(S_mid) is closer to N than to mean(R), with the CI of
  mean(S_mid - R) excluding 0 (negative). "Prices the Kalshi definition" if CI of mean(S_mid - R) includes 0.
Trade trials (taker, YES at ask + 0.07*P*(1-P) fee rounded up per 1-lot, event-clustered bootstrap 2000):
  GT1: buy YES on every market with ask in [0.03, 0.95] at T_ref, all three N pooled (one unit per market).
  GT2: same, ask in [0.10, 0.60] (excludes extreme longshots where FLB dominates).
  Success: ROI > 0 with event-cluster CI lower bound > 0 AND p below the edgelab family bar. Kill: ROI <= 0.
  Dev = events with first-round day < 2026-07-01; holdout (sealed, user yes only) = >= 2026-07-01.
Amendment G1 (11-15, before any price pulled): first-round day = Thursday on/before the LAST close_time of the event
(ET). expected_expiration_time is ~2 weeks after the end and min(close_time) is the Friday cut, so both were wrong
anchors (my first version used expected_expiration -> 3M Open R1 = Aug 6 instead of Jul 23). Events with < 30 settled
markets (in-progress AUCPBCS26) dropped. G-a (settlement counts only, no prices; all 43 complete events,
results/rules_quirks/golf_meta.py): mean R/N = 1.15 (top5), 1.12 (top10), 1.09 (top20); R > N in 57% / 66% / 73% of
events. Example: KXPGATOP20-THCCBN26 R = 30, KXPGATOP5-TOC26 R = 10.

Amendment G2 (11-15, before prices): the shared Kalshi throttle gives ~0.3 req/s and dev events (Jan-Jun 2026) only
exist on /historical/markets/{t}/candlesticks (one call per market; the event-level candles endpoint returns empty for
historical events). So candles are fetched for DEV events only, order TOP10 -> TOP20 -> TOP5; if time runs out the
scored set is whichever series completed (stated in the result). edgelab trials rq_golf_gt1-510722d2,
rq_golf_gt2-db46068a registered (family kalshi_rules_quirks, bar .025 after 2 trials) BEFORE any price was read.

Amendment G3 (11-15): scored dev set = KXPGATOP10 only (2,949 dev markets, 2,653 quoted at T_ref; TOP20/TOP5 candle
fetch stopped -- ~6 h more at the shared rate). edgelab rejected v1 at point-in-time: T_ref 11:00 UTC can be AFTER the
first tee (6:40 ET = 10:40 UTC in EDT months). It printed 5 example outcomes in the error (no ROI seen). v1 trials
retired; v2 rq_golf_gt1b-56f3cdd4 / rq_golf_gt2b-cf872080 = same rule with T_ref 10:00 UTC, event_start 10:40 UTC
(edgelab/rules/golf_topn_b.py). Family now 6 trials, bar p < .0083.

## Step 1 census result (11-15; 4,750 open series, results/rules_quirks/rules_by_series.jsonl, rules_features.csv)
Regex classes (scripts/rules_quirks_classify.py) then manual read of the top-volume hits. Reference = book convention.
| class | series | vs book | flag? |
|---|---|---|---|
| top-N "including ties" (full $1 each) | KXPGATOP5/10/20, KXDPWTTOP5/10/20, KXPGAR3TOP10, R1/R2 TOPN | books dead-heat (sum pays N) | **FLAG 1** (9-15% extra mass) |
| tie/NR/abandoned -> all markets $0.50 | KXT20MATCH, KXODIMATCH, KXWT20MATCH, KXCRICKETT20IMATCH, KXIPLGAME; KXUFCFIGHT draw/NC; KXNFLGAME tie | books VOID (stake back) | **FLAG 2** for cricket (NR frequent); UFC/NFL rare |
| tennis "after a ball has been played"; ITF: retiring player -> No, walkover -> 0.50 | KXATPMATCH, KXATPCHALLENGERMATCH, KXITFWMATCH | books differ book-by-book (1 set rule / void) | weak flag 3 (retirements ~2-3%, convention not uniform) |
| OT counted (NFL props say "including overtime"; NFL/NBA totals silent = final score); NHL shootout = 1 goal | | same as books | no |
| soccer 90'+stoppage | KXUEFANL*, KXLALIGA*, ... | same as books | no |
| round leader / 3-ball ties pay $1/N | KXPGAR*LEAD, KXPGA3BALL | = dead heat | no |
| UFC title: interim excluded; vacant -> all NO (KXUFCTITLE) vs "Vacant" market (KXUFCHEAVYWEIGHTTITLE) | | no book analogue | judgment, not systematic |
| "5 PM EST" literal in crypto (KXBTCD) during EDT; weather = Weather Company CLI | | | known; not a book-gap |

## Flag 2 frequency (settlements only; scripts/rules_quirks_cricket.py, results/rules_quirks/cricket_markets.parquet)
Share of settled events where every market settled at $0.50 (scalar): KXT20MATCH 11.9% of 1,929; KXTESTMATCH 10.6%/66
(has a Draw market, so scalar = abandoned); KXODIMATCH 7.6%/251; KXCRICKETT20IMATCH 4.4%/68; KXWT20MATCH 3.9%/459;
KXIPLGAME 2.7%/75; KXBBLGAME 0/243. Mechanism: fair Kalshi price of a team = p_win*(1-q) + 0.5*q where p_win is the
void-adjusted (book) prob and q = P(NR). At q = 12%, a 0.80 book favourite is worth 0.764 on Kalshi (-3.6c) and the
0.20 dog 0.236 (+3.6c).

## Pre-registration addendum C (cricket NR) -- written BEFORE any cricket price is pulled
No book odds for these minor leagues, so the test is Kalshi-internal. T_ref = last hourly candle ending <= scheduled
start - 1 h (start parsed from rules_primary "originally scheduled for <date> at <time> EST"; events without a
parseable start dropped). Two-team events only (exclude Test draw markets).
  C0 (descriptive): payout-vs-price calibration of the favourite (mid >= 0.5) by price bin, payout counts NR as 0.5.
  C1 trade: buy YES on the UNDERDOG at ask + fee when its ask is in [0.10, 0.45] at T_ref. One unit per event.
  C2 trade: C1 restricted to series with dev NR rate >= 7% (KXT20MATCH, KXODIMATCH, KXTESTMATCH excluded as 3-way).
  Event-clustered = one bet per event, so plain bootstrap over bets. Success: ROI CI lower bound > 0 AND p below the
  edgelab family bar; kill: ROI <= 0. Dev = scheduled start < 2026-07-01; holdout >= 2026-07-01 (user yes only).
  (C2 series list fixed as KXT20MATCH + KXODIMATCH. KXCRICKETT20IMATCH has no start time in its ticker -> dropped.)
  edgelab: rq_cricket_c1-f1f528d9, rq_cricket_c2-aea39410 registered before prices (family bar .0125 after 4 trials);
  vault outcome kind 'none' because the evaluator settles non-0/1 outcomes as a push -> script is authoritative.

## Result C (11-15, DEV, `scripts/rules_quirks_cricket_px.py score`; bets in results/rules_quirks/cricket_C*_dev_bets.csv)
671/695 dev events quoted at start-1 h; NR share among them 13.6%.
- C0 favourite calibration, payout counting NR as 0.5: mid .543 -> pay .556 (n252); .646 -> .657 (102); .744 -> .709
  (67); .852 -> .826 (46); .939 -> .963 (41). No systematic overpricing of favourites -> prices already reflect the
  $0.50 NR rule (no book-copy gap to harvest).
- C1 underdog YES at ask: n 216, 59 wins, 9 NR, ROI **-14.7% [-30.8, +2.2]** (p(boot<=0) .96). KILL (ROI <= 0).
- C2 (T20 + ODI): n 132, ROI **-10.5% [-31.3, +11.4]**. KILL. NR payouts contributed +1.15 of -4.85 units.
- Underdogs at ask lose: the favourite-longshot bias plus the spread outweigh the q(p-0.5) bonus.
Verdict: **DEAD**. Holdout not requested.

## Result G (11-15, DEV = Jan-Jun 2026 KXPGATOP10; `scripts/rules_quirks_golf.py score`, edgelab evaluate --split dev)
- G-a definition gap (all 43 complete events, settlements only): mean R/N = 1.147 [1.105, 1.191] (top10), 1.113
  [1.080, 1.148] (top20), 1.177 [1.121, 1.237] (top5); R > N in 67% / 74% / 58% of events. The rules gap is real
  and large: books dead-heat pays N per event, Kalshi pays ~11.5 per top-10 event.
- G-b pricing (22 dev TOP10 events with >= 90% quoted at 10:00 UTC Thursday): sum of mids 11.08 vs realized 11.45,
  S - R = -0.38 [-1.12, +0.32] -> CI includes 0 -> **Kalshi prices the KALSHI (ties-included) definition**, not the
  book's N = 10.
- Trades (taker at ask, fee): rq_golf_gt1b-56f3cdd4 n 2,376, ROI **-18.0% [-23.8, -12.1]**, halves -20.2/-15.8%;
  rq_golf_gt2b-cf872080 n 1,404, ROI **-14.5% [-22.1, -6.5]**, halves -14.0/-14.9%. Both DEAD (edgelab state dead).
  The script with cent-rounded fees gives TOP10 -20.7% / -16.3% (same picture). Mean ask .13: the spread + fee on
  cheap YES, plus the longshot premium, swamp any residual tie value.
Verdict: **DEAD** -- the gap exists but is priced. Holdout (Jul-Nov) not requested.

## Final verdict (rules_quirks)
The census of 4,750 open series found two systematic settlement-definition gaps vs the book convention (golf top-N
pays every tie in full; cricket NR/tie settles at $0.50 where books void). Both change 9-18% of outcomes or payout
mass, and in both the Kalshi price already reflects the Kalshi definition. Buying the side the gap favours loses at
the ask (golf -18%, cricket -15%). Weak flag 3 (tennis retirement conventions differ book to book) was not tested: no
book data, and the convention isn't uniform, so there is no single "book price" to be wrong against. Sports totals and
spreads have no quirk: NFL/NBA/NCAA totals count OT, NHL counts the shootout as 1 goal, soccer is 90'+stoppage --
all identical to books.

## Learnings
### A Kalshi rules gap vs books is not an edge unless prices copy the book
Golf top-N: Kalshi pays $1 to every tied player (realized YES per top-10 event 11.47, R/N 1.147 [1.105,1.191]), books
dead-heat to N. But the sum of Kalshi mids at Thursday 10:00 UTC was 11.08 vs realized 11.45 (S-R -0.38 [-1.12,+0.32]):
the market prices the ties. Buying YES at ask lost -18.0% [-23.8,-12.1] (rq_golf_gt1b). Source notes/rules_quirks.md.
Check the sum of the field's prices against realized counts before building a "definition gap" trade.

### Kalshi cricket NR/tie/abandon settles every team at $0.50 (books void)
NR share of settled events: KXT20MATCH 11.9% (1,929), KXODIMATCH 7.6% (251), KXWT20MATCH 3.9% (459), KXIPLGAME 2.7%;
13.6% of quoted dev events. Prices already reflect it: favourite payout incl. NR = 0.5 tracks the mid in every bin;
underdog YES at ask -14.7% [-30.8,+2.2] (n216). scripts/rules_quirks_cricket_px.py.

### Kalshi settlement conventions that match the books (do not re-audit)
NFL/NBA/NCAA totals and spreads count OT (rules silent; NFL props say "including overtime"); NHL totals count OT and
score a shootout as 1 goal to the winner; soccer = 90' + stoppage; golf round-leader / 3-ball ties pay $1/N (= dead
heat). results/rules_quirks/rules_features.csv has the regex classes for all 4,750 open series (2026-11-15).

### edgelab rejects a decision time that is not strictly before event_start
Golf T_ref 11:00 UTC was pit-rejected because the first tee in EDT months is ~10:40 UTC. A trial cannot be amended
once evaluated, so retire it and register v2 (family 4 -> 6 trials, bar .0125 -> .0083). Pick the decision time against
the EARLIEST possible start before registering.

### edgelab's evaluator cannot settle a fractional ($0.50) payout
cost_and_pnl treats any outcome other than 0/1 as a push (pnl 0). Markets that settle at $0.50 (cricket NR, NFL tie,
UFC draw) must be scored by script; the vault entry uses outcome kind 'none' (kalshi_cricket_nr).

### The shared Kalshi throttle gives one agent ~0.15-0.3 req/s; historical candles cost one call per market
Event-level candles (/series/{s}/events/{e}/candlesticks, one call for all markets) work only for markets still on the
live API (~Sep 2026+); /historical needs one call per market. 2,949 golf markets took ~3 h. Plan dev sets around this.
