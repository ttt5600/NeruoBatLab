# kalshi_first_listing -- how stale is Kalshi's first tradeable sports price, across ALL sports series?

Agent `kalshi_first_listing`, started 2026-11-12. Script: `scripts/kalshi_first_listing.py`. Scratch: `results/kalshi_first_listing/`.
Sibling `kalshi_openers_xleague` owns model-vs-opener (totals/spreads); this file is MARKET-ONLY staleness.
Prior: L35 NFL early listings CLV +3.3c p<.0002 (109 games), maker at first bid adverse +16% p .09, taker first ask -4.5%.

## PRE-REGISTRATION (written 2026-11-12 BEFORE any candle / price path was pulled; seen beforehand: the Sports series
list (4,293 series) and settled-market METADATA for the window only)

### Reference choice (decided before data)
- Pinnacle history at Kalshi listing time does NOT exist locally: gap_logger keeps only flagged (EV>=1%) sides,
  open_lines/open_totals state files are pruned to current games and their journals hold model-pick rows only
  (871 rows, NHL/NFL/NBA-totals; that is the sibling's model-vs-opener territory). So the census reference is the
  **Kalshi mid at T-30 min** (the task's fallback). Pinnacle-at-listing would need a new forward logger -> proposal only.
- Start time proxy (market-only, all series): T_start = expected_expiration_time - 3 h (repo convention, checked vs ESPN
  on 2025 US leagues in scripts/kalshi_start_times.py; 2026 NFL sometimes exp = start + 6 h, which makes the proxy EARLIER
  than start = still pre-game, conservative). T_ref = T_start - 30 min. Guard: an event is flagged "in-game at T_ref"
  if any hourly candle in [T_ref-2h, T_ref] has trade range >= 0.25; series with > 20% flagged are reported but cannot pass.

### Data
- Window: markets with close_time in [2026-10-20, 2026-11-11) UTC, status settled, multivariate excluded, series in the
  Kalshi Sports category. This is DEV (descriptive census); any bet rule that passes is tested FORWARD only.
- Sample: series with >= 30 events in the window (top 40 by volume). Up to 60 events per series (random, seed 0).
  Per event ONE market: the market whose first qualifying mid is closest to 0.50 (no lookahead: chosen from first quotes).
- Prices: hourly candles (period_interval=60) from open_time to T_ref. FIRST quote = close bid/ask of the first candle
  with bid > 0, ask < 1 and ask - bid <= 0.10 (L35 rule). Mid at T_ref = last such candle ending <= T_ref.

### Census columns (per series)
lead_h (T_start - open_time, median), first_quote_delay_h (first qualifying candle - open_time), first_spread (median),
|D| = |mid_Tref - mid_first| (median, mean), vol before T_ref (median), n events.
Direction predictability, market-only, all info known at first quote + 2 h:
  P1 early drift: d1 = mid(first+2h) - mid(first); r = mid_Tref - mid(first+2h). Spearman(d1, r).
  P2 level: Spearman(mid_first - 0.5, D).  (favourite/longshot drift)

### Census PASS (pre-registered)
n >= 30 AND median |D| >= 0.03 AND median first_spread <= 0.05 AND in-game flag rate <= 20% AND (P1 or P2 has
Spearman p < 0.001 [~Bonferroni over 40 series x 2 predictors] with the same sign in both date halves of the window).
Only passing series get a bet rule, written below BEFORE its outcome is computed, registered in edgelab family
kalshi_sports_pregame, and tested forward (dev window already used for selection).

### Capturability diagnostics (uncounted, described for every series with n >= 30)
- Taker unconditional: buy YES at first ask + fee 0.07*a*(1-a), hold to settlement; ROI.
- Maker at first bid (YES), filled iff a later candle (ending <= T_ref) has trade low <= bid - 0.01 (trade-through);
  maker fee 0.0175*b*(1-b). Report fill rate, ROI of filled, and YES-rate of filled vs unfilled (adverse selection,
  E5 lesson). Same for NO side (maker at 1 - first ask on NO = YES ask; filled iff trade high >= ask + 0.01).
These are not bet rules; they only say whether staleness, if any, could be harvested at real prices.

## Log
- 2026-11-12: inbox `started` row. universe pull running.
- 2026-11-12: global settled-markets pull (all categories) was ~90 min (2,000 pages covered only 11-07..11-11), killed.
  Selection changed BEFORE any prices: rank Sports series by volume on those cached 11-07..11-11 pages
  (results/kalshi_first_listing/phaseA_rank.csv), take the top 55 with >= 5 events, pull each one's full window
  per-series, then apply the pre-registered >= 30 events / top 40 by volume / 60 events rule.
  Ladder events: candles only for the middle 5 markets by strike (metadata-only choice) to keep API calls bounded.
- 2026-11-12 10:05: candle fetch ran at ~19 markets/min (Kalshi 429s, the IP is shared with sibling agents) -> 6 h.
  Killed after ~300 markets, NO census computed. Sample cut BEFORE any outcome: 40 events/series, ONE market per event
  (ladders: middle strike by metadata; team/3-way events: middle ticker alphabetically, e.g. 2-team ML = first team).
  The "first mid closest to 0.5" rule is therefore dropped (metadata choice instead; still no lookahead).
- 2026-11-12 ~10:50: candles done (1,571 markets). `python scripts/kalshi_first_listing.py census` ->
  results/kalshi_first_listing/census.csv (per series) + per_event.parquet (1,578 rows, one market per event).

## RESULT (dev window 10-20..11-11, 40 series, 1,578 events): CENSUS PASS 0/40 -> no bet rule, nothing registered
Gate by gate (census.csv): median |D| >= 3c in 19/40; median first spread <= 5c in 18/40; both in 5/40
(KXT20MATCH 5.0c/5c, KXNBATOTAL 4.5c/4.5c, KXNFLGAME 4.2c/5c, KXLALIGAGAME 3.0c/4c, KXNFLPASSYDS 3.0c/5c);
NONE of those 5 has a listing-time market-only predictor (best: LaLiga level P2 rho +0.41 p .024; NBA totals early-drift
P1 rho +0.34 p .032; NFL game P2 +0.31 p .055; bar p < .001). Only two series cross p < .001 at all: KXITFWMATCH
(P2 +0.51 p .0008, but 42% of events flagged in-game at T_ref = start proxy broken for ITF tennis) and KXNFLTD
(P2 -0.61 p .00006: favourites drift down, longshots up) but median |D| 1.5c on a 2c spread -> nothing to capture.
Lead time (median h, listing -> start proxy): soccer top-5 ~350, CFB 335, NFL 302, NHL 167, NFL spread/total 164,
NBA/CBB game ~96, props 19-111, tennis 12-24. First-quote spread median 6c pooled; tennis challengers 2c.

Pooled (1,431 events not flagged in-game; per_event.parquet; bootstrap over events, 2,000 draws, seed 0):
- Staleness is REAL but small: Brier first mid 0.1974 vs T-30 mid 0.1889; median |D| 2.5c = exactly one half-spread
  (median |D| / half-spread = 1.00); |D| > half-spread + 2c in 27.1% [24.8, 29.5] of events.
- Taker YES at first ask + fee: ROI -10.0% [-15.2, -4.8].
- Maker YES at first bid, trade-through fill (31% fill): ROI of filled -11.0% [-21.5, -0.7].
  Maker NO at 1 - first ask (45% fill): -5.0% [-11.8, +1.6] (not distinguishable from 0).
- ADVERSE SELECTION (E5 lesson replicated across all sports): (outcome - first mid) of filled YES makers minus unfilled
  = -9.3c [-14.5, -4.4]. Per series, yes-rate filled vs unfilled: NBA game .14 vs .68, NHL .33 vs .61, NFL .47 vs .63.
  The first bid gets hit exactly when the price is about to move down: resting at the listing price is selling an
  option to whoever learns first.
Consistent with L35 (NFL first-listing CLV real, maker adverse, taker negative); this census says the same holds in
every big Kalshi sports series and that market-only info at listing does not tell you the direction.

Caveats: hourly candles hide intra-hour quotes (first quote is the first hourly close); no order-book depth; start
proxy = expected_expiration - 3 h (wrong for ITF/ITFW tennis, UFC 22%, NFL rush yds 28% flagged); 3-week window;
one market per event; futures not covered (none settle inside a 3-week window).

## Not done / proposal (needs the lead: new launchd logger)
The only untested reference is a SHARP price AT LISTING. No local history exists (gap_logger logs only flagged sides;
open_lines/open_totals journals hold model-pick rows only). Proposal: a read-only logger that, on each new Kalshi
sports event (KXNFLGAME/KXNBAGAME/KXNHLGAME/KXNBATOTAL/KXNFLTOTAL/soccer top-5), snapshots Kalshi bid/ask + Pinnacle
no-vig (if posted) at first sighting and every 2 h to start. Note Pinnacle is often NOT yet posted at Kalshi listing
(NFL/soccer list 12-15 days ahead), so the gap can only be measured once Pinnacle opens = the sibling's opener question.
Given the -9.3c adverse selection, any such rule must be TAKER at the ask; a maker version starts ~9c in the hole.

## Learnings
### Kalshi first-listing staleness is about one half-spread
Across 40 Kalshi sports series (1,578 events, 10-20..11-11), median |first mid - T-30 mid| = 2.5c vs median first
spread 6c (results/kalshi_first_listing/census.csv). The first price is less accurate (Brier .1974 vs .1889) but the
move is not predictable from market-only info at listing in any series (0/40 pass at p<.001).
### Resting at the first Kalshi bid is adversely selected in every sport
Filled maker YES orders at the first bid end 9.3c [4.4, 14.5] worse vs first mid than unfilled ones; maker ROI -11.0%
[-21.5, -0.7], taker at first ask -10.0% [-15.2, -4.8] (per_event.parquet, pooled bootstrap). Same lesson as E5 and L35:
a fill at a stale quote is information that the quote was wrong in your direction.
### Kalshi rate limit is shared across sibling agents
Hourly-candle fetches ran at ~20-25 markets/min (429s) while other agents were pulling; plan ~1 market per event and
<= 1.5k markets per study, or the fetch takes hours.
### expected_expiration - 3 h is not a start time for tennis/UFC
42-58% of ITF/ITFW tennis events and 22% of UFC fights showed in-game price ranges before that proxy; use ESPN/real
start times for those series.
