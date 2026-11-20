# derived_ladders -- are Kalshi SECONDARY sports ladders mispriced in shape? (agent, 2026-11-19)

Hypothesis (from KFG1, notes/kalshi_fg_markets.md): main game lines are made off sharp books; derived ladders (team totals,
halves, player yards, corners, shots...) are priced less carefully, with systematic SHAPE errors (KFG1: 3+ FG mid .218 vs
.269 hit; placebo +5%). Do-not-redo: KXNFLFG (KFG1), KXNBAPTS/REB/AST (prop ladders paper), NFL/NBA/NHL main totals/winners.

Plan: (1) census of Kalshi sports series (public read-only GETs) -> derived ones with >= ~80 settled events + real volume;
(2) descriptive calibration of mid AND ask at T = start - 60 min by rung position / price bucket, event-clustered bootstrap,
count cells; (3) prereg best 1-2 BEFORE returns, edgelab register, dev only (sealed latest ~30% unopened).
Scripts/output: sports/results/derived_ladders/.

## Log
- 2026-11-19 census: results/derived_ladders/census.py over 698 candidate game-level series (regex over the 4,316 Sports
  series in series_sports.json, main GAME/SPREAD/TOTAL + futures excluded, KXNFLFG/KXNBAPTS/REB/AST excluded) -> census.json.
  Settled markets from /markets?status=settled + /historical/markets, capped at 8 pages each (flag `capped`).
- Timing: Kalshi occurrence_datetime - 3 h == ESPN start exactly (std 0) on the 67 ESPN-matched KXEPLCORNERS events;
  fetch_ev.py uses ESPN start, else occ-3h when >= 95% of the series' ESPN-matched events agree within 5 min; MLB from ticker HHMM.
- Quotes: event-level candlesticks endpoint (/series/S/events/E/candlesticks, 1 call per event, works for historical too),
  1-min over [T-6h, T], T = start - 60 min, last candle with end <= T. Output results/derived_ladders/ev/q_<S>.jsonl.
- **SPLIT fixed before any outcome is looked at** (calib.py `split`): per series, events sorted by start; DEV = earliest 70%,
  SEALED = latest 30% (cut = start of event at index floor(0.7 n)). calib.py loads DEV only. fetch_ev.py pulls quotes for all
  events (one call per event, cheaper than re-fetching later), but sealed quotes are never joined to outcomes or scored.
- Descriptive pass (calib.py): valid quote = market open_time <= T, bid > 0, ask < 1, ask > bid. Cells = mid buckets (10)
  and, for strike ladders, relative rung (strike rank minus rank of the rung nearest .5, clipped +-4); cell kept if n >= 30
  and >= 15 events. Per cell: hit rate - mid (event-clustered bootstrap 95% CI), taker ROI YES at ask / NO at 1-bid incl.
  quadratic fee x series multiplier (10-lot, cent-rounded). Descriptive only; every cell counted for the multiplicity note.
- 11-19 lead message: prioritise STRATEGY_v2 derived candidates: soccer correct score (7 leagues), KXNFL1HFT, NFL win margin,
  NFL OT/tie, NBA win margin, KXNCAAMB1HWINNER; adopt gate = base-rate-shape placebo + per-rung calibration (no returns) first.
  Soccer 1H/2H/team totals were listed 11-18/19 -> forward paper only (not enough dev history).
- **Data limit found**: markets older than Kalshi's historical cutoff (/historical/cutoff market_settled_ts 2026-09-19) have NO
  pre-game candles unless they traded (e.g. KXNBA1HTOTAL-26FEB10LACHOU-98: first candle 01:52Z, tip 01:00Z), and the event
  candlestick endpoint returns empty for them. So usable pre-game quote history starts ~2026-09-19 for every series:
  MLB derived series (season ended) mostly drop out; NBA from 10-20, NFL/NCAAF/soccer ~2 months. fetch_ev.py now fetches only
  live-src (post-cutoff) events; the 70/30 split is defined over those events only (amended before any outcome was viewed).
- **AMENDMENT (11-19, after viewing only KXEPLCORNERS and KXNBA1HTOTAL full-dev tables, neither has a usable cell):**
  to avoid choosing and testing a rung class on the same rows, DEV is halved by start time: devA (first 50% of dev events)
  = discovery only (calib.py default part); devB (rest of dev) = the dev test of any pre-registered rule; SEALED (latest
  30%) untouched. KXNBA1HTOTAL full-dev looked at: 19 cells, only mid .5-.6 gap CI > 0 (+.083 [.004,.153], spread .095 =
  stale wide quotes); YES-at-ask ROI negative everywhere, NO ~0 on longshots. Treated as burned for discovery.

## Discovery (devA) highlights -- descriptive, multiplicity noted in the final table
- 21 series / 257 cells by ~11-19 late: 25 cells with hit-minus-mid CI excluding 0 (~13 expected by chance at 5%), 1 taker-YES
  and 1 taker-NO cell with ROI CI > 0 (~6 each expected one-sided). Most "significant" cells are (a) quotes with spreads .25-.53
  (KXNFLTEAMYDS, KXNFLTEAMSACK: mid meaningless) or (b) whole-ladder LEVEL shifts in a few weeks (KXNBATEAMTOTAL all rungs
  gap ~-.05 = teams scored under; KXNCAAF1HTOTAL all over-rungs +.03..+.08 = 1H scoring above line), not shape.
- One coherent SHAPE pattern, NFL count ladders, upper rungs (2+ and higher) cheap while the 1+ rung is calibrated:
  KXNFLPASSTDS strike 2.5 mid .1-.2: hit .286 vs mid .150 (n49, 36 ev); 3.5 mid .05-.1: .200 vs .073 (n25); 1+ mid .6-.8 .738 vs
  .731 (n61). KXNFLTD strike 1.5 (2+ TDs) mid .1-.2: .260 vs .134 (n50); 1+ rungs calibrated except noisy .4-.6 cells.
  Same direction as KFG1 (3+ FG .218 vs .269). Mechanism hypothesis: Kalshi prices count ladders with too-thin upper tails
  (Poisson-like from the mean) while NFL counts are overdispersed (game script). Alternative: a high-scoring 3-week window.

## PRE-REGISTRATION NFLUP (written 11-19 BEFORE any devB / sealed row of KXNFLTD or KXNFLPASSTDS is joined to outcomes)
- Series: KXNFLPASSTDS (QB N+ passing TDs) and KXNFLTD (player N+ TDs). KXNFLFG excluded (KFG1 owns it).
- Unit: one market (rung). Decision T = start - 60 min (ESPN start, else occurrence-3h; validated 100% on matched events);
  quote = last 1-min candle ending <= T from the event candlesticks; valid = open_time <= T, bid > 0, ask < 1, ask > bid.
- Rule NFLUP: buy 1 YES contract on every valid rung with floor_strike >= 1.5 (2+ and up) and YES ask in [0.05, 0.30], at the
  ask + 10-lot quadratic taker fee (0.07, mult 1, cent-rounded). Hold to settlement. No model.
- Split (mechanical, per series, fixed before outcomes): devA = start < 2026-10-04 17:00Z (discovery, used above, not scored
  for the decision); **devB = 2026-10-04 17:00Z .. 2026-10-27 00:15Z (56 events per series) = the dev test**;
  SEALED = start >= 2026-10-27 00:15Z (45 events) untouched, opened only on lead/user yes; forward = after 2026-11-19.
- Metric: ROI = sum pnl / sum (ask + fee). CI: 10,000-resample bootstrap clustered by GAME (event suffix date+teams, so the
  TD and pass-TD ladders of one game are one cluster). Also report by-week pnl.
- Control (shape vs level, the lead's base-rate placebo): C1 = same rule on the 1+ rungs (floor_strike 0.5) with ask in
  [0.05, 0.30] in the same series/games. If the upper-rung edge is a TAIL-SHAPE error, C1 ~ 0; if it is a high-scoring
  window, C1 also profits. Placebo P2 = rule outcome shuffled across games within strike (200x) is NOT needed for a
  model-free rule; instead report C1 and the per-strike calibration table.
- PASS (-> request sealed): n >= 60 bets AND ROI > 0 with game-clustered 95% CI lower > 0 AND ROI(rule) - ROI(C1) > 0 with
  game-clustered CI lower > 0 (paired by game). KILL: ROI <= 0 or game-clustered CI upper < 0. Else not distinguishable.
- Edgelab: trial nflup_upper_rungs_t60 (family kalshi_sports_pregame), registered before devB is scored.

## NFLUP devB RESULT (results/derived_ladders/nflup_dev.py -> nflup_devB_bets.parquet), edgelab nflup_upper_rungs_t60-4e8e6ef1 (bar p < 2.08e-3, 24 trials)
- devB: 2,545 valid markets, 56 games (2026-10-04 17:00Z .. 10-26 00:20Z). Quote staleness median 264 min (player ladders rarely re-quote).
- NFLUP: n 480 bets, 47 wins, hit .098 vs mean ask .114, ROI -19.3%, game-clustered 95% CI [-41.4%, +5.0%] (p(ROI<=0) .94).
  By strike: PASSTDS 2.5 hit .155 vs ask .169 (n97), 3.5 .054 vs .085 (n74); TD 2+ .101 vs .100 (n258), TD 3+ 0/27.
  By week pnl: -3.66, +0.61, -3.75, -4.43.
- Control C1 (1+ rungs, ask .05-.30): n 1001, ROI -17.1% [-30.7%, -3.8%]. NFLUP - C1: CI [-26.4%, +23.6%].
- **Verdict per prereg: KILL (ROI <= 0).** The devA upper-rung "underpricing" (pass-TD 3+ .286 vs .150) did not replicate in
  the next 3 weeks: it was a high-scoring window (discovery noise), not a tail-shape error. Upper rungs are priced ~fair
  (TD 2+ .101 vs ask .100) and lose the fee + spread at the ask. No sealed request.
- edgelab: nflup_upper_rungs_t60-4e8e6ef1 retired -> dead (reason = devB kill).

## Final discovery tally (devA only; calib_cells.csv, summarize.py; census of quoted series census_quotes.txt)
- 41 series quoted at T-60 (one event-candles call per event); 32 series/pools with >= 1 reportable cell; **339 cells examined**
  (mid buckets + relative rungs). 32 cells with hit-minus-mid 95% CI excluding 0 (~17 expected by chance), 1 cell taker-YES
  ROI CI > 0, 3 cells taker-NO ROI CI > 0 (~8 each expected one-sided). The excess over chance is explained by
  (a) wide-spread series where mid is meaningless (KXNFLTEAMYDS spread .27, KXNFLTEAMSACK .07, NCAAMB 1H .06-.12) and
  (b) whole-ladder level shifts in a 3-5 week window (NBA team totals under, NCAAF 1H overs, NFL team totals over).
- Tight-spread derived ladders (NFL 1H total/spread, OT, win margin, 1HFT, pass yds, NHL 1P total; spread .01-.02) show no shape cell.
- Soccer correct score pooled 7 leagues (69 events devA, 888 valid rungs): mid .0-.1 hit .055 vs mid .052 -> calibrated.
- Soccer corners pooled EPL/LaLiga/UCL (65 events): median rungs hit ~.12 below mid (rel -1: .492 vs .622), taker-NO ROI CI
  crosses 0. Lead only; 4 corner series failed to fetch (empty settled-market listing -> no custom_strike column; not retried).
- KXNCAAMB1HWINNER: 1,159 events but all before the 2026-09-19 cutoff -> no pre-game quotes; untestable until new season data.
- Not pre-registered: NCAAF 1H totals overs (+.03..+.08 gap in devA, taker-YES ROI ~+2% CI crossing 0) -- looks like a
  level shift, too thin after fees; soccer corners (n 65).

## Verdict
No derived-ladder shape error survives. The one coherent shape (NFL upper count rungs cheap, matching KFG1's 3+ FG) was a
discovery-window artefact: NFLUP devB ROI -19.3% [-41.4,+5.0], killed. KFG1's +5% placebo "shape" profit is consistent with
the same window noise. Kalshi derived ladders at T-60 look calibrated within ~+-0.05 and lose the spread+fee at the ask.

## Learnings
### Kalshi pre-game candles exist only after the historical cutoff
Markets settled before /historical/cutoff (market_settled_ts 2026-09-19) return no pre-game candles unless they traded
(KXNBA1HTOTAL-26FEB10LACHOU-98: first candle 52 min after tip); /series/S/events/E/candlesticks returns empty for them.
So every Kalshi derived-ladder backtest has ~2 months of quote history (NFL ~149 games, NBA ~220 events). Source fetch_ev.py logs.
### Event candlesticks endpoint = one call per event for all rungs
/series/{S}/events/{E}/candlesticks returns every market's candles in one call (19 rungs for an NBA 1H total event), ~20x fewer
requests than per-market candles. Source results/derived_ladders/fetch_ev.py.
### Kalshi occurrence_datetime minus 3 h equals the real start
On ESPN-matched events the offset was exactly 0 (std 0) for EPL corners and NFL team totals, 98% within 5 min for NBA 1H totals.
Source starts_*.parquet off_min column.
### A short discovery window turns a scoring level shift into a fake ladder-shape error
devA (NFL weeks 2-4): pass-TD 3+ hit .286 vs mid .150, TD 2+ .260 vs .134; devB (next 3 weeks): TD 2+ .101 vs ask .100, rule
-19.3% [-41.4,+5.0]. Split dev into discovery and test halves; check a same-series control rung (1+) for level shifts.
### Kalshi derived sports ladders are calibrated at T-60
339 cells across 32 series/pools, no taker cell beyond chance; tight NFL ladders (spread .01) calibrated. Source calib_cells.csv.
