# nba_l32b_exec -- can L32b be executed on Kalshi? (agent, 2026-11-11)

## Question
L32b (notes/l32b_debiased_totals.md) beats the Pinnacle OPENER (dev +8.1% no-vig p .006; forward 31 settled, CLV +0.82
pts [-0.10, +1.75] per lead brief). Our only US venue for NBA totals is Kalshi KXNBATOTAL. Q004 (raw model vs the first
tight Kalshi anchor strike) is near-kill (NBA -12.7%/48 per lead brief).
1. When do KXNBATOTAL ladders list / first go tight vs the Pinnacle opener; is Kalshi's first price nearer open or close?
2. Re-price the L32b forward picks at the actual Kalshi ask+fee available at pick time.
3. Why does Q004 lose while L32b shows CLV?
4. Power: picks needed for CLV CI to exclude 0; pick rate.

## Disclosed before pre-registering
- Known from the brief: forward 31 settled 17-14, CLV +0.82, no-vig +9.3%. NOT yet looked at: any Kalshi price for an
  L32b pick, any per-pick Kalshi P&L.
- Code read: Q004 pick_tight (~/.claude/quant-kalshi/scripts/nba_open_totals.py) uses the RAW model (no c level
  correction) vs the anchor strike. L32b subtracts c (~+2.7 at launch). That is a structural difference (Q3 hypothesis).

## Pre-registration (written before computing any Kalshi-priced L32b outcome)
Trial K1 `l32b_kalshi_taker` (family nba_totals):
- Universe: L32b "live" decisions (data in ~/.claude/quant-sports/data/l32b_state.json) with a pick.
- Decision time T = the L32b decision row's t (>= 30 min before tip). Kalshi quote = last 1-min candlestick
  (GET /series/KXNBATOTAL/markets/{ticker}/candlesticks, public) ending <= T, yes_ask / yes_bid close; if no candle at
  or before T within 60 min -> "no quote" (counted, not bet).
- STRIKE: Kalshi strike S nearest to the Pinnacle OPEN line (ties -> the strike more favorable to our side, i.e. lower
  for over, higher for under). Bet only if the de-biased signal still holds at S: over needs model - c - S >= 3.0,
  under needs model - c - S <= -3.0 (same threshold as L32b, now measured vs the strike we actually buy).
- PRICE: over = YES ask, under = NO ask (= 1 - yes_bid) + exact Kalshi taker fee (100-ct order). Graded on final vs S.
- ROI per bet = (1-ask-fee)/(ask+fee) win, -1 loss. Bootstrap by game, 4,000 resamples, seed 0.
- DEV = every L32b live pick logged before 2026-11-11 (this file's date). Dev is descriptive + a go/no-go for forward:
  GO forward paper if dev mean ROI > 0 AND n >= 15 priced bets; KILL (no forward) if dev mean ROI <= -10% with
  n >= 20. Anything between: forward paper anyway (cheap) but label "needs data".
- FORWARD judge (from 2026-11-12 on): after 100 settled, ROI@ask+fee > 0 with one-sided p < .10 AND mean CLV
  (Pinnacle close - S signed) > 0 -> sealed/real-money ask to the user. Kill if ROI CI upper < 0 after 60 or mean CLV
  <= 0 after 100.
- Descriptive (not judged): maker variant (rest at the bid+1c, fill if a later candle's ask <= our price before tip).

## Results

### Q1 listing timing (results/nba_l32b_exec/q1_timing.py -> q1_timing.csv; L32 state + open_totals_tight.json)
- 114 games (not pre-existing) with a Pinnacle first sighting AND a KXNBATOTAL event.
- Pinnacle first sighting (10-min polling, "open") is a median 21.8 h before tip (p10 18.1, p90 24.2).
- Kalshi open_time is BEFORE our Pinnacle first sighting in 95% of games: median 3.4 h earlier (p10 11.9 h, p90 0.6 h).
- First tight anchor (<= 4c): uncensored n 91, median 1.5 h BEFORE the Pinnacle sighting; incl. censored, 84% tight at
  or before it. First two-sided spread median 16.5c, first-tight spread median 3.0c.
- Kalshi implied total at the SAME tick as the Pinnacle open (108 of 114 games): |Kalshi - Pin open| 0.33 pts vs
  |Kalshi - Pin close| 1.82; difference -1.50 [-1.79, -1.23]; closer to open in 80% (ties 14%). Kalshi - Pin open mean
  -0.04. So Kalshi at open time IS the opener: it tracks Pinnacle's open, not its close.

### Q3 why Q004 loses (results/nba_l32b_exec/q3_q004.py -> q3_q004_rows.csv)
- Q004 pick_tight uses the RAW model vs the anchor strike: mean model - strike +3.85 (sd 3.27, 91 decisions) -> 47 of
  48 settled picks are OVERS. Settled 48: ROI@ask+fee -12.7% [-39.9, +14.6], CLV -0.27 pts.
- Strike vs Pinnacle open at those ticks: mean +0.02, |.| 0.62 -> timing/pricing is not the problem.
- Split by the L32b de-bias (c = trailing-100 mean(model - open), same as the logger): picks that ALSO pass
  |model - c - S| >= 3 (n 18): CLV +1.11, ROI -4.6% [-47.7, +39.0]; picks de-biasing would SKIP (n 24): CLV -1.44,
  ROI -36.0% [-69.0, +3.6]. (6 early picks had no c yet.) The loss lives in the level-biased overs.
- Answer: different model (no level correction), not timing, not Kalshi pricing. Small n; descriptive.

### Q4 power / pick rate (L32b state, live picks)
- 30 live picks in 9 days (11-02 .. 11-11) = ~23/week (38% of ~8.8 decided games/day). CLV settled n 28:
  mean +0.79 pts, sd 2.69, se 0.51.
- At the current point estimate: ~46 picks for a two-sided 95% CI to exclude 0 (~32 one-sided) -> ~1-2 more weeks.
  If the true effect is half (+0.4), ~180 picks (~8 weeks). The L32b judge (200 settled) is ~8-9 weeks out.

### Q2 K1 dev (registered before scoring: edgelab l32b_k1_kalshi_taker-8d6fb950, family nba_totals 31 trials, bar p < 1.61e-03)
Script: results/nba_l32b_exec/q2_fetch.py (public candlesticks, cached per game) + q2_score.py dev -> q2_k1_rows_dev.csv,
q2_k1_summary_dev.json. 28 L32b live picks before 11-11, all settled; 27 bet, 1 skipped (gap 2.79 < 3 at S).
(Wrong turn: first fetch died on HTTP 429 and `| tail` hid the exit code; re-ran with throttle + per-game cache.)

| Variant | n | W | ROI @ ask+fee [95%] | p | CLV vs S pts [95%] | mean cost |
|---|---|---|---|---|---|---|
| **K1 primary** (L32b tick, recheck gap at S) | 27 | 15 | **+1.9% [-32.6, +36.3]** | .433 | +1.37 [+0.35, +2.43] | 0.543 |
| overs | 20 | 10 | -7.0% [-45.2, +31.8] | .60 | +1.07 [-0.23, +2.45] | 0.538 |
| unders | 7 | 5 | +27.4% [-27.0, +79.6] | .14 | +2.21 [+1.00, +3.43] | 0.557 |
| no recheck, nearest strike | 28 | 16 | +4.9% [-28.6, +37.7] | .41 | +1.18 [+0.11, +2.23] | 0.543 |
| at L32's decision tick (= open time) | 28 | 16 | +5.5% [-28.0, +38.5] | .41 | +1.18 [+0.11, +2.23] | 0.539 |
| reference: same 28 picks at Pinnacle no-vig open | 28 | | +13.9% | | +0.79 (vs open) | |

- Rows inspected (all 28): quotes 0.1-23 min old (median 2.3), spreads 1-3c, prices 0.45-0.61, S within 1 pt of the open.
- Friction: fee ~1.75c + half-spread ~0.85c ~= 2.6c -> break-even ~52.6% (~ -111 American); Kalshi taker costs about
  what -110 costs. L32b dev at -110 was +4.8% [-1.2, +11.1] -> expect low single digits, not +8%.
- Pre-registered go/no-go: dev ROI > 0 with n >= 15 -> GO forward paper. ROI is NOT distinguishable from 0.
- Maker variant (pre-reg descriptive): not computed -- candles cannot show queue position; E5/L35 showed maker fills
  are adversely selected, so a candle-touch fill model would flatter it.
- L32 decision-tick depth on the L32 side within 1c: median $4,723, p10 $3,110 (n 10) -> size is not the constraint.

## Verdict
EXECUTABLE in mechanics (Kalshi lists ~3.4 h before the Pinnacle open, tight at ~3c, prices the opener not the close,
thousands of $ deep). Profitability NOT shown: +1.9% [-32.6, +36.3] on 27. NEEDS MORE DATA -> forward paper K1.

## Forward-paper rule K1 (from 2026-11-12; no orders)
For every L32b "live" pick (l32b_logger tick >= 30 min before tip): S = KXNBATOTAL strike nearest the Pinnacle open
(tie: lower for over / higher for under); skip unless model - c - S clears +/-3.0; price = last 1-min candle ending
<= the L32b tick (YES ask for over, 1 - yes_bid for under) + exact taker fee (100 ct); grade final vs S.
Run: `q2_fetch.py` then `q2_score.py forward` (candles are public and historical, so after-the-fact pricing uses only
pre-decision quotes). Judge after 100 settled (~4-5 weeks at ~23 picks/wk): ROI > 0 one-sided p < .10 AND CLV vs S
> 0 -> sealed/real-money ask; kill if ROI CI upper < 0 after 60 or CLV <= 0 after 100. Better: lead wires the same
quote into l32b_logger at the tick (live orderbook) -- a logger change, so the lead's call. edgelab rule is
`forward:unwired` (fails closed) until an adapter reads q2_k1_rows_forward.csv.

## Learnings

### Kalshi NBA totals list before the Pinnacle opener and price the opener
KXNBATOTAL open_time precedes our first Pinnacle sighting in 95% of 114 games (median 3.4 h earlier); at the
Pinnacle-open tick the Kalshi implied total is 0.33 pts from the open vs 1.82 from the close (diff -1.50 [-1.79,-1.23]).
Source results/nba_l32b_exec/q1_timing.py. Matters: "the edge is vs the opener so we can't bet it" was wrong for NBA.

### Q004 NBA loses because the model is not de-biased, not because of timing
Raw model - Kalshi strike +3.85 mean -> 47/48 Q004 picks were overs. Picks the c-correction keeps: CLV +1.11 (n 18);
picks it drops: CLV -1.44, ROI -36% (n 24). Source results/nba_l32b_exec/q3_q004.py. Matters: a level-biased model
turns any timing edge into a one-sided over bet.

### Kalshi taker on NBA totals costs about -110
Fee ~1.75c + half-spread ~0.85c at 50c -> break-even ~52.6%. Judge Kalshi-executed totals edges against the -110
dev number (L32b +4.8% [-1.2, +11.1]), not the no-vig one (+8.1%). Source q2_k1_rows_dev.csv.

### L32b executed on Kalshi: +1.9% [-32.6, +36.3] on 27 dev bets
CLV vs the bought strike +1.37 [+0.35, +2.43], but ROI not distinguishable from 0; forward K1 paper from 11-12.
Source q2_k1_summary_dev.json; edgelab l32b_k1_kalshi_taker-8d6fb950.

### A pipe into tail hides a crashed fetch
`python fetch.py | tail` exited 0 after an HTTP 429 traceback; the output file never existed. Check outputs, not exit codes.
