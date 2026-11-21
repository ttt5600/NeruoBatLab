# props_forward_1121 -- scoring the Kalshi NBA prop-ladder FORWARD paper trial (agent notes, 2026-11-20)

## Question (verbatim from the prereg, notes/kalshi_prop_ladders.md, committed d2898cf 2026-11-01 17:41 PT)
Is the nba_sim player distribution better calibrated than Kalshi mids on KXNBAPTS/REB/AST ladders, and do rungs where
the two disagree by >= 8c make money after fees?

## Rule applied (copied from the prereg BEFORE looking at outcomes; no new variants)
- Picks: YES if sim >= ask + 0.08; NO if sim <= bid - 0.08; volume_fp >= 100; projected minutes >= 20; 1 contract taker.
- Cost YES = ask + 0.07 ask(1-ask); NO = (1-bid) + 0.07 (1-bid) bid (prereg ignores fee rounding; brief asks for the
  rounded-up fee too -> reported as a sensitivity, decision uses the prereg's own cost formula AND I report both).
- Void: player did not play (absent / 0 min).
- ROI = sum(profit)/sum(cost); 95% CI + one-sided p from game-cluster bootstrap.
- CLV = last Kalshi trade before tip minus entry (YES close-ask; NO bid-close).
- Judge at the FIRST report with >= 150 settled picks AND >= 20 games:
  - PASS: ROI > 0 one-sided p < .05 AND mean CLV > 0.
  - KILL: ROI 95% CI upper < 0, OR mean CLV < 0 with p < .05.
  - Otherwise not distinguishable: keep logging to 300 picks, same rule once more; then stop.
- Secondary: Brier(sim) - Brier(mid) on all non-void rungs with a sim price, game-cluster CI.

"Judge at the first report with >= 150 picks & >= 20 games" -> I apply the rule at the chronological point where the
settled pick count first reached >= 150 with >= 20 games (judge-at-n), and also report the full-sample number. If the
first judgement was "not distinguishable", the second look is at >= 300 picks.

## Plan
1. Logger health: rows per day, game coverage vs ESPN slate, log time vs tip (no post-tip quotes), sim version.
2. Re-derive picks from the log with the frozen constants; join grades one-to-one; void handling.
3. Score per prereg (judge-at-n and full); calibration sim vs mid paired Brier/logloss with game-cluster bootstrap.
4. Splits (descriptive): stat x side, stars' upper rungs, fee rounding, CLV, top-5 games.
5. Post-hoc (flagged): minutes errors (DNP/blowout), questionable-status exclusion.
6. edgelab: check whether the registered trial scores; note vault window.

## Log
(filled as I go)

### Descriptive definitions fixed BEFORE computing outcomes (2026-11-20 ~23:20 PT)
- "Star" = player whose market-implied points median (linear interpolation of the logged pts-ladder mids across k,
  where mid crosses 0.5) is >= 20 pts in that game. "Upper rung" = a rung with logged mid < 0.5 (above the market
  median), any stat. These are descriptive splits only, not a test.
- Blowout (post-hoc, descriptive) = final margin >= 20. Minutes error = ESPN minutes - min_proj.
- Questionable (post-hoc) = player named in the game row's q_home / q_away list at log time.
- Judge-at-n operationalised as: order grading batches by graded_ts; the FIRST batch after which settled picks >= 150
  AND games with a settled pick >= 20 is the judgement point (the logger's `report` would have said this then).
- Fee: decision uses the prereg cost (exact 0.07 p(1-p)); rounded-up-to-the-cent fee reported alongside.
- Settlement authority: Kalshi `result` from /markets (read-only) cross-checked against the ESPN-box grades.

### Step 1 -- logger health (results/props_forward_1121/health.py)
- 140 game rows / 19,474 rung rows, 0 duplicate tickers, 1 sim tag (v22) on all games, 0 unmatched names, 0 null sims,
  49 absent rungs. Logged 84.4-119.8 min before ESPN tip (recomputed with Timedelta); 0 rungs with ts >= tip.
- Coverage 11-02..11-20: every ESPN game logged except ORL@PHX + MIA@GS 11-05 (Mac asleep 12:51-19:23 PT) and CHA@ATL
  11-20 (gap 12:33-15:05 PT). 11-01's 4 games tipped before the prereg commit (17:41 PT). Misses are clock-driven
  (sleep), not outcome-driven. 11-03 = no NBA games (election day) per ESPN slate.
- Grades: 18,341 rows, 0 duplicates, 0 orphans; all logged rungs graded except the 8 11-20 games (pending).
- Tick gaps > 25 min inside 08-22 PT exist on 11-03, 11-05, 11-07, 11-09..11, 11-13, 11-14, 11-20 (sleep); window is
  60 min wide so most gaps cost nothing.

### Step 2 -- score per prereg (results/props_forward_1121/score.py -> score.txt, score.json)
- Picks recomputed from logged sim/bid/ask/vol/min_proj with the frozen constants: 1,721 = logged 1,721, 0 mismatches.
  My profit and CLV equal the logger's stored values exactly (max abs diff 0).
- 1,609 picks graded, 47 void (DNP), 1,562 settled non-void in 131 games (11-02..11-19); 112 picks pending (11-20).
- FIRST LOOK (grading batch 11-05 08:00 PT; 115 picks/14 games before it, 283/28 after): ROI -10.37%
  [-23.76, +4.11] p(ROI<=0) .916; CLV -1.42c [-1.94, -0.90], p(CLV>=0) 0.0000 -> KILL (CLV < 0 with p < .05).
- SECOND LOOK would be 11-07 (319 picks): ROI -12.40% [-24.42, +0.87]; CLV -1.46c [-1.96, -0.95] -> KILL.
- FULL (1,562 picks / 131 games): W-L 750-812, -73.91 units on 823.91; ROI -8.97% [-14.83, -2.97] (rounded fee
  -9.65% [-15.47, -3.70]); CLV -0.70c [-0.95, -0.46] -> KILL on both arms.
- Calibration (17,577 rungs / 132 games): Brier sim .1640 vs mid .1568, diff +.0072 [+.0053, +.0091]; log loss .5056 vs
  .4801, diff +.0256 [+.0191, +.0323]. Sim WORSE than mids on every stat (pts +.0083, reb +.0062, ast +.0072, all CIs > 0).
  Reliability: sim bins <0.75 hit ABOVE the sim (sim .024 -> hit .067; .097 -> .140) and the top bin below (.976 -> .917):
  the sim's distributions are too narrow. Mid bins track hit within ~1-2pp.
- Logger's own `report` (deployed copy, read-only) reproduces the full-sample numbers exactly: ROI -0.090
  [-0.148, -0.030], CLV -0.0070 [-0.0095, -0.0046], verdict KILL (it judges at current n, not at the first look).

### Step 3 -- descriptive splits (results/props_forward_1121/splits.py -> splits.txt)
- Mechanism: the sim is biased LOW. All 17,577 calib rows: mean sim .3925 vs mid .4235 vs realised .4247. Gap largest
  mid-ladder (sim-mid -.055 for mid .2-.4, -.064 for .4-.6). Mids are ~unbiased (hit-mid within +-1.2pp by bin).
  Consequence: 1,315 of 1,562 picks are NO, and NO loses: all NO ROI -8.2% [-14.7, -1.5], all YES -13.3% [-25.6, -0.7].
- By stat (both sides): pts -9.1% [-18.5, +0.3] n677; reb -9.1% [-19.1, +0.7] n439; ast -8.7% [-16.7, -0.8] n446.
  CLV negative for every stat x NO cell; YES cells' CLV not distinguishable from 0.
- Prereg's flagged failure mode (NO on stars' upper rungs): NOT where the money was lost. Star (implied pts median >=20)
  upper rungs: sim .194 / mid .232 / hit .212 -> market over-prices by ~2pp, sim under-prices more; NO picks there
  n233 ROI +1.6% [-9.5, +12.1] (CLV -1.28c). The loss is on stars' LOWER rungs (mid >= .5): sim .722 / mid .748 /
  hit .783; NO picks n324 ROI -20.4% [-36.5, -3.1]. Sim is below market on stars everywhere and the market is closer.
- Concentration: top-5 games +5.63/+4.66/+4.55/+4.08/+3.35 units (PHI@CLE 11-06, LAC@PHX 11-10, BOS@PHI 11-10,
  SAC@DAL 11-02, CHI@BKN 11-18); worst-5 -5.67..-4.37. ROI ex best game -9.77% [-15.61, -3.97]; ex worst -8.40%
  [-14.35, -2.45]. Not driven by one game.
- POST-HOC minutes (conditioning on an outcome; not actionable): losers mean (ESPN min - proj) +2.84 vs winners +0.59;
  |err| >= 10 in 12.2% of losers vs 6.9% of winners, mostly players playing MORE than projected (9.7% vs 4.3%).
  NO picks with |err| < 5: ROI -2.3% [-9.4, +4.7] n778 (CLV still -0.58c [-0.85, -0.31]). Minutes projections run
  low for 20-28 min players (+1.81 min) and high for 36+ (-2.99): the v2.2 minutes model is too compressed.
- POST-HOC blowouts (margin >= 20): ROI +3.5% [-9.0, +15.5] n402 vs -13.3% [-19.6, -6.9] n1,160 otherwise (NO picks
  cash when starters sit; can't be known pre-game).
- POST-HOC questionable exclusion: 49 picks on questionable players ROI -31.3% [-54.5, -7.8]; excluding them -8.3%
  [-14.3, -2.2] n1,513. Conclusion unchanged.

### Step 4 -- edgelab (results/props_forward_1121/edgelab_sandbox.txt; registry COPY, real registry sha unchanged)
- Trial kalshi_prop_ladders-1315b2c4 is registered (state `registered`, rule unwired) with a forward adapter.
  `evaluate --split forward` CRASHES on the real code: edgelab/adapters/prop_ladders.py line 69 reads f["result"],
  which vault.load hides (HIDDEN = result,y,stat_value,profit) -> KeyError 'result' (the pre-existing error noted in
  notes/ops_goal_amendment.md). Fix for the lead: `(ok & f["void"].fillna(False).astype(bool))`.
- With that one line patched in memory on a sandbox registry: verdict fail, n 1,609 (47 voids as push), net ROI
  -8.72% [-15.40, -1.81], CLV -0.70c p_negative .0005, spec kill clv_negative_significant TRIGGERED; forward SPRT
  out_of_scope (prices outside [.4,.6]). Vault forward window is [2026-11-01, open] -> no vault.yaml step needed.

### Step 5 -- Kalshi settlement cross-check (results/props_forward_1121/fetch_settle.py, settle_check.txt)
- 23,089 markets fetched for the 420 logged event tickers (all finalized); all 18,341 graded rungs found.
- Non-void: ESPN-box y agrees with Kalshi result on 17,609/17,616; 7 disagree (stat corrections: Kalshi
  expiration_value differs from ESPN box by 1), 1 of them a pick (Coward reb 8+ YES 11-16: ESPN 8 = win, Kalshi NO).
- All 725 ESPN-void rungs settled by Kalshi as `scalar` (fair price, median .45), 0 settled yes/no -> void rule matches
  the rules text ("inactive, or active but never takes the court -> fair price").
- Every graded pick settled at Kalshi's own value (voids at fair price): n 1,609, -76.71 units, ROI -9.05%
  [-14.77, -3.22]. Same conclusion.

### Step 6 -- traps.py checks (results/props_forward_1121/traps_check.txt)
- OK: no lookahead (rung ts < tip on all rows), epoch units, name normalizer injective per game, player join one-to-one
  (1 athlete_id per (game, player) and vice versa), CI reportable, CLV not degenerate.
- TRAP side-imbalance: 239/283 first-look picks were NO (binomial p 1.3e-33); full 1,315/1,562 (scipy p 5.9e-176).
- TRAP model-bias: mean sim - mid = -0.03 over 17,577 rungs (> .01). Both traps would have flagged the problem at the
  first look (and the 11-01 dry run was already "all NO").
- Harness bug: traps.check_pick_side_balance overflows (OverflowError in _binom_two_sided_p, comb * p**i as float) at
  n = 1,562. Use scipy.stats.binomtest or log-space. Not edited (harness owned by the lead/user).
- CLV caveat: the prereg CLV (YES close-ask, NO bid-close) includes the entry half-spread. Mean half-spread at entry:
  0.80c full / 1.25c first look, vs CLV -0.70c / -1.42c. So the CLV arm of the KILL is mostly the spread (any taker
  with no information scores ~ -half-spread); the mid-to-mid drift is in Step 7.

## Verdict (per the prereg, applied literally)
KILL. The first look (grading batch 2026-11-05 08:00 PT, 283 settled picks / 28 games) met the kill arm
"mean CLV < 0 with p < .05": CLV -1.42c [-1.94, -0.90]. ROI there -10.37% [-23.76, +4.11] (not distinguishable on
its own). Every later cut agrees; the full sample (1,562 picks / 131 games) meets BOTH kill arms (ROI CI upper -2.97%).
Secondary: the sim is worse calibrated than the Kalshi mids (Brier +.0072 [+.0053, +.0091]), so no ROI result could
have been trusted anyway. edgelab (sandbox, adapter one-line fix) agrees: fail, spec kill triggered.
Proposals (lead/user, not done by me): stop the props launchd job's `tick` (prereg says stop; it keeps logging 4-15
games/day), or keep it as free data for a de-biased sim -- that would be a NEW prereg. Fix edgelab adapter line 69 and
retire the trial in the real registry. Fix traps.check_pick_side_balance overflow.

### Step 7 -- CLV vs the last pre-tip MID (results/props_forward_1121/fetch_candles.py, clv_mid.py -> clv_mid.txt)
- 1-min Kalshi candles (read-only) at or before ESPN tip for all 1,562 settled picks (100% coverage, 0 candles after tip,
  median gap 0 min).
- Full: prereg CLV (last trade) -0.70c [-0.95, -0.46]; close MID minus entry price -0.20c [-0.46, +0.06] (not
  distinguishable); mid-to-mid drift toward the pick +0.60c [+0.35, +0.85]; entry half-spread 0.80c.
  First look: -1.42c / -0.67c [-1.11, -0.22] / drift +0.58c [+0.19, +0.97] / half-spread 1.25c.
- By side: NO drift +0.75c [+0.45, +1.04] n1,315; YES -0.19c [-0.59, +0.26] n247. The market drifts slightly toward
  the NO picks before tip but by less than the half-spread, and no control (drift of un-picked rungs) was run, so this
  is not evidence of signal. Outcomes still went the market's way (ROI -9%).

## Learnings
### A biased model turns a two-sided disagreement rule into a one-sided bet
nba_sim v2.2 sat 3.1pp below the Kalshi mids on average (mean sim .3925 vs mid .4235 vs realised .4247 over 17,577
rungs), so the ">= 8c disagreement" rule became 84% NO picks (1,315/1,562) and lost -8.97% [-14.83, -2.97]. The
existing traps.check_pick_side_balance / check_residual_bias flag it at the first look (239/283 NO, p 1.3e-33). Run them
before a forward trial starts, on the dry run. Source: results/props_forward_1121/traps_check.txt, splits.txt.
### Player-stat sims that are too narrow look like edge on both tails
Reliability: sim < 5% rungs hit 6.7%, sim > 95% hit 91.7% (score.txt). Minutes projections were compressed (+1.81 min
for 20-28 min players, -2.99 for 36+). A distribution model must pass a reliability check against the market before its
disagreements are treated as picks. Brier sim .1640 vs mid .1568, diff +.0072 [+.0053, +.0091].
### Taker CLV measured against the entry ask/bid contains the half-spread
Prereg CLV = close - ask (YES) / bid - close (NO) scores an uninformed taker at about -half-spread. Here half-spread
0.80c vs CLV -0.70c; the mid-to-mid drift was +0.60c toward the picks. Report CLV as close-mid minus entry-mid next to
the prereg number, or the CLV kill arm fires by construction. Source: results/props_forward_1121/clv_mid.txt.
### The pre-registered "expected failure mode" was not where the money went
Prereg flagged NO on stars' upper rungs. Those picks were ~flat (+1.6% [-9.5, +12.1], n233); stars' LOWER rungs lost
-20.4% [-36.5, -3.1] (n324). The sim was below the market on stars at every rung and the market was closer.
Source: splits.txt [F].
### An edgelab adapter that reads a hidden column crashes every forward evaluation silently for weeks
edgelab/adapters/prop_ladders.py line 69 reads f["result"], which vault.load hides -> KeyError, so the trial had no
edgelab forward verdict from 11-02 to 11-20. Adapters must only read non-hidden columns; add a selftest that runs each
forward adapter on its live data. Source: results/props_forward_1121/edgelab_sandbox.txt.
### ESPN box grading matched Kalshi settlement on 99.96% of rungs
17,609/17,616 non-void rungs agreed; 7 stat-correction disagreements (1 pick). All 725 DNP rungs settled at a Kalshi
fair price, matching the void rule. Source: settle_check.txt.
