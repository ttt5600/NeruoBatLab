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
