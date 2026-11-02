# Kalshi NBA player-prop ladders vs nba_sim -- FORWARD PAPER test (pre-registration)

Written 2026-11-01 ~17:40 PT and committed BEFORE the first logged pick (the log `data/kalshi_prop_log.jsonl` is empty
at the commit that adds this file). Logger: `scripts/kalshi_prop_logger.py`. Paper only: no orders, no keys.

## Question
Kalshi's player-prop ladders (KXNBAPTS / KXNBAREB / KXNBAAST; rung = "Player: k+", YES iff stat >= k) are thin retail
markets. Is the nba_sim player distribution (default model at log time; v2.2 from 2026-11-01) better calibrated than
the Kalshi mids on these ladders, and do the rungs where the two disagree by >= 8c make money after fees?

Prior evidence (descriptive only, not a test): 10-31 slate, 718 rungs, sim v2 - Kalshi mid mean -0.012, sd 0.121;
the large gaps were mostly minutes errors (Edey 13 min, Turner 17, Ware 13). A dry run on TOR@LAL 11-01 (not logged)
flagged 14 rungs, all NO on the top scorers' upper rungs (sim below market on stars). This is the expected failure
mode to watch: the sim regresses stars' projections; whether the market over-prices them is the question.

## What is logged (`tick`)
- Games: every NBA game on today's ESPN slate whose tip is 60-120 min away, ESPN status SCHEDULED, not yet logged.
  Logged once, at the first in-window tick. ESPN status is re-checked right before writing; a game that has tipped
  is never logged.
- Model: `nba_sim` default (fresh ESPN games/box refresh + injury feed + roster), 20,000 sims, seed 7. Players matched
  to Kalshi by normalized name (accents, punctuation, Jr/II/III stripped); unique match required, else sim = null.
  Players on the roster but not in tonight's set get sim = 0 and `absent = true` (excluded from calibration).
- Rungs: ALL two-sided rungs (0 < yes_bid < yes_ask < 1) of the three series: sim prob, bid, ask, mid, last,
  volume_fp, open interest, projected minutes, sim version tag.

## Frozen pick rule (constants at the top of the script; changing them = a new test with a new note)
- YES if sim >= ask + 0.08; NO if sim <= bid - 0.08 (E = 0.08).
- Only rungs with volume_fp >= 100 (V) and players projected >= 20 min.
- 1 contract per pick, taker at the logged quote. Cost YES = ask + 0.07 ask (1-ask); NO = (1-bid) + 0.07 (1-bid) bid.
  (Kalshi rounds fees up to the cent per order; ignored for 1-unit paper, it would only make results worse.)

## Grading (`grade`) and report (`report`)
- Settled from ESPN box scores: y = 1 iff stat >= k. A player who did not play (absent / 0 min) is VOID (Kalshi
  settles inactive players at a fair price).
- ROI = sum(profit) / sum(cost); 95% CI and one-sided p from a game-cluster bootstrap (rungs of one game are
  correlated: same minutes, same pace; the ladder rungs of one player even more so).
- CLV = last Kalshi trade before tip (`/markets/trades?ticker=..&max_ts=<tip epoch>&limit=1`, yes price) minus entry:
  YES: close - ask; NO: bid - close.
- Calibration of ALL non-void logged rungs with a sim price: sim-probability bins vs hit rate, next to Kalshi-mid bins
  vs hit rate on the same rows; Brier and log-loss sim vs mid with a game-cluster CI of the Brier difference. This
  is the key diagnostic: is the sim better calibrated than the mids on these ladders?

## Pass / kill (evaluated at the first `report` with >= 150 settled picks AND >= 20 games)
- PASS: ROI > 0 with one-sided p < .05 AND mean CLV > 0.
- KILL: ROI 95% CI upper < 0, OR mean CLV < 0 with p < .05.
- Otherwise: not distinguishable; keep logging to 300 picks, then the same rule once more; then stop.
- The >= 20 games floor is mine (2026-11-01): 150 picks can come from ~10 games when one star's ladder gives 4-5 picks.
- Secondary (reported, not a pass condition): Brier(sim) - Brier(mid) on all rungs, its CI; a sim that is NOT better
  calibrated than the mids makes any ROI pass suspect (check for lookahead / stale quotes first).

## How to run
```
cd sports
../.venv_quant/bin/python scripts/kalshi_prop_logger.py tick      # every 10-15 min on game days (idempotent per game)
../.venv_quant/bin/python scripts/kalshi_prop_logger.py grade     # next morning (or any time; only FINAL games)
../.venv_quant/bin/python scripts/kalshi_prop_logger.py report
../.venv_quant/bin/python scripts/kalshi_prop_logger.py tick --dry --event <espn id>   # price any game, write nothing
```
Scheduling (the lead's job): a tick every 10-15 minutes from 2 h before the first tip to the last tip covers every game,
because the window is 60 min wide. Each tick costs ~3 s of sim per game + ~6-8 Kalshi requests per game (paced
<= 2 req/s, backoff on 429). If ticks are missed for a whole window, that game is simply not logged (no backfill).
