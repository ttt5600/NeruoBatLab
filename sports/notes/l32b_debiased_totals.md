# L32b: de-biased NBA opening totals (pre-registration, 2026-11-01)

Written and committed BEFORE any L32b P&L was computed. Script: `scripts/l32b_debiased_totals.py` (dev scorer),
`scripts/l32b_logger.py` (forward paper logger, only built if dev passes). Ledger dataset `l32b_debiased_totals`.

## Why
L32 (forward paper, `~/.claude/quant-kalshi/scripts/nba_open_totals.py`, frozen; not modified) picks ~all OVERS:
model - first Pinnacle open = +2.90 mean (n 54, `report` 2026-11-01). Question: is that a level bias to remove?

## What I already looked at before writing this (disclosed)
- Descriptive only, no P&L of any de-biased rule:
  - dev `wf.pkl` + `markets()`: mean(T2_strict - opener) = +3.48 (2023-24), +1.41 (2024-25); weeks 1-3 of each
    season +2.9..+3.6 / -0.1..+2.9. No decay over the season.
  - 2026-27 recompute (`nba_simple_live.predict(d, refresh=False)`, 90 finals): model - DK open -0.26 overall;
    week of 10-20 -1.39, week of 10-27 +1.37. On L32's 42 settled-window finals: model - Pinnacle open +2.75,
    Pinnacle first sighting - DK open -1.02, all lines ~5 pts above the actual totals (se ~3.3).
- Already known from notes/nba_simple_model.md: T5 split overs +6.9% (932) / unders +15.3% (258).

## Rule (fixed a priori)
- MODEL = T2_strict (dev: walk-forward `wf.pkl`, exactly the T5 predictions). OPEN = T5's opener (DraftKings open, else
  ESPN BET open; `nba_simple_model.markets()`), priced at its proportional no-vig open price (`pf_over_open`).
- LEVEL CORRECTION c(g) = mean of (MODEL - OPEN) over the most recent **N = 100** games of the SAME season that have both
  MODEL and OPEN and whose ET date is **strictly earlier** than g's ET date. Requires **>= 30** such games, else no bet
  (game skipped). No cross-season carry-over (forward cannot see 2025-26, which is sealed).
- PICK: adj = MODEL - c - OPEN. OVER if adj >= +3.0, UNDER if adj <= -3.0. Flat 1u, one bet per game.
- GRADED vs the OPEN line (push = 0 on exact integer).

## Data / split
- DEV only: scored games 2023-24 and 2024-25 (seasons 2024, 2025 in the code) = T5's games. 2025-26 is sealed (the
  L32 sealed shot was already spent); 2026-27 is forward. Neither is read by the dev scorer.

## Test
- PRIMARY: ROI at the no-vig open price; game bootstrap 4,000 resamples, seed 0, one-sided p = P(resampled mean <= 0).
- PASS = p < .05 AND ROI > 0 AND both seasons (halves) ROI > 0.
- Also reported (descriptive): ROI at -110 with 2.5/97.5; win rate; n overs / n unders (expect roughly balanced);
  CLV vs same-book close; the ORIGINAL T5 rule on the same games (games with c available) for comparison.
- Too-good guard: ROI > +10% on > 200 bets -> inspect rows before reporting.
- Kill / no-forward: anything short of PASS -> no logger, L32b DEAD on dev.
- Power: ~1,000 bets at sd ~1u -> SE ~3%; at a true +4% power ~ 0.5. A miss is weak evidence, not a refutation.

## If PASS: forward paper (L32b logger)
- Inputs: L32's own state file (`~/.claude/quant-kalshi/data/open_totals_state.json`, read-only): per game the first
  Pinnacle open line/prices and the decision-tick MODEL (`decision.model`). c(g) from L32's decided games with both
  model and open on strictly earlier ET dates, last 100, >= 30. Pick |MODEL - c - OPEN| >= 3; priced at the open
  no-vig and -110; graded vs the open; CLV vs last pre-tip Pinnacle line. One pick per game, decided once, when L32's
  decision exists and >= 30 min before tip. Paper only.
- JUDGE: same as L32 (after 200 settled: CLV > 0 one-sided p < .05; kill if mean CLV <= 0 after 100).

## Results
(appended after scoring; see below this line)

### Dev result (2026-11-01, after prereg commit 1cd5cf5; `results/l32b_debiased_totals/dev.json`, `dev_bets.csv`)
| Rule (same 2,209 games with c) | Bets | Over / under | Win | No-vig ROI [2.5, 97.5] | p | -110 ROI [2.5, 97.5] | CLV pts | 2023-24 / 2024-25 no-vig |
|---|---|---|---|---|---|---|---|---|
| **L32b de-biased** | 967 | 473 / 494 | .549 | **+8.1% [+1.9, +14.4]** | .006 | +4.8% [-1.2, +11.1] | +2.39 | +9.3% / +6.8% |
| T5 original (comparison) | 1,140 | 898 / 242 | .548 | +8.0% [+2.3, +13.8] | .003 | +4.6% [-0.9, +10.3] | +1.84 | +8.5% / +7.3% |

- **PASS** on the pre-registered bar (p .006 < .05, both seasons > 0). Too-good guard not triggered (+8.1% < 10%).
- Not new evidence of an edge: same games, same model, same opener as T5 (which was itself the selected trial).
  What it shows is that the edge does NOT depend on the over tilt: balanced 473/494, unders +8.7%, overs +7.4%,
  and CLV per bet is higher (+2.39 vs +1.84).
- At -110 neither rule clears zero (p .057 / .052). The bar is the no-vig open (a maker order at fair).
- Row checks: c recomputed independently for 4 random bets (exact match, 100 prior games each); no duplicate events.
  Something's off, inherited from T5: 39 bets have an opener price outside 40/60 (as extreme as 24/76, juiced
  stale openers). They lose -3.9u; excluding them is +8.8% (post-hoc, not the test).
- c (dev): 2023-24 mean +3.26 (range -0.5 .. +8.7, rising into April); 2024-25 +1.39 (-0.5 .. +2.6).

### Forward: `scripts/l32b_logger.py` built (reads L32 state read-only; logs data/l32b_log.jsonl)
- Run every 10 min from a session shell (launchd cannot read ~/Desktop):
  `cd .../quant-trader/sports && while true; do ../.venv_quant/bin/python scripts/l32b_logger.py tick; sleep 600; done`
- Needs the L32 launchd logger running (it supplies open, model, close). First tick 2026-11-02 01:37Z: 21 decisions
  (11 live, 10 backfill), c = +2.67 on 44 games.
