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
