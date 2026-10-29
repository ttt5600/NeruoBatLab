# Protocol — how to backtest without fooling ourselves

1. Freeze first (`scripts/freeze.py`); mine only non-sealed seasons; sealed per league in MANIFEST.
2. Every trial counts. Lab: `lab/run.py dev` appends lab/ledger.jsonl, bar p < 0.05/#trials.
   Prop loop: lab/reports/prop_loop_ledger.jsonl. Pattern mine: BH over all cells + both-halves sign check.
3. Sealed/lockbox: one shot, human "yes" required (`lab/run.py promote --confirm-lockbox`,
   `pattern_mine.py confirm --confirm`, `consensus_props.py sealed --confirm`). Never re-score.
4. Cluster duplicates: same player-game-market at many books/lines = one observation.
5. Evaluate vs the market, not 50%: target = outcome - de-vigged market prob; skewed stats make unders >50% for free.
6. Distrust: EV claims > ~5%, profit from one book/side, anything relying on line moves or "current" timestamps.
7. Kelly: shrink edge first, then 1/4 Kelly. 2% ROI at -110 needs ~14k independent bets to detect.

## 8. Edge math rule (binding from 2026-10-29)
- Every prereg and every finding carries the EDGE_MATH table for its edge, produced by `scripts/edge_math.py bet`
  (price -> implied -> de-vig -> fees -> break-even -> p with Wilson CI -> EV -> Kelly on the LCB -> night/correlation
  cap -> liquidity cap -> bets needed -> CLV). Standard and worked examples: notes/EDGE_MATH.md; why each step exists:
  notes/TRADING_PRINCIPLES.md; situational factors already tested: notes/SITUATIONAL_FACTORS.md.
- Size = k x Kelly on the LOWER confidence bound of p (k = 1/4 default, 1/2 max), then the per-pick (5%), per-night
  (10%, correlated picks = one exposure, across venues too) and liquidity caps. Kelly <= 0 at the LCB = paper only.
- De-vig with power or Shin whenever a longshot is involved; proportional only for near-even two-way markets.
- Social-media / vendor / "system" claims are judged against the ACTUAL obtainable price and the BASE RATE of the
  same bet without the filter, never against 50% or a quoted hit rate.

### Claim check (recipe, ~30 min, no trial budget needed for steps 1-3)
1. Write the claim exactly: market, side, line, price quoted, hit rate claimed, period. Mark anything unverifiable.
2. Price: find the real obtainable price (our odds files / ESPN core / Kalshi / venue API). No price -> say so; a
   claim at a price that does not exist is void (kicker: -200 claimed, lines open at 4.5-9.5, never 3.5).
3. Break-even: `edge_math.py bet --p <claimed> --odds <price>`; then the fair price of the claimed rate
   (`fair_american_at_p`). If the claimed Kelly is enormous, the price is the suspect, not the opportunity.
4. Base rate: the hit rate of the same bet on ALL eligible games (DEV seasons only), and a matched baseline
   (e.g. logistic on the market's own total/spread). Edge = filter rate - matched baseline, with a p-value.
5. Market test: residual vs the de-vigged market probability at the price you could get; ROI at that price.
6. Log ONE ledger trial (`lab/ledger.jsonl`), write notes/<claim>.md with the 12-step table, add a row to
   FUND_BOOK 3d (dead) or 3c (lead). Model write-up: notes/nfl_kicker_system.md.
