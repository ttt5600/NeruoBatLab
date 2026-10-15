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
