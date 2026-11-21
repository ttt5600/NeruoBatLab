# Experiment budget (user 2026-11-21: "$50 experiment budget" + "just try it be aggressive")
Real money on unproven leads, to MEASURE them. Hard cap: total cost of all experiment orders <= $50 (max loss = spend).
Journal: ~/.claude/quant-kalshi/data/exp_budget.jsonl. Status: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/exp_exec.py status`.
Stop everything: `touch ~/.claude/quant-kalshi/data/EXP_STOP` (also stops on LEAD_KILL / kalshi KILL).

| date | lead | order | cost | thesis | settles |
|---|---|---|---|---|---|
| 11-21 05:45Z | OpenRouter BYOK definition (notes/ai_nowcast.md) | KXTOKENUSE-26NOV23-T238 YES 180 @ .0735 (+.0047 fee), IOC | $14.08 | all 7 past weeks settled BYOK-inclusive; incl. view already 254.8T > 238T; book prices the excluded view (~7%) | 11-23 10:00 ET |
| from 11-22 | KFG1 field-goal ladders (notes/kalshi_fg_markets.md) | $2 IOC per KFG1 'bet' row at the logged ask, via scripts/exp_exec.py in the kfg1 launchd job | <= $35.92 left | dev +23% [-0.7,+47.3] n56 (not distinguishable); measuring live fills vs paper | per game |

## Sizing check (user 11-21: "adjust based on our confidence and kelly interval")
OpenRouter T238: cost .0782/ct incl fee. Full Kelly on $552 Kalshi balance across my 10-20% estimate: p .07 (= market) $0;
p .10 $13.05; p .15 $43.00; p .20 $72.94 (quarter Kelly $3.26 / $10.75 / $18.23). Rule = quarter Kelly at the interval's
LOWER bound (STRATEGY_v2) -> $3.26; position $14.08 already >= full Kelly at the lower bound -> NO add.
Memory feedback_longshot_kelly_stale_model: Kelly-sizing 1-9c contracts off our own estimate lost ~$150 once.
KFG1: dev CI lower bound < 0 -> Kelly 0; $2/bet is a measurement stake, not a Kelly size.
