# Experiment budget (user 2026-11-21: "$50 experiment budget" + "just try it be aggressive")
Real money on unproven leads, to MEASURE them. Hard cap: total cost of all experiment orders <= $50 (max loss = spend).
Journal: ~/.claude/quant-kalshi/data/exp_budget.jsonl. Status: `cd ~/.claude/quant-kalshi && .venv/bin/python scripts/exp_exec.py status`.
Stop everything: `touch ~/.claude/quant-kalshi/data/EXP_STOP` (also stops on LEAD_KILL / kalshi KILL).

| date | lead | order | cost | thesis | settles |
|---|---|---|---|---|---|
| 11-21 05:45Z | OpenRouter BYOK definition (notes/ai_nowcast.md) | KXTOKENUSE-26NOV23-T238 YES 180 @ .0735 (+.0047 fee), IOC | $14.08 | all 7 past weeks settled BYOK-inclusive; incl. view already 254.8T > 238T; book prices the excluded view (~7%) | 11-23 10:00 ET |
| from 11-22 | KFG1 field-goal ladders (notes/kalshi_fg_markets.md) | $2 IOC per KFG1 'bet' row at the logged ask, via scripts/exp_exec.py in the kfg1 launchd job | <= $35.92 left | dev +23% [-0.7,+47.3] n56 (not distinguishable); measuring live fills vs paper | per game |
