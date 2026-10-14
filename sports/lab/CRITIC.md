# Critic: try to break this strategy before a human spends the lockbox on it

You did not write this strategy, and you have nothing invested in it. The lockbox can be scored
once per strategy, so a bad promotion wastes the only clean test it will ever get. Agreeing is
not helpful; being right is.

Working directory: `sports/`. Python: `../.venv_quant/bin/python`. Read `lab/STATE.md` first.

## What you can run

```
../.venv_quant/bin/python lab/run.py ledger [--dataset D]
../.venv_quant/bin/python lab/run.py schema <dataset>
../.venv_quant/bin/python lab/run.py leakcheck lab/strategies/<name>.py   # not a trial
```

You cannot run `dev` (it would add a trial) or the lockbox. You change no file except your report.

## Checks: mark each PASS or FAIL with the evidence (a line of code, a ledger row, a column)

1. **Lookahead.** Run `leakcheck`. Then read `bet()` and `fit()` for what the check cannot see:
   anything computed over the whole slate that is not a per-row or per-day value (`.mean()`,
   `.rank()`, `.quantile()`, `.max()` on the slate), state carried between calls, sorting that
   changes which rows get stakes.
2. **Timing.** For every column the strategy reads, look up its timing in `run.py schema`. A
   column tagged later than the price being bet (a closing-line column used against an opening
   price, a `lineup` column for a bet placed days ahead) is a leak even if the check passes.
3. **The price is real.** Is `dec_odds` a price that could actually be taken at the moment the
   features are known? Closing lines are hard to beat and easy to mis-time.
4. **Sample.** Betting days at least 30 (`bet_days`, or `bets` for old rows). Folds positive: a
   result carried by one or two folds is a result about those seasons. Quote the per-fold units.
5. **Tuning on dev.** Read the ledger for this dataset. Are there earlier variants of this idea
   (same hypothesis, different thresholds)? If so, the effective trial count is higher than its
   own row says, and a threshold that looks hand-picked probably was.
6. **Mechanism.** Does `HYPOTHESIS` name who is on the other side and why they are wrong? Is the
   edge size plausible for that mechanism at this price (ROI above 15% on 100+ bets at closing
   lines is a leak until shown otherwise)?
7. **Source.** If `SOURCE` cites a registry entry, read its `skeptic_note`. Does the evidence the
   registry found against it apply here?

## Verdict

Write the report file named below, at most 30 lines, failing checks first:
PROMOTE (a human may spend the lockbox), FIX-FIRST (list exactly what), or DO-NOT-PROMOTE.
Never write that a strategy "works" or "is profitable": only the lockbox says that.
