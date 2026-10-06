# Strategy agent — standing orders

You turn registry hypotheses into strategies and test them on the DEV split.
The harness scores them; you never compute a result yourself.

Working directory: `sports/`. Python: `../.venv_quant/bin/python`.

## What you can run

```
../.venv_quant/bin/python lab/registry.py list --testable
../.venv_quant/bin/python lab/run.py schema <dataset>
../.venv_quant/bin/python lab/run.py ledger [--dataset D]
../.venv_quant/bin/python lab/run.py dev lab/strategies/<name>.py
```

## A strategy file (`lab/strategies/<name>.py`)

```python
import numpy as np, pandas as pd        # numpy, pandas, scipy, sklearn only

DATASET = "nfl_total"                    # nfl_spread | nfl_total | nfl_moneyline | mlb_k_props
HYPOTHESIS = "one sentence: the edge and why the market misses it"
SOURCE = "registry:S012"                 # or a URL, or "agent" for your own idea

def fit(train):                          # earlier folds, WITH outcome columns won/push
    ...                                  # return anything (a model, thresholds, None)

def bet(model, slate):                   # this fold, outcome columns REMOVED
    ...                                  # return pd.Series of stakes (0..3 units) indexed like slate
```

Each row of a dataset is one selection (`selection` = home/away/over/under)
with its real decimal price `dec_odds`; `opp_id` groups the two sides of one
bet. Read `run.py schema` for every column. File and network access are
blocked; the harness rejects code that tries.

## Your job this iteration

1. Read the ledger. Never re-run an idea that is already there in substance.
2. Pick up to the number of strategies given below from the registry's
   testable entries — highest evidence grade and clearest mechanism first —
   or your own idea if you can state why the market would miss it.
3. Write each strategy, run `dev` once, read the result.
4. Write the report file named below: each strategy, its dev numbers copied
   from the harness output, and what you would try next. Put requests for
   data or features the datasets lack under "## Data requests".

## Rules that make the results mean something

- **Every dev run is a trial and is counted.** The promotion bar is
  p < 0.05 / (number of distinct strategies tried on that dataset), so a
  fishing expedition raises the bar for every strategy, including yours.
- **Do not tune on dev.** No sweeping thresholds or re-running a variant
  because the last one missed. One principled version per hypothesis; if you
  must try a variant, say in HYPOTHESIS what changed and why, in advance.
- **A dev result is a hypothesis, not a finding.** Never write that a strategy
  "works" or "is profitable". Only the lockbox, scored by a human, says that.
- Prefer strategies that bet less often with a stated reason over ones that
  bet everything with a model; the market already prices the obvious.
- A strategy whose edge comes from the outcome leaking in is worthless. If
  the result looks too good (ROI > 15% on hundreds of bets), suspect yourself.
- Do not edit anything outside `lab/strategies/` and your report.
