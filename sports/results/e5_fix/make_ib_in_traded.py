"""Picks for events 11-10/11-11 whose side contract printed per the FX daily files (input for ib_ticks.py)."""
from pathlib import Path

import pandas as pd

D = Path(__file__).parent
FX = Path.home() / ".claude/quant-kalshi/data/forecastex_paper_files"
p = pd.read_csv(D / "ib_in_recent.csv")
keep = set()
for f in ["20261110", "20261111"]:
    t = pd.read_csv(FX / f"uh_{f}.csv")
    t = t[t.pair_quantity > 0]
    keep.update(zip(t.event_contract, t.subtype))
m = [(a, b) in keep for a, b in zip(p.ticker, p.side)]
q = p[pd.Series(m, index=p.index) & (p.event_date >= "2026-11-10")]
q.to_csv(D / "ib_in_traded.csv", index=False)
print(len(q))
