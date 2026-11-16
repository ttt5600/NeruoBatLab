"""Print the last manual (forced) run's quote rows with the new CB2/CB3 columns (no outcomes exist)."""
import json
from pathlib import Path

import pandas as pd

D = Path.home() / ".claude/quant-sports/data/kx_forward"
q = pd.DataFrame([json.loads(l) for l in (D / "crypto_barrier_quotes_manual.jsonl").read_text().splitlines()])
last = q[q.quote_time >= q.quote_time.max()[:13]]          # rows of the latest hour
last = last[last.decided_at >= sorted(last.decided_at.unique())[0]]
x = last[last.trial & last.p2.notna()].copy()
x["mid"] = (x.yes_bid + x.yes_ask) / 2
pd.set_option("display.width", 220)
print(x[["ticker", "barrier", "strike", "spot", "tau_d", "yes_bid", "yes_ask", "mid", "p1", "p2", "ps", "sigma_b",
         "fires", "fires_cb2", "fires_cb3"]].round(4).to_string(index=False))
print("cols present:", [c for c in ("ps", "sigma_b", "fires_cb2", "fires_cb3", "cb2_error") if c in last])
print("cb2_error rows:", int(last.get("cb2_error", pd.Series(dtype=object)).notna().sum()))
print((D / "crypto_barrier_v2_manual.jsonl").read_text()[-800:])
