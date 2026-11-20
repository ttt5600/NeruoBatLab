"""Trade tape for DEV KXTSAD markets with lifetime volume > 0. Read-only."""
import json, sys
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[1] / "scripts"))
from kx_throttle import kget
t = pd.read_parquet(OUT / "tsa_dev_quotes.parquet"); t = t[t.vol > 0]
res = {}
for tk in t.ticker:
    tr, cur = [], None
    while True:
        p = {"ticker": tk, "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets/trades", p) or {}
        tr += d.get("trades", []); cur = d.get("cursor")
        if not cur or not d.get("trades"): break
    res[tk] = tr
json.dump(res, open(OUT / "tsa_trades_dev.json", "w"))
print("markets", len(res), "trades", sum(map(len, res.values())))
