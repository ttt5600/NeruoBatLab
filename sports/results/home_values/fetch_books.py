"""Read-only: orderbook + public trades for every *HOME market (results/home_values/markets.json)."""
import json, sys, time
from pathlib import Path
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1] / "scripts"))
from kx_throttle import kget

m = json.load(open(H / "markets.json"))
books, trades = {}, []
for x in m:
    t = x["ticker"]
    books[t] = kget(f"/markets/{t}/orderbook")
    cur = None
    while True:
        p = {"ticker": t, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = kget("/markets/trades", p)
        trades += r.get("trades", [])
        cur = r.get("cursor")
        if not cur:
            break
json.dump({"ts": time.time(), "books": books}, open(H / "books.json", "w"))
json.dump(trades, open(H / "trades.json", "w"))
print(len(books), len(trades))
