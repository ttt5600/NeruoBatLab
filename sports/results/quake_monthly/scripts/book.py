"""Read-only: current orderbooks, full trade tape and hourly candles for KXBIGGESTQUAKEMONTH. No orders."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from kx_throttle import kget

OUT = Path(__file__).resolve().parents[1]
raw = json.load(open(OUT / "census_raw.json"))
tick = [m["ticker"] for m in raw["markets"]["open"]]
res = {"pulled": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "books": {}, "trades": {}, "mk": {}}
for t in tick:
    res["mk"][t] = kget(f"/markets/{t}")["market"]
    res["books"][t] = kget(f"/markets/{t}/orderbook", {"depth": 10})
    tr, cur = [], None
    while True:
        p = {"ticker": t, "limit": 1000}
        if cur: p["cursor"] = cur
        d = kget("/markets/trades", p)
        tr += d.get("trades", [])
        cur = d.get("cursor")
        if not cur: break
    res["trades"][t] = tr
json.dump(res, open(OUT / "book_snapshot.json", "w"), indent=1)
for t in tick:
    m = res["mk"][t]
    print(t, "bid", m["yes_bid_dollars"], m.get("yes_bid_size_fp"), "ask", m["yes_ask_dollars"], m.get("yes_ask_size_fp"),
          "vol", m["volume_fp"], "ntrades", len(res["trades"][t]), "updated", m.get("updated_time"))
    print("   book", json.dumps(res["books"][t])[:300])
    for x in res["trades"][t][:8]:
        print("   tr", x.get("created_time"), x.get("taker_side"), x.get("yes_price_dollars"), x.get("count_fp"))
