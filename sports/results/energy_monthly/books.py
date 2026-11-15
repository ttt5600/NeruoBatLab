"""Orderbooks + trade tape for the open ERCOT / TTF events (read-only)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget  # noqa: E402

OUT = Path(__file__).resolve().parent
d = json.loads((OUT / "census.json").read_text())
res = {}
for s in ["KXERCOTHOUM", "KXERCOTNORTHM", "KXERCOTWESTM", "KXTTFMON"]:
    for e in d[s]["events"]["open"]:
        for m in sorted(e["markets"], key=lambda m: m.get("floor_strike") or 0):
            t = m["ticker"]
            ob = (kget(f"/markets/{t}/orderbook") or {}).get("orderbook") or (kget(f"/markets/{t}/orderbook") or {}).get("orderbook_fp") or {}
            mk = (kget(f"/markets/{t}") or {}).get("market", {})
            tr = (kget("/markets/trades", {"ticker": t, "limit": 1000}) or {}).get("trades", [])
            res[t] = {"ob": ob, "mk": mk, "trades": tr}
            def lv(side):
                v = ob.get(side) or ob.get(side + "_dollars") or []
                return v[-3:] if v else []
            print(t, "yes_bids", lv("yes"), "no_bids", lv("no"), "ntrades", len(tr),
                  "vol", mk.get("volume_fp", mk.get("volume")), "oi", mk.get("open_interest_fp", mk.get("open_interest")),
                  "last", mk.get("last_price_dollars", mk.get("last_price")))
(OUT / "books_2026-11-15.json").write_text(json.dumps(res, default=str))
