"""Read-only: trade tape + markets + orderbooks for KXTSAW 26NOV08 / 26NOV15 / 26NOV22 (tsa_fill_review agent)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget
out = {}
for ev in ["KXTSAW-26NOV08", "KXTSAW-26NOV15", "KXTSAW-26NOV22"]:
    ms = (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])
    out[ev] = {"markets": ms, "trades": {}, "books": {}}
    for m in ms:
        tk = m["ticker"]; tr = []; cur = None
        while True:
            p = {"ticker": tk, "limit": 1000}
            if cur: p["cursor"] = cur
            d = kget("/markets/trades", p) or {}
            tr += d.get("trades", []); cur = d.get("cursor")
            if not cur: break
        out[ev]["trades"][tk] = tr
        if m.get("status") in ("active", "open"):
            out[ev]["books"][tk] = (kget(f"/markets/{tk}/orderbook", {"depth": 10}) or {})
    print(ev, len(ms), sum(len(v) for v in out[ev]["trades"].values()))
json.dump(out, open(HERE / "tape.json", "w"))
