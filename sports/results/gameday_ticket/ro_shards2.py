"""READ-ONLY: shard of the BTC/NBA markets traded 11-13/14 and of today's game series."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/.claude/quant-kalshi/scripts"))
import kalshi_exec as kx
e = kx.env(); cfg = kx.config(e); c = kx.Kalshi(e, cfg["env"])
for t in ["KXBTC15M-26NOV131700-00", "KXNBAGAME-26NOV14DENNOP-DEN", "KXNBAPTS-26NOV14DENNOP-DENNJOKIC15-25"]:
    code, out = c.request("GET", f"/markets/{t}", auth=False); m = out.get("market", {})
    print(t, code, m.get("exchange_index"), m.get("status"), m.get("result"))
for s in ["KXNHLGAME", "KXNBAGAME", "KXNBATOTAL", "KXNFLGAME", "KXNFLTOTAL"]:
    code, out = c.request("GET", "/markets", params={"series_ticker": s, "status": "open", "limit": 200}, auth=False)
    ms = out.get("markets", [])
    evs = sorted({m["event_ticker"] for m in ms})
    print(s, len(ms), sorted({m.get("exchange_index") for m in ms}), evs[:20])
