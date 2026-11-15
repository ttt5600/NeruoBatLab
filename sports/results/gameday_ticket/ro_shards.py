"""READ-ONLY: which exchange shard each relevant market lives on; recent portfolio transfers/fills around the 11-13 cash drop."""
import sys, json, os
sys.path.insert(0, os.path.expanduser("~/.claude/quant-kalshi/scripts"))
import kalshi_exec as kx

e = kx.env(); cfg = kx.config(e); c = kx.Kalshi(e, cfg["env"])
for ev in ["KXNFLTOTAL-26NOV15BUFNYJ", "KXHIGHDEN-26NOV15", "KXTSAW-26NOV15", "KXNBATOTAL-26NOV15", "KXDHOUSESEATS-27",
           "KXNHLGAME-26NOV15"]:
    code, out = c.request("GET", "/markets", params={"event_ticker": ev, "limit": 3}, auth=False)
    ms = out.get("markets", []) if code == 200 else []
    print(ev, code, [(m.get("ticker"), m.get("exchange_index"), m.get("status")) for m in ms[:2]])
for t in ["KXDHOUSESEATS-27-249", "KXNVDACEO-29JAN-TCOO"]:
    code, out = c.request("GET", f"/markets/{t}", auth=False)
    m = out.get("market", {})
    print(t, code, {k: m.get(k) for k in ("exchange_index", "status", "title")})
code, f = c.request("GET", "/portfolio/fills", params={"limit": 40, "min_ts": 1794600000})
print("fills since 11-13 ~19:20Z:", code)
for x in f.get("fills", []):
    print(" ", x.get("created_time"), x.get("ticker"), x.get("side"), x.get("action"), x.get("count_fp", x.get("count")),
          x.get("yes_price_dollars"), x.get("no_price_dollars"), x.get("is_taker"))
for path in ["/portfolio/transfers", "/portfolio/balance/transfers", "/portfolio/deposits"]:
    code, out = c.request("GET", path, params={"limit": 20})
    print(path, code, json.dumps(out)[:600])
