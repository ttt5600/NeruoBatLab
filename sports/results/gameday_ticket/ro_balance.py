"""READ-ONLY: Kalshi balance + open positions + resting orders (GET only). Run with the deployed venv."""
import sys, json, os
sys.path.insert(0, os.path.expanduser("~/.claude/quant-kalshi/scripts"))
import kalshi_exec as kx

e = kx.env(); cfg = kx.config(e); c = kx.Kalshi(e, cfg["env"])
code, b = c.request("GET", "/portfolio/balance"); print(code, json.dumps(b)[:800])
code, p = c.request("GET", "/portfolio/positions", params={"limit": 200, "count_filter": "position"})
mp = p.get("market_positions", []); print("n positions", len(mp))
tot = 0
for m in sorted(mp, key=lambda m: -float(m.get("market_exposure_dollars") or 0)):
    ex = float(m.get("market_exposure_dollars") or 0); tot += ex
    if ex > 2:
        print(" ", m.get("ticker"), m.get("position_fp", m.get("position")), ex)
print("total exposure", round(tot, 2))
code, r = c.request("GET", "/portfolio/orders", params={"status": "resting", "limit": 200})
os_ = r.get("orders", []); print("resting orders", len(os_))
for o in os_[:30]:
    print(" ", o.get("ticker"), o.get("side"), o.get("remaining_count_fp", o.get("remaining_count")),
          o.get("yes_price_dollars"), o.get("no_price_dollars"))
