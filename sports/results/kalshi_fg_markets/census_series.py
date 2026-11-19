import sys, json
sys.path.insert(0, "scripts")
from kx_throttle import kget
S = ["KXNFLGAMEFG","KXNFLFG","KXNFLTEAMFG","KXNFLLONGFG","KXNFLLONGESTFG","KXNFL60YARDFGS","KXNCAAFTEAMFG","KXNCAAFTOTALFG"]
out = {}
for s in S:
    j = kget(f"/series/{s}") or {}
    ser = j.get("series") or {}
    # events, all statuses
    evs = []; cur = None
    while True:
        p = {"series_ticker": s, "limit": 200}
        if cur: p["cursor"] = cur
        e = kget("/events", p) or {}
        evs += e.get("events") or []
        cur = e.get("cursor")
        if not cur or not e.get("events"): break
    # markets
    mk = []; cur = None
    while True:
        p = {"series_ticker": s, "limit": 1000}
        if cur: p["cursor"] = cur
        m = kget("/markets", p) or {}
        mk += m.get("markets") or []
        cur = m.get("cursor")
        if not cur or not m.get("markets"): break
    # historical markets
    hk = []; cur = None
    while True:
        p = {"series_ticker": s, "limit": 1000}
        if cur: p["cursor"] = cur
        m = kget("/historical/markets", p) or {}
        hk += m.get("markets") or []
        cur = m.get("cursor")
        if not cur or not m.get("markets"): break
    out[s] = {"series": ser, "events": evs, "markets": mk, "hist_markets": hk}
    print(s, "found" if ser else "NOSERIES", ser.get("fee_type"), ser.get("fee_multiplier"), "events", len(evs), "mk", len(mk), "hist", len(hk), flush=True)
json.dump(out, open("results/kalshi_fg_markets/census_raw.json","w"))
