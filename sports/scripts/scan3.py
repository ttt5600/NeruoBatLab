"""Three-way House scan: NYT needle vs Kalshi vs Polymarket (read-only). Flags Kalshi outliers."""
import json, re, sys, time, urllib.request
sys.argv = [sys.argv[0]]
import nyt_needle_house as B

def poly():
    out, off = {}, 0
    while True:
        req = urllib.request.Request(f"https://gamma-api.polymarket.com/events?tag_slug=house-elections&closed=false&limit=100&offset={off}",
                                     headers={"User-Agent": "Mozilla/5.0"})
        ev = json.load(urllib.request.urlopen(req, timeout=30))
        if not ev: break
        for e in ev:
            for m in e.get("markets") or []:
                q = m.get("question") or ""
                mm = re.search(r"win the ([A-Z]{2})-(\d+|AL) House seat", q)
                pa = "R" if "Republican" in q else "D" if "Democratic" in q else None
                if not mm or not pa or m.get("closed"): continue
                st, d = mm.groups()
                d = "AL" if d == "AL" or st in B.AT_LARGE else str(int(d))
                bb, ba = B.f(m.get("bestBid")), B.f(m.get("bestAsk"))
                if bb is None or ba is None: continue
                out[((st, d), pa)] = dict(bid=bb, ask=ba, mid=(bb + ba) / 2, liq=B.f(m.get("liquidity")) or 0)
        off += 100
    return out

N = B.needles(); P = poly(); M = B.kalshi_markets(); C = B.cnn_calls()
rows = []
for m in M:
    k, pa = m["_key"], m["_party"]
    yb, ya = B.f(m.get("yes_bid_dollars")), B.f(m.get("yes_ask_dollars"))
    if yb is None or ya is None or ya <= 0: continue
    p = P.get((k, pa)); n = N.get(k) or {}
    km = (yb + ya) / 2
    pm = p["mid"] if p else None
    nd = n.get(pa)
    rows.append((k, pa, yb, ya, km, p, nd, n, C.get(k)))
print(f"needles={len(N)} poly={len(P)} kalshi={len(M)} cnn_called={len(C)}  {time.strftime('%H:%M:%S')}")
print("--- Kalshi vs Polymarket gaps (buy Kalshi if poly bid > kalshi ask, or poly ask < kalshi bid) ---")
for k, pa, yb, ya, km, p, nd, n, c in sorted(rows, key=lambda r: -abs(r[4] - (r[5]["mid"] if r[5] else r[4]))):
    if not p: continue
    gap_buy_yes = p["bid"] - ya       # poly bid above kalshi yes ask -> kalshi YES cheap
    gap_buy_no = yb - p["ask"]         # kalshi yes bid above poly ask -> kalshi NO cheap
    g = max(gap_buy_yes, gap_buy_no)
    if g < 0.03 and abs(km - p["mid"]) < 0.08: continue
    side = "YES" if gap_buy_yes >= gap_buy_no else "NO"
    print(f"{k[0]}-{k[1]:>2} {pa} kalshi {yb:.2f}/{ya:.2f} poly {p['bid']:.3f}/{p['ask']:.3f} liq{p['liq']:>7.0f}  "
          f"nyt {nd if nd is None else round(nd,3)} cnt {n.get('counted')} upd {str(n.get('updated'))[11:19]} call {n.get('call')}/{c}  "
          f"=> Kalshi {side} cheap by {g:+.3f}")
print("--- contested races (Kalshi mid 0.05-0.95), D leg ---")
for k, pa, yb, ya, km, p, nd, n, c in sorted(rows, key=lambda r: r[0]):
    if pa != "D" or not (0.05 < km < 0.95): continue
    import datetime as dt
    age = None
    try: age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(n["updated"].replace("Z","+00:00"))).total_seconds()/60
    except Exception: pass
    print(f"{k[0]}-{k[1]:>2} D kalshi {yb:.2f}/{ya:.2f} poly {p['mid'] if p else float('nan'):.3f} nyt(tilt) {nd if nd is None else round(nd,3)} nytraw {n.get('nyt_D', nd) if n.get('nyt_D', nd) is None else round(n.get('nyt_D', nd),3)} eevp {n.get('eevp')} age {age if age is None else round(age)}m call {n.get('call')}/{c}")
