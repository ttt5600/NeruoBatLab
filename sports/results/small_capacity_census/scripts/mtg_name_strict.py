"""Name the strict T-MTG-signal cards (max(TCG,Manapool) < $50, net >= $3 at 9.25% tax + $1 ship) via CK tier match ->
Scryfall /cards/collection (retry x3, honest UA). Usage: python3 -I mtg_name_strict.py <AllPricesToday.json> <ck.json> <out.json>"""
import json, sys, time, urllib.request
from collections import defaultdict
d = json.load(open(sys.argv[1]))["data"]; ck = json.load(open(sys.argv[2]))["data"]
TAX, SHIP, MARGIN = 0.0925, 1.0, 3.0
idx = defaultdict(list)
for r in ck:
    idx[(round(float(r["price_buy"]), 2), round(float(r["price_retail"]), 2), r["is_foil"] == "true")].append(r)
cands = []
for u, rec in d.items():
    p = rec.get("paper") or {}; k = p.get("cardkingdom") or {}; t = p.get("tcgplayer") or {}; m = p.get("manapool") or {}
    for f in ("normal", "foil"):
        b = (k.get("buylist") or {}).get(f) or {}; r = (t.get("retail") or {}).get(f) or {}
        mm = (m.get("retail") or {}).get(f) or {}; cr = (k.get("retail") or {}).get(f) or {}
        if not b or not r or not mm: continue
        tv, mv = r[max(r)] or 0, mm[max(mm)] or 0
        px = max(tv, mv)
        if px <= 0 or px >= 50: continue
        bv = b[max(b)]; net = bv - (px * (1 + TAX) + SHIP)
        if net >= MARGIN:
            cands.append(dict(foil=f == "foil", ck_buy=bv, tcg=tv, mp=mv, ck_retail=cr[max(cr)] if cr else None, net=round(net, 2)))
need = set()
for c in cands:
    if c["ck_retail"] is None: continue
    for r in idx.get((round(c["ck_buy"], 2), round(c["ck_retail"], 2), c["foil"]), []):
        need.add(r["scryfall_id"])
ids = sorted(need); print("cands", len(cands), "ids", len(ids), flush=True)
H = {"User-Agent": "small-capacity-census/0.1 (read-only research script)", "Accept": "application/json",
     "Content-Type": "application/json"}
sc = {}
for i in range(0, len(ids), 75):
    body = json.dumps({"identifiers": [{"id": x} for x in ids[i:i + 75]]}).encode()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request("https://api.scryfall.com/cards/collection", data=body,
                                                               headers=H, method="POST"), timeout=180) as resp:
                js = json.load(resp)
            break
        except Exception as e:
            print("retry", i, attempt, repr(e)[:80], flush=True); time.sleep(5)
    else:
        continue
    for c in js.get("data", []):
        sc[c["id"]] = (c["name"], c["set_name"], c["prices"].get("usd"), c["prices"].get("usd_foil"), c.get("released_at"))
    print("batch", i // 75 + 1, "of", (len(ids) + 74) // 75, flush=True); time.sleep(0.3)
out = []
for c in cands:
    hits = []
    if c["ck_retail"] is not None:
        for r in idx.get((round(c["ck_buy"], 2), round(c["ck_retail"], 2), c["foil"]), []):
            s = sc.get(r["scryfall_id"])
            if not s: continue
            pz = s[3] if c["foil"] else s[2]
            if pz is not None and abs(float(pz) - c["tcg"]) < 0.005: hits.append((r, s))
    c["n_hits"] = len(hits)
    if len(hits) == 1:
        r, s = hits[0]
        c.update(name=r["name"], edition=r["edition"], variation=r["variation"], released=s[4], qty_buying=r["qty_buying"],
                 qty_retail=r["qty_retail"], url="https://www.cardkingdom.com/" + r["url"].replace("\\/", "/"))
    out.append(c)
json.dump(out, open(sys.argv[3], "w"), indent=1)
for c in sorted(out, key=lambda x: -x["net"]):
    print(f'{c.get("name","?")[:28]:28s}|{c.get("edition","?")[:24]:24s}|{str(c.get("released"))[:10]}|f={c["foil"]!s:5s} '
          f'buy={c["ck_buy"]:6.2f} tcg={c["tcg"]:6.2f} mp={c["mp"]:6.2f} ckret={c["ck_retail"]} qbuy={c.get("qty_buying")} net={c["net"]:6.2f} hits={c["n_hits"]}')
