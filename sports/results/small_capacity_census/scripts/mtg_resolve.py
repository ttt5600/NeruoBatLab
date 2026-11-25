"""Resolve MTGJSON sub-$50 buylist-arb candidates to named cards: Scryfall search usd=<exact TCG price> (unique=prints),
intersect with CK pricelist rows at the same price_buy/foil. Read-only, honest UA, 0.5 s spacing, max 2 pages/query.
Usage: python3 -I mtg_resolve.py <AllPricesToday.json> <ck_pricelist_v2.json> <out.json>"""
import json, sys, time, urllib.request, urllib.parse
from collections import defaultdict
mj, ckf, out = sys.argv[1:4]
TAX, SHIP = 0.0925, 1.0
d = json.load(open(mj))["data"]
ck = json.load(open(ckf))["data"]
idx = defaultdict(list)
for r in ck:
    idx[(round(float(r["price_buy"]), 2), r["is_foil"] == "true")].append(r)
cands = []
for u, rec in d.items():
    p = rec.get("paper") or {}; k = p.get("cardkingdom") or {}; t = p.get("tcgplayer") or {}
    for f in ("normal", "foil"):
        b = (k.get("buylist") or {}).get(f) or {}; rr = (t.get("retail") or {}).get(f) or {}
        if not b or not rr: continue
        bv, rv = b[max(b)], rr[max(rr)]
        if not rv or rv <= 0: continue
        net = bv - (rv * (1 + TAX) + SHIP)
        if net >= 2 and rv < 50:
            cands.append(dict(uuid=u, foil=(f == "foil"), ck_buy=bv, tcg=rv, net=round(net, 2)))
H = {"User-Agent": "small-capacity-census/0.1 (read-only research script)", "Accept": "application/json"}
cache = {}
def search(price):
    if price in cache: return cache[price]
    ids = {}
    url = "https://api.scryfall.com/cards/search?" + urllib.parse.urlencode({"q": f"usd={price:.2f}", "unique": "prints"})
    for _ in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=60) as resp:
                j = json.load(resp)
        except Exception as e:
            break
        for c in j.get("data", []):
            ids[c["id"]] = (c["name"], c["set_name"], c["prices"].get("usd"), c["prices"].get("usd_foil"), c.get("rarity"))
        time.sleep(0.5)
        if not j.get("has_more"): break
        url = j["next_page"]
    cache[price] = ids
    return ids
res = []
for c in cands:
    ids = search(c["tcg"])
    want = f'{c["tcg"]:.2f}'
    hits = []
    for r in idx[(round(c["ck_buy"], 2), c["foil"])]:
        s = ids.get(r["scryfall_id"])
        if not s: continue
        price = s[3] if c["foil"] else s[2]
        if price == want:
            hits.append((r, s))
    c["n_hits"] = len(hits)
    if len(hits) == 1:
        r, s = hits[0]
        c.update(name=s[0], set=s[1], rarity=s[4], ck_edition=r["edition"], variation=r["variation"],
                 qty_buying=r["qty_buying"], ck_retail=r["price_retail"], qty_retail=r["qty_retail"],
                 ck_nm=(r.get("condition_values") or {}).get("nm_price"), ck_url=r["url"])
    res.append(c)
resolved = [c for c in res if c["n_hits"] == 1]
summ = dict(n=len(res), resolved=len(resolved), zero=sum(1 for c in res if c["n_hits"] == 0),
            multi=sum(1 for c in res if c["n_hits"] > 1),
            resolved_qty_gt0=sum(1 for c in resolved if c["qty_buying"] > 0),
            resolved_net_sum_qty_le4=round(sum(c["net"] * min(4, c["qty_buying"]) for c in resolved), 2))
json.dump(dict(summary=summ, rows=res), open(out, "w"), indent=1)
print(json.dumps(summ))
for c in sorted(resolved, key=lambda x: -x["net"]):
    print(f'{c["name"][:30]:30s}|{c["ck_edition"][:24]:24s}|{c["variation"][:14]:14s}|foil={c["foil"]!s:5s} ckbuy={c["ck_buy"]:6.2f} tcg={c["tcg"]:6.2f} ckret={c["ck_retail"]} qret={c["qty_retail"]} qbuy={c["qty_buying"]} net={c["net"]:6.2f}')
