"""Name 'both venues' candidates: CK rows matching (price_buy, price_retail, foil) -> Scryfall /cards/collection prices ->
keep the row whose Scryfall usd/usd_foil == MTGJSON TCGplayer price (to the cent). Honest UA, 0.3 s spacing.
Usage: python3 -I mtg_name_scry.py <mtg_manapool_out.json> <ck_pricelist_v2.json> <out.json>"""
import json, sys, time, urllib.request
from collections import defaultdict
both = json.load(open(sys.argv[1]))["both"]
ck = json.load(open(sys.argv[2]))["data"]
idx = defaultdict(list)
for r in ck:
    idx[(round(float(r["price_buy"]), 2), round(float(r["price_retail"]), 2), r["is_foil"] == "true")].append(r)
need = {}
for b in both:
    if b["ck_retail"] is None: continue
    for r in idx.get((round(b["ck_buy"], 2), round(b["ck_retail"], 2), b["foil"]), []):
        if r.get("scryfall_id"): need[r["scryfall_id"]] = None
ids = list(need)
print("ids to fetch", len(ids), flush=True)
H = {"User-Agent": "small-capacity-census/0.1 (read-only research script)", "Accept": "application/json",
     "Content-Type": "application/json"}
sc = {}
for i in range(0, len(ids), 75):
    body = json.dumps({"identifiers": [{"id": x} for x in ids[i:i + 75]]}).encode()
    with urllib.request.urlopen(urllib.request.Request("https://api.scryfall.com/cards/collection", data=body,
                                                       headers=H, method="POST"), timeout=120) as resp:
        js = json.load(resp)
    for c in js.get("data", []):
        sc[c["id"]] = (c["name"], c["set_name"], c["prices"].get("usd"), c["prices"].get("usd_foil"), c.get("rarity"))
    time.sleep(0.3)
named = []
for b in both:
    if b["ck_retail"] is None: continue
    hits = []
    for r in idx.get((round(b["ck_buy"], 2), round(b["ck_retail"], 2), b["foil"]), []):
        s = sc.get(r.get("scryfall_id"))
        if not s: continue
        p = s[3] if b["foil"] else s[2]
        if p is not None and abs(float(p) - b["tcg"]) < 0.005:
            hits.append((r, s))
    if len(hits) == 1:
        r, s = hits[0]; cv = r.get("condition_values") or {}
        b.update(name=r["name"], edition=r["edition"], variation=r["variation"], rarity=s[4], qty_buying=r["qty_buying"],
                 qty_retail=r["qty_retail"], nm_qty=cv.get("nm_qty"), url=r["url"])
        named.append(b)
summ = dict(n_both=len(both), n_with_ck_retail=sum(1 for b in both if b["ck_retail"] is not None), n_named=len(named),
            n_qty_buying_gt0=sum(1 for b in named if b["qty_buying"] > 0), n_foil=sum(1 for b in named if b["foil"]),
            sum_qty_buying=sum(b["qty_buying"] for b in named),
            cap_net_all_qty=round(sum(min(b["net_tcg"], b["net_mp"]) * b["qty_buying"] for b in named), 2))
json.dump(dict(summary=summ, named=named), open(sys.argv[3], "w"), indent=1)
print(json.dumps(summ))
for b in sorted(named, key=lambda x: -min(x["net_tcg"], x["net_mp"])):
    print(f'{b["name"][:28]:28s}|{b["edition"][:22]:22s}|{b["variation"][:16]:16s}|f={b["foil"]!s:5s} buy={b["ck_buy"]:6.2f} '
          f'tcg={b["tcg"]:6.2f} mp={b["manapool"]:6.2f} ckret={b["ck_retail"]:6.2f} qbuy={b["qty_buying"]} qret={b["qty_retail"]} '
          f'net={min(b["net_tcg"], b["net_mp"]):6.2f}')
