"""Name the 'both venues' candidates (mtg_manapool_out.json) by matching CK pricelist on price_buy + price_retail + foil.
Usage: python3 -I mtg_name_both.py <mtg_manapool_out.json> <ck_pricelist_v2.json> <out.json>"""
import json, sys
from collections import defaultdict
both = json.load(open(sys.argv[1]))["both"]
ck = json.load(open(sys.argv[2]))["data"]
idx = defaultdict(list)
for r in ck:
    idx[(round(float(r["price_buy"]), 2), round(float(r["price_retail"]), 2), r["is_foil"] == "true")].append(r)
out = []
for b in both:
    if b["ck_retail"] is None:
        b["match"] = "no_ck_retail"; out.append(b); continue
    rows = idx.get((round(b["ck_buy"], 2), round(b["ck_retail"], 2), b["foil"]), [])
    b["n_match"] = len(rows)
    if len(rows) == 1:
        r = rows[0]; cv = r.get("condition_values") or {}
        b.update(name=r["name"], edition=r["edition"], variation=r["variation"], qty_buying=r["qty_buying"],
                 qty_retail=r["qty_retail"], nm_qty=cv.get("nm_qty"), url=r["url"])
    out.append(b)
named = [b for b in out if b.get("name")]
summ = dict(n_both=len(both), n_named=len(named), n_qty_buying_gt0=sum(1 for b in named if b["qty_buying"] > 0),
            n_foil=sum(1 for b in named if b["foil"]),
            cap_net_qty_buying=round(sum(min(b["net_tcg"], b["net_mp"]) * b["qty_buying"] for b in named), 2),
            sum_qty_buying=sum(b["qty_buying"] for b in named))
json.dump(dict(summary=summ, rows=out), open(sys.argv[3], "w"), indent=1)
print(json.dumps(summ))
for b in sorted(named, key=lambda x: -min(x["net_tcg"], x["net_mp"])):
    print(f'{b["name"][:30]:30s}|{b["edition"][:26]:26s}|{b["variation"][:18]:18s}|foil={b["foil"]!s:5s} buy={b["ck_buy"]:6.2f} '
          f'tcg={b["tcg"]:6.2f} mp={b["manapool"]:6.2f} ckret={b["ck_retail"]:6.2f} qbuy={b["qty_buying"]} qret={b["qty_retail"]} '
          f'net={min(b["net_tcg"], b["net_mp"]):6.2f}')
