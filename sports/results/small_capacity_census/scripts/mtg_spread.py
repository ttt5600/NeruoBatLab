"""MTG buylist-vs-retail spread census from MTGJSON AllPricesToday (read-only, MIT data).
Question: how many printings have Card Kingdom (CK) buylist CASH >= TCGplayer retail * (1+tax) + per-card shipping share,
i.e. buy on TCGplayer, sell to CK, net >= threshold? Prints counts by threshold and the top rows.
Caveats: MTGJSON 'retail.normal' for tcgplayer is a market/low-type price (exact definition per MTGJSON docs: not
verified here); CK buylist is for NM; TCGplayer listing at that price may be LP/one copy; no depth.
Usage: python3 -I mtg_spread.py <AllPricesToday.json> <out.json>
"""
import json, sys

src, out = sys.argv[1], sys.argv[2]
d = json.load(open(src))
data = d["data"]
TAX = 0.0925          # CA ~7.25-10.25%; use 9.25% midpoint-ish (TCGplayer collects sales tax)
SHIP = 1.00           # per-card share of seller shipping (orders <$5 pay ~$1-1.3; >$5 often free) -- assumption
rows = []
prov_seen = {}
for uuid, rec in data.items():
    paper = rec.get("paper") or {}
    for p in paper:
        prov_seen[p] = prov_seen.get(p, 0) + 1
    ck = paper.get("cardkingdom") or {}
    tcg = paper.get("tcgplayer") or {}
    for finish in ("normal", "foil"):
        b = ((ck.get("buylist") or {}).get(finish) or {})
        r = ((tcg.get("retail") or {}).get(finish) or {})
        ckr = ((ck.get("retail") or {}).get(finish) or {})
        if not b or not r:
            continue
        bd = max(b); rd = max(r)
        bv, rv = b[bd], r[rd]
        if rv is None or bv is None or rv <= 0:
            continue
        cost = rv * (1 + TAX) + SHIP
        rows.append({"uuid": uuid, "finish": finish, "ck_buy": bv, "tcg_retail": rv,
                     "ck_retail": ckr.get(max(ckr)) if ckr else None, "buy_date": bd, "ret_date": rd,
                     "net": round(bv - cost, 2), "ratio": round(bv / rv, 3)})
rows.sort(key=lambda x: -x["net"])
summ = {"meta": d.get("meta"), "providers_seen": prov_seen, "n_pairs": len(rows),
        "assumptions": {"tax": TAX, "ship_per_card": SHIP}}
for thr in (0, 1, 2, 5, 10, 20):
    summ[f"n_net_ge_{thr}"] = sum(1 for x in rows if x["net"] >= thr)
summ["n_ratio_ge_1"] = sum(1 for x in rows if x["ratio"] >= 1)
summ["sum_net_ge_2"] = round(sum(x["net"] for x in rows if x["net"] >= 2), 2)
summ["ratio_quantiles"] = {q: sorted(x["ratio"] for x in rows)[int(q * (len(rows) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if rows else {}
summ["top40"] = rows[:40]
json.dump(summ, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in summ.items() if k != "top40"}, indent=1))
for x in rows[:15]:
    print(x)
