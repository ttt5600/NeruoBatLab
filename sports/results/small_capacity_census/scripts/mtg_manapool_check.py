"""Offline robustness check of sub-$50 CK-buylist arb candidates: require a SECOND US marketplace (Manapool retail,
from the same MTGJSON AllPricesToday file) to also price the card low enough. Net computed vs min and vs max of the two.
Usage: python3 -I mtg_manapool_check.py <AllPricesToday.json> <out.json>"""
import json, sys
d = json.load(open(sys.argv[1]))["data"]
TAX, SHIP = 0.0925, 1.0
rows = []
for u, rec in d.items():
    p = rec.get("paper") or {}; k = p.get("cardkingdom") or {}; t = p.get("tcgplayer") or {}; m = p.get("manapool") or {}
    for f in ("normal", "foil"):
        b = (k.get("buylist") or {}).get(f) or {}; rr = (t.get("retail") or {}).get(f) or {}
        mm = (m.get("retail") or {}).get(f) or {}; cr = (k.get("retail") or {}).get(f) or {}
        if not b or not rr: continue
        bv, rv = b[max(b)], rr[max(rr)]
        mv = mm[max(mm)] if mm else None
        cv = cr[max(cr)] if cr else None
        if not rv or rv <= 0 or rv >= 50: continue
        net_t = bv - (rv * (1 + TAX) + SHIP)
        if net_t < 2: continue
        net_m = bv - (mv * (1 + TAX) + SHIP) if mv else None
        rows.append(dict(uuid=u, foil=f == "foil", ck_buy=bv, tcg=rv, manapool=mv, ck_retail=cv,
                         net_tcg=round(net_t, 2), net_mp=round(net_m, 2) if net_m is not None else None,
                         ck_buy_over_ck_retail=round(bv / cv, 2) if cv else None))
both = [r for r in rows if r["net_mp"] is not None and r["net_mp"] >= 2]
summ = dict(n_sub50_net2_tcg=len(rows), n_with_manapool=sum(1 for r in rows if r["manapool"]),
            n_both_venues_net2=len(both), sum_net_min_both=round(sum(min(r["net_tcg"], r["net_mp"]) for r in both), 2),
            n_ck_retail_missing=sum(1 for r in rows if r["ck_retail"] is None),
            buy_over_retail_quantiles=sorted(r["ck_buy_over_ck_retail"] for r in rows if r["ck_buy_over_ck_retail"])[::10])
json.dump(dict(summary=summ, both=sorted(both, key=lambda r: -min(r["net_tcg"], r["net_mp"])), rows=rows),
          open(sys.argv[2], "w"), indent=1)
print(json.dumps(summ, indent=1))
for r in sorted(both, key=lambda r: -min(r["net_tcg"], r["net_mp"]))[:40]:
    print(r)
