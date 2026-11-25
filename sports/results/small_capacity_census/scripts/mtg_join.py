"""Join MTGJSON spread candidates to the Card Kingdom v2 pricelist (names, qty_buying, NM retail) and cross-check
TCGplayer price against Scryfall (prices.usd = TCGplayer market price) via POST /cards/collection (75 ids/call,
honest UA, 250 ms spacing). Read-only. Usage:
python3 -I mtg_join.py <AllPricesToday.json> <ck_pricelist_v2.json> <out.json> [min_net]
"""
import json, sys, time, urllib.request
from collections import defaultdict

mj, ckf, out = sys.argv[1], sys.argv[2], sys.argv[3]
MIN_NET = float(sys.argv[4]) if len(sys.argv) > 4 else 2.0
TAX, SHIP = 0.0925, 1.00

data = json.load(open(mj))["data"]
cands = []
for uuid, rec in data.items():
    paper = rec.get("paper") or {}
    ck, tcg = paper.get("cardkingdom") or {}, paper.get("tcgplayer") or {}
    for fin in ("normal", "foil"):
        b = (ck.get("buylist") or {}).get(fin) or {}
        r = (tcg.get("retail") or {}).get(fin) or {}
        cr = (ck.get("retail") or {}).get(fin) or {}
        if not b or not r:
            continue
        bv, rv = b[max(b)], r[max(r)]
        crv = cr[max(cr)] if cr else None
        if not rv or rv <= 0 or bv is None:
            continue
        net = bv - (rv * (1 + TAX) + SHIP)
        if net >= MIN_NET:
            cands.append(dict(uuid=uuid, foil=(fin == "foil"), ck_buy=bv, mj_tcg=rv, mj_ck_retail=crv, mj_net=round(net, 2)))

ck = json.load(open(ckf))
print("ck meta", ck.get("meta"), "rows", len(ck["data"]))
idx = defaultdict(list)
for row in ck["data"]:
    try:
        pb = round(float(row["price_buy"]), 2)
    except Exception:
        continue
    idx[(pb, row["is_foil"] == "true")].append(row)

matched = []
amb = 0
for c in cands:
    rows = idx.get((round(c["ck_buy"], 2), c["foil"]), [])
    if c["mj_ck_retail"] is not None:
        rows2 = [r for r in rows if abs(float(r["price_retail"]) - c["mj_ck_retail"]) < 0.005]
        rows = rows2 or rows
    if len(rows) == 1:
        c.update(name=rows[0]["name"], edition=rows[0]["edition"], variation=rows[0]["variation"],
                 scryfall_id=rows[0]["scryfall_id"], qty_buying=rows[0]["qty_buying"], qty_retail=rows[0]["qty_retail"],
                 ck_nm_price=(rows[0].get("condition_values") or {}).get("nm_price"))
        matched.append(c)
    elif rows:
        amb += 1
print("candidates", len(cands), "unique CK matches", len(matched), "ambiguous", amb)

# Scryfall cross-check
UA = {"User-Agent": "small-capacity-census/0.1 (read-only research script)", "Accept": "application/json",
      "Content-Type": "application/json"}
scry = {}
ids = list({m["scryfall_id"] for m in matched if m.get("scryfall_id")})
for i in range(0, len(ids), 75):
    body = json.dumps({"identifiers": [{"id": x} for x in ids[i:i + 75]]}).encode()
    req = urllib.request.Request("https://api.scryfall.com/cards/collection", data=body, headers=UA, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        js = json.load(resp)
    for card in js.get("data", []):
        scry[card["id"]] = dict(sf_name=card.get("name"), set=card.get("set"), rarity=card.get("rarity"),
                                reserved=card.get("reserved"), usd=card.get("prices", {}).get("usd"),
                                usd_foil=card.get("prices", {}).get("usd_foil"))
    time.sleep(0.25)
for m in matched:
    s = scry.get(m.get("scryfall_id"), {})
    m.update(s)
    p = s.get("usd_foil") if m["foil"] else s.get("usd")
    m["sf_tcg"] = float(p) if p else None
    m["sf_net"] = round(m["ck_buy"] - (m["sf_tcg"] * (1 + TAX) + SHIP), 2) if m["sf_tcg"] else None
    m["agree"] = (m["sf_tcg"] is not None and abs(m["sf_tcg"] - m["mj_tcg"]) / max(m["mj_tcg"], 0.01) < 0.15)

conf = [m for m in matched if m["sf_net"] is not None and m["sf_net"] >= MIN_NET]
summ = dict(min_net=MIN_NET, n_cands=len(cands), n_matched=len(matched), n_ambiguous=amb,
            n_scryfall_priced=sum(1 for m in matched if m["sf_tcg"] is not None),
            n_confirmed_by_scryfall=len(conf),
            n_confirmed_under_50=sum(1 for m in conf if m["sf_tcg"] < 50),
            n_reserved_list=sum(1 for m in conf if m.get("reserved")),
            qty_buying_quantiles=sorted(m["qty_buying"] for m in conf)[::max(1, len(conf) // 10)] if conf else [],
            cap_profit_sum_qty_le4=round(sum(m["sf_net"] * min(4, m["qty_buying"] or 0) for m in conf), 2),
            cap_profit_sum_under50_qty_le4=round(sum(m["sf_net"] * min(4, m["qty_buying"] or 0) for m in conf if m["sf_tcg"] < 50), 2))
json.dump(dict(summary=summ, confirmed=sorted(conf, key=lambda x: -x["sf_net"]), matched=matched), open(out, "w"), indent=1)
print(json.dumps(summ, indent=1))
for m in sorted(conf, key=lambda x: (x["sf_tcg"] >= 50, -x["sf_net"]))[:30]:
    print(f'{m["name"][:28]:28s} | {m["edition"][:22]:22s} | foil={m["foil"]!s:5s} ckbuy={m["ck_buy"]:8.2f} '
          f'mjTCG={m["mj_tcg"]:8.2f} sfTCG={m["sf_tcg"]:8.2f} ckNM={m.get("ck_nm_price")} qtyBuy={m["qty_buying"]} '
          f'net={m["sf_net"]:7.2f} RL={m.get("reserved")}')
