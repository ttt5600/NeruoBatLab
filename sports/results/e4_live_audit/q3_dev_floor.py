"""Q3 (DEV only): M3_N10 per-pick units sliced by NO ask at T, and by city. Sealed files are NOT read."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import kalshi_temp_maker as K

p = pd.read_parquet(K.PICKS); p["date"] = pd.to_datetime(p["date"])
assert ((p.cohort == "E1") & (p.date >= "2025-07-01")).sum() == 0 and ((p.cohort == "X1") & (p.date >= "2026-03-01")).sum() == 0
tr = pd.read_parquet(K.TRADES); qk = pd.read_parquet(K.QUOTES)
mk, bet, filled = K.variant(p, tr, qk, "M3_N10")
base = K.taker(p)
ask = 1 - p.yes_bid.to_numpy(float)
q = K.maker_price(p, "3")
dates = p.date.dt.strftime("%Y-%m-%d").to_numpy()
rng = np.random.default_rng(0)
out = {"picks": int(len(p)), "fills": int(filled.sum()), "units_all": round(float(mk.sum()), 2)}


def summ(mask, tag):
    m = mask
    bu = K.boot(dates[m], mk[m], rng)
    br = K.boot(dates[m], mk[m], rng, denom=filled[m].astype(float))
    won = p.won.to_numpy()[m]
    return {"slice": tag, "picks": int(m.sum()), "fills": int(filled[m].sum()), "fill_rate": round(float(filled[m].mean()), 3),
            "units": round(float(mk[m].sum()), 2), "units_ci": [round(float(x), 2) for x in np.quantile(bu, [.025, .975])],
            "roi_fill": round(float(mk[m][filled[m]].mean()), 4) if filled[m].any() else None,
            "roi_ci": [round(float(x), 4) for x in np.quantile(br, [.025, .975])],
            "loss_rate_filled": round(float(1 - won[filled[m]].mean()), 4) if filled[m].any() else None,
            "taker_units": round(float(base[m].sum()), 2), "maker_minus_taker": round(float((mk - base)[m].sum()), 2)}

rows = [summ(ask < 0.90, "ask<0.90 (dropped by floor)"), summ(ask >= 0.90, "ask>=0.90 (kept)")]
for lo, hi in [(0, .80), (.80, .90), (.90, .95), (.95, .98), (.98, 1.01)]:
    rows.append(summ((ask >= lo) & (ask < hi), f"ask [{lo},{hi})"))
rows.append(summ(q >= 0.97, "q>=0.97 (live-typical)"))
out["slices"] = rows
city = p["ticker"].str.split("-").str[0].to_numpy()
cr = []
for c in np.unique(city):
    s = summ(city == c, c); cr.append(s)
out["cities"] = cr
json.dump(out, open("results/e4_live_audit/q3_dev_floor.json", "w"), indent=1)
for r in rows:
    print(r)
print("cities:")
for r in sorted(cr, key=lambda r: r["units"]):
    print({k: r[k] for k in ("slice", "fills", "units", "units_ci", "roi_fill", "loss_rate_filled")})
