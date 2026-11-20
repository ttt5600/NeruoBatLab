"""Census table of the series whose T-60 quotes were fetched (quotes only, no outcomes) -> census_quotes.txt"""
import json, glob
import numpy as np, pandas as pd
c = json.load(open("results/derived_ladders/census.json"))
rows = []
for f in sorted(glob.glob("results/derived_ladders/ev/q_*.jsonl")):
    s = f.split("q_")[1][:-6]
    L = [json.loads(l) for l in open(f)]
    L = [x for x in L if x["q"]]
    sp, v = [], []
    for x in L:
        for q in x["q"].values():
            if not q:
                continue
            v.append(q["vol6h"])
            if q["ask"] and q["bid"] and float(q["bid"]) > 0 and float(q["ask"]) < 1:
                sp.append(float(q["ask"]) - float(q["bid"]))
    cs = c.get(s, {})
    rows.append(dict(series=s, settled_ev=cs.get("n_ev"), mk_per_ev=round(cs.get("mk_per_ev", 0), 1), vol_med_ct=round(cs.get("vol_med", 0)),
                     fee=cs.get("fee"), quoted_ev=len(L), T60_spread_med=round(float(np.median(sp)), 3) if sp else None,
                     frac_traded_6h=round(float(np.mean([x > 0 for x in v])), 2) if v else None))
t = pd.DataFrame(rows).sort_values("quoted_ev", ascending=False)
open("results/derived_ladders/census_quotes.txt", "w").write(t.to_string(index=False))
print(t.to_string(index=False))
