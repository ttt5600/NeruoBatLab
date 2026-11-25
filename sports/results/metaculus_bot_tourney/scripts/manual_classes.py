"""MANUAL classification (read every title in raw/mad_meta.jsonl, 2026-11-24) of the 180 MiniBench questions in the
three complete rounds logged by github.com/m-ad/claude_forecasting_auto (rounds starting 2026-10-19, 11-02, 11-16).
A = official US series the fund already nowcasts (NWS/NOAA weather, precip, tides, buoys; AAA gas; BEA/BLS/EIA macro
    releases; PortWatch-style Hormuz AIS counts)
B = market price / FX reference close (spot + implied-vol random walk answers it; any bot with yfinance can too)
C = other official/public data series we do not run (foreign CPI/GDP, disease surveillance, election tallies, company
    results, ratings, social counts, USDA/CDC/NHC series)
D = event needing news judgment (politics, conflict, courts, approvals, sports results)"""
import json, statistics as st
A = {46124, 46132, 46161, 46378, 46386, 46395, 46406, 46418, 46579, 46587, 46594, 46597, 46619}
B = {46126, 46127, 46130, 46139, 46143, 46145, 46154, 46156, 46170, 46176, 46178, 46400, 46409, 46422,
     46563, 46564, 46567, 46571, 46576, 46592, 46595, 46601, 46602, 46607}
D = {46123, 46125, 46135, 46137, 46146, 46147, 46150, 46155, 46157, 46159, 46160, 46162, 46163, 46165, 46167, 46168,
     46169, 46171, 46172, 46173, 46175, 46179, 46180,
     46370, 46373, 46375, 46376, 46377, 46382, 46384, 46388, 46391, 46393, 46399, 46402, 46404, 46407, 46412, 46414,
     46415, 46420, 46423, 46425, 46426, 46427,
     46561, 46565, 46566, 46568, 46570, 46573, 46574, 46575, 46578, 46581, 46584, 46585, 46586, 46588, 46589, 46596,
     46599, 46603, 46608, 46610, 46612, 46613, 46614, 46615, 46620}
rows = [json.loads(l) for l in open("/Users/jonathanwang/Desktop/vocalizations_lab/.claude/worktrees/quant-trader/"
                                     "sports/results/metaculus_bot_tourney/raw/mad_meta.jsonl")]
mb = [r for r in rows if r.get("tourn") == "minibench"]
def pid(r): return int(r["slug"].split("-mc")[1].split("-")[0])
out = {"A": [], "B": [], "C": [], "D": []}
for r in mb:
    p = pid(r); k = "A" if p in A else "B" if p in B else "D" if p in D else "C"
    out[k].append(r)
n = len(mb)
print("MiniBench questions:", n, "rounds:", sorted({r["slug"][:10] for r in mb})[:1], "...")
for k, v in out.items():
    sc = [r["score"] for r in v if isinstance(r.get("score"), (int, float))]
    types = {}
    for r in v: types[r["type"]] = types.get(r["type"], 0) + 1
    print(f"{k}: {len(v):3d} ({len(v)/n:.0%})  types {types}  m-ad bot scored n={len(sc)} mean peer {st.mean(sc) if sc else float('nan'):.1f}"
          + (f" median {st.median(sc):.1f}" if sc else ""))
print("official-data resolvable (A+B+C):", sum(len(out[k]) for k in "ABC"), "of", n)
print("fund-pipeline or price-model (A+B):", len(out["A"]) + len(out["B"]), "of", n)
