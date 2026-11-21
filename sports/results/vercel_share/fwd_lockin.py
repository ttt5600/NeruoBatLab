"""Forward intraday lock-in from fwd/snapshots.jsonl: for each series and UTC day, the partial value at each snapshot vs the
final value (first snapshot >= 01:00Z of the next day). Prints |partial - final| by hours into the day, and post-midnight drift."""
import json, collections
import pandas as pd
from source_panel import SER
recs = [json.loads(l) for l in open("fwd/snapshots.jsonl")]
rows = []
for r in recs:
    ts = pd.Timestamp(r["ts"])
    L = {(x["date"], x["name"], x["metric"]): x["share_percent"] for x in r.get("labs", [])}
    W = {(o["day"][:10], k, o["metric"]): v for o in r.get("wc", []) for k, v in o["chef_values"]}
    for s, (src, name, metric) in SER.items():
        D = W if src == "W" else L
        for (d, n, m), v in D.items():
            if n == name and m == metric:
                rows.append(dict(ts=ts, series=s, day=d, v=v, h=(ts - pd.Timestamp(d, tz="UTC")) / pd.Timedelta(hours=1)))
df = pd.DataFrame(rows)
if df.empty: raise SystemExit("no rows")
fin = df[df.h >= 25].sort_values("ts").groupby(["series", "day"]).v.first().rename("final")
last = df.sort_values("ts").groupby(["series", "day"]).v.last().rename("last")
x = df.join(fin, on=["series", "day"]).join(last, on=["series", "day"])
x["ref"] = x.final.fillna(x["last"]); x["err"] = (x.v - x.ref).abs()
x["hb"] = pd.cut(x.h, [0, 4, 8, 12, 16, 20, 24, 30, 48])
print("days with a final:", fin.reset_index().day.unique().tolist())
print(x.groupby(["hb"], observed=True).agg(n=("err", "size"), med_err=("err", "median"), p90=("err", lambda e: e.quantile(.9))).round(3))
print(x[x.h.between(0, 24)].groupby("series").agg(n=("err", "size"), med_err=("err", "median")).round(3))
x.to_csv("fwd/lockin_rows.csv", index=False)
