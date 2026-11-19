"""CS2F sealed score (prereg notes/cs2_model.md). Run ONCE after the sealed candle fetch completes."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.argv = sys.argv[:1]
import dev, traps
OUT = Path(__file__).resolve().parent
S0, S1 = pd.Timestamp("2026-10-19 13:45", tz="UTC"), pd.Timestamp("2026-11-16", tz="UTC")
BAR = 0.05 / 21
q = pd.read_parquet(OUT / "preds.parquet")
q = q[(q.start >= S0) & (q.start < S1) & q.timed]
qt = dev.quotes()
traps.check_epoch_seconds(qt.start_ts, "start_ts")
d = q.merge(qt.add_suffix("_a"), left_on="ta", right_on="ticker_a").merge(qt.add_suffix("_b"), left_on="tb", right_on="ticker_b")
d["start_s"] = (d.start - dev.EPOCH) // pd.Timedelta(seconds=1)
assert (d.start_s == d.start_ts_a).all() and (d.start_s == d.start_ts_b).all()
T = d.start_s - 3600
d = d[d.q_end_a.notna() & d.q_end_b.notna()].copy()
assert (d.q_end_a <= T.loc[d.index]).all() and (d.q_end_b <= T.loc[d.index]).all(), "post-T quote"
print("sealed events in window", len(q), "with quotes at T", len(d))
for s in "ab":
    d[f"pay_{s}"] = np.where(d[f"res_{s}"] == "yes", 1.0, np.where(d[f"res_{s}"] == "no", 0.0, d[f"sv_{s}"]))
    d[f"mid_{s}"] = (d[f"ask_{s}"] + d[f"bid_{s}"]) / 2
rows = []
for s, o in (("a", "b"), ("b", "a")):
    x = d[d[f"mid_{s}"] > d[f"mid_{o}"]].copy()
    x["ask"], x["bid"], x["mid"], x["pay"] = x[f"ask_{s}"], x[f"bid_{s}"], x[f"mid_{s}"], x[f"pay_{s}"]
    rows.append(x)
f = pd.concat(rows)
f = f[(f.ask <= 0.95) & (f.bid > 0)].copy()
assert f.ev.is_unique
f["cost"] = f.ask + dev.fee(f.ask); f["pnl"] = f.pay - f.cost
f["date"] = f.start.dt.tz_convert("America/New_York").dt.date
out = {}
for name, lo, hi, sub in (("PRIMARY_65_80", .65, .80, None), ("desc_50_65", .50, .65, None), ("desc_80_95", .80, .95, None),
                          ("desc_65_80_elo_eligible", .65, .80, "elig")):
    b = f[(f.mid >= lo) & (f.mid < hi)]
    if sub:
        b = b[b.n_series >= 10]
    roi, ci, p = dev.boot(b, "pnl", "cost", "date")
    out[name] = dict(n=int(len(b)), dates=int(b.date.nunique()), roi=round(roi, 4), ci=np.round(ci, 4).tolist(), p=round(p, 5),
                     win=round(b.pay.mean(), 3), mean_ask=round(b.ask.mean(), 3), scalar=int((b.pay % 1 != 0).sum()))
pr = out["PRIMARY_65_80"]
out["bar"] = BAR
out["verdict"] = "PASS" if (pr["ci"][0] > 0 and pr["p"] < BAR) else "KILL"
print(json.dumps(out, indent=1))
(OUT / "result_sealed_cs2f.json").write_text(json.dumps(out, indent=1))
f.to_parquet(OUT / "rows_sealed_cs2f.parquet", index=False)
