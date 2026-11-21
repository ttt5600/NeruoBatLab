"""Verify the UFC-live horn stamp against the Kalshi KXUFCDISTANCE tape (DIST YES -> ~1 when the fight goes the distance).
Price-only + timing; DEV 2026 fights that have a DIST market."""
import json
from pathlib import Path
import numpy as np, pandas as pd
OUT = Path(__file__).resolve().parent
F = pd.read_parquet(OUT / "fights_timed.parquet").set_index("event_ticker")
recs = {}
for cf in ("dev_raw.jsonl", "dev_raw_rev.jsonl", "dev_raw_v2.jsonl"):
    p = OUT / cf
    if p.exists():
        for l in open(p):
            try: x = json.loads(l)
            except json.JSONDecodeError: continue
            recs.setdefault(x["event_ticker"], x)
rows = []
for ev, r in recs.items():
    if not r.get("tapeD") or ev not in F.index or pd.isna(F.loc[ev, "t_horn"]): continue
    d = pd.DataFrame(r["tapeD"]); d["t"] = pd.to_datetime(d.created_time, utc=True, format="ISO8601"); d["px"] = d.yes_price_dollars.astype(float)
    d = d.sort_values("t"); h = F.loc[ev, "t_horn"]
    hi = d.px >= .97; ok = hi[::-1].cummin()[::-1]
    t97 = d.t[ok].iloc[0] if ok.any() else pd.NaT
    pre = d[d.t <= h - pd.Timedelta(seconds=20)]
    rows.append({"ev": ev, "t97_minus_horn": (t97 - h).total_seconds() if pd.notna(t97) else np.nan,
                 "dist_last_px_20s_before_horn": pre.px.iloc[-1] if len(pre) else np.nan, "n": len(d),
                 "dist_close_minus_horn": (pd.Timestamp(r["dist_close"]) - h).total_seconds() if r.get("dist_close") else np.nan})
x = pd.DataFrame(rows)
print("fights with DIST tape", len(x))
print(x[["t97_minus_horn", "dist_last_px_20s_before_horn", "dist_close_minus_horn"]].describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
