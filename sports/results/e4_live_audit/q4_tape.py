"""Live maker orders vs the public tape (read-only GETs): (a) does the dev fill model (>= N printed at <= q after post)
agree with the real fill?  (b) capacity: printed volume at <= q / < q after post, vs 1x/2x/4x of our size."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import kalshi_temp_maker as K

d = pd.read_parquet("results/e4_live_audit/live_rows.parquet")
m = d[(d.exec_style == "maker") & d.status.isin(["filled", "no_fill", "partial"])].copy()
rows = []
for r in m.itertuples():
    t0 = int(pd.Timestamp(r.order_ts).timestamp()); t1 = int(r.expiration_time)
    tp = pd.DataFrame(K._tape(r.ticker, t0, t1))
    q = float(r.maker_q)
    if len(tp):
        tp = tp[(tp.ts > t0) & (tp.ts <= t1)]
    le = tp[tp.no_price <= q + 1e-9]["count"].sum() if len(tp) else 0.0
    lt = tp[tp.no_price < q - 1e-9]["count"].sum() if len(tp) else 0.0
    at = le - lt
    tot = tp["count"].sum() if len(tp) else 0.0
    rows.append({"ticker": r.ticker, "cohort": r.cohort, "q": q, "count": float(r["count"]) if False else float(r._asdict()["count"]),
                 "filled": float(r.filled), "result": r.paper_result, "vol_le_q": le, "vol_lt_q": lt, "vol_at_q": at,
                 "vol_total": tot, "n_trades": len(tp)})
t = pd.DataFrame(rows)
t.to_csv("results/e4_live_audit/q4_tape.csv", index=False)
t["live_fill"] = t.filled > 0
for n in (1, 10):
    t[f"model_N{n}"] = t.vol_le_q >= n
out = {"orders": int(len(t)), "live_fill_rate": round(t.live_fill.mean(), 3)}
for n in (1, 10):
    c = pd.crosstab(t[f"model_N{n}"], t.live_fill)
    out[f"model_N{n}_fill_rate"] = round(t[f"model_N{n}"].mean(), 3)
    out[f"crosstab_N{n}"] = {f"model={a},live={b}": int(c.loc[a, b]) for a in c.index for b in c.columns}
out["tradethrough_lt_q_rate_among_live_fills"] = round((t[t.live_fill].vol_lt_q > 0).mean(), 3)
out["tradethrough_lt_q_rate_among_live_nofills"] = round((t[~t.live_fill].vol_lt_q > 0).mean(), 3)
f = t[t.live_fill]
for k in (2, 4):
    # necessary condition for k x size: printed volume at <= q >= k * our filled size (ignores queue ahead -> optimistic)
    out[f"vol_le_q_ge_{k}x_size"] = f"{int((f.vol_le_q >= k * f['count']).sum())}/{len(f)}"
    # conservative: volume strictly through our price >= k x size (level cleared, would fill regardless of queue)
    out[f"vol_lt_q_ge_{k}x_size"] = f"{int((f.vol_lt_q >= k * f['count']).sum())}/{len(f)}"
out["median_vol_le_q_filled"] = float(f.vol_le_q.median())
out["median_vol_le_q_nofill"] = float(t[~t.live_fill].vol_le_q.median())
json.dump(out, open("results/e4_live_audit/q4_tape.json", "w"), indent=1)
print(json.dumps(out, indent=1))
print(t.sort_values(["live_fill", "vol_le_q"]).to_string())
