"""Step-3 depth check (pre-registered in notes/SIZING_RULE.md 2026-11-25): orig-7 maker orders since 2026-11-12 vs the
public Kalshi tape. Read-only GETs (kalshi_temp_maker._tape, cached), >= 1.1 s between tickers.
Input: results/e4_live_audit/q5_orders.json (dumped from the live journal). Output: q5_depth_step3.json + q5_depth_step3.csv."""
import json, sys, time
import pandas as pd
sys.path.insert(0, "scripts")
import kalshi_temp_maker as K

NEW = 48
orders = json.load(open("results/e4_live_audit/q5_orders.json"))
rows = []
for o in orders:
    t0 = int(pd.Timestamp(o["ts"]).timestamp()); t1 = int(o["expiration_time"])
    tp = pd.DataFrame(K._tape(o["ticker"], t0, t1)); time.sleep(1.1)
    q = float(o["maker_q"])
    if len(tp):
        tp = tp[(tp.ts > t0) & (tp.ts <= t1)]
    le = float(tp[tp.no_price <= q + 1e-9]["count"].sum()) if len(tp) else 0.0
    lt = float(tp[tp.no_price < q - 1e-9]["count"].sum()) if len(tp) else 0.0
    rows.append({**{k: o[k] for k in ("ticker", "ts", "maker_q", "count", "filled", "result")}, "vol_le_q": le, "vol_lt_q": lt})
t = pd.DataFrame(rows); t.to_csv("results/e4_live_audit/q5_depth_step3.csv", index=False)
f = t[t.filled > 0]; u = t[t.filled <= 0]
def n(df, col, k): return f"{int((df[col] >= k).sum())}/{len(df)}"
out = {"orders": len(t), "filled": len(f), "new_size": NEW,
       "PRIMARY_filled_vol_lt_q_ge_2x_new(96)": n(f, "vol_lt_q", 2 * NEW),
       "primary_pass": bool((f.vol_lt_q >= 2 * NEW).sum() > len(f) / 2),
       "filled_vol_lt_q_ge_1x_new(48)": n(f, "vol_lt_q", NEW),
       "filled_vol_le_q_ge_2x_new(96)": n(f, "vol_le_q", 2 * NEW),
       "unfilled_vol_lt_q_ge_48": n(u, "vol_lt_q", NEW), "unfilled_vol_le_q_ge_1": n(u, "vol_le_q", 1),
       "median_vol_lt_q_filled": float(f.vol_lt_q.median()) if len(f) else None,
       "median_vol_le_q_filled": float(f.vol_le_q.median()) if len(f) else None}
json.dump(out, open("results/e4_live_audit/q5_depth_step3.json", "w"), indent=1)
print(json.dumps(out, indent=1)); print(t.to_string())
