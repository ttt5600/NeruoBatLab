"""DEV re-score of M3_N10 with a live-calibrated queue haircut (exploratory, NOT a promotion trial).
Live (q4_tape.csv, excluding the 6 orders cancelled externally on 11-04): orders whose tape printed at <= q but never
below q filled at rate h = k/n; orders traded through (< q) filled ~always. Apply h to dev fills that had no
trade-through. Bootstrap: dates + Beta(k+1, n-k+1) draw for h."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
import kalshi_temp_maker as K

EXT = {"KXHIGHTTTN-26NOV03-B74.5", "KXHIGHDEN-26NOV03-B61.5", "KXHIGHTLV-26NOV03-B72.5", "KXHIGHMIA-26NOV04-B87.5",
       "KXHIGHDEN-26NOV04-B60.5", "KXHIGHDEN-26NOV04-T61"}
t = pd.read_csv("results/e4_live_audit/q4_tape.csv")
t = t[~t.ticker.isin(EXT)]
t["live_fill"] = t.filled > 0
atq = t[(t.vol_le_q >= 10) & (t.vol_lt_q <= 0)]
thr = t[t.vol_lt_q > 0]
k, n = int(atq.live_fill.sum()), len(atq)
out = {"live_orders_ex_ext_cancel": int(len(t)), "live_fill_rate_ex_ext": round(t.live_fill.mean(), 3),
       "model_N10_rate_ex_ext": round((t.vol_le_q >= 10).mean(), 3),
       "at_q_only_fill": f"{k}/{n}", "tradethrough_fill": f"{int(thr.live_fill.sum())}/{len(thr)}"}

p = pd.read_parquet(K.PICKS); p["date"] = pd.to_datetime(p["date"])
tr = pd.read_parquet(K.TRADES); qk = pd.read_parquet(K.QUOTES)
mk, bet, filled = K.variant(p, tr, qk, "M3_N10")
base = K.taker(p)
q = K.maker_price(p, "3")
through = K.fill_flags(p, tr, q - 0.01 + 1e-6, 1, p.close_ts.to_numpy())  # any print at no_price <= q - 1c, i.e. < q
won = p.won.to_numpy()
out["dev_fills"] = int(filled.sum()); out["dev_fills_tradethrough"] = int((filled & through).sum())
out["dev_losers_filled"] = int((filled & (won == 0)).sum()); out["dev_losers_filled_tradethrough"] = int((filled & through & (won == 0)).sum())
w = np.where(filled & ~through, 1.0, 0.0)  # haircut weight applies here
dates = p.date.dt.strftime("%Y-%m-%d").to_numpy()
u, inv = np.unique(dates, return_inverse=True)
rng = np.random.default_rng(0)
B = 4000
res_units, res_d, res_roi = [], [], []
for _ in range(B):
    h = rng.beta(k + 1, n - k + 1)
    wt = np.where(w > 0, h, filled.astype(float))
    idx = rng.integers(0, len(u), len(u))
    cnt = np.bincount(idx, minlength=len(u))[inv]
    units = (mk * wt * cnt).sum(); fills = (wt * cnt).sum(); tk = (base * cnt).sum()
    res_units.append(units); res_d.append(units - tk); res_roi.append(units / fills)
h0 = k / n
wt0 = np.where(w > 0, h0, filled.astype(float))
out.update({"h_point": round(h0, 3), "units_point": round(float((mk * wt0).sum()), 2),
            "units_model": round(float(mk.sum()), 2), "taker_units": round(float(base.sum()), 2),
            "roi_fill_point": round(float((mk * wt0).sum() / wt0.sum()), 4), "roi_fill_model": round(float(mk[filled].mean()), 4),
            "units_ci": [round(float(x), 2) for x in np.quantile(res_units, [.025, .975])],
            "maker_minus_taker_point": round(float((mk * wt0).sum() - base.sum()), 2),
            "maker_minus_taker_ci": [round(float(x), 2) for x in np.quantile(res_d, [.025, .975])],
            "roi_fill_ci": [round(float(x), 4) for x in np.quantile(res_roi, [.025, .975])]})
json.dump(out, open("results/e4_live_audit/q2b_queue_haircut.json", "w"), indent=1)
print(json.dumps(out, indent=1))
