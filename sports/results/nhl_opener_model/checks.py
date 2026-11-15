"""Post-hoc diagnostics (NOT pre-registered): is trial A's CLV model information or reversion of a noisy L_K?"""
import json
import numpy as np
import pandas as pd
d = pd.read_pickle("kalshi_rows.pkl"); out = {}
for tr in "AB":
    x = d[(d.trial == tr) & d.c.notna() & d.L_K.notna() & d.L_close.notna()].copy()
    x["mc"] = x.m - x.c
    x = x.sort_values("Tdec").reset_index(drop=True)
    x["Lbar"] = [x.L_K[x.tip < x.Tdec[i]].tail(100).mean() for i in range(len(x))]  # placebo: no team info
    x = x[x.Lbar.notna()].reset_index(drop=True)
    y = (x.L_close - x.L_K).values
    X = np.column_stack([np.ones(len(x)), x.mc - x.L_K, x.Lbar - x.L_K])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(0); dd = x.date.astype(str).values; u = np.unique(dd)
    grp = {k: np.where(dd == k)[0] for k in u}; bs = []
    for _ in range(2000):
        ii = np.concatenate([grp[k] for k in rng.choice(u, len(u))]); bs.append(np.linalg.lstsq(X[ii], y[ii], rcond=None)[0])
    bs = np.array(bs)
    out[tr] = {"n": int(len(x)), "beta_model_gap": [float(beta[1]), *map(float, np.percentile(bs[:, 1], [2.5, 97.5]))],
               "beta_placebo_gap": [float(beta[2]), *map(float, np.percentile(bs[:, 2], [2.5, 97.5]))]}
    gp = x.Lbar - x.L_K; b = x[gp.abs() >= 0.4]
    out[tr]["placebo_k0.4"] = {"n": int(len(b)), "clv_goals": float((np.sign(gp[b.index]) * (b.L_close - b.L_K)).mean())}
    # model-only regression on the move of the DK open -> DK close (reference independent of L_K noise)
    z = x.dropna(subset=["dk_open_med", "dk_close_med"])
    gz = (z.mc - z.dk_open_med).values; yz = (z.dk_close_med - z.dk_open_med).values
    out[tr]["dk_move_on_model_gap_vs_dk_open_slope"] = float(np.polyfit(gz, yz, 1)[0])
    out[tr]["corr_dk_move_model_gap"] = float(np.corrcoef(gz, yz)[0, 1])
json.dump(out, open("checks.json", "w"), indent=1); print(json.dumps(out, indent=1))
