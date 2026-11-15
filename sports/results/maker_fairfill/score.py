"""Score the pre-registered maker_fairfill rule (notes/maker_fairfill.md). DEV only.

    ../.venv_quant/bin/python results/maker_fairfill/score.py   -> rows.csv, summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
XS = (0, 1, 2)
BAR = {"nba": 1.35e-3, "cfb": 2.38e-3, "nfl": 3.70e-4, "cbb": 1.25e-2}   # edgelab bars after registration


def mfee(p, n=100):
    return math.ceil(round(n * 0.0175 * p * (1 - p) * 100, 6)) / 100 / n


def tape(tk):
    t = pd.DataFrame(json.load(open(OUT / "tapes" / f"{tk}.json"))["trades"])
    if t.empty:
        return t
    t["ts"] = pd.to_datetime(t.created_time, utc=True, format="ISO8601")
    t["yes"] = t.yes_price_dollars.astype(float)
    t["no"] = t.no_price_dollars.astype(float)
    t["n"] = t.count_fp.astype(float)
    return t


def build(cancel_min=5):
    B = pd.read_csv(OUT / "bets.csv")
    rows = []
    for r in B.itertuples():
        t = tape(r.ticker)
        T, end = pd.Timestamp(r.T), pd.Timestamp(r.start) - pd.Timedelta(minutes=cancel_min)
        assert T < end, r
        w = t[(t.ts > T) & (t.ts <= end)] if len(t) else t
        mid = r.side_ask - r.spread_c / 200
        for x in XS:
            b = math.floor(round((mid - x / 100) * 100, 6)) / 100
            b = min(b, round(r.side_ask - 0.01, 2))
            px = (w.yes if r.yes_side else w.no) if len(w) else pd.Series(dtype=float)
            thru = w.n[px < b - 1e-9].sum() if len(w) else 0.0
            touch = w.n[px <= b + 1e-9].sum() if len(w) else 0.0
            cost = b + mfee(b)
            rows.append(dict(lg=r.lg, mk=r.mk, key=r.key, date=r.date, ticker=r.ticker, x=x, bid=b, cost=cost,
                             fill=thru > 0, fill_touch=touch > 0, fill_full=thru >= 100, thru_ct=thru,
                             touch_ct=touch, won=r.won, clv=r.clv, taker_cost=r.taker_cost,
                             roi=(1 - cost) / cost if r.won else -1.0, n_tr_window=len(w),
                             hrs=(end - T) / pd.Timedelta(hours=1)))
    return pd.DataFrame(rows)


def cboot(d, f, B=4000, seed=0):
    """f(sub-frame) -> scalar; resample ET dates with replacement."""
    rng = np.random.default_rng(seed)
    g = {k: v for k, v in d.groupby("date")}
    ks = list(g)
    est = f(d)
    vals = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(ks, len(ks))])
        v = f(s)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return [round(float(est), 4), round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4),
            round(float((vals <= 0).mean()), 4)]


def gap(s, col="fill"):
    a, b = s[s[col]], s[~s[col]]
    return a.won.mean() - b.won.mean() if len(a) and len(b) else np.nan


def summ(d, col="fill"):
    f = d[d[col]]
    out = dict(n=len(d), dates=d.date.nunique(), fill_rate=round(d[col].mean(), 3), n_filled=len(f),
               W_filled=int(f.won.sum()), win_filled=round(f.won.mean(), 3) if len(f) else None,
               win_unfilled=round(d[~d[col]].won.mean(), 3) if (~d[col]).any() else None,
               win_all=round(d.won.mean(), 3), avg_cost=round(f.cost.mean(), 4) if len(f) else None,
               thru_ct_median=float(d.thru_ct.median()), thru_ct_filled_median=float(f.thru_ct.median()) if len(f) else None)
    if len(f) >= 3 and f.date.nunique() >= 2:
        out["roi_filled"] = cboot(f, lambda s: s.roi.mean())
        out["clv_filled"] = cboot(f, lambda s: s.clv.mean())
    if len(f) and (~d[col]).any() and d.date.nunique() >= 2:
        out["as_gap"] = cboot(d, lambda s: gap(s, col))
    out["roi_all_if_filled"] = round(d.roi.mean(), 4)
    return out


def verdict(s, lg):
    r = s.get("roi_filled")
    if not r:
        return "NEEDS DATA"
    if r[3] < BAR[lg] and s["n_filled"] >= 15 and s["clv_filled"][0] > 0:
        return "PASS dev"
    a = s.get("as_gap")
    if r[2] < 0 or (a and a[2] < 0 and r[0] < 0):
        return "KILL"
    if r[0] > 0 and s["n_filled"] >= 15:
        return "GO forward paper"
    return "NEEDS DATA"


def main():
    d = build()
    d.to_csv(OUT / "rows.csv", index=False)
    S = {}
    for (lg, x), g in d.groupby(["lg", "x"]):
        s = summ(g)
        s["verdict"] = verdict(s, lg)
        S[f"{lg}_x{x}"] = s
        print(lg, x, json.dumps(s))
    for x in XS:
        g = d[d.x == x]
        S[f"pooled_x{x}"] = summ(g)
        S[f"pooled_x{x}_touch"] = summ(g, "fill_touch")
        S[f"pooled_x{x}_full100"] = summ(g, "fill_full")
        print("POOLED", x, json.dumps(S[f"pooled_x{x}"]))
    d30 = build(30)
    for x in XS:
        S[f"pooled_x{x}_cancel30"] = summ(d30[d30.x == x])
    json.dump(S, open(OUT / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
