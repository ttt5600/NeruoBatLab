"""Kalshi test (pre-registered, notes/cbb_opener_model.md; edgelab cbb_t1_kalshi_taker-4dd8f3e6).
At Kalshi first-tight hour (>= 30 min pre-tip): S = anchor strike (mid nearest 50c; tie -> favourable to our side);
bet if |model - c - S| >= 4; over -> YES ask, under -> NO = 1 - bid; + exact taker fee; graded final vs S;
CLV = DK close - S signed. c from 2026-27 games (model - DK open), last 200, >= 50, strictly earlier ET dates.

    python results/cbb_opener_model/kalshi.py -> kalshi_rows.csv, kalshi.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
KX = OUT.parent / "kalshi_openers_xleague"
sys.path.insert(0, str(OUT))
from dev import N_C, MIN_C, K, lines  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def fee(p, n=100):
    return math.ceil(round(n * 0.07 * p * (1 - p) * 100, 6)) / 100 / n


def boot(v, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [float(v.mean()) if len(v) else None, None, None, None]
    bs = np.random.default_rng(seed).choice(v, (4000, len(v))).mean(1)
    return [round(v.mean(), 4), round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4),
            round(float((bs <= 0).mean()), 4)]


P = pd.read_parquet(OUT / "preds_2027.parquet").merge(lines(2027), on="eid", how="left")
H = P[P.total_open.notna()].sort_values(["date_et", "start", "eid"])
gap, hd = (H.model - H.total_open).values, H.date_et.values
cmap = {}
for d in np.unique(P.date_et.values):
    idx = np.where(hd < d)[0]
    cmap[pd.Timestamp(d)] = gap[idx[-N_C:]].mean() if len(idx) >= MIN_C else np.nan
P["c"] = P.date_et.map(cmap)

out = []
for tag in ("", "_cbbom"):
    R = pd.read_csv(KX / f"q1_cbb{tag}_rows.csv")
    R = R[(R.mk == "total")]
    C = pd.read_parquet(KX / f"cbb{tag}_candles.parquet")
    G = pd.read_parquet(KX / f"cbb{tag}_games.parquet")
    Km = pd.read_parquet(KX / f"cbb{tag}_kmarkets.parquet")
    ev2eid = Km.drop_duplicates("event").merge(G[["key", "event_id"]].drop_duplicates("key"), on="key").set_index("event").event_id.astype(str)
    for r in R.itertuples():
        eid = ev2eid.get(r.event)
        p = P[P.eid == eid]
        base = dict(event=r.event, eid=eid)
        if p.empty:
            out.append(dict(base, status="no model")); continue
        p = p.iloc[0]
        if pd.isna(r.ft_t):
            out.append(dict(base, status="never tight")); continue
        T, occ = pd.Timestamp(r.ft_t), pd.Timestamp(r.occ)
        if occ - T < pd.Timedelta(minutes=30):
            out.append(dict(base, status="ft < 30 min pre-tip")); continue
        if pd.isna(p.c):
            out.append(dict(base, status="no c")); continue
        t = int((T - EPOCH) // pd.Timedelta(seconds=1))
        c = C[(C.event == r.event) & (C.end <= t) & (C.end >= t - 6 * 3600) & C.bid.notna() & C.ask.notna()]
        q = c.sort_values("end").groupby("strike").tail(1)
        q = q[(q.bid > 0) & (q.ask < 1)].copy()
        if q.empty:
            out.append(dict(base, status="no quote")); continue
        mdl = p.model - p.c
        q["d50"] = ((q.bid + q.ask) / 2 - .5).abs().round(6)
        best = q[q.d50 == q.d50.min()]
        # tie -> strike favourable to our side: over prefers the lower strike, under the higher
        up = mdl > best.strike.mean()
        z = best.sort_values("strike").iloc[0 if up else -1]
        S = float(z.strike)
        gp = mdl - S
        row = dict(base, status="ok", ft_t=T, tip=occ, model=p.model, c=p.c, S=S, gap=gp, open=p.total_open,
                   close=p.total_close, final=p.final, bid=z.bid, ask=z.ask, spread_c=round((z.ask - z.bid) * 100))
        if abs(gp) < K:
            out.append(dict(row, bet=False)); continue
        up = gp > 0
        px = z.ask if up else 1 - z.bid
        cost = px + fee(px)
        won = (p.final > S) if up else (p.final < S)
        out.append(dict(row, bet=True, side="over" if up else "under", px=px, cost=cost, won=won,
                        pnl=(1 - cost) if won else -cost, roi=((1 - cost) if won else -cost) / cost,
                        clv=((p.total_close - S) * (1 if up else -1)) if pd.notna(p.total_close) else np.nan))
D = pd.DataFrame(out)
D.to_csv(OUT / "kalshi_rows.csv", index=False)
B = D[D.get("bet", False) == True]  # noqa: E712
ok = D[D.status == "ok"]
res = dict(status=D.status.value_counts().to_dict(), n_ok=len(ok), n_bets=len(B),
           overs=int((B.side == "over").sum()) if len(B) else 0,
           wins=int((B.won == True).sum()) if len(B) else 0,  # noqa: E712
           roi_at_cost=boot(B.roi) if len(B) else None, pnl_per_contract=boot(B.pnl) if len(B) else None,
           clv_pts=boot(B.clv.dropna()) if len(B) else None, mean_cost=round(float(B.cost.mean()), 4) if len(B) else None,
           c_range=[round(float(P.c.min()), 2), round(float(P.c.max()), 2)],
           S_minus_open_abs=round(float((ok.S - ok.open).abs().mean()), 3) if len(ok) else None,
           ok_gap_abs_mean=round(float(ok.gap.abs().mean()), 3) if len(ok) else None)
for k in (3.0, 5.0, 6.0):  # descriptive: re-threshold on the same rows (bet side/price recomputed)
    x = ok[ok.gap.abs() >= k].copy()
    up = x.gap > 0
    px = np.where(up, x.ask, 1 - x.bid)
    cost = px + np.array([fee(v) for v in px])
    won = np.where(up, x.final > x.S, x.final < x.S)
    res[f"k{k:g}_roi"] = boot(np.where(won, 1 - cost, -cost) / cost) + [len(x)]
json.dump(res, open(OUT / "kalshi.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
