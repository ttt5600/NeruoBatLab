"""Dev scorer (pre-registered, notes/cbb_opener_model.md): de-biased model vs DK total open, 2023-24 + 2024-25.

    python results/cbb_opener_model/dev.py  -> dev.json, dev_bets.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
N_C, MIN_C, K = 200, 50, 4.0


def am_p(a):
    return 1 / (1 + (a / 100 if a > 0 else 100 / -a))


def am_win(a):
    return a / 100 if a > 0 else 100 / -a


def lines(s):
    o = pd.read_parquet(OUT / f"odds_{s}.parquet")
    o["pr"] = o.provider.str.replace(" ", "").map({"DraftKings": 0, "ESPNBET": 1})
    o = o[o.pr.notna() & o.total_open.notna() & o.total_close.notna()].sort_values("pr").drop_duplicates("eid")
    return o[["eid", "provider", "total_open", "total_close", "over_open", "under_open"]]


def debias(P):
    """c(g) = mean(model - open) over last N_C games with ET date strictly earlier; >= MIN_C else NaN."""
    P = P.sort_values(["date_et", "start", "eid"]).reset_index(drop=True)
    gap = (P.model - P.total_open).values
    c = np.full(len(P), np.nan)
    dates = P.date_et.values
    for d in np.unique(dates):
        idx = np.where(dates < d)[0]
        if len(idx) >= MIN_C:
            c[dates == d] = gap[idx[-N_C:]].mean()
    P["c"] = c
    return P


def bets(P, k):
    P = P.copy()
    P["adj"] = P.model - P.c - P.total_open
    B = P[P.c.notna() & (P.adj.abs() >= k)].copy()
    B["side"] = np.where(B.adj > 0, 1, -1)
    m = (B.final - B.total_open) * B.side
    B["res"] = np.sign(m)  # +1 win, 0 push, -1 loss
    po = B.over_open.map(am_p) / (B.over_open.map(am_p) + B.under_open.map(am_p))
    p_side = np.where(B.side > 0, po, 1 - po)
    B["roi_nv"] = np.where(B.res > 0, 1 / p_side - 1, np.where(B.res < 0, -1.0, 0.0))
    pr = np.where(B.side > 0, B.over_open, B.under_open)
    B["roi_px"] = np.where(B.res > 0, pd.Series(pr).map(am_win).values, np.where(B.res < 0, -1.0, 0.0))
    B["clv"] = (B.total_close - B.total_open) * B.side
    return B


def cboot(B, col, n=4000, seed=0):
    """Cluster bootstrap over ET dates; returns mean, 2.5, 97.5, one-sided p(mean<=0)."""
    B = B[B[col].notna()]
    grp = B.groupby("date_et")[col].agg(["sum", "count"])
    s, c = grp["sum"].values, grp["count"].values
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(s), (n, len(s)))
    bs = s[i].sum(1) / c[i].sum(1)
    return dict(mean=round(float(s.sum() / c.sum()), 4), lo=round(float(np.percentile(bs, 2.5)), 4),
                hi=round(float(np.percentile(bs, 97.5)), 4), p=round(float((bs <= 0).mean()), 4), n=int(c.sum()))


if __name__ == "__main__":
    allP = []
    for s in (2024, 2025):
        P = pd.read_parquet(OUT / f"preds_{s}.parquet").merge(lines(s), on="eid")
        P = P[P.over_open.notna() & P.under_open.notna()]
        allP.append(debias(P))
    P = pd.concat(allP)
    res = {"n_games_with_open": int(len(P)), "n_with_c": int(P.c.notna().sum()),
           "gap_raw_mean": round(float((P.model - P.total_open).mean()), 3),
           "mae_model": round(float((P.model - P.final).abs().mean()), 3),
           "mae_open": round(float((P.total_open - P.final).abs().mean()), 3),
           "mae_close": round(float((P.total_close - P.final).abs().mean()), 3),
           "mae_model_debiased": round(float((P.model - P.c - P.final).abs().mean()), 3)}
    for k in (K, 3.0, 5.0, 6.0):
        B = bets(P, k)
        r = {m: cboot(B, m) for m in ("roi_nv", "roi_px", "clv")}
        r["by_season_nv"] = {int(s): round(float(x.roi_nv.mean()), 4) for s, x in B.groupby("season")}
        r["by_season_px"] = {int(s): round(float(x.roi_px.mean()), 4) for s, x in B.groupby("season")}
        r["by_season_clv"] = {int(s): round(float(x.clv.mean()), 3) for s, x in B.groupby("season")}
        r["overs"], r["unders"] = int((B.side > 0).sum()), int((B.side < 0).sum())
        r["win"] = round(float((B.res > 0).sum() / max((B.res != 0).sum(), 1)), 4)
        res[f"k{k:g}"] = r
        if k == K:
            B.to_csv(OUT / "dev_bets.csv", index=False)
    json.dump(res, open(OUT / "dev.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
