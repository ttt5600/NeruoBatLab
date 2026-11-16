"""ot1_extension descriptives (run only AFTER the one edgelab dev evaluation of each trial).
Strata: pre-2025-07 vs since, BTC vs ETH, X2 by arm, series. Cluster bootstrap by crypto|close date (B 4000, seed 0).
-> results/ot1_extension/desc.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edgelab.rules.opt_tails_ext import taker  # noqa: E402

R = ROOT / "results" / "ot1_extension"


def boot(cl, pnl, cost, B=4000):
    rng = np.random.default_rng(0)
    u, inv = np.unique(cl, return_inverse=True)
    P = np.bincount(inv, pnl, len(u)); C = np.bincount(inv, cost, len(u))
    idx = rng.integers(0, len(u), (B, len(u)))
    r = P[idx].sum(1) / C[idx].sum(1)
    return float(pnl.sum() / cost.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)), len(u)


def bets(fr, arm_set):
    b = taker(fr, {"arm_set": arm_set})
    m = fr.drop_duplicates("ticker").set_index("ticker")
    b["result"] = m.loc[b.bet_id, "result"].to_numpy()
    b["cur"] = m.loc[b.bet_id, "cur"].to_numpy()
    b["series"] = m.loc[b.bet_id, "series"].to_numpy()
    b["date"] = pd.to_datetime(m.loc[b.bet_id, "date"].to_numpy())
    a = fr.set_index(["ticker", "T"])["arm"]
    b["arm"] = [a.get((t, int(d.timestamp()))) for t, d in zip(b.bet_id, b.decided_at)]
    fee = 0.07 * b.price * (1 - b.price)
    b["cost"] = b.price + fee
    b["pnl"] = (b.result == b.side).astype(float) - b["cost"]
    return b


def summ(b):
    if not len(b):
        return {"n": 0}
    roi, lo, hi, k = boot(b.cluster.to_numpy(), b.pnl.to_numpy(), b.cost.to_numpy())
    return {"n": len(b), "clusters": k, "roi": round(roi, 4), "ci": [round(lo, 4), round(hi, 4)],
            "win": round(float((b.result == b.side).mean()), 4), "mean_price": round(float(b.price.mean()), 4)}


def main():
    fr = pd.read_parquet(R / "frame.parquet")
    fr = fr[fr.result.isin(["yes", "no"])]
    out = {}
    for a in ("x1", "x2", "x3"):
        b = bets(fr, a)
        d = {"all": summ(b), "pre_2025_07": summ(b[b.date < "2025-07-01"]), "since_2025_07": summ(b[b.date >= "2025-07-01"])}
        for c, g in b.groupby("cur"):
            d[f"cur_{c}"] = summ(g)
        for s, g in b.groupby("series"):
            d[f"series_{s}"] = summ(g)
        if a == "x2":
            for s, g in b.groupby("arm"):
                d[f"arm_{s}"] = summ(g)
        d["side"] = {s: summ(g) for s, g in b.groupby("side")}
        out[a] = d
        b.to_parquet(R / f"bets_{a}.parquet")
    (R / "desc.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
