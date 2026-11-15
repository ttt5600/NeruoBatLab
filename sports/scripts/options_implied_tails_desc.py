"""Descriptives for options_implied_tails AFTER the edgelab dev evaluations (verdicts come only from edgelab).
  - maker filled vs unfilled win rates (adverse selection)
  - sigma x 1.25 sensitivity (implied_s1.25.parquet) for the taker rules
  - calibration of options-implied p vs Kalshi mid vs realised, by p bucket (tail candidates only)
DEV rows only (frame.parquet holds DEV only). Writes results/options_implied_tails/desc.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edgelab.rules import opt_tails as R  # noqa: E402

D = ROOT / "data" / "opt_tails"
OUT = ROOT / "results" / "options_implied_tails"


def clus_ci(df, val, cl, n=5000, seed=7):
    g = df.groupby(cl)[val].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    return [float(np.percentile(s / c, 2.5)), float(np.percentile(s / c, 97.5))]


def score(b, fr):
    res = fr.set_index("ticker")["result"]
    b = b.copy()
    b["win"] = (b["bet_id"].map(res) == b["side"]).astype(float)
    fee = np.where(b["price_source"] == "taker", 0.07 * b["price"] * (1 - b["price"]), 0.0)
    b["cost"] = b["price"] + fee
    b["pnl"] = b["win"] - b["cost"]
    if not len(b):
        return {"n": 0}
    roi = b.pnl.sum() / b.cost.sum()
    b["r"] = b.pnl / b.cost.mean()
    return {"n": int(len(b)), "clusters": int(b.cluster.nunique()), "roi": float(roi), "win": float(b.win.mean()),
            "mean_cost": float(b.cost.mean()), "roi_ci": clus_ci(b, "r", "cluster")}


def main():
    fr = pd.read_parquet(D / "frame.parquet")
    fr.attrs["tape"] = pd.read_parquet(D / "tape.parquet")
    fr = fr[fr.result.isin(["yes", "no"])].copy()
    out = {}
    # DESCRIPTIVE strata requested after the freeze (lead, 11-15): weekend close date and listing age at T
    fr["weekend"] = pd.to_datetime(fr["date"]).dt.dayofweek >= 5
    fr["age_h"] = (fr["T"] - fr["open_ts"]) / 3600
    for ac in ("crypto", "index"):
        tb = R.taker(fr, {"asset_class": ac})
        mb = R.maker(fr, {"asset_class": ac})
        info = fr.set_index("ticker")[["weekend", "age_h"]]
        strata = {"listing_age_h_quartiles": fr.loc[fr.asset_class == ac, "age_h"].quantile([0, .25, .5, .75, 1]).round(1).tolist()}
        for nm, b in (("taker", tb), ("maker", mb)):
            if not len(b):
                continue
            b = b.join(info, on="bet_id")
            strata[f"{nm}_weekend"] = score(b[b.weekend].drop(columns=["weekend", "age_h"]), fr)
            strata[f"{nm}_weekday"] = score(b[~b.weekend].drop(columns=["weekend", "age_h"]), fr)
            med = b.age_h.median()
            strata[f"{nm}_age_le_median({med:.1f}h)"] = score(b[b.age_h <= med].drop(columns=["weekend", "age_h"]), fr)
            strata[f"{nm}_age_gt_median"] = score(b[b.age_h > med].drop(columns=["weekend", "age_h"]), fr)
        out[f"{ac}_strata_descriptive"] = strata
    for ac in ("crypto", "index"):
        out[f"{ac}_taker"] = score(R.taker(fr, {"asset_class": ac}), fr)
        m = R.maker(fr, {"asset_class": ac, "return_unfilled": True})
        res = fr.set_index("ticker")["result"]
        m["win"] = (m["bet_id"].map(res) == m["side"]).astype(float)
        out[f"{ac}_maker_posted"] = {
            "posted": int(len(m)), "filled": int(m._filled.sum()),
            "win_filled": float(m.loc[m._filled, "win"].mean()) if m._filled.any() else None,
            "win_unfilled": float(m.loc[~m._filled, "win"].mean()) if (~m._filled).any() else None,
            "mean_q_filled": float(m.loc[m._filled, "price"].mean()) if m._filled.any() else None}
        out[f"{ac}_maker_filled"] = score(m[m._filled].drop(columns=["_filled", "win"]), fr)
    s = D / "implied_s1.25.parquet"
    if s.exists():
        imp = pd.read_parquet(s)[["ticker", "p_yes"]].rename(columns={"p_yes": "p125"})
        f2 = fr.merge(imp, on="ticker", how="left")
        f2["p_yes"] = f2["p125"]
        f2.attrs = fr.attrs
        for ac in ("crypto", "index"):
            out[f"{ac}_taker_sigma1.25"] = score(R.taker(f2, {"asset_class": ac}), f2)
    # calibration: tail-side implied prob vs Kalshi tail-side mid vs realised, quotes with spread <= 5c
    q = fr[np.isfinite(fr.yes_bid) & np.isfinite(fr.yes_ask) & (fr.yes_ask - fr.yes_bid <= 0.05 + 1e-9)].copy()
    q["tail_yes"] = q.p_yes <= 0.10
    q["pT"] = np.where(q.tail_yes, q.p_yes, 1 - q.p_yes)
    mid = (q.yes_bid + q.yes_ask) / 2
    q["midT"] = np.where(q.tail_yes, mid, 1 - mid)
    q["hitT"] = np.where(q.tail_yes, q.result == "yes", q.result == "no").astype(float)
    q["b"] = pd.cut(q.pT, [0, .005, .02, .05, .10001])
    cal = q.groupby(["asset_class", "b"], observed=True).agg(n=("hitT", "size"), pT=("pT", "mean"),
                                                             midT=("midT", "mean"), realised=("hitT", "mean"))
    out["calibration"] = cal.reset_index().astype({"b": str}).to_dict("records")
    for ac in ("crypto", "index"):
        z = q[q.asset_class == ac]
        out[f"{ac}_brier_tail"] = {"implied": float(((z.pT - z.hitT) ** 2).mean()),
                                   "mid": float(((z.midT - z.hitT) ** 2).mean()), "n": int(len(z))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "desc.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
