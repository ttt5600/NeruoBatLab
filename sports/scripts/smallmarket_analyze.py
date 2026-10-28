"""Analysis for scripts/smallmarket_calibration.py (EXPLORATORY).  python scripts/smallmarket_analyze.py -> tables.json"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from smallmarket_calibration import HORIZONS, OUT

PB = [0, .05, .15, .35, .65, .85, .95, 1.0001]
PB_LBL = ["<5", "5-15", "15-35", "35-65", "65-85", "85-95", ">95"]
PM_RATE = {"crypto_fees": 0.07, "sports_fees": 0.05}  # docs.polymarket.com/trading/fees (2026-10-27): fee=C*rate*p*(1-p)
VB = [-1, 1e3, 1e4, 1e5, 1e6, 1e13]


def obs_pm() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "pm_priced2.parquet")
    sp = pd.read_parquet(OUT / "pm_live_books.parquet")
    sp = sp[(sp.volume > 0) & (sp.ask - sp.bid < 0.5)].assign(spr=lambda x: x.ask - x.bid)
    mid = (sp.ask + sp.bid) / 2
    sp["vb"] = pd.cut(sp.volume, VB, labels=False)
    sp["ext"] = (np.minimum(mid, 1 - mid) < 0.15).astype(int)
    prox = sp.groupby(["vb", "ext"]).spr.median().to_dict()
    rows = []
    for k in HORIZONS:
        g = df[df[f"p_{k}"].notna()].copy()
        g["h"], g["p"] = k, g[f"p_{k}"]
        rows.append(g)
    o = pd.concat(rows)
    o["y"] = o.y0
    o["venue"] = "polymarket"
    o["event"] = "pm" + o.event_id.astype(str)
    o["vb"] = pd.cut(o.volume, VB, labels=False).fillna(0).astype(int)
    o["ext"] = (np.minimum(o.p, 1 - o.p) < 0.15).astype(int)
    o["spread"] = [prox.get((a, b), 0.05) for a, b in zip(o.vb, o.ext)]
    o["yes_ask"] = np.minimum(o.p + o.spread / 2, 0.999)
    o["yes_bid"] = np.maximum(o.p - o.spread / 2, 0.001)
    ft = o.fee_schedule.map(lambda s: "crypto_fees" if '"rate": 0.25' in str(s) else ("sports_fees" if "0.0175" in str(s) else None))
    o["fee_rate"] = np.where(o.fees_enabled, ft.map(PM_RATE).fillna(0.05), 0.0)
    o["yesno"] = o.outcome0.eq("Yes")
    o["excl"] = o.neg_risk
    o["early_ok"] = True
    o["vol_life"] = o.volume
    o["v24"] = np.nan
    o["series"] = None
    return o[["venue", "event", "market_id", "cat", "h", "p", "y", "yes_bid", "yes_ask", "spread", "fee_rate", "yesno",
              "excl", "n_mkts_event", "vol_life", "v24", "early_ok", "series"]].rename(columns={"market_id": "mkt"})


def obs_k() -> pd.DataFrame:
    df = pd.read_parquet(OUT / "k_priced2.parquet")
    rows = []
    for k in HORIZONS:
        g = df.copy()
        g["h"] = k
        g["yes_bid"], g["yes_ask"], g["v24"] = g[f"bid_{k}"], g[f"ask_{k}"], g[f"v24_{k}"]
        rows.append(g)
    o = pd.concat(rows)
    o = o[(o.yes_bid > 0) & (o.yes_ask < 1) & (o.yes_ask > o.yes_bid)].copy()
    o["p"] = (o.yes_bid + o.yes_ask) / 2
    o["spread"] = o.yes_ask - o.yes_bid
    o["y"] = (o.result == "yes").astype(float)
    o["venue"] = "kalshi"
    o["event"] = "k" + o.event_ticker
    o["fee_rate"] = 0.07 * o.fee_mult
    o["yesno"] = True
    o["excl"] = o.mutually_exclusive.fillna(False).astype(bool)
    o["vol_life"] = o.volume
    # can_close_early is True for ~100% of markets, so flag ACTUAL early closes: closed > 1 day before expected expiry
    o["early_ok"] = ~((o.exp_time - o.close_time) > pd.Timedelta(days=1))
    return o[["venue", "event", "ticker", "cat", "h", "p", "y", "yes_bid", "yes_ask", "spread", "fee_rate", "yesno",
              "excl", "n_mkts_event", "vol_life", "v24", "early_ok", "series"]].rename(columns={"ticker": "mkt"})


def boot(d: pd.DataFrame, fn, B=300, seed=0):
    """Point estimate + 95% CI, resampling EVENTS with replacement."""
    codes, uniq = pd.factorize(d.event.values)
    n = len(uniq)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n + 1))
    rng = np.random.default_rng(seed)
    est = fn(d)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, n, n)
        ii = np.concatenate([order[bounds[i]:bounds[i + 1]] for i in pick])
        vals.append(fn(d.iloc[ii]))
    vals = np.array(vals, dtype=float)
    return est, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def brier(d):
    return float(np.mean((d.p.values - d.y.values) ** 2))


def ece(d, nb=10):
    p, y = d.p.values, d.y.values
    b = np.minimum((p * nb).astype(int), nb - 1)
    tot = 0.0
    for i in np.unique(b):
        m = b == i
        tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(tot / len(p))


def ls_gap(d):
    """Longshot overpricing (price points): mean longshot-side price - longshot-side win rate, longshot price .01-.20."""
    p, y = d.p.values, d.y.values
    q = np.minimum(p, 1 - p)
    w = np.where(p <= 0.5, y, 1 - y)
    m = (q >= 0.01) & (q <= 0.20)
    return float(q[m].mean() - w[m].mean()) if m.sum() else np.nan


def fee(a, rate):
    return rate * a * (1 - a)


def rule_frame(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    p = d.p.values
    if rule == "FAV":       # buy whichever side has mid in [.80,.98]
        yes = (p >= 0.80) & (p <= 0.98); no = (p <= 0.20) & (p >= 0.02)
    elif rule == "LONG":    # buy whichever side has mid in [.02,.20]
        yes = (p <= 0.20) & (p >= 0.02); no = (p >= 0.80) & (p <= 0.98)
    elif rule == "NO_LONG":  # YES/NO markets only: buy NO when YES mid in [.02,.20]
        yes = np.zeros(len(p), bool); no = d.yesno.values & (p <= 0.20) & (p >= 0.02)
    a = np.where(yes, d.yes_ask.values, 1 - d.yes_bid.values)
    mid = np.where(yes, p, 1 - p)
    win = np.where(yes, d.y.values, 1 - d.y.values)
    out = pd.DataFrame({"event": d.event.values, "cost": a + fee(a, d.fee_rate.values), "cost_mid": mid, "win": win,
                        "spread": d.spread.values})
    return out[yes | no]


def roi(r):
    return float((r.win.sum() - r.cost.sum()) / r.cost.sum()) if len(r) else np.nan


def roi_mid(r):
    return float((r.win.sum() - r.cost_mid.sum()) / r.cost_mid.sum()) if len(r) else np.nan


def R(x, k=4):
    return [round(v, k) for v in x]


def summarize(d: pd.DataFrame, B=300) -> dict:
    out = {"n": int(len(d)), "events": int(d.event.nunique())}
    if len(d) < 30 or out["events"] < 10:
        return out
    out["brier"] = R(boot(d, brier, B))
    out["ece"] = R(boot(d, ece, B))
    out["ls_gap"] = R(boot(d, ls_gap, B))
    out["med_spread"] = round(float(d.spread.median()), 4)
    for rule in ("FAV", "LONG", "NO_LONG"):
        r = rule_frame(d, rule)
        if len(r) >= 30 and r.event.nunique() >= 10:
            out[rule] = {"n": int(len(r)), "events": int(r.event.nunique()), "roi_net": R(boot(r, roi, B)),
                         "roi_mid": R(boot(r, roi_mid, B)), "med_spread": round(float(r.spread.median()), 3)}
    return out


def nbucket(n):
    return "1" if n <= 1 else ("2-4" if n <= 4 else ("5-10" if n <= 10 else "11+"))


def sums(d: pd.DataFrame) -> dict:
    """Q2: mutually exclusive events, every leg priced at the same horizon, exactly one leg settled YES."""
    res = {}
    for (venue, h), g in d[d.excl].groupby(["venue", "h"]):
        ev = g.groupby("event").agg(n=("p", "size"), N=("n_mkts_event", "first"), S=("p", "sum"), Sb=("yes_bid", "sum"),
                                    Sa=("yes_ask", "sum"), wins=("y", "sum"), cat=("cat", "first"))
        ev = ev[(ev.n == ev.N) & (ev.n >= 3) & (ev.wins == 1)]
        if len(ev) < 20:
            continue
        legs = g[g.event.isin(ev.index)].merge(ev[["S"]], left_on="event", right_index=True)
        legs["pn"] = legs.p / legs.S
        legs["Sg"] = pd.cut(legs.S, [0, 0.97, 1.03, 1.10, 99], labels=["<0.97", "0.97-1.03", "1.03-1.10", ">1.10"])
        tab = {}
        for sg, l in legs.groupby("Sg", observed=True):
            if l.event.nunique() < 10:
                continue
            lg = l[(l.p <= 0.20) & (l.p >= 0.02)]
            r = rule_frame(l, "NO_LONG")
            ent = {"events": int(l.event.nunique()), "legs": int(len(l)), "brier_raw": round(brier(l), 4),
                   "brier_norm": round(float(np.mean((l.pn - l.y) ** 2)), 4), "longshot_legs": int(len(lg))}
            if len(lg) >= 20:
                ent.update({"longshot_price": round(float(lg.p.mean()), 4), "longshot_winrate": round(float(lg.y.mean()), 4),
                            "NO_LONG_roi_net": R(boot(r, roi, 300)), "NO_LONG_roi_mid": round(roi_mid(r), 4)})
            tab[str(sg)] = ent
        res[f"{venue}_{h}"] = {
            "events": int(len(ev)), "S_median": round(float(ev.S.median()), 4),
            "S_q10_q90": [round(float(ev.S.quantile(.1)), 4), round(float(ev.S.quantile(.9)), 4)],
            "share_S_gt_1.03": round(float((ev.S > 1.03).mean()), 3), "share_S_lt_0.97": round(float((ev.S < 0.97).mean()), 3),
            "corr_S_vs_n": round(float(np.corrcoef(ev.S, ev.n)[0, 1]), 3),
            "S_by_n": ev.groupby(ev.n.map(nbucket)).S.median().round(4).to_dict(),
            "by_cat": ev.groupby("cat").S.agg(["size", "median"]).round(4).to_dict("index"),
            "by_S": tab}
        if venue == "kalshi":
            res[f"{venue}_{h}"]["arb_buy_all_NO_events"] = int((ev.Sb > 1.0).sum())
            res[f"{venue}_{h}"]["arb_buy_all_YES_events"] = int((ev.Sa < 1.0).sum())
            res[f"{venue}_{h}"]["Sbid_median"] = round(float(ev.Sb.median()), 4)
            res[f"{venue}_{h}"]["Sask_median"] = round(float(ev.Sa.median()), 4)
    return res


def main() -> int:
    parts = []
    if (OUT / "pm_priced2.parquet").exists():
        parts.append(obs_pm())
    if (OUT / "k_priced2.parquet").exists():
        parts.append(obs_k())
    d = pd.concat(parts, ignore_index=True)
    d = d[(d.p > 0) & (d.p < 1) & d.y.notna()].copy()
    d["pb"] = pd.cut(d.p, PB, labels=PB_LBL, right=False)
    d["nb"] = d.n_mkts_event.map(nbucket)
    # fixed lifetime-volume bins (many Kalshi markets have 0-1 contracts, so quintiles would split ties arbitrarily)
    # kalshi contracts: 0-1 | 2-100 | 101-1k | 1k-10k | >10k ; polymarket USD: <1k | 1k-10k | 10k-100k | 100k-1M | >1M
    kb = pd.cut(d.vol_life, [-1, 1, 100, 1e3, 1e4, 1e13], labels=False) + 1
    pb_ = pd.cut(d.vol_life, [-1, 1e3, 1e4, 1e5, 1e6, 1e13], labels=False) + 1
    d["volq"] = np.where(d.venue == "kalshi", kb, pb_).astype(int)
    d.to_parquet(OUT / "obs.parquet", index=False)
    T = {}
    for (v, h), g in d.groupby(["venue", "h"]):
        key = f"{v}_{h}"
        print(key, len(g), flush=True)
        T[key] = {"all": summarize(g)}
        T[key]["volq"] = {int(q): {**summarize(gq), "vol_median": float(gq.vol_life.median())} for q, gq in g.groupby("volq")}
        T[key]["cat"] = {c: summarize(gc) for c, gc in g.groupby("cat")}
        T[key]["nb"] = {c: summarize(gc) for c, gc in g.groupby("nb")}
        T[key]["pb"] = {str(c): {"n": int(len(gc)), "events": int(gc.event.nunique()), "price": round(float(gc.p.mean()), 4),
                                 "rate": round(float(gc.y.mean()), 4)} for c, gc in g.groupby("pb", observed=True)}
        T[key]["pb_x_vol"] = {f"{c}|{'low' if q <= 2 else 'high'}": {"n": int(len(gc)), "price": round(float(gc.p.mean()), 4),
                                                                     "rate": round(float(gc.y.mean()), 4)}
                              for (c, q), gc in g[g.volq != 3].assign(lh=lambda x: np.where(x.volq <= 2, 1, 5))
                              .groupby(["pb", "lh"], observed=True)}
        T[key]["cat_x_vol"] = {}
        for c, gc in g.groupby("cat"):
            lo, hi = gc[gc.volq <= 2], gc[gc.volq >= 4]
            if len(lo) >= 100 and len(hi) >= 100:
                T[key]["cat_x_vol"][c] = {"low": summarize(lo, B=200), "high": summarize(hi, B=200)}
        if v == "kalshi":
            T[key]["no_early_close"] = summarize(g[g.early_ok])
            g2 = g.assign(v24b=pd.cut(g.v24, [-1, 0, 10, 100, 1000, 1e13], labels=["0", "1-10", "11-100", "101-1k", ">1k"]))
            T[key]["v24"] = {str(b): summarize(gb) for b, gb in g2.groupby("v24b", observed=True)}
            T[key]["spread_by_volq"] = g.groupby("volq").spread.describe()[["25%", "50%", "75%"]].round(3).to_dict("index")
    T["sums"] = sums(d)
    (OUT / "tables.json").write_text(json.dumps(T, indent=1, default=str))
    print("-> data/smallmarket/tables.json", len(d), "observations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
