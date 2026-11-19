"""KFG1 dev scorer (prereg notes/kalshi_fg_markets.md). --mech: pipeline/timestamp checks only, no outcomes/returns."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps
OUT = Path(__file__).resolve().parent
THR = 0.04
def fee(p):
    return np.ceil(np.round(100 * 0.07 * 10 * p * (1 - p), 9)) / 1000
def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan
def quotes():
    rows = []
    for l in (OUT / "candles_dev.jsonl").open():
        j = json.loads(l); T = j["T"]; best = None
        for c in j["c"]:
            e = int(c["end_period_ts"])
            if e <= T and (best is None or e > int(best["end_period_ts"])): best = c
        g = lambda k: fnum(best[k].get("close_dollars", best[k].get("close"))) if best and best.get(k) else np.nan
        rows.append(dict(ticker=j["ticker"], T=T, q_end=int(best["end_period_ts"]) if best else np.nan,
                         ask=g("yes_ask"), bid=g("yes_bid"), n_c=len(j["c"])))
    return pd.DataFrame(rows)
def build():
    m = pd.read_parquet(OUT / "markets_matched.parquet")
    m = m[m.week <= 7]
    q = quotes()
    traps.check_epoch_seconds(q["T"], "T")
    d = m.merge(q, on="ticker", how="inner", validate="one_to_one")
    pr = pd.read_parquet(OUT / "preds_2026.parquet")
    d = d.merge(pr[["game_id", "team", "implied", "indoor", "p1", "p2", "p3", "p4"]], on=["game_id", "team"], how="left", validate="many_to_one")
    d["p"] = d.apply(lambda r: r[f"p{r.k}"], axis=1)
    kick = (d.kick - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    assert ((d["T"] - kick) == -3600).all()
    assert (d.q_end.dropna() <= d["T"][d.q_end.notna()]).all()
    return d
def bets(d, pcol="p"):
    yc, nc = d.ask, 1 - d.bid
    ok = (d.bid > 0)
    ey = np.where(ok & yc.between(.03, .97), d[pcol] - yc - fee(yc), -np.inf)
    en = np.where(ok & nc.between(.03, .97), (1 - d[pcol]) - nc - fee(nc), -np.inf)
    x = pd.concat([d.assign(side="yes", cost=yc, edge=ey), d.assign(side="no", cost=nc, edge=en)])
    x = x.sort_values("edge", ascending=False).drop_duplicates(["game_id", "team"])
    return x[x.edge >= THR].copy()
def boot(b, cl, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = b.groupby(cl).agg(p=("pnl", "sum"), c=("tc", "sum")); P, C = g.p.to_numpy(), g.c.to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g))); r = P[idx].sum(1) / C[idx].sum(1)
    return np.percentile(r, [2.5, 97.5]).round(4).tolist(), float((r <= 0).mean())
def score(b):
    b["tc"] = b.cost + fee(b.cost)
    b["win"] = np.where(b.side == "yes", b.result == "yes", b.result == "no").astype(float)
    b["pnl"] = b.win - b.tc
    return b
if __name__ == "__main__":
    d = build()
    print("dev markets with candles row", len(d), "with quote at T", d.ask.notna().sum(), "games", d.game_id.nunique())
    print("quote staleness min (T - q_end)/60:", ((d["T"] - d.q_end) / 60).describe().round(1).to_dict())
    print("missing preds", d.p.isna().sum())
    print("spread (ask-bid) by k:", d.assign(sp=d.ask - d.bid).groupby("k").sp.median().to_dict())
    print("mid by k vs model:", d.assign(mid=(d.ask + d.bid) / 2).groupby("k")[["mid", "p"]].mean().round(3).to_dict())
    if "--mech" in sys.argv: sys.exit()
    d = d[d.result.isin(["yes", "no"])]
    d["y"] = (d.result == "yes").astype(int); d["mid"] = (d.ask + d.bid) / 2
    q = d.dropna(subset=["mid"])
    print("\n== calibration per strike (dev): n, outcome rate, Kalshi mid, model p, Brier mid, Brier model")
    for k, g in q.groupby("k"):
        print(k, len(g), round(g.y.mean(), 3), round(g.mid.mean(), 3), round(g.p.mean(), 3), round(((g.mid - g.y) ** 2).mean(), 4), round(((g.p - g.y) ** 2).mean(), 4))
    print("all Brier mid", round(((q.mid - q.y) ** 2).mean(), 4), "model", round(((q.p - q.y) ** 2).mean(), 4))
    q["mb"] = pd.cut(q.mid, [0, .1, .2, .35, .5, .65, .8, .9, 1])
    print(q.groupby("mb", observed=True).agg(n=("y", "size"), y=("y", "mean"), mid=("mid", "mean")).round(3).to_string())
    b = score(bets(d))
    roi = b.pnl.sum() / b.tc.sum()
    cig, pg = boot(b, "game_id"); cid, pdt = boot(b, "gd")
    print(f"\n== KFG1 dev: n {len(b)} (yes {int((b.side=='yes').sum())}, no {int((b.side=='no').sum())}) wins {int(b.win.sum())} ROI {roi:+.4f} "
          f"game-CI {cig} p {pg:.4f} | date-CI {cid} p {pdt:.4f}; mean edge {b.edge.mean():.3f}")
    print(b.groupby(["side", "k"]).agg(n=("pnl", "size"), pnl=("pnl", "sum"), cost=("tc", "mean"), p=("p", "mean"), win=("win", "mean")).round(3).to_string())
    rng = np.random.default_rng(1); pl = []
    tg = d[["game_id", "team"]].drop_duplicates().reset_index(drop=True)
    for i in range(200):
        perm = tg.sample(frac=1, random_state=rng.integers(1e9)).reset_index(drop=True)
        mp = dict(zip(zip(tg.game_id, tg.team), zip(perm.game_id, perm.team)))
        src = d.set_index(["game_id", "team", "k"]).p
        dd = d.copy(); dd["pp"] = [src.get((*mp[(g, t)], k), np.nan) for g, t, k in zip(d.game_id, d.team, d.k)]
        bb = score(bets(dd, "pp")); pl.append(bb.pnl.sum() / bb.tc.sum() if len(bb) else np.nan)
    pl = np.array(pl); pct = float((pl < roi).mean() * 100)
    print(f"placebo ROI mean {np.nanmean(pl):+.4f} p95 {np.nanpercentile(pl,95):+.4f}; observed pct {pct:.1f}")
    b.to_parquet(OUT / "bets_dev.parquet")
    json.dump(dict(n=len(b), roi=roi, ci_game=cig, p_game=pg, ci_date=cid, p_date=pdt, placebo_mean=float(np.nanmean(pl)),
                   placebo_p95=float(np.nanpercentile(pl, 95)), placebo_pct=pct), open(OUT / "result_dev.json", "w"), indent=1)
