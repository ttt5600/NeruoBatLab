"""CS2E dev scoring (prereg notes/cs2_model.md). Usage: dev.py [--preds preds.parquet] [--split dev] [--mech]
--mech: pipeline/timestamp checks only, no returns printed."""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps
OUT = Path(__file__).resolve().parent
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
DEV0, SEAL0 = pd.Timestamp("2026-04-15", tz="UTC"), pd.Timestamp("2026-10-19 13:45", tz="UTC")
PCOL, NMIN, THR = "p_series_32", 10, 0.05
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d


def fee(p):
    return np.ceil(np.round(100 * 0.07 * 10 * p * (1 - p), 9)) / 1000


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def quotes():
    rows = []
    for line in (OUT / "candles_raw.jsonl").open():
        j = json.loads(line)
        s0 = j["start_ts"]
        T = s0 - 3600
        best, close = None, None
        for c in j["c"]:
            e = int(c["end_period_ts"])
            if e <= T and (best is None or e > int(best["end_period_ts"])):
                best = c
            if e <= s0 and (close is None or e > int(close["end_period_ts"])):
                close = c
        g = lambda c, k: fnum((c or {}).get(k, {}).get("close", (c or {}).get(k, {}).get("close_dollars"))) if c else np.nan
        rows.append(dict(ticker=j["ticker"], start_ts=s0, q_end=int(best["end_period_ts"]) if best else np.nan,
                         ask=g(best, "yes_ask"), bid=g(best, "yes_bid"), ask0=g(close, "yes_ask"), bid0=g(close, "yes_bid"),
                         n_c=len(j["c"])))
    return pd.DataFrame(rows)


def boot(df, col_pnl, col_cost, cluster, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = df.groupby(cluster).agg(p=(col_pnl, "sum"), c=(col_cost, "sum"))
    P, C = g.p.to_numpy(), g.c.to_numpy()
    idx = rng.integers(0, len(g), size=(n, len(g)))
    r = P[idx].sum(1) / C[idx].sum(1)
    return P.sum() / C.sum(), np.percentile(r, [2.5, 97.5]), (r <= 0).mean()


def main():
    preds = pd.read_parquet(OUT / arg("--preds", "preds.parquet"))
    qt = quotes()
    traps.check_epoch_seconds(qt.start_ts, "start_ts")
    d = preds.merge(qt.add_suffix("_a"), left_on="ta", right_on="ticker_a", how="inner")
    d = d.merge(qt.add_suffix("_b"), left_on="tb", right_on="ticker_b", how="inner")
    d["start_s"] = (d.start - EPOCH) // pd.Timedelta(seconds=1)
    assert (d.start_s == d.start_ts_a).all() and (d.start_s == d.start_ts_b).all()
    d["T_s"] = d.start_s - 3600
    ok = d.q_end_a.notna() & d.q_end_b.notna()
    assert (d.loc[ok, "q_end_a"] <= d.loc[ok, "T_s"]).all() and (d.loc[ok, "q_end_b"] <= d.loc[ok, "T_s"]).all()
    split = arg("--split", "dev")
    if split == "dev":
        d = d[(d.start >= DEV0) & (d.start < SEAL0) & d.timed]
    elif split == "warm":
        d = d[d.start < DEV0]
    else:
        raise SystemExit("sealed is not opened by this script without the lead's yes")
    print(f"{split}: events with both candle sets {len(d)}; with quotes at T {ok.loc[d.index].sum()}")
    d = d[ok.loc[d.index]].copy()
    if "--mech" in sys.argv:
        print("quote age (T - q_end) h:", ((d.T_s - d.q_end_a) / 3600).describe().round(2).to_dict())
        print("ask+ask sum:", (d.ask_a + d.ask_b).describe().round(3).to_dict())
        print("start range", d.start.min(), d.start.max())
        return
    # payoff per side
    for s in "ab":
        res, sv = d[f"res_{s}"], d[f"sv_{s}"]
        d[f"pay_{s}"] = np.where(res == "yes", 1.0, np.where(res == "no", 0.0, sv))
    d["pa"] = d[PCOL]; d["pb"] = 1 - d[PCOL]
    valid = lambda s: (d[f"ask_{s}"] >= 0.05) & (d[f"ask_{s}"] <= 0.95) & (d[f"bid_{s}"] > 0)
    for s, p in (("a", "pa"), ("b", "pb")):
        d[f"edge_{s}"] = np.where(valid(s), d[p] - (d[f"ask_{s}"] + fee(d[f"ask_{s}"])), -np.inf)
    d["side"] = np.where(d.edge_a >= d.edge_b, "a", "b")
    d["edge"] = np.maximum(d.edge_a, d.edge_b)
    pick = lambda c: np.where(d.side == "a", d[c + "_a"], d[c + "_b"])
    d["ask"], d["pay"], d["ask0"] = pick("ask"), pick("pay"), pick("ask0")
    d["cost"] = d.ask + fee(d.ask)
    d["pnl"] = d.pay - d.cost
    d["date"] = d.start.dt.tz_convert("America/New_York").dt.date
    d["mid_a"] = (d.ask_a + d.bid_a) / 2
    elig = d[d[f"n_series"] >= NMIN]
    out = {"split": split, "preds": arg("--preds", "preds.parquet"), "events_quoted": int(len(d)), "eligible": int(len(elig))}
    for thr in [0.02, 0.03, 0.05, 0.08]:
        b = elig[elig.edge >= thr].sort_values("start")
        if len(b) < 10:
            out[f"thr{thr}"] = {"n": int(len(b))}; continue
        roi, ci, p = boot(b, "pnl", "cost", "date")
        roi_e, ci_e, _ = boot(b, "pnl", "cost", "ev")
        h = len(b) // 2
        h1 = b.iloc[:h].pnl.sum() / b.iloc[:h].cost.sum(); h2 = b.iloc[h:].pnl.sum() / b.iloc[h:].cost.sum()
        clv = (b.ask0 - b.ask).mean()
        out[f"thr{thr}"] = dict(n=int(len(b)), roi=round(roi, 4), ci_date=np.round(ci, 4).tolist(), p_le0=round(p, 4),
                                ci_event=np.round(ci_e, 4).tolist(), half1=round(h1, 4), half2=round(h2, 4),
                                clv_ask_T_to_start=round(clv, 4), mean_ask=round(b.ask.mean(), 3),
                                win=round(b.pay.mean(), 3), mean_edge=round(b.edge.mean(), 3))
    # diagnostics on eligible binary events
    e2 = elig[elig.res_a.isin(["yes", "no"])].copy()
    y = (e2.res_a == "yes").astype(float)
    ll = lambda p: float(-(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(np.clip(1 - p, 1e-4, 1 - 1e-4))).mean())
    mid = e2.mid_a.clip(0.01, 0.99)
    out["diag"] = dict(n=int(len(e2)), ll_model=round(ll(e2.pa), 4), ll_mid=round(ll(mid), 4),
                       brier_model=round(float(((e2.pa - y) ** 2).mean()), 4), brier_mid=round(float(((mid - y) ** 2).mean()), 4))
    # stacking: y ~ logit(mid) + logit(model), date-clustered bootstrap of model coefficient
    import statsmodels.api as sm
    X = pd.DataFrame({"lm": np.log(mid / (1 - mid)), "le": np.log(e2.pa / (1 - e2.pa))})
    X = sm.add_constant(X)
    fit = sm.Logit(y.values, X.values).fit(disp=0)
    dates = e2.start.dt.tz_convert("America/New_York").dt.date.values
    ud = np.unique(dates); rng = np.random.default_rng(1); co = []
    idx_by = {u: np.where(dates == u)[0] for u in ud}
    for _ in range(500):
        ii = np.concatenate([idx_by[u] for u in rng.choice(ud, len(ud))])
        try:
            co.append(sm.Logit(y.values[ii], X.values[ii]).fit(disp=0).params[2])
        except Exception:
            pass
    out["stack"] = dict(coef_mid=round(float(fit.params[1]), 3), coef_elo=round(float(fit.params[2]), 3),
                        coef_elo_ci=np.round(np.percentile(co, [2.5, 97.5]), 3).tolist())
    # calibration of mid in deciles
    e2["bin"] = pd.qcut(mid, 10, duplicates="drop")
    out["calib_mid"] = e2.assign(y=y).groupby("bin", observed=True).agg(mid=("mid_a", "mean"), y=("y", "mean"), n=("y", "size")).round(3).reset_index(drop=True).to_dict("records")
    print(json.dumps(out, indent=1, default=str))
    tag = Path(arg("--preds", "preds.parquet")).stem
    (OUT / f"result_{split}_{tag}.json").write_text(json.dumps(out, indent=1, default=str))
    elig.to_parquet(OUT / f"rows_{split}_{tag}.parquet", index=False)


if __name__ == "__main__":
    main()
