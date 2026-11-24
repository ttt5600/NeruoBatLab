"""wx_bot_flow scoring (pre-registered in notes/wx_bot_flow.md section 3). Read-only, local files.
    ../.venv_quant/bin/python scripts/wx_bot_flow_score.py  -> results/wx_bot_flow/score.json, frame.parquet
"""
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results" / "wx_bot_flow"
CITY = {"KXHIGHNY": "NY", "KXHIGHCHI": "CHI", "KXHIGHMIA": "MIA", "KXHIGHAUS": "AUS", "KXHIGHLAX": "LAX",
        "KXHIGHDEN": "DEN", "KXHIGHPHIL": "PHIL"}
LST = {"NY": 5, "CHI": 6, "MIA": 5, "AUS": 6, "LAX": 8, "DEN": 7, "PHIL": 5}   # hours behind UTC, standard time
OFF = {"NY": .18, "CHI": .60, "MIA": -.02, "LAX": .22, "DEN": -.40, "AUS": 0.0, "PHIL": 0.0}  # omcheck.json
rng = np.random.default_rng(20261124)


def member_max():
    g = pd.read_parquet(R / "gefs_points.parquet")
    assert g.tK.between(200, 330).all(), g.tK.describe()
    assert not g.duplicated(["run_init", "member", "fhr", "point"]).any()
    g["tF"] = (g.tK - 273.15) * 9 / 5 + 32
    g["city"] = g.point.str.split("_").str[0]
    g["kind"] = g.point.str.split("_").str[1]
    g["run_h"] = g.run_init.dt.hour
    g["D"] = (g.run_init + pd.Timedelta(days=1)).dt.normalize()
    g["vh"] = (g.valid - g.D) / pd.Timedelta(hours=1)          # hours after 00Z D
    bot = g[(g.kind == "bot") & (g.vh >= 0) & (g.vh <= 24)].copy()
    bot["tF"] = bot.tF - bot.city.map(OFF)
    bm = bot.groupby(["city", "D", "run_h", "member"]).tF.agg(["max", "size"]).reset_index()
    stn = g[g.kind == "stn"].copy()
    stn["lo"] = stn.city.map(LST)
    stn = stn[(stn.vh >= stn.lo) & (stn.vh <= stn.lo + 24)]
    sm = stn.groupby(["city", "D", "run_h", "member"]).tF.agg(["max", "size"]).reset_index()
    return bm, sm


def bracket_p(maxes, st, fl, cap):
    t = np.asarray(maxes)
    if st == "between":
        w = ((t >= fl - .5) & (t < cap + .5)).mean()
    elif st == "greater":
        w = (t >= fl + .5).mean()
    else:   # less: AwpDemon threshold = cap, YES if t < cap + .5 (its own off-by-one, replicated)
        w = (t < cap + .5).mean()
    return float(np.clip(w, .05, .95))


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b


def boot(df, cols, ycol, key="sigdate", n=2000):
    if len(df) < 30:
        return {"insufficient_n": len(df)}, np.full((1, len(cols) + 1), np.nan)
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols]); y = df[ycol].values
    b = ols(X, y)
    dates = df[key].unique(); grp = {d: np.where(df[key].values == d)[0] for d in dates}
    bs = []
    for _ in range(n):
        idx = np.concatenate([grp[d] for d in rng.choice(dates, len(dates))])
        bs.append(ols(X[idx], y[idx]))
    bs = np.array(bs)
    return {c: dict(beta=round(float(b[i + 1]), 4), ci=[round(float(np.percentile(bs[:, i + 1], 2.5)), 4),
                                                       round(float(np.percentile(bs[:, i + 1], 97.5)), 4)])
            for i, c in enumerate(cols)}, bs


def main():
    bm, sm = member_max()
    MB = {k: v["max"].values for k, v in bm.groupby(["city", "D", "run_h"])}
    MS = {k: v["max"].values for k, v in sm.groupby(["city", "D", "run_h"])}
    nmem = bm.groupby(["city", "D", "run_h"]).size()
    mk = pd.read_parquet(R / "markets.parquet")
    mk["city"] = mk.series.map(CITY)
    mk["D"] = pd.to_datetime(mk.D)
    mk["won"] = (mk.result == "yes").astype(int)
    rows = []
    for r in mk.itertuples():
        rec = dict(ticker=r.ticker, event=r.event, city=r.city, D=r.D, st=r.strike_type, won=r.won, result=r.result)
        for rh in (6, 12):
            k = (r.city, r.D, rh)
            rec[f"p{rh:02d}"] = bracket_p(MB[k], r.strike_type, r.floor, r.cap) if k in MB else np.nan
            rec[f"ps{rh:02d}"] = bracket_p(MS[k], r.strike_type, r.floor, r.cap) if k in MS else np.nan
            rec[f"nm{rh:02d}"] = nmem.get(k, 0)
        k7 = (r.city, r.D - pd.Timedelta(days=7), 12)
        rec["p12_wrong"] = bracket_p(MB[k7], r.strike_type, r.floor, r.cap) if k7 in MB else np.nan
        rows.append(rec)
    fr = pd.DataFrame(rows)
    fr["sigdate"] = fr.D - pd.Timedelta(days=1)
    # prices: hourly candles, end_ts = end of the hour
    c = pd.read_parquet(R / "candles.parquet")
    traps.check_epoch_seconds(c.end_ts, "candle end_ts")
    c["end"] = pd.to_datetime(c.end_ts, unit="s", utc=True)
    c = c.merge(fr[["ticker", "sigdate"]], on="ticker")
    c["hr"] = (c.end - c.sigdate.dt.tz_localize("UTC")) / pd.Timedelta(hours=1)
    for h, nm in ((16, "16"), (19, "19")):
        cc = c[c.hr == h].set_index("ticker")
        fr[f"ask{nm}"] = fr.ticker.map(cc.yes_ask); fr[f"bid{nm}"] = fr.ticker.map(cc.yes_bid)
    fr["mid19"] = (fr.ask19 + fr.bid19) / 2
    # flow
    t = pd.read_parquet(R / "trades.parquet")
    t["ts"] = traps.epoch_seconds(t.created_time)
    traps.check_epoch_seconds(t.ts, "trade ts")
    t = t.merge(fr[["ticker", "sigdate"]], on="ticker")
    t["hr"] = (t.ts - traps.epoch_seconds(t.sigdate.dt.tz_localize("UTC"))) / 3600
    t["signed"] = np.where(t.taker_side == "yes", t["count"], -t["count"])

    def nyf(lo, hi, name):
        w = t[(t.hr >= lo) & (t.hr < hi)].groupby("ticker").agg(net=("signed", "sum"), vol=("count", "sum"), n=("count", "size"))
        fr[f"nyf_{name}"] = fr.ticker.map(w.net / w.vol); fr[f"vol_{name}"] = fr.ticker.map(w.vol).fillna(0)
        fr[f"net_{name}"] = fr.ticker.map(w.net).fillna(0)
    nyf(16, 19, "after"); nyf(15, 16, "before")
    for h in (14, 15, 16, 17, 18, 19):
        nyf(h, h + 1, f"h{h}")
    fr["edge12"] = fr.p12 - fr.ask16; fr["edge06"] = fr.p06 - fr.ask16
    fr["edge12s"] = fr.ps12 - fr.ask16; fr["edge06s"] = fr.ps06 - fr.ask16
    fr["edge12w"] = fr.p12_wrong - fr.ask16
    fr["edge12_19"] = fr.p12 - fr.ask19
    fr.to_parquet(R / "frame.parquet")
    out = {"n_markets": len(fr), "n_events": int(fr.event.nunique()), "members_min": int(fr[["nm06", "nm12"]].min().min()),
           "ask16_coverage": float(fr.ask16.notna().mean()), "trades": len(t)}
    base = fr[(fr.ask16 >= .03) & (fr.ask16 <= .90) & fr.p12.notna() & fr.p06.notna()]
    A = base[base.vol_after > 0]
    out["H1"], bsA = boot(A, ["edge12", "edge06", "ask16"], "nyf_after"); out["H1"]["n"] = len(A); out["H1"]["dates"] = int(A.sigdate.nunique())
    B = base[base.vol_before > 0]
    out["P1_before"], bsB = boot(B, ["edge12", "edge06", "ask16"], "nyf_before"); out["P1_before"]["n"] = len(B)
    # beta1 - beta1_P1, paired over the same resampled dates (same rng draws are not shared; use independent boot)
    diff = bsA[:, 1] - bsB[:, 1]
    out["H1_minus_P1"] = dict(diff=round(out["H1"]["edge12"]["beta"] - out["P1_before"]["edge12"]["beta"], 4),
                              ci=[round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)])
    W = A[A.edge12w.notna()]
    out["P2_wrongdate"], _ = boot(W, ["edge12w", "edge06", "ask16"], "nyf_after"); out["P2_wrongdate"]["n"] = len(W)
    out["P2_wrongdate"]["dates"] = int(W.sigdate.nunique()) if len(W) else 0
    out["H1b_bot_vs_station"], _ = boot(A[A.edge12s.notna()], ["edge12", "edge06", "edge12s", "edge06s", "ask16"], "nyf_after")
    # descriptive: by hour
    out["by_hour"] = {}
    for h in (14, 15, 16, 17, 18, 19):
        H = base[base[f"vol_h{h}"] > 0]
        if len(H) > 50:
            r_, _ = boot(H, ["edge12", "edge06", "ask16"], f"nyf_h{h}", n=500)
            out["by_hour"][h] = dict(edge12=r_["edge12"], n=len(H))
    # signal counts
    out["signals_at_16Z"] = dict(
        bot_buy_yes_08=int(((base.edge12 >= .08)).sum()), bot_buy_no_08=int((((1 - base.p12) - (1 - base.bid16)) >= .08).sum()),
        new_buy_yes_08=int(((base.edge12 >= .08) & (base.edge06 < .08)).sum()), n=len(base),
        bot_buy_yes_cheap=int(((base.edge12 >= .08) & (base.ask16 <= .15)).sum()))
    # H2 pricing
    P = fr[(fr.mid19 >= .02) & (fr.mid19 <= .90) & fr.p12.notna() & fr.result.isin(["yes", "no"])].copy()
    P["resid"] = P.won - P.mid19
    out["H2"], _ = boot(P, ["edge12_19", "mid19"], "resid"); out["H2"]["n"] = len(P)   # mid19 control: amended 04:30Z after a scrambled-outcome dry run showed the shared-price mechanical slope
    ch = P[P.mid19 <= .15]
    g1, g0 = ch[ch.edge12_19 >= .08], ch[ch.edge12_19 < .08]
    dates = ch.sigdate.unique(); bsd = []
    for _ in range(2000):
        s = rng.choice(dates, len(dates)); m = pd.concat([ch[ch.sigdate == d] for d in s])
        a, b0 = m[m.edge12_19 >= .08], m[m.edge12_19 < .08]
        if len(a) and len(b0):
            bsd.append(a.resid.mean() - b0.resid.mean())
    out["H2_cheap"] = dict(n_botbuy=len(g1), n_other=len(g0), resid_botbuy=round(float(g1.resid.mean()), 4) if len(g1) else None,
                           resid_other=round(float(g0.resid.mean()), 4), hit_botbuy=round(float(g1.won.mean()), 4) if len(g1) else None,
                           mid_botbuy=round(float(g1.mid19.mean()), 4) if len(g1) else None,
                           diff=round(float(g1.resid.mean() - g0.resid.mean()), 4) if len(g1) else None,
                           ci=[round(float(np.percentile(bsd, 2.5)), 4), round(float(np.percentile(bsd, 97.5)), 4)] if bsd else None)
    json.dump(out, open(R / "score.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
