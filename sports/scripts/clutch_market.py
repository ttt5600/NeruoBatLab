"""clutch_market: does Kalshi's in-game NBA price misprice comebacks conditional on a team's comeback / clutch record?

Pre-registration: notes/clutch_market.md (written 2026-11-12 before outcomes were joined). DEV ONLY.

    python scripts/clutch_market.py build     # traits (prior games only) + snapshots -> data/clutch_market/frame.parquet
    python scripts/clutch_market.py calib     # descriptive calibration by trait tercile -> results/clutch_market/calib.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))
from traps import check_epoch_seconds, epoch_seconds  # noqa: E402

SEAL = pd.Timestamp("2026-01-15")
DEV = (pd.Timestamp("2025-04-15"), pd.Timestamp("2026-01-14"))
OUT_D = ROOT / "data" / "clutch_market"
RES = ROOT / "results" / "clutch_market"
FRAME = OUT_D / "frame.parquet"
CANDLES = OUT_D / "candles.parquet"
WIN = 82
MIN_N = 10
CELLS_D = [(6, 10, "D1"), (11, 15, "D2"), (16, 99, "D3")]


def load_states():
    cols = ["eid", "seq", "period", "clock_s", "hs", "as", "wall", "home_t", "away_t", "home_won", "date", "season",
            "gtype", "F_home", "F_away"]
    s = pd.read_parquet(ROOT / "data" / "nba_ingame_lineup_states.parquet", columns=cols)
    s = s[s["date"] < SEAL].copy()          # sealed rows never read beyond this line
    s["m"] = s["hs"] - s["as"]
    return s.sort_values(["eid", "seq"]).reset_index(drop=True)


def game_table(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby("eid").agg(date=("date", "first"), home_t=("home_t", "first"), away_t=("away_t", "first"),
                             home_won=("home_won", "first"), mmin=("m", "min"), mmax=("m", "max"))
    late = s[((s["period"] == 4) & (s["clock_s"] <= 300) & (s["m"].abs() <= 5)) | (s["period"] >= 5)]
    g["clutch"] = g.index.isin(set(late["eid"]))
    g = g.reset_index()
    h = pd.DataFrame({"eid": g.eid, "date": g.date, "team": g.home_t, "won": g.home_won,
                      "trailed10": g.mmin <= -10, "clutch": g.clutch})
    a = pd.DataFrame({"eid": g.eid, "date": g.date, "team": g.away_t, "won": 1 - g.home_won,
                      "trailed10": g.mmax >= 10, "clutch": g.clutch})
    return g, pd.concat([h, a], ignore_index=True).sort_values(["team", "date", "eid"]).reset_index(drop=True)


def traits(tg: pd.DataFrame) -> pd.DataFrame:
    """For each team-game: T_CB, T_CL over the last WIN games dated strictly before this game's date."""
    out = []
    for team, d in tg.groupby("team"):
        d = d.reset_index(drop=True)
        dates = d["date"].to_numpy()
        for i in range(len(d)):
            prior = d.iloc[: np.searchsorted(dates, dates[i], side="left")].tail(WIN)
            cb = prior[prior.trailed10]
            cl = prior[prior.clutch]
            out.append({"eid": d.eid[i], "team": team, "n_cb": len(cb), "n_cl": len(cl),
                        "T_CB": cb.won.mean() if len(cb) >= MIN_N else np.nan,
                        "T_CL": cl.won.mean() if len(cl) >= MIN_N else np.nan,
                        "prior_last": prior.date.max() if len(prior) else pd.NaT, "date": d.date[i]})
    t = pd.DataFrame(out)
    assert (t.dropna(subset=["prior_last"]).eval("prior_last < date")).all(), "lookahead in trait"
    return t


def load_candles(eids) -> pd.DataFrame:
    c = pd.concat([pd.read_parquet(ROOT / "data" / "nba_ingame.parquet"),
                   pd.read_parquet(ROOT / "data" / "nba_late_candles_2025po.parquet")], ignore_index=True)
    c = c[(c["date"] >= DEV[0]) & (c["date"] <= DEV[1]) & c["eid"].isin(eids)]
    c = c.drop_duplicates(["eid", "side", "end"])
    check_epoch_seconds(c["end"], "candle end")
    return c


def build():
    OUT_D.mkdir(parents=True, exist_ok=True)
    s = load_states()
    g, tg = game_table(s)
    t = traits(tg)
    dev_eids = set(g.loc[(g.date >= DEV[0]) & (g.date <= DEV[1]), "eid"])
    c = load_candles(dev_eids)
    c.drop(columns=["result"]).to_parquet(CANDLES)
    s = s[s["eid"].isin(set(c.eid))].copy()
    s["wall_s"] = epoch_seconds(s["wall"])
    check_epoch_seconds(s["wall_s"], "play wall")
    rows = []
    for eid, ce in c.groupby("eid"):
        se = s[s.eid == eid].sort_values("wall_s")
        piv = ce.pivot_table(index="end", columns="side", values=["bid", "ask"]).dropna()
        if piv.empty or se.empty:
            continue
        q = pd.DataFrame({"end": piv.index.to_numpy(float), "h_bid": piv[("bid", "home")].to_numpy(),
                          "h_ask": piv[("ask", "home")].to_numpy(), "a_bid": piv[("bid", "away")].to_numpy(),
                          "a_ask": piv[("ask", "away")].to_numpy()})
        q["k"] = q["end"] - 60
        st = se[["wall_s", "period", "clock_s", "m", "F_home", "F_away", "seq"]].rename(columns={"wall_s": "k"})
        x = pd.merge_asof(q.sort_values("k"), st.sort_values("k").assign(state_wall=st["k"].to_numpy()),
                          on="k", direction="backward").dropna(subset=["m"])
        x = x[x["period"].between(2, 4) & (x["m"] != 0)]
        game_end = float(se["wall_s"].max())
        x = x[x["end"] <= game_end]
        g0 = g[g.eid == eid].iloc[0]
        for r in x.itertuples(index=False):
            home_trails = r.m < 0
            d = abs(int(r.m))
            cell_d = next((n for lo, hi, n in CELLS_D if lo <= d <= hi), None)
            if cell_d is None:
                continue
            tb, ta = (r.h_bid, r.h_ask) if home_trails else (r.a_bid, r.a_ask)
            lb, la = (r.a_bid, r.a_ask) if home_trails else (r.h_bid, r.h_ask)
            if not (tb >= 0.01 and ta <= 0.99 and ta - tb <= 0.03 + 1e-9):
                continue
            rows.append({"eid": eid, "date": g0.date, "end": r.end, "state_wall": r.state_wall, "game_end": game_end,
                         "period": int(r.period), "clock_s": r.clock_s, "deficit": d,
                         "cell": cell_d + ("_MID" if r.period <= 3 else "_Q4"),
                         "trailer_side": "home" if home_trails else "away",
                         "trailer": g0.home_t if home_trails else g0.away_t,
                         "leader": g0.away_t if home_trails else g0.home_t,
                         "t_bid": tb, "t_ask": ta, "l_bid": lb, "l_ask": la,
                         "F_trailer": r.F_home if home_trails else r.F_away,
                         "F_leader": r.F_away if home_trails else r.F_home})
    f = pd.DataFrame(rows).sort_values(["eid", "end"])
    f = f.drop_duplicates(["eid", "cell"], keep="first").reset_index(drop=True)
    tt = t[["eid", "team", "T_CB", "T_CL", "n_cb", "n_cl", "prior_last"]]
    f = f.merge(tt.rename(columns={"team": "trailer"}), on=["eid", "trailer"], how="left", validate="many_to_one")
    assert (f.dropna(subset=["prior_last"])["prior_last"] < f.dropna(subset=["prior_last"])["date"]).all()
    # outcome kept in a separate column so edgelab can hide it; calib reads it
    res = c[c.side == "home"].groupby("eid")["result"].first()
    f["home_yes"] = f["eid"].map(res)
    f["trailer_won"] = np.where(f.trailer_side == "home", f.home_yes == "yes", f.home_yes == "no").astype(int)
    f["ticker_trailer"] = f["eid"] + "|" + f["trailer_side"]
    f.drop(columns=["home_yes"]).to_parquet(FRAME)
    q = f["T_CB"].dropna().quantile([1 / 3, 2 / 3]).round(4).tolist()
    q2 = f["T_CL"].dropna().quantile([1 / 3, 2 / 3]).round(4).tolist()
    info = {"games_dev": len(dev_eids), "games_with_candles": int(c.eid.nunique()), "snapshots": len(f),
            "snap_games": int(f.eid.nunique()), "with_T_CB": int(f.T_CB.notna().sum()),
            "with_T_CL": int(f.T_CL.notna().sum()), "T_CB_cuts": q, "T_CL_cuts": q2,
            "cells": f.cell.value_counts().to_dict()}
    print(json.dumps(info, indent=1, default=str))
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "build.json").write_text(json.dumps(info, indent=1, default=str))


def _boot(df, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    games = df["eid"].unique()
    idx = {e: np.flatnonzero(df["eid"].to_numpy() == e) for e in games}
    out = []
    for _ in range(B):
        pick = rng.choice(games, len(games), replace=True)
        out.append(fn(df.iloc[np.concatenate([idx[e] for e in pick])]))
    return np.nanpercentile(np.asarray(out, float), [2.5, 97.5], axis=0)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _trait_coef(d, col):
    import statsmodels.api as sm
    z = (d[col] - d[col].mean()) / d[col].std()
    X = sm.add_constant(np.column_stack([_logit(d["mid"].to_numpy()), z.to_numpy()]))
    try:
        return sm.Logit(d["trailer_won"].to_numpy(), X).fit(disp=0).params[2]
    except Exception:
        return np.nan


def calib():
    f = pd.read_parquet(FRAME)
    f["mid"] = (f.t_bid + f.t_ask) / 2
    f["r"] = f.trailer_won - f.mid
    out = {"n_snap": len(f), "overall": {"mean_mid": f.mid.mean(), "won": f.trailer_won.mean(), "r": f.r.mean(),
                                         "r_ci": _boot(f, lambda d: d.r.mean()).tolist()}}
    for col in ["T_CB", "T_CL"]:
        d = f.dropna(subset=[col]).copy()
        cuts = d[col].quantile([1 / 3, 2 / 3]).to_numpy()
        d["terc"] = np.where(d[col] <= cuts[0], "bottom", np.where(d[col] <= cuts[1], "middle", "top"))
        tab = {}
        for k, dd in d.groupby("terc"):
            tab[k] = {"n": len(dd), "games": int(dd.eid.nunique()), "trait_mean": dd[col].mean(),
                      "mid": dd.mid.mean(), "won": dd.trailer_won.mean(), "r": dd.r.mean(),
                      "r_ci": _boot(dd, lambda x: x.r.mean()).tolist()}
        diff = lambda x: x.loc[x.terc == "top", "r"].mean() - x.loc[x.terc == "bottom", "r"].mean()
        coef = _trait_coef(d, col)
        out[col] = {"cuts": cuts.tolist(), "terciles": tab, "diff_top_minus_bottom": diff(d),
                    "diff_ci": _boot(d, diff).tolist(), "logit_coef_z": coef,
                    "logit_coef_ci": _boot(d, lambda x: _trait_coef(x, col), B=500).tolist()}
        # by phase
        out[col]["by_phase"] = {}
        for ph in ["MID", "Q4"]:
            dp = d[d.cell.str.endswith(ph)]
            out[col]["by_phase"][ph] = {"n": len(dp), "diff": diff(dp), "diff_ci": _boot(dp, diff, B=1000).tolist()}
    # secondary: foul trouble difference
    f["dF"] = f.F_leader - f.F_trailer
    slope = lambda x: np.polyfit(x.dF, x.r, 1)[0] if x.dF.nunique() > 1 else np.nan
    out["foul"] = {"dist": f.dF.value_counts().to_dict(), "slope": slope(f), "slope_ci": _boot(f, slope).tolist(),
                   "r_leader_in_trouble": f.loc[f.dF > 0, "r"].mean(), "n_leader_in_trouble": int((f.dF > 0).sum()),
                   "r_ci_leader_in_trouble": _boot(f[f.dF > 0], lambda x: x.r.mean()).tolist()}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "calib.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    {"build": build, "calib": calib}[sys.argv[1]]()
