"""L7 situational NBA prop features vs the de-vigged OPEN prop price + L5 shrink-P2-toward-market check. DEV only.

    python scripts/props_situational.py build    # frame -> results/props_situational/frame.parquet
    python scripts/props_situational.py check    # feature check (pre-registered in notes/props_situational.md)
    python scripts/props_situational.py l5       # L5 calibration check

Pre-registration: notes/props_situational.md (written before any outcome was computed).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from nba_props_model import D, MK  # noqa: E402

OUT = ROOT / "results" / "props_situational"
OUT.mkdir(parents=True, exist_ok=True)
DEV = [2023, 2024, 2025]
ANCHOR = {2023: "DraftKings", 2024: "DraftKings", 2025: "ESPN BET"}

# home arena (lat, lon, UTC offset standard time) by ESPN home abbreviation
ARENA = {
    "ATL": (33.757, -84.396, -5), "BOS": (42.366, -71.062, -5), "BKN": (40.683, -73.975, -5),
    "CHA": (35.225, -80.839, -5), "CHI": (41.881, -87.674, -6), "CLE": (41.497, -81.688, -5),
    "DAL": (32.790, -96.810, -6), "DEN": (39.749, -105.008, -7), "DET": (42.341, -83.055, -5),
    "GS": (37.768, -122.388, -8), "HOU": (29.751, -95.362, -6), "IND": (39.764, -86.156, -5),
    "LAC": (34.043, -118.267, -8), "LAL": (34.043, -118.267, -8), "MEM": (35.138, -90.051, -6),
    "MIA": (25.781, -80.188, -5), "MIL": (43.045, -87.917, -6), "MIN": (44.980, -93.276, -6),
    "NO": (29.949, -90.082, -6), "NY": (40.751, -73.994, -5), "OKC": (35.463, -97.515, -6),
    "ORL": (28.539, -81.384, -5), "PHI": (39.901, -75.172, -5), "PHX": (33.446, -112.071, -7),
    "POR": (45.532, -122.667, -8), "SAC": (38.580, -121.500, -8), "SA": (29.427, -98.438, -6),
    "TOR": (43.643, -79.379, -5), "UTAH": (40.768, -111.901, -7), "WSH": (38.898, -77.021, -5),
}
NIGHT = {"NY", "BKN", "LAL", "LAC", "MIA"}
ALT = {"DEN", "UTAH"}


def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


def schedule() -> pd.DataFrame:
    """Per (event, team): situational features from the team's previous game (all seasons, so Oct games see nothing
    from the prior June: rest >= 3 by construction)."""
    g = pd.concat([pd.read_parquet(D / f"games_{s}.parquet") for s in DEV], ignore_index=True)
    g = g[g.home.isin(ARENA) & g.away.isin(ARENA) & g.season_type.isin(["regular-season", "post-season", "play-in-season"])]
    g = g[~g.neutral.fillna(False).astype(bool)].copy()
    g["day"] = g.start_utc.dt.tz_convert("US/Eastern").dt.tz_localize(None).dt.normalize()
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"event_id": g.event_id, "season": g.season, "season_type": g.season_type,
                                  "day": g.day, "team_id": g[f"{side}_team_id"], "team": g[side],
                                  "opp_id": g[f"{opp}_team_id"], "venue": g.home, "away": side == "away"}))
    t = pd.concat(rows, ignore_index=True)
    t["team_id"] = pd.to_numeric(t.team_id, errors="coerce").astype(float)
    t["opp_id"] = pd.to_numeric(t.opp_id, errors="coerce").astype(float)
    t["event_id"] = t.event_id.astype(str)
    t = t.sort_values(["team_id", "day"])
    t["prev_day"] = t.groupby("team_id").day.shift(1)
    t["prev_venue"] = t.groupby("team_id").venue.shift(1)
    t["prev2_day"] = t.groupby("team_id").day.shift(2)
    rest = (t.day - t.prev_day).dt.days
    t["b2b"] = (rest == 1).astype(float)
    t["rest3"] = ((rest >= 3) | rest.isna()).astype(float)
    t["three_in_four"] = ((t.day - t.prev2_day).dt.days <= 3).astype(float)
    km = [hav(ARENA[p], ARENA[v]) if isinstance(p, str) else 0.0 for p, v in zip(t.prev_venue, t.venue)]
    t["travel"] = np.log1p(km)
    t["tz_east"] = [ARENA[v][2] - ARENA[p][2] if isinstance(p, str) else 0.0 for p, v in zip(t.prev_venue, t.venue)]
    t["alt_visitor"] = (t.away & t.venue.isin(ALT)).astype(float)
    t["night_own"] = ((t.b2b == 1) & t.prev_venue.isin(NIGHT)).astype(float)
    t["away_f"] = t.away.astype(float)
    opp = t[["event_id", "team_id", "b2b", "night_own"]].rename(
        columns={"team_id": "opp_id", "b2b": "opp_b2b", "night_own": "night_opp"})
    t = t.merge(opp, on=["event_id", "opp_id"], how="left")
    return t


def spreads() -> pd.DataFrame:
    l = pd.concat([pd.read_parquet(D / f"lines_{s}.parquet") for s in DEV], ignore_index=True)
    l["event_id"] = l.event_id.astype(str)
    s = l.groupby("event_id").home_spread_open.median().rename("spread_open").reset_index()
    s["blowout"] = (s.spread_open.abs() >= 10).astype(float)
    return s.dropna()


def build():
    from nba_props_dist import rolling_cols
    from nba_props_model import history
    h = history(DEV)
    r = rolling_cols(h)
    box = pd.concat([pd.read_parquet(D / f"box_{s}.parquet") for s in DEV], ignore_index=True)
    team = box.drop_duplicates(["event_id", "athlete_id"])[["event_id", "athlete_id", "team_id"]]
    team["team_id"] = pd.to_numeric(team.team_id, errors="coerce").astype(float)
    team["event_id"] = team.event_id.astype(str)
    r["event_id"] = r.event_id.astype(str)
    frames = []
    for s in DEV:
        gr = pd.read_parquet(D / f"graded_{s}.parquet")
        gr = gr[(gr.provider == ANCHOR[s]) & gr.market_n.isin(MK) & gr.line_open.notna() & (gr.dec_open > 1)
                & gr.actual.notna()]
        p = gr.pivot_table(index=["event_id", "athlete_id", "market_n", "line_open"], columns="side",
                           values=["dec_open", "dec", "line"], aggfunc="first")
        p.columns = [f"{a}_{b}" for a, b in p.columns]
        p = p.reset_index().dropna(subset=["dec_open_over", "dec_open_under"])
        act = gr.drop_duplicates(["event_id", "athlete_id", "market_n", "line_open"])[
            ["event_id", "athlete_id", "market_n", "line_open", "actual"]]
        p = p.merge(act, on=["event_id", "athlete_id", "market_n", "line_open"])
        # one line per (player, game, market): keep the row with the most balanced open prices (main line)
        p["bal"] = (p.dec_open_over - p.dec_open_under).abs()
        p = p.sort_values("bal").drop_duplicates(["event_id", "athlete_id", "market_n"]).assign(season=s)
        frames.append(p)
    x = pd.concat(frames, ignore_index=True)
    x["athlete_id"] = x.athlete_id.astype(str)
    x["event_id"] = x.event_id.astype(str)
    team["athlete_id"] = team.athlete_id.astype(str)
    r["athlete_id"] = r.athlete_id.astype(str)
    x = x.merge(team, on=["event_id", "athlete_id"], how="inner")
    x = x.merge(r, on=["event_id", "athlete_id"], how="inner")
    x = x[x.min20 >= 20]
    hist = [row[f"h::{mk}"] for mk, row in zip(x.market_n, x.to_dict("records"))]
    x["p_emp"] = [np.mean(v < l) + 0.5 * np.mean(v == l) for v, l in zip(hist, x.line_open)]
    x = x.drop(columns=[c for c in x.columns if c.startswith("h::")])
    sch = schedule()
    x = x.merge(sch.drop(columns=["season"]), on=["event_id", "team_id"], how="inner")
    x = x.merge(spreads(), on="event_id", how="left")
    x = x[x.season_type == "regular-season"]
    x["p_mkt"] = (1 / x.dec_open_under) / (1 / x.dec_open_under + 1 / x.dec_open_over)
    push = x.actual == x.line_open
    x["under_win"] = np.where(push, np.nan, (x.actual < x.line_open).astype(float))
    x["e"] = x.under_win - x.p_mkt
    x["pnl_under"] = np.where(push, 0.0, np.where(x.actual < x.line_open, x.dec_open_under - 1, -1.0))
    same = x.line_under == x.line_open
    x["p_close"] = np.where(same & x.dec_under.notna() & x.dec_over.notna() & (x.line_over == x.line_open),
                            (1 / x.dec_under) / (1 / x.dec_under + 1 / x.dec_over), np.nan)
    x.to_parquet(OUT / "frame.parquet")
    print(len(x), x.groupby("season").size().to_dict())


FEATS = ["b2b", "rest3", "three_in_four", "travel", "tz_east", "alt_visitor", "night_own", "night_opp", "blowout",
         "opp_b2b", "away_f"]


def ols_cluster(y, X, cl):
    """OLS with cluster-robust (CR1) SEs."""
    XtX = np.linalg.inv(X.T @ X)
    b = XtX @ X.T @ y
    u = y - X @ b
    codes, inv = np.unique(cl, return_inverse=True)
    S = np.zeros((len(codes), X.shape[1]))
    np.add.at(S, inv, X * u[:, None])
    G, n, k = len(codes), len(y), X.shape[1]
    V = XtX @ (S.T @ S) @ XtX * G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(V))


def check():
    from scipy.stats import norm
    x = pd.read_parquet(OUT / "frame.parquet").dropna(subset=["e"])
    mk = pd.get_dummies(x.market_n, drop_first=True).astype(float).to_numpy()
    out = {"n": len(x), "dates": int(x.day.nunique()), "mean_e": float(x.e.mean()),
           "under_roi_all": float(x.pnl_under.sum() / len(x)), "features": {}}
    for f in FEATS:
        d = x.dropna(subset=[f])
        mkd = pd.get_dummies(d.market_n, drop_first=True).astype(float).to_numpy()
        X = np.column_stack([np.ones(len(d)), d[f].to_numpy(float), mkd])
        b, se = ols_cluster(d.e.to_numpy(), X, d.day.astype(str).to_numpy())
        p = 2 * (1 - norm.cdf(abs(b[1] / se[1])))
        by = {}
        for s in DEV:
            ds = d[d.season == s]
            if len(ds) < 200 or ds[f].nunique() < 2:
                by[s] = None; continue
            m = pd.get_dummies(ds.market_n, drop_first=True).astype(float).to_numpy()
            bs, ss = ols_cluster(ds.e.to_numpy(), np.column_stack([np.ones(len(ds)), ds[f].to_numpy(float), m]),
                                 ds.day.astype(str).to_numpy())
            by[s] = round(float(bs[1]), 4)
        out["features"][f] = {"n": len(d), "share_or_mean": float(d[f].mean()), "coef": float(b[1]),
                              "ci95": [float(b[1] - 1.96 * se[1]), float(b[1] + 1.96 * se[1])], "p": float(p),
                              "passes": bool(p < 0.005), "by_season": by}
        print(f"{f:14s} n={len(d):6d} mean={d[f].mean():.3f} coef={b[1]:+.4f} "
              f"[{b[1]-1.96*se[1]:+.4f},{b[1]+1.96*se[1]:+.4f}] p={p:.4f} {by}")
    print("n", out["n"], "dates", out["dates"], "mean e", round(out["mean_e"], 4), "under ROI all", round(out["under_roi_all"], 4))
    json.dump(out, open(OUT / "feature_check.json", "w"), indent=1)


def l5():
    from scipy.stats import norm
    x = pd.read_parquet(OUT / "frame.parquet").dropna(subset=["under_win"])
    lg = lambda p: np.log(np.clip(p, 0.02, 0.98) / (1 - np.clip(p, 0.02, 0.98)))
    tr = x[x.season.isin([2023, 2024])]
    import statsmodels.api as sm
    X = np.column_stack([np.ones(len(tr)), lg(tr.p_mkt), lg(tr.p_emp) - lg(tr.p_mkt)])
    m = sm.GLM(tr.under_win.to_numpy(), X, family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(tr.day)[0]})
    b, se = m.params, m.bse
    res = {"n_fit": len(tr), "coef": b.tolist(), "ci_c": [b[2] - 1.96 * se[2], b[2] + 1.96 * se[2]],
           "p_c": float(m.pvalues[2])}
    te = x[x.season == 2025].copy()
    Xt = np.column_stack([np.ones(len(te)), lg(te.p_mkt), lg(te.p_emp) - lg(te.p_mkt)])
    te["p_fit"] = 1 / (1 + np.exp(-Xt @ b))
    bt = te[te.p_fit >= te.p_mkt + 0.03]
    res["test2025"] = {"bets": len(bt), "roi": float(bt.pnl_under.mean()) if len(bt) else None}
    print(json.dumps(res, indent=1, default=float))
    json.dump(res, open(OUT / "l5_check.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    {"build": build, "check": check, "l5": l5}[sys.argv[1]]()
