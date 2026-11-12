"""B2B recovery trend (notes/b2b_recovery_trend.md, pre-registered 2026-11-12). DEV ONLY: 2025-26 never loaded.

    ../.venv_quant/bin/python scripts/b2b_recovery_trend.py build     # -> results/b2b_recovery_trend/games.parquet
    ../.venv_quant/bin/python scripts/b2b_recovery_trend.py trend     # per-season + pooled trend regressions
    ../.venv_quant/bin/python scripts/b2b_recovery_trend.py bets      # B1/B2 bet files for edgelab (dev 2019-20..2024-25)
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
import traps  # noqa: E402

DB = ROOT / "data" / "nba_odds" / "OddsData.sqlite"
OUT = ROOT / "results" / "b2b_recovery_trend"
OUT.mkdir(parents=True, exist_ok=True)
STD = 1 + 100 / 110

# official regular-season last day for seasons before ESPN game_type coverage (nba_games.parquet starts 2015-16)
REG_END = {2007: "2008-04-16", 2008: "2009-04-16", 2009: "2010-04-14", 2010: "2011-04-13", 2011: "2012-04-26",
           2012: "2013-04-17", 2013: "2014-04-16", 2014: "2015-04-15"}
BUBBLE = ("2020-07-30", "2020-10-11")

# franchise arena (lat, lon); old names map to the same franchise city at the time
ARENA = {
    "Atlanta Hawks": (33.757, -84.396), "Boston Celtics": (42.366, -71.062), "Brooklyn Nets": (40.683, -73.976),
    "New Jersey Nets": (40.734, -74.171), "Charlotte Hornets": (35.225, -80.839), "Charlotte Bobcats": (35.225, -80.839),
    "Chicago Bulls": (41.881, -87.674), "Cleveland Cavaliers": (41.497, -81.688), "Dallas Mavericks": (32.790, -96.810),
    "Denver Nuggets": (39.749, -105.008), "Detroit Pistons": (42.341, -83.055), "Golden State Warriors": (37.768, -122.388),
    "Houston Rockets": (29.751, -95.362), "Indiana Pacers": (39.764, -86.155), "Los Angeles Clippers": (34.043, -118.267),
    "LA Clippers": (34.043, -118.267), "Los Angeles Lakers": (34.043, -118.267), "Memphis Grizzlies": (35.138, -90.051),
    "Miami Heat": (25.781, -80.188), "Milwaukee Bucks": (43.045, -87.917), "Minnesota Timberwolves": (44.979, -93.276),
    "New Orleans Hornets": (29.949, -90.082), "New Orleans Pelicans": (29.949, -90.082),
    "New Orleans/Oklahoma City Hornets": (29.949, -90.082), "New York Knicks": (40.751, -73.993),
    "Oklahoma City Thunder": (35.463, -97.515), "Seattle SuperSonics": (47.622, -122.354), "Orlando Magic": (28.539, -81.384),
    "Philadelphia 76ers": (39.901, -75.172), "Phoenix Suns": (33.446, -112.071), "Portland Trail Blazers": (45.532, -122.667),
    "Sacramento Kings": (38.580, -121.500), "San Antonio Spurs": (29.427, -98.438), "Toronto Raptors": (43.643, -79.379),
    "Utah Jazz": (40.768, -111.901), "Washington Wizards": (38.898, -77.021),
}
# Golden State played in Oakland until 2019 (37.750,-122.203); the ~10 mi difference is irrelevant for travel bins.


def season_tables(c):
    names = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
    out = {}
    for y in range(2007, 2025):
        s = f"{y}-{str(y + 1)[2:]}"
        t = f"odds_{s}_new" if f"odds_{s}_new" in names and y <= 2022 else s
        out[y] = t
    return out


def hav(a, b):
    la1, lo1, la2, lo2 = map(np.radians, [a[:, 0], a[:, 1], b[:, 0], b[:, 1]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def build():
    c = sqlite3.connect(DB)
    g = pd.read_parquet(ROOT / "data" / "nba_games.parquet", columns=["season", "game_type", "date"])
    g = g[g.season < 2025]  # 2025-26 held out: dropped at read
    reg_end = {**{k: pd.Timestamp(v) for k, v in REG_END.items()},
               **g[g.game_type == "regular"].groupby("season").date.max().to_dict()}
    fs = []
    for y, t in season_tables(c).items():
        d = pd.read_sql(f'select Date, Home, Away, OU, Spread, ML_Home, ML_Away, Points, Win_Margin, '
                        f'Days_Rest_Home, Days_Rest_Away from "{t}"', c)
        d["season"] = y
        d["table"] = t
        fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    d["date"] = pd.to_datetime(d.Date)
    assert d.date.dt.year.between(2007, 2025).all(), "bad dates"
    for k in ["OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    unknown = set(d.Home) | set(d.Away)
    unknown -= set(ARENA)
    assert not unknown, unknown
    # rest computed from the full table (incl. playoffs, before filtering) per team-season
    long = pd.concat([d[["season", "date", "Home"]].rename(columns={"Home": "team"}).assign(venue=d.Home),
                      d[["season", "date", "Away"]].rename(columns={"Away": "team"}).assign(venue=d.Home)])
    long = long.sort_values(["team", "season", "date"]).drop_duplicates(["team", "season", "date"])
    grp = long.groupby(["team", "season"])
    long["prev_date"] = grp.date.shift(1)
    long["prev_venue"] = grp.venue.shift(1)
    long["d3"] = grp.date.shift(2)
    long["rest"] = (long.date - long.prev_date).dt.days
    long["b2b"] = (long.rest == 1).astype(int)
    long["in3of4"] = ((long.date - long.d3).dt.days <= 3).astype(int)
    pv = long.prev_venue.fillna(long.venue)
    long["travel"] = hav(np.array([ARENA[v] for v in pv]), np.array([ARENA[v] for v in long.venue]))
    long["travel"] = np.where(long.b2b == 1, long.travel, 0.0)
    feat = long[["team", "season", "date", "rest", "b2b", "in3of4", "travel"]]
    for side in ["Home", "Away"]:
        f = feat.rename(columns={c_: f"{c_}_{side[0].lower()}" for c_ in ["rest", "b2b", "in3of4", "travel"]})
        d = traps.checked_merge(d, f.rename(columns={"team": side}), on=[side, "season", "date"], how="left",
                                validate="many_to_one")
    # regular season only, bubble out
    d["reg"] = d.date <= d.season.map(reg_end)
    d["bubble"] = d.date.between(*map(pd.Timestamp, BUBBLE))
    # validate own rest vs file's Days_Rest (file: 1 = played yesterday)
    ok = d.rest_h.notna()
    agree = (d.loc[ok, "rest_h"] == pd.to_numeric(d.loc[ok, "Days_Rest_Home"], errors="coerce")).mean()
    # sign the spread from the ML favourite (file spread is unsigned), + = home favoured
    d["spread"] = d.Spread.abs() * np.where(d.ML_Home < d.ML_Away, 1, -1)
    d = d[d.reg & ~d.bubble].dropna(subset=["OU", "Spread", "ML_Home", "ML_Away", "Points", "Win_Margin"])
    d = d[(d.Points > 120)]  # drops 0-score placeholder rows
    # line-validity filter (no outcome used): OU/spread swapped or typo'd rows, the 2012-04-21 160/5 placeholder rows
    n0 = len(d)
    bad = (d.OU < 150) | (d.OU > 300) | ((d.Spread.abs() > 25) & (d[["ML_Home", "ML_Away"]].abs().max(axis=1) < 1000)) \
        | ((d.OU == 160) & (d.Spread == 5))
    d = d[~bad]
    print("dropped bad-line rows:", n0 - len(d))
    d["margin"] = d.Win_Margin
    d["cover"] = d.margin - d.spread
    d["ou_res"] = d.Points - d.OU
    d = d.drop_duplicates(["date", "Home", "Away"]).reset_index(drop=True)
    d.to_parquet(OUT / "games.parquet")
    info = {"rows": len(d), "rest_agree_with_file": round(float(agree), 4),
            "by_season": d.groupby("season").agg(n=("margin", "size"), b2b_h=("b2b_h", "mean"),
                                                 b2b_a=("b2b_a", "mean")).round(3).reset_index().to_dict("records"),
            "corr_spread_margin": round(float(np.corrcoef(d.spread, d.margin)[0, 1]), 3),
            "mean_cover": round(float(d.cover.mean()), 3), "mean_ou_res": round(float(d.ou_res.mean()), 3)}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "by_season"}, indent=1))
    print(pd.DataFrame(info["by_season"]).to_string())


def _fe(d, total=False):
    """team-season strength dummies: home +1, away -1 (margin) or +1/+1 (totals)."""
    ts_h = d.Home + "|" + d.season.astype(str)
    ts_a = d.Away + "|" + d.season.astype(str)
    cats = sorted(set(ts_h) | set(ts_a))
    H = pd.get_dummies(pd.Categorical(ts_h, categories=cats), dtype=float).values
    A = pd.get_dummies(pd.Categorical(ts_a, categories=cats), dtype=float).values
    M = H + A if total else H - A
    return M[:, 1:] if not total else M  # drop one column for identification in the margin case


def _ols(y, X, groups, names):
    import statsmodels.api as sm
    r = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = {}
    for n in names:
        i = n[1]
        b, se = float(r.params[i]), float(r.bse[i])
        out[n[0]] = {"b": round(b, 3), "lo": round(b - 1.96 * se, 3), "hi": round(b + 1.96 * se, 3),
                     "p": float(r.pvalues[i])}
    return out


def _design(d, extra_cols, total=False, pooled=True, trend_cols=()):
    seas = pd.get_dummies(d.season, dtype=float).values  # per-season intercept (HCA / scoring level)
    cols = [d[c].values.astype(float)[:, None] for c in extra_cols]
    t = (d.season.values - 2016).astype(float)
    tcols = [(d[c].values * t)[:, None] for c in trend_cols]
    X = np.hstack(cols + tcols + [seas, _fe(d, total)])
    return X


def trend():
    d = pd.read_parquet(OUT / "games.parquet")
    d["dB2B"] = d.b2b_h - d.b2b_a
    d["d3in4"] = d.in3of4_h - d.in3of4_a
    d["dtrav"] = (d.travel_h - d.travel_a) / 1000
    d["nB2B"] = d.b2b_h + d.b2b_a
    grp = d.date.dt.strftime("%Y-%m-%d").values
    res = {"per_season": [], "pooled": {}}
    ys_m = {"realized": "margin", "market": "spread", "residual": "cover"}
    ys_t = {"realized": "Points", "market": "OU", "residual": "ou_res"}
    for s, g in d.groupby("season"):
        gg = g.date.dt.strftime("%Y-%m-%d").values
        row = {"season": int(s), "n": len(g), "n_b2b_diff": int((g.dB2B != 0).sum())}
        for k, y in ys_m.items():
            X = np.hstack([g[["dB2B"]].values.astype(float), np.ones((len(g), 1)), _fe(g)])
            row["side_" + k] = _ols(g[y].values, X, gg, [("dB2B", 0)])["dB2B"]
        for k, y in ys_t.items():
            X = np.hstack([g[["nB2B"]].values.astype(float), np.ones((len(g), 1)), _fe(g, True)])
            row["tot_" + k] = _ols(g[y].values, X, gg, [("nB2B", 0)])["nB2B"]
        res["per_season"].append(row)
    for label, sub in [("all", d), ("ex2020_21", d[d.season != 2020])]:
        sg = sub.date.dt.strftime("%Y-%m-%d").values
        for k, y in ys_m.items():
            X = _design(sub, ["dB2B"], trend_cols=["dB2B"])
            res["pooled"][f"{label}_side_{k}"] = _ols(sub[y].values, X, sg, [("level_2016", 0), ("trend_per_season", 1)])
            X = _design(sub, ["dB2B", "d3in4", "dtrav"], trend_cols=["dB2B"])
            res["pooled"][f"{label}_side_{k}_ctl"] = _ols(sub[y].values, X, sg,
                                                          [("dB2B_2016", 0), ("d3in4", 1), ("dtrav_per1000mi", 2),
                                                           ("trend_per_season", 3)])
        for k, y in ys_t.items():
            X = _design(sub, ["nB2B"], total=True, trend_cols=["nB2B"])
            res["pooled"][f"{label}_tot_{k}"] = _ols(sub[y].values, X, sg, [("level_2016", 0), ("trend_per_season", 1)])
    (OUT / "trend.json").write_text(json.dumps(res, indent=1))
    rows = []
    for r in res["per_season"]:
        f = lambda x: f"{x['b']:+.2f} [{x['lo']:+.2f},{x['hi']:+.2f}]"
        rows.append([r["season"], r["n_b2b_diff"], f(r["side_realized"]), f(r["side_market"]), f(r["side_residual"]),
                     f(r["tot_realized"]), f(r["tot_market"]), f(r["tot_residual"])])
    print(pd.DataFrame(rows, columns=["season", "n_dB2B", "side_real", "side_mkt", "side_resid", "tot_real", "tot_mkt",
                                      "tot_resid"]).to_string(index=False))
    for k, v in res["pooled"].items():
        print(k, {a: f"{b['b']:+.3f} [{b['lo']:+.3f},{b['hi']:+.3f}] p={b['p']:.3g}" for a, b in v.items()})


def bets():
    """edgelab frame for B1 (back the B2B team ATS vs a non-B2B opponent) and B2 (over when >= 1 team on a B2B).
    Dev 2019-20..2024-25 only (bubble already excluded in build). Closing line, -110."""
    d = pd.read_parquet(OUT / "games.parquet")
    d = d[d.season.between(2019, 2024)].copy()
    d["game_id"] = d.date.dt.strftime("%Y-%m-%d") + "|" + d.Home + "|" + d.Away
    one = (d.b2b_h + d.b2b_a) == 1
    b1 = d[one].copy()
    b1["sel"] = np.where(b1.b2b_h == 1, "home", "away")
    c = np.where(b1.sel == "home", b1.cover, -b1.cover)
    b1["won"], b1["push"], b1["trial"] = (c > 0).astype(int), (c == 0).astype(int), "B1"
    b2 = d[(d.b2b_h + d.b2b_a) >= 1].copy()
    b2["sel"] = "over"
    b2["won"], b2["push"], b2["trial"] = (b2.ou_res > 0).astype(int), (b2.ou_res == 0).astype(int), "B2"
    f = pd.concat([b1, b2])[["date", "game_id", "trial", "sel", "won", "push", "season"]]
    f["dec"] = STD
    f.to_parquet(OUT / "edgelab_frame.parquet")
    print(f.groupby("trial").size())


if __name__ == "__main__":
    {"build": build, "trend": trend, "bets": bets}[sys.argv[1]]()
