"""game_release_slump: totals UNDER after mega video-game launches (pre-registration notes/game_release_slump.md). DEV ONLY.

    ../.venv_quant/bin/python -I scripts/game_release_slump.py build    # frame: one row per game, window labels (no outcome stats)
    ../.venv_quant/bin/python -I scripts/game_release_slump.py power    # game counts + residual SD -> MDE (SD is outcome-
                                                                        #   derived but direction-free; pooled over ALL games)
    ../.venv_quant/bin/python -I scripts/game_release_slump.py score    # D (window - control) by league/event, bootstrap by event

Leagues: NFL (lab nfl_total, real close odds, 2006-2019), NBA (SBR OddsData.sqlite close OU at -110, 2007-08..2018-19),
NHL (SBR close total + price, 2007-08..2022-23), CFB (SBR close total at -110, 2007..2022). No sealed season is read.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "game_release_slump"
FRAME = OUT / "frame.parquet"

EVENTS = [  # (event, release date, rockstar?) -- inclusion rule in notes/game_release_slump.md
    ("GTA IV", "2008-04-29", True), ("CoD MW2", "2009-11-10", False), ("CoD Black Ops", "2010-11-09", False),
    ("CoD MW3", "2011-11-08", False), ("CoD Black Ops II", "2012-11-13", False), ("GTA V", "2013-09-17", True),
    ("CoD Ghosts", "2013-11-05", False), ("Destiny", "2014-09-09", False), ("CoD BO3 + Fallout 4", "2015-11-06", False),
    ("CoD WWII", "2017-11-03", False), ("CoD Black Ops 4", "2018-10-12", False), ("RDR2", "2018-10-26", True),
    ("CoD MW 2019", "2019-10-25", False), ("CoD MW II 2022", "2022-10-28", False)]
# Fallout 4 (2015-11-10) is merged into the BO3 cluster (overlapping windows, pre-registered).
CTRL = [(-21, -8), (21, 34)]
STD = 1 + 100 / 110


def am_dec(a):
    a = np.asarray(a, float)
    return np.where(a > 0, 1 + a / 100, 1 + 100 / np.abs(a))


# ------------------------------------------------------------------------------------------------- leagues
def nfl():
    d = pd.read_parquet(ROOT / "data" / "lab" / "nfl_total.parquet")
    d = d[d["season"] <= 2019]
    o = d[d.selection == "over"].set_index("opp_id"); u = d[d.selection == "under"].set_index("opp_id")
    g = pd.DataFrame({"date": pd.to_datetime(o["date"]).dt.normalize(), "season": o["season"], "home": o["home"],
                      "away": o["away"], "line": o["total_line"], "final": o["home_score"] + o["away_score"],
                      "dec_over": o["dec_odds"], "dec_under": u.loc[o.index, "dec_odds"]}).reset_index()
    g["game_id"] = "NFL|" + g["opp_id"].astype(str)
    g["league"], g["price_real"] = "NFL", True
    return g.drop(columns="opp_id")


def nba():
    c = sqlite3.connect(ROOT / "data" / "nba_odds" / "OddsData.sqlite")
    fs = []
    for y in range(2007, 2019):
        s = f"{y}-{str(y + 1)[2:]}"
        d = pd.read_sql(f'select * from "odds_{s}"', c)
        d["season"] = y
        fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    md = d["Date"].str[-4:].astype(int)
    mon, day = md // 100, md % 100
    yr = np.where(mon >= 8, d.season, d.season + 1)
    d["date"] = pd.to_datetime(dict(year=yr, month=mon, day=day), errors="coerce")
    d["line"] = pd.to_numeric(d.OU, errors="coerce"); d["final"] = pd.to_numeric(d.Points, errors="coerce")
    g = d[["date", "season", "Home", "Away", "line", "final"]].rename(columns={"Home": "home", "Away": "away"})
    g = g.dropna(subset=["date", "line", "final"])
    g = g[(g.line > 150) & (g.line < 280)]
    g["dec_over"] = g["dec_under"] = STD
    g["game_id"] = "NBA|" + g.date.dt.strftime("%Y%m%d") + "|" + g.home + "|" + g.away
    g["league"], g["price_real"] = "NBA", False
    return g


def nhl():
    import nhl_luck as N
    g = pd.concat([N.parse_sbr(y) for y in sorted(N.SEASON_FILES)], ignore_index=True)
    g = g.rename(columns={"home_sbr": "home", "away_sbr": "away", "tot_close": "line"})
    g["final"] = g.home_final + g.away_final
    g["dec_over"], g["dec_under"] = am_dec(g.over_close_px), am_dec(g.under_close_px)
    g = g.dropna(subset=["date", "line", "final", "dec_over", "dec_under"])
    g = g[(g.line >= 4) & (g.line <= 8)]
    g["game_id"] = "NHL|" + g.date.dt.strftime("%Y%m%d") + "|" + g.home + "|" + g.away
    g["league"], g["price_real"] = "NHL", True
    return g[["date", "season", "home", "away", "line", "final", "dec_over", "dec_under", "game_id", "league", "price_real"]]


def cfb():
    import cfb_simple_model as C
    g = C.load_all()
    g = g.rename(columns={"total_close": "line", "points": "final"}).dropna(subset=["line", "final"])
    g["dec_over"] = g["dec_under"] = STD
    g["game_id"] = "CFB|" + g.date.dt.strftime("%Y%m%d") + "|" + g.rot.astype(str)
    g["league"], g["price_real"] = "CFB", False
    return g[["date", "season", "home", "away", "line", "final", "dec_over", "dec_under", "game_id", "league", "price_real"]]


# ------------------------------------------------------------------------------------------------- labels
def label(g: pd.DataFrame) -> pd.DataFrame:
    """Each game: nearest event day offset k (days since release); window W7 (0..6), W14 (0..13), control."""
    rel = pd.to_datetime([e[1] for e in EVENTS])
    rows = []
    for (name, d0, rock), r0 in zip(EVENTS, rel):
        k = (g["date"] - r0).dt.days
        m = (k >= -21) & (k <= 34)
        s = g[m].copy(); s["event"], s["k"], s["rockstar"] = name, k[m], rock
        rows.append(s)
    x = pd.concat(rows, ignore_index=True)
    x["w7"], x["w14"] = x.k.between(0, 6), x.k.between(0, 13)
    x["ctrl"] = x.k.between(*CTRL[0]) | x.k.between(*CTRL[1])
    # a game in ANY event's W14 cannot be control for another event; a game in two events' windows is kept once
    # (first event) -- rare (BO4 / RDR2 14 days apart)
    in_win = set(x.loc[x.w14, "game_id"])
    x = x[~(x.ctrl & x.game_id.isin(in_win))]
    x = x[x.w14 | x.ctrl]
    x = x.sort_values(["game_id", "k"]).drop_duplicates(["game_id", "w14"])
    x = x[~(x.ctrl & x.duplicated("game_id", keep=False))]
    return x.reset_index(drop=True)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    g = pd.concat([nfl(), nba(), nhl(), cfb()], ignore_index=True)
    g["date"] = pd.to_datetime(g["date"]).astype("datetime64[ns]")
    g.to_parquet(OUT / "all_games.parquet", index=False)
    x = label(g)
    x.to_parquet(FRAME, index=False)
    print(x.groupby(["league", "event"]).agg(w7=("w7", "sum"), w14=("w14", "sum"), ctrl=("ctrl", "sum")).to_string())
    print(x.groupby("league").agg(w7=("w7", "sum"), w14=("w14", "sum"), ctrl=("ctrl", "sum")))


def export():
    """edgelab frame: one row per W14 game (UNDER side), hidden won/push/final. Loader game_release_totals."""
    x = pd.read_parquet(FRAME)
    x = x[x.w14].copy()
    r = x.final - x.line
    e = pd.DataFrame({"date": x.date, "game_id": x.game_id, "sel": "under", "league": x.league, "event": x.event,
                      "k": x.k, "w7": x.w7, "w14": x.w14, "line": x.line, "dec_under": x.dec_under,
                      "price_real": x.price_real, "won": (r < 0).astype(int), "push": (r == 0).astype(int),
                      "final": x.final})
    e.to_parquet(OUT / "edgelab_frame.parquet", index=False)
    print(len(e), e.groupby("league").size().to_dict())


def power():
    g = pd.read_parquet(OUT / "all_games.parquet")
    g["r"] = g.final - g.line
    sd = g.groupby("league")["r"].std()
    x = pd.read_parquet(FRAME)
    print("residual SD by league (all dev games, direction-free):\n", sd.round(2))
    for w in ("w7", "w14"):
        n = x[x[w]].groupby("league").size(); nc = x[x.ctrl].groupby("league").size()
        se = sd * np.sqrt(1 / n + 1 / nc)
        se_z = np.sqrt(1 / n.sum() + 1 / nc.sum())        # pooled, in SD units
        se_win = np.sqrt(0.25 / n.sum())
        print(f"\n{w}: games {n.to_dict()} control {nc.to_dict()}")
        print(" MDE (80% power, 2-sided .05) per league, points:", (2.8 * se).round(2).to_dict())
        print(f" pooled MDE in SD units {2.8 * se_z:.3f}; under win-rate MDE {2.8 * se_win * 100:.1f}pp "
              f"(ROI ~{2.8 * se_win * 2 * 0.909 * 100:.1f}%) -- iid; event clustering widens it")


def boot_events(df, stat, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    ev = df.event.unique(); grp = {e: df[df.event == e] for e in ev}
    out = []
    for _ in range(B):
        pick = rng.choice(ev, len(ev))
        out.append(stat(pd.concat([grp[e] for e in pick])))
    return np.nanpercentile(out, [2.5, 97.5])


def score():
    x = pd.read_parquet(FRAME)
    g = pd.read_parquet(OUT / "all_games.parquet")
    sd = (g.final - g.line).groupby(g.league).std()
    x["r"] = x.final - x.line
    x["z"] = x.r / x.league.map(sd)
    x["push"] = x.r == 0
    x["under_pnl"] = np.where(x.push, 0.0, np.where(x.r < 0, x.dec_under - 1, -1.0))
    lines = []

    def D(df, w, col="z"):
        return df.loc[df[w], col].mean() - df.loc[df.ctrl, col].mean()

    for w in ("w7", "w14"):
        lines.append(f"\n=== {w} ===")
        for lg, df in [("ALL", x)] + list(x.groupby("league")):
            nw, nc = int(df[w].sum()), int(df.ctrl.sum())
            if nw == 0:
                continue
            d_pts = D(df, w, "r") if lg != "ALL" else np.nan
            dz = D(df, w, "z"); ci = boot_events(df, lambda q: D(q, w, "z"))
            roi = df.loc[df[w], "under_pnl"].mean(); roic = df.loc[df.ctrl, "under_pnl"].mean()
            rci = boot_events(df, lambda q: q.loc[q[w], "under_pnl"].mean())
            uw = (df.loc[df[w] & ~df.push, "r"] < 0).mean()
            lines.append(f"{lg:4s} n_win {nw:5d} n_ctrl {nc:5d} | D pts {d_pts:+.2f} | D sd {dz:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"
                         f" | under win% {uw * 100:.1f} | under ROI win {roi * 100:+.2f}% [{rci[0] * 100:+.2f},{rci[1] * 100:+.2f}]"
                         f" ctrl {roic * 100:+.2f}%")
        ev = x.groupby("event").apply(lambda q: pd.Series({"n": int(q[w].sum()), "Dz": D(q, w, "z"),
                                                            "roi": q.loc[q[w], "under_pnl"].mean()}))
        lines.append(ev.round(3).to_string())
        rk = x[x.rockstar]
        lines.append(f"Rockstar subset (GTA IV/V, RDR2): n {int(rk[w].sum())} D sd {D(rk, w, 'z'):+.3f} "
                     f"[{boot_events(rk, lambda q: D(q, w, 'z'))[0]:+.3f},{boot_events(rk, lambda q: D(q, w, 'z'))[1]:+.3f}] "
                     f"under ROI {rk.loc[rk[w], 'under_pnl'].mean() * 100:+.2f}%")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "score.txt").write_text(txt)
    x.to_parquet(OUT / "scored_frame.parquet", index=False)


if __name__ == "__main__":
    {"build": build, "export": export, "power": power, "score": score}[sys.argv[1]]()
