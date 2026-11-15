"""nhl_opener_model: a-priori xG + goalie GSAx + b2b NHL totals model (spec in notes/nhl_opener_model.md).
predict(home, away, D, goalie_h, goalie_a) -> (lam_h, lam_a, ladder median). D = decision ET date; only games with
gameDate < D are used. goalie_* = {playerId: weight} (projected) or {id: 1.0} (actual)."""
import glob, math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

MP = Path(__file__).parent / "mp"
HL_TEAM, HL_G, PSEUDO_T, PSEUDO_G = 45.0, 365.0, 10.0, 100.0
B2B_OWN, B2B_OPP, HOME = -0.24, 0.21, 0.02
ESPN2MP = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "UTAH": "UTA"}


def season_of(d):  # MoneyPuck season label = start year
    d = pd.Timestamp(d)
    return d.year if d.month >= 8 else d.year - 1


def _load():
    t = []
    for f in glob.glob(str(MP / "team_*.csv")):
        x = pd.read_csv(f, usecols=["season", "gameId", "playerTeam", "opposingTeam", "home_or_away", "gameDate",
                                    "situation", "xGoalsFor", "xGoalsAgainst", "goalsFor", "goalsAgainst"])
        t.append(x[(x.situation == "all") & (x.season >= 2020)])
    T = pd.concat(t, ignore_index=True)
    T["date"] = pd.to_datetime(T.gameDate.astype(str))
    T = T.drop_duplicates(["gameId", "playerTeam"])
    T["off"] = 0.7 * T.xGoalsFor + 0.3 * T.goalsFor
    g = []
    for f in glob.glob(str(MP / "goalie_*.csv")):
        x = pd.read_csv(f, usecols=["playerId", "season", "gameId", "playerTeam", "gameDate", "situation", "icetime",
                                    "xGoals", "goals"])
        g.append(x[(x.situation == "all") & (x.season >= 2020)])
    G = pd.concat(g, ignore_index=True).drop_duplicates(["gameId", "playerId"])
    G["date"] = pd.to_datetime(G.gameDate.astype(str))
    G["starter"] = G.groupby(["gameId", "playerTeam"]).icetime.transform("max") == G.icetime
    return T, G


T, G = _load()
STARTS = G[G.starter].drop_duplicates(["gameId", "playerTeam"])[["gameId", "playerTeam", "playerId", "date", "season"]]


@lru_cache(maxsize=None)
def season_final(s):
    x = T[T.season == s]
    if x.empty:
        return None
    tm = x.groupby("playerTeam").agg(off=("off", "mean"), xga=("xGoalsAgainst", "mean"))
    return tm, x.goalsFor.mean(), x.off.mean(), x.xGoalsAgainst.mean()


@lru_cache(maxsize=None)
def state(D):
    D = pd.Timestamp(D)
    s = season_of(D)
    prev = season_final(s - 1)
    ptm, pL, pLO, pLD = prev
    x = T[(T.season == s) & (T.date < D)]
    w = 0.5 ** ((D - x.date).dt.days / HL_TEAM)
    # league levels: weighted current + 100 pseudo team-games of last season
    L = (np.sum(w * x.goalsFor) + 100 * pL) / (w.sum() + 100)
    LO = (np.sum(w * x.off) + 100 * pLO) / (w.sum() + 100)
    LD = (np.sum(w * x.xGoalsAgainst) + 100 * pLD) / (w.sum() + 100)
    x = x.assign(w=w, wo=w * x.off, wd=w * x.xGoalsAgainst)
    agg = x.groupby("playerTeam")[["w", "wo", "wd"]].sum()
    teams = set(ptm.index) | set(agg.index)
    R = {}
    for tm in teams:
        po = 0.5 * (ptm.off.get(tm, LO) if tm in ptm.index else LO) + 0.5 * LO
        pd_ = 0.5 * (ptm.xga.get(tm, LD) if tm in ptm.index else LD) + 0.5 * LD
        a = agg.loc[tm] if tm in agg.index else pd.Series({"w": 0, "wo": 0, "wd": 0})
        R[tm] = ((a.wo + PSEUDO_T * po) / (a.w + PSEUDO_T), (a.wd + PSEUDO_T * pd_) / (a.w + PSEUDO_T))
    # goalies: GSAx rate across seasons, half-life 365 d
    gx = G[G.date < D]
    gw = 0.5 ** ((D - gx.date).dt.days / HL_G)
    gg = gx.assign(a=gw * (gx.xGoals - gx.goals), b=gw * gx.xGoals).groupby("playerId")[["a", "b"]].sum()
    gs = (gg.a / (gg.b + PSEUDO_G)).to_dict()
    return R, L, LO, LD, gs


def projected(team, D, n=10):
    st = STARTS[(STARTS.playerTeam == team) & (STARTS.date < pd.Timestamp(D))].sort_values("date").tail(n)
    if st.empty:
        return {}
    return (st.playerId.value_counts() / len(st)).to_dict()


def played_prev(team, D):
    d1 = pd.Timestamp(D) - pd.Timedelta(days=1)
    return bool(((T.playerTeam == team) & (T.date == d1)).any())


def gfac(gd, gs):
    if not gd:
        return 1.0
    return sum(wt * (1 - gs.get(pid, 0.0)) for pid, wt in gd.items()) / sum(gd.values())


def p_over_ladder(lh, la, xs=np.arange(2.5, 11.0, 1.0)):
    k = np.arange(0, 25)
    ph = np.array([math.exp(-lh) * lh ** i / math.factorial(i) for i in k])
    pa = np.array([math.exp(-la) * la ** i / math.factorial(i) for i in k])
    J = np.outer(ph, pa)
    tot = k[:, None] + k[None, :] + (k[:, None] == k[None, :])
    return [(x, float(J[tot > x].sum())) for x in xs]


def median_from(pts):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "kalshi_openers_xleague"))
    from kx import implied
    return implied(pts)


def predict(home, away, D, gh, ga, b2b_date=None):
    """b2b_date: the game's ET date (b2b = team played the day before the GAME, schedule-known)."""
    home, away = ESPN2MP.get(home, home), ESPN2MP.get(away, away)
    R, L, LO, LD, gs = state(pd.Timestamp(D))
    gd = pd.Timestamp(b2b_date if b2b_date is not None else D)
    bh, ba = played_prev(home, gd), played_prev(away, gd)
    lh = L * (R[home][0] / LO) * (R[away][1] / LD) * gfac(ga, gs) * (1 + HOME)
    la = L * (R[away][0] / LO) * (R[home][1] / LD) * gfac(gh, gs) * (1 - HOME)
    lh += B2B_OWN * bh + B2B_OPP * ba
    la += B2B_OWN * ba + B2B_OPP * bh
    return lh, la, median_from(p_over_ladder(lh, la)), bh, ba
