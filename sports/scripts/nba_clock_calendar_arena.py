"""NBA clock / calendar / arena factors. Pre-registration: notes/nba_clock_calendar_arena.md (read it first).

    ../.venv_quant/bin/python scripts/nba_clock_calendar_arena.py [--ledger]

One run: B1-B4 on ESPN game lines (close, one book per event), B5 arena gate + props, B6 altitude props at OPEN.
DEV 2021-22..2024-25 (props 2022-23..2024-25); SEALED 2025-26 only for trials that pass DEV. No orders, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESPN = ROOT / "data" / "espn" / "nba"
OUT = ROOT / "results" / "nba_clock_calendar_arena"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_clock_calendar_arena"
DEV = [2022, 2023, 2024, 2025]
H1, H2 = (2022, 2023), (2024, 2025)
PROP_DEV = [2023, 2024, 2025]
SEALED = 2026
BOOK = {2023: "DraftKings", 2024: "DraftKings", 2025: "ESPN BET", 2026: "ESPN BET"}
K = 6
BAR = 0.05 / K
NB = 10000
SEED = 2026

NYZ, CHZ, DNZ, PHZ, LAZ = "America/New_York", "America/Chicago", "America/Denver", "America/Phoenix", "America/Los_Angeles"
TEAM_TZ = {**{t: NYZ for t in "ATL BOS BKN CHA CLE DET IND MIA NY ORL PHI TOR WSH".split()},
           **{t: CHZ for t in "CHI DAL HOU MEM MIL MIN NO OKC SA".split()},
           "DEN": DNZ, "UTAH": DNZ, "PHX": PHZ, **{t: LAZ for t in "GS LAC LAL POR SAC".split()}}
STATE_TZ = {"AZ": PHZ, "CO": DNZ, "UT": DNZ, "CA": LAZ, "OR": LAZ, "WA": LAZ, "NV": LAZ,
            "TX": CHZ, "OK": CHZ, "TN": CHZ, "LA": CHZ, "MN": CHZ, "WI": CHZ, "IL": CHZ,
            **{s: NYZ for s in "GA MA NY NC OH MI IN FL PA DC ON NJ QC".split()}}
CITY_TZ = {"Mexico City": "America/Mexico_City", "Paris": "Europe/Paris", "Berlin": "Europe/Berlin",
           "London": "Europe/London"}
METRO = {"Los Angeles": "LA", "Inglewood": "LA", "New York": "NY", "Brooklyn": "NY", "Miami": "MIA",
         "Las Vegas": "LV"}
HOME_METRO = {"LAL": "LA", "LAC": "LA", "NY": "NY", "BKN": "NY", "MIA": "MIA"}
PROP_MKT = {"total points": "points", "total rebounds": "rebounds", "total 3-point field goals": "threes"}


# ------------------------------------------------------------------------------------------------ games + lines
def venue_tz(city, state):
    if city in CITY_TZ:
        return CITY_TZ[city]
    return STATE_TZ.get(state)


def games(seasons) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(ESPN / f"games_{s}.parquet") for s in seasons], ignore_index=True)
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")
          & g.home.isin(TEAM_TZ) & g.away.isin(TEAM_TZ)].copy()
    g["event_id"] = g.event_id.astype(str)
    g["tz"] = [venue_tz(c, s) for c, s in zip(g.city, g.state)]
    g["et_date"] = g.start_utc.dt.tz_convert(NYZ).dt.tz_localize(None).dt.normalize()
    loc = [t.tz_convert(z) if z else pd.NaT for t, z in zip(g.start_utc, g.tz)]
    g["loc_hr"] = [t.hour + t.minute / 60 if z else np.nan for t, z in zip(loc, g.tz)]
    g["loc_dow"] = [t.dayofweek if z else -1 for t, z in zip(loc, g.tz)]  # Mon=0 .. Sun=6
    for side in ("home", "away"):
        g[f"bc_{side}"] = [(lambda u: u.hour + u.minute / 60)(t.tz_convert(TEAM_TZ[tm]))
                           for t, tm in zip(g.start_utc, g[side])]
    g["metro"] = g.city.map(METRO)
    g["margin"] = g.home_score - g.away_score
    g["pts"] = g.home_score + g.away_score
    return g.sort_values("start_utc").reset_index(drop=True)


def lines(season: int) -> pd.DataFrame:
    ln = pd.read_parquet(ESPN / f"lines_{season}.parquet")
    ln["event_id"] = ln.event_id.astype(str)
    ln = ln[ln.provider.isin(["DraftKings", "Draft Kings", "ESPN BET"])].dropna(subset=["home_ml", "away_ml"]).copy()
    ln["pri"] = (ln.provider == "ESPN BET").astype(int)
    ln = ln.sort_values("pri").drop_duplicates("event_id")
    return ln[["event_id", "provider", "home_ml", "away_ml", "spread", "home_spread_odds", "away_spread_odds",
               "total", "over_odds", "under_odds"]]


def schedule_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per game: away team's previous game (date, metro), days since, b2b, road-trip game #; home b2b too.
    Uses only that team's EARLIER games (sorted by start_utc)."""
    h = pd.DataFrame({"event_id": g.event_id, "team": g.home, "is_home": True, "t": g.start_utc,
                      "d": g.et_date, "metro": g.metro, "season": g.season})
    a = h.assign(team=g.away.values, is_home=False)
    tl = pd.concat([h, a], ignore_index=True).sort_values(["team", "t"])
    grp = tl.groupby(["team", "season"], sort=False)
    tl["prev_d"] = grp.d.shift(1)
    tl["prev_metro"] = grp.metro.shift(1)
    tl["days"] = (tl.d - tl.prev_d).dt.days
    # road-trip game number: consecutive away games ending here
    trip = np.zeros(len(tl), int)
    for _, x in tl.groupby(["team", "season"], sort=False):
        c = 0
        for i, home in zip(x.index, x.is_home.values):
            c = 0 if home else c + 1
            trip[tl.index.get_loc(i)] = c
    tl["trip_n"] = trip
    a_ = tl[~tl.is_home].set_index("event_id")[["prev_metro", "days", "trip_n"]].add_prefix("a_")
    h_ = tl[tl.is_home].set_index("event_id")[["days"]].add_prefix("h_")
    return g.join(a_, on="event_id").join(h_, on="event_id")


def ml_dec(o):
    o = np.asarray(o, float)
    return np.where(o > 0, 1 + o / 100, 1 + 100 / np.abs(o))


def bet_ats(x: pd.DataFrame, home: np.ndarray) -> pd.DataFrame:
    cov = x.margin.to_numpy() + x.spread.to_numpy()  # home cover margin
    s = np.where(home, cov, -cov)
    odds = np.where(home, x.home_spread_odds, x.away_spread_odds)
    d = ml_dec(odds)
    prof = np.where(s > 0, d - 1, np.where(s < 0, -1.0, 0.0))
    return pd.DataFrame({"event_id": x.event_id.values, "season": x.season.values, "date": x.et_date.values,
                         "side": np.where(home, "home_ats", "away_ats"), "odds": odds, "win": (s > 0).astype(int),
                         "push": (s == 0).astype(int), "profit": prof}).dropna(subset=["odds"])


def bet_ml(x: pd.DataFrame, home: np.ndarray) -> pd.DataFrame:
    odds = np.where(home, x.home_ml, x.away_ml)
    w = np.where(home, x.margin > 0, x.margin < 0)
    d = ml_dec(odds)
    return pd.DataFrame({"event_id": x.event_id.values, "season": x.season.values, "date": x.et_date.values,
                         "side": np.where(home, "home_ml", "away_ml"), "odds": odds, "win": w.astype(int),
                         "push": 0, "profit": np.where(w, d - 1, -1.0)}).dropna(subset=["odds"])


def bet_under(x: pd.DataFrame) -> pd.DataFrame:
    s = x.total.to_numpy() - x.pts.to_numpy()
    d = ml_dec(x.under_odds)
    return pd.DataFrame({"event_id": x.event_id.values, "season": x.season.values, "date": x.et_date.values,
                         "side": "under", "odds": x.under_odds.values, "win": (s > 0).astype(int),
                         "push": (s == 0).astype(int),
                         "profit": np.where(s > 0, d - 1, np.where(s < 0, -1.0, 0.0))}).dropna(subset=["odds"])


# ------------------------------------------------------------------------------------------------ stats
def cboot(v, clus, seed=SEED, nb=NB):
    c = pd.Series(np.asarray(v, float)).groupby(pd.factorize(pd.Series(clus))[0])
    s, k = c.sum().to_numpy(), c.size().to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), (nb, len(s)))
    return s[idx].sum(1) / k[idx].sum(1)


def summarize(b: pd.DataFrame, clus_col: str, halves="season") -> dict:
    if len(b) == 0:
        return {"bets": 0, "passed": False}
    v = b.profit.to_numpy()
    bs = cboot(v, b[clus_col].astype(str).to_numpy())
    out = {"bets": int(len(b)), "units": float(v.sum()), "roi": float(v.mean()), "win_rate": float(b.win.mean()),
           "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)),
           "boot_mean": float(bs.mean()), "cluster": clus_col,
           "by_season": {int(s): {"n": int(len(y)), "roi": float(y.profit.mean())} for s, y in b.groupby("season")}}
    if halves == "season":
        hs = [b[b.season.isin(H1)], b[b.season.isin(H2)]]
    else:
        ds = np.sort(pd.to_datetime(b.date).unique())
        med = ds[len(ds) // 2]
        hs = [b[pd.to_datetime(b.date) < med], b[pd.to_datetime(b.date) >= med]]
    out["half1"] = float(hs[0].profit.mean()) if len(hs[0]) else None
    out["half2"] = float(hs[1].profit.mean()) if len(hs[1]) else None
    out["half_n"] = [int(len(hs[0])), int(len(hs[1]))]
    out["passed"] = bool(out["p"] < BAR and (out["half1"] or -1) > 0 and (out["half2"] or -1) > 0)
    return out


# ------------------------------------------------------------------------------------------------ B1-B4
def game_trials(x: pd.DataFrame) -> dict:
    res, bets = {}, {}
    lt = lambda c: x[c] >= 22.0
    er = lambda c: x[c] <= 10.0
    dh = lt("bc_home") | er("bc_home")
    da = lt("bc_away") | er("bc_away")
    one = dh ^ da
    sel = x[one & x.spread.notna()]
    # bet the opponent of the disadvantaged team: home if away disadvantaged
    b1 = bet_ats(sel, (da[one & x.spread.notna()]).to_numpy())
    b1["comp"] = np.where((lt("bc_home") | lt("bc_away"))[sel.index], "late", "early")
    bets["B1"] = b1
    res["B1"] = summarize(b1, "event_id")
    res["B1"]["components"] = {c: summarize(y, "event_id") for c, y in b1.groupby("comp")}
    selm = x[one]
    res["B1"]["ml_close"] = summarize(bet_ml(selm, da[one].to_numpy()), "event_id")
    res["B1"]["who_disadvantaged"] = {"away": int(da[one].sum()), "home": int(dh[one].sum())}

    b2 = bet_under(x[(x.loc_hr <= 15.5) & x.total.notna()])
    bets["B2"] = b2
    res["B2"] = summarize(b2, "event_id")

    m3 = (x.loc_dow == 4) | (x.loc_dow.isin([5, 6]) & (x.loc_hr <= 15.5))
    s3 = x[m3]
    a3 = bet_ats(s3[s3.spread.notna()], np.ones(int(s3.spread.notna().sum()), bool))
    u3 = bet_under(s3[s3.total.notna()])
    b3 = pd.concat([a3, u3], ignore_index=True)
    bets["B3"] = b3
    res["B3"] = summarize(b3, "event_id")
    res["B3"]["components"] = {"home_ats": summarize(a3, "event_id"), "under": summarize(u3, "event_id"),
                               "friday_ats": summarize(a3[a3.event_id.isin(x[x.loc_dow == 4].event_id)], "event_id"),
                               "wkend_matinee_ats": summarize(a3[~a3.event_id.isin(x[x.loc_dow == 4].event_id)],
                                                              "event_id")}

    road = ~x.neutral & (x.a_days == 2) & x.spread.notna()
    party = x.a_prev_metro.notna() & (x.a_prev_metro != x.away.map(HOME_METRO))
    b4 = bet_ats(x[road & party], np.ones(int((road & party).sum()), bool))
    bets["B4"] = b4
    res["B4"] = summarize(b4, "event_id")
    res["B4"]["by_metro"] = {m: summarize(b4[b4.event_id.isin(x[x.a_prev_metro == m].event_id)], "event_id")
                             for m in ["LA", "NY", "MIA", "LV"]}
    ctrl = bet_ats(x[road & x.a_prev_metro.isna()], np.ones(int((road & x.a_prev_metro.isna()).sum()), bool))
    res["B4"]["control_one_off_day_not_party"] = summarize(ctrl, "event_id")
    return res, bets


# ------------------------------------------------------------------------------------------------ B5 arena gate
def team_box(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = games([s])
        g["home_team_id"], g["away_team_id"] = g.home_team_id.astype(str), g.away_team_id.astype(str)
        b = pd.read_parquet(ESPN / f"box_{s}.parquet")
        b["event_id"], b["team_id"] = b.event_id.astype(str), b.team_id.astype(str)
        b = b[b.stat.isin(["fieldGoalsMade", "fieldGoalsAttempted", "threePointFieldGoalsMade",
                           "threePointFieldGoalsAttempted"])]
        t = b.pivot_table(index=["event_id", "team_id"], columns="stat", values="value", aggfunc="sum").reset_index()
        t = t.merge(g[["event_id", "season", "home_team_id", "away_team_id", "home", "away", "venue_id", "neutral"]],
                    on="event_id")
        rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    t["is_home"] = t.team_id == t.home_team_id
    t["team"] = np.where(t.is_home, t.home, t.away)
    t["host"] = t.home  # home team of the game
    for st, m, a in (("p3", "threePointFieldGoalsMade", "threePointFieldGoalsAttempted"),
                     ("fg", "fieldGoalsMade", "fieldGoalsAttempted")):
        sm = t.groupby(["team", "season"])[m].transform("sum")
        sa = t.groupby(["team", "season"])[a].transform("sum")
        t[f"r_{st}"] = t[m] / t[a] - (sm - t[m]) / (sa - t[a])
    return t


def arena_effect(t: pd.DataFrame, min_home=60) -> pd.DataFrame:
    """E per (host team, main venue): visitor residual at host's main venue minus host's road opponents' residual."""
    t = t[~t.neutral]
    vis = t[~t.is_home]  # visitors shooting at host's venue
    main = vis.groupby("host").venue_id.agg(lambda v: v.value_counts().index[0])
    out = []
    for host, ven in main.items():
        v = vis[(vis.host == host) & (vis.venue_id == ven)]
        road_opp = t[t.is_home & (t.away == host)]  # host on the road; opponent (home) shooting
        if len(v) < min_home:
            continue
        r = {"host": host, "venue_id": ven, "n_home": int(len(v)), "n_road": int(len(road_opp))}
        for st in ("p3", "fg"):
            r[f"E_{st}"] = float(v[f"r_{st}"].mean() - road_opp[f"r_{st}"].mean())
        out.append(r)
    return pd.DataFrame(out)


def b5_gate() -> dict:
    t = team_box(range(2019, 2026))
    ev = arena_effect(t[t.season % 2 == 0]).set_index(["host", "venue_id"])
    od = arena_effect(t[t.season % 2 == 1]).set_index(["host", "venue_id"])
    j = ev.join(od, lsuffix="_even", rsuffix="_odd", how="inner")
    rng = np.random.default_rng(SEED)
    out = {"arenas": int(len(j))}
    for st in ("p3", "fg"):
        a, b = j[f"E_{st}_even"].to_numpy(), j[f"E_{st}_odd"].to_numpy()
        r = float(np.corrcoef(a, b)[0, 1])
        bs = []
        for _ in range(NB):
            i = rng.integers(0, len(a), len(a))
            if np.std(a[i]) > 0 and np.std(b[i]) > 0:
                bs.append(np.corrcoef(a[i], b[i])[0, 1])
        out[st] = {"r": r, "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                   "sd_E_even": float(a.std()), "sd_E_odd": float(b.std()), "gate": bool(np.quantile(bs, .025) > 0)}
    j.reset_index().to_csv(OUT / "b5_arena_split_half.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------------ props
def player_box(seasons, gsched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = gsched[gsched.season == s]
        b = pd.read_parquet(ESPN / f"box_{s}.parquet")
        b["event_id"], b["athlete_id"], b["team_id"] = (b.event_id.astype(str), b.athlete_id.astype(str),
                                                         b.team_id.astype(str))
        b = b[b.stat.isin(["minutes", "points", "rebounds", "threePointFieldGoalsMade"])]
        w = b.pivot_table(index=["event_id", "athlete_id", "team_id"], columns="stat", values="value",
                          aggfunc="first").reset_index().rename(columns={"threePointFieldGoalsMade": "threes"})
        w = w.merge(g[["event_id", "season", "et_date", "home_team_id", "away_team_id", "home", "away", "venue_id",
                       "neutral", "a_days", "a_trip_n"]], on="event_id")
        rows.append(w)
    p = pd.concat(rows, ignore_index=True)
    p = p[p.minutes > 0]
    p["home_team_id"], p["away_team_id"] = p.home_team_id.astype(str), p.away_team_id.astype(str)
    p["is_away"] = p.team_id == p.away_team_id
    return p


def quotes(seasons) -> pd.DataFrame:
    out = []
    for s in seasons:
        g = pd.read_parquet(ESPN / f"graded_{s}.parquet")
        g = g[g.market_n.isin(PROP_MKT) & (g.provider == BOOK[s]) & g.athlete_id.notna() & g.line_open.notna()]
        key = ["event_id", "athlete_id", "market_n", "line_open"]
        o = g[g.side == "over"].drop_duplicates(key).set_index(key)
        u = g[g.side == "under"].drop_duplicates(key).set_index(key)
        j = o[["dec_open", "actual"]].rename(columns={"dec_open": "d_o"}).join(
            u[["dec_open"]].rename(columns={"dec_open": "d_u"}), how="inner").reset_index()
        j = j[(j.d_o > 1) & (j.d_u > 1) & j.actual.notna()].rename(columns={"line_open": "L"})
        io, iu = 1 / j.d_o, 1 / j.d_u
        j["p_over"] = io / (io + iu)
        j["dist"] = (j.p_over - .5).abs()
        j = j.sort_values("dist").drop_duplicates(["event_id", "athlete_id", "market_n"])
        j["stat"] = j.market_n.map(PROP_MKT)
        out.append(j)
    q = pd.concat(out, ignore_index=True)
    q["event_id"], q["athlete_id"] = q.event_id.astype(str), q.athlete_id.astype(str)
    return q[q.actual != q.L].copy()


def prop_frame(seasons, gsched: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = player_box(seasons, gsched)
    q = quotes(seasons)
    m = q.merge(p, on=["event_id", "athlete_id"], how="left", indicator=True)
    audit = {"quotes": int(len(q)), "matched_box": int((m._merge == "both").sum())}
    m = m[m._merge == "both"].drop(columns="_merge")
    m["box_stat"] = [r[s] for r, s in zip(m[["points", "rebounds", "threes"]].to_dict("records"), m.stat)]
    audit["actual_eq_box"] = float((m.box_stat == m.actual).mean())
    m["won"] = (m.actual < m.L).astype(int)  # UNDER
    m["profit"] = np.where(m.won == 1, m.d_u - 1, -1.0)
    m["p_side"] = 1 - m.p_over
    m["win"] = m.won
    m["date"] = m.et_date
    return m, audit


def prop_summary(b: pd.DataFrame) -> dict:
    s = summarize(b, "date", halves="median_date")
    if len(b):
        s["hit_minus_novig"] = float(b.won.mean() - b.p_side.mean())
        s["by_stat"] = {k: {"n": int(len(y)), "roi": float(y.profit.mean())} for k, y in b.groupby("stat")}
    return s


def b6(m: pd.DataFrame) -> dict:
    alt = m.home.isin(["DEN", "UTAH"]) & ~m.neutral & m.is_away & m.stat.isin(["points", "rebounds"])
    b = m[alt]
    r = prop_summary(b)
    tired = (b.a_days == 1) | (b.a_trip_n >= 2)
    r["sub_b2b_or_trip2plus"] = prop_summary(b[tired])
    r["sub_rest"] = prop_summary(b[~tired])
    base = m[~m.home.isin(["DEN", "UTAH"]) & ~m.neutral & m.is_away & m.stat.isin(["points", "rebounds"])]
    r["baseline_visitor_unders_other_arenas"] = prop_summary(base)
    return r, b


def b5_props(m: pd.DataFrame, gate: dict) -> tuple[dict, pd.DataFrame]:
    stats = (["threes"] if gate["p3"]["gate"] else []) + (["points"] if gate["fg"]["gate"] else [])
    picks = []
    for s in sorted(m.season.unique()):
        t = team_box(range(2019, int(s)))
        e = arena_effect(t)
        cols = [f"E_{'p3' if st == 'threes' else 'fg'}" for st in stats]
        for st, col in zip(stats, cols):
            cut = e[col].quantile(.2)
            low = set(e[e[col] <= cut].venue_id)
            picks.append(m[(m.season == s) & m.is_away & ~m.neutral & (m.stat == st) & m.venue_id.isin(low)])
    b = pd.concat(picks) if picks else m.iloc[:0]
    return prop_summary(b), b


# ------------------------------------------------------------------------------------------------ checks
def checks(x: pd.DataFrame) -> dict:
    c = {"missing_tz": int(x.tz.isna().sum())}
    hm = x[~x.neutral & (x.loc_dow.isin([0, 1, 2, 3]))]
    c["home_local_tip_weeknight_share_17_to_21"] = {t: round(float(((y.loc_hr >= 17) & (y.loc_hr <= 21)).mean()), 3)
                                                    for t, y in hm.groupby("home")}
    # DST: PHX home weeknight local tip modal hour in Nov vs Mar (should be equal; ET differs by 1h)
    for tm in ("PHX", "LAL", "BOS"):
        y = hm[hm.home == tm]
        c[f"{tm}_modal_local_tip_by_month"] = {int(mo): float(z.loc_hr.mode().iloc[0])
                                              for mo, z in y.groupby(y.et_date.dt.month)}
    c["spread_sign_agrees_with_ml_fav"] = float(((x.spread < 0) == (x.home_ml < x.away_ml))[x.spread != 0].mean())
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    seasons_all = DEV + [SEALED]
    g = schedule_features(games(seasons_all))
    ln = pd.concat([lines(s) for s in seasons_all], ignore_index=True)
    x = g.merge(ln, on="event_id", how="left")
    out = {"checks": checks(x[x.season.isin(DEV)])}
    dev = x[x.season.isin(DEV)]
    res, bets = game_trials(dev)
    for k, b in bets.items():
        b.to_csv(OUT / f"bets_dev_{k}.csv", index=False)

    out["B5_gate"] = gate = b5_gate()
    m, audit = prop_frame(PROP_DEV, g)
    out["props_audit"] = audit
    out["props_base_rate_under"] = {"n": int(len(m)), "roi": float(m.profit.mean()),
                                    "hit_minus_novig": float(m.won.mean() - m.p_side.mean())}
    if gate["p3"]["gate"] or gate["fg"]["gate"]:
        res["B5"], b5b = b5_props(m, gate)
        b5b.to_csv(OUT / "bets_dev_B5.csv", index=False)
    else:
        res["B5"] = {"bets": 0, "passed": False, "note": "DEAD at step 1: neither split-half r CI excludes 0"}
    res["B6"], b6b = b6(m)
    b6b.to_csv(OUT / "bets_dev_B6.csv", index=False)
    out["dev"] = res

    passed = [k for k, r in res.items() if r.get("passed")]
    out["sealed"] = {}
    if passed:
        sbar = 0.05 / len(passed)
        sx = x[x.season == SEALED]
        sres, _ = game_trials(sx)
        ms, saudit = prop_frame([SEALED], g)
        out["sealed_props_audit"] = saudit
        if "B5" in passed:
            sres["B5"], _ = b5_props(ms, gate)
        sres["B6"], _ = b6(ms)
        for k in passed:
            r = sres[k]
            r["sealed_bar"] = sbar
            r["sealed_pass"] = bool(r.get("bets", 0) > 0 and r["p"] < sbar and r["roi"] > 0)
            out["sealed"][k] = r
    (OUT / "results.json").write_text(json.dumps(out, indent=1, default=str))

    print(json.dumps(out["checks"], default=str)[:1500])
    print("B5 gate", json.dumps(gate))
    print("props audit", audit, "base under", out["props_base_rate_under"])
    for k, r in res.items():
        if r.get("bets"):
            print(f"{k}: n {r['bets']} ROI {r['roi']:+.4f} CI [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p {r['p']:.4f} "
                  f"halves {r['half1']:+.4f}/{r['half2']:+.4f} pass {r['passed']}")
        else:
            print(k, r)
    print("sealed", json.dumps(out["sealed"], default=str)[:2000])

    if a.ledger:
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        names = {"B1": "ncca_B1_body_clock_ats", "B2": "ncca_B2_early_tip_under", "B3": "ncca_B3_fri_wkend_matinee",
                 "B4": "ncca_B4_nightlife_ats", "B5": "ncca_B5_arena_visitor_props", "B6": "ncca_B6_altitude_visitor_props"}
        with LEDGER.open("a") as fh:
            for i, (k, r) in enumerate(res.items(), 1):
                keep = {kk: vv for kk, vv in r.items() if not isinstance(vv, dict) or kk == "by_season"}
                fh.write(json.dumps({"ts": ts, "strategy": names[k], "sha": sha, "dataset": DATASET,
                                     "hypothesis": "pre-registered in notes/nba_clock_calendar_arena.md",
                                     "source": "coordinator brief 10-29", "status": "ok",
                                     "split": "DEV 2021-22..2024-25 (props 2022-23..2024-25)", "trial_number": i,
                                     "study_trials": K, "bonferroni_bar": BAR, **keep}, default=str) + "\n")
            for k, r in out["sealed"].items():
                keep = {kk: vv for kk, vv in r.items() if not isinstance(vv, dict) or kk == "by_season"}
                fh.write(json.dumps({"ts": ts, "strategy": names[k] + "_sealed", "sha": sha, "dataset": DATASET,
                                     "status": "ok", "split": "SEALED 2025-26", **keep}, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
