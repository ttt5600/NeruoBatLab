"""NFL comebacks, style matchups, coaching and referees -- all measured against the MARKET.

    python scripts/nfl_comebacks.py            # full run, writes notes/nfl_comebacks_matchups_tables.md + json

DEV SEASONS ONLY: 2006-2019 (the lab NFL lockbox is 2020-2025, frozen manifest seals 2025; 2026 is forward).
Benchmarks:
  * in-game  : nflfastR vegas_wp (spread-aware win probability) at the game state.  residual = outcome - vegas_wp
  * pre-game : closing spread / total in the nflverse schedule.  ATS resid = (home - away) - spread_line,
               total resid = points - total_line.
Every team trait is SEASON-TO-DATE from earlier games only (>= MIN_GP prior games); coach and referee
traits are expanding over all earlier dev games of that coach / referee (>= 8 prior games).
Every test is counted; BH (q .05) and Bonferroni over the whole family; sign checked on 2006-12 and 2013-19.

Data: data/pbp/play_by_play_YYYY.parquet (nflverse), data/pbp/pbp_participation_YYYY.parquet (2016-19:
pressure / pass rushers; man-zone only 2018-19), data/nfl_schedule.csv (closing lines, referee, coaches).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEV = list(range(2006, 2020))
EARLY, LATE = range(2006, 2013), range(2013, 2020)
MIN_GP = 4
CHECKPOINTS = {"end Q1 (45:00 left)": 2700, "half (30:00)": 1800, "mid Q3 (22:30)": 1350,
               "end Q3 (15:00)": 900, "10:00 Q4": 600, "5:00 Q4": 300, "2:00 Q4": 120}
DEF_BINS = [0, 3, 7, 10, 14, 17, 21, 28, 99]
DEF_LBL = ["1-3", "4-7", "8-10", "11-14", "15-17", "18-21", "22-28", "29+"]
OUT_MD = ROOT / "notes" / "nfl_comebacks_matchups_tables.md"
OUT_JSON = ROOT / "data" / "lab" / "nfl_comebacks_results.json"

PBP_COLS = ["game_id", "play_id", "season", "week", "season_type", "home_team", "away_team", "posteam",
            "defteam", "qtr", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
            "yardline_100", "posteam_score", "defteam_score", "score_differential", "wp", "vegas_wp",
            "vegas_home_wp", "play_type", "pass", "rush", "qb_dropback", "no_huddle", "epa", "pass_oe",
            "cpoe", "interception", "fumble_lost", "penalty", "penalty_team", "penalty_type", "penalty_yards",
            "home_coach", "away_coach", "result", "total", "spread_line", "total_line", "qb_kneel", "qb_spike"]


# ----------------------------------------------------------------------------------------------- load
def load_pbp() -> pd.DataFrame:
    out = []
    for y in DEV:
        p = pd.read_parquet(DATA / "pbp" / f"play_by_play_{y}.parquet", columns=PBP_COLS)
        pf = DATA / "pbp" / f"pbp_participation_{y}.parquet"
        if pf.exists():
            pt = pd.read_parquet(pf, columns=["nflverse_game_id", "play_id", "was_pressure",
                                              "number_of_pass_rushers", "defense_man_zone_type"])
            pt = pt.rename(columns={"nflverse_game_id": "game_id"}).drop_duplicates(["game_id", "play_id"])
            p = p.merge(pt, on=["game_id", "play_id"], how="left")
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    assert p.season.between(2006, 2019).all(), "sealed season leaked in"
    return p


def load_sched() -> pd.DataFrame:
    s = pd.read_csv(DATA / "nfl_schedule.csv")
    s = s[s.season.isin(DEV)].dropna(subset=["home_score", "spread_line", "total_line"]).copy()
    s["ats"] = s["result"] - s["spread_line"]          # >0 home covered
    s["tot_resid"] = s["total"] - s["total_line"]
    s["gdate"] = pd.to_datetime(s["gameday"])
    return s


# ------------------------------------------------------------------------------- team-game aggregates
def team_games(p: pd.DataFrame) -> pd.DataFrame:
    sc = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & (p.qb_kneel != 1) & (p.qb_spike != 1)].copy()
    sc["neutral"] = sc.wp.between(.2, .8) & (sc.half_seconds_remaining > 120) & sc.down.isin([1, 2])
    sc["trail_late"] = (sc.qtr >= 4) & (sc.score_differential < 0)
    sc["lead_late"] = (sc.qtr >= 4) & (sc.score_differential > 0)
    sc["db"] = sc.qb_dropback == 1
    sc["press"] = sc.db & (sc.was_pressure == True)  # noqa: E712
    sc["nopress"] = sc.db & (sc.was_pressure == False)  # noqa: E712
    sc["blitz"] = sc.db & (sc.number_of_pass_rushers >= 5)
    sc["noblitz"] = sc.db & (sc.number_of_pass_rushers <= 4)
    sc["man"] = sc.db & (sc.defense_man_zone_type == "MAN_COVERAGE")
    sc["zone"] = sc.db & (sc.defense_man_zone_type == "ZONE_COVERAGE")
    sc["to"] = sc.interception.fillna(0) + sc.fumble_lost.fillna(0)
    sc["one"] = 1.0

    sc["isrush"] = sc["rush"] == 1
    for m in ("db", "isrush", "trail_late", "lead_late", "press", "nopress", "blitz", "noblitz", "man", "zone"):
        sc[f"e_{m}"] = sc.epa.where(sc[m], 0.0)
        sc[m] = sc[m].astype(float)
    names = {"db": "passepa", "isrush": "rushepa", "press": "pressepa", "nopress": "nopressepa",
             "blitz": "blitzepa", "noblitz": "noblitzepa", "man": "manepa", "zone": "zoneepa"}
    cnt = {"db": "ndb", "isrush": "nrush", "press": "npress", "nopress": "nnopress", "blitz": "nblitz",
           "noblitz": "nnoblitz", "man": "nman", "zone": "nzone"}

    def agg(key: str, role: str) -> pd.DataFrame:
        late = "trail_late" if role == "o" else "lead_late"
        spec = {f"{role}_n": ("one", "sum"), f"{role}_epa": ("epa", "sum"), f"{role}_to": ("to", "sum"),
                f"{role}_nlate": (late, "sum"), f"{role}_lateepa": (f"e_{late}", "sum")}
        for m in names:
            spec[f"{role}_{names[m]}"] = (f"e_{m}", "sum")
            spec[f"{role}_{cnt[m]}"] = (m, "sum")
        d = sc.groupby(["game_id", key]).agg(**spec)
        d.index.names = ["game_id", "team"]
        return d

    off = agg("posteam", "o")
    de = agg("defteam", "d")
    ne = sc[sc.neutral & sc.pass_oe.notna()].groupby(["game_id", "posteam"]).agg(
        o_proe=("pass_oe", "sum"), o_nproe=("pass_oe", "size"))
    ne.index.names = ["game_id", "team"]
    cp = sc[sc.cpoe.notna()].groupby(["game_id", "posteam"]).agg(o_cpoe=("cpoe", "sum"), o_ncpoe=("cpoe", "size"))
    cp.index.names = ["game_id", "team"]
    # pace: no-huddle share of all snaps; urgency: no-huddle share when trailing in the 2nd half
    sc["trail2h"] = (sc.qtr >= 3) & (sc.score_differential < 0)
    nh = sc.groupby(["game_id", "posteam"]).agg(o_nh=("no_huddle", "sum"), o_nsnap=("one", "sum"))
    nh2 = sc[sc.trail2h].groupby(["game_id", "posteam"]).agg(o_nhtr=("no_huddle", "sum"), o_ntr=("one", "sum"))
    for x in (nh, nh2):
        x.index.names = ["game_id", "team"]
    # 4th-down decisions vs a league model (same-season league norm; team side is prior games only)
    f4 = p[(p.down == 4) & p.play_type.isin(["pass", "run", "punt", "field_goal"]) & p.posteam.notna()].copy()
    f4 = f4[f4.wp.between(.05, .95)]  # drop garbage-time / desperation
    f4["go"] = f4.play_type.isin(["pass", "run"]).astype(float)
    X = pd.DataFrame({"c": 1.0, "ytg": np.log1p(f4.ydstogo.clip(1, 30)), "yl": f4.yardline_100 / 100,
                      "yl2": (f4.yardline_100 / 100) ** 2, "sd": f4.score_differential.clip(-21, 21) / 7,
                      "t": f4.game_seconds_remaining / 3600, "sdt": f4.score_differential.clip(-21, 21) / 7 *
                      (1 - f4.game_seconds_remaining / 3600), "ytg_yl": np.log1p(f4.ydstogo.clip(1, 30)) * f4.yardline_100 / 100})
    f4["xgo"] = np.nan
    for y in DEV:
        m = f4.season == y
        fit = sm.Logit(f4.go[m], X[m]).fit(disp=0)
        f4.loc[m, "xgo"] = fit.predict(X[m])
    g4 = f4.groupby(["game_id", "posteam"]).agg(o_go=("go", "sum"), o_xgo=("xgo", "sum"), o_n4=("go", "size"))
    g4.index.names = ["game_id", "team"]
    # penalties against the team
    pen = p[p.penalty == 1].groupby(["game_id", "penalty_team"]).agg(pen_n=("penalty", "size"),
                                                                  pen_yds=("penalty_yards", "sum"))
    pen.index.names = ["game_id", "team"]
    tg = off.join(de, how="outer").join([ne, cp, nh, nh2, g4, pen], how="left").fillna(0).reset_index()
    return tg


def season_to_date(tg: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): pre-game season-to-date trait means using only earlier games."""
    h = sched[["game_id", "season", "gdate", "home_team", "away_team", "home_score", "away_score", "result"]]
    rows = pd.concat([
        h.assign(team=h.home_team, opp=h.away_team, pf=h.home_score, pa=h.away_score),
        h.assign(team=h.away_team, opp=h.home_team, pf=h.away_score, pa=h.home_score)])
    rows = rows.merge(tg, on=["game_id", "team"], how="left").sort_values(["team", "season", "gdate"])
    opp_to = tg[["game_id", "team", "o_to"]].rename(columns={"team": "opp", "o_to": "takeaways"})
    rows = rows.merge(opp_to, on=["game_id", "opp"], how="left")
    rows["win"] = np.sign(rows.pf - rows.pa)
    rows["one_score"] = ((rows.pf - rows.pa).abs() <= 8).astype(float)
    rows["one_score_win"] = rows.one_score * (rows.win > 0)
    rows["gp"] = 1.0
    sumcols = [c for c in tg.columns if c not in ("game_id", "team")] + ["takeaways", "one_score", "one_score_win",
                                                                        "gp", "pf", "pa"]
    cum = rows.groupby(["team", "season"])[sumcols].cumsum() - rows[sumcols].fillna(0)
    c = cum
    r = lambda a, b: c[a] / c[b].where(c[b] > 0)  # noqa: E731
    f = pd.DataFrame({
        "game_id": rows.game_id, "team": rows.team, "gp_prior": c.gp,
        "off_epa": r("o_epa", "o_n"), "def_epa": r("d_epa", "d_n"),
        "off_pass_epa": r("o_passepa", "o_ndb"), "off_rush_epa": r("o_rushepa", "o_nrush"),
        "def_pass_epa": r("d_passepa", "d_ndb"), "def_rush_epa": r("d_rushepa", "d_nrush"),
        "proe": r("o_proe", "o_nproe"), "cpoe": r("o_cpoe", "o_ncpoe"),
        "nohuddle": r("o_nh", "o_nsnap"), "urgency": r("o_nhtr", "o_ntr"),
        "off_late_trail_epa": r("o_lateepa", "o_nlate"), "def_late_lead_epa": r("d_lateepa", "d_nlate"),
        "press_sens": r("o_nopressepa", "o_nnopress") - r("o_pressepa", "o_npress"),
        "def_press_rate": c.d_npress / (c.d_npress + c.d_nnopress).where((c.d_npress + c.d_nnopress) > 0),
        "blitz_sens": r("o_noblitzepa", "o_nnoblitz") - r("o_blitzepa", "o_nblitz"),
        "def_blitz_rate": c.d_nblitz / (c.d_nblitz + c.d_nnoblitz).where((c.d_nblitz + c.d_nnoblitz) > 0),
        "man_minus_zone": r("o_manepa", "o_nman") - r("o_zoneepa", "o_nzone"),
        "def_man_rate": c.d_nman / (c.d_nman + c.d_nzone).where((c.d_nman + c.d_nzone) > 0),
        "to_margin_pg": (c.takeaways - c.o_to) / c.gp.where(c.gp > 0),
        "epa_margin": r("o_epa", "o_n") - r("d_epa", "d_n"),
        "one_score_luck": (c.one_score_win - 0.5 * c.one_score) / c.gp.where(c.gp > 0),
        "pt_margin_pg": (c.pf - c.pa) / c.gp.where(c.gp > 0),
        "plays_pg": (c.o_n + c.d_n) / c.gp.where(c.gp > 0),
        "pen_pg": c.pen_n / c.gp.where(c.gp > 0),
    })
    f["qb"] = f.off_pass_epa + f.cpoe.fillna(0) / 100  # pass EPA/dropback + CPOE/100 (nflfastR-style composite)
    # small-sample guards: participation splits need volume, otherwise NaN
    f.loc[(c.o_npress < 20) | (c.o_nnopress < 60), "press_sens"] = np.nan
    f.loc[(c.o_nblitz < 20) | (c.o_nnoblitz < 60), "blitz_sens"] = np.nan
    f.loc[(c.o_nman < 20) | (c.o_nzone < 40), "man_minus_zone"] = np.nan
    f.loc[(c.d_nman + c.d_nzone) < 60, "def_man_rate"] = np.nan
    f.loc[(c.d_npress + c.d_nnopress) < 60, "def_press_rate"] = np.nan
    f.loc[(c.d_nblitz + c.d_nnoblitz) < 60, "def_blitz_rate"] = np.nan
    f.loc[c.o_ntr < 15, "urgency"] = np.nan
    f.loc[c.o_nlate < 15, "off_late_trail_epa"] = np.nan
    f.loc[c.d_nlate < 15, "def_late_lead_epa"] = np.nan
    f.loc[f.gp_prior < MIN_GP] = f.loc[f.gp_prior < MIN_GP].assign(
        **{k: np.nan for k in f.columns if k not in ("game_id", "team", "gp_prior")})
    return f


def expanding_entity(sched: pd.DataFrame, tg: pd.DataFrame, p: pd.DataFrame):
    """Coach 4th-down aggressiveness and referee penalty rates from ALL earlier dev games (>= 8)."""
    s = sched.sort_values("gdate")
    # coach
    rows = pd.concat([s.assign(team=s.home_team, coach=s.home_coach), s.assign(team=s.away_team, coach=s.away_coach)])
    rows = rows[["game_id", "gdate", "team", "coach"]].merge(tg[["game_id", "team", "o_go", "o_xgo", "o_n4"]],
                                                             on=["game_id", "team"], how="left").fillna(
        {"o_go": 0, "o_xgo": 0, "o_n4": 0}).sort_values(["coach", "gdate"])
    rows["gp"] = 1.0
    cs = rows.groupby("coach")[["o_go", "o_xgo", "o_n4", "gp"]].cumsum() - rows[["o_go", "o_xgo", "o_n4", "gp"]]
    rows["coach_aggr"] = ((cs.o_go - cs.o_xgo) / cs.o_n4.where(cs.o_n4 >= 30))
    rows.loc[cs.gp < 8, "coach_aggr"] = np.nan
    coach = rows[["game_id", "team", "coach", "coach_aggr"]]
    # referee (head referee from schedule; nflverse does not list full crews)
    pg = p[p.penalty == 1].copy()
    pg["is_home_pen"] = pg.penalty_team == pg.home_team
    pg["dpi"] = pg.penalty_type == "Defensive Pass Interference"
    pg["hold"] = pg.penalty_type.isin(["Offensive Holding", "Defensive Holding"])
    gp = pg.groupby("game_id").agg(flags=("penalty", "size"), dpi=("dpi", "sum"), hold=("hold", "sum"),
                                   home_pen_yds=("penalty_yards", lambda x: x[pg.loc[x.index, "is_home_pen"]].sum()),
                                   all_pen_yds=("penalty_yards", "sum"))
    gp["home_minus_away_yds"] = 2 * gp.home_pen_yds - gp.all_pen_yds
    r = s[["game_id", "gdate", "referee"]].merge(gp, on="game_id", how="left").sort_values(["referee", "gdate"])
    r["gp"] = 1.0
    cols = ["flags", "dpi", "hold", "home_minus_away_yds", "gp"]
    cr = r.groupby("referee")[cols].cumsum() - r[cols].fillna(0)
    ref = pd.DataFrame({"game_id": r.game_id, "referee": r.referee, "ref_gp": cr.gp,
                        "ref_flags": cr["flags"] / cr.gp, "ref_dpi": cr.dpi / cr.gp, "ref_hold": cr.hold / cr.gp,
                        "ref_home_bias": cr.home_minus_away_yds / cr.gp, "flags_this_game": r["flags"]})
    ref.loc[ref.ref_gp < 8, ["ref_flags", "ref_dpi", "ref_hold", "ref_home_bias"]] = np.nan
    return coach, ref


# --------------------------------------------------------------------------------------- statistics
TESTS: list[dict] = []


def record(family: str, name: str, coef: float, se: float, p: float, n: int, unit: str,
           early: float, late: float, note: str = "", control: bool = False):
    TESTS.append(dict(family=family, name=name, coef=coef, lo=coef - 1.96 * se, hi=coef + 1.96 * se, p=p, n=n,
                      unit=unit, early=early, late=late, note=note, control=control))


def zs(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / x.std()


def ols(y, X, groups=None):
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X, missing="drop")
    if groups is not None:
        return m.fit(cov_type="cluster", cov_kwds={"groups": groups})
    return m.fit(cov_type="HC1")


def ols_test(d: pd.DataFrame, y: str, x: str, controls=(), cluster=None, season_col="season"):
    cols = [y, x, *controls, season_col] + ([cluster] if cluster else [])
    d = d[cols].dropna()
    if len(d) < 50:
        return None
    X = d[[x, *controls]].copy()
    X[x] = zs(X[x])
    for c in controls:
        X[c] = zs(X[c])
    fit = ols(d[y], X, d[cluster].astype("category").cat.codes if cluster else None)
    res = [fit.params[x], fit.bse[x], fit.pvalues[x], len(d)]
    halves = []
    for yrs in (EARLY, LATE):
        dd = d[d[season_col].isin(yrs)]
        Xh = d.loc[dd.index, [x, *controls]].copy()
        Xh[x] = (Xh[x] - d[x].mean()) / d[x].std()
        for c in controls:
            Xh[c] = (Xh[c] - d[c].mean()) / d[c].std()
        halves.append(ols(dd[y], Xh).params[x] if len(dd) > 30 else np.nan)
    return res + halves


# ------------------------------------------------------------------------------------- PART 1 comebacks
def game_states(p: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    q = p[p.posteam.notna() & p.vegas_home_wp.notna() & p.posteam_score.notna()].copy()
    q = q.sort_values(["game_id", "play_id"])
    q["home_diff"] = np.where(q.posteam == q.home_team, q.posteam_score - q.defteam_score,
                              q.defteam_score - q.posteam_score)
    out = []
    for lbl, t in CHECKPOINTS.items():
        w = q[(q.game_seconds_remaining <= t) & (q.game_seconds_remaining > t - 240) & (q.qtr <= 4)]
        w = w.groupby("game_id").head(1).assign(checkpoint=lbl, cp_sec=t)
        out.append(w)
    st = pd.concat(out)
    st = st.merge(sched[["game_id", "result", "spread_line", "total_line", "home_coach", "away_coach"]],
                  on="game_id", suffixes=("_pbp", ""))
    st = st[st.home_diff != 0].copy()
    st["trail"] = np.where(st.home_diff < 0, st.home_team, st.away_team)
    st["lead"] = np.where(st.home_diff < 0, st.away_team, st.home_team)
    st["trail_home"] = st.home_diff < 0
    st["deficit"] = st.home_diff.abs()
    st["dbin"] = pd.cut(st.deficit, DEF_BINS, labels=DEF_LBL)
    final = np.where(st.trail_home, st.result, -st.result)
    st["won"] = np.where(final > 0, 1.0, np.where(final == 0, 0.5, 0.0))
    st["vwp"] = np.where(st.trail_home, st.vegas_home_wp, 1 - st.vegas_home_wp)
    st["resid"] = st.won - st.vwp
    st["trail_spread"] = np.where(st.trail_home, st.spread_line, -st.spread_line)  # + = trailing team was favoured
    st["sbin"] = pd.cut(st.trail_spread, [-99, -7.25, -3.25, 3.25, 99],
                        labels=["trail dog >7", "trail dog 3.5-7", "within 3", "trail fav >3"])
    st["trail_ball"] = st.posteam == st.trail
    return st


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.96
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def comeback_tables(st: pd.DataFrame) -> dict:
    T = {}
    order = list(CHECKPOINTS)
    g = st.groupby(["checkpoint", "dbin"], observed=True).agg(n=("won", "size"), win=("won", "mean"),
                                                               vwp=("vwp", "mean"))
    T["main"] = g
    g2 = st[st.dbin.isin(["4-7", "8-10", "11-14", "15-17", "18-21"])].groupby(
        ["checkpoint", "dbin", "sbin"], observed=True).agg(n=("won", "size"), win=("won", "mean"), vwp=("vwp", "mean"))
    T["spread"] = g2
    g3 = st[st.cp_sec <= 900].groupby(["checkpoint", "dbin", "trail_ball"], observed=True).agg(
        n=("won", "size"), win=("won", "mean"), vwp=("vwp", "mean"))
    T["poss"] = g3
    # calibration of vegas_wp for trailing teams overall
    cal = st.copy()
    cal["vbin"] = pd.cut(cal.vwp, [0, .02, .05, .1, .2, .3, .4, .5])
    T["calib"] = cal.groupby("vbin", observed=True).agg(n=("won", "size"), win=("won", "mean"), vwp=("vwp", "mean"))
    T["order"] = order
    return T


def max_deficit_overcome(p: pd.DataFrame):
    q = p[p.posteam.notna() & p.posteam_score.notna()].copy()
    q["hd"] = np.where(q.posteam == q.home_team, q.posteam_score - q.defteam_score, q.defteam_score - q.posteam_score)
    mx = q.groupby("game_id").agg(minhd=("hd", "min"), maxhd=("hd", "max"), res=("result", "first"))
    mx["wmd"] = np.where(mx.res > 0, -mx.minhd, np.where(mx.res < 0, mx.maxhd, np.nan))
    return [(k, int((mx.wmd >= k).sum()), 100 * (mx.wmd >= k).mean()) for k in (7, 10, 14, 17, 21, 24, 28)]


def comeback_trait_tests(st: pd.DataFrame, f: pd.DataFrame, coach: pd.DataFrame):
    d = st[(st.deficit >= 7) & (st.cp_sec <= 1800) & st.vwp.between(.01, .99)].copy()
    tf = f.add_prefix("T_").rename(columns={"T_game_id": "game_id", "T_team": "trail"})
    lf = f.add_prefix("L_").rename(columns={"L_game_id": "game_id", "L_team": "lead"})
    d = d.merge(tf, on=["game_id", "trail"], how="left").merge(lf, on=["game_id", "lead"], how="left")
    d = d.merge(coach.rename(columns={"team": "trail", "coach_aggr": "T_coach_aggr"})[["game_id", "trail",
                                                                                        "T_coach_aggr"]],
                on=["game_id", "trail"], how="left")
    traits = {
        "T_off_epa": "trailing offense EPA/play", "T_proe": "trailing pass rate over expected",
        "T_nohuddle": "trailing no-huddle (pace) rate", "T_urgency": "trailing no-huddle rate when behind in 2H",
        "T_qb": "trailing QB (pass EPA/db + CPOE)", "T_off_late_trail_epa": "trailing Q4-when-behind offense EPA",
        "T_off_pass_epa": "trailing pass EPA/dropback", "L_def_pass_epa": "leader defense pass EPA allowed",
        "L_def_rush_epa": "leader defense rush EPA allowed", "L_def_late_lead_epa": "leader Q4-when-ahead D EPA allowed",
        "L_off_epa": "leader offense EPA/play", "T_coach_aggr": "trailing coach 4th-down aggressiveness",
        "total_line": "pregame total (vegas_wp ignores it)",
    }
    d["logit_v"] = np.log(d.vwp / (1 - d.vwp))
    # POSITIVE CONTROL: the same traits DO predict comebacks vs a WP model that ignores team strength (nflfastR wp,
    # no spread). If this is strong and the vegas_wp version is null, the trait is real but already in the spread.
    d["wp_trail"] = np.where(d.posteam == d.trail, d.wp, 1 - d.wp)
    d["epa_gap"] = d.T_epa_margin - d.L_epa_margin
    dd = d[["won", "wp_trail", "epa_gap", "game_id", "vwp"]].dropna()
    dd = dd[dd.wp_trail.between(.01, .99)]
    for off, lbl in (("wp_trail", "nflfastR wp (no spread)"), ("vwp", "vegas_wp (spread-aware)")):
        X = sm.add_constant(pd.DataFrame({"z": zs(dd.epa_gap)}))
        fit = sm.GLM(dd.won, X, family=sm.families.Binomial(), offset=np.log(dd[off] / (1 - dd[off]))).fit(
            cov_type="cluster", cov_kwds={"groups": dd.game_id.astype("category").cat.codes})
        pbar = dd[off].mean()
        halves = []
        for yrs in (EARLY, LATE):
            h = dd[d.loc[dd.index, "season"].isin(yrs)]
            Xh = sm.add_constant(pd.DataFrame({"z": (h.epa_gap - dd.epa_gap.mean()) / dd.epa_gap.std()}))
            halves.append(sm.GLM(h.won, Xh, family=sm.families.Binomial(),
                                 offset=np.log(h[off] / (1 - h[off]))).fit().params["z"])
        ctrl = off == "wp_trail"
        record("1 comeback vs vegas_wp", ("CONTROL: " if ctrl else "") + f"EPA-margin gap (trail - lead) vs {lbl}",
               100 * fit.params["z"] * pbar * (1 - pbar), 100 * fit.bse["z"] * pbar * (1 - pbar), fit.pvalues["z"],
               dd.game_id.nunique(), "pp per 1 SD", halves[0], halves[1], control=ctrl,
               note=f"states {len(dd)}")
    for x, lbl in traits.items():
        dd = d[["won", "vwp", "resid", "logit_v", x, "season", "game_id"]].dropna()
        if len(dd) < 200:
            continue
        z = zs(dd[x])
        # logistic with vegas_wp as offset; cluster by game
        X = sm.add_constant(pd.DataFrame({"z": z}))
        fit = sm.GLM(dd.won, X, family=sm.families.Binomial(), offset=dd.logit_v).fit(
            cov_type="cluster", cov_kwds={"groups": dd.game_id.astype("category").cat.codes})
        b, se, pv = fit.params["z"], fit.bse["z"], fit.pvalues["z"]
        # translate to pp at the sample mean vegas_wp
        pbar = dd.vwp.mean()
        pp = 100 * b * pbar * (1 - pbar)
        ppse = 100 * se * pbar * (1 - pbar)
        halves = []
        for yrs in (EARLY, LATE):
            h = dd[dd.season.isin(yrs)]
            Xh = sm.add_constant(pd.DataFrame({"z": z[h.index]}))
            halves.append(sm.GLM(h.won, Xh, family=sm.families.Binomial(), offset=h.logit_v).fit().params["z"])
        # tercile resid spread (top - bottom) in pp
        q = pd.qcut(dd[x], 3, labels=False, duplicates="drop")
        top, bot = dd.resid[q == q.max()], dd.resid[q == 0]
        record("1 comeback vs vegas_wp", f"{lbl} [{x}]", pp, ppse, pv, dd.game_id.nunique(), "pp per 1 SD",
               halves[0], halves[1], note=f"states {len(dd)}; top-bottom tercile resid {100 * (top.mean() - bot.mean()):+.1f} pp")
    # POST-HOC (added after reading table B): trailing team's pregame spread -> resid. vegas_wp already contains the
    # spread, so a nonzero slope means nflfastR lets the spread decay too fast (a benchmark flaw, maybe not a market one)
    dd = d[["won", "logit_v", "trail_spread", "season", "game_id", "vwp"]].dropna()
    X = sm.add_constant(pd.DataFrame({"z": zs(dd.trail_spread)}))
    fit = sm.GLM(dd.won, X, family=sm.families.Binomial(), offset=dd.logit_v).fit(
        cov_type="cluster", cov_kwds={"groups": dd.game_id.astype("category").cat.codes})
    pbar = dd.vwp.mean()
    halves = [sm.GLM(h.won, sm.add_constant(pd.DataFrame({"z": X.z[h.index]})), family=sm.families.Binomial(),
                     offset=h.logit_v).fit().params["z"] for h in (dd[dd.season.isin(EARLY)], dd[dd.season.isin(LATE)])]
    record("1 comeback vs vegas_wp", "POST-HOC: trailing team's pregame spread (fav +) [trail_spread]",
           100 * fit.params["z"] * pbar * (1 - pbar), 100 * fit.bse["z"] * pbar * (1 - pbar), fit.pvalues["z"],
           dd.game_id.nunique(), "pp per 1 SD", halves[0], halves[1], note=f"states {len(dd)}; chosen after seeing table B")
    return d


# --------------------------------------------------------------------------------- PART 2-5 game level
def game_frame(sched, f, coach, ref):
    g = sched.copy()
    hf = f.add_prefix("h_").rename(columns={"h_game_id": "game_id", "h_team": "home_team"})
    af = f.add_prefix("a_").rename(columns={"a_game_id": "game_id", "a_team": "away_team"})
    g = g.merge(hf, on=["game_id", "home_team"], how="left").merge(af, on=["game_id", "away_team"], how="left")
    ch = coach.rename(columns={"team": "home_team", "coach_aggr": "h_coach_aggr"})[["game_id", "home_team", "h_coach_aggr"]]
    ca = coach.rename(columns={"team": "away_team", "coach_aggr": "a_coach_aggr"})[["game_id", "away_team", "a_coach_aggr"]]
    g = g.merge(ch, on=["game_id", "home_team"], how="left").merge(ca, on=["game_id", "away_team"], how="left")
    g = g.merge(ref.drop(columns=["referee"]), on="game_id", how="left")
    return g


def game_tests(g: pd.DataFrame):
    # POSITIVE CONTROL: team quality predicts the raw margin (it is information), then is it in the spread?
    g["D_epa_margin_pc"] = g["h_epa_margin"] - g["a_epa_margin"]
    r = ols_test(g, "result", "D_epa_margin_pc")
    if r:
        record("5 team traits vs spread (ATS, home-away)", "CONTROL: EPA margin -> RAW home margin (no line)", r[0],
               r[1], r[2], r[3], "pts per 1 SD", r[4], r[5], control=True)
    # ---- 5: main effects of known team quality / style vs the closing spread (home - away diffs)
    diffs = {"epa_margin": "EPA margin/play", "off_epa": "offense EPA/play", "def_epa": "defense EPA allowed",
             "off_pass_epa": "pass offense EPA", "off_rush_epa": "rush offense EPA", "proe": "pass rate over expected",
             "qb": "QB composite", "nohuddle": "no-huddle rate", "to_margin_pg": "turnover margin/game",
             "one_score_luck": "one-score-game record luck", "pt_margin_pg": "point margin/game",
             "pen_pg": "penalties/game", "coach_aggr": "coach 4th-down aggressiveness"}
    for k, lbl in diffs.items():
        g[f"D_{k}"] = g[f"h_{k}"] - g[f"a_{k}"]
        r = ols_test(g, "ats", f"D_{k}")
        if r:
            record("5 team traits vs spread (ATS, home-away)", lbl, r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
    # "luck" traits net of EPA margin (classic market over-reaction tests)
    for k, lbl in {"to_margin_pg": "turnover margin | EPA margin", "one_score_luck": "one-score luck | EPA margin",
                   "pt_margin_pg": "point margin | EPA margin"}.items():
        r = ols_test(g, "ats", f"D_{k}", controls=("D_epa_margin",))
        if r:
            record("5 team traits vs spread (ATS, home-away)", lbl, r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
    sums = {"off_epa": "both offenses EPA", "def_epa": "both defenses EPA allowed", "proe": "both PROE",
            "nohuddle": "both no-huddle", "plays_pg": "both plays/game (pace)", "pen_pg": "both penalties/game",
            "coach_aggr": "both coaches 4th-down aggr"}
    for k, lbl in sums.items():
        g[f"S_{k}"] = g[f"h_{k}"] + g[f"a_{k}"]
        r = ols_test(g, "tot_resid", f"S_{k}")
        if r:
            record("5 team traits vs total", lbl, r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])

    # ---- 2: style x style interactions (home side matchup minus away side matchup), main effects controlled
    def mk(off, de, sign_off=1, sign_def=1):
        zh = sign_off * zs(g[f"h_{off}"]) * sign_def * zs(g[f"a_{de}"])
        za = sign_off * zs(g[f"a_{off}"]) * sign_def * zs(g[f"h_{de}"])
        return zh, za
    inter = {
        "pass-heavy O x pass-weak D": ("proe", "def_pass_epa", 1, 1),
        "pass-efficient O x pass-weak D": ("off_pass_epa", "def_pass_epa", 1, 1),
        "run-heavy O x run-weak D": ("proe", "def_rush_epa", -1, 1),
        "run-efficient O x run-weak D": ("off_rush_epa", "def_rush_epa", 1, 1),
        "pressure-sensitive QB x pressure D (2016-19)": ("press_sens", "def_press_rate", 1, 1),
        "blitz-sensitive QB x blitz-heavy D (2016-19)": ("blitz_sens", "def_blitz_rate", 1, 1),
        "QB better vs man x man-heavy D (2018-19)": ("man_minus_zone", "def_man_rate", 1, 1),
    }
    for lbl, spec in inter.items():
        off, de, so, sd = spec
        zh, za = mk(off, de, so, sd)
        g["I"] = zh - za
        g["Itot"] = zh + za
        g["c1"] = g[f"h_{off}"] - g[f"a_{off}"]
        g["c2"] = g[f"h_{de}"] - g[f"a_{de}"]
        r = ols_test(g, "ats", "I", controls=("c1", "c2"))
        if r:
            record("2 matchup -> ATS", lbl, r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
        g["c1"] = g[f"h_{off}"] + g[f"a_{off}"]
        g["c2"] = g[f"h_{de}"] + g[f"a_{de}"]
        r = ols_test(g, "tot_resid", "Itot", controls=("c1", "c2"))
        if r:
            record("2 matchup -> total", lbl, r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
    # tercile cell grid for the headline style matchup (home offence PROE tercile x away pass-D tercile), stacked both sides
    cells = []
    for side, opp, sgn in (("h", "a", 1), ("a", "h", -1)):
        x = g[[f"{side}_proe", f"{opp}_def_pass_epa", "ats", "season"]].dropna().copy()
        x["o"] = pd.qcut(x[f"{side}_proe"], 3, labels=["run-lean O", "mid O", "pass-heavy O"])
        x["d"] = pd.qcut(x[f"{opp}_def_pass_epa"], 3, labels=["good pass D", "mid pass D", "bad pass D"])
        x["y"] = sgn * x.ats
        cells.append(x)
    cells = pd.concat(cells)
    grid = cells.groupby(["o", "d"], observed=True).y.agg(["mean", "std", "size"])
    for (o, d_), row in grid.iterrows():
        se = row["std"] / np.sqrt(row["size"])
        sub = cells[(cells.o == o) & (cells.d == d_)]
        record("2 matchup grid (side ATS, both sides stacked)", f"{o} vs {d_}", row["mean"], se,
               2 * stats.norm.sf(abs(row["mean"] / se)), int(row["size"]), "pts ATS (side)",
               sub[sub.season.isin(EARLY)].y.mean(), sub[sub.season.isin(LATE)].y.mean(),
               note="stacked home+away rows: games appear twice across cells")

    # ---- 4: referees
    r = ols_test(g, "flags_this_game", "ref_flags")
    if r:
        record("4 referee", "POSITIVE CONTROL: ref prior flags/game -> flags this game", r[0], r[1], r[2], r[3],
               "flags per 1 SD", r[4], r[5], control=True)
    for x, lbl in {"ref_flags": "ref flags/game", "ref_dpi": "ref DPI/game", "ref_hold": "ref holding/game"}.items():
        r = ols_test(g, "tot_resid", x)
        if r:
            record("4 referee", f"{lbl} -> total resid", r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
    for x, lbl in {"ref_home_bias": "ref home-minus-away penalty yds", "ref_flags": "ref flags/game"}.items():
        r = ols_test(g, "ats", x)
        if r:
            record("4 referee", f"{lbl} -> home ATS", r[0], r[1], r[2], r[3], "pts per 1 SD", r[4], r[5])
    return g


def coach_tests(st: pd.DataFrame, g: pd.DataFrame, coach: pd.DataFrame):
    # 3a: halftime-trailing comeback residual by coach -- does a coach's residual persist (odd vs even seasons)?
    h = st[(st.cp_sec == 1800) & (st.deficit >= 1)].copy()
    h["tcoach"] = np.where(h.trail_home, h.home_coach, h.away_coach)
    h["par"] = h.season % 2
    piv = h.groupby(["tcoach", "par"]).resid.agg(["mean", "size"]).unstack()
    piv = piv[(piv["size"][0] >= 15) & (piv["size"][1] >= 15)]
    rr, pv = stats.pearsonr(piv["mean"][0], piv["mean"][1])
    se = (1 - rr ** 2) / np.sqrt(len(piv) - 2)
    record("3 coaching", "coach trailing-at-half resid, odd vs even seasons (persistence r)", rr, se, pv, len(piv),
           "Pearson r across coaches", np.nan, np.nan, note=f"coaches with >=15 trailing halves in both parities")
    # 3b: same for coach ATS
    rows = pd.concat([g.assign(coach=g.home_coach, y=g.ats), g.assign(coach=g.away_coach, y=-g.ats)])
    rows["par"] = rows.season % 2
    piv = rows.groupby(["coach", "par"]).y.agg(["mean", "size"]).unstack()
    piv = piv[(piv["size"][0] >= 24) & (piv["size"][1] >= 24)]
    rr, pv = stats.pearsonr(piv["mean"][0], piv["mean"][1])
    record("3 coaching", "coach ATS, odd vs even seasons (persistence r)", rr, (1 - rr ** 2) / np.sqrt(len(piv) - 2), pv,
           len(piv), "Pearson r across coaches", np.nan, np.nan, note=">=24 games in both parities")
    # 3c: trailing coach's 4th-down aggressiveness -> halftime trailing residual (already in part 1 family for 2H states)
    h = h.merge(coach.rename(columns={"team": "trail", "coach_aggr": "aggr"})[["game_id", "trail", "aggr"]],
                on=["game_id", "trail"], how="left")
    r = ols_test(h, "resid", "aggr", cluster=None)
    if r:
        record("3 coaching", "trailing coach 4th-down aggr -> trailing-at-half resid vs vegas_wp", 100 * r[0],
               100 * r[1], r[2], r[3], "pp per 1 SD", 100 * r[4], 100 * r[5])


def kalshi_dev_check() -> dict:
    """Does KALSHI already price what vegas_wp misses (trailing favourites / stronger trailing teams)?
    Uses ONLY the in-game DEV split pre-registered in scripts/nfl_ingame.py (2025 REG weeks 1-12, already mined for
    H1-H3). 2025 wk13+/POST and 2026 are that trial's sealed set and are not read here."""
    cd = pd.read_parquet(DATA / "nfl_ingame.parquet")
    games = pd.read_parquet(DATA / "nfl_ingame_games.parquet")
    dev = games[(games.season == 2025) & (games.stype == "REG") & (games.week <= 12)].index
    # running score: nfl_ingame_plays.home_score is the FINAL score, so read pbp (dev weeks only) for the live score
    pb = pd.read_parquet(DATA / "pbp" / "play_by_play_2025.parquet",
                         columns=["game_id", "play_id", "week", "season_type", "game_seconds_remaining", "posteam",
                                  "home_team", "posteam_score", "defteam_score", "vegas_home_wp", "time_of_day",
                                  "spread_line"])
    pb = pb[(pb.season_type == "REG") & (pb.week <= 12) & pb.game_id.isin(dev) & pb.posteam.notna()
            & pb.posteam_score.notna() & pb.vegas_home_wp.notna() & pb.time_of_day.notna()].copy()
    pb["t"] = pd.to_datetime(pb.time_of_day, utc=True, format="ISO8601")
    hs = np.where(pb.posteam == pb.home_team, pb.posteam_score, pb.defteam_score)
    as_ = np.where(pb.posteam == pb.home_team, pb.defteam_score, pb.posteam_score)
    plays = pb.assign(home_score=hs, away_score=as_)
    sl = pb.groupby("game_id").spread_line.first().reset_index()
    rows = []
    for gid, gp in plays.groupby("game_id"):
        c = cd[(cd.game_id == gid) & (cd.side == "home")].sort_values("end")
        if c.empty:
            continue
        gp = gp.sort_values(["t", "play_id"])
        ends = c.end.to_numpy()
        for lbl, t in CHECKPOINTS.items():
            w = gp[(gp.game_seconds_remaining <= t) & (gp.game_seconds_remaining > t - 240)]
            if w.empty:
                continue
            r = w.iloc[0]
            ts = (r.t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
            i = np.searchsorted(ends, ts + 60)
            if i >= len(c):
                continue
            b, a = c.bid.iloc[i], c.ask.iloc[i]
            if not (0 < b and a < 1 and a - b <= .05):
                continue
            rows.append(dict(game_id=gid, cp=t, hd=r.home_score - r.away_score, vh=r.vegas_home_wp,
                             kh=(b + a) / 2, result=games.loc[gid, "result"]))
    k = pd.DataFrame(rows).merge(sl, on="game_id", how="left")
    k = k[k.hd != 0].copy()
    th = k.hd < 0
    k["v"] = np.where(th, k.vh, 1 - k.vh)
    k["kal"] = np.where(th, k.kh, 1 - k.kh)
    k["won"] = np.where(np.where(th, k.result, -k.result) > 0, 1.0, 0.0)
    k["trail_spread"] = np.where(th, k.spread_line, -k.spread_line)
    k["deficit"] = k.hd.abs()
    k["gap"] = k.kal - k.v
    out = {"games": int(k.game_id.nunique()), "states": int(len(k)),
           "mean_kalshi_minus_vegaswp_pp": 100 * k.gap.mean()}
    for name, m in (("trail fav >3", k.trail_spread > 3), ("within 3", k.trail_spread.abs() <= 3),
                    ("trail dog >3", k.trail_spread < -3)):
        x = k[m & (k.deficit >= 4) & (k.cp <= 1800)]
        gm = x.groupby("game_id")[["gap", "won", "kal", "v"]].mean()
        out[name] = dict(states=int(len(x)), games=int(len(gm)), kalshi_minus_vwp_pp=100 * x.gap.mean(),
                         se_pp=100 * gm.gap.std() / np.sqrt(max(len(gm), 1)), win=x.won.mean(), kalshi=x.kal.mean(),
                         vwp=x.v.mean())
    x = k[(k.deficit >= 4) & (k.cp <= 1800)].dropna(subset=["trail_spread"])
    fit = ols(x.gap, pd.DataFrame({"z": zs(x.trail_spread)}), x.game_id.astype("category").cat.codes)
    record("1b Kalshi dev check (2025 wk1-12)", "Kalshi minus vegas_wp for trailing team ~ trailing pregame spread",
           100 * fit.params["z"], 100 * fit.bse["z"], fit.pvalues["z"], int(x.game_id.nunique()), "pp per 1 SD",
           np.nan, np.nan, note=f"states {len(x)}; + means Kalshi already gives trailing favourites more than vegas_wp")
    return out


def bh(ps, q=.05):
    ps = np.asarray(ps)
    n = len(ps)
    o = np.argsort(ps)
    thr = q * (np.arange(1, n + 1)) / n
    ok = ps[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    res = np.zeros(n, bool)
    res[o[:k]] = True
    return res


def fmt_tables(T, st, tests):
    L = []
    L.append("## A. Comeback base rates (dev 2006-2019, one snapshot per game per checkpoint)\n")
    L.append("Cell = trailing team's actual win rate (ties 0.5) / mean nflfastR vegas_wp at the same states / n games.\n")
    m = T["main"]
    hdr = "| time left | " + " | ".join(f"down {l}" for l in DEF_LBL) + " |"
    L += [hdr, "|" + "---|" * (len(DEF_LBL) + 1)]
    for cp in T["order"]:
        row = [cp]
        for l in DEF_LBL:
            try:
                r = m.loc[(cp, l)]
                row.append(f"{100 * r.win:.0f}% / {100 * r.vwp:.0f}% / {int(r.n)}")
            except KeyError:
                row.append("-")
        L.append("| " + " | ".join(row) + " |")
    L.append("\n## B. Same, split by the trailing team's PREGAME spread (actual / vegas_wp / n)\n")
    s = T["spread"]
    sb = ["trail fav >3", "within 3", "trail dog 3.5-7", "trail dog >7"]
    L += ["| time left | down | " + " | ".join(sb) + " |", "|---|---|" + "---|" * len(sb)]
    for cp in T["order"]:
        for l in ["4-7", "8-10", "11-14", "15-17", "18-21"]:
            row = [cp, l]
            for b in sb:
                try:
                    r = s.loc[(cp, l, b)]
                    row.append(f"{100 * r.win:.0f}% / {100 * r.vwp:.0f}% / {int(r.n)}")
                except KeyError:
                    row.append("-")
            L.append("| " + " | ".join(row) + " |")
    L.append("\n## C. Possession matters (2nd half+Q4 checkpoints): trailing team HAS the ball vs NOT\n")
    pz = T["poss"]
    L += ["| time left | down | trailing team has ball | opponent has ball |", "|---|---|---|---|"]
    for cp in T["order"]:
        if CHECKPOINTS[cp] > 900:
            continue
        for l in ["1-3", "4-7", "8-10", "11-14", "15-17"]:
            row = [cp, l]
            for b in (True, False):
                try:
                    r = pz.loc[(cp, l, b)]
                    row.append(f"{100 * r.win:.0f}% / {100 * r.vwp:.0f}% / {int(r.n)}")
                except KeyError:
                    row.append("-")
            L.append("| " + " | ".join(row) + " |")
    L.append("\n## D. Calibration of vegas_wp for the trailing team (all checkpoints, dev)\n")
    L += ["| vegas_wp bin | n states | actual | vegas_wp | 95% CI actual |", "|---|---|---|---|---|"]
    for b, r in T["calib"].iterrows():
        lo, hi = wilson(r.win * r.n, r.n)
        L.append(f"| {b} | {int(r.n)} | {100 * r.win:.1f}% | {100 * r.vwp:.1f}% | {100 * lo:.1f}-{100 * hi:.1f}% |")
    L.append("\n## E. How big a comeback is possible? Winner's largest deficit at any point (3,738 dev games)\n")
    L += ["| winner trailed by at least | games | share of all games |", "|---|---|---|"]
    for k, n_, sh in T["maxdef"]:
        L.append(f"| {k} | {n_} | {sh:.2f}% |")
    L.append(f"\n## F. Every hypothesis test ({int((~tests.control).sum())} counted; positive controls listed but not counted)\n")
    L.append("coef = effect per 1 SD of the trait (pp of win prob for family 1/3c; points for ATS/total). "
             "early/late = same coefficient fit on 2006-12 / 2013-19 (family 1: logit units, sign is what matters).\n")
    L += ["| family | test | effect | 95% CI | p | BH q.05 | Bonf | n | sign 06-12 / 13-19 | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, t in tests.iterrows():
        both = "both +" if (t.early > 0 and t.late > 0) else "both -" if (t.early < 0 and t.late < 0) else (
            "n/a" if pd.isna(t.early) else "FLIPS")
        L.append(f"| {t.family} | {t['name']} | {t.coef:+.3f} {t.unit} | [{t.lo:+.3f}, {t.hi:+.3f}] | {t.p:.3g} | "
                 f"{'PASS' if t.bh else '-'} | {'PASS' if t.bonf else '-'} | {t.n} | "
                 f"{t.early:+.3f} / {t.late:+.3f} ({both}) | {t.note} |")
    return "\n".join(L)


def main() -> int:
    tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "lab"
    p = load_pbp()
    sched = load_sched()
    print(f"pbp plays {len(p):,}  games {p.game_id.nunique():,}  seasons {p.season.min()}-{p.season.max()}")
    tg = team_games(p)
    f = season_to_date(tg, sched)
    coach, ref = expanding_entity(sched, tg, p)
    st = game_states(p, sched)
    print(f"states {len(st):,} over {st.game_id.nunique():,} games")
    T = comeback_tables(st)
    T["maxdef"] = max_deficit_overcome(p)
    comeback_trait_tests(st, f, coach)
    g = game_frame(sched, f, coach, ref)
    g = game_tests(g)
    coach_tests(st, g, coach)
    kd = kalshi_dev_check()
    print("KALSHI DEV CHECK", json.dumps(kd, indent=1, default=float))

    tests = pd.DataFrame(TESTS)
    fam = tests[~tests.control].copy()
    fam["bh"] = bh(fam.p.values)
    fam["bonf"] = fam.p < .05 / len(fam)
    ctrl = tests[tests.control].assign(bh=False, bonf=False)
    allt = pd.concat([fam, ctrl])
    print(f"tests counted: {len(fam)}; Bonferroni bar p < {.05 / len(fam):.5f}; BH passes {fam.bh.sum()}; "
          f"Bonf passes {fam.bonf.sum()}")
    print(allt.sort_values("p")[["family", "name", "coef", "lo", "hi", "p", "n", "early", "late", "bh"]]
          .head(25).to_string())
    OUT_MD.write_text("# NFL comebacks / matchups -- generated tables (scripts/nfl_comebacks.py)\n\n"
                      + fmt_tables(T, st, allt) + "\n")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"n_tests": int(len(fam)), "bonf_bar": .05 / len(fam),
                                    "tests": allt.replace({np.nan: None}).to_dict("records")}, indent=1, default=str))
    st.to_parquet(tmp / "nfl_states.parquet")
    g.to_parquet(tmp / "nfl_games_feat.parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
