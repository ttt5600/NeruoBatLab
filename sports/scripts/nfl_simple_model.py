"""Simple, explainable NFL game model (team EPA -> + QB -> + style), scored against the OPENER.

    python scripts/nfl_simple_model.py dev [--ledger]   # walk-forward dev backtest; writes json + tables
    python scripts/nfl_simple_model.py week [--refresh] [--week N]   # live: model spread/total for this week

No xgboost, no tuning: every number below was fixed BEFORE the market was joined (2026-10-25).

DATA / SPLITS (respecting lab/datasets.py SPLITS: NFL dev 2006-2019, lockbox 2020-2025; 2026 forward)
  * nflverse play-by-play 2006-2019 (features), nflverse schedule (results, starting QB ids, neutral flag).
  * OPENERS: sportsbookreviewsonline NFL archive, seasons 2007-2019 (dev only; 2020+ pages never fetched).
    Saved raw to data/sbr_nfl/nfl_odds_<season>.html. "Open"/"Close" are SBR's columns; which book the open
    comes from, and when it was posted, is undocumented. Spreads assumed -110 both sides.
  * `week` uses 2025 pbp only as the prior-season shrinkage target for live 2026 features; nothing from
    2020-2026 is ever scored or used to fit coefficients.

FEATURES (pre-game, from earlier games only; REG+POST plays with EPA, no kneels/spikes)
  Team rates are season-to-date, shrunk toward a prior = league + 0.5*(team last season - league):
      est = (sum_this_season + K * prior) / (n_this_season + K)
  K: pass EPA 250 dropbacks, run EPA 200 rushes, PROE 200 neutral plays, explosive 250 plays,
     pressure 250 dropbacks, pace 4 games (offensive plays per game).
  1. Team EPA (model A): offence/defence EPA per dropback and per designed run.
  2. QB (model B adds): QB rating = shrunk EPA/dropback (qb_epa) + 0.02 * shrunk CPOE.
       Per-QB state carries across seasons, decayed x0.6 at each new season; shrink K = 150 dropbacks
       toward a replacement prior = league EPA/db - 0.10. Starter = schedule home_qb_id/away_qb_id.
       qb_adj = rating(today's starter) - dropback-weighted rating of the QBs behind the team's
       season-to-date pass EPA (last season's mix weighted to 250 db as the prior).
     Key-player (WR1/LT/edge) availability: NOT built (injury/snap data are not local and are not clean
     back to 2007); stated as a gap, not silently dropped.
  3. Style (model C adds): pass rate over expected (neutral: wp .2-.8, downs 1-3), pace (off plays/game),
     explosive rate (pass >= 20 yds, run >= 10) made/allowed, pressure (sack or QB hit per dropback)
     generated/allowed; plus SIX pre-chosen interactions (z-scored with TRAIN-fold mean/SD):
       I1 spread: pass-heavy O x weak pass D   = z(h_proe)*z(a_dpass) - z(a_proe)*z(h_dpass)
       I2 spread: pass rush x leaky pass block = z(h_dpress)*z(a_opress) - z(a_dpress)*z(h_opress)
       I3 spread: explosive O x explosive-allowing D = z(h_oexpl)*z(a_dexpl) - z(a_oexpl)*z(h_dexpl)
       I4 total : pace x pace                  = z(h_pace)*z(a_pace)
       I5 total : bad D x bad D                = z(h_dEPA)*z(a_dEPA)     (lead from nfl_comebacks.py)
       I6 total : explosive O x explosive D    = z(h_oexpl)*z(a_dexpl) + z(a_oexpl)*z(h_dexpl)
  Targets: home margin (spread models), total points (total models). OLS, walk-forward by season:
  test season s in 2010-2019 is fit on 2007..s-1 only.

PRE-REGISTERED TRIALS (14; bar p < 0.05/14 = 0.00357; one-sided week-cluster bootstrap, 5000 draws)
  Bet rule: spread bet when |model - open| >= 2.0 pts, total bet when |model - open| >= 2.5 pts,
  side = the model's side of the OPEN line.
  T01-T03 spread CLV A/B/C : mean (close - open) moved toward the bet, in points.
  T04-T06 spread ROI A/B/C : result graded at the OPEN line, -110 (push = 0).
  T07-T09 total  CLV A/B/C
  T10-T12 total  ROI A/B/C
  T13 opener excess error in QB-change games: mean(|margin-open| - |margin-close|) in games where either
      team's starter differs from its previous game's starter (same season) MINUS the same in other games.
  T14 model B spread CLV restricted to QB-change games.
  Pass = p < bar AND effect > 0 AND positive in both halves (2010-14, 2015-19).
  Reported but not counted: no-vig ROI (Kalshi maker proxy), stacking coefficients
  (margin-open ~ b(model-open); margin-close ~ b(model-close); close-open ~ b(model-open)),
  blind drift baselines, coefficient tables, n needed for 80% power at the bar.
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_JSON = DATA / "lab" / "nfl_simple_model_results.json"
OUT_TAB = ROOT / "notes" / "nfl_simple_model_tables.md"
LEDGER = ROOT / "lab" / "ledger.jsonl"
FEAT_SEASONS = list(range(2006, 2020))   # 2006 = prior only
TEST_SEASONS = list(range(2010, 2020))
FIRST_FIT = 2007
TEAM_MAP = {"SD": "LAC", "OAK": "LV", "STL": "LA", "LAR": "LA"}
K = {"pass": 250, "run": 200, "proe": 200, "expl": 250, "press": 250, "pace": 4}
QB_K, QB_DECAY, QB_REPL, CPOE_W = 150.0, 0.6, -0.10, 0.02
SPREAD_T, TOTAL_T = 2.0, 2.5
N_TRIALS = 14
BAR = 0.05 / N_TRIALS
PBP_COLS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "pass", "rush", "qb_dropback",
            "qb_kneel", "qb_spike", "epa", "qb_epa", "cpoe", "pass_oe", "sack", "qb_hit", "yards_gained", "id",
            "wp", "down"]


# ------------------------------------------------------------------------------------------- data
def load_pbp(seasons) -> pd.DataFrame:
    out = []
    for y in seasons:
        p = pd.read_parquet(DATA / "pbp" / f"play_by_play_{y}.parquet", columns=PBP_COLS)
        out.append(p)
    p = pd.concat(out, ignore_index=True)
    p = p[p["epa"].notna() & ((p["pass"] == 1) | (p["rush"] == 1))
          & (p["qb_kneel"] != 1) & (p["qb_spike"] != 1) & p["posteam"].notna()]
    for c in ("posteam", "defteam"):
        p[c] = p[c].replace(TEAM_MAP)
    return p


def load_schedule() -> pd.DataFrame:
    s = pd.read_csv(DATA / "nfl_schedule.csv")
    s = s[s["game_type"].notna()].copy()
    for c in ("home_team", "away_team"):
        s[c] = s[c].replace(TEAM_MAP)
    s["gameday"] = pd.to_datetime(s["gameday"])
    s["neutral"] = (s["location"] == "Neutral").astype(int)
    return s


def team_games(p: pd.DataFrame) -> pd.DataFrame:
    p = p.assign(
        db=(p["qb_dropback"] == 1).astype(int),
        run=((p["rush"] == 1) & (p["qb_dropback"] != 1)).astype(int),
        expl=(((p["pass"] == 1) & (p["yards_gained"] >= 20)) | ((p["rush"] == 1) & (p["yards_gained"] >= 10))).astype(int),
        neutral=(p["pass_oe"].notna() & p["wp"].between(.2, .8) & p["down"].between(1, 3)).astype(int))
    p["press"] = (p["db"] == 1) & ((p["sack"] == 1) | (p["qb_hit"] == 1))
    p["pass_epa"] = np.where(p["db"] == 1, p["epa"], 0.0)
    p["run_epa"] = np.where(p["run"] == 1, p["epa"], 0.0)
    p["proe"] = np.where(p["neutral"] == 1, p["pass_oe"] / 100.0, 0.0)
    agg = dict(db=("db", "sum"), pass_epa=("pass_epa", "sum"), run=("run", "sum"), run_epa=("run_epa", "sum"),
               expl=("expl", "sum"), plays=("epa", "size"), press=("press", "sum"),
               proe=("proe", "sum"), neutral=("neutral", "sum"))
    o = p.groupby(["game_id", "season", "week", "posteam"]).agg(**agg).reset_index().rename(columns={"posteam": "team"})
    d = p.groupby(["game_id", "posteam", "defteam"]).agg(**{k: v for k, v in agg.items()
                                                            if k in ("db", "pass_epa", "run", "run_epa", "expl", "plays", "press")})
    d = d.reset_index().rename(columns={"defteam": "team"}).drop(columns="posteam")
    d.columns = ["game_id", "team"] + ["d_" + c for c in d.columns[2:]]
    o.columns = list(o.columns[:4]) + ["o_" + c for c in o.columns[4:]]
    return o.merge(d, on=["game_id", "team"])


RATES = {  # name: (numerator, denominator, K key)
    "o_pass": ("o_pass_epa", "o_db", "pass"), "o_run": ("o_run_epa", "o_run", "run"),
    "d_pass": ("d_pass_epa", "d_db", "pass"), "d_run": ("d_run_epa", "d_run", "run"),
    "o_proe": ("o_proe", "o_neutral", "proe"), "o_expl": ("o_expl", "o_plays", "expl"),
    "d_expl": ("d_expl", "d_plays", "expl"), "o_press": ("o_press", "o_db", "press"),
    "d_press": ("d_press", "d_db", "press"),
}


def team_features(tg: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date shrunk rates for each (game_id, team), from strictly earlier games."""
    tg = tg.sort_values(["team", "season", "week"]).copy()
    tg["games"] = 1
    sums = [c for c in tg.columns if c.startswith(("o_", "d_"))] + ["games"]
    season_tot = tg.groupby(["season", "team"])[sums].sum()
    league = tg.groupby("season")[sums].sum()
    cum = tg.groupby(["season", "team"])[sums].cumsum() - tg[sums]
    out = tg[["game_id", "season", "week", "team"]].copy()
    for name, (num, den, kk) in RATES.items():
        lg = (league[num] / league[den])
        last = season_tot[num] / season_tot[den]
        prior_lg = tg["season"].map(lambda s: lg.get(s - 1, np.nan))
        prior_tm = pd.Series([last.get((s - 1, t), np.nan) for s, t in zip(tg["season"], tg["team"])], index=tg.index)
        prior = prior_lg + 0.5 * (prior_tm.fillna(prior_lg) - prior_lg)
        out[name] = (cum[num] + K[kk] * prior) / (cum[den] + K[kk])
    lg_ppg = league["o_plays"] / league["games"]
    tm_ppg = season_tot["o_plays"] / season_tot["games"]
    prior_lg = tg["season"].map(lambda s: lg_ppg.get(s - 1, np.nan))
    prior_tm = pd.Series([tm_ppg.get((s - 1, t), np.nan) for s, t in zip(tg["season"], tg["team"])], index=tg.index)
    prior = prior_lg + 0.5 * (prior_tm.fillna(prior_lg) - prior_lg)
    out["pace"] = (cum["o_plays"] + K["pace"] * prior) / (cum["games"] + K["pace"])
    out["gp"] = cum["games"]
    return out


def team_features_live(tg: pd.DataFrame, season: int) -> pd.DataFrame:
    """Same estimator, evaluated AFTER every game played so far in `season` (for the upcoming week)."""
    nxt = tg[tg["season"] == season].groupby("team").tail(1)[["team", "season"]].copy()
    fake = nxt.assign(game_id="NEXT_" + nxt["team"], week=99)
    for c in tg.columns:
        if c.startswith(("o_", "d_")):
            fake[c] = 0.0
    f = team_features(pd.concat([tg, fake], ignore_index=True))
    return f[f["game_id"].str.startswith("NEXT_")]


# ------------------------------------------------------------------------------------------- QB layer
def qb_games(p: pd.DataFrame) -> pd.DataFrame:
    d = p[(p["qb_dropback"] == 1) & p["id"].notna()]
    d = d.assign(cp=d["cpoe"].fillna(0.0), cpn=d["cpoe"].notna().astype(int))
    return d.groupby(["game_id", "season", "week", "posteam", "id"]).agg(
        db=("qb_dropback", "size"), epa=("qb_epa", "sum"), cp=("cp", "sum"), cpn=("cpn", "sum")).reset_index()


def qb_features(qg: pd.DataFrame, sched: pd.DataFrame, live_season: int | None = None):
    """Walk week by week: features for a week's games use only earlier weeks. Returns per (game_id, team)
    qb_rating, qb_base, qb_adj, qbchg. With live_season, also returns current states for the next week."""
    st: dict = {}                         # qb id -> [db, epa, cp, cpn]
    lg_epa = {}
    for s, g in qg.groupby("season"):
        lg_epa[s] = g["epa"].sum() / g["db"].sum()
    rows = []
    games = sched[sched["season"].isin(qg["season"].unique())].sort_values(["season", "week", "gameday"])
    qg_by_week = {k: v for k, v in qg.groupby(["season", "week"])}
    team_mix: dict = {}                   # (season, team) -> {qb: db}
    last_starter: dict = {}               # (season, team) -> qb id
    cur_season = None

    def rating(q, s):
        prior = lg_epa.get(s - 1, lg_epa.get(s, 0.0)) + QB_REPL
        v = st.get(q)
        if v is None:
            return prior
        return (v[1] + QB_K * prior) / (v[0] + QB_K) + CPOE_W * v[2] / (v[3] + QB_K)

    def base(s, t):
        cur = team_mix.get((s, t), {})
        prev = team_mix.get((s - 1, t), {})
        w = dict(cur)
        tot_prev = sum(prev.values())
        if tot_prev:
            for q, d in prev.items():
                w[q] = w.get(q, 0) + d * K["pass"] / tot_prev
        if not w:
            return rating(None, s)
        return sum(d * rating(q, s) for q, d in w.items()) / sum(w.values())

    for (s, wk), g in games.groupby(["season", "week"], sort=True):
        if s != cur_season:
            for v in st.values():
                for i in range(4):
                    v[i] *= QB_DECAY
            cur_season = s
        for _, r in g.iterrows():
            for side in ("home", "away"):
                t, q = r[f"{side}_team"], r[f"{side}_qb_id"]
                rq = rating(q, s)
                b = base(s, t)
                prev = last_starter.get((s, t))
                rows.append(dict(game_id=r["game_id"], team=t, qb_rating=rq, qb_base=b, qb_adj=rq - b,
                                 qbchg=int(prev is not None and isinstance(q, str) and q != prev)))
        wkq = qg_by_week.get((s, wk))
        if wkq is not None:
            for _, r in wkq.iterrows():
                v = st.setdefault(r["id"], [0.0, 0.0, 0.0, 0.0])
                v[0] += r["db"]; v[1] += r["epa"]; v[2] += r["cp"]; v[3] += r["cpn"]
                mix = team_mix.setdefault((s, r["posteam"]), {})
                mix[r["id"]] = mix.get(r["id"], 0) + r["db"]
        for _, r in g.iterrows():
            for side in ("home", "away"):
                if isinstance(r[f"{side}_qb_id"], str) and pd.notna(r["home_score"]):
                    last_starter[(s, r[f"{side}_team"])] = r[f"{side}_qb_id"]
    out = pd.DataFrame(rows)
    if live_season is None:
        return out
    return out, (lambda q, t: (rating(q, live_season), base(live_season, t), last_starter.get((live_season, t))))


# ------------------------------------------------------------------------------------------- design
BASE_S = ["off_pass", "def_pass", "off_run", "def_run"]
BASE_T = ["off_pass_sum", "def_pass_sum", "off_run_sum", "def_run_sum", "lg_pts"]
QB_S, QB_T = ["qb_adj_diff"], ["qb_adj_sum"]
STYLE_S = ["proe_diff", "expl_net", "press_net", "I1_passO_x_passD", "I2_rush_x_block", "I3_explO_x_explD"]
STYLE_T = ["pace_sum", "proe_sum", "expl_sum", "I4_pace_x_pace", "I5_badD_x_badD", "I6_explO_x_explD"]
MODELS = {"A": (BASE_S, BASE_T), "B": (BASE_S + QB_S, BASE_T + QB_T),
          "C": (BASE_S + QB_S + STYLE_S, BASE_T + QB_T + STYLE_T)}


def assemble(sched: pd.DataFrame, tf: pd.DataFrame, qf: pd.DataFrame, lg_pts: pd.Series) -> pd.DataFrame:
    g = sched.copy()
    feat = [c for c in tf.columns if c not in ("game_id", "team", "season", "week")]
    qcols = [c for c in qf.columns if c not in ("game_id", "team")]
    for side in ("home", "away"):
        f = tf[["game_id", "team"] + feat].rename(columns={c: f"{side[0]}_{c}" for c in feat} | {"team": f"{side}_team"})
        g = g.merge(f, on=["game_id", f"{side}_team"], how="inner")
        q = qf.rename(columns={c: f"{side[0]}_{c}" for c in qcols} | {"team": f"{side}_team"})
        g = g.merge(q, on=["game_id", f"{side}_team"], how="left")
    g = _derive(g, g["season"].map(lg_pts))
    g["margin"] = g["home_score"] - g["away_score"]
    g["points"] = g["home_score"] + g["away_score"]
    return g


def add_interactions(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """z-score with the TRAIN fold's pooled home+away mean/SD, then the six pre-chosen interactions."""
    df = df.copy()
    zs = {}
    for c in ["o_proe", "d_pass", "d_press", "o_press", "o_expl", "d_expl", "pace", "d_epa"]:
        pool = pd.concat([ref[f"h_{c}"], ref[f"a_{c}"]])
        mu, sd = pool.mean(), pool.std()
        zs[c] = {s: (df[f"{s}_{c}"] - mu) / sd for s in "ha"}
    z = lambda c, s: zs[c][s]
    df["I1_passO_x_passD"] = z("o_proe", "h") * z("d_pass", "a") - z("o_proe", "a") * z("d_pass", "h")
    df["I2_rush_x_block"] = z("d_press", "h") * z("o_press", "a") - z("d_press", "a") * z("o_press", "h")
    df["I3_explO_x_explD"] = z("o_expl", "h") * z("d_expl", "a") - z("o_expl", "a") * z("d_expl", "h")
    df["I4_pace_x_pace"] = z("pace", "h") * z("pace", "a")
    df["I5_badD_x_badD"] = z("d_epa", "h") * z("d_epa", "a")
    df["I6_explO_x_explD"] = z("o_expl", "h") * z("d_expl", "a") + z("o_expl", "a") * z("d_expl", "h")
    return df


def ols(X: pd.DataFrame, y: pd.Series):
    import statsmodels.api as sm
    return sm.OLS(y.to_numpy(float), sm.add_constant(X.to_numpy(float), has_constant="add")).fit()


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols, target, neutral_term: bool):
    train = add_interactions(train, train)
    test = add_interactions(test, train)
    xs = list(cols) + (["hfa_off"] if neutral_term else [])
    tr = train.assign(hfa_off=-train["neutral"])
    te = test.assign(hfa_off=-test["neutral"])
    m = ols(tr[xs], tr[target])
    import statsmodels.api as sm
    pred = m.predict(sm.add_constant(te[xs].to_numpy(float), has_constant="add"))
    return pred, m, xs


# ------------------------------------------------------------------------------------------- build
def build_dev():
    p = load_pbp(FEAT_SEASONS)
    sched = load_schedule()
    sched = sched[sched["season"].isin(FEAT_SEASONS) & sched["home_score"].notna()]
    tg = team_games(p)
    tf = team_features(tg)
    qf = qb_features(qb_games(p), sched)
    pts = sched.groupby("season").apply(lambda d: (d["home_score"] + d["away_score"]).mean())
    lg_pts = {s: pts.get(s - 1, np.nan) for s in FEAT_SEASONS}
    g = assemble(sched, tf, qf, pd.Series(lg_pts))
    g = g[g["season"] >= FIRST_FIT].dropna(subset=BASE_S + BASE_T + ["h_pace", "a_pace", "h_o_press", "a_o_press"])
    return g, p


# ------------------------------------------------------------------------------------------- SBR openers
SBR_NAME = {"Arizona": "ARI", "Atlanta": "ATL", "Baltimore": "BAL", "Buffalo": "BUF", "BuffaloBills": "BUF",
            "Carolina": "CAR", "Chicago": "CHI", "Cincinnati": "CIN", "Cleveland": "CLE", "Dallas": "DAL",
            "Denver": "DEN", "Detroit": "DET", "GreenBay": "GB", "Houston": "HOU", "HoustonTexans": "HOU",
            "Indianapolis": "IND", "Jacksonville": "JAX", "KansasCity": "KC", "LAChargers": "LAC", "LARams": "LA",
            "Miami": "MIA", "Minnesota": "MIN", "NYGiants": "NYG", "NYJets": "NYJ", "NewEngland": "NE",
            "NewOrleans": "NO", "Oakland": "LV", "Philadelphia": "PHI", "Pittsburgh": "PIT", "SanDiego": "LAC",
            "SanFrancisco": "SF", "Seattle": "SEA", "St.Louis": "LA", "TampaBay": "TB", "Tennessee": "TEN",
            "Washington": "WAS", "NewYork": None, "LosAngeles": None}


def _num(x):
    x = str(x).strip().lower()
    if x in ("pk", "p", "pick"):
        return 0.0
    try:
        return float(x)
    except ValueError:
        return np.nan


def load_sbr(sched: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in range(2007, 2020):
        t = pd.read_html(io.StringIO((DATA / "sbr_nfl" / f"nfl_odds_{y}.html").read_text()))[0]
        t.columns = t.iloc[0]
        t = t.iloc[1:].reset_index(drop=True)
        for i in range(0, len(t) - 1, 2):
            v, h = t.iloc[i], t.iloc[i + 1]
            md = int(v["Date"])
            yr = y if md // 100 >= 8 else y + 1
            date = pd.Timestamp(yr, md // 100, md % 100)
            rec = dict(season=y, date=date, v_name=v["Team"], h_name=h["Team"], v_final=int(v["Final"]),
                       h_final=int(h["Final"]), vh=v["VH"])
            for col, tag in (("Open", "open"), ("Close", "close")):
                a, b = _num(v[col]), _num(h[col])
                if np.isnan(a) or np.isnan(b):
                    rec[f"spread_{tag}"] = rec[f"total_{tag}"] = np.nan
                    continue
                if a >= b:   # total on visitor row, spread on home row -> home favoured
                    rec[f"total_{tag}"], rec[f"spread_{tag}"] = a, b
                else:
                    rec[f"total_{tag}"], rec[f"spread_{tag}"] = b, -a
                if rec[f"total_{tag}"] < 25:
                    rec[f"spread_{tag}"] = rec[f"total_{tag}"] = np.nan
            rows.append(rec)
    sb = pd.DataFrame(rows)
    # match to nflverse by season, date (+-1 day) and the final score; neutral games may swap V/H
    sc = sched[["game_id", "season", "gameday", "home_team", "away_team", "home_score", "away_score"]]
    out = []
    for _, r in sb.iterrows():
        c = sc[(sc["season"] == r["season"]) & ((sc["gameday"] - r["date"]).abs() <= pd.Timedelta(days=1))]
        hn, vn = SBR_NAME.get(r["h_name"]), SBR_NAME.get(r["v_name"])
        m = c[(c["home_score"] == r["h_final"]) & (c["away_score"] == r["v_final"])]
        flip = False
        if len(m) == 0:
            m = c[(c["home_score"] == r["v_final"]) & (c["away_score"] == r["h_final"])]
            flip = True
        if len(m) > 1:
            m = m[(m["home_team"] == (vn if flip else hn)) | (m["away_team"] == (hn if flip else vn))]
        if len(m) != 1:
            continue
        rec = dict(game_id=m["game_id"].iloc[0], spread_open=r["spread_open"], spread_close=r["spread_close"],
                   total_open=r["total_open"], total_close=r["total_close"])
        if flip:
            rec["spread_open"], rec["spread_close"] = -rec["spread_open"], -rec["spread_close"]
        out.append(rec)
    o = pd.DataFrame(out).drop_duplicates("game_id", keep=False)
    return o, len(sb)


# ------------------------------------------------------------------------------------------- scoring
def boot_p(vals: np.ndarray, clusters: np.ndarray, n=5000, seed=0):
    """One-sided p (mean <= 0) and 95% CI of the mean, resampling week clusters."""
    df = pd.DataFrame({"v": vals, "c": clusters})
    g = df.groupby("c")["v"].agg(["sum", "size"])
    s, k = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    means = s[idx].sum(1) / k[idx].sum(1)
    return float(np.mean(means <= 0)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grade(side: np.ndarray, line: np.ndarray, outcome: np.ndarray, payout: float):
    """side +1 = home/over, -1 = away/under; line on the same scale as outcome (margin or points)."""
    r = side * (outcome - line)
    return np.where(r > 0, payout, np.where(r < 0, -1.0, 0.0))


def score_rule(d: pd.DataFrame, model_col: str, mkt: str, thresh: float):
    o, c = d[f"{mkt}_open"].to_numpy(), d[f"{mkt}_close"].to_numpy()
    y = d["margin" if mkt == "spread" else "points"].to_numpy()
    gap = d[model_col].to_numpy() - o
    bet = np.abs(gap) >= thresh
    side = np.sign(gap)
    b = d[bet]
    clv = (side * (c - o))[bet]
    roi110 = grade(side, o, y, 100 / 110)[bet]
    roinv = grade(side, o, y, 1.0)[bet]
    cl = (b["season"].astype(str) + "_" + b["week"].astype(str)).to_numpy()
    res = {}
    for name, v in (("clv", clv), ("roi110", roi110), ("roi_novig", roinv)):
        pv, lo, hi = boot_p(v, cl)
        early = v[(b["season"] <= 2014).to_numpy()].mean()
        late = v[(b["season"] >= 2015).to_numpy()].mean()
        res[name] = dict(n=int(bet.sum()), mean=float(v.mean()), lo=lo, hi=hi, p=pv, early=float(early), late=float(late),
                         sd=float(v.std()))
    res["win_rate_open"] = float((roinv > 0).sum() / max((roinv != 0).sum(), 1))
    res["home_or_over_share"] = float((side[bet] > 0).mean())
    return res


def stack(d: pd.DataFrame, model_col: str, mkt: str):
    import statsmodels.api as sm
    y = d["margin" if mkt == "spread" else "points"]
    out = {}
    for name, dep, ref in (("outcome_vs_open", y - d[f"{mkt}_open"], d[f"{mkt}_open"]),
                           ("outcome_vs_close", y - d[f"{mkt}_close"], d[f"{mkt}_close"]),
                           ("move_vs_open", d[f"{mkt}_close"] - d[f"{mkt}_open"], d[f"{mkt}_open"])):
        x = (d[model_col] - ref).to_numpy(float)
        cl = (d["season"].astype(str) + "_" + d["week"].astype(str)).astype("category").cat.codes
        m = sm.OLS(dep.to_numpy(float), sm.add_constant(x)).fit(cov_type="cluster", cov_kwds={"groups": cl})
        out[name] = dict(b=float(m.params[1]), lo=float(m.conf_int()[1][0]), hi=float(m.conf_int()[1][1]),
                         p=float(m.pvalues[1]))
    return out


def n_needed(sd: float, effect: float, z_a: float = None) -> float:
    from scipy import stats
    z_a = stats.norm.ppf(1 - BAR) if z_a is None else z_a
    if effect <= 0:
        return float("inf")
    return float(((z_a + 0.8416) * sd / effect) ** 2)


def run_dev(ledger: bool):
    g, _ = build_dev()
    sched = load_schedule()
    sbr, n_sbr = load_sbr(sched)
    preds = []
    coefs = {}
    for s in TEST_SEASONS:
        tr = g[(g["season"] >= FIRST_FIT) & (g["season"] < s)]
        te = g[g["season"] == s].copy()
        for mname, (cs, ct) in MODELS.items():
            ps, ms, xs = fit_predict(tr, te, cs, "margin", True)
            pt, mt, xt = fit_predict(tr, te, ct, "points", False)
            te[f"spread_{mname}"], te[f"total_{mname}"] = ps, pt
        preds.append(te)
    P = pd.concat(preds, ignore_index=True)
    # full-dev coefficient tables (fit 2007-2019) for the "playing style" explanation
    for mname, (cs, ct) in MODELS.items():
        for tgt, cols, neu in (("margin", cs, True), ("points", ct, False)):
            _, m, xs = fit_predict(g, g.head(1), cols, tgt, neu)
            gi = add_interactions(g, g).assign(hfa_off=-g["neutral"])
            sds = gi[xs].std()
            coefs[f"{mname}_{tgt}"] = dict(n=int(m.nobs), r2=float(m.rsquared), rmse=float(np.sqrt(m.mse_resid)),
                                           terms={k: dict(b=float(b), se=float(se), p=float(pv), per_sd=float(b * sds[k]))
                                                  for k, b, se, pv in zip(["const"] + xs, m.params, m.bse, m.pvalues)
                                                  if k != "const"} | {"const": dict(b=float(m.params[0]))})
    # model accuracy (no market)
    acc = {}
    for mname in MODELS:
        acc[mname] = dict(spread_rmse=float(np.sqrt(((P["margin"] - P[f"spread_{mname}"]) ** 2).mean())),
                          total_rmse=float(np.sqrt(((P["points"] - P[f"total_{mname}"]) ** 2).mean())))
    # ---- market join (first done 2026-10-25 after the docstring was frozen)
    M = P.merge(sbr, on="game_id", how="inner")
    M = M.dropna(subset=["spread_open", "spread_close", "total_open", "total_close"])
    chk = M.merge(sched[["game_id", "spread_line", "total_line"]].rename(columns={"spread_line": "nv_spread", "total_line": "nv_total"}),
                  on="game_id")
    mk = dict(n_sbr_rows=n_sbr, n_pred=len(P), n_joined=len(M),
              spread_close_vs_nflverse_mae=float((chk["spread_close"] - chk["nv_spread"]).abs().mean()),
              spread_close_vs_nflverse_corr=float(np.corrcoef(chk["spread_close"], chk["nv_spread"])[0, 1]),
              total_close_vs_nflverse_mae=float((chk["total_close"] - chk["nv_total"]).abs().mean()))
    for name, line in (("open", "spread_open"), ("close", "spread_close")):
        mk[f"rmse_margin_{name}"] = float(np.sqrt(((M["margin"] - M[line]) ** 2).mean()))
        mk[f"rmse_points_{name}"] = float(np.sqrt(((M["points"] - M[line.replace('spread', 'total')]) ** 2).mean()))
    for mname in MODELS:
        mk[f"rmse_margin_{mname}"] = float(np.sqrt(((M["margin"] - M[f"spread_{mname}"]) ** 2).mean()))
        mk[f"rmse_points_{mname}"] = float(np.sqrt(((M["points"] - M[f"total_{mname}"]) ** 2).mean()))
    mk["drift_home_spread_close_minus_open"] = float((M["spread_close"] - M["spread_open"]).mean())
    mk["drift_total_close_minus_open"] = float((M["total_close"] - M["total_open"]).mean())
    mk["mean_abs_spread_move"] = float((M["spread_close"] - M["spread_open"]).abs().mean())
    mk["mean_abs_total_move"] = float((M["total_close"] - M["total_open"]).abs().mean())

    trials, detail = [], {}
    tn = 0
    for mkt, th in (("spread", SPREAD_T), ("total", TOTAL_T)):
        for metric in ("clv", "roi110"):
            for mname in MODELS:
                tn += 1
                r = score_rule(M, f"{mkt}_{mname}", mkt, th)
                detail[f"{mkt}_{mname}"] = r | {"stack": stack(M, f"{mkt}_{mname}", mkt)}
                x = r[metric]
                trials.append(dict(trial=f"T{tn:02d}_{mkt}_{metric}_{mname}", **x))
    # QB-change isolation
    q = M.assign(exc=(M["margin"] - M["spread_open"]).abs() - (M["margin"] - M["spread_close"]).abs(),
                 mv=(M["spread_close"] - M["spread_open"]).abs())
    a, b = q[q["qbchg"] == 1], q[q["qbchg"] == 0]
    rng = np.random.default_rng(1)
    diffs = np.array([rng.choice(a["exc"].to_numpy(), len(a)).mean() - rng.choice(b["exc"].to_numpy(), len(b)).mean()
                      for _ in range(5000)])
    obs = a["exc"].mean() - b["exc"].mean()
    trials.append(dict(trial="T13_qbchg_opener_excess_error", n=int(len(a)), mean=float(obs),
                       lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)),
                       p=float(np.mean(diffs <= 0)),
                       early=float(a[a.season <= 2014]["exc"].mean() - b[b.season <= 2014]["exc"].mean()),
                       late=float(a[a.season >= 2015]["exc"].mean() - b[b.season >= 2015]["exc"].mean()),
                       sd=float(a["exc"].std())))
    qb_detail = dict(n_qbchg=int(len(a)), n_other=int(len(b)), excess_qbchg=float(a["exc"].mean()),
                     excess_other=float(b["exc"].mean()), move_qbchg=float(a["mv"].mean()), move_other=float(b["mv"].mean()))
    r = score_rule(a, "spread_B", "spread", SPREAD_T)
    trials.append(dict(trial="T14_qbchg_spread_clv_B", **r["clv"]))
    qb_detail["B_in_qbchg"] = r
    qb_detail["A_in_qbchg"] = score_rule(a, "spread_A", "spread", SPREAD_T)
    for t in trials:
        t["passed"] = bool(t["p"] < BAR and t["mean"] > 0 and t["early"] > 0 and t["late"] > 0)
        t["n_needed_80pct"] = n_needed(t["sd"], abs(t["mean"]) if t["mean"] > 0 else 0.0)
    res = dict(ts=datetime.now(timezone.utc).isoformat(), bar=BAR, trials=trials, detail=detail, qb=qb_detail,
               market=mk, accuracy=acc, coefs=coefs)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    write_tables(res)
    for t in trials:
        print(f"{t['trial']:34s} n={t['n']:5d} mean={t['mean']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}] p={t['p']:.4f} "
              f"halves {t['early']:+.3f}/{t['late']:+.3f} pass={t['passed']}")
    print(json.dumps(mk, indent=1))
    if ledger:
        with LEDGER.open("a") as fh:
            for i, t in enumerate(trials, 1):
                fh.write(json.dumps(dict(ts=res["ts"], strategy=f"nfl_simple_{t['trial']}", dataset="nfl_simple_model",
                                         metric="clv" if "clv" in t["trial"] else ("roi" if "roi" in t["trial"] else "excess_err"),
                                         hypothesis="pre-registered in scripts/nfl_simple_model.py docstring",
                                         source="user 2026-10-25: simple EPA/QB/style model vs NFL opener",
                                         status="ok", trial_number=i, bonferroni_bar=BAR, passed=t["passed"],
                                         bets=t["n"], mean=t["mean"], ci_lo=t["lo"], ci_hi=t["hi"], p=t["p"],
                                         per_half={"2010-14": t["early"], "2015-19": t["late"]})) + "\n")
        print("appended", len(trials), "trials to", LEDGER)


def write_tables(res):
    L = ["# nfl_simple_model tables (generated by scripts/nfl_simple_model.py dev)", "",
         f"Bar p < {res['bar']:.5f} (0.05/{N_TRIALS}). Week-cluster bootstrap. CLV in points; ROI per unit staked.", "",
         "| trial | n | mean | 95% CI | p | 2010-14 | 2015-19 | pass | n for 80% power |", "|---|---|---|---|---|---|---|---|---|"]
    for t in res["trials"]:
        L.append(f"| {t['trial']} | {t['n']} | {t['mean']:+.4f} | [{t['lo']:+.4f}, {t['hi']:+.4f}] | {t['p']:.4f} | "
                 f"{t['early']:+.4f} | {t['late']:+.4f} | {'YES' if t['passed'] else 'no'} | "
                 f"{t['n_needed_80pct']:.0f} |")
    L += ["", "## Secondary (not counted): no-vig ROI, win rate at open, stacking", "",
          "| model | n bets | win% at open | ROI -110 | ROI no-vig [CI] | home/over share | stack outcome~(model-open) | stack outcome~(model-close) | stack move~(model-open) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, d in res["detail"].items():
        s = d["stack"]
        f = lambda x: f"{x['b']:+.3f} [{x['lo']:+.3f},{x['hi']:+.3f}] p{x['p']:.3g}"
        L.append(f"| {k} | {d['clv']['n']} | {100 * d['win_rate_open']:.1f}% | {d['roi110']['mean']:+.4f} | "
                 f"{d['roi_novig']['mean']:+.4f} [{d['roi_novig']['lo']:+.3f},{d['roi_novig']['hi']:+.3f}] | "
                 f"{d['home_or_over_share']:.2f} | {f(s['outcome_vs_open'])} | {f(s['outcome_vs_close'])} | {f(s['move_vs_open'])} |")
    L += ["", "## Market / accuracy", "", "```", json.dumps(res["market"], indent=1), json.dumps(res["accuracy"], indent=1),
          json.dumps({k: v for k, v in res["qb"].items() if not isinstance(v, dict)}, indent=1), "```", "",
          "## Coefficients (fit on all dev 2007-2019; per_sd = coefficient x feature SD, in points)", ""]
    for k, c in res["coefs"].items():
        L += [f"### {k}  (n {c['n']}, R2 {c['r2']:.3f}, RMSE {c['rmse']:.2f})", "", "| term | coef | se | p | pts per SD |",
              "|---|---|---|---|---|"]
        for t, v in c["terms"].items():
            if t == "const":
                L.append(f"| const | {v['b']:+.3f} | | | |")
            else:
                L.append(f"| {t} | {v['b']:+.3f} | {v['se']:.3f} | {v['p']:.3g} | {v['per_sd']:+.2f} |")
        L.append("")
    OUT_TAB.write_text("\n".join(L))


# ------------------------------------------------------------------------------------------- live week
def run_week(refresh: bool, week: int | None):
    import requests
    if refresh:
        r = requests.get("https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2026.parquet", timeout=300)
        r.raise_for_status()
        (DATA / "pbp" / "play_by_play_2026.parquet").write_bytes(r.content)
        r = requests.get("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", timeout=120)
        r.raise_for_status()
        (DATA / "nfl_schedule.csv").write_bytes(r.content)
    LIVE = 2026
    # coefficients: dev only (2007-2019)
    g, _ = build_dev()
    p = load_pbp([2025, LIVE])
    sched = load_schedule()
    tg = team_games(p)
    played = tg[tg["season"] == LIVE]["game_id"].unique()
    tf = team_features_live(tg, LIVE)
    s26 = sched[sched["season"] == LIVE]
    if week is None:
        pend = s26[s26["home_score"].isna()]
        week = int(pend["week"].min())
    wk = s26[s26["week"] == week].copy()
    done = sched[sched["season"].isin([2025, LIVE]) & sched["home_score"].notna() & sched["game_id"].isin(set(tg["game_id"]))]
    _, rate_fn = qb_features(qb_games(p), done, live_season=LIVE)
    rows = []
    for _, r in wk.iterrows():
        rec = r.to_dict()
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            f = tf[tf["team"] == t]
            for c in f.columns:
                if c not in ("game_id", "team", "season", "week"):
                    rec[f"{side[0]}_{c}"] = float(f[c].iloc[0]) if len(f) else np.nan
            q = r[f"{side}_qb_id"]
            rq, b, last = rate_fn(q if isinstance(q, str) else None, t)
            rec[f"{side[0]}_qb_adj"] = (rq - b) if isinstance(q, str) else 0.0
            rec[f"{side[0]}_qbchg"] = int(isinstance(q, str) and last is not None and q != last)
            rec[f"{side[0]}_qb"] = r[f"{side}_qb_name"]
        rows.append(rec)
    W = pd.DataFrame(rows)
    pts = sched[sched["season"] == 2025]
    lg = float((pts["home_score"] + pts["away_score"]).mean())
    # reuse assemble's arithmetic on the already-merged frame
    W = _derive(W, lg)
    out = W[["gameday", "away_team", "home_team", "a_qb", "h_qb", "qbchg", "spread_line", "total_line"]].copy()
    for mname, (cs, ct) in MODELS.items():
        ps, _, _ = fit_predict(g, W, cs, "margin", True)
        pt, _, _ = fit_predict(g, W, ct, "points", False)
        out[f"spr_{mname}"], out[f"tot_{mname}"] = np.round(ps, 1), np.round(pt, 1)
    out["spr_gap_B"] = (out["spr_B"] - out["spread_line"]).round(1)
    out["tot_gap_B"] = (out["tot_B"] - out["total_line"]).round(1)
    pd.set_option("display.width", 250)
    print(f"NFL {LIVE} week {week}: model home margin (spr_*) and total (tot_*); spread_line = nflverse current "
          f"line (home favoured +). Features from {len(played)} games played so far; coefficients fit on dev 2007-2019.")
    print("Model B is the proposed forward model. LIVE PREDICTION ONLY -- not a scored result.")
    print(out.to_string(index=False))
    flag = out[(out["spr_gap_B"].abs() >= SPREAD_T) | (out["tot_gap_B"].abs() >= TOTAL_T)]
    print(f"\nGames where model B differs from the current line by >= {SPREAD_T} (spread) or >= {TOTAL_T} (total): "
          f"{len(flag)} (paper only; the rule is defined vs the OPENER, which this table does not have)")


def _derive(g: pd.DataFrame, lg_pts: float) -> pd.DataFrame:
    g = g.copy()
    g["neutral"] = (g["location"] == "Neutral").astype(int)
    g["h_proe"], g["a_proe"] = g["h_o_proe"], g["a_o_proe"]
    g["h_d_epa"] = (g["h_d_pass"] + g["h_d_run"]) / 2
    g["a_d_epa"] = (g["a_d_pass"] + g["a_d_run"]) / 2
    g["off_pass"] = 10 * (g["h_o_pass"] - g["a_o_pass"])
    g["def_pass"] = 10 * (g["a_d_pass"] - g["h_d_pass"])
    g["off_run"] = 10 * (g["h_o_run"] - g["a_o_run"])
    g["def_run"] = 10 * (g["a_d_run"] - g["h_d_run"])
    g["off_pass_sum"] = 10 * (g["h_o_pass"] + g["a_o_pass"])
    g["def_pass_sum"] = 10 * (g["h_d_pass"] + g["a_d_pass"])
    g["off_run_sum"] = 10 * (g["h_o_run"] + g["a_o_run"])
    g["def_run_sum"] = 10 * (g["h_d_run"] + g["a_d_run"])
    g["lg_pts"] = lg_pts
    g["qb_adj_diff"] = 10 * (g["h_qb_adj"].fillna(0) - g["a_qb_adj"].fillna(0))
    g["qb_adj_sum"] = 10 * (g["h_qb_adj"].fillna(0) + g["a_qb_adj"].fillna(0))
    g["proe_diff"] = 100 * (g["h_o_proe"] - g["a_o_proe"])
    g["expl_net"] = 100 * ((g["h_o_expl"] - g["a_d_expl"]) - (g["a_o_expl"] - g["h_d_expl"]))
    g["press_net"] = 100 * ((g["h_d_press"] - g["a_d_press"]) - (g["h_o_press"] - g["a_o_press"]))
    g["pace_sum"] = g["h_pace"] + g["a_pace"]
    g["proe_sum"] = 100 * (g["h_o_proe"] + g["a_o_proe"])
    g["expl_sum"] = 100 * (g["h_o_expl"] + g["a_o_expl"] + g["h_d_expl"] + g["a_d_expl"])
    g["qbchg"] = ((g["h_qbchg"].fillna(0) + g["a_qbchg"].fillna(0)) > 0).astype(int)
    return g


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if cmd == "dev":
        run_dev("--ledger" in sys.argv)
    elif cmd == "week":
        wk = int(sys.argv[sys.argv.index("--week") + 1]) if "--week" in sys.argv else None
        run_week("--refresh" in sys.argv, wk)
    else:
        print(__doc__)
