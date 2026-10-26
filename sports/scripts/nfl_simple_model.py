"""Simple, explainable NFL game model (team EPA -> + QB -> + style), scored against the OPENER.

    python scripts/nfl_simple_model.py dev [--ledger]   # walk-forward dev backtest; writes json + tables
    python scripts/nfl_simple_model.py refit            # live refit on 2020-2025: LOSO + legacy compare + 2026 level
    python scripts/nfl_simple_model.py week [--refresh] [--week N] [--fit modern|legacy] [--json]

LIVE REFIT (added 2026-10-25, see the "live refit" section below and notes/nfl_simple_model.md):
  `week` now defaults to coefficients fit on 2020-2025 (recency weight half-life 2 seasons) with the same A/B/C
  structure minus the six matchup interactions (all p > .27 in dev). `--fit legacy` reproduces the 2007-2019 fit.
  DECISION: fitting on 2020-2025 uses the seasons proposed as L33's SEALED set -> the NFL sealed test is FORFEITED
  and L33 is FORWARD-PAPER ONLY. 2026 is used for a level check only. The `dev` command below is unchanged.

No xgboost, no tuning: every number below was fixed BEFORE the market was joined (2026-10-25).

DATA / SPLITS (respecting lab/datasets.py SPLITS: NFL dev 2006-2019, lockbox 2020-2025; 2026 forward)
  * nflverse play-by-play 2006-2019 (features), nflverse schedule (results, starting QB ids, neutral flag).
  * OPENERS: sportsbookreviewsonline NFL archive, seasons 2007-2019 (dev only; 2020+ pages never fetched).
    Saved raw to data/sbr_nfl/nfl_odds_<season>.html. "Open"/"Close" are SBR's columns; which book the open
    comes from, and when it was posted, is undocumented. Spreads assumed -110 both sides.
  * (original 10-25 text, now superseded for `week`/`refit`): `week` used 2025 pbp only as the prior-season
    shrinkage target for live 2026 features. `dev` still never scores or fits on 2020-2026; `refit` and
    `week --fit modern` DO fit on 2020-2025 (sealed test forfeited, see top).

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


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols, target, neutral_term: bool, weights=None):
    train = add_interactions(train, train)
    test = add_interactions(test, train)
    xs = list(cols) + (["hfa_off"] if neutral_term else [])
    tr = train.assign(hfa_off=-train["neutral"])
    te = test.assign(hfa_off=-test["neutral"])
    import statsmodels.api as sm
    if weights is None:
        m = ols(tr[xs], tr[target])
    else:
        m = sm.WLS(tr[target].to_numpy(float), sm.add_constant(tr[xs].to_numpy(float), has_constant="add"),
                   weights=np.asarray(weights, float)).fit()
    pred = m.predict(sm.add_constant(te[xs].to_numpy(float), has_constant="add"))
    return pred, m, xs


# ------------------------------------------------------------------------------------------- build
def build_dev():
    return build_all(FEAT_SEASONS, FIRST_FIT)


def build_all(feat_seasons, first_fit):
    """Walk-forward pre-game features for every COMPLETED game in feat_seasons (season >= first_fit)."""
    p = load_pbp(feat_seasons)
    sched = load_schedule()
    sched = sched[sched["season"].isin(feat_seasons) & sched["home_score"].notna()
                  & sched["game_id"].isin(set(p["game_id"]))]
    tg = team_games(p)
    tf = team_features(tg)
    qf = qb_features(qb_games(p), sched)
    pts = sched.groupby("season").apply(lambda d: (d["home_score"] + d["away_score"]).mean())
    lg_pts = {s: pts.get(s - 1, np.nan) for s in feat_seasons}
    g = assemble(sched, tf, qf, pd.Series(lg_pts))
    g = g[g["season"] >= first_fit].dropna(subset=BASE_S + BASE_T + ["h_pace", "a_pace", "h_o_press", "a_o_press"])
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


# ------------------------------------------------------------------------------------------- live refit (2020-2025)
# DECISION (lead, 2026-10-25, on the user's "do the refit and get it ready"): the live coefficients are refit on
# 2020-2025. Those seasons were the proposed SEALED set for L33 (scoring T02/T12 once on 2020-25 openers). Fitting
# on them forfeits that sealed test: L33 is now FORWARD-PAPER ONLY. 2020-2025 must never be quoted as an
# out-of-sample betting result for this model. 2026 is used only for a LEVEL check (mean model - close), never for
# betting-performance claims.
LIVE = 2026
MODERN_FIT = list(range(2020, 2026))
HALF_LIFE = 2.0            # recency weight w = 0.5 ** ((2025 - season) / 2): 2025 = 1, 2023 = .5, 2021 = .25, 2020 = .18
ALL_FEAT = list(range(2006, LIVE + 1))   # features need history (QB state, last-season priors); fit seasons are separate
STYLE_S_M = ["proe_diff", "expl_net", "press_net"]   # the 6 matchup interactions are DROPPED (all p > .27 in dev)
STYLE_T_M = ["pace_sum", "proe_sum", "expl_sum"]
MODELS_MODERN = {"A": (BASE_S, BASE_T), "B": (BASE_S + QB_S, BASE_T + QB_T),
                 "C": (BASE_S + QB_S + STYLE_S_M, BASE_T + QB_T + STYLE_T_M)}
LIVE_PICK = {"modern": ("B", "C"), "legacy": ("B", "B")}   # (spread model, total model) used for the JSON output
REFIT_JSON = DATA / "lab" / "nfl_simple_model_refit.json"
REFIT_TAB = ROOT / "notes" / "nfl_simple_model_refit_tables.md"
DEPTH = DATA / "nfl_depth_charts_2026.parquet"
QB_OUT = {"Out", "Injured Reserve", "Doubtful", "Suspension", "Physically Unable to Perform", "Reserve"}
NV = "https://github.com/nflverse/nflverse-data/releases/download"


def recency_w(season):
    return 0.5 ** ((MODERN_FIT[-1] - np.asarray(season, float)) / HALF_LIFE)


def fit_spec(fit: str):
    """(models, train selector, weight fn) for --fit modern|legacy."""
    if fit == "legacy":
        return MODELS, (lambda g: g[(g["season"] >= FIRST_FIT) & (g["season"] <= 2019)]), None
    return MODELS_MODERN, (lambda g: g[g["season"].isin(MODERN_FIT)]), recency_w


def predict_all(models, train, test, wfn):
    out, fits = {}, {}
    w = None if wfn is None else wfn(train["season"])
    for m, (cs, ct) in models.items():
        ps, ms, xs = fit_predict(train, test, cs, "margin", True, w)
        pt, mt, xt = fit_predict(train, test, ct, "points", False, w)
        out[f"spread_{m}"], out[f"total_{m}"] = np.asarray(ps), np.asarray(pt)
        fits[m] = ((ms, xs), (mt, xt))
    return out, fits


def load_closes():
    """Closing lines from data/nfl_games.parquet; where it has none (it stops at 2026 week 4), fall back to the
    nflverse schedule's spread_line/total_line (same source: they agree on all 1,757 overlapping 2020-26 games
    to 0.0006 pts mean abs). For games not yet played the schedule value is the CURRENT line, not a close."""
    c = pd.read_parquet(DATA / "nfl_games.parquet", columns=["game_id", "spread_line", "total_line", "start_utc"])
    s = pd.read_csv(DATA / "nfl_schedule.csv", usecols=["game_id", "spread_line", "total_line"])
    c = c.merge(s, on="game_id", how="outer", suffixes=("", "_nv"))
    c["close_spread"] = c["spread_line"].fillna(c["spread_line_nv"])
    c["close_total"] = c["total_line"].fillna(c["total_line_nv"])
    return c[["game_id", "close_spread", "close_total", "start_utc"]]


def _metrics(d: pd.DataFrame, models) -> dict:
    d = d.dropna(subset=["close_spread", "close_total"])
    r = dict(n=int(len(d)), close_rmse_margin=float(np.sqrt(((d["margin"] - d["close_spread"]) ** 2).mean())),
             close_rmse_points=float(np.sqrt(((d["points"] - d["close_total"]) ** 2).mean())))
    for m in models:
        gs, gt = d[f"spread_{m}"] - d["close_spread"], d[f"total_{m}"] - d["close_total"]
        r[m] = dict(spread_mae_vs_close=float(gs.abs().mean()), spread_bias=float(gs.mean()),
                    total_mae_vs_close=float(gt.abs().mean()), total_bias=float(gt.mean()),
                    spread_rmse_outcome=float(np.sqrt(((d["margin"] - d[f"spread_{m}"]) ** 2).mean())),
                    total_rmse_outcome=float(np.sqrt(((d["points"] - d[f"total_{m}"]) ** 2).mean())),
                    share_spread_gap_ge2=float((gs.abs() >= SPREAD_T).mean()),
                    share_total_gap_ge25=float((gt.abs() >= TOTAL_T).mean()))
    return r


def coef_table(m, xs, fitdf) -> dict:
    sds = add_interactions(fitdf, fitdf).assign(hfa_off=-fitdf["neutral"])[xs].std()
    return dict(n=int(m.nobs), r2=float(m.rsquared), rmse=float(np.sqrt(np.mean(np.asarray(m.resid) ** 2))),
                terms={k: dict(b=float(b), se=float(se), p=float(pv), per_sd=float(b * sds[k]))
                       for k, b, se, pv in zip(["const"] + xs, m.params, m.bse, m.pvalues) if k != "const"}
                | {"const": dict(b=float(m.params[0]))})


def run_refit():
    g, _ = build_all(ALL_FEAT, FIRST_FIT)
    g = g.merge(load_closes(), on="game_id", how="left")
    res = dict(ts=datetime.now(timezone.utc).isoformat(), half_life=HALF_LIFE, fit_seasons=MODERN_FIT,
               decision="2020-2025 used to fit live coefficients -> L33 sealed test FORFEITED; L33 forward-paper only",
               n_close_joined={int(s): int(d["close_spread"].notna().sum()) for s, d in g[g.season >= 2020].groupby("season")})
    # 1) leave-one-season-out on 2020-2025 (modern structure, recency-weighted)
    loso = []
    for s in MODERN_FIT:
        tr = g[g["season"].isin(MODERN_FIT) & (g["season"] != s)]
        te = g[g["season"] == s].copy()
        pr, _ = predict_all(MODELS_MODERN, tr, te, recency_w)
        loso.append(te.assign(**pr))
    L = pd.concat(loso, ignore_index=True)
    res["loso_modern"] = {"pooled": _metrics(L, MODELS_MODERN)} | {int(s): _metrics(d, MODELS_MODERN) for s, d in L.groupby("season")}
    # 2) legacy 2007-2019 fit (unchanged structure incl. interactions) applied to 2020-2026
    models_l, sel_l, _ = fit_spec("legacy")
    te = g[g["season"] >= 2020].copy()
    pr, fits_l = predict_all(models_l, sel_l(g), te, None)
    Lg = te.assign(**pr)
    res["legacy_on_2020_25"] = {"pooled": _metrics(Lg[Lg.season <= 2025], models_l)} | {
        int(s): _metrics(d, models_l) for s, d in Lg[Lg.season <= 2025].groupby("season")}
    res["legacy_coef_check"] = {m: {k: v["b"] for k, v in coef_table(*fits_l[m][0], sel_l(g))["terms"].items()} for m in models_l}
    # 3) final modern fit on all of 2020-2025 -> coefficients + 2026 level check (completed 2026 games only)
    fitdf = g[g["season"].isin(MODERN_FIT)]
    te = g[g["season"] == LIVE].copy()
    pr, fits = predict_all(MODELS_MODERN, fitdf, te, recency_w)
    T = te.assign(**pr)
    res["coefs"] = {}
    for m in MODELS_MODERN:
        (ms, xs), (mt, xt) = fits[m]
        res["coefs"][f"{m}_margin"] = coef_table(ms, xs, fitdf)
        res["coefs"][f"{m}_points"] = coef_table(mt, xt, fitdf)
    lv = {}
    for name, D, mods in (("modern", T, MODELS_MODERN), ("legacy", Lg[Lg.season == LIVE], models_l)):
        lv[name] = {"all": _metrics(D, mods)} | {f"wk{int(w)}": _metrics(d, mods) for w, d in D.groupby("week")}
    res["level_2026"] = lv
    REFIT_JSON.write_text(json.dumps(res, indent=1, default=float))
    write_refit_tables(res)
    print(REFIT_TAB.read_text())


def write_refit_tables(res):
    f2 = lambda x: f"{x:+.2f}"
    L = ["# nfl_simple_model REFIT tables (generated by `scripts/nfl_simple_model.py refit`)", "",
         f"Live coefficients fit on {res['fit_seasons'][0]}-{res['fit_seasons'][-1]} with recency weight half-life "
         f"{res['half_life']} seasons. {res['decision']}.", "",
         "MAE/bias are model minus nflverse CLOSING line (data/nfl_games.parquet). RMSE is vs the actual outcome. "
         "Accuracy only: no betting result is scored here.", "",
         f"Closing lines joined per season: {res['n_close_joined']}", "",
         "## Leave-one-season-out (modern fit) vs the legacy 2007-19 fit, on 2020-2025", "",
         "| fit | season | n | model | spread MAE vs close | spread bias | total MAE vs close | total bias | "
         "RMSE margin (close) | RMSE points (close) | share |gap|>=2 spr | share |gap|>=2.5 tot |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for fit, key in (("modern LOSO", "loso_modern"), ("legacy 07-19", "legacy_on_2020_25")):
        for s, r in res[key].items():
            for m in ("A", "B", "C"):
                x = r[m]
                L.append(f"| {fit} | {s} | {r['n']} | {m} | {x['spread_mae_vs_close']:.2f} | {f2(x['spread_bias'])} | "
                         f"{x['total_mae_vs_close']:.2f} | {f2(x['total_bias'])} | {x['spread_rmse_outcome']:.2f} "
                         f"({r['close_rmse_margin']:.2f}) | {x['total_rmse_outcome']:.2f} ({r['close_rmse_points']:.2f}) | "
                         f"{x['share_spread_gap_ge2']:.2f} | {x['share_total_gap_ge25']:.2f} |")
    L += ["", "## 2026 LEVEL check (completed 2026 games, walk-forward features, actual starters; NOT a betting score)", "",
          "| fit | slice | n | model | spread bias | spread MAE | total bias | total MAE | share |gap|>=2 spr | share |gap|>=2.5 tot |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for fit, d in res["level_2026"].items():
        for sl, r in d.items():
            for m in ("A", "B", "C"):
                x = r[m]
                L.append(f"| {fit} | {sl} | {r['n']} | {m} | {f2(x['spread_bias'])} | {x['spread_mae_vs_close']:.2f} | "
                         f"{f2(x['total_bias'])} | {x['total_mae_vs_close']:.2f} | {x['share_spread_gap_ge2']:.2f} | "
                         f"{x['share_total_gap_ge25']:.2f} |")
    L += ["", f"## Coefficients, live fit {res['fit_seasons'][0]}-{res['fit_seasons'][-1]} (WLS, recency-weighted; "
          "per_sd = coef x unweighted feature SD, in points)", ""]
    for k, c in res["coefs"].items():
        L += [f"### {k}  (n {c['n']}, weighted R2 {c['r2']:.3f}, unweighted RMSE {c['rmse']:.2f})", "", "| term | coef | se | p | pts per SD |",
              "|---|---|---|---|---|"]
        for t, v in c["terms"].items():
            L.append(f"| const | {v['b']:+.3f} | | | |" if t == "const" else
                     f"| {t} | {v['b']:+.3f} | {v['se']:.3f} | {v['p']:.3g} | {v['per_sd']:+.2f} |")
        L.append("")
    REFIT_TAB.write_text("\n".join(L))


# ------------------------------------------------------------------------------------------- live week
def _download(url: str, dst: Path):
    import requests
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    dst.write_bytes(r.content)


def _espn_injuries(warn: list) -> dict:
    """ESPN league injury report -> {espn_athlete_id: status}. Empty dict (and a warning) on failure."""
    import re
    import requests
    try:
        j = requests.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries", timeout=60).json()
    except Exception as e:      # noqa: BLE001
        warn.append(f"ESPN injuries unavailable ({type(e).__name__}); QB statuses NOT checked")
        return {}
    out = {}
    for t in j.get("injuries", []):
        for i in t.get("injuries", []):
            for ln in i.get("athlete", {}).get("links", []):
                m = re.search(r"/id/(\d+)", ln.get("href", ""))
                if m:
                    out[m.group(1)] = (i.get("status"), pd.to_datetime(i.get("date"), utc=True))
                    break
    return out


def live_starters(wk: pd.DataFrame, qb_known: set) -> dict:
    """Projected starting QB per (game_id, side): nflverse depth chart (ESPN-sourced, latest snapshot before
    kickoff), QB1 unless ESPN lists him Out/IR/Doubtful/Suspended -> next QB on the chart. Only ESPN entries
    dated BEFORE kickoff are applied (so a finished game is rated as it looked pre-game). Cross-checked against the
    nflverse schedule's QB (projected pre-game, actual starter once played; it can be stale, e.g. PHI 2026 wk7)."""
    gw: list = []
    inj = _espn_injuries(gw)
    dc = pd.read_parquet(DEPTH) if DEPTH.exists() else pd.DataFrame()
    if len(dc):
        dc = dc[dc["pos_abb"] == "QB"].copy()
        dc["team"] = dc["team"].replace(TEAM_MAP)
        dc["dt"] = pd.to_datetime(dc["dt"], utc=True)
    else:
        gw.append("no depth chart file; using nflverse schedule starters")
    now = pd.Timestamp.now(tz="UTC")
    out = {}
    for _, r in wk.iterrows():
        ko = r["kickoff_utc"]
        for side in ("home", "away"):
            t, w = r[f"{side}_team"], []
            sq, sn = r[f"{side}_qb_id"], r[f"{side}_qb_name"]
            chosen = (sq if isinstance(sq, str) else None, sn)
            if len(dc):
                d = dc[(dc["team"] == t) & (dc["dt"] <= ko)]
                if len(d):
                    d = d[d["dt"] == d["dt"].max()].sort_values("pos_rank").drop_duplicates("gsis_id")
                    cands = list(zip(d["gsis_id"], d["player_name"], d["espn_id"].astype(str)))
                    pick = cands[0]
                    for c in cands:
                        st, sdt = inj.get(c[2], (None, None))
                        if sdt is not None and sdt > ko:
                            # ESPN keeps only the latest entry; one written after kickoff (post-game notes,
                            # in-game injuries) says nothing about pre-game availability -> ignore it.
                            if c == cands[0] and ko <= now:
                                w.append(f"{t}: ESPN entry for {c[1]} updated after kickoff; pre-game status unknown")
                            st = None
                        if st in QB_OUT:
                            w.append(f"{t} QB {c[1]} listed {st} (ESPN, {sdt:%m-%d %H:%MZ})")
                            continue
                        if st == "Questionable":
                            w.append(f"{t} QB {c[1]} Questionable (ESPN)")
                        pick = c
                        break
                    if pick != cands[0]:
                        w.append(f"{t}: backup {pick[1]} rated instead of depth-chart QB1 {cands[0][1]}")
                    chosen = (pick[0], pick[1])
                    if isinstance(sq, str) and sq != pick[0]:
                        w.append(f"{t}: depth chart QB {pick[1]} != nflverse schedule QB {sn}")
                else:
                    w.append(f"{t}: no depth-chart snapshot before kickoff; nflverse schedule QB used")
            if chosen[0] is None:
                w.append(f"{t}: no starter identified -> QB term 0")
            elif chosen[0] not in qb_known:
                w.append(f"{t}: {chosen[1]} has no dropbacks in pbp -> replacement-level prior")
            out[(r["game_id"], side)] = (chosen[0], chosen[1], w)
    return out, gw


def run_week(refresh: bool, week: int | None, fit: str = "modern", as_json: bool = False):
    if refresh:
        _download(f"{NV}/pbp/play_by_play_{LIVE}.parquet", DATA / "pbp" / f"play_by_play_{LIVE}.parquet")
        _download("https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv", DATA / "nfl_schedule.csv")
        _download(f"{NV}/depth_charts/depth_charts_{LIVE}.parquet", DEPTH)
    models, sel, wfn = fit_spec(fit)
    p_all = load_pbp(ALL_FEAT)
    sched = load_schedule()
    s26 = sched[sched["season"] == LIVE]
    if week is None:
        week = int(s26[s26["home_score"].isna()]["week"].min())
    # coefficients: legacy = 2007-2019 (original dev fit); modern = 2020-2025 recency-weighted
    g, _ = build_all(ALL_FEAT, FIRST_FIT)
    train = sel(g)
    # live features: only plays from games BEFORE the target week (pbp may already contain this week's games)
    p = p_all[(p_all["season"] < LIVE) | (p_all["week"] < week)]
    tg = team_games(p)
    played = tg[tg["season"] == LIVE]["game_id"].unique()
    tf = team_features_live(tg, LIVE)
    wk = s26[s26["week"] == week].copy()
    ko = load_closes()[["game_id", "start_utc"]]
    wk = wk.merge(ko, on="game_id", how="left")
    et = pd.to_datetime(wk["gameday"].dt.strftime("%Y-%m-%d") + " " + wk["gametime"].astype(str)).dt.tz_localize(
        "America/New_York").dt.tz_convert("UTC")
    wk["kickoff_utc"] = pd.to_datetime(wk["start_utc"], utc=True).fillna(et)
    done = sched[sched["home_score"].notna() & sched["game_id"].isin(set(tg["game_id"]))]
    qg = qb_games(p)
    _, rate_fn = qb_features(qg, done, live_season=LIVE)
    starters, gwarn = live_starters(wk, set(qg["id"]))
    rows = []
    for _, r in wk.iterrows():
        rec = r.to_dict()
        for side in ("home", "away"):
            t = r[f"{side}_team"]
            f = tf[tf["team"] == t]
            for c in f.columns:
                if c not in ("game_id", "team", "season", "week"):
                    rec[f"{side[0]}_{c}"] = float(f[c].iloc[0]) if len(f) else np.nan
            q, qn, w = starters[(r["game_id"], side)]
            rq, b, last = rate_fn(q, t)
            rec[f"{side[0]}_qb_adj"] = (rq - b) if q is not None else 0.0
            rec[f"{side[0]}_qbchg"] = int(q is not None and last is not None and q != last)
            rec[f"{side[0]}_qb"] = qn
            rec[f"{side[0]}_qb_rating"] = rq
            rec[f"{side[0]}_warn"] = w
        rows.append(rec)
    W = pd.DataFrame(rows)
    pts = sched[sched["season"] == LIVE - 1]
    W = _derive(W, float((pts["home_score"] + pts["away_score"]).mean()))
    pr, _ = predict_all(models, train, W, wfn)
    ps_m, pt_m = LIVE_PICK[fit]
    out = W[["game_id", "kickoff_utc", "away_team", "home_team", "a_qb", "h_qb", "qbchg", "spread_line", "total_line"]].copy()
    for k, v in pr.items():
        out[k] = np.round(v, 1)
    out["spr_gap"] = (out[f"spread_{ps_m}"] - out["spread_line"]).round(1)
    out["tot_gap"] = (out[f"total_{pt_m}"] - out["total_line"]).round(1)
    if as_json:
        js = []
        for (_, o), (_, w) in zip(out.iterrows(), W.iterrows()):
            js.append(dict(game_id=o["game_id"], kickoff_utc=o["kickoff_utc"].strftime("%Y-%m-%dT%H:%MZ"),
                           home=o["home_team"], away=o["away_team"],
                           model_spread_home=float(o[f"spread_{ps_m}"]), model_total=float(o[f"total_{pt_m}"]),
                           qb_home=o["h_qb"], qb_away=o["a_qb"], warnings=list(w["h_warn"]) + list(w["a_warn"]) + gwarn,
                           fit=fit, spread_model=ps_m, total_model=pt_m,
                           market_spread_line=None if pd.isna(o["spread_line"]) else float(o["spread_line"]),
                           market_total_line=None if pd.isna(o["total_line"]) else float(o["total_line"])))
        print(json.dumps(js, indent=1))
        return
    pd.set_option("display.width", 250)
    fit_txt = "2007-2019 (legacy dev fit)" if fit == "legacy" else f"2020-2025, recency half-life {HALF_LIFE} seasons"
    print(f"NFL {LIVE} week {week}: model home margin (spread_*, + = home favoured, nflverse sign) and total (total_*). "
          f"Features from {len(played)} {LIVE} games before week {week}; coefficients fit on {fit_txt}.")
    print(f"Live picks: spread model {ps_m}, total model {pt_m}. LIVE PREDICTION ONLY -- not a scored result.")
    print(out.drop(columns=["game_id"]).assign(kickoff_utc=out["kickoff_utc"].dt.strftime("%m-%d %H:%M")).to_string(index=False))
    print(f"\nmean gap vs current line: spread {out['spr_gap'].mean():+.2f}, total {out['tot_gap'].mean():+.2f}")
    print(f"|spread gap| >= {SPREAD_T}: {int((out['spr_gap'].abs() >= SPREAD_T).sum())}/{len(out)};  "
          f"|total gap| >= {TOTAL_T}: {int((out['tot_gap'].abs() >= TOTAL_T).sum())}/{len(out)}  "
          f"(paper only; the forward rule is defined vs the OPENER)")
    for _, w in W.iterrows():
        for x in list(w["h_warn"]) + list(w["a_warn"]):
            print("  warn:", x)
    for x in gwarn:
        print("  warn:", x)


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
    elif cmd == "refit":
        run_refit()
    elif cmd == "week":
        wk = int(sys.argv[sys.argv.index("--week") + 1]) if "--week" in sys.argv else None
        fit = sys.argv[sys.argv.index("--fit") + 1] if "--fit" in sys.argv else "modern"
        assert fit in ("modern", "legacy"), fit
        run_week("--refresh" in sys.argv, wk, fit, "--json" in sys.argv)
    else:
        print(__doc__)
