"""NBA SIMPLE model: team rating + player availability + playing style. Every coefficient printable. No xgboost.

    ../.venv_quant/bin/python scripts/nba_simple_model.py build              # features + model accuracy only (no market)
    ../.venv_quant/bin/python scripts/nba_simple_model.py backtest [--ledger]  # pre-registered market trials (dev)
    ../.venv_quant/bin/python scripts/nba_simple_model.py today [--date YYYY-MM-DD] [--json] [--out "Name,Name"]
                                                       [--no-refresh] [--no-cache]   (-> scripts/nba_simple_live.py)

User 2026-10-25: "Kalshi follows Pinnacle. Get the edge via a SIMPLE model that takes into account the betting rates and
stats on each team, runs simple analysis to determine playing style; simpler might be better; look at the players and
their performances. Top teams use xgboost which might not be that useful."

=====================================================================================================================
PRE-REGISTERED 2026-10-25, written BEFORE any model output was joined to any market price. Nothing below is tuned:
every hyperparameter is fixed a priori (team-rating ones are O1's frozen values from notes/player_sim.md).
=====================================================================================================================
DATA (backtest = DEV only): ESPN data/espn/nba/{games,box,lines}_{2022..2025} = seasons 2021-22..2024-25, plus
  data/nba_comebacks/teamlogs.parquet (stats.nba.com, for paint points only; its labels 2021..2024 = same seasons).
  NBA 2026 (= 2025-26) is SEALED and data/nba_ingame.parquet is never read by build/backtest (asserted).
  `today` may read 2026 + 2027 (= 2026-27) for LIVE PREDICTION ONLY; it never scores anything.

MODEL (all state is updated in ET-date batches: every game on date D uses only games dated < D)
 M0 TEAM RATING (= O1's ridge): margin_home = h + r_home - r_away + b2b/rest/travel terms, weighted ridge, half-life
    60 d, lambda 5. Totals analogue: team points = mu + off[team] + def_allowed[opp] + h*home, same weights, lambda 5;
    total0 = pts_home + pts_away.
 PLAYER RATING ("BPM-lite"): for season S, OLS (weights = minutes) over player-seasons of seasons < S with >= 300 min
    (season 2022 features use 2022 itself -- in-sample, they only ever train the 2023 stack, never get scored):
    on-court plus-minus per 48 ~ per-48 [pts, fga, fta, 3pm, orb, drb, ast, tov, stl, blk, pf].
    A player's rates are decayed sums over all his past games (half-life 180 d -> last season is the prior), shrunk to
    REPLACEMENT rates (minutes-weighted mean of 100-500-minute player-seasons) with 500 pseudo-minutes.
    rating r_i = beta . rates_i (+ intercept). Points per minute shrunk the same way.
 TEAM STRENGTH tonight = sum_i m_i * r_i / 240 over players EXPECTED to play; m_i = EWMA(0.3) of his past minutes,
    rescaled to 240 with a 42-minute cap. O = sum_i m_i * ppm_i = expected points from those players.
    FULL roster = players whose latest team is T and who appeared in any of T's last 10 games.
    Availability variants:  STRICT = players who played T's previous game (pure no-lookahead);
                            ACTUAL = players who actually played tonight (proxy for the injury report at tip:
                                     LOOKAHEAD -- late scratches/minutes restrictions are known only near tip).
    Features: dPfull = Pfull_h - Pfull_a; dAbs = (Psel_h - Pfull_h) - (Psel_a - Pfull_a); for totals
    Ofull = Ofull_h + Ofull_a and oAbs = (Osel_h - Ofull_h) + (Osel_a - Ofull_a).
 STYLE (team, decayed half-life 60 d over its past games): pace, 3PA rate (own/allowed), FT rate (own/allowed),
    ORB%/DRB%, TOV% (own/forced), 3P% (own/allowed), paint-points share (own/allowed, stats.nba.com).
    Spread matchups (pre-chosen, 5):  S1 paint_def = h.opp_paint_share - a.opp_paint_share   (L30 survivor)
      S2 three = z(h.3PAr) z(a.opp3PAr) - z(a.3PAr) z(h.opp3PAr)      S3 reb = z(a.ORB) z(h.DRB) - z(h.ORB) z(a.DRB)
      S4 tov = z(h.forcedTOV) z(a.TOV) - z(a.forcedTOV) z(h.TOV)      S5 ft = z(h.FTr) z(a.oppFTr) - z(a.FTr) z(h.oppFTr)
    Totals (5): U1 pace sum, U2 sum 3P%, U3 sum opp 3P% (L31), U4 sum 3PA rate, U5 b2b count.
    Rest/travel are already inside M0. Every final feature is standardised on the training fold (pts per SD).
 NESTED MODELS (OLS stacks, walk-forward by season: season S is fit on scored games of seasons < S, 2022 from
    2021-12-01): M0 = mu0 as is; M1 = a + b*mu0 + c*dPfull + d*dAbs; M2 = M1 + S1..S5. Totals T0/T1/T2 likewise with
    total0, Ofull, oAbs, U1..U5. Win prob = Phi(mu / residual SD of the training fit).
 SCORED games: regular season, non-neutral, both teams >= 5 earlier games, seasons 2023-2025.

MARKETS. OPENER = DraftKings open where present, else ESPN BET open (same rule as O1's amendment; DK only exists in
  2024, so 2025 = ESPN BET). CLV reference = the SAME book's close. CLOSE = median over real books of
  home_spread_close / total_close (2023: ESPN BET only). No-vig = proportional de-vig of that book's two-way spread or
  total prices (approximates resting a maker order on Kalshi at fair, ~0 maker fee). No Pinnacle in this data.
TRIALS (dataset nba_simple_model; 8 trials -> bar one-sided p < 0.05/8 = 0.00625; flat 1u, one bet per game;
  model M2/T2; spread side when |model margin - market implied margin| >= 2.0 pts; total side when |gap| >= 3.0 pts;
  one-sided game bootstrap, 4000 resamples; pass = p < bar and effect > 0):
  T1 spr_strict_open_clv     STRICT vs opener, mean CLV in points (bet-side line move open -> same-book close)
  T2 spr_strict_open_novig   same bets, ATS at the opening line paid at the no-vig opening price
  T3 spr_strict_open_110     same bets, ATS at the opening line at -110
  T4 tot_strict_open_clv     STRICT totals vs opener, mean CLV in points
  T5 tot_strict_open_novig   same bets, O/U at the opening total, no-vig price
  T6 spr_actual_close_novig  ACTUAL vs median close spread, ATS no-vig (injury-report-at-tip proxy vs the close)
  T7 tot_actual_close_novig  ACTUAL vs median close total, O/U no-vig
  T8 spr_actual_open_clv_news ACTUAL vs opener, CLV, ONLY games with player-absence news: a top-5-projected-minutes
                              player of either team who played its previous game does not play tonight. (Upper bound:
                              the news may have broken after the open.)
DESCRIPTIVE (not trials): out-of-sample margin/total MAE for M0/M1/M2 x STRICT/ACTUAL vs open/close lines; stacking
  coefficient margin ~ market line + model (given open, given close); corr(model - open, close - open); CLV/ATS split
  by favourite/dog, by news/no-news, by season (both-halves sign check); full coefficient tables.

AMENDMENT 2026-10-25, after `build` (model accuracy only) and BEFORE any market join: season 2022 is a cold start
  (no prior-season player history: every player begins at replacement), so Ofull/dPfull have a different level there
  and the 2023 totals stack trained on 2022 came out +4 pts biased (T1 MAE 15.2 vs T0 14.65). Fix: 2022 is WARM-UP
  ONLY. Stacks are fit on scored games of 2023 onward; walk-forward folds = 2024 (fit 2023) and 2025 (fit 2023-24).
  All trials are therefore scored on 2024-2025 (open trials were 2024-25 anyway; close trials T6/T7 lose 2023).
  Nothing else changed (no hyperparameter, feature, threshold or trial changed).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from nba_player_sim import CITY, LOC, TI, km, ridge_preds
from sports.market import devig_two_way

ESPN = ROOT / "data" / "espn" / "nba"
LEDGER = ROOT / "lab" / "ledger.jsonl"
FROZEN = ROOT / "results" / "nba_simple_model_frozen.json"
OUTD = ROOT / "data" / "nba_simple_model"
DATASET = "nba_simple_model"
DEV = [2022, 2023, 2024, 2025]
assert 2026 not in DEV and 2027 not in DEV
NOT_BOOKS = {"consensus", "accuscore", "teamrankings", "betegy"}
H_TEAM, LAM, HP, KSHRINK, HS = 60, 5.0, 180.0, 500.0, 60.0
SPR_X, TOT_X, N_TRIALS = 2.0, 3.0, 8
TUNE_FROM = pd.Timestamp("2021-12-01").date()
STATS = ["points", "fieldGoalsAttempted", "freeThrowsAttempted", "threePointFieldGoalsMade", "offensiveRebounds",
         "defensiveRebounds", "assists", "turnovers", "steals", "blocks", "fouls"]
SHORT = ["pts", "fga", "fta", "3pm", "orb", "drb", "ast", "tov", "stl", "blk", "pf"]
ESPN2NBA = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}
SPR_STYLE = ["S1_paint_def", "S2_three", "S3_reb", "S4_tov", "S5_ft"]
TOT_STYLE = ["U1_pace", "U2_fg3pct", "U3_oppfg3pct", "U4_fg3arate", "U5_b2b"]
FEAT = {"M1": ["mu0", "dPfull", "dAbs"], "T1": ["tot0", "Ofull", "oAbs"]}
FEAT["M2"], FEAT["T2"] = FEAT["M1"] + SPR_STYLE, FEAT["T1"] + TOT_STYLE


# ------------------------------------------------------------------------------------------------------------- data
def load_games(seasons, live_date=None) -> pd.DataFrame:
    g = pd.concat([pd.read_parquet(ESPN / f"games_{y}.parquet").assign(season=y) for y in seasons], ignore_index=True)
    g["date"] = g.start_utc.dt.tz_convert("US/Eastern").dt.date
    keep = (g.status == "STATUS_FINAL")
    if live_date is not None:
        keep = (keep & (g.date < live_date)) | (g.date == live_date)
    g = g[keep & (g.season_type != "preseason") & g.home.isin(LOC) & g.away.isin(LOC)].copy()
    g = g.sort_values("start_utc").reset_index(drop=True)
    g["margin"] = g.home_score - g.away_score
    g["total"] = g.home_score + g.away_score
    g["loc"] = [CITY.get(c, LOC[h]) if n else LOC[h] for c, h, n in zip(g.city, g.home, g.neutral.astype(bool))]
    last, cnt = {}, {}
    cols = {f"{a}_{s}": [] for a in ("rest", "trav", "n") for s in ("home", "away")}
    for r in g.itertuples():
        for side in ("home", "away"):
            t = getattr(r, side)
            if t in last and last[t][2] == r.season:
                d0, l0, _ = last[t]
                cols[f"rest_{side}"].append(min((r.date - d0).days, 4))
                cols[f"trav_{side}"].append(km(l0, r.loc) / 1000)
            else:
                cols[f"rest_{side}"].append(4)
                cols[f"trav_{side}"].append(0.0)
            last[t] = (r.date, r.loc, r.season)
            cols[f"n_{side}"].append(cnt.get((t, r.season), 0))
            cnt[(t, r.season)] = cnt.get((t, r.season), 0) + 1
    for k, v in cols.items():
        g[k] = v
    g["b2b_home"], g["b2b_away"] = (g.rest_home == 1).astype(float), (g.rest_away == 1).astype(float)
    g["restdiff"] = g.rest_home.clip(upper=3) - g.rest_away.clip(upper=3)
    g["home_ind"] = (~g.neutral.astype(bool)).astype(float)
    g["scored"] = (g.season_type == "regular-season") & (g.home_ind == 1) & (g.n_home >= 5) & (g.n_away >= 5)
    return g


def load_box(g: pd.DataFrame, seasons) -> pd.DataFrame:
    b = pd.concat([pd.read_parquet(ESPN / f"box_{y}.parquet", columns=["event_id", "team_id", "athlete_id", "name",
                                                                       "stat", "value"]) for y in seasons])
    b = b[b.event_id.isin(set(g.event_id))]
    nm = b.drop_duplicates("athlete_id").set_index("athlete_id").name
    w = b.pivot_table(index=["event_id", "team_id", "athlete_id"], columns="stat", values="value",
                      aggfunc="first").reset_index()
    w = w.dropna(subset=["minutes"])
    w = w[w.minutes > 0].fillna(0)
    ids = pd.concat([g[["event_id", "home_team_id", "home"]].rename(columns={"home_team_id": "team_id", "home": "team"}),
                     g[["event_id", "away_team_id", "away"]].rename(columns={"away_team_id": "team_id", "away": "team"})])
    w = w.merge(ids, on=["event_id", "team_id"], how="inner")
    w["name"] = w.athlete_id.map(nm)
    return w


def load_paint(g: pd.DataFrame, live=False) -> pd.DataFrame:
    """(event_id, team) -> paint pts scored. Dev: stats.nba.com teamlogs on disk. Live: nba_api pull (cached)."""
    if not live:
        t = pd.read_parquet(ROOT / "data" / "nba_comebacks" / "teamlogs.parquet",
                            columns=["GAME_DATE", "TEAM_ABBREVIATION", "PTS_PAINT", "season"])
        t = t[t.season.between(2021, 2024)]
    else:
        frames = []
        for s in sorted(g.season.unique()):
            lab = f"{s - 1}-{str(s)[2:]}"
            p = OUTD / f"paint_{lab}.parquet"
            try:
                from nba_api.stats.endpoints import teamgamelogs
                d = teamgamelogs.TeamGameLogs(season_nullable=lab, measure_type_player_game_logs_nullable="Misc",
                                              timeout=15).get_data_frames()[0]
                OUTD.mkdir(parents=True, exist_ok=True)
                d[["GAME_DATE", "TEAM_ABBREVIATION", "PTS_PAINT"]].to_parquet(p)
            except Exception as ex:  # stats.nba.com throttles / blocks: fall back to cache, else no paint term
                print(f"  (paint pull {lab} failed: {str(ex)[:60]}; using cache if any)")
            if p.exists():
                frames.append(pd.read_parquet(p))
        if not frames:
            return pd.DataFrame(columns=["event_id", "team", "paint"])
        t = pd.concat(frames)
    t = t.assign(date=pd.to_datetime(t.GAME_DATE.astype(str).str[:10]).dt.date, abbr=t.TEAM_ABBREVIATION)
    rows = []
    for side in ("home", "away"):
        x = g[["event_id", "date", side]].rename(columns={side: "team"})
        x["abbr"] = x.team.map(lambda a: ESPN2NBA.get(a, a))
        rows.append(x.merge(t[["date", "abbr", "PTS_PAINT"]], on=["date", "abbr"], how="inner"))
    return pd.concat(rows).rename(columns={"PTS_PAINT": "paint"})[["event_id", "team", "paint"]]


# ----------------------------------------------------------------------------------------------------- player model
def player_seasons(box: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    s = box.merge(g[["event_id", "season"]], on="event_id")
    return s.groupby(["season", "athlete_id"])[STATS + ["minutes", "plusMinus"]].sum().reset_index()


def fit_bpm(ps: pd.DataFrame) -> dict:
    """plus-minus per 48 ~ per-48 box rates, minutes-weighted OLS over player-seasons with >= 300 min."""
    x = ps[ps.minutes >= 300]
    R = x[STATS].to_numpy() / x.minutes.to_numpy()[:, None] * 48
    y = x.plusMinus.to_numpy() / x.minutes.to_numpy() * 48
    f = sm.WLS(y, sm.add_constant(R), weights=x.minutes.to_numpy()).fit()
    lo = ps[(ps.minutes >= 100) & (ps.minutes <= 500)]
    repl = (lo[STATS].sum() / lo.minutes.sum()).to_numpy() * 48
    repl_ppm = float(lo.points.sum() / lo.minutes.sum())
    return {"b0": float(f.params[0]), "beta": f.params[1:].tolist(), "se": f.bse[1:].tolist(), "repl48": repl.tolist(),
            "repl_ppm": repl_ppm, "n": int(len(x)), "r2": float(f.rsquared)}


def betas_by_season(box, g) -> dict:
    ps = player_seasons(box, g)
    out = {}
    for s in sorted(g.season.unique()):
        tr = ps[ps.season < s]
        out[int(s)] = fit_bpm(tr if len(tr) else ps[ps.season == s])
    return out


def minutes_proj(em: np.ndarray) -> np.ndarray:
    if len(em) == 0:
        return em
    m = em * 240 / em.sum()
    for _ in range(3):
        over = m > 42
        if not over.any():
            break
        m[over] = 42
        rest = ~over
        m[rest] *= (240 - 42 * over.sum()) / max(m[rest].sum(), 1e-9)
    return m


def player_pass(g: pd.DataFrame, box: pd.DataFrame, betas: dict, out_names=None) -> pd.DataFrame:
    """Walk dates; per game per team: Pfull/Pstrict/Pact, Ofull/Ostrict/Oact, news flag, starter lists."""
    out_names = out_names or set()
    bx = {}
    for r in box.itertuples(index=False):
        bx.setdefault((r.event_id, r.team), []).append(r)
    ST = {}                       # athlete -> dict(s: decayed stat vec per-minute sums, m: decayed min, t, em, team, last)
    TH = {}                       # team -> list of player-sets of its past games (most recent last)
    names = dict(zip(box.athlete_id, box.name))
    dn = np.array([d.toordinal() for d in g.date])
    feats = []
    for d in np.unique(dn):
        idx = np.where(dn == d)[0]
        for i in idx:
            r = g.iloc[i]
            bt = betas[int(r.season)]
            beta, repl, rppm = np.array(bt["beta"]), np.array(bt["repl48"]) / 48, bt["repl_ppm"]
            row = {"event_id": r.event_id}
            for side in ("home", "away"):
                T = r[side]
                hist = TH.get(T, [])
                full = set().union(*hist[-10:]) if hist else set()
                full = {a for a in full if ST[a]["team"] == T}
                prev = hist[-1] if hist else set()
                act = {x.athlete_id for x in bx.get((r.event_id, T), [])}

                def rate(a):
                    s = ST.get(a)
                    if s is None:
                        return bt["b0"] + repl @ beta * 48, rppm, 15.0
                    dec = 0.5 ** ((d - s["t"]) / HP)
                    mm = s["m"] * dec
                    rt = (s["s"] * dec + KSHRINK * repl) / (mm + KSHRINK)
                    ppm = (s["s"][0] * dec + KSHRINK * rppm) / (mm + KSHRINK)
                    return bt["b0"] + rt @ beta * 48, ppm, s["em"]

                def strength(S):
                    S = [a for a in S if names.get(a) not in out_names]
                    if not S:
                        return 0.0, 100.0
                    v = np.array([rate(a) for a in S])
                    m = minutes_proj(v[:, 2].copy())
                    return float((m * v[:, 0]).sum() / 240), float((m * v[:, 1]).sum())

                starters = sorted(full, key=lambda a: -ST[a]["em"])[:5]
                for tag, S in (("full", full), ("strict", prev), ("act", act)):
                    row[f"P{tag}_{side}"], row[f"O{tag}_{side}"] = strength(S) if S else (np.nan, np.nan)
                row[f"news_{side}"] = [names.get(a, a) for a in starters if a in prev and a not in act]
                row[f"miss_strict_{side}"] = [names.get(a, a) for a in starters if a not in prev]
            feats.append(row)
        for i in idx:                                       # update after the whole date
            r = g.iloc[i]
            for side in ("home", "away"):
                T = r[side]
                ps = bx.get((r.event_id, T), [])
                for x in ps:
                    v = np.array([getattr(x, c) for c in STATS], float)
                    s = ST.get(x.athlete_id)
                    if s is None:
                        s = {"s": np.zeros(len(STATS)), "m": 0.0, "t": d, "em": x.minutes}
                    dec = 0.5 ** ((d - s["t"]) / HP)
                    s["s"], s["m"], s["t"] = s["s"] * dec + v, s["m"] * dec + x.minutes, d
                    s["em"] = 0.7 * s["em"] + 0.3 * x.minutes
                    s["team"] = T
                    ST[x.athlete_id] = s
                if ps:
                    TH.setdefault(T, []).append({x.athlete_id for x in ps})
    return pd.DataFrame(feats)


# ------------------------------------------------------------------------------------------------------ team models
def ridge_totals(g: pd.DataFrame, H=H_TEAM, lam=LAM) -> np.ndarray:
    n = len(g)
    hi, ai = g.home.map(TI).to_numpy(), g.away.map(TI).to_numpy()
    X = np.zeros((2 * n, 62))
    X[:, 0] = 1
    r = np.arange(n)
    X[r, 1 + hi] = 1; X[r, 31 + ai] = 1; X[r, 61] = g.home_ind          # home scoring: off[home] + def_allowed[away]
    X[n + r, 1 + ai] = 1; X[n + r, 31 + hi] = 1                           # away scoring
    y = np.r_[g.home_score.to_numpy(float), g.away_score.to_numpy(float)]
    dn = np.array([d.toordinal() for d in g.date])
    dn2 = np.r_[dn, dn]
    pen = np.r_[1e-6, np.full(60, lam), 1e-6]
    out = np.full(n, np.nan)
    for d in np.unique(dn):
        tr = (dn2 < d) & ~np.isnan(y)
        if tr.sum() < 100:
            continue
        w = 0.5 ** ((d - dn2[tr]) / H)
        Xt = X[tr]
        beta = np.linalg.solve((Xt * w[:, None]).T @ Xt + np.diag(pen), (Xt * w[:, None]).T @ y[tr])
        te = np.where(dn == d)[0]
        out[te] = X[te] @ beta + X[n + te] @ beta
    return out


def style_pass(g: pd.DataFrame, box: pd.DataFrame, paint: pd.DataFrame) -> pd.DataFrame:
    """Decayed (half-life HS) team style stats from each team's PAST games."""
    tg = box.groupby(["event_id", "team"])[["fieldGoalsAttempted", "threePointFieldGoalsAttempted",
                                            "threePointFieldGoalsMade", "freeThrowsAttempted", "offensiveRebounds",
                                            "defensiveRebounds", "turnovers", "points"]].sum()
    tg.columns = ["fga", "fg3a", "fg3m", "fta", "orb", "drb", "tov", "pts"]
    tg = tg.reset_index().merge(paint, on=["event_id", "team"], how="left")
    tg["poss"] = tg.fga - tg.orb + tg.tov + 0.44 * tg.fta
    TG = {(r.event_id, r.team): r for r in tg.itertuples(index=False)}
    K = ["n", "poss", "fga", "fg3a", "fg3m", "fta", "orb", "drb", "tov", "pts", "paint", "ppts",
         "o_fga", "o_fg3a", "o_fg3m", "o_fta", "o_orb", "o_drb", "o_tov", "o_pts", "o_paint", "o_ppts"]
    S = {}
    dn = np.array([d.toordinal() for d in g.date])
    rows = []

    def feats(T, d):
        if T not in S:
            return {}
        v, t0 = S[T]
        v = dict(zip(K, np.array(v) * 0.5 ** ((d - t0) / HS)))
        f = lambda a, b: a / b if b > 0 else np.nan
        return {"pace": f(v["poss"], v["n"]), "fg3a_rate": f(v["fg3a"], v["fga"]), "opp_fg3a_rate": f(v["o_fg3a"], v["o_fga"]),
                "ft_rate": f(v["fta"], v["fga"]), "opp_ft_rate": f(v["o_fta"], v["o_fga"]),
                "orb_pct": f(v["orb"], v["orb"] + v["o_drb"]), "drb_pct": f(v["drb"], v["drb"] + v["o_orb"]),
                "tov_pct": f(v["tov"], v["poss"]), "opp_tov_pct": f(v["o_tov"], v["poss"]),
                "fg3_pct": f(v["fg3m"], v["fg3a"]), "opp_fg3_pct": f(v["o_fg3m"], v["o_fg3a"]),
                "paint_share": f(v["paint"], v["ppts"]), "opp_paint_share": f(v["o_paint"], v["o_ppts"])}

    for d in np.unique(dn):
        idx = np.where(dn == d)[0]
        for i in idx:
            r = g.iloc[i]
            row = {"event_id": r.event_id}
            for side in ("home", "away"):
                for k, v in feats(r[side], d).items():
                    row[f"{side}_{k}"] = v
            rows.append(row)
        for i in idx:
            r = g.iloc[i]
            a, b = TG.get((r.event_id, r.home)), TG.get((r.event_id, r.away))
            if a is None or b is None:
                continue
            poss = (a.poss + b.poss) / 2
            for me, op in ((a, b), (b, a)):
                hasp = not (pd.isna(me.paint) or pd.isna(op.paint))
                x = [1, poss, me.fga, me.fg3a, me.fg3m, me.fta, me.orb, me.drb, me.tov, me.pts,
                     me.paint if hasp else 0, me.pts if hasp else 0,
                     op.fga, op.fg3a, op.fg3m, op.fta, op.orb, op.drb, op.tov, op.pts,
                     op.paint if hasp else 0, op.pts if hasp else 0]
                v, t0 = S.get(me.team, (np.zeros(len(K)), d))
                S[me.team] = (np.array(v) * 0.5 ** ((d - t0) / HS) + np.array(x, float), d)
    return pd.DataFrame(rows)


def build(seasons, live_date=None, out_names=None, frozen=None) -> pd.DataFrame:
    live = live_date is not None
    g = load_games(seasons, live_date)
    box = load_box(g, seasons)
    betas = {int(s): frozen["bpm"] for s in g.season.unique()} if frozen else betas_by_season(box, g)
    g["mu0"] = ridge_preds(g, H_TEAM, LAM)
    g["tot0"] = ridge_totals(g)
    pf = player_pass(g, box, betas, out_names)
    sf = style_pass(g, box, load_paint(g, live))
    g = g.merge(pf, on="event_id", how="left").merge(sf, on="event_id", how="left")
    g["dPfull"] = g.Pfull_home - g.Pfull_away
    g["Ofull"] = g.Ofull_home + g.Ofull_away
    for v in ("strict", "act"):
        g[f"dAbs_{v}"] = (g[f"P{v}_home"] - g.Pfull_home) - (g[f"P{v}_away"] - g.Pfull_away)
        g[f"oAbs_{v}"] = (g[f"O{v}_home"] - g.Ofull_home) + (g[f"O{v}_away"] - g.Ofull_away)
    g["news"] = [bool(len(a) or len(b)) for a, b in zip(g.news_home, g.news_away)]
    return g, betas


# ------------------------------------------------------------------------------------------------- style + stacking
RAW = ["fg3a_rate", "opp_fg3a_rate", "orb_pct", "drb_pct", "tov_pct", "opp_tov_pct", "ft_rate", "opp_ft_rate", "pace"]


def style_feats(d: pd.DataFrame, zc: dict) -> pd.DataFrame:
    z = lambda side, k: (d[f"{side}_{k}"] - zc[k][0]) / zc[k][1]
    o = pd.DataFrame(index=d.index)
    o["S1_paint_def"] = (d.home_opp_paint_share - d.away_opp_paint_share).fillna(0)
    o["S2_three"] = z("home", "fg3a_rate") * z("away", "opp_fg3a_rate") - z("away", "fg3a_rate") * z("home", "opp_fg3a_rate")
    o["S3_reb"] = z("away", "orb_pct") * z("home", "drb_pct") - z("home", "orb_pct") * z("away", "drb_pct")
    o["S4_tov"] = z("home", "opp_tov_pct") * z("away", "tov_pct") - z("away", "opp_tov_pct") * z("home", "tov_pct")
    o["S5_ft"] = z("home", "ft_rate") * z("away", "opp_ft_rate") - z("away", "ft_rate") * z("home", "opp_ft_rate")
    o["U1_pace"] = z("home", "pace") + z("away", "pace")
    o["U2_fg3pct"] = d.home_fg3_pct + d.away_fg3_pct
    o["U3_oppfg3pct"] = d.home_opp_fg3_pct + d.away_opp_fg3_pct
    o["U4_fg3arate"] = z("home", "fg3a_rate") + z("away", "fg3a_rate")
    o["U5_b2b"] = d.b2b_home + d.b2b_away
    return o


def zconst(d: pd.DataFrame) -> dict:
    return {k: (float(pd.concat([d[f"home_{k}"], d[f"away_{k}"]]).mean()),
                float(pd.concat([d[f"home_{k}"], d[f"away_{k}"]]).std())) for k in RAW}


def design(d, model, var, zc, sc=None):
    st = style_feats(d, zc)
    base = {"mu0": d.mu0, "dPfull": d.dPfull, "dAbs": d[f"dAbs_{var}"], "tot0": d.tot0, "Ofull": d.Ofull,
            "oAbs": d[f"oAbs_{var}"]}
    X = pd.DataFrame({c: base[c] if c in base else st[c] for c in FEAT[model]}, index=d.index)
    if sc is None:
        sc = {c: (float(X[c].mean()), float(X[c].std())) for c in X.columns if c not in ("mu0", "tot0")}
    for c, (m, s) in sc.items():
        X[c] = (X[c] - m) / s
    return X, sc


def fit_stack(tr, model, var, zc=None):
    y = tr.margin if model.startswith("M") else tr.total
    zc = zc or zconst(tr)
    X, sc = design(tr, model, var, zc)
    ok = X.notna().all(axis=1) & y.notna()
    f = sm.OLS(y[ok], sm.add_constant(X[ok])).fit()
    return {"model": model, "var": var, "params": f.params.to_dict(), "bse": f.bse.to_dict(), "zc": zc, "sc": sc,
            "sigma": float(np.sqrt(f.scale)), "n": int(ok.sum())}


def predict(fit, d):
    X, _ = design(d, fit["model"], fit["var"], fit["zc"], fit["sc"])
    p = fit["params"]
    return p["const"] + sum(p[c] * X[c] for c in X.columns)


def walk_forward(g: pd.DataFrame) -> pd.DataFrame:
    base = g[g.scored & (g.season >= 2023)].copy()          # 2022 = warm-up only (AMENDMENT)
    out = []
    for s in (2024, 2025):
        tr, te = base[base.season < s], base[base.season == s].copy()
        for model in ("M1", "M2", "T1", "T2"):
            for var in ("strict", "act"):
                f = fit_stack(tr.dropna(subset=["mu0", "tot0", "dPfull", "Ofull"]), model, var)
                te[f"{model}_{var}"] = predict(f, te)
                te[f"{model}_{var}_sig"] = f["sigma"]
        out.append(te)
    return pd.concat(out)


# ---------------------------------------------------------------------------------------------------------- markets
def markets(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y in DEV:
        if y < 2023:
            continue
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        bk = ln[~ln.provider.isin(NOT_BOOKS)]
        c = bk.groupby("event_id").agg(spr_close=("home_spread_close", "median"), tot_close=("total_close", "median"),
                                       hso_c=("home_spread_odds_close", "median"), aso_c=("away_spread_odds_close", "median"),
                                       oo_c=("over_odds_close", "median"), uo_c=("under_odds_close", "median")).reset_index()
        o = bk[bk.provider.isin(["DraftKings", "ESPN BET"])].dropna(subset=["home_spread_open"]).copy()
        o["pri"] = (o.provider != "DraftKings").astype(int)
        o = o.sort_values("pri").drop_duplicates("event_id")
        o = o[["event_id", "provider", "home_spread_open", "home_spread_close", "home_spread_odds_open",
               "away_spread_odds_open", "total_open", "total_close", "over_odds_open", "under_odds_open"]].rename(
            columns={"provider": "open_book", "home_spread_close": "spr_close_same", "total_close": "tot_close_same"})
        rows.append(c.merge(o, on="event_id", how="left"))
    m = pd.concat(rows)
    m["pf_spr_open"] = devig_two_way(m.home_spread_odds_open, m.away_spread_odds_open)[0]
    m["pf_over_open"] = devig_two_way(m.over_odds_open, m.under_odds_open)[0]
    m["pf_spr_close"] = devig_two_way(m.hso_c, m.aso_c)[0]
    m["pf_over_close"] = devig_two_way(m.oo_c, m.uo_c)[0]
    return d.merge(m, on="event_id", how="left")


def boot(v, season, seed=0):
    v = np.asarray(v, float)
    if len(v) == 0:
        return {"bets": 0, "roi": float("nan"), "p": 1.0}
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    per = pd.Series(v, index=season.index).groupby(season).agg(["mean", "size"])
    return {"bets": int(len(v)), "units": float(v.sum()), "roi": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)),
            "folds_positive": int((per["mean"] > 0).sum()), "folds": int(len(per)),
            "per_fold": {str(k): [round(float(a), 4), int(b)] for k, (a, b) in per.iterrows()}}


def bets(d, kind, pred, line, close_line, pf_col, thr):
    """kind 'spr': pred = model home margin, line = home spread (neg = home fav). kind 'tot': pred = model total."""
    x = d[d[pred].notna() & d[line].notna()].copy()
    gap = (x[pred] + x[line]) if kind == "spr" else (x[pred] - x[line])
    x = x[gap.abs() >= thr].copy()
    side = np.sign(gap[x.index])                              # +1 = home / over
    res = (x.margin + x[line]) if kind == "spr" else (x.total - x[line])
    res = side * res
    pf = np.where(side > 0, x[pf_col], 1 - x[pf_col]) if pf_col else np.full(len(x), np.nan)
    x["side"], x["win"], x["push"] = side, (res > 0).astype(float), (res == 0)
    x["pnl_novig"] = np.where(res > 0, 1 / pf - 1, np.where(res == 0, 0.0, -1.0))
    x["pnl_110"] = np.where(res > 0, 100 / 110, np.where(res == 0, 0.0, -1.0))
    if close_line:
        mv = (x[line] - x[close_line]) if kind == "spr" else (x[close_line] - x[line])
        x["clv"] = side * mv
    if kind == "spr":
        x["fav"] = np.where(side > 0, x[line] < 0, x[line] > 0)       # bet side is the favourite at that line
    else:
        x["fav"] = side > 0                                            # 'fav' = over for totals
    return x


def describe_stack(d, pred, line, kind, tag):
    x = d[d[pred].notna() & d[line].notna()]
    y = x.margin if kind == "spr" else x.total
    mk = -x[line] if kind == "spr" else x[line]
    f = sm.OLS(y.to_numpy(float), sm.add_constant(np.column_stack([mk.to_numpy(float), x[pred].to_numpy(float)]))).fit()
    ci = f.conf_int()
    mae_m, mae_k = np.mean(np.abs(x[pred] - y)), np.mean(np.abs(mk - y))
    print(f"  {tag:34s} n {len(x):5d} MAE model {mae_m:6.2f} line {mae_k:6.2f} | stack b_line {f.params[1]:+.2f} "
          f"b_model {f.params[2]:+.2f} [{ci[2][0]:+.2f},{ci[2][1]:+.2f}]")
    return {"n": int(len(x)), "mae_model": float(mae_m), "mae_line": float(mae_k), "b_model": float(f.params[2]),
            "b_model_ci": [float(ci[2][0]), float(ci[2][1])], "b_line": float(f.params[1])}


# --------------------------------------------------------------------------------------------------------- commands
def print_coefs(fit, title):
    print(f"\n  {title}  (n {fit['n']}, residual SD {fit['sigma']:.2f}; style terms are per training SD)")
    for k, v in fit["params"].items():
        print(f"    {k:14s} {v:+8.3f}  (se {fit['bse'][k]:.3f})")


def cmd_build(save=True):
    g, betas = build(DEV)
    print(f"games {len(g)}; scored {g.scored.sum()}; news games {g.news.sum()} ({g[g.scored].news.mean():.1%} of scored)")
    for s, b in betas.items():
        print(f"  BPM-lite beta for season {s} (fit on {b['n']} player-seasons, R2 {b['r2']:.2f}): " +
              " ".join(f"{k} {v:+.2f}" for k, v in zip(SHORT, b["beta"])))
    wf = walk_forward(g)
    print("\nOUT-OF-SAMPLE accuracy (no market), margin MAE / total MAE by season:")
    for s in (2024, 2025):
        x = wf[wf.season == s]
        msg = " ".join(f"{m}_{v} {np.mean(np.abs(x[f'{m}_{v}'] - x.margin)):.2f}" for m in ("M1", "M2") for v in ("strict", "act"))
        tmsg = " ".join(f"{m}_{v} {np.mean(np.abs(x[f'{m}_{v}'] - x.total)):.2f}" for m in ("T1", "T2") for v in ("strict", "act"))
        print(f"  {s} n {len(x)}: M0 {np.mean(np.abs(x.mu0 - x.margin)):.2f} {msg} | T0 {np.mean(np.abs(x.tot0 - x.total)):.2f} {tmsg}")
    if save:
        OUTD.mkdir(parents=True, exist_ok=True)
        wf.drop(columns=["loc"]).to_pickle(OUTD / "wf.pkl")
        # frozen live parameters: BPM on all dev player-seasons; stacks on all dev scored games
        box = load_box(g, DEV)
        bpm = fit_bpm(player_seasons(box, g))
        base = g[g.scored & (g.season >= 2023)]
        base = base.dropna(subset=["mu0", "tot0", "dPfull", "Ofull"])
        fits = {f"{m}_{v}": fit_stack(base, m, v) for m in ("M1", "M2", "T1", "T2") for v in ("strict", "act")}
        FROZEN.parent.mkdir(parents=True, exist_ok=True)
        FROZEN.write_text(json.dumps({"bpm": bpm, "fits": fits, "built": datetime.now(timezone.utc).isoformat()}, indent=1))
        print("\nBPM-lite (all dev, used live):")
        for k, v, se in zip(SHORT, bpm["beta"], bpm["se"]):
            print(f"    {k:4s} per 48 {v:+.3f} (se {se:.3f})")
        for k in ("M2_strict", "M2_act", "T2_strict", "T2_act"):
            print_coefs(fits[k], k)
    return g, wf


def cmd_backtest(ledger=False):
    wf = pd.read_pickle(OUTD / "wf.pkl") if (OUTD / "wf.pkl").exists() else cmd_build()[1]
    d = markets(wf)
    print(f"\nscored 2024-25 {len(d)}; with open {d.home_spread_open.notna().sum()} "
          f"({d[d.home_spread_open.notna()].open_book.value_counts().to_dict()}); with close {d.spr_close.notna().sum()}")
    res = {"descriptive": {}}
    print("\nDESCRIPTIVE: margin ~ line + model (stacking), out of sample")
    for m in ("mu0", "M1_strict", "M2_strict", "M1_act", "M2_act"):
        res["descriptive"][f"spr_open_{m}"] = describe_stack(d, m, "home_spread_open", "spr", f"{m} vs OPEN spread")
        res["descriptive"][f"spr_close_{m}"] = describe_stack(d, m, "spr_close", "spr", f"{m} vs CLOSE spread")
    for m in ("tot0", "T1_strict", "T2_strict", "T1_act", "T2_act"):
        res["descriptive"][f"tot_open_{m}"] = describe_stack(d, m, "total_open", "tot", f"{m} vs OPEN total")
        res["descriptive"][f"tot_close_{m}"] = describe_stack(d, m, "tot_close", "tot", f"{m} vs CLOSE total")
    o = d[d.home_spread_open.notna() & d.spr_close_same.notna()]
    for m in ("mu0", "M2_strict", "M2_act"):
        c = np.corrcoef(o[m] + o.home_spread_open, o.home_spread_open - o.spr_close_same)[0, 1]
        print(f"  corr({m} - open, line move toward home) {c:+.3f} n {len(o)}")
    print(f"  mean |close - open| spread {np.mean(np.abs(o.spr_close_same - o.home_spread_open)):.2f} pts")

    plan = [
        ("T1_spr_strict_open_clv", "spr", "M2_strict", "home_spread_open", "spr_close_same", "pf_spr_open", SPR_X, "clv", None),
        ("T2_spr_strict_open_novig", "spr", "M2_strict", "home_spread_open", "spr_close_same", "pf_spr_open", SPR_X, "pnl_novig", None),
        ("T3_spr_strict_open_110", "spr", "M2_strict", "home_spread_open", "spr_close_same", "pf_spr_open", SPR_X, "pnl_110", None),
        ("T4_tot_strict_open_clv", "tot", "T2_strict", "total_open", "tot_close_same", "pf_over_open", TOT_X, "clv", None),
        ("T5_tot_strict_open_novig", "tot", "T2_strict", "total_open", "tot_close_same", "pf_over_open", TOT_X, "pnl_novig", None),
        ("T6_spr_actual_close_novig", "spr", "M2_act", "spr_close", None, "pf_spr_close", SPR_X, "pnl_novig", None),
        ("T7_tot_actual_close_novig", "tot", "T2_act", "tot_close", None, "pf_over_close", TOT_X, "pnl_novig", None),
        ("T8_spr_actual_open_clv_news", "spr", "M2_act", "home_spread_open", "spr_close_same", "pf_spr_open", SPR_X, "clv", True),
    ]
    assert len(plan) == N_TRIALS
    bar = 0.05 / N_TRIALS
    print(f"\nTRIALS (bar one-sided p < {bar:.5f}); effect = mean per bet (CLV in points; P&L in units)")
    recs, res["trials"] = [], {}
    for name, kind, pred, line, cl, pf, thr, metric, news in plan:
        dd = d[d.news] if news else d
        x = bets(dd, kind, pred, line, cl, pf, thr)
        x = x[x[metric].notna()]
        r = boot(x[metric], x.season)
        r["win_rate_ex_push"] = float(x.win[~x.push].mean()) if len(x) else float("nan")
        split = {}
        for lab, mask in (("fav_or_over", x.fav), ("dog_or_under", ~x.fav)):
            split[lab] = [round(float(x[metric][mask].mean()), 4), int(mask.sum())]
        r["split"] = split
        r["passed"] = bool(r["p"] < bar and r["roi"] > 0)
        res["trials"][name] = r
        print(f"  {name:30s} n {r['bets']:5d} mean {r['roi']:+.4f} CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p']:.4f} "
              f"win {r['win_rate_ex_push']:.3f} seasons+ {r['folds_positive']}/{r['folds']} {r['per_fold']} "
              f"fav/dog {split} {'PASS' if r['passed'] else ''}")
        sha = hashlib.sha256((DATASET + ":" + name).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_simple_{name}",
                     "sha": sha, "dataset": DATASET, "metric": metric, "trial": name,
                     "hypothesis": f"pre-registered in scripts/nba_simple_model.py ({name})",
                     "source": "simple team+player+style model vs open/close (user 2026-10-25)", "status": "ok",
                     "trial_number": len(recs) + 1, "bonferroni_bar": bar, **r})

    print("\nDESCRIPTIVE splits (not trials):")
    extra = {}
    for tag, pred, news in (("M2_act news", "M2_act", True), ("M2_act no-news", "M2_act", False),
                            ("M2_strict news", "M2_strict", True), ("M2_strict no-news", "M2_strict", False),
                            ("M0 all", "mu0", None), ("M1_strict all", "M1_strict", None), ("M2_strict all", "M2_strict", None)):
        dd = d if news is None else d[d.news == news]
        x = bets(dd, "spr", pred, "home_spread_open", "spr_close_same", "pf_spr_open", SPR_X).dropna(subset=["clv"])
        b = boot(x.clv, x.season)
        extra[tag] = {"n": b["bets"], "clv": b.get("roi"), "ci": [b.get("ci_lo"), b.get("ci_hi")], "p": b.get("p"),
                      "novig": float(x.pnl_novig.mean()), "win": float(x.win[~x.push].mean())}
        print(f"  spread vs OPEN {tag:20s} n {b['bets']:5d} CLV {b['roi']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] "
              f"ATS win {extra[tag]['win']:.3f} no-vig {extra[tag]['novig']:+.4f}")
    for tag, pred in (("T0", "tot0"), ("T1_strict", "T1_strict"), ("T2_strict", "T2_strict"), ("T2_act", "T2_act")):
        x = bets(d, "tot", pred, "total_open", "tot_close_same", "pf_over_open", TOT_X).dropna(subset=["clv"])
        b = boot(x.clv, x.season)
        extra[f"tot {tag}"] = {"n": b["bets"], "clv": b.get("roi"), "novig": float(x.pnl_novig.mean())}
        print(f"  total vs OPEN {tag:21s} n {b['bets']:5d} CLV {b['roi']:+.3f} [{b['ci_lo']:+.3f},{b['ci_hi']:+.3f}] "
              f"O/U win {x.win[~x.push].mean():.3f} no-vig {x.pnl_novig.mean():+.4f}")
    for thr in (1.0, 2.0, 3.0, 4.0, 6.0):
        x = bets(d, "spr", "M2_strict", "home_spread_open", "spr_close_same", "pf_spr_open", thr).dropna(subset=["clv"])
        print(f"  dose: M2_strict spread gap >= {thr:.0f}: n {len(x):5d} CLV {x.clv.mean():+.3f} ATS {x.win[~x.push].mean():.3f}"
              f" no-vig {x.pnl_novig.mean():+.4f}")
    res["descriptive"]["splits"] = extra
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nba_simple_model.json").write_text(json.dumps(res, indent=1, default=float))
    if ledger:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


def cmd_today(date, refresh, out_names, as_json=False, no_cache=False):
    """LIVE PREDICTION ONLY (reads 2025-26 + current season; never scores). Delegates to scripts/nba_simple_live.py,
    which adds ESPN paint points, current rosters, the ESPN injury feed, --json and a per-day state cache. Frozen
    coefficients and feature formulas are this file's. The old today body (stats.nba.com paint, last-10-games roster,
    no injuries) is what `model_*_prereg` in the live output reproduces."""
    import nba_simple_live
    argv = (["--date", str(date)] if date else []) + (["--json"] if as_json else []) + \
        (["--out", ",".join(sorted(out_names))] if out_names else []) + ([] if refresh else ["--no-refresh"]) + \
        (["--no-cache"] if no_cache else [])
    return nba_simple_live.main(argv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "backtest", "today"])
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--date", default=None)
    ap.add_argument("--refresh", action="store_true", help="(today) no-op kept for compatibility: refresh is default")
    ap.add_argument("--no-refresh", action="store_true", help="(today) skip the incremental ESPN games/box update")
    ap.add_argument("--no-cache", action="store_true", help="(today) rebuild the per-day state cache")
    ap.add_argument("--json", action="store_true", help="(today) JSON list on stdout, one object per game")
    ap.add_argument("--out", default="", help="comma-separated player names to treat as OUT tonight")
    a = ap.parse_args()
    if a.cmd == "build":
        cmd_build()
    elif a.cmd == "backtest":
        return cmd_backtest(a.ledger)
    else:
        date = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
        return cmd_today(date, not a.no_refresh, {s.strip() for s in a.out.split(",") if s.strip()}, a.json,
                         a.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
