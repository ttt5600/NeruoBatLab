"""NHL PDO / shooting-luck regression vs the market, and blind unders by opening total (DEV ONLY).

Pre-registered 2026-10-27, written BEFORE any market data was joined to luck features or outcomes.

    ../.venv_quant/bin/python -I scripts/nhl_luck.py build              # parse SBR + NHL API, join, luck features
    ../.venv_quant/bin/python -I scripts/nhl_luck.py backtest [--ledger] # the 17 pre-registered trials

Lead being tested: notes/nhl_simple_model.md -- blind UNDER at an ESPN BET opening 6.5 made +5.8% no-vig on 2023-25
(1,448 games, post-hoc, found while looking at the model). This script tests it on ~15 OLDER seasons from a different
source, plus a PDO-luck hypothesis.

DATA (all dev; NHL sealed season = 2026 = 2025-26 per data/frozen/2026-10-14/MANIFEST.json, never read here):
  - data/sbr_nhl/nhl_odds_<season>.html: sportsbookreviewsonline NHL archive, 2007-08 .. 2022-23 (2022-23 is partial;
    "2021" = the 2020-21 season). Rows in V/H pairs: Open ML, Close ML, (puck line from 2014-15), open total + price,
    close total + price. KNOWN GAP: SBR says nothing about which book or what time "Open" is; the close is also an
    unnamed consensus/book. Totals price convention (first row of a pair = OVER, second = UNDER) is VERIFIED in
    `build` against ESPN close over/under odds for overlapping games (market-vs-market only, no outcomes), not assumed.
  - data/nhl_api_teamgames/team_summary_<seasonId>.json: NHL stats REST API (api.nhle.com, public, keyless),
    team summary per game, regular season (gameTypeId=2): goalsFor/Against, shotsFor/Against, winsInShootout.
    Used for (a) restricting SBR rows to regular-season games (join on date + teams + final score) and (b) shots.
  - Step-0 looks taken before this docstring: the SBR table layout and its first ~5 rows for 4 seasons; the API row
    layout for one game. No luck feature was computed and no outcome was aggregated.
  - Descriptive only (NOT a trial; it is the discovery sample): T1 recomputed on ESPN BET openers 2023-24/2024-25
    (data/espn/nhl/lines_2024/2025.parquet).

GRADING: payout per 1 unit staked at the American price of the side bet (juiced totals priced exactly: -120 wins
  0.833, +100 wins 1.0); win/push(0)/loss(-1). Integer totals (6.0) can push. Finals include OT; a shootout counts as one
  goal (the SBR/API final), as books grade NHL totals. Fair probs: proportional two-way de-vig.

LUCK FEATURES (season-to-date, games strictly BEFORE the game date, same season only):
  Per team T: SF, SA = shots for/against; GF, GA = goals for/against EXCLUDING the shootout goal; n = games played.
  League mean lg = sum(GF)/sum(SF) over all games before the date this season (first game-day: previous season's
  league mean; 2007-08 first day: 2007-08 full-season mean -- 2007-08 is warm-up only for T2-T4 anyway).
  Prior: shooting% and save% each regressed toward the league with K = 2500 pseudo-shots:
     ex_sh = sh - (GF + K*lg)/(SF + K)          sh = GF/SF
     ex_sv = sv - (SA - GA + K*(1-lg))/(SA + K)   sv = 1 - GA/SA
  a_T = ex_sh * SF/n   (goals/game the team's scoring rate is inflated by luck)
  d_T = ex_sv * SA/n   (goals/game the team's goals-against rate is deflated by luck)
  Game TOTAL luck  LT = (a_H + a_A) - (d_H + d_A)   > 0 -> raw-goal ratings overstate the total -> bet UNDER
  Game ML luck     LM = (a_H + d_H) - (a_A + d_A)   > 0 -> home overrated (PDO luck)            -> bet AWAY
  (i.e. each team's (sh% + sv% - 1) excess over its regressed value, times shots.)
  Extreme quintiles: cutoffs = 20th / 80th percentiles of LT (LM) over eligible games of ALL PRIOR seasons (expanding,
  no lookahead). 2007-08 = cutoff warm-up only. Top quintile -> UNDER / AWAY; bottom quintile -> OVER / HOME.
  Eligible: T2/T4 = both teams have >= 16 prior games; T3 (early) = both teams have 1..15 prior games (cutoffs from
  prior seasons' early games). Disjoint sets.

TRIALS (17; dataset bar p < 0.05/17 = 0.00294; pass needs p < bar AND effect > 0; BH q=0.05 over all 17 also
  reported). p = one-sided cluster bootstrap over game-DATES (4,000 resamples), share of resampled mean <= 0.
  T1 blind UNDER by SBR opening total, 2007-08 .. 2022-23 (all regular-season games with that open total):
    T1a open 6.5, graded at OPEN line+price         T1b open 6.5, under at the CLOSE line+price
    T1c open 5.5, at open                           T1d open 5.5, under at close
    T1e open 6.0, at open                           T1f open 6.0, under at close
  T2 PDO luck, full season (>= 16 prior games), scored 2008-09 .. 2022-23:
    T2a totals luck bets (LT quintiles), ROI at CLOSE line+price
    T2b totals luck bets, ROI at OPEN line+price
    T2c ML luck bets (LM quintiles), residual = win - de-vigged CLOSE prob of the side (prob pts)
    T2d ML luck bets, ROI at CLOSE price
    T2e ML luck bets, ROI at OPEN price
  T3 early season (both teams 1..15 prior games), scored 2008-09 .. 2022-23:
    T3a totals luck, ROI at close    T3b totals luck, ROI at open
    T3c ML luck, ROI at close        T3d ML luck, ROI at open
  T4 does the market move on luck (T2 bet sets)? CLV toward the luck side, open -> close:
    T4a totals: side * (mu_close - mu_open) in implied goals (side +1 over / -1 under; mu from Poisson total with
        the OT rule, lh = la = mu/2, solved to the de-vigged over prob at each line; handles line + juice moves)
    T4b ML: de-vigged close prob - de-vigged open prob of the side (prob pts)
REPORT per trial: n, effect, 95% CI, p, hit rate vs break-even (mean of 1/decimal price of the bets; pushes excluded),
  both halves (seasons <= 2014-15 vs >= 2015-16), per-season sign count, over/under or home/away split.
DESCRIPTIVE (not trials): OLS of (final - close total) on LT and of (home win - p_close) on LM; OLS of open->close
  move on LT/LM; T1 on ESPN BET 2023-25; rows dropped at each join step.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
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

from sports.market import devig_two_way
from nhl_simple_model import implied_mu  # noqa: E402

SBR = ROOT / "data" / "sbr_nhl"
API = ROOT / "data" / "nhl_api_teamgames"
ESPN = ROOT / "data" / "espn" / "nhl"
OUTD = ROOT / "data" / "nhl_luck"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nhl_luck"
K = 2500.0
N_TRIALS = 17
SEASON_FILES = {2008: "2007-08", 2009: "2008-09", 2010: "2009-10", 2011: "2010-11", 2012: "2011-12", 2013: "2012-13",
                2014: "2013-14", 2015: "2014-15", 2016: "2015-16", 2017: "2016-17", 2018: "2017-18", 2019: "2018-19",
                2020: "2019-20", 2021: "2021", 2022: "2021-22", 2023: "2022-23"}   # season key = end year
assert max(SEASON_FILES) < 2026
HALF_SPLIT = 2015   # first half: seasons ending <= 2015


def num(s):
    return pd.to_numeric(s.astype(str).str.strip().replace({"pk": "0", "PK": "0", "NL": np.nan, "": np.nan}),
                         errors="coerce")


def dec(am):
    am = np.asarray(am, float)
    return np.where(am > 0, 1 + am / 100, 1 + 100 / np.abs(am))


# ------------------------------------------------------------------------------------------------------------ parse
def parse_sbr(y: int) -> pd.DataFrame:
    t = pd.read_html(io.StringIO((SBR / f"nhl_odds_{SEASON_FILES[y]}.html").read_text()))[0]
    t = t.iloc[1:].reset_index(drop=True)
    if t.shape[1] == 14:
        t.columns = ["date", "rot", "vh", "team", "p1", "p2", "p3", "final", "ml_open", "ml_close",
                     "tot_open", "tot_open_px", "tot_close", "tot_close_px"]
    else:
        t.columns = ["date", "rot", "vh", "team", "p1", "p2", "p3", "final", "ml_open", "ml_close", "pl", "pl_px",
                     "tot_open", "tot_open_px", "tot_close", "tot_close_px"]
    a, b = t.iloc[0::2].reset_index(drop=True), t.iloc[1::2].reset_index(drop=True)
    ok = a.vh.isin(["V", "N"]) & b.vh.isin(["H", "N"])
    flipped = a.vh.eq("H") & b.vh.eq("V")
    assert (ok | flipped).all(), y          # anything else = pairing broken
    a, b = a[ok].reset_index(drop=True), b[ok].reset_index(drop=True)   # H-first pairs dropped (O/U row order unclear)
    md = a.date.astype(int)
    mon, day = md // 100, md % 100
    yr = np.where(mon >= 8 if y != 2021 else mon >= 13, y - 1, y)
    if y == 2020:   # 2019-20: Aug 2020 bubble games belong to year 2020
        yr = np.where(mon >= 9, 2019, 2020)
    g = pd.DataFrame({
        "season": y, "date": pd.to_datetime(dict(year=yr, month=mon, day=day), errors="coerce"),
        "away_sbr": a.team.str.strip(), "home_sbr": b.team.str.strip(), "neutral": a.vh.eq("N"),
        "away_final": num(a.final), "home_final": num(b.final),
        "ml_open_a": num(a.ml_open), "ml_open_h": num(b.ml_open),
        "ml_close_a": num(a.ml_close), "ml_close_h": num(b.ml_close),
        "tot_open": num(a.tot_open), "tot_open_b": num(b.tot_open),
        "over_open_px": num(a.tot_open_px), "under_open_px": num(b.tot_open_px),
        "tot_close": num(a.tot_close), "tot_close_b": num(b.tot_close),
        "over_close_px": num(a.tot_close_px), "under_close_px": num(b.tot_close_px)})
    return g


def load_api() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(API / "team_summary_*.json"))):
        d = pd.DataFrame(json.load(open(f))["data"])
        d["season"] = int(Path(f).stem[-4:])
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["date"] = pd.to_datetime(d.gameDate)
    d["SF"], d["SA"] = d.shotsForPerGame, d.shotsAgainstPerGame
    so_w = d.winsInShootout.fillna(0) > 0
    so = d.groupby("gameId").winsInShootout.transform(lambda s: (s.fillna(0) > 0).any())
    # VERIFIED in build v1: API goalsFor/Against EXCLUDE the shootout goal (SO games show tied goals; SBR finals
    # = API + 1 for the SO winner). So GF/GA are used as-is for luck; graded finals add the SO goal back.
    d["GF"], d["GA"] = d.goalsFor, d.goalsAgainst
    d["GF_raw"] = d.goalsFor + so_w.astype(int)
    d["GA_raw"] = d.goalsAgainst + (so & ~so_w).astype(int)
    d["so"] = so
    return d


def game_table(api: pd.DataFrame) -> pd.DataFrame:
    h = api[api.homeRoad == "H"][["gameId", "season", "date", "teamId", "teamFullName", "opponentTeamAbbrev",
                                  "GF_raw", "GA_raw", "so"]]
    r = api[api.homeRoad == "R"][["gameId", "teamId", "teamFullName", "opponentTeamAbbrev"]]
    g = h.merge(r, on="gameId", suffixes=("_h", "_a"))
    g = g.rename(columns={"opponentTeamAbbrev_a": "home", "opponentTeamAbbrev_h": "away",
                          "GF_raw": "home_score", "GA_raw": "away_score"})
    return g[["gameId", "season", "date", "home", "away", "teamId_h", "teamId_a", "home_score", "away_score", "so"]]


def code_map(sbr: pd.DataFrame, g: pd.DataFrame) -> dict:
    from collections import Counter
    homes = g.groupby("date").home.apply(set)
    votes: dict = {}
    for d, c in zip(sbr.date, sbr.home_sbr):
        for ab in homes.get(d, ()):
            votes.setdefault(c, Counter())[ab] += 1
    return {c: v.most_common(1)[0][0] for c, v in votes.items()}


# ------------------------------------------------------------------------------------------------------------- luck
def luck_features(api: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    api = api.sort_values(["date", "gameId"]).copy()
    out = []
    prev_lg = None
    for s, a in api.groupby("season"):
        days = np.sort(a.date.unique())
        full_lg = a.GF.sum() / a.SF.sum()
        cum = {}
        lg_g = lg_s = 0.0
        for dday in days:
            lg = lg_g / lg_s if lg_s > 0 else (prev_lg if prev_lg is not None else full_lg)
            today = a[a.date == dday]
            for _, row in today.iterrows():
                c = cum.get(row.teamId)
                if c is None:
                    out.append((row.gameId, row.teamId, 0, np.nan, np.nan, np.nan, np.nan, lg))
                    continue
                n, SF, SA, GF, GA = c
                ex_sh = GF / SF - (GF + K * lg) / (SF + K)
                ex_sv = (1 - GA / SA) - (SA - GA + K * (1 - lg)) / (SA + K)
                out.append((row.gameId, row.teamId, n, ex_sh * SF / n, ex_sv * SA / n, GF / SF, 1 - GA / SA, lg))
            for _, row in today.iterrows():
                c = cum.get(row.teamId, (0, 0.0, 0.0, 0.0, 0.0))
                cum[row.teamId] = (c[0] + 1, c[1] + row.SF, c[2] + row.SA, c[3] + row.GF, c[4] + row.GA)
            lg_g += today.GF.sum()
            lg_s += today.SF.sum()
        prev_lg = full_lg
    f = pd.DataFrame(out, columns=["gameId", "teamId", "n_prior", "a", "d", "sh", "sv", "lg"])
    g = g.merge(f.add_suffix("_h").rename(columns={"gameId_h": "gameId", "teamId_h": "teamId_h"}),
                on=["gameId", "teamId_h"])
    g = g.merge(f.add_suffix("_a").rename(columns={"gameId_a": "gameId", "teamId_a": "teamId_a"}),
                on=["gameId", "teamId_a"])
    g["LT"] = (g.a_h + g.a_a) - (g.d_h + g.d_a)
    g["LM"] = (g.a_h + g.d_h) - (g.a_a + g.d_a)
    g["elig_full"] = (g.n_prior_h >= 16) & (g.n_prior_a >= 16)
    g["elig_early"] = g.n_prior_h.between(1, 15) & g.n_prior_a.between(1, 15)
    return g.drop(columns=["lg_a"]).rename(columns={"lg_h": "lg"})


# ------------------------------------------------------------------------------------------------------------ build
def cmd_build():
    OUTD.mkdir(exist_ok=True)
    api = load_api()
    g = game_table(api)
    log = {}
    js = []
    for y in SEASON_FILES:
        s = parse_sbr(y)
        gs = g[g.season == y]
        cm = code_map(s, gs)
        s["home"], s["away"] = s.home_sbr.map(cm), s.away_sbr.map(cm)
        s = s.drop(columns="season")
        j = s.merge(gs, on=["date", "home", "away"], how="inner")
        if s.neutral.any():   # neutral-site rows may be listed in either orientation
            sw = s[s.neutral].rename(columns={"home": "away", "away": "home"})
            jn = sw.merge(gs, on=["date", "home", "away"], how="inner")
            if len(jn):
                for c1, c2 in (("ml_open_a", "ml_open_h"), ("ml_close_a", "ml_close_h"), ("away_final", "home_final")):
                    jn[c1], jn[c2] = jn[c2].to_numpy(), jn[c1].to_numpy()
                j = pd.concat([j, jn[~jn.gameId.isin(j.gameId)]])
        j["score_ok"] = (j.home_final == j.home_score) & (j.away_final == j.away_score)
        log[y] = {"sbr_games": len(s), "api_regular": len(gs), "joined": len(j), "score_ok": int(j.score_ok.sum()),
                  "unmapped_codes": sorted(set(s.home_sbr) - set(cm))}
        js.append(j[j.score_ok])
    j = pd.concat(js, ignore_index=True)
    print(json.dumps(log, indent=0))
    # totals price convention: compare SBR close O/U prices with ESPN close odds where both exist (market-only check)
    conv = {}
    for y in (2022, 2023):
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        ga = pd.read_parquet(ESPN / f"games_{y}.parquet")[["event_id", "start_utc", "home", "away"]]
        ga["date"] = ga.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
        mp = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL"}
        ga["home"], ga["away"] = ga.home.replace(mp), ga.away.replace(mp)
        ln = ln.dropna(subset=["total", "over_odds", "under_odds"]).copy()   # last ESPN snapshot ~ close
        ln["po"] = devig_two_way(ln.over_odds, ln.under_odds)[0]
        e = ln.merge(ga, on="event_id").groupby(["date", "home", "away", "total"]).po.median().reset_index()
        m = j[j.season == y].merge(e, left_on=["date", "home", "away", "tot_close"], right_on=["date", "home", "away", "total"])
        sp = devig_two_way(m.over_close_px, m.under_close_px)[0]
        ok = np.isfinite(sp) & np.isfinite(m.po)
        conv[y] = {"n": int(ok.sum()), "corr_row1_as_over_vs_espn_over": float(np.corrcoef(sp[ok], m.po[ok])[0, 1]),
                   "mean_abs_diff": float(np.mean(np.abs(sp[ok] - m.po[ok])))}
    print("price convention check (share where SBR row-1 price has same sign-ordering as ESPN over):", conv)
    feats = luck_features(api, g)
    d = j.merge(feats[["gameId", "n_prior_h", "n_prior_a", "a_h", "a_a", "d_h", "d_a", "sh_h", "sv_h", "sh_a", "sv_a",
                       "LT", "LM", "elig_full", "elig_early", "lg"]], on="gameId", how="left")
    d["total"] = d.home_score + d.away_score
    d["home_win"] = (d.home_score > d.away_score).astype(int)
    d.to_pickle(OUTD / "games.pkl")
    json.dump({"join": log, "convention": conv}, open(OUTD / "build_log.json", "w"), indent=1, default=str)
    print(f"saved {len(d)} games; luck finite {d.LT.notna().sum()}; full-elig {d.elig_full.sum()} early {d.elig_early.sum()}")
    return 0


# --------------------------------------------------------------------------------------------------------- backtest
def boot(pnl, dates, seed=0):
    v = pd.Series(np.asarray(pnl, float)).reset_index(drop=True)
    dd = pd.Series(np.asarray(dates)).reset_index(drop=True)
    if len(v) == 0:
        return {"bets": 0, "roi": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p": 1.0}
    grp = pd.DataFrame({"v": v, "d": dd}).groupby("d").v.agg(["sum", "size"])
    S, N = grp["sum"].to_numpy(), grp["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), (4000, len(S)))
    bs = S[idx].sum(1) / N[idx].sum(1)
    return {"bets": int(len(v)), "roi": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)),
            "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0))}


def grade_tot(x, side, line, opx, upx):
    """side +1 over / -1 under at line with (over, under) American prices."""
    res = side * (x.total.to_numpy() - line)
    px = np.where(side > 0, opx, upx)
    pnl = np.where(res > 0, dec(px) - 1, np.where(res == 0, 0.0, -1.0))
    return pnl, res, 1 / dec(px)


def grade_ml(x, side, ph, pa):
    win = np.where(side > 0, x.home_win == 1, x.home_win == 0)
    px = np.where(side > 0, ph, pa)
    return np.where(win, dec(px) - 1, -1.0), win.astype(float), 1 / dec(px)


def cutoffs(d, col, elig):
    """20/80 percentiles from all PRIOR seasons' eligible games."""
    out = {}
    for s in sorted(d.season.unique()):
        prior = d[(d.season < s) & d[elig] & d[col].notna()][col]
        out[s] = (np.quantile(prior, .2), np.quantile(prior, .8)) if len(prior) > 200 else (np.nan, np.nan)
    return out


def luck_side(d, col, elig, against_high=True):
    c = cutoffs(d, col, elig)
    lo = d.season.map(lambda s: c[s][0])
    hi = d.season.map(lambda s: c[s][1])
    side = np.where(d[col] >= hi, -1, np.where(d[col] <= lo, 1, 0))
    m = d[elig] & d[col].notna() & lo.notna()
    return pd.Series(np.where(m, side, 0), index=d.index)


def summarize(name, x, metric_vals, be=None, hit=None, splitlab=None, metric="roi"):
    r = boot(metric_vals, x.date)
    v = pd.Series(np.asarray(metric_vals, float), index=x.index)
    per = v.groupby(x.season).agg(["mean", "size"])
    r["per_season"] = {str(k): [round(float(a), 4), int(b)] for k, (a, b) in per.iterrows()}
    r["seasons_positive"] = int((per["mean"] > 0).sum())
    r["seasons"] = int(len(per))
    h1, h2 = v[x.season <= HALF_SPLIT], v[x.season > HALF_SPLIT]
    r["halves"] = {"<=2014-15": [float(h1.mean()) if len(h1) else None, int(len(h1))],
                   ">=2015-16": [float(h2.mean()) if len(h2) else None, int(len(h2))]}
    if hit is not None:
        r["hit_rate"] = float(np.nanmean(hit))
        r["breakeven"] = float(np.nanmean(be))
    if splitlab is not None:
        r["split"] = {str(k): [round(float(a), 4), int(b)] for k, (a, b) in v.groupby(splitlab).agg(["mean", "size"]).iterrows()}
    r["metric"] = metric
    return r


def mu(line, po):
    out = np.full(len(line), np.nan)
    ok = np.isfinite(line) & np.isfinite(po)
    if ok.any():
        out[ok] = implied_mu(np.asarray(line)[ok], np.asarray(po)[ok])
    return out


def cmd_backtest(ledger=False):
    d = pd.read_pickle(OUTD / "games.pkl").reset_index(drop=True)
    d["po_open"] = devig_two_way(d.over_open_px, d.under_open_px)[0]
    d["po_close"] = devig_two_way(d.over_close_px, d.under_close_px)[0]
    d["ph_open"] = devig_two_way(d.ml_open_h, d.ml_open_a)[0]
    d["ph_close"] = devig_two_way(d.ml_close_h, d.ml_close_a)[0]
    okO = d[["tot_open", "over_open_px", "under_open_px"]].notna().all(1)
    okC = d[["tot_close", "over_close_px", "under_close_px"]].notna().all(1)
    okMO = d[["ml_open_h", "ml_open_a"]].notna().all(1) & (d.ml_open_h.abs() >= 100) & (d.ml_open_a.abs() >= 100)
    okMC = d[["ml_close_h", "ml_close_a"]].notna().all(1) & (d.ml_close_h.abs() >= 100) & (d.ml_close_a.abs() >= 100)
    print(f"games {len(d)}; open total+prices {okO.sum()}; close total+prices {okC.sum()}; ML open {okMO.sum()} close {okMC.sum()}")
    print("open total distribution:", d.tot_open.value_counts().head(6).to_dict())
    print("overround open tot median", float(np.nanmedian(devig_two_way(d.over_open_px, d.under_open_px)[2])),
          "close tot", float(np.nanmedian(devig_two_way(d.over_close_px, d.under_close_px)[2])),
          "ML open", float(np.nanmedian(devig_two_way(d.ml_open_h, d.ml_open_a)[2][okMO])),
          "ML close", float(np.nanmedian(devig_two_way(d.ml_close_h, d.ml_close_a)[2][okMC])))
    res, recs = {}, []

    def add(name, r, metric):
        res[name] = r
        print(f"  {name:26s} n {r['bets']:5d} eff {r['roi']:+.4f} CI [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p']:.4f} "
              f"hit {r.get('hit_rate', float('nan')):.3f} be {r.get('breakeven', float('nan')):.3f} "
              f"seasons+ {r['seasons_positive']}/{r['seasons']} halves {r['halves']} split {r.get('split')}")

    # T1 blind unders
    for tag, L in (("a", 6.5), ("c", 5.5), ("e", 6.0)):
        x = d[okO & (d.tot_open == L)]
        side = -np.ones(len(x))
        pnl, rs, be = grade_tot(x, side, x.tot_open.to_numpy(), x.over_open_px.to_numpy(), x.under_open_px.to_numpy())
        hit = np.where(rs == 0, np.nan, (rs > 0).astype(float))
        add(f"T1{tag}_under_open{L}_at_open", summarize("", x, pnl, be, hit), "roi")
        x = d[okO & okC & (d.tot_open == L)]
        side = -np.ones(len(x))
        pnl, rs, be = grade_tot(x, side, x.tot_close.to_numpy(), x.over_close_px.to_numpy(), x.under_close_px.to_numpy())
        hit = np.where(rs == 0, np.nan, (rs > 0).astype(float))
        add(f"T1{chr(ord(tag) + 1)}_under_open{L}_at_close", summarize("", x, pnl, be, hit), "roi")

    # T2/T3 totals + ML luck
    d["sideT_full"] = luck_side(d, "LT", "elig_full")
    d["sideM_full"] = luck_side(d, "LM", "elig_full")   # high LM -> -1 = AWAY, low -> +1 = HOME
    d["sideT_early"] = luck_side(d, "LT", "elig_early")
    d["sideM_early"] = luck_side(d, "LM", "elig_early")
    sc = d.season >= 2009
    for per, T, tA, tB, M in (("full", "T2", "a", "b", ("c", "d", "e")), ("early", "T3", "a", "b", None)):
        st = d[f"sideT_{per}"]
        x = d[sc & (st != 0) & okC]
        s = st[x.index].to_numpy()
        pnl, rs, be = grade_tot(x, s, x.tot_close.to_numpy(), x.over_close_px.to_numpy(), x.under_close_px.to_numpy())
        add(f"{T}{tA}_tot_luck_{per}_close", summarize("", x, pnl, be, np.where(rs == 0, np.nan, (rs > 0) * 1.0),
                                                       np.where(s > 0, "over", "under")), "roi")
        x = d[sc & (st != 0) & okO]
        s = st[x.index].to_numpy()
        pnl, rs, be = grade_tot(x, s, x.tot_open.to_numpy(), x.over_open_px.to_numpy(), x.under_open_px.to_numpy())
        add(f"{T}{tB}_tot_luck_{per}_open", summarize("", x, pnl, be, np.where(rs == 0, np.nan, (rs > 0) * 1.0),
                                                      np.where(s > 0, "over", "under")), "roi")
        sm_ = d[f"sideM_{per}"]
        x = d[sc & (sm_ != 0) & okMC]
        s = sm_[x.index].to_numpy()
        lab = np.where(s > 0, "home", "away")
        if per == "full":
            pside = np.where(s > 0, x.ph_close, 1 - x.ph_close)
            win = np.where(s > 0, x.home_win == 1, x.home_win == 0).astype(float)
            add("T2c_ml_luck_full_resid_close", summarize("", x, win - pside, None, None, lab, "prob_pts"), "resid")
        pnl, win, be = grade_ml(x, s, x.ml_close_h.to_numpy(), x.ml_close_a.to_numpy())
        add(f"{T}{'d' if per == 'full' else 'c'}_ml_luck_{per}_close", summarize("", x, pnl, be, win, lab), "roi")
        x = d[sc & (sm_ != 0) & okMO]
        s = sm_[x.index].to_numpy()
        pnl, win, be = grade_ml(x, s, x.ml_open_h.to_numpy(), x.ml_open_a.to_numpy())
        add(f"{T}{'e' if per == 'full' else 'd'}_ml_luck_{per}_open", summarize("", x, pnl, be, win,
                                                                               np.where(s > 0, "home", "away")), "roi")

    # T4 line movement toward the luck side
    d["mu_open"] = mu(d.tot_open.to_numpy(float), d.po_open.to_numpy(float))
    d["mu_close"] = mu(d.tot_close.to_numpy(float), d.po_close.to_numpy(float))
    st = d.sideT_full
    x = d[sc & (st != 0) & okO & okC]
    add("T4a_tot_luck_clv_goals", summarize("", x, st[x.index] * (x.mu_close - x.mu_open), None, None,
                                            np.where(st[x.index] > 0, "over", "under"), "goals"), "clv")
    sm_ = d.sideM_full
    x = d[sc & (sm_ != 0) & okMO & okMC]
    s = sm_[x.index]
    add("T4b_ml_luck_clv_prob", summarize("", x, s * (x.ph_close - x.ph_open), None, None,
                                          np.where(s > 0, "home", "away"), "prob_pts"), "clv")
    assert len(res) == N_TRIALS, len(res)

    # BH + bar
    bar = 0.05 / N_TRIALS
    names = list(res)
    ps = np.array([res[n]["p"] for n in names])
    order = np.argsort(ps)
    bh = np.zeros(len(ps), bool)
    kmax = 0
    for i, o in enumerate(order, 1):
        if ps[o] <= 0.05 * i / len(ps):
            kmax = i
    bh[order[:kmax]] = True
    for i, n in enumerate(names):
        r = res[n]
        r["passed"] = bool(r["p"] < bar and r["roi"] > 0)
        r["bh_reject"] = bool(bh[i] and r["roi"] > 0)
        sha = hashlib.sha256((DATASET + ":" + n).encode()).hexdigest()[:12]
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nhl_luck_{n}",
                     "sha": sha, "dataset": DATASET, "metric": r["metric"], "trial": n,
                     "hypothesis": f"pre-registered in scripts/nhl_luck.py ({n})",
                     "source": "NHL PDO luck + blind opening-total unders, SBR 2007-08..2022-23 (2026-10-27)",
                     "status": "ok", "trial_number": i + 1, "study_trials": N_TRIALS, "bonferroni_bar": bar,
                     "bets": r["bets"], "roi": r["roi"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "p": r["p"],
                     "per_fold": r["per_season"], "halves": r["halves"], "bh_reject": r["bh_reject"],
                     "passed": r["passed"]})
    print("\nPASS (bar %.5f):" % bar, [n for n in names if res[n]["passed"]], " BH:", [n for n in names if res[n]["bh_reject"]])

    # DESCRIPTIVE
    print("\nDESCRIPTIVE (not trials)")
    desc = {}
    for col, el in (("LT", "elig_full"), ("LT", "elig_early")):
        y = d[sc & d[el] & okC & d[col].notna()]
        f = sm.OLS((y.total - y.tot_close).to_numpy(float), sm.add_constant(y[col].to_numpy(float))).fit(
            cov_type="cluster", cov_kwds={"groups": y.date.factorize()[0]})
        desc[f"resid_total_on_{col}_{el}"] = [float(f.params[1]), float(f.bse[1]), int(len(y))]
        print(f"  (final - close total) ~ {col} [{el}]: slope {f.params[1]:+.3f} se {f.bse[1]:.3f} n {len(y)}  "
              f"sd(LT) {y[col].std():.3f}")
        y2 = y[okO[y.index]]
        f = sm.OLS((y2.mu_close - y2.mu_open).to_numpy(float), sm.add_constant(y2[col].to_numpy(float))).fit(
            cov_type="cluster", cov_kwds={"groups": y2.date.factorize()[0]})
        desc[f"move_mu_on_{col}_{el}"] = [float(f.params[1]), float(f.bse[1]), int(len(y2))]
        print(f"  (mu_close - mu_open) ~ {col} [{el}]: slope {f.params[1]:+.4f} se {f.bse[1]:.4f}")
    for el in ("elig_full", "elig_early"):
        y = d[sc & d[el] & okMC & d.LM.notna()]
        f = sm.OLS((y.home_win - y.ph_close).to_numpy(float), sm.add_constant(y.LM.to_numpy(float))).fit(
            cov_type="cluster", cov_kwds={"groups": y.date.factorize()[0]})
        desc[f"resid_ml_on_LM_{el}"] = [float(f.params[1]), float(f.bse[1]), int(len(y))]
        print(f"  (home_win - p_close) ~ LM [{el}]: slope {f.params[1]:+.4f} se {f.bse[1]:.4f} n {len(y)} sd(LM) {y.LM.std():.3f}")
        y2 = y[okMO[y.index]]
        f = sm.OLS((y2.ph_close - y2.ph_open).to_numpy(float), sm.add_constant(y2.LM.to_numpy(float))).fit(
            cov_type="cluster", cov_kwds={"groups": y2.date.factorize()[0]})
        desc[f"move_ml_on_LM_{el}"] = [float(f.params[1]), float(f.bse[1]), int(len(y2))]
        print(f"  (p_close - p_open) ~ LM [{el}]: slope {f.params[1]:+.4f} se {f.bse[1]:.4f}")
    # blind unders per season table + under at close by close line
    for L in (5.5, 6.0, 6.5):
        x = d[okO & (d.tot_open == L)]
        pnl, rs, _ = grade_tot(x, -np.ones(len(x)), x.tot_open.to_numpy(), x.over_open_px.to_numpy(), x.under_open_px.to_numpy())
        nv = np.where(rs > 0, 1 / (1 - x.po_open.to_numpy()) - 1, np.where(rs == 0, 0, -1.0))
        desc[f"T1_open{L}_novig"] = float(np.mean(nv))
        print(f"  open {L}: under no-vig {np.mean(nv):+.4f}; mean under price {np.median(x.under_open_px):+.0f}; "
              f"mean fair P(under) {1 - x.po_open.mean():.3f}; push share {(rs == 0).mean():.3f}; n {len(x)}")
        print("     by season (at open):", {int(k): round(float(v), 3) for k, v in pd.Series(pnl, index=x.index).groupby(x.season).mean().items()})
    # ESPN BET 2023-25 replication of T1 (discovery sample)
    rows = []
    for y in (2024, 2025):
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        gm = pd.read_parquet(ESPN / f"games_{y}.parquet")
        gm = gm[(gm.season_type == "regular-season") & (gm.status == "STATUS_FINAL")]
        o = ln[ln.provider == "ESPN BET"].dropna(subset=["total_open", "under_odds_open"]).drop_duplicates("event_id")
        o = o.merge(gm[["event_id", "home_score", "away_score"]], on="event_id")
        o["season"] = y
        rows.append(o)
    e = pd.concat(rows)
    e["total"] = e.home_score + e.away_score
    for L in (5.5, 6.5):
        x = e[e.total_open == L]
        res_ = x.total - L
        pnl = np.where(res_ < 0, dec(x.under_odds_open) - 1, -1.0)
        desc[f"espnbet_under_open{L}_offered"] = [float(pnl.mean()), int(len(x))]
        print(f"  ESPN BET 2023-25 under at open {L} offered price: {pnl.mean():+.4f} n {len(x)}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nhl_luck.json").write_text(json.dumps({"trials": res, "descriptive": desc}, indent=1, default=float))
    if ledger:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=float) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "backtest"])
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    return cmd_build() if a.cmd == "build" else cmd_backtest(a.ledger)


if __name__ == "__main__":
    sys.exit(main())
