"""NBA 3-point LUCK: does the market overreact to small-sample 3P shooting (early season and generally)? DEV ONLY.

    ../.venv_quant/bin/python scripts/nba_luck.py features            # feature build + distribution only (NO market)
    ../.venv_quant/bin/python scripts/nba_luck.py backtest [--ledger]  # pre-registered trials (joins lines)

PRE-REGISTRATION (written 2026-10-27 before any market join in this script; the only prior market look is the
post-hoc scripts/nba_luck_early_calib.py on 2022-23..2024-25 -- "both-cold early games +7.0 over the close, n 59" --
which is WHY this study exists; those 3 seasons are inside DEV here, so the 2015-16..2021-22 subset is the cleaner
read and is reported descriptively).

DATA (DEV only). Seasons 2015-16..2024-25 regular season (comebacks `season` 2015..2024). SEALED and never read:
NBA 2025-26 (ESPN games/box/lines_2026+, OddsData odds_2025-26, nba_games.parquet season 2025) and
data/nba_ingame.parquet. Asserted in code.
  - 3P makes/attempts per team-game, own and opponent: data/nba_comebacks/teamgames.parquet (stats.nba.com logs).
  - Closing lines: data/nba_comebacks/games.parquet `total_line` / `spread` (expected HOME margin) -- ESPN pickcenter
    spreads 2015-22, OddsData 2023-24; totals OddsData (170-270). Assumed close (unverified, see comebacks caveats).
  - Openers (T2 only): ESPN lines_2024 / lines_2025 (= 2023-24, 2024-25): DraftKings total_open, else ESPN BET
    (simple-model rule). Same-book total_close for the line move. Joined on ESPN event id (games.eid).
  - Model cross-check: data/nba_simple_model/wf.pkl `T2_strict` (walk-forward total, 2023-24 + 2024-25).

FEATURE (same definition as nba_tonight_sheet.py; season-to-date, strictly before this game, no lookahead):
  shrunk 3P% = (3PM + K*prior) / (3PA + K), K = 750 attempts.
  Offence prior = the team's own PRIOR-season 3P%; defence prior = PRIOR-season league 3P%.
  2015-16 has no prior season in the data -> both priors = league 3P% of all 2015-16 games strictly before this
  game's date (no lookahead; the season's first date has no prior -> games with 0 prior games are dropped anyway).
  luck_off(team) = (3PM_td - 3PA_td * shrunk_off) / GP * 3   (pts/g of scoring from 3P above the shrunk rate)
  luck_def(team) = (opp3PM_td - opp3PA_td * shrunk_def) / GP * 3  (pts/g ALLOWED above shrunk; + = opponents hot)
  GAME LUCK  = 0.5 * (luck_off_h + luck_off_a + luck_def_h + luck_def_a).  + = recent form overstates scoring.
  MARGIN LUCK(team) = luck_off - luck_def;  D = margin_luck_home - margin_luck_away  (+ = home flattered).
  Games need both teams with >= 1 prior game this season.
BUCKETS: EARLY = both teams have played 1..10 prior games this season (tonight is game 2..11 for each).
  REST = every other game with both >= 1 prior game.  ALL = EARLY + REST.
QUINTILE CUTS: q20 / q80 of the feature (game luck, D) computed WITHIN bucket over all DEV games -- features only,
  no outcomes; frozen into results for forward use.
HYPOTHESIS (market overreacts / under-regresses 3P luck): total residual (final - line) falls with game luck
  (slope < 0); home ATS residual (margin - home spread) falls with D (slope < 0).

TRIALS (14; one-sided in the predicted direction unless noted). Slope p = OLS, SE clustered by game date.
ROI trials at -110 (win +0.9091, push 0, loss -1), one bet per game, one-sided game bootstrap (4,000).
  T1a  ALL  total res vs CLOSE ~ game luck, slope < 0
  T1b  ALL  CLOSE totals bet: luck <= q20 -> OVER, luck >= q80 -> UNDER; ROI at -110 > 0
  T2a  ALL  total res vs OPENER ~ game luck, slope < 0               (2023-24, 2024-25 only)
  T2b  ALL  OPENER totals quintile bet (cuts as T1b), ROI at -110 > 0
  T2c  ALL  line move (same-book close - open) ~ game luck, slope != 0 (TWO-sided: does the market move on luck?)
  T3a  ALL  home ATS res vs CLOSE ~ D, slope < 0
  T3b  ALL  CLOSE sides: D >= q80 -> bet AWAY, D <= q20 -> bet HOME (against the lucky team); ROI at -110 > 0
  T4-1a/T4-1b, T4-2a/T4-2b/T4-2c, T4-3a/T4-3b: the same seven, EARLY bucket only (EARLY cuts).
METRICS per trial: slope + 95% CI, hit rate vs 52.4% (binomial), ROI at -110, both halves (2015-19 = seasons
  2015..2018 vs 2020-25 = 2019..2024; for opener trials halves = 2023-24 vs 2024-25), season sign count.
MULTIPLICITY: Bonferroni 0.05/14 and BH(q=.05) over the 14; whole-ledger bar 0.05/(ledger rows + 14) reported.
PASS (per trial) = Bonferroni-14 p AND same sign in both halves (ROI trials: ROI > 0 both halves).
DESCRIPTIVE (not trials): REST-bucket versions, season FE robustness, dose by decile, 2015-22-only (pre-generation)
  subset, and "is it already in our model": res vs close ~ luck + (T2_strict - close), and luck's partial effect after
  L31's season 3P% for/against terms (z of summed raw 3P% for and allowed).
Ledger: lab/ledger.jsonl, dataset `nba_luck`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "data" / "nba_comebacks"
ESPN = ROOT / "data" / "espn" / "nba"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_luck"
K3 = 750.0
DEV = list(range(2015, 2025))           # comebacks season = start year; 2025 (=2025-26) is SEALED
assert 2025 not in DEV
OPEN_FILES = {2023: "lines_2024.parquet", 2024: "lines_2025.parquet"}   # ESPN file year = end year
NOT_BOOKS = {"consensus", "accuscore", "teamrankings", "betegy"}
N_TRIALS = 14
WIN110 = 100 / 110


# ------------------------------------------------------------------------------------------------ features (no market)
def features() -> pd.DataFrame:
    t = pd.read_parquet(CB / "teamgames.parquet")
    t = t[(t.stype == "Regular Season") & t.season.isin(DEV)].copy()
    assert t.season.max() <= 2024
    t = t.sort_values(["team", "season", "date", "GAME_ID"]).reset_index(drop=True)
    gb = t.groupby(["team", "season"], sort=False)
    for c in ("fg3m", "fg3a", "o_fg3m", "o_fg3a"):
        t["td_" + c] = gb[c].cumsum() - t[c]
    t["gp"] = gb.cumcount()
    # priors
    full = t.groupby(["team", "season"]).agg(m=("fg3m", "sum"), a=("fg3a", "sum"))
    own_prev = (full.m / full.a).rename("p_prev").reset_index().assign(season=lambda x: x.season + 1)
    lg = t.groupby("season").agg(m=("fg3m", "sum"), a=("fg3a", "sum"))
    lg_prev = (lg.m / lg.a).rename("lg_prev").reset_index().assign(season=lambda x: x.season + 1)
    t = t.merge(own_prev, on=["team", "season"], how="left").merge(lg_prev, on="season", how="left")
    # 2015-16: league 3P% of all games strictly before this date
    s15 = t[t.season == 2015]
    daily = s15.groupby("date").agg(m=("fg3m", "sum"), a=("fg3a", "sum")).sort_index()
    cm, ca = daily.m.cumsum().shift(1), daily.a.cumsum().shift(1)
    lg_td = (cm / ca).rename("lg_td")
    t = t.merge(lg_td, left_on="date", right_index=True, how="left")
    first = t.season == 2015
    t.loc[first, "p_prev"] = t.loc[first, "lg_td"]
    t.loc[first, "lg_prev"] = t.loc[first, "lg_td"]
    t["p_prev"] = t.p_prev.fillna(t.lg_prev)
    s_off = (t.td_fg3m + K3 * t.p_prev) / (t.td_fg3a + K3)
    s_def = (t.td_o_fg3m + K3 * t.lg_prev) / (t.td_o_fg3a + K3)
    gpn = t.gp.replace(0, np.nan)
    t["luck_off"] = (t.td_fg3m - t.td_fg3a * s_off) / gpn * 3
    t["luck_def"] = (t.td_o_fg3m - t.td_o_fg3a * s_def) / gpn * 3
    t["raw3"] = t.td_fg3m / t.td_fg3a
    t["raw3_allowed"] = t.td_o_fg3m / t.td_o_fg3a
    h = t[t.is_home].set_index("GAME_ID")
    a = t[~t.is_home].set_index("GAME_ID")
    cols = ["luck_off", "luck_def", "gp", "raw3", "raw3_allowed", "team"]
    g = h[["season", "date"] + cols].add_prefix("h_").rename(columns={"h_season": "season", "h_date": "date"}).join(
        a[cols].add_prefix("a_"), how="inner")
    g["game_luck"] = 0.5 * (g.h_luck_off + g.a_luck_off + g.h_luck_def + g.a_luck_def)
    g["D"] = (g.h_luck_off - g.h_luck_def) - (g.a_luck_off - g.a_luck_def)
    g["gpmin"], g["gpmax"] = g[["h_gp", "a_gp"]].min(1), g[["h_gp", "a_gp"]].max(1)
    g = g[g.gpmin >= 1].copy()
    g["bucket"] = np.where(g.gpmax <= 10, "EARLY", "REST")
    return g.reset_index()


def cuts(g: pd.DataFrame) -> dict:
    out = {}
    for b in ("EARLY", "REST", "ALL"):
        x = g if b == "ALL" else g[g.bucket == b]
        out[b] = {v: [float(x[v].quantile(.2)), float(x[v].quantile(.8))] for v in ("game_luck", "D")}
    return out


def cmd_features() -> int:
    g = features()
    print(f"games {len(g)}  by bucket {g.bucket.value_counts().to_dict()}")
    print(g.groupby(["bucket"])[["game_luck", "D"]].describe().T.round(2))
    print("cuts", json.dumps(cuts(g), indent=1))
    print(g.groupby("season").game_luck.agg(["mean", "std", "size"]).round(2))
    return 0


# ---------------------------------------------------------------------------------------------------------- markets
def markets(g: pd.DataFrame) -> pd.DataFrame:
    m = pd.read_parquet(CB / "games.parquet")
    m = m[(m.stype == "Regular Season") & m.season.isin(DEV)]
    m = m[["GAME_ID", "eid", "spread", "total_line", "hpts", "apts", "margin", "line_src"]]
    d = g.merge(m, on="GAME_ID", how="left")
    d["total"] = d.hpts + d.apts
    rows = []
    for s, fn in OPEN_FILES.items():
        ln = pd.read_parquet(ESPN / fn)
        bk = ln[~ln.provider.isin(NOT_BOOKS)]
        o = bk[bk.provider.isin(["DraftKings", "ESPN BET"])].dropna(subset=["total_open"]).copy()
        o["pri"] = (o.provider != "DraftKings").astype(int)
        o = o.sort_values("pri").drop_duplicates("event_id")
        c = bk.groupby("event_id").total_close.median().rename("tot_close_med")
        o = o.merge(c, left_on="event_id", right_index=True, how="left")
        rows.append(o[["event_id", "provider", "total_open", "total_close", "tot_close_med", "over_odds_open",
                       "under_odds_open"]].rename(columns={"provider": "open_book", "total_close": "tot_close_same"}))
    o = pd.concat(rows)
    o["event_id"] = o.event_id.astype(str)
    d = d.merge(o, left_on="eid", right_on="event_id", how="left")
    w = pd.read_pickle(ROOT / "data" / "nba_simple_model" / "wf.pkl")[["event_id", "T2_strict"]]
    w["event_id"] = w.event_id.astype(str)
    d = d.merge(w.rename(columns={"event_id": "eid"}), on="eid", how="left")
    d["res_close"] = d.total - d.total_line
    d["res_open"] = d.total - d.total_open
    d["move"] = d.tot_close_same - d.total_open
    d["ats_res"] = d.margin - d.spread
    d["half"] = np.where(d.season <= 2018, "2015-19", "2020-25")
    return d


# ------------------------------------------------------------------------------------------------------------ stats
def slope(d, y, x, two_sided=False, fe=False):
    z = d.dropna(subset=[y, x])
    X = z[[x]].copy()
    if fe:
        X = X.join(pd.get_dummies(z.season, prefix="s", drop_first=True, dtype=float))
    f = sm.OLS(z[y], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(z.date)[0]})
    b, se = float(f.params[x]), float(f.bse[x])
    tt = b / se
    p = float(2 * stats.norm.sf(abs(tt))) if two_sided else float(stats.norm.cdf(tt))   # one-sided: slope < 0
    return {"n": int(len(z)), "slope": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se, "p": p}


def bet_frame(d, kind, line, cut):
    """kind 'tot': feature game_luck, low -> OVER (+1), high -> UNDER (-1). kind 'spr': feature D, high -> AWAY (-1)."""
    if kind == "tot":
        x = d.dropna(subset=["game_luck", line, "total"]).copy()
        side = np.where(x.game_luck <= cut[0], 1, np.where(x.game_luck >= cut[1], -1, 0))
        res = x.total - x[line]
    else:
        x = d.dropna(subset=["D", line, "margin"]).copy()
        side = np.where(x.D <= cut[0], 1, np.where(x.D >= cut[1], -1, 0))
        res = x.margin - x[line]
    x["side"] = side
    x = x[x.side != 0].copy()
    r = x.side * res[x.index]
    x["win"], x["push"] = (r > 0).astype(float), (r == 0)
    x["pnl"] = np.where(r > 0, WIN110, np.where(r == 0, 0.0, -1.0))
    return x


def roi_stats(x, seed=0):
    v = x.pnl.to_numpy(float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (4000, len(v)))].mean(1)
    nw = int(x.win[~x.push].sum()); nn = int((~x.push).sum())
    return {"n": int(len(v)), "roi": float(v.mean()), "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)),
            "p": float(np.mean(bs <= 0)), "hit": nw / nn if nn else float("nan"),
            "p_hit_vs_524": float(stats.binomtest(nw, nn, 0.5238, alternative="greater").pvalue) if nn else 1.0,
            "units": float(v.sum())}


def by_group(x, col, val):
    return {str(k): [round(float(v.mean()), 4), int(len(v))] for k, v in x.groupby(col)[val]}


def slope_groups(d, y, xv, col, two_sided=False):
    out = {}
    for k, z in d.groupby(col):
        if z[[y, xv]].dropna().shape[0] > 30:
            s = slope(z, y, xv, two_sided)
            out[str(k)] = [round(s["slope"], 4), s["n"]]
    return out


def bh(ps, q=.05):
    ps = np.asarray(ps); n = len(ps); o = np.argsort(ps)
    thr = q * (np.arange(1, n + 1)) / n
    ok = ps[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(n, bool); rej[o[:k]] = True
    return rej


# ---------------------------------------------------------------------------------------------------------- backtest
def cmd_backtest(ledger=False) -> int:
    g = features()
    C = cuts(g)
    d = markets(g)
    print(f"games {len(d)}; close total {d.total_line.notna().sum()}, spread {d.spread.notna().sum()}, "
          f"opener {d.total_open.notna().sum()} ({d.open_book.value_counts().to_dict()}), model {d.T2_strict.notna().sum()}")
    res = {"cuts": C, "trials": {}, "descriptive": {}}
    specs = []
    for tag, B in (("", "ALL"), ("T4-", "EARLY")):
        sub = d if B == "ALL" else d[d.bucket == B]
        cu = C[B]
        p1 = "T1" if not tag else "T4-1"; p2 = "T2" if not tag else "T4-2"; p3 = "T3" if not tag else "T4-3"
        specs += [(f"{p1}a", sub, "slope", ("res_close", "game_luck", False), "half"),
                  (f"{p1}b", sub, "bet", ("tot", "total_line", cu["game_luck"]), "half"),
                  (f"{p2}a", sub, "slope", ("res_open", "game_luck", False), "season"),
                  (f"{p2}b", sub, "bet", ("tot", "total_open", cu["game_luck"]), "season"),
                  (f"{p2}c", sub, "slope", ("move", "game_luck", True), "season"),
                  (f"{p3}a", sub, "slope", ("ats_res", "D", False), "half"),
                  (f"{p3}b", sub, "bet", ("spr", "spread", cu["D"]), "half")]
    assert len(specs) == N_TRIALS
    for name, sub, kind, args, halfcol in specs:
        if kind == "slope":
            y, xv, two = args
            r = slope(sub, y, xv, two)
            r["halves"] = slope_groups(sub, y, xv, halfcol, two)
            r["seasons"] = slope_groups(sub, y, xv, "season", two)
            sgn = -1 if not two else np.sign(r["slope"])
            r["seasons_pred_sign"] = f"{sum(np.sign(v[0]) == sgn for v in r['seasons'].values())}/{len(r['seasons'])}"
            r["halves_ok"] = all(np.sign(v[0]) == sgn for v in r["halves"].values()) and len(r["halves"]) == 2
            r["fe"] = slope(sub, y, xv, two, fe=True)
            r["metric"] = "slope"
            # quintile mean residual (descriptive)
            msg = (f"slope {r['slope']:+.3f} [{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] p {r['p']:.4f} n {r['n']} halves "
                   f"{r['halves']} seasons {r['seasons_pred_sign']} FE {r['fe']['slope']:+.3f} (p {r['fe']['p']:.3f})")
        else:
            k, line, cu = args
            x = bet_frame(sub, k, line, cu)
            r = roi_stats(x)
            r["halves"] = by_group(x, halfcol, "pnl")
            r["seasons"] = by_group(x, "season", "pnl")
            r["seasons_pred_sign"] = f"{sum(v[0] > 0 for v in r['seasons'].values())}/{len(r['seasons'])}"
            r["halves_ok"] = all(v[0] > 0 for v in r["halves"].values()) and len(r["halves"]) == 2
            r["by_side"] = by_group(x, "side", "pnl")
            r["metric"] = "roi_110"
            msg = (f"ROI {r['roi']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] p {r['p']:.4f} hit {r['hit']:.3f} "
                   f"(p vs .524 {r['p_hit_vs_524']:.3f}) n {r['n']} halves {r['halves']} seasons {r['seasons_pred_sign']} "
                   f"side {r['by_side']}")
        res["trials"][name] = r
        print(f"{name:7s} {msg}")
    names = list(res["trials"])
    ps = [res["trials"][n]["p"] for n in names]
    rej = bh(ps)
    n_led = sum(1 for _ in LEDGER.open()) if LEDGER.exists() else 0
    whole = 0.05 / (n_led + N_TRIALS)
    print(f"\nBonferroni-14 bar {0.05/N_TRIALS:.5f}; whole-ledger bar {whole:.6f}")
    for n, rj in zip(names, rej):
        r = res["trials"][n]
        r["bh_reject"] = bool(rj)
        r["bonf_pass"] = bool(r["p"] < 0.05 / N_TRIALS)
        good = (r["roi"] > 0) if r["metric"] == "roi_110" else (r["slope"] < 0 or n.endswith("c"))
        r["passed"] = bool(r["bonf_pass"] and r["halves_ok"] and good)
        print(f"  {n:7s} p {r['p']:.4f} BH {'rej' if rj else '-'} bonf {'PASS' if r['bonf_pass'] else '-'} "
              f"halves_ok {r['halves_ok']} -> {'PASS' if r['passed'] else 'fail'}")

    # ---------------- descriptive
    print("\nDESCRIPTIVE")
    D = res["descriptive"]
    rest = d[d.bucket == "REST"]
    D["rest_T1a"] = slope(rest, "res_close", "game_luck")
    D["rest_T1b"] = roi_stats(bet_frame(rest, "tot", "total_line", C["REST"]["game_luck"]))
    D["rest_T3a"] = slope(rest, "ats_res", "D")
    D["rest_T3b"] = roi_stats(bet_frame(rest, "spr", "spread", C["REST"]["D"]))
    for k in ("rest_T1a", "rest_T3a"):
        print(f"  {k}: slope {D[k]['slope']:+.3f} [{D[k]['ci_lo']:+.3f},{D[k]['ci_hi']:+.3f}] p {D[k]['p']:.3f} n {D[k]['n']}")
    for k in ("rest_T1b", "rest_T3b"):
        print(f"  {k}: ROI {D[k]['roi']:+.4f} hit {D[k]['hit']:.3f} p {D[k]['p']:.3f} n {D[k]['n']}")
    # pre-generation subset (2015-16..2021-22)
    pre = d[d.season <= 2021]
    for B in ("ALL", "EARLY"):
        z = pre if B == "ALL" else pre[pre.bucket == "EARLY"]
        s = slope(z, "res_close", "game_luck"); bt = roi_stats(bet_frame(z, "tot", "total_line", C[B]["game_luck"]))
        D[f"pre2022_{B}"] = {"slope": s, "bet": bt}
        print(f"  2015-22 {B}: totals slope {s['slope']:+.3f} (p {s['p']:.3f}, n {s['n']}); quintile bet ROI {bt['roi']:+.4f} "
              f"hit {bt['hit']:.3f} n {bt['n']}")
    # deciles: mean residual by luck decile (ALL and EARLY)
    for B in ("ALL", "EARLY"):
        z = (d if B == "ALL" else d[d.bucket == "EARLY"]).dropna(subset=["res_close", "game_luck"]).copy()
        z["dec"] = pd.qcut(z.game_luck, 10, labels=False)
        t = z.groupby("dec").agg(luck=("game_luck", "mean"), res=("res_close", "mean"), over=("res_close", lambda s: (s > 0).mean()), n=("res_close", "size"))
        D[f"deciles_{B}"] = t.round(3).to_dict("index")
        print(f"  deciles {B}:\n" + t.round(2).to_string())
        z["decD"] = pd.qcut(z.D, 10, labels=False)
        zz = z.dropna(subset=["ats_res"])
        t2 = zz.groupby("decD").agg(D=("D", "mean"), ats=("ats_res", "mean"), n=("ats_res", "size"))
        D[f"decilesD_{B}"] = t2.round(3).to_dict("index")
        print(f"  D deciles {B} (home ATS res):\n" + t2.round(2).to_string())
    # the post-hoc seed rule on the full DEV window: both teams 1-5 games, luck <= -4
    z = d[(d.gpmax <= 5)]
    for thr in (-4, -6, -8):
        y = z[z.game_luck <= thr].dropna(subset=["res_close"])
        D[f"seed_le{thr}"] = {"n": int(len(y)), "res": float(y.res_close.mean()), "over": float((y.res_close > 0).mean()),
                              "by_season": by_group(y, "season", "res_close")}
        print(f"  seed rule gp<=5 luck<={thr}: n {len(y)} res {y.res_close.mean():+.2f} over {(y.res_close > 0).mean():.3f} "
              f"seasons+ {sum(v[0] > 0 for v in D[f'seed_le{thr}']['by_season'].values())}/{len(D[f'seed_le{thr}']['by_season'])}")
    # in the model? (2023-24, 2024-25)
    z = d.dropna(subset=["res_close", "game_luck", "T2_strict", "total_line"]).copy()
    z["mgap"] = z.T2_strict - z.total_line
    f0 = sm.OLS(z.res_close, sm.add_constant(z[["game_luck"]])).fit(cov_type="HC1")
    f1 = sm.OLS(z.res_close, sm.add_constant(z[["game_luck", "mgap"]])).fit(cov_type="HC1")
    z["mres"] = z.total - z.T2_strict
    f2 = sm.OLS(z.mres, sm.add_constant(z[["game_luck"]])).fit(cov_type="HC1")
    D["model"] = {"n": int(len(z)), "luck_alone": [float(f0.params.game_luck), float(f0.bse.game_luck)],
                  "luck_given_model": [float(f1.params.game_luck), float(f1.bse.game_luck)],
                  "mgap_coef": [float(f1.params.mgap), float(f1.bse.mgap)],
                  "model_res_on_luck": [float(f2.params.game_luck), float(f2.bse.game_luck)],
                  "corr_luck_mgap": float(z.game_luck.corr(z.mgap))}
    print(f"  MODEL (n {len(z)}): res_close~luck {f0.params.game_luck:+.3f} ({f0.bse.game_luck:.3f}); + model gap -> luck "
          f"{f1.params.game_luck:+.3f} ({f1.bse.game_luck:.3f}), gap {f1.params.mgap:+.3f}; (final - model) ~ luck "
          f"{f2.params.game_luck:+.3f} ({f2.bse.game_luck:.3f}); corr(luck, model-close) {z.game_luck.corr(z.mgap):+.3f}")
    # L31 overlap: raw season 3P% for / allowed sums (z within season)
    z = d.dropna(subset=["res_close", "game_luck"]).copy()
    z["A"] = z.h_raw3 + z.a_raw3; z["Bz"] = z.h_raw3_allowed + z.a_raw3_allowed
    for c in ("A", "Bz"):
        z[c] = z.groupby("season")[c].transform(lambda s: (s - s.mean()) / s.std())
    for B in ("ALL", "EARLY", "REST"):
        zz = z if B == "ALL" else z[z.bucket == B]
        f = sm.OLS(zz.res_close, sm.add_constant(zz[["game_luck", "A", "Bz"]])).fit(cov_type="HC1")
        D[f"L31_{B}"] = {k: [float(f.params[k]), float(f.bse[k])] for k in ("game_luck", "A", "Bz")} | {"n": int(f.nobs)}
        print(f"  L31 overlap {B}: luck {f.params.game_luck:+.3f} ({f.bse.game_luck:.3f}) A {f.params.A:+.3f} ({f.bse.A:.3f}) "
              f"B {f.params.Bz:+.3f} ({f.bse.Bz:.3f}) n {int(f.nobs)}; corr(luck,A) {zz.game_luck.corr(zz.A):+.2f}")
    # opener -> close move of median-close vs same-book (sanity)
    out = ROOT / "results" / "nba_luck.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {out}")
    if ledger:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with LEDGER.open("a") as f:
            for i, n in enumerate(names):
                r = res["trials"][n]
                rec = {"ts": ts, "strategy": f"nba_luck_{n}", "sha": hashlib.sha256((DATASET + ":" + n).encode()).hexdigest()[:12],
                       "dataset": DATASET, "metric": r["metric"], "trial": n,
                       "hypothesis": f"pre-registered in scripts/nba_luck.py ({n})",
                       "source": "lead brief 2026-10-27: NBA market overreacts to small-sample 3P luck?", "status": "ok",
                       "trial_number": i + 1, "study_trials": N_TRIALS, "bonferroni_bar": 0.05 / N_TRIALS,
                       "bets": r["n"], "roi": r.get("roi", r.get("slope")), "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "p": r["p"], "per_fold": r["halves"], "passed": r["passed"]}
                f.write(json.dumps(rec, default=float) + "\n")
        print(f"appended {len(names)} trials to {LEDGER}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if not a or a[0] == "features":
        return cmd_features()
    if a[0] == "backtest":
        return cmd_backtest("--ledger" in a)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
