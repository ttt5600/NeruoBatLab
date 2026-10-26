"""Two NFL leads from nfl_comebacks.py re-tested against the OPENER (DEV 2010-2019 only).

    ../.venv_quant/bin/python scripts/nfl_leads_open.py [--ledger]

PRE-REGISTRATION (written 2026-10-25, BEFORE this script joined any market line; nothing below was changed after)
==============================================================================================================
Seasons: test games 2010-2019 (REG + POST), the walk-forward test seasons of nfl_simple_model.py `dev`.
2020+ is NOT read (2020-25 was used for the live refit; 2026 is forward). Lines: SBR open/close 2010-19
(data/sbr_nfl/), nflverse schedule close prices (spread/total odds) for no-vig at the close.
Population: games where BOTH teams have >= 4 prior games this season (the comebacks lead's MIN_GP) and both
SBR open and close exist for the market being tested.

Features (pre-game, earlier games only; the SAME shrunk estimator as nfl_simple_model.team_features:
est = (sum_this_season + K*prior) / (n + K), prior = league + 0.5*(team last season - league), K 250 db / 200 runs)
  dEPA(team) = (shrunk pass-D EPA/dropback allowed + shrunk run-D EPA/designed-run allowed) / 2.  HIGHER = WORSE D.
  oEPA(team) = (shrunk pass-O EPA/dropback + shrunk run-O EPA/designed-run) / 2.                 HIGHER = BETTER O.
  Normalisation is CAUSAL and cross-sectional: z within the same (season, week) across every team playing that
  week (its current pre-game value); tercile = rank percentile within the same (season, week).

Lead BD ("two bad defences go UNDER"; comebacks: both-D EPA sum -> total resid vs close -0.79 pts/SD, p .0049)
  Rule BD: both teams' dEPA in the worst within-week tercile (pct rank > 2/3)  ->  bet UNDER the OPEN total.
  Linear BD: S = z_h(dEPA) + z_a(dEPA), then standardised over the sample; prediction: negative slope.
Lead OE ("market under-weights season-to-date offence EPA"; comebacks: +0.56 pts/SD ATS vs close, p .04)
  Rule OE: D = z_h(oEPA) - z_a(oEPA); |D| >= 1.0  ->  bet the better-offence side ATS at the OPEN spread.
  Linear OE: D standardised over the sample; prediction: positive slope on home (margin - line).

COUNTED TRIALS (10; bar p < 0.05/10 = 0.005; one-sided in the predicted direction; rule metrics use a
week-cluster bootstrap (5000), slopes use week-clustered OLS SEs)
  L01 BD rule CLV vs opener (pts toward the under: open - close)
  L02 BD rule ROI at the OPEN total, -110 (push 0)
  L03 BD rule ROI of the UNDER at the CLOSE (SBR close), -110      [replication of the close lead with this feature]
  L04 BD linear slope: (points - open total) ~ S, pts/SD (sign flipped so + = supports)
  L05 BD incremental: (points - open) ~ S + (modelC_total - open)  -> coefficient on S (flipped)
  L06 OE rule CLV vs opener (pts toward the bet side)
  L07 OE rule ROI ATS at the OPEN spread, -110
  L08 OE rule ROI ATS at the CLOSE (SBR close), -110
  L09 OE linear slope: (margin - open spread) ~ D, pts/SD
  L10 OE incremental: (margin - open) ~ D + (modelB_spread - open)  -> coefficient on D
  Model predictions = nfl_simple_model walk-forward (test season s fit on 2007..s-1), total C / spread B
  (the live picks). Pass = p < bar AND effect > 0 AND same sign in 2010-14 and 2015-19.
REPORTED, NOT COUNTED: no-vig ROI at the close using nflverse close odds (proportional de-vig, graded at the
nflverse close line); even-money ROI at the open (SBR has no open prices); win rate vs close; the same incremental
regressions vs the CLOSE and for CLV ((close-open) ~ feature + (model-open)); rule overlap with the model's own
bets; the comebacks-style unshrunk version is NOT re-run (that is the discovery data).
CAVEAT stated up front: 2010-2019 is the same data on which both leads were found (vs the close). L03/L08 are
therefore NOT independent replications; L01/L02/L06/L07 (vs the opener) ask a new question of the same games.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfl_simple_model as nsm  # noqa: E402

ROOT = nsm.ROOT
OUT_JSON = ROOT / "data" / "lab" / "nfl_leads_open_results.json"
N_TRIALS = 10
BAR = 0.05 / N_TRIALS
MIN_GP = 4
OE_T = 1.0
TEST = list(range(2010, 2020))


def model_preds() -> pd.DataFrame:
    g, _ = nsm.build_dev()
    out = []
    for s in TEST:
        tr = g[(g["season"] >= nsm.FIRST_FIT) & (g["season"] < s)]
        te = g[g["season"] == s].copy()
        cs, ct = nsm.MODELS["B"]
        te["spread_B"] = nsm.fit_predict(tr, te, cs, "margin", True)[0]
        cs, ct = nsm.MODELS["C"]
        te["total_C"] = nsm.fit_predict(tr, te, ct, "points", False)[0]
        out.append(te)
    P = pd.concat(out, ignore_index=True)
    assert P["season"].between(2010, 2019).all()
    return P


def add_features(P: pd.DataFrame) -> pd.DataFrame:
    P = P.copy()
    P["h_oEPA"] = (P["h_o_pass"] + P["h_o_run"]) / 2
    P["a_oEPA"] = (P["a_o_pass"] + P["a_o_run"]) / 2
    P["h_dEPA"] = (P["h_d_pass"] + P["h_d_run"]) / 2
    P["a_dEPA"] = (P["a_d_pass"] + P["a_d_run"]) / 2
    # long table of every team playing in each (season, week): pre-game values, causal cross-section
    L = pd.concat([P[["game_id", "season", "week", f"{s}_oEPA", f"{s}_dEPA"]].set_axis(
        ["game_id", "season", "week", "oEPA", "dEPA"], axis=1).assign(side=s) for s in "ha"])
    grp = L.groupby(["season", "week"])
    for c in ("oEPA", "dEPA"):
        L[f"z_{c}"] = (L[c] - grp[c].transform("mean")) / grp[c].transform("std")
        L[f"pct_{c}"] = grp[c].rank(pct=True)
    for s in "ha":
        x = L[L["side"] == s].set_index("game_id")
        for c in ("z_oEPA", "z_dEPA", "pct_dEPA"):
            P[f"{s}_{c}"] = P["game_id"].map(x[c])
    P["BD_rule"] = (P["h_pct_dEPA"] > 2 / 3) & (P["a_pct_dEPA"] > 2 / 3)
    P["S_raw"] = P["h_z_dEPA"] + P["a_z_dEPA"]
    P["D_raw"] = P["h_z_oEPA"] - P["a_z_oEPA"]
    P["gp_min"] = np.minimum(P["h_gp"], P["a_gp"])
    return P


def nflverse_close(sched):
    c = sched[["game_id", "spread_line", "total_line", "home_spread_odds", "away_spread_odds", "over_odds",
               "under_odds"]].copy()

    def imp(o):
        o = o.astype(float)
        return np.where(o < 0, -o / (-o + 100), 100 / (o + 100))
    for a, b, tag in (("home_spread_odds", "away_spread_odds", "home"), ("over_odds", "under_odds", "over")):
        pa, pb = imp(c[a]), imp(c[b])
        c[f"nv_{tag}"] = pa / (pa + pb)
    return c.rename(columns={"spread_line": "nvl_spread", "total_line": "nvl_total"})


def rule_metrics(b: pd.DataFrame, side: np.ndarray, mkt: str):
    """side: +1 home/over, -1 away/under. Returns dict of per-bet value arrays."""
    y = (b["margin"] if mkt == "spread" else b["points"]).to_numpy()
    o, c = b[f"{mkt}_open"].to_numpy(), b[f"{mkt}_close"].to_numpy()
    nvl = b["nvl_spread" if mkt == "spread" else "nvl_total"].to_numpy()
    nvp = b["nv_home" if mkt == "spread" else "nv_over"].to_numpy()
    p_side = np.where(side > 0, nvp, 1 - nvp)
    r_nv = side * (y - nvl)
    novig_close = np.where(r_nv > 0, (1 - p_side) / p_side, np.where(r_nv < 0, -1.0, 0.0))
    return dict(clv=side * (c - o), roi_open=nsm.grade(side, o, y, 100 / 110), roi_close=nsm.grade(side, c, y, 100 / 110),
                roi_open_evens=nsm.grade(side, o, y, 1.0), novig_close=novig_close,
                win_close=np.where(side * (y - c) > 0, 1.0, np.where(side * (y - c) < 0, 0.0, np.nan)),
                win_open=np.where(side * (y - o) > 0, 1.0, np.where(side * (y - o) < 0, 0.0, np.nan)))


def summarise(v: np.ndarray, b: pd.DataFrame):
    ok = ~np.isnan(v)
    v, bb = v[ok], b[ok]
    cl = (bb["season"].astype(str) + "_" + bb["week"].astype(str)).to_numpy()
    p, lo, hi = nsm.boot_p(v, cl)
    return dict(n=int(len(v)), mean=float(v.mean()), lo=lo, hi=hi, p=p, sd=float(v.std()),
                early=float(v[(bb["season"] <= 2014).to_numpy()].mean()),
                late=float(v[(bb["season"] >= 2015).to_numpy()].mean()))


def slope(d: pd.DataFrame, dep: pd.Series, x: str, controls=(), sign=1.0):
    cl = (d["season"].astype(str) + "_" + d["week"].astype(str)).astype("category").cat.codes
    X = pd.DataFrame({x: (d[x] - d[x].mean()) / d[x].std()}, index=d.index)
    for c in controls:
        X[c] = d[c]
    m = sm.OLS(dep.to_numpy(float), sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": cl})
    b, se = sign * float(m.params[x]), float(m.bse[x])
    from scipy import stats
    halves = []
    for msk in (d["season"] <= 2014, d["season"] >= 2015):
        mh = sm.OLS(dep[msk].to_numpy(float), sm.add_constant(X[msk])).fit()
        halves.append(sign * float(mh.params[x]))
    out = dict(n=int(len(d)), mean=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=float(stats.norm.sf(b / se)), sd=np.nan,
               early=halves[0], late=halves[1])
    out["controls"] = {c: dict(b=float(m.params[c]), se=float(m.bse[c])) for c in controls}
    return out


def main(ledger: bool):
    P = add_features(model_preds())
    P = P[P["gp_min"] >= MIN_GP].copy()
    feat_summary = dict(n_games_feat=int(len(P)), n_BD_rule=int(P["BD_rule"].sum()),
                        n_OE_rule=int((P["D_raw"].abs() >= OE_T).sum()))
    # ------------------------------------------------------------- market join (first done after the docstring)
    sched = nsm.load_schedule()
    sbr, _ = nsm.load_sbr(sched)
    M = P.merge(sbr, on="game_id", how="inner").merge(nflverse_close(sched), on="game_id", how="left")
    assert M["season"].between(2010, 2019).all()
    T = M.dropna(subset=["total_open", "total_close"]).copy()
    S = M.dropna(subset=["spread_open", "spread_close"]).copy()
    T["S"] = T["S_raw"]
    S["D"] = S["D_raw"]
    T["gapC"] = T["total_C"] - T["total_open"]
    S["gapB"] = S["spread_B"] - S["spread_open"]
    T["gapC_close"] = T["total_C"] - T["total_close"]
    S["gapB_close"] = S["spread_B"] - S["spread_close"]

    trials, extra = [], {}
    # ---- BD
    bd = T[T["BD_rule"]]
    rm = rule_metrics(bd, -np.ones(len(bd)), "total")
    trials.append(dict(trial="L01_BD_clv_open", **summarise(rm["clv"], bd)))
    trials.append(dict(trial="L02_BD_roi_open110", **summarise(rm["roi_open"], bd)))
    trials.append(dict(trial="L03_BD_roi_close110", **summarise(rm["roi_close"], bd)))
    trials.append(dict(trial="L04_BD_slope_vs_open", **slope(T, T["points"] - T["total_open"], "S", sign=-1)))
    trials.append(dict(trial="L05_BD_incremental_modelC", **slope(T, T["points"] - T["total_open"], "S", ("gapC",), sign=-1)))
    extra["BD"] = dict(
        n_rule=int(len(bd)),
        win_open=float(np.nanmean(rm["win_open"])), win_close=float(np.nanmean(rm["win_close"])),
        roi_open_evens=summarise(rm["roi_open_evens"], bd), novig_close=summarise(rm["novig_close"], bd),
        mean_open=float(bd["total_open"].mean()), mean_close=float(bd["total_close"].mean()),
        mean_points=float(bd["points"].mean()), mean_modelC=float(bd["total_C"].mean()),
        model_over_share=float((bd["gapC"] > 0).mean()),
        model_bets_under_share=float((bd["gapC"] <= -nsm.TOTAL_T).mean()),
        model_bets_over_share=float((bd["gapC"] >= nsm.TOTAL_T).mean()),
        slope_vs_close=slope(T, T["points"] - T["total_close"], "S", sign=-1),
        slope_clv=slope(T, T["total_close"] - T["total_open"], "S", sign=-1),
        incr_vs_close=slope(T, T["points"] - T["total_close"], "S", ("gapC_close",), sign=-1),
        incr_clv=slope(T, T["total_close"] - T["total_open"], "S", ("gapC",), sign=-1),
        model_resid=slope(T, T["points"] - T["total_C"], "S", sign=-1),
        corr_S_gapC=float(np.corrcoef(T["S"], T["gapC"])[0, 1]),
        n_total_games=int(len(T)))
    # ---- OE
    oe = S[S["D"].abs() >= OE_T]
    side = np.sign(oe["D"].to_numpy())
    rm = rule_metrics(oe, side, "spread")
    trials.append(dict(trial="L06_OE_clv_open", **summarise(rm["clv"], oe)))
    trials.append(dict(trial="L07_OE_roi_open110", **summarise(rm["roi_open"], oe)))
    trials.append(dict(trial="L08_OE_roi_close110", **summarise(rm["roi_close"], oe)))
    trials.append(dict(trial="L09_OE_slope_vs_open", **slope(S, S["margin"] - S["spread_open"], "D")))
    trials.append(dict(trial="L10_OE_incremental_modelB", **slope(S, S["margin"] - S["spread_open"], "D", ("gapB",))))
    mside = np.sign(oe["gapB"].to_numpy())
    extra["OE"] = dict(
        n_rule=int(len(oe)), home_share=float((side > 0).mean()),
        win_open=float(np.nanmean(rm["win_open"])), win_close=float(np.nanmean(rm["win_close"])),
        roi_open_evens=summarise(rm["roi_open_evens"], oe), novig_close=summarise(rm["novig_close"], oe),
        model_agrees_share=float((mside == side).mean()),
        model_bets_same_side_share=float(((oe["gapB"].abs() >= nsm.SPREAD_T) & (mside == side)).mean()),
        slope_vs_close=slope(S, S["margin"] - S["spread_close"], "D"),
        slope_clv=slope(S, S["spread_close"] - S["spread_open"], "D"),
        incr_vs_close=slope(S, S["margin"] - S["spread_close"], "D", ("gapB_close",)),
        incr_clv=slope(S, S["spread_close"] - S["spread_open"], "D", ("gapB",)),
        model_resid=slope(S, S["margin"] - S["spread_B"], "D"),
        corr_D_gapB=float(np.corrcoef(S["D"], S["gapB"])[0, 1]),
        n_spread_games=int(len(S)))
    for t in trials:
        t["passed"] = bool(t["p"] < BAR and t["mean"] > 0 and t["early"] > 0 and t["late"] > 0)
    res = dict(ts=datetime.now(timezone.utc).isoformat(), bar=BAR, features=feat_summary, trials=trials, extra=extra)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    for t in trials:
        print(f"{t['trial']:30s} n={t['n']:5d} mean={t['mean']:+.4f} [{t['lo']:+.4f},{t['hi']:+.4f}] p={t['p']:.4f} "
              f"halves {t['early']:+.3f}/{t['late']:+.3f} pass={t['passed']}")
    print(json.dumps(extra, indent=1, default=float))
    print(json.dumps(feat_summary))
    if ledger:
        with nsm.LEDGER.open("a") as fh:
            for i, t in enumerate(trials, 1):
                fh.write(json.dumps(dict(
                    ts=res["ts"], strategy=f"nfl_leads_open_{t['trial']}", dataset="nfl_leads_open",
                    metric="clv" if "clv" in t["trial"] else ("roi" if "roi" in t["trial"] else "slope_pts_per_sd"),
                    hypothesis="pre-registered in scripts/nfl_leads_open.py docstring",
                    source="lead 2026-10-25: comebacks leads (bad-D unders, offence EPA ATS) vs the opener",
                    status="ok", trial_number=i, bonferroni_bar=BAR, passed=t["passed"], bets=t["n"],
                    mean=t["mean"], ci_lo=t["lo"], ci_hi=t["hi"], p=t["p"],
                    per_half={"2010-14": t["early"], "2015-19": t["late"]})) + "\n")
        print("appended", len(trials), "trials to", nsm.LEDGER)


if __name__ == "__main__":
    main("--ledger" in sys.argv)
