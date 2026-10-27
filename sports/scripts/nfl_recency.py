"""NFL recency bias: does the line OVERREACT to last week's result? (DEV 2007-2019 only)

    ../.venv_quant/bin/python scripts/nfl_recency.py features   # feature build only, no join to this game's lines/outcome
    ../.venv_quant/bin/python scripts/nfl_recency.py run [--ledger]

PRE-REGISTRATION (written 2026-10-27, BEFORE any join of this-week outcomes to this-week lines; not edited after)
==============================================================================================================
Seasons: target games 2007-2019 (REG + POST) with SBR open AND close for the market tested (data/sbr_nfl/).
2020+ is NOT read (2020-25 spent on the simple-model refit, 2025+ lab-sealed, 2026 forward).
Close used for scoring = SBR close (paired with the SBR open). Previous-game lines = nflverse schedule closing
spread_line / total_line (data/nfl_schedule.csv; data/nfl_games.parquet only starts 2015 and is the same nflverse
close). Spread convention: line = expected HOME margin (+ = home favoured); margin = home - away.

Previous game ("last week") = the team's most recent earlier game in the SAME season (bye weeks skipped; week 1 has
none -> game dropped). Both teams need a previous game. No lookahead: only games that kicked off earlier.

Team features from its previous game (team perspective; spread_prev = expected margin FOR the team):
  cover(X)  = margin_prev - spread_prev                      (last-week cover margin)
  blow(X)   = sign(cover) if |cover| >= 14 else 0            (blowout cover / blowout non-cover)
  upset(X)  = 1 if spread_prev < 0 (closing dog) and margin_prev > 0 (won outright), else 0
  luck(X)   = margin_prev - epa_implied_prev                 (turnovers, fumble recoveries, return TDs, special teams)
  skill(X)  = epa_implied_prev - spread_prev                 (so cover = luck + skill exactly)
     epa_implied: OLS (no intercept) of home margin on [home-flag (0 at neutral sites), clean-EPA/play diff], where clean
     EPA/play = mean nflverse EPA over the team's offensive scrimmage plays (pass or rush, no kneels/spikes) that are
     NOT interceptions and NOT lost fumbles; differential = own offence minus opponent offence. The mapping is fit once
     on all 2006-2019 games' box scores (no market data; a fixed points-per-EPA scale, not a predictor of next week).
  totres(X) = points_prev - total_line_prev                  (last game's total vs its close)
  tblow(X)  = sign(totres) if |totres| >= 14 else 0
Game features (home minus away for sides, sum for totals): dCover, dLuck, dSkill, sTot; rule signals
  Fb = blow(h) - blow(a), U = upset(h) - upset(a), Fl = sign(luck) flags at |luck| >= 14 (h - a), Ft = tblow(h) + tblow(a).
Overreaction hypothesis = the line over-weights last week => NEGATIVE slopes of this week's residual on the feature.
All slope estimates are reported SIGN-FLIPPED so + = supports overreaction (fade), except T2b (see below).

COUNTED TRIALS (12; Bonferroni bar p < 0.05/12 = .00417; BH at q .05 over all 12 also reported; one-sided in the
predicted direction except T2b two-sided. Slopes: week-clustered OLS, pts per SD of the feature. Rules: per-bet ROI at
-110 (push 0), week-cluster bootstrap 5000; ROI > 0 <=> hit rate > 52.38%. Pass = p < bar AND effect > 0 AND same
sign in 2007-12 and 2013-19.)
  T1a slope (margin - CLOSE) ~ dCover
  T1b blowout-fade rule at CLOSE: Fb != 0 -> bet side -sign(Fb) (against the big coverer / on the big non-coverer)
  T1c upset-fade rule at CLOSE: U != 0 -> bet against the team that won outright as a dog
  T2a slope (margin - OPEN) ~ dCover
  T2b line move (close - open) ~ dCover, two-sided (+ = the line moves FURTHER toward last week's coverer after the open)
  T2c blowout-fade rule at the OPEN, -110
  T3a slope (margin - CLOSE) ~ dLuck, controlling dSkill  (fade luck: market prices luck as skill?)
  T3b slope (margin - OPEN)  ~ dLuck, controlling dSkill
  T3c luck-fade rule at CLOSE: Fl != 0 -> bet -sign(Fl)
  T4a slope (points - CLOSE total) ~ sTot
  T4b slope (points - OPEN total) ~ sTot
  T4c totals-fade rule at CLOSE: Ft != 0 -> bet UNDER if Ft > 0, OVER if Ft < 0
REPORTED, NOT COUNTED: hit rates per rule at open and close; season sign counts (13 seasons) for every trial; the
dSkill coefficient in T3; positive control: raw margin ~ dLuck + dSkill (luck should NOT persist, skill should);
line move (close-open) ~ dLuck + dSkill; T1/T3 slopes on the clipped residual (+-14); blow/upset each side separately.
Season halves: 2007-12 vs 2013-19.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nfl_simple_model as nsm  # noqa: E402

ROOT = nsm.ROOT
DATA = nsm.DATA
OUT_JSON = DATA / "lab" / "nfl_recency_results.json"
N_TRIALS = 12
BAR = 0.05 / N_TRIALS
BLOW = 14.0
SEASONS = list(range(2007, 2020))
PBP_SEASONS = list(range(2006, 2020))


# ------------------------------------------------------------------------------------------- features
def epa_games() -> pd.DataFrame:
    cols = ["game_id", "posteam", "pass", "rush", "qb_kneel", "qb_spike", "epa", "interception", "fumble_lost"]
    out = []
    for y in PBP_SEASONS:
        p = pd.read_parquet(DATA / "pbp" / f"play_by_play_{y}.parquet", columns=cols)
        p = p[p["epa"].notna() & ((p["pass"] == 1) | (p["rush"] == 1)) & (p["qb_kneel"] != 1)
              & (p["qb_spike"] != 1) & p["posteam"].notna()]
        p = p[(p["interception"] != 1) & (p["fumble_lost"] != 1)]
        p["posteam"] = p["posteam"].replace(nsm.TEAM_MAP)
        out.append(p.groupby(["game_id", "posteam"])["epa"].mean().rename("clean_epa").reset_index())
    return pd.concat(out, ignore_index=True)


def team_rows(sched: pd.DataFrame) -> pd.DataFrame:
    s = sched[sched["season"].between(2006, 2019) & sched["home_score"].notna()].copy()
    e = epa_games().set_index(["game_id", "posteam"])["clean_epa"]
    s["h_ce"] = [e.get((g, t), np.nan) for g, t in zip(s["game_id"], s["home_team"])]
    s["a_ce"] = [e.get((g, t), np.nan) for g, t in zip(s["game_id"], s["away_team"])]
    s["margin"] = s["home_score"] - s["away_score"]
    s["dce"] = s["h_ce"] - s["a_ce"]
    s["hflag"] = 1 - s["neutral"]
    fit = s.dropna(subset=["dce"])
    m = sm.OLS(fit["margin"], fit[["hflag", "dce"]]).fit()   # no intercept: neutral-site "home" is arbitrary
    s["epa_impl"] = m.params["hflag"] * s["hflag"] + m.params["dce"] * s["dce"]
    mapping = dict(hflag=float(m.params["hflag"]), per_epa=float(m.params["dce"]),
                   r2=float(m.rsquared), n=int(m.nobs))
    rows = []
    for side, opp, sg in (("home", "away", 1), ("away", "home", -1)):
        r = pd.DataFrame(dict(
            game_id=s["game_id"], season=s["season"], week=s["week"], gameday=s["gameday"], team=s[f"{side}_team"],
            margin=sg * s["margin"], spread=sg * s["spread_line"], points=s["home_score"] + s["away_score"],
            total_line=s["total_line"], epa_impl=sg * s["epa_impl"]))
        rows.append(r)
    T = pd.concat(rows, ignore_index=True)
    T["cover"] = T["margin"] - T["spread"]
    T["luck"] = T["margin"] - T["epa_impl"]
    T["skill"] = T["epa_impl"] - T["spread"]
    T["upset"] = ((T["spread"] < 0) & (T["margin"] > 0)).astype(int)
    T["totres"] = T["points"] - T["total_line"]
    T = T.sort_values(["team", "season", "gameday"])
    feats = ["cover", "luck", "skill", "upset", "totres"]
    prev = T.groupby(["team", "season"])[feats].shift(1)
    T[[f"p_{c}" for c in feats]] = prev.to_numpy()
    return T, mapping


def game_features(sched: pd.DataFrame):
    T, mapping = team_rows(sched)
    P = T.set_index(["game_id", "team"])
    g = sched[sched["season"].isin(SEASONS) & sched["home_score"].notna()][
        ["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"]].copy()
    for sd in ("home", "away"):
        k = sd[0]
        sub = P.loc[list(zip(g["game_id"], g[f"{sd}_team"]))]
        for c in ("cover", "luck", "skill", "upset", "totres"):
            g[f"{k}_{c}"] = sub[f"p_{c}"].to_numpy()
    g = g.dropna(subset=["h_cover", "a_cover", "h_luck", "a_luck", "h_totres", "a_totres"]).copy()
    g["margin"] = g["home_score"] - g["away_score"]
    g["points"] = g["home_score"] + g["away_score"]

    def flag(x):
        return np.where(np.abs(x) >= BLOW, np.sign(x), 0.0)
    g["dCover"] = g["h_cover"] - g["a_cover"]
    g["dLuck"] = g["h_luck"] - g["a_luck"]
    g["dSkill"] = g["h_skill"] - g["a_skill"]
    g["sTot"] = g["h_totres"] + g["a_totres"]
    g["Fb"] = flag(g["h_cover"]) - flag(g["a_cover"])
    g["U"] = g["h_upset"] - g["a_upset"]
    g["Fl"] = flag(g["h_luck"]) - flag(g["a_luck"])
    g["Ft"] = flag(g["h_totres"]) + flag(g["a_totres"])
    return g, mapping


def feature_report(g, mapping):
    d = dict(mapping=mapping, n_games=int(len(g)),
             sd={c: float(g[c].std()) for c in ("dCover", "dLuck", "dSkill", "sTot")},
             corr_luck_skill=float(np.corrcoef(g["dLuck"], g["dSkill"])[0, 1]),
             n_rule=dict(Fb=int((g["Fb"] != 0).sum()), U=int((g["U"] != 0).sum()), Fl=int((g["Fl"] != 0).sum()),
                         Ft=int((g["Ft"] != 0).sum())))
    return d


# ------------------------------------------------------------------------------------------- scoring
def clusters(d):
    return (d["season"].astype(str) + "_" + d["week"].astype(str)).to_numpy()


def summarise(v, d):
    ok = ~np.isnan(v)
    v, dd = v[ok], d[ok]
    p, lo, hi = nsm.boot_p(v, clusters(dd))
    seas = dd.assign(v=v).groupby("season")["v"].mean()
    return dict(n=int(len(v)), mean=float(v.mean()), lo=lo, hi=hi, p=p,
                early=float(v[(dd["season"] <= 2012).to_numpy()].mean()),
                late=float(v[(dd["season"] >= 2013).to_numpy()].mean()),
                seasons_pos=int((seas > 0).sum()), seasons_n=int(len(seas)))


def rule(d, side, mkt, line_tag):
    y = (d["margin"] if mkt == "spread" else d["points"]).to_numpy()
    line = d[f"{mkt}_{line_tag}"].to_numpy()
    roi = nsm.grade(side, line, y, 100 / 110)
    r = side * (y - line)
    hits = (r > 0).sum()
    dec = (r != 0).sum()
    out = summarise(roi, d)
    out.update(hit=float(hits / dec), wins=int(hits), decided=int(dec))
    return out


def slope(d, dep, x, controls=(), sign=-1.0, two_sided=False):
    cl = pd.Series(clusters(d)).astype("category").cat.codes.to_numpy()
    X = pd.DataFrame({x: (d[x] - d[x].mean()) / d[x].std()}, index=d.index)
    for c in controls:
        X[c] = (d[c] - d[c].mean()) / d[c].std()
    Xc = sm.add_constant(X)
    m = sm.OLS(dep.to_numpy(float), Xc).fit(cov_type="cluster", cov_kwds={"groups": cl})
    b, se = sign * float(m.params[x]), float(m.bse[x])
    p = float(2 * stats.norm.sf(abs(b / se))) if two_sided else float(stats.norm.sf(b / se))
    halves = []
    for msk in (d["season"] <= 2012, d["season"] >= 2013):
        halves.append(sign * float(sm.OLS(dep[msk].to_numpy(float), Xc[msk]).fit().params[x]))
    pos = 0
    for s in sorted(d["season"].unique()):
        msk = (d["season"] == s).to_numpy()
        pos += int(sign * sm.OLS(dep[msk].to_numpy(float), Xc[msk]).fit().params[x] > 0)
    out = dict(n=int(len(d)), mean=b, lo=b - 1.96 * se, hi=b + 1.96 * se, p=p, early=halves[0], late=halves[1],
               seasons_pos=pos, seasons_n=int(d["season"].nunique()))
    out["controls"] = {c: dict(b=float(m.params[c]), se=float(m.bse[c])) for c in controls}
    return out


def bh(ps, q=0.05):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    m = len(ps)
    thr = q * np.arange(1, m + 1) / m
    ok = ps[o] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[o[:k]] = True
    qv = np.minimum.accumulate((ps[o] * m / np.arange(1, m + 1))[::-1])[::-1]
    qq = np.empty(m)
    qq[o] = np.minimum(qv, 1)
    return rej, qq


def run(ledger: bool):
    sched = nsm.load_schedule()
    g, mapping = game_features(sched)
    feat = feature_report(g, mapping)
    # ------------------------------------------------------------- market join (first done here)
    sbr, _ = nsm.load_sbr(sched)
    M = g.merge(sbr, on="game_id", how="inner")
    assert M["season"].between(2007, 2019).all()
    S = M.dropna(subset=["spread_open", "spread_close"]).copy()
    T = M.dropna(subset=["total_open", "total_close"]).copy()
    trials, extra = [], {}
    rc, ro = S["margin"] - S["spread_close"], S["margin"] - S["spread_open"]
    trials.append(dict(trial="T1a_slope_close_dCover", **slope(S, rc, "dCover")))
    b = S[S["Fb"] != 0]
    trials.append(dict(trial="T1b_blowfade_close", **rule(b, -np.sign(b["Fb"].to_numpy()), "spread", "close")))
    u = S[S["U"] != 0]
    trials.append(dict(trial="T1c_upsetfade_close", **rule(u, -np.sign(u["U"].to_numpy()), "spread", "close")))
    trials.append(dict(trial="T2a_slope_open_dCover", **slope(S, ro, "dCover")))
    trials.append(dict(trial="T2b_move_dCover_2s", **slope(S, S["spread_close"] - S["spread_open"], "dCover",
                                                          sign=1.0, two_sided=True)))
    trials.append(dict(trial="T2c_blowfade_open", **rule(b, -np.sign(b["Fb"].to_numpy()), "spread", "open")))
    trials.append(dict(trial="T3a_slope_close_dLuck", **slope(S, rc, "dLuck", ("dSkill",))))
    trials.append(dict(trial="T3b_slope_open_dLuck", **slope(S, ro, "dLuck", ("dSkill",))))
    lk = S[S["Fl"] != 0]
    trials.append(dict(trial="T3c_luckfade_close", **rule(lk, -np.sign(lk["Fl"].to_numpy()), "spread", "close")))
    trials.append(dict(trial="T4a_slope_close_sTot", **slope(T, T["points"] - T["total_close"], "sTot")))
    trials.append(dict(trial="T4b_slope_open_sTot", **slope(T, T["points"] - T["total_open"], "sTot")))
    t = T[T["Ft"] != 0]
    trials.append(dict(trial="T4c_totfade_close", **rule(t, -np.sign(t["Ft"].to_numpy()), "total", "close")))
    for tr in trials:
        tr["passed"] = bool(tr["p"] < BAR and tr["mean"] > 0 and tr["early"] > 0 and tr["late"] > 0)
    rej, qv = bh([tr["p"] for tr in trials])
    for tr, r_, q_ in zip(trials, rej, qv):
        tr["bh_reject"], tr["bh_q"] = bool(r_), float(q_)
    # ------------------------------------------------------------- reported, not counted
    extra["rules_other_line"] = {
        "T1b_blowfade_open_hit": rule(b, -np.sign(b["Fb"].to_numpy()), "spread", "open"),
        "T1c_upsetfade_open": rule(u, -np.sign(u["U"].to_numpy()), "spread", "open"),
        "T3c_luckfade_open": rule(lk, -np.sign(lk["Fl"].to_numpy()), "spread", "open"),
        "T4c_totfade_open": rule(t, -np.sign(t["Ft"].to_numpy()), "total", "open"),
        "blowfade_close_bigcover_only": rule(b[b["Fb"].abs() == 1], -np.sign(b[b["Fb"].abs() == 1]["Fb"].to_numpy()),
                                              "spread", "close"),
    }
    # one-sided-flag subsets: fade a big coverer (opponent unflagged) vs back a big non-coverer
    hb = np.where(np.abs(S["h_cover"]) >= BLOW, np.sign(S["h_cover"]), 0)
    ab = np.where(np.abs(S["a_cover"]) >= BLOW, np.sign(S["a_cover"]), 0)
    S["_hb"], S["_ab"] = hb, ab
    pos = S[((S["_hb"] == 1) & (S["_ab"] != 1)) | ((S["_ab"] == 1) & (S["_hb"] != 1))]
    side_pos = np.where(pos["_hb"] == 1, -1.0, 1.0)
    neg = S[((S["_hb"] == -1) & (S["_ab"] != -1)) | ((S["_ab"] == -1) & (S["_hb"] != -1))]
    side_neg = np.where(neg["_hb"] == -1, 1.0, -1.0)
    extra["fade_big_coverer_close"] = rule(pos, side_pos, "spread", "close")
    extra["back_big_noncoverer_close"] = rule(neg, side_neg, "spread", "close")
    extra["T3_skill_coef_close"] = trials[6]["controls"]
    extra["T3_skill_coef_open"] = trials[7]["controls"]
    # positive control: does luck / skill persist in RAW margin?
    cl = pd.Series(clusters(S)).astype("category").cat.codes.to_numpy()
    X = sm.add_constant(pd.DataFrame({c: (S[c] - S[c].mean()) / S[c].std() for c in ("dLuck", "dSkill")}))
    for name, dep in (("raw_margin", S["margin"]), ("close_spread", S["spread_close"]),
                      ("open_spread", S["spread_open"]), ("move_close_minus_open", S["spread_close"] - S["spread_open"]),
                      ("resid_close", rc), ("resid_open", ro)):
        m = sm.OLS(dep.to_numpy(float), X).fit(cov_type="cluster", cov_kwds={"groups": cl})
        extra[f"luck_skill_{name}"] = {c: dict(b=float(m.params[c]), se=float(m.bse[c])) for c in ("dLuck", "dSkill")}
    m = sm.OLS(rc.to_numpy(float), sm.add_constant(pd.DataFrame({"dCover": S["dCover"]}))).fit(
        cov_type="cluster", cov_kwds={"groups": cl})
    extra["T1a_raw_pts_per_pt"] = dict(b=float(m.params["dCover"]), se=float(m.bse["dCover"]))
    extra["T1a_clipped14"] = slope(S, rc.clip(-14, 14), "dCover")
    extra["T3a_clipped14"] = slope(S, rc.clip(-14, 14), "dLuck", ("dSkill",))
    extra["T4_move_sTot"] = slope(T, T["total_close"] - T["total_open"], "sTot", sign=1.0, two_sided=True)
    extra["n"] = dict(spread_games=int(len(S)), total_games=int(len(T)))
    res = dict(ts=datetime.now(timezone.utc).isoformat(), bar=BAR, features=feat, trials=trials, extra=extra)
    OUT_JSON.write_text(json.dumps(res, indent=1, default=float))
    for tr in trials:
        hit = f" hit={tr['hit']:.3f} ({tr['wins']}/{tr['decided']})" if "hit" in tr else ""
        print(f"{tr['trial']:26s} n={tr['n']:5d} eff={tr['mean']:+.4f} [{tr['lo']:+.4f},{tr['hi']:+.4f}] p={tr['p']:.4f} "
              f"q={tr['bh_q']:.3f} halves {tr['early']:+.3f}/{tr['late']:+.3f} seasons+ {tr['seasons_pos']}/"
              f"{tr['seasons_n']}{hit} pass={tr['passed']}")
    print(json.dumps(extra, indent=1, default=float))
    print(json.dumps(feat, indent=1))
    if ledger:
        with nsm.LEDGER.open("a") as fh:
            for i, tr in enumerate(trials, 1):
                fh.write(json.dumps(dict(
                    ts=res["ts"], strategy=f"nfl_recency_{tr['trial']}", dataset="nfl_recency",
                    metric="roi" if ("fade" in tr["trial"]) else "slope_pts_per_sd",
                    trial=tr["trial"], hypothesis="pre-registered in scripts/nfl_recency.py docstring",
                    source="lead brief 2026-10-27: NFL line overreacts to last week's result (recency bias)?",
                    status="ok", trial_number=i, study_trials=N_TRIALS, bonferroni_bar=BAR, passed=tr["passed"],
                    bh_q=tr["bh_q"], bets=tr["n"], mean=tr["mean"], ci_lo=tr["lo"], ci_hi=tr["hi"], p=tr["p"],
                    hit=tr.get("hit"), per_half={"2007-12": tr["early"], "2013-19": tr["late"]},
                    seasons_pos=f"{tr['seasons_pos']}/{tr['seasons_n']}")) + "\n")
        print("appended", len(trials), "trials to", nsm.LEDGER)


def features_only():
    g, mapping = game_features(nsm.load_schedule())
    print(json.dumps(feature_report(g, mapping), indent=1))
    print(g[["dCover", "dLuck", "dSkill", "sTot"]].describe().round(2))
    print("check cover = luck + skill:", float((g["h_cover"] - g["h_luck"] - g["h_skill"]).abs().max()))


if __name__ == "__main__":
    if sys.argv[1:2] == ["features"]:
        features_only()
    else:
        run("--ledger" in sys.argv)
