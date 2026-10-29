"""NBA early-season closing-price bias. Pre-registration: notes/nba_early_season_bias.md (read it first).

    ../.venv_quant/bin/python scripts/nba_early_season_bias.py [--ledger]

One run: builds point-in-time form (ET date < game date only), scores DEV (2021-22..2024-25), and scores SEALED
(2025-26 first 30 days) ONLY for trials that pass DEV. Prices = ESPN plain fields (= close), one book per event.
No orders, no network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import edge_math as em  # noqa: E402

ESPN = ROOT / "data" / "espn" / "nba"
OUT = ROOT / "results" / "nba_early_season_bias"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DEV = (2022, 2023, 2024, 2025)
H1, H2 = (2022, 2023), (2024, 2025)
SEALED = 2026
K = 5
BAR = 0.05 / K
NB = 10000
SEED = 2026
GP_MAX, GP_MIN_FORM, GP_SENS = 15, 3, 8


# ------------------------------------------------------------------------------------------------ data
def games(season: int) -> pd.DataFrame:
    g = pd.read_parquet(ESPN / f"games_{season}.parquet")
    g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL")].copy()
    g["event_id"] = g.event_id.astype(str)
    g["date"] = g.start_utc.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    g["margin"] = g.home_score - g.away_score
    return g.sort_values("start_utc").reset_index(drop=True)


def lines(season: int) -> pd.DataFrame:
    ln = pd.read_parquet(ESPN / f"lines_{season}.parquet")
    ln["event_id"] = ln.event_id.astype(str)
    ln = ln[ln.provider.isin(["DraftKings", "Draft Kings", "ESPN BET"])].dropna(subset=["home_ml", "away_ml"]).copy()
    ln["pri"] = (ln.provider == "ESPN BET").astype(int)
    ln = ln.sort_values("pri").drop_duplicates("event_id")
    return ln[["event_id", "provider", "home_ml", "away_ml", "spread", "home_spread_odds", "away_spread_odds"]]


def team_long(g: pd.DataFrame) -> pd.DataFrame:
    h = pd.DataFrame({"event_id": g.event_id, "date": g.date, "team": g.home, "opp": g.away, "m": g.margin})
    a = pd.DataFrame({"event_id": g.event_id, "date": g.date, "team": g.away, "opp": g.home, "m": -g.margin})
    return pd.concat([h, a], ignore_index=True)


def form_asof(tl: pd.DataFrame, d) -> pd.DataFrame:
    """gp, MOV, adjMOV per team using ONLY rows with date < d."""
    p = tl[tl.date < d]
    if p.empty:
        return pd.DataFrame(columns=["gp", "mov", "adj"])
    s = p.groupby("team").m.agg(gp="size", mov="mean")
    opp_mov = p.opp.map(s["mov"])
    s["adj"] = s["mov"] + opp_mov.groupby(p.team).mean()
    return s


def build(season: int) -> pd.DataFrame:
    g = games(season)
    tl = team_long(g)
    rows = []
    for d, gd in g.groupby("date"):
        f = form_asof(tl, d)
        for r in gd.itertuples():
            fh = f.loc[r.home] if r.home in f.index else None
            fa = f.loc[r.away] if r.away in f.index else None
            rows.append({"event_id": r.event_id, "date": d, "home": r.home, "away": r.away,
                         "gp_h": int(fh.gp) if fh is not None else 0, "gp_a": int(fa.gp) if fa is not None else 0,
                         "adj_h": fh.adj if fh is not None else np.nan, "adj_a": fa.adj if fa is not None else np.nan,
                         "mov_h": fh.mov if fh is not None else np.nan, "mov_a": fa.mov if fa is not None else np.nan})
    x = pd.DataFrame(rows)
    x["season"] = season
    x["day"] = (x.date - x.date.min()).dt.days
    # reputation = previous regular season MOV per game
    if (ESPN / f"games_{season-1}.parquet").exists() and season - 1 >= 2022:
        rep = team_long(games(season - 1)).groupby("team").m.mean()
        x["rep_h"], x["rep_a"] = x.home.map(rep), x.away.map(rep)
    else:
        x["rep_h"] = x["rep_a"] = np.nan
    x = x.merge(lines(season), on="event_id", how="left")
    return x, g, tl


def lookahead_test(season: int, x: pd.DataFrame, g: pd.DataFrame, n: int = 60) -> int:
    """Recompute form for sampled rows from a dataset TRUNCATED to games before the date; must match exactly."""
    rng = np.random.default_rng(1)
    bad = 0
    for i in rng.choice(len(x), n, replace=False):
        r = x.iloc[i]
        f = form_asof(team_long(g[g.date < r.date]), pd.Timestamp("2100-01-01"))
        for side in ("h", "a"):
            t = r["home" if side == "h" else "away"]
            v = f.loc[t].adj if t in f.index else np.nan
            if not ((np.isnan(v) and np.isnan(r[f"adj_{side}"])) or abs(v - r[f"adj_{side}"]) < 1e-9):
                bad += 1
    return bad


# ------------------------------------------------------------------------------------------------ scoring helpers
def ml_profit(odds, win):
    dec = np.array([em.american_to_decimal(o) for o in odds])
    return np.where(win, dec - 1, -1.0)


def pdv(a, b):
    return np.array([em.devig([u, v], "power")[0] for u, v in zip(a, b)])


def boot_mean(v, clus=None, seed=SEED):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    if clus is None:
        bs = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    else:
        c = pd.Series(v).groupby(np.asarray(clus))
        s, k = c.sum().to_numpy(), c.size().to_numpy()
        idx = rng.integers(0, len(s), (NB, len(s)))
        bs = s[idx].sum(1) / k[idx].sum(1)
    return bs


def summarize(b: pd.DataFrame, halves=(H1, H2)) -> dict:
    v = b.profit.to_numpy()
    if len(v) == 0:
        return {"bets": 0}
    bs = boot_mean(v)
    bd = boot_mean(v, b.date.astype(str).to_numpy())
    out = {"bets": int(len(v)), "units": float(v.sum()), "roi": float(v.mean()),
           "ci_lo": float(np.quantile(bs, .025)), "ci_hi": float(np.quantile(bs, .975)), "p": float(np.mean(bs <= 0)),
           "boot_mean": float(bs.mean()), "win_rate": float(b.win.mean()),
           "date_clustered": {"ci_lo": float(np.quantile(bd, .025)), "ci_hi": float(np.quantile(bd, .975)),
                              "p": float(np.mean(bd <= 0))},
           "by_season": {int(s): {"n": int(len(y)), "roi": float(y.profit.mean())} for s, y in b.groupby("season")}}
    for i, h in enumerate(halves, 1):
        y = b[b.season.isin(h)]
        out[f"half{i}"], out[f"half{i}_n"] = (float(y.profit.mean()) if len(y) else None), int(len(y))
    return out


def passed(s: dict, bar: float) -> bool:
    return s.get("bets", 0) > 0 and s["p"] < bar and (s["half1"] or -1) > 0 and (s["half2"] or -1) > 0


# ------------------------------------------------------------------------------------------------ trials
def t1(x, X):
    R = x.R
    b = x[(R >= X) | (R <= -X)].copy()
    home = b.R >= X
    return bet_ml(b, home)


def t2(x, Y):
    b = x[(x.G >= Y) | (x.G <= -Y)].copy()
    home = b.G <= -Y
    return bet_ml(b, home)


def t3(x):
    b = x[x.p_home < 0.5].copy()
    return bet_ml(b, pd.Series(True, index=b.index))


def bet_ml(b, home):
    b = b.copy()
    b["side"] = np.where(home, "home", "away")
    b["odds"] = np.where(home, b.home_ml, b.away_ml)
    b["win"] = np.where(home, b.homewin, ~b.homewin.astype(bool)).astype(bool)
    b["profit"] = ml_profit(b.odds, b.win)
    return b


def t4(x, actual_odds=False):
    b = x[x.spread.abs() >= 5].copy()
    home_dog = b.spread > 0
    cov = b.margin + b.spread  # >0 home covers, <0 away covers
    b["side"] = np.where(home_dog, "home", "away")
    b["push"] = cov == 0
    b["win"] = np.where(home_dog, cov > 0, cov < 0)
    if actual_odds:
        b["odds"] = np.where(home_dog, b.home_spread_odds, b.away_spread_odds)
        b = b.dropna(subset=["odds"])
    else:
        b["odds"] = -110.0
    b["profit"] = np.where(b.push, 0.0, ml_profit(b.odds, b.win))
    return b


def t5(x, nb=NB, clus=None):
    y = x.homewin.astype(int).to_numpy()
    lc = np.log(x.p_home / (1 - x.p_home)).to_numpy()
    F = x.F.to_numpy()
    X = sm.add_constant(np.column_stack([lc, F]))
    m = sm.Logit(y, X).fit(disp=0)
    rng = np.random.default_rng(SEED)
    co = []
    if clus is None:
        groups = [np.array([i]) for i in range(len(y))]
    else:
        groups = [np.asarray(ix) for ix in pd.Series(range(len(y))).groupby(np.asarray(clus)).groups.values()]
    for _ in range(nb):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        try:
            co.append(sm.Logit(y[idx], X[idx]).fit(disp=0, start_params=m.params).params[2])
        except Exception:
            pass
    co = np.array(co)
    ci = m.conf_int()
    return {"n": int(len(y)), "coef_close": float(m.params[1]), "coef_form": float(m.params[2]),
            "wald_ci": [float(ci[2][0]), float(ci[2][1])], "wald_p_two": float(m.pvalues[2]),
            "boot_ci": [float(np.quantile(co, .025)), float(np.quantile(co, .975))], "boot_p": float(np.mean(co <= 0)),
            "boot_n": int(len(co))}


# ------------------------------------------------------------------------------------------------ main
def prepare(seasons):
    xs, checks = [], {}
    for s in seasons:
        x, g, tl = build(s)
        checks[s] = {"lookahead_mismatches": lookahead_test(s, x, g), "games": int(len(g)),
                     "with_ml": int(x.home_ml.notna().sum()), "with_spread": int(x.spread.notna().sum()),
                     "providers": x.provider.value_counts().to_dict()}
        # spread sign check: favourite by spread == favourite by ML
        y = x.dropna(subset=["spread", "home_ml"])
        y = y[(y.spread != 0) & (y.home_ml != y.away_ml)]
        checks[s]["spread_ml_sign_agree"] = float(((y.spread < 0) == (y.home_ml < y.away_ml)).mean())
        xs.append(x)
    x = pd.concat(xs, ignore_index=True)
    x = x.dropna(subset=["home_ml", "away_ml"]).copy()
    x["p_home"] = pdv(x.home_ml, x.away_ml)
    x["M"] = -x.spread
    return x, checks


def add_outcomes(x):
    o = pd.concat([games(s)[["event_id", "margin"]] for s in sorted(x.season.unique())])
    x = x.merge(o, on="event_id", how="left")
    x = x[x.margin.notna() & (x.margin != 0)].copy()
    x["homewin"] = x.margin > 0
    return x


def windows(x, gmax):
    w = x[(x.gp_h <= gmax) & (x.gp_a <= gmax)].copy()
    Hs = w.groupby("season").M.mean()
    w["Hs"] = w.season.map(Hs)
    w["R"] = (w.adj_h - w.adj_a) - (w.M - w.Hs)
    w["G"] = (w.rep_h - w.adj_h) - (w.rep_a - w.adj_a)
    w["F"] = (w.adj_h - w.adj_a) / 10
    wf = w[(w.gp_h >= GP_MIN_FORM) & (w.gp_a >= GP_MIN_FORM)].dropna(subset=["adj_h", "adj_a"])
    return w, wf


def run_all(w, wf, X, Y, halves=(H1, H2)):
    res, bets = {}, {}
    b1 = t1(wf.dropna(subset=["R"]), X); bets["T1"] = b1; res["T1"] = summarize(b1, halves)
    b2 = t2(wf.dropna(subset=["G"]), Y); bets["T2"] = b2; res["T2"] = summarize(b2, halves)
    b3 = t3(w); bets["T3"] = b3; res["T3"] = summarize(b3, halves)
    w4 = w.dropna(subset=["spread"])
    b4 = t4(w4); bets["T4"] = b4; res["T4"] = summarize(b4, halves)
    res["T4"]["sens_actual_odds"] = summarize(t4(w4, True), halves)
    return res, bets


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    x, checks = prepare(DEV + (SEALED,))
    print("checks", json.dumps(checks, indent=1, default=str))
    assert all(c["lookahead_mismatches"] == 0 for c in checks.values()), "LOOKAHEAD TEST FAILED"
    dev = x[x.season.isin(DEV)]
    # thresholds frozen from DEV prices + form only (no outcomes joined yet)
    wd, wdf = windows(dev, GP_MAX)
    X = float(np.quantile(wdf.R.dropna().abs(), 2 / 3))
    Y = float(np.quantile(wdf.G.dropna().abs(), 2 / 3))
    print(f"frozen X (T1) {X:.3f} pts, Y (T2) {Y:.3f} pts; DEV window games {len(wd)}, form window {len(wdf)}")

    # ---- outcomes joined here
    dev = add_outcomes(dev)
    wd, wdf = windows(dev, GP_MAX)
    res, bets = run_all(wd, wdf, X, Y)
    res["T2"]["note"] = "T2 DEV = 2022-23..2024-25 (no REP for 2021-22); half1 = 2022-23 only"
    t5d = t5(wdf)
    t5d["date_clustered"] = {k: v for k, v in t5(wdf, nb=2000, clus=wdf.date.astype(str).to_numpy()).items()
                             if k in ("boot_ci", "boot_p", "boot_n")}
    t5d["half1"] = t5(wdf[wdf.season.isin(H1)], nb=200)["coef_form"]
    t5d["half2"] = t5(wdf[wdf.season.isin(H2)], nb=200)["coef_form"]
    res["T5"] = t5d
    for k in ("T1", "T2", "T3", "T4"):
        res[k]["passed"] = passed(res[k], BAR)
    res["T5"]["passed"] = bool(t5d["boot_p"] < BAR and t5d["half1"] > 0 and t5d["half2"] > 0)

    # sensitivity: <= 8 games
    s8, s8f = windows(dev, GP_SENS)
    sens8, _ = run_all(s8, s8f, X, Y)
    sens8["T5"] = t5(s8f, nb=1000)
    # descriptive: calibration of close in window vs rest of season
    full = dev.copy()
    full["early"] = (full.gp_h <= GP_MAX) & (full.gp_a <= GP_MAX)
    full["fav_p"] = np.maximum(full.p_home, 1 - full.p_home)
    full["fav_win"] = np.where(full.p_home >= .5, full.homewin, ~full.homewin)
    calib = {("early" if e else "rest"): {"n": int(len(y)), "mean_fav_p": float(y.fav_p.mean()),
                                         "fav_win_rate": float(y.fav_win.mean()),
                                         "home_p": float(y.p_home.mean()), "home_win": float(y.homewin.mean())}
             for e, y in full.groupby("early")}

    print(json.dumps({"dev": res, "sens_le8": sens8, "calib": calib}, indent=1, default=str))
    for k, b in bets.items():
        b.to_csv(OUT / f"bets_dev_{k}.csv", index=False)
    wdf.to_csv(OUT / "dev_form_window.csv", index=False)

    # ---- SEALED, only for DEV passes
    pas = [k for k in ("T1", "T2", "T3", "T4", "T5") if res[k]["passed"]]
    sealed = {}
    if pas:
        sx = x[x.season == SEALED]
        sx = sx[sx.day < 30]
        sx = add_outcomes(sx)
        ws, wsf = windows(sx, GP_MAX)
        sres, sbets = run_all(ws, wsf, X, Y, halves=((SEALED,), (SEALED,)))
        sres["T5"] = t5(wsf)
        sbar = 0.05 / len(pas)
        for k in pas:
            s = sres[k]
            if k == "T5":
                s["passed"] = bool(s["boot_p"] < sbar and s["coef_form"] > 0)
            else:
                s["passed"] = bool(s.get("bets", 0) > 0 and s["p"] < sbar and s["roi"] > 0)
                sbets[k].to_csv(OUT / f"bets_sealed_{k}.csv", index=False)
            sealed[k] = s
        print("SEALED", json.dumps(sealed, indent=1, default=str))
    else:
        print("SEALED not scored: no DEV pass")

    (OUT / "score.json").write_text(json.dumps({"X": X, "Y": Y, "checks": checks, "dev": res, "sens_le8": sens8,
                                                "calib": calib, "sealed": sealed}, indent=1, default=str))
    if a.ledger:
        sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        names = {"T1": "nesb_T1_form_vs_market_ml", "T2": "nesb_T2_reputation_fade_ml", "T3": "nesb_T3_home_dog_ml",
                 "T4": "nesb_T4_dog5_ats", "T5": "nesb_T5_form_coef_logit"}
        with LEDGER.open("a") as fh:
            for i, k in enumerate(("T1", "T2", "T3", "T4", "T5"), 1):
                r = {"ts": ts, "strategy": names[k], "sha": sha, "dataset": "nba_early_season_bias",
                     "hypothesis": "pre-registered in notes/nba_early_season_bias.md; early-season (both teams <=15 gp) "
                                   "NBA close slow to price current-season form", "source": "user hypothesis 10-29",
                     "status": "ok", "split": "DEV 2021-22..2024-25", "trial_number": i, "study_trials": K,
                     "bonferroni_bar": BAR, **{kk: vv for kk, vv in res[k].items()}}
                r["verdict"] = "PASS_DEV" if res[k]["passed"] else "FAIL"
                fh.write(json.dumps(r, default=str) + "\n")
            for k, s in sealed.items():
                r = {"ts": ts, "strategy": names[k] + "_sealed", "sha": sha, "dataset": "nba_early_season_bias",
                     "status": "ok", "split": "SEALED 2025-26 first 30 days", "study_trials": len(pas),
                     "bonferroni_bar": 0.05 / len(pas), **s, "verdict": "PASS_SEALED" if s["passed"] else "FAIL_SEALED"}
                fh.write(json.dumps(r, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
