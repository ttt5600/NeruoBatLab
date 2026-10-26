"""Simple college-football (FBS) model scored against the OPENING line. DEV ONLY. No orders.

    python scripts/cfb_simple_model.py model          # fit ratings walk-forward; outcome-only diagnostics (no market)
    python scripts/cfb_simple_model.py dev [--ledger] # join SBR openers/closers and score the pre-registered trials

PRE-REGISTERED 2026-10-25, written BEFORE any market column was joined or looked at.

DATA / SPLITS
  * CFB is NOT in the frozen MANIFEST (sealed = nba/nhl/mlb/nfl only), so per the brief: CFB 2025+ is treated as
    SEALED and is never read. data/espn/cfb/*_2025.parquet are not opened by this script.
  * Results + markets: sportsbookreviewsonline (SBR) NCAAF archive, seasons 2007-2022, saved raw to
    data/sbr_cfb/cfb_odds_<season>.html. Columns used: Date (MMDD), Rot, VH, Team, Final, Open, Close.
    The 2022 page stops on 2022-11-15 (partial season). SBR's "Open"/"Close" book and timestamp are undocumented.
    SBR gives no spread/total PRICES -> -110 both sides is ASSUMED (a stated limitation, not data).
  * ESPN data/espn/cfb lines: opener columns are empty in 2022, ~450 games in 2023, ~530 in 2024 (ESPN BET only)
    and the 2024 game feed is partial (548 games) -> not used for the counted trials (too few seasons/games).
  * Burn-in: 2007 has no prior season -> ratings start at 0; scored test seasons are 2008-2022.
    Halves for the both-halves check: 2008-2014 and 2015-2022.

FBS DEFINITION (per season, from results only)
  * Rows with 6-digit rotation numbers (FCS-vs-FCS games SBR added in 2019/2021) are dropped from everything.
  * A team is FBS in season s if it has >= 6 games in that season's remaining rows (>= 3 in 2020, the COVID season).
  * AMENDMENT (2026-10-25, still BEFORE any market join; found by an outcome-only name-continuity check):
    SBR renames teams across seasons (Kent->KentState, SoMississippi->SouthernMiss, ArizonaU->Arizona, the 2020 page
    uses long names, ...). ALIAS maps them to one name so priors carry over. In 2020 the >=3-games rule admitted
    FCS teams (spring/fall FBS opponents) -> 2020 FBS = >=3 games AND FBS in 2019 or 2021.
  * All non-FBS teams are pooled into ONE pseudo-team "FCS" for rating purposes. Games involving FCS are used to fit
    ratings but are NEVER bet (FBS-vs-FCS excluded).

MARGIN MODEL (points, from scores only; no EPA -- no CFB play-by-play is local)
  pred_margin(home) = HFA * (not neutral) + r_home - r_away
  * Ratings fit before EACH game date d by weighted ridge on this season's games with date < d:
      minimise sum_i w_i (m_i - HFA*nn_i - (r_h - r_a))^2 + LAMBDA * sum_t (r_t - prior_t)^2
    m_i = home margin capped at +-28 (blowout cap). LAMBDA = 4 (game-equivalents).
  * Recency weight within season: w_i = 0.5 ** (days_ago / 42)   (half-life 6 weeks).
  * Prior (regression to a recruiting-free baseline): prior_t = 0.6 * (team's end-of-season rating in s-1).
    Teams not FBS in s-1 (new FBS members): prior -10. FCS pseudo-team: prior = 0.6 * its s-1 rating (2007: 0).
    End-of-season ratings use the same ridge on all of season s (no recency weight).
  * HFA: fixed per season = mean of the unpenalised end-of-season HFA estimates of all prior seasons
    (2008 uses 2007's). HFA is not re-estimated within the season.
  * QB availability: SKIPPED (no cheap historical CFB starter data).

TOTALS MODEL (offence/defence points ratings)
  pts(team A vs B) = mu + o_A + d_B + 0.5*HFA_pts*(home) - 0.5*HFA_pts*(away)   [HFA_pts reuses the margin HFA]
  * Same ridge machinery: o, d penalised toward 0.6 * last season's end values (new FBS: o -5, d +5),
    LAMBDA_OD = 4; mu penalised toward last season's mean points per team-game with LAMBDA_MU = 50.
    Points per team capped at 70. Same recency weight.
  * pred_total = 2*mu + o_h + d_a + o_a + d_h.
  * Pace proxy: NOT AVAILABLE (SBR has no plays/possessions); o_t + d_t already carries a team's pace/total tendency.

MARKET PARSE (SBR convention, as in the NFL script)
  Per game the two rows hold, in Open and in Close, one spread (smaller number, on the favourite's row) and one total
  (larger). "pk" = 0, "NL" = missing. Expected home margin from market = +s if the spread is on the home row else -s.
  Sanity filters (pre-set): total outside 25-100 -> missing; |spread| > 70 -> missing;
  |close - open| > 14 for spread or total -> that market is treated as a data error and dropped.
  N/N (neutral) games: second row = "home" for orientation, neutral flag set, HFA = 0.

BET RULE: spread bet when |pred_margin - open_margin| >= 3.0; total bet when |pred_total - open_total| >= 3.5.
  Side = model's side of the OPEN. One spread bet and at most one total bet per game. FBS-vs-FBS only.
  Week = 1 + floor((date - season's first date) / 7). Early = weeks 1-4.

PRE-REGISTERED TRIALS (14; bar p < 0.05/14 = 0.00357; one-sided season-week cluster bootstrap, 5000 draws)
  T01 spread CLV (pts, close - open toward bet)        T02 total CLV
  T03 spread ROI at open, -110                          T04 total ROI at open, -110
  T05 spread ROI at open, no-vig (even money)           T06 total ROI at open, no-vig
  T07 spread -110 ROI, bets on the opening FAVOURITE    T08 spread -110 ROI, bets on the opening DOG
  T09 spread -110 ROI, weeks 1-4                        T10 spread -110 ROI, weeks 5+
  T11 total -110 ROI, weeks 1-4                         T12 total -110 ROI, weeks 5+
  T13 spread CLV, weeks 1-4                             T14 total CLV, weeks 1-4
  Pass = p < bar AND mean > 0 AND mean > 0 in both halves (2008-14, 2015-22).
  Reported, not counted: model and open MAE vs outcome, stacking coefficients, by-season tables.
"""
from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SBR = DATA / "sbr_cfb"
OUT_JSON = DATA / "lab" / "cfb_simple_model_results.json"
LEDGER = ROOT / "lab" / "ledger.jsonl"
SEASONS = list(range(2007, 2023))
TEST = list(range(2008, 2023))
LAMBDA, LAMBDA_OD, LAMBDA_MU = 4.0, 4.0, 50.0
HALF_LIFE_D = 42.0
CAP_M, CAP_P = 28.0, 70.0
REG = 0.6
TH_S, TH_T = 3.0, 3.5
NTRIALS = 14
BAR = 0.05 / NTRIALS


# ------------------------------------------------------------------------------------------- parse
def _num(x):
    x = str(x).strip().lower()
    if x in ("pk", "p", "pick", "-pk"):
        return 0.0
    try:
        return float(x)
    except ValueError:
        return np.nan


def parse_season(y: int) -> pd.DataFrame:
    t = pd.read_html(io.StringIO((SBR / f"cfb_odds_{y}.html").read_text()))[0]
    t.columns = t.iloc[0]
    t = t.iloc[1:].reset_index(drop=True)
    rows = []
    for i in range(0, len(t) - 1, 2):
        v, h = t.iloc[i], t.iloc[i + 1]
        try:
            rv, rh = int(v["Rot"]), int(h["Rot"])
            md = int(v["Date"])
            vf, hf = int(v["Final"]), int(h["Final"])
        except (ValueError, TypeError):
            continue
        if rh != rv + 1:   # misaligned pair
            continue
        yr = y if md // 100 >= 7 else y + 1
        try:
            date = pd.Timestamp(yr, md // 100, md % 100)
        except ValueError:
            continue
        rec = dict(season=y, date=date, rot=rv, away=str(v["Team"]).strip(), home=str(h["Team"]).strip(),
                   away_pts=vf, home_pts=hf, neutral=(str(h["VH"]).strip() == "N"))
        for col, tag in (("Open", "open"), ("Close", "close")):
            a, b = _num(v[col]), _num(h[col])
            rec[f"spread_{tag}"] = rec[f"total_{tag}"] = np.nan
            if np.isnan(a) or np.isnan(b):
                continue
            if a >= b:   # total on away row, spread on home row -> home favoured
                tot, mg = a, b
            else:        # spread on away row -> away favoured
                tot, mg = b, -a
            if 25 <= tot <= 100:
                rec[f"total_{tag}"] = tot
            if abs(mg) <= 70:
                rec[f"spread_{tag}"] = mg   # expected HOME margin
        rows.append(rec)
    return pd.DataFrame(rows)


ALIAS = {"Kent": "KentState", "SoMississippi": "SouthernMiss", "ArizonaU": "Arizona", "HoustonU": "Houston",
         "MississippiSt": "MississippiState", "WashingtonU": "Washington", "AppalachianState": "AppalachianSt",
         "Buffalo": "BuffaloU", "Cincinnati": "CincinnatiU", "FloridaInternational": "FloridaIntl",
         "MiddleTennesseeState": "MiddleTennSt", "Minnesota": "MinnesotaU", "NorthernIllinois": "NoIllinois",
         "Pittsburgh": "PittsburghU", "Tennessee": "TennesseeU", "UTSA": "TexSanAntonio", "TexasAM": "TexasA&M"}


def load_all() -> pd.DataFrame:
    g = pd.concat([parse_season(y) for y in SEASONS], ignore_index=True)
    g["home"] = g["home"].replace(ALIAS); g["away"] = g["away"].replace(ALIAS)
    g = g[g["rot"] < 100000].copy()   # drop 6-digit rotation (FCS-vs-FCS) rows
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["points"] = g["home_pts"] + g["away_pts"]
    fbs = {}
    for s, d in g.groupby("season"):
        c = pd.concat([d["home"], d["away"]]).value_counts()
        fbs[s] = set(c[c >= (3 if s == 2020 else 6)].index)
    fbs[2020] = {t for t in fbs[2020] if t in fbs[2019] or t in fbs[2021]}
    g["fbs_game"] = [(r.home in fbs[r.season]) and (r.away in fbs[r.season]) for r in g.itertuples()]
    g["h_id"] = [r.home if r.home in fbs[r.season] else "FCS" for r in g.itertuples()]
    g["a_id"] = [r.away if r.away in fbs[r.season] else "FCS" for r in g.itertuples()]
    first = g.groupby("season")["date"].transform("min")
    g["week"] = ((g["date"] - first).dt.days // 7 + 1).astype(int)
    g = g.sort_values(["date", "rot"]).reset_index(drop=True)
    g.attrs["fbs"] = fbs
    return g


# ------------------------------------------------------------------------------------------- ridge
def ridge_margin(d: pd.DataFrame, teams: list, prior: dict, hfa: float | None, w: np.ndarray):
    """Returns ratings dict (and fitted HFA if hfa is None)."""
    ix = {t: i for i, t in enumerate(teams)}
    n, k = len(d), len(teams)
    est_h = hfa is None
    X = np.zeros((n, k + est_h))
    hi = d["h_id"].map(ix).to_numpy(); ai = d["a_id"].map(ix).to_numpy()
    X[np.arange(n), hi] += 1; X[np.arange(n), ai] -= 1
    nn = (~d["neutral"].to_numpy()).astype(float)
    p = np.array([prior.get(t, 0.0) for t in teams])
    y = np.clip(d["margin"].to_numpy(float), -CAP_M, CAP_M) - X[:, :k] @ p
    if est_h:
        X[:, k] = nn
    else:
        y = y - hfa * nn
    pen = np.full(X.shape[1], LAMBDA);
    if est_h:
        pen[k] = 1e-6
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (w * y))
    r = dict(zip(teams, p + beta[:k]))
    return (r, float(beta[k])) if est_h else (r, hfa)


def ridge_od(d: pd.DataFrame, teams: list, prior_o: dict, prior_d: dict, mu0: float, hfa: float, w: np.ndarray):
    ix = {t: i for i, t in enumerate(teams)}
    k = len(teams); n = len(d)
    # two rows per game: home scoring, away scoring. params: o (k), d (k), mu (1)
    X = np.zeros((2 * n, 2 * k + 1))
    hi = d["h_id"].map(ix).to_numpy(); ai = d["a_id"].map(ix).to_numpy()
    r = np.arange(n)
    X[r, hi] = 1; X[r, k + ai] = 1
    X[n + r, ai] = 1; X[n + r, k + hi] = 1
    X[:, 2 * k] = 1
    nn = (~d["neutral"].to_numpy()).astype(float)
    y = np.concatenate([np.minimum(d["home_pts"].to_numpy(float), CAP_P) - 0.5 * hfa * nn,
                        np.minimum(d["away_pts"].to_numpy(float), CAP_P) + 0.5 * hfa * nn])
    ww = np.concatenate([w, w])
    pvec = np.array([prior_o.get(t, -5.0) for t in teams] + [prior_d.get(t, 5.0) for t in teams] + [mu0])
    y = y - X @ pvec
    pen = np.array([LAMBDA_OD] * (2 * k) + [LAMBDA_MU])
    A = X.T @ (X * ww[:, None]) + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (ww * y)) + pvec
    return dict(zip(teams, beta[:k])), dict(zip(teams, beta[k:2 * k])), float(beta[2 * k])


def build_predictions(g: pd.DataFrame) -> pd.DataFrame:
    fbs = g.attrs["fbs"]
    out = []
    prev_r, prev_o, prev_d, prev_mu, hfas = {}, {}, {}, None, []
    for s in SEASONS:
        d = g[g["season"] == s]
        teams = sorted(set(d["h_id"]) | set(d["a_id"]))
        prev_fbs = fbs.get(s - 1, set())
        prior_r = {t: (REG * prev_r[t] if (t in prev_r and (t == "FCS" or t in prev_fbs)) else (-10.0 if t != "FCS" else 0.0))
                   for t in teams} if prev_r else {t: 0.0 for t in teams}
        prior_o = {t: (REG * prev_o[t] if t in prev_o else (-5.0 if t != "FCS" else 0.0)) for t in teams} if prev_o else {t: 0.0 for t in teams}
        prior_dd = {t: (REG * prev_d[t] if t in prev_d else (5.0 if t != "FCS" else 0.0)) for t in teams} if prev_d else {t: 0.0 for t in teams}
        mu0 = prev_mu if prev_mu is not None else 28.0
        hfa = float(np.mean(hfas)) if hfas else None
        if s in TEST:
            for date, gd in d.groupby("date"):
                past = d[d["date"] < date]
                if len(past):
                    w = 0.5 ** ((date - past["date"]).dt.days.to_numpy(float) / HALF_LIFE_D)
                    r, _ = ridge_margin(past, teams, prior_r, hfa, w)
                    o, dd, mu = ridge_od(past, teams, prior_o, prior_dd, mu0, hfa, w)
                else:
                    r, o, dd, mu = prior_r, prior_o, prior_dd, mu0
                for row in gd.itertuples():
                    nn = 0.0 if row.neutral else 1.0
                    pm = hfa * nn + r[row.h_id] - r[row.a_id]
                    pt = 2 * mu + o[row.h_id] + dd[row.a_id] + o[row.a_id] + dd[row.h_id]
                    out.append(dict(idx=row.Index, pred_margin=pm, pred_total=pt))
        # end-of-season fit (all games, unweighted) -> priors + HFA for next season
        w1 = np.ones(len(d))
        prev_r, h_end = ridge_margin(d, teams, prior_r, None, w1)
        hfas.append(h_end)
        prev_o, prev_d, prev_mu = ridge_od(d, teams, prior_o, prior_dd, mu0, float(np.mean(hfas)), w1)
    p = pd.DataFrame(out).set_index("idx")
    return g.join(p, how="inner")


# ------------------------------------------------------------------------------------------- scoring
def boot_p(vals, clusters, n=5000, seed=0):
    df = pd.DataFrame({"v": vals, "c": clusters})
    gg = df.groupby("c")["v"].agg(["sum", "size"])
    s, k = gg["sum"].to_numpy(), gg["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    means = s[idx].sum(1) / k[idx].sum(1)
    return float(np.mean(means <= 0)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def grade(side, line, outcome, payout):
    r = side * (outcome - line)
    return np.where(r > 0, payout, np.where(r < 0, -1.0, 0.0))


def bets(P: pd.DataFrame, mkt: str) -> pd.DataFrame:
    pred = "pred_margin" if mkt == "spread" else "pred_total"
    y = "margin" if mkt == "spread" else "points"
    th = TH_S if mkt == "spread" else TH_T
    d = P[P["fbs_game"] & P[f"{mkt}_open"].notna() & P[f"{mkt}_close"].notna()].copy()
    d = d[(d[f"{mkt}_close"] - d[f"{mkt}_open"]).abs() <= 14]
    gap = d[pred] - d[f"{mkt}_open"]
    d = d[gap.abs() >= th].copy()
    d["side"] = np.sign(d[pred] - d[f"{mkt}_open"])
    o, c, out = d[f"{mkt}_open"].to_numpy(), d[f"{mkt}_close"].to_numpy(), d[y].to_numpy(float)
    d["clv"] = d["side"] * (c - o)
    d["roi110"] = grade(d["side"].to_numpy(), o, out, 100 / 110)
    d["roinv"] = grade(d["side"].to_numpy(), o, out, 1.0)
    if mkt == "spread":   # favourite = side whose expected margin at open is positive
        d["on_fav"] = (d["side"] * o) > 0
    d["cl"] = d["season"].astype(str) + "_" + d["week"].astype(str)
    return d


def summarise(d: pd.DataFrame, col: str) -> dict:
    v = d[col].to_numpy(float)
    if len(v) == 0:
        return dict(n=0)
    p, lo, hi = boot_p(v, d["cl"].to_numpy())
    h1 = v[(d["season"] <= 2014).to_numpy()]; h2 = v[(d["season"] >= 2015).to_numpy()]
    return dict(n=int(len(v)), mean=float(v.mean()), lo=lo, hi=hi, p=p,
                h1=float(h1.mean()) if len(h1) else None, h2=float(h2.mean()) if len(h2) else None)


def run_model():
    g = load_all()
    P = build_predictions(g)
    f = P[P["fbs_game"]]
    print("games parsed", len(g), "scored test games", len(P), "FBS-vs-FBS", len(f))
    print("FBS teams per season", {s: len(v) for s, v in g.attrs["fbs"].items()})
    for s, d in f.groupby("season"):
        print(s, len(d), "margin MAE %.2f  total MAE %.2f  total bias %+.2f" % (
            (d.pred_margin - d.margin).abs().mean(), (d.pred_total - d.points).abs().mean(), (d.pred_total - d.points).mean()))
    return g, P


def run_dev(ledger: bool):
    g = load_all()
    P = build_predictions(g)
    S, T = bets(P, "spread"), bets(P, "total")
    early = lambda d: d[d["week"] <= 4]
    late = lambda d: d[d["week"] >= 5]
    trials = [
        ("T01_spread_clv", S, "clv"), ("T02_total_clv", T, "clv"),
        ("T03_spread_roi110", S, "roi110"), ("T04_total_roi110", T, "roi110"),
        ("T05_spread_roi_novig", S, "roinv"), ("T06_total_roi_novig", T, "roinv"),
        ("T07_spread_roi110_fav", S[S["on_fav"]], "roi110"), ("T08_spread_roi110_dog", S[~S["on_fav"]], "roi110"),
        ("T09_spread_roi110_wk1_4", early(S), "roi110"), ("T10_spread_roi110_wk5p", late(S), "roi110"),
        ("T11_total_roi110_wk1_4", early(T), "roi110"), ("T12_total_roi110_wk5p", late(T), "roi110"),
        ("T13_spread_clv_wk1_4", early(S), "clv"), ("T14_total_clv_wk1_4", early(T), "clv"),
    ]
    res = {}
    for name, d, col in trials:
        r = summarise(d, col)
        r["passed"] = bool(r["n"] and r["p"] < BAR and r["mean"] > 0 and r["h1"] > 0 and r["h2"] > 0)
        res[name] = r
        print(f"{name:26s} n={r['n']:5d} mean={r['mean']:+.4f} CI[{r['lo']:+.4f},{r['hi']:+.4f}] p={r['p']:.4f} "
              f"h1={r['h1']:+.4f} h2={r['h2']:+.4f} {'PASS' if r['passed'] else ''}")
    # diagnostics (not counted)
    diag = {}
    for mkt, pred, y in (("spread", "pred_margin", "margin"), ("total", "pred_total", "points")):
        d = P[P["fbs_game"] & P[f"{mkt}_open"].notna() & P[f"{mkt}_close"].notna()]
        d = d[(d[f"{mkt}_close"] - d[f"{mkt}_open"]).abs() <= 14]
        diag[mkt] = dict(n=int(len(d)), mae_model=float((d[pred] - d[y]).abs().mean()),
                         mae_open=float((d[f"{mkt}_open"] - d[y]).abs().mean()),
                         mae_close=float((d[f"{mkt}_close"] - d[y]).abs().mean()),
                         mean_abs_model_minus_open=float((d[pred] - d[f"{mkt}_open"]).abs().mean()),
                         bias_model_minus_open=float((d[pred] - d[f"{mkt}_open"]).mean()))
        import statsmodels.api as sm
        cl = (d["season"].astype(str) + "_" + d["week"].astype(str)).astype("category").cat.codes
        for nm, dep, ref in (("outcome_vs_open", d[y] - d[f"{mkt}_open"], d[f"{mkt}_open"]),
                             ("outcome_vs_close", d[y] - d[f"{mkt}_close"], d[f"{mkt}_close"]),
                             ("move_vs_open", d[f"{mkt}_close"] - d[f"{mkt}_open"], d[f"{mkt}_open"])):
            x = (d[pred] - ref).to_numpy(float)
            m = sm.OLS(dep.to_numpy(float), sm.add_constant(x)).fit(cov_type="cluster", cov_kwds={"groups": cl})
            ci = m.conf_int()
            diag[mkt][nm] = dict(b=float(m.params[1]), lo=float(ci[1][0]), hi=float(ci[1][1]), p=float(m.pvalues[1]))
        # by season (bets)
        B = S if mkt == "spread" else T
        diag[mkt]["by_season"] = {int(s): dict(n=int(len(x)), clv=float(x.clv.mean()), roi110=float(x.roi110.mean()))
                                  for s, x in B.groupby("season")}
        diag[mkt]["bets_win_rate_open"] = float((B.roinv > 0).sum() / max((B.roinv != 0).sum(), 1))
    print(json.dumps(diag, indent=1))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(dict(trials=res, diag=diag, bar=BAR), indent=1))
    if ledger:
        ts = datetime.now(timezone.utc).isoformat()
        with LEDGER.open("a") as fh:
            for i, (name, r) in enumerate(res.items(), 1):
                metric = "clv" if "clv" in name else ("roi_novig" if "novig" in name else "roi110")
                fh.write(json.dumps(dict(ts=ts, strategy=f"cfb_simple_{name}", dataset="cfb_simple_model", metric=metric,
                                         hypothesis="pre-registered in scripts/cfb_simple_model.py docstring",
                                         source="user 2026-10-25: simple CFB ridge model vs SBR opener (dev 2008-22)",
                                         status="ok", trial_number=i, bonferroni_bar=BAR, passed=r["passed"],
                                         bets=r["n"], mean=r["mean"], ci_lo=r["lo"], ci_hi=r["hi"], p=r["p"],
                                         per_half={"2008-14": r["h1"], "2015-22": r["h2"]})) + "\n")
        print("ledger: appended", len(res))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "model"
    if cmd == "model":
        run_model()
    elif cmd == "dev":
        run_dev("--ledger" in sys.argv)
    else:
        print(__doc__)
