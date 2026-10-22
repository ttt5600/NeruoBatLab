"""Do PUBLIC NBA game models beat the closing moneyline?  (user 2026-10-21: "No way the ESPN model is the best source")

    ../.venv_quant/bin/python scripts/nba_public_models.py [--ledger]

PRE-REGISTERED 2026-10-21, written BEFORE any model probability was joined to a market price or outcome.

Models with FREE archived per-game PREGAME probabilities (the only kind that can be backtested without a rebuild):
  - 538 NBA Elo (elo_prob1), CARM-Elo (carm-elo_prob1, 2015-16..2018-19), RAPTOR (raptor_prob1, 2018-19..2022-23)
    from 538's nba_elo.csv. projects.fivethirtyeight.com now redirects to ABC News, so the file is the Wayback
    snapshot https://web.archive.org/web/20231024142254id_/https://projects.fivethirtyeight.com/nba-model/nba_elo.csv
    (10,835,526 bytes, md5 b11901fb009e1ba393b4a2e653c83aff) -> data/fte_nba/nba_elo.csv (gitignored).
    CAVEAT: Elo is a deterministic function of past scores, so elo_prob1 has no look-ahead by construction (its K and
    home edge were tuned on history). RAPTOR/CARM-Elo rows may be 538 re-computations, not the as-published numbers.
  - ESPN's own pregame home win prob p0 = first value of ESPN summary winprobability[] (cached in
    data/nba_late_states.parquet by nba_late_wp.py). This is "the ESPN model" -- the benchmark the user doubts.

Markets (team1 = home in the 538 file; verified by the score join below):
  A. SBR sqlite (data/nba_odds/OddsData.sqlite), tables odds_2007-08 .. odds_2018-19 ONLY = the nba_sbr DEV window
     (2019-20+ is reserved as nba_sbr SEALED in nba_angles.py and is never opened here). ML_Home/ML_Away (American);
     the file does not say whether the ML is the close (UNVERIFIED; SBR archive MLs are usually the closing ML).
     The unsigned Spread column is not used at all. Joined to 538 on date + home margin + total points.
  B. ESPN lines (data/espn/nba/lines_{2022..2025}.parquet = seasons 2021-22..2024-25; 2026 is SEALED and never read).
     Real books only (drop consensus/accuscore/teamrankings/betegy). Column home_ml ~= close (equals home_ml_close in
     98-100% of rows where both exist). Market prob = median over books of proportional de-vig; bet price = median
     decimal odds over books for the chosen side. Joined to 538 on US/Eastern date + home/away score.
  Both: regular season only; drop rows with overround outside [0, 0.10]. Market prob = proportional de-vig.

TRIALS (dataset nba_public_models; 7 trials -> bar one-sided p < 0.05/7 = 0.0071; flat 1u at the market price;
one bet per game; one-sided bootstrap over games, 4000 resamples, p = P(boot units <= 0)):
  M1 elo_sbr_x05      A, 2007-08..2018-19: bet home if elo_prob1 - p_mkt >= 0.05, away if <= -0.05
  M2 elo_sbr_x10      A, same with 0.10
  M3 carmelo_sbr_x05  A, 2015-16..2018-19: carm-elo_prob1, 0.05
  M4 raptor_espn_x05  B, 2021-22..2022-23 (538 stopped after 2022-23): raptor_prob1, 0.05
  M5 espnwp_espn_x05  B, 2021-22..2024-25: ESPN p0, 0.05
  M6 stack_sbr_elo    A: walk-forward by season (fit on all earlier dev seasons, first test season 2008-09):
                      logistic y ~ logit(p_mkt) + logit(elo_prob1); bet when p_stack - p_mkt >= 0.03 (either side)
  M7 stack_espn_wp    B: same with ESPN p0, fit on earlier seasons, test 2022-23..2024-25
DESCRIPTIVE (not trials): Brier + log loss of every model vs the de-vigged market on the same games, paired bootstrap
of per-game log-loss difference; in-sample stacking coefficients with 95% CIs; whether ESPN's 'accuscore' and
'teamrankings' provider rows are independent forecasts or relayed market lines (correlation with the book median).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sports.market import american_to_decimal, brier, devig_two_way, log_loss, paired_logloss_bootstrap

FTE = ROOT / "data" / "fte_nba" / "nba_elo.csv"
DB = ROOT / "data" / "nba_odds" / "OddsData.sqlite"
ESPN = ROOT / "data" / "espn" / "nba"
STATES = ROOT / "data" / "nba_late_states.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "nba_public_models"
SBR_DEV = [f"{y}-{str(y + 1)[2:]}" for y in range(2007, 2019)]
ESPN_DEV = [2022, 2023, 2024, 2025]          # end-year labels; 2026 SEALED
NOT_BOOKS = {"consensus", "accuscore", "teamrankings", "betegy"}
N_TRIALS = 7


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fte() -> pd.DataFrame:
    e = pd.read_csv(FTE)
    e = e[e.playoff.isna() & (e.neutral == 0)].copy()
    e["date"] = pd.to_datetime(e.date).dt.date
    return e[["date", "season", "team1", "team2", "score1", "score2", "elo_prob1", "carm-elo_prob1", "raptor_prob1"]]


def sbr() -> pd.DataFrame:
    c = sqlite3.connect(DB)
    fs = []
    for s in SBR_DEV:
        d = pd.read_sql(f'select Date, Home, Away, ML_Home, ML_Away, Win_Margin, Points from "odds_{s}"', c)
        y0 = int(s[:4])
        mmdd = d.Date.str[-4:]
        yr = np.where(mmdd.str[:2].astype(int) >= 8, y0, y0 + 1)
        d["date"] = pd.to_datetime(pd.Series(yr, index=d.index).astype(str) + mmdd, format="%Y%m%d").dt.date
        d["season"] = y0 + 1
        fs.append(d)
    d = pd.concat(fs, ignore_index=True)
    for k in ["ML_Home", "ML_Away", "Win_Margin", "Points"]:
        d[k] = pd.to_numeric(d[k], errors="coerce")
    d = d.dropna(subset=["ML_Home", "ML_Away", "Win_Margin", "Points"])
    d = d[(d.ML_Home != 0) & (d.ML_Away != 0) & (d.Win_Margin != 0)]
    ph, _, ov = devig_two_way(d.ML_Home, d.ML_Away)
    d["p_mkt"], d["over"] = ph, ov
    d["dec_h"], d["dec_a"] = american_to_decimal(d.ML_Home), american_to_decimal(d.ML_Away)
    d["y"] = (d.Win_Margin > 0).astype(int)
    return d[(d.over >= 0) & (d.over <= 0.10)]


def join_sbr(d: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    e = e.assign(margin=e.score1 - e.score2, tot=e.score1 + e.score2)
    k = ["date", "margin", "tot"]
    e1 = e.drop_duplicates(k, keep=False)
    m = d.rename(columns={"Win_Margin": "margin", "Points": "tot"}).merge(e1, on=k, how="inner", suffixes=("", "_538"))
    # flipped-orientation check: would the join work if team1 were the AWAY team?
    flip = d.rename(columns={"Points": "tot"}).assign(margin=-d.Win_Margin).merge(e1, on=k, how="inner")
    print(f"[A] SBR dev games {len(d)}; joined to 538 (team1=home) {len(m)}; with team1=away instead {len(flip)}")
    return m


def espn() -> pd.DataFrame:
    rows = []
    for y in ESPN_DEV:
        g = pd.read_parquet(ESPN / f"games_{y}.parquet")
        g = g[(g.season_type == "regular-season") & (g.status == "STATUS_FINAL") & (~g.neutral.astype(bool))]
        ln = pd.read_parquet(ESPN / f"lines_{y}.parquet")
        bk = ln[~ln.provider.isin(NOT_BOOKS)].dropna(subset=["home_ml", "away_ml"]).copy()
        ph, _, ov = devig_two_way(bk.home_ml, bk.away_ml)
        bk["p"], bk["over"] = ph, ov
        bk = bk[(bk.over >= 0) & (bk.over <= 0.10)]
        bk["dh"], bk["da"] = american_to_decimal(bk.home_ml), american_to_decimal(bk.away_ml)
        agg = bk.groupby("event_id").agg(p_mkt=("p", "median"), dec_h=("dh", "median"), dec_a=("da", "median"),
                                         n_books=("p", "size")).reset_index()
        side = {}
        for prov in ("accuscore", "teamrankings"):
            x = ln[ln.provider == prov].dropna(subset=["home_ml", "away_ml"])
            if len(x):
                side[prov] = pd.Series(devig_two_way(x.home_ml, x.away_ml)[0], index=x.event_id.values)
        g = g.merge(agg, on="event_id", how="inner")
        for prov, s in side.items():
            g[f"p_{prov}"] = g.event_id.map(s[~s.index.duplicated()])
        g["season"] = y
        rows.append(g)
    g = pd.concat(rows, ignore_index=True)
    g["date"] = pd.to_datetime(g.start_utc).dt.tz_convert("America/New_York").dt.date
    g = g[g.home_score != g.away_score]
    g["y"] = (g.home_score > g.away_score).astype(int)
    st = pd.read_parquet(STATES, columns=["eid", "p0", "season"]).drop_duplicates("eid")
    st = st[st.season <= 2024]                     # start-year 2025 = 2025-26 = SEALED; never used
    g = g.merge(st[["eid", "p0"]].rename(columns={"eid": "event_id"}).astype({"event_id": g.event_id.dtype}),
                on="event_id", how="left")
    return g


def join_espn(g: pd.DataFrame, e: pd.DataFrame) -> pd.DataFrame:
    k = ["date", "score1", "score2"]
    e1 = e.drop_duplicates(k, keep=False)
    m = g.merge(e1.drop(columns=["season"]), left_on=["date", "home_score", "away_score"], right_on=k, how="left")
    print(f"[B] ESPN dev regular-season games {len(g)}; with 538 row {m.elo_prob1.notna().sum()}; "
          f"with ESPN p0 {m.p0.notna().sum()}")
    return m


def bets(d, pm, thr):
    diff = d[pm] - d.p_mkt
    home, away = diff >= thr, diff <= -thr
    m = (home | away) & d[pm].notna()
    x = d[m]
    hs = home[m].to_numpy()
    won = np.where(hs, x.y == 1, x.y == 0)
    dec = np.where(hs, x.dec_h, x.dec_a)
    return x, np.where(won, dec - 1, -1.0)


def score(pnl, season, rng) -> dict:
    if len(pnl) == 0:
        return {"bets": 0}
    boots = np.array([rng.choice(pnl, len(pnl)).sum() for _ in range(4000)])
    per = pd.Series(pnl, index=season.index).groupby(season).sum().round(2)
    return {"bets": int(len(pnl)), "staked": float(len(pnl)), "units": float(pnl.sum()), "roi": float(pnl.mean()),
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()), "folds": int(len(per)),
            "per_fold": {str(k): v for k, v in per.items()}}


def stack_walk(d, pm, seasons_test) -> pd.Series:
    out = pd.Series(np.nan, index=d.index)
    for s in seasons_test:
        tr, te = d[(d.season < s) & d[pm].notna()], d[(d.season == s) & d[pm].notna()]
        if len(tr) < 200 or len(te) == 0:
            continue
        X = sm.add_constant(np.column_stack([logit(tr.p_mkt), logit(tr[pm])]))
        f = sm.Logit(tr.y.to_numpy(), X).fit(disp=0)
        Xt = sm.add_constant(np.column_stack([logit(te.p_mkt), logit(te[pm])]), has_constant="add")
        out.loc[te.index] = f.predict(Xt)
    return out


def describe(tag, d, pm):
    x = d[d[pm].notna()]
    if len(x) < 50:
        return
    b = paired_logloss_bootstrap(x[pm], x.p_mkt, x.y, n_boot=2000)
    X = sm.add_constant(np.column_stack([logit(x.p_mkt), logit(x[pm])]))
    f = sm.Logit(x.y.to_numpy(), X).fit(disp=0)
    ci = f.conf_int()
    print(f"  {tag:28s} n {len(x):5d}  model LL {log_loss(x[pm], x.y):.4f} Brier {brier(x[pm], x.y):.4f} | market LL "
          f"{log_loss(x.p_mkt, x.y):.4f} Brier {brier(x.p_mkt, x.y):.4f} | dLL {b['observed_diff']:+.4f} "
          f"[{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] | stack b_mkt {f.params[1]:.2f} b_model {f.params[2]:+.2f} "
          f"[{ci[2][0]:+.2f},{ci[2][1]:+.2f}] | corr {np.corrcoef(x[pm], x.p_mkt)[0, 1]:.3f}")


def main() -> int:
    e = fte()
    A = join_sbr(sbr(), e)
    B = join_espn(espn(), e)
    print(f"sanity A: market LL {log_loss(A.p_mkt, A.y):.4f}, home win {A.y.mean():.3f}, overround med {A.over.median():.3f}")
    print("DESCRIPTIVE: model vs de-vigged market on the same games (dLL < 0 = model better)")
    for s0, s1, tag in ((2008, 2019, "A all"), (2008, 2013, "A 2007-08..2012-13"), (2014, 2019, "A 2013-14..2018-19")):
        describe(f"elo_prob1 {tag}", A[(A.season >= s0) & (A.season <= s1)], "elo_prob1")
    describe("carm-elo_prob1 A", A, "carm-elo_prob1")
    describe("raptor_prob1 A (2018-19)", A, "raptor_prob1")
    for pm in ("elo_prob1", "raptor_prob1", "p0", "p_accuscore", "p_teamrankings"):
        describe(f"{pm} B", B, pm)
    for y in ESPN_DEV:
        describe(f"p0 B {y}", B[B.season == y], "p0")
    same = B[B.elo_prob1.notna() & B.raptor_prob1.notna() & B.p0.notna()]   # apples-to-apples, 2021-22..2022-23
    for pm in ("elo_prob1", "raptor_prob1", "p0"):
        describe(f"{pm} B same games", same, pm)
    same = same.assign(p_avg=(same.raptor_prob1 + same.p0) / 2)
    describe("mean(raptor,p0) B same games", same, "p_avg")

    A["p_stack"] = stack_walk(A, "elo_prob1", range(2009, 2020))
    B["p_stack"] = stack_walk(B, "p0", [2023, 2024, 2025])
    describe("stack(elo) A walk-forward", A, "p_stack")
    describe("stack(p0) B walk-forward", B, "p_stack")

    plan = [("elo_sbr_x05", A, "elo_prob1", .05), ("elo_sbr_x10", A, "elo_prob1", .10),
            ("carmelo_sbr_x05", A, "carm-elo_prob1", .05),
            ("raptor_espn_x05", B[B.season <= 2023], "raptor_prob1", .05), ("espnwp_espn_x05", B, "p0", .05),
            ("stack_sbr_elo", A, "p_stack", .03), ("stack_espn_wp", B, "p_stack", .03)]
    assert len(plan) == N_TRIALS
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r.get("dataset") == DATASET}
    recs = []
    print(f"TRIALS (bar p < {0.05 / N_TRIALS:.4f})")
    for name, d, pm, thr in plan:
        x, pnl = bets(d, pm, thr)
        r = score(pnl, x.season, np.random.default_rng(0))
        sha = hashlib.sha256(("nba_public_models:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"  {name:16s} bets {r['bets']:5d} units {r.get('units', 0):+8.1f} ROI {r.get('roi', 0):+7.2%} "
              f"CI [{r.get('ci_lo', 0):+.1f},{r.get('ci_hi', 0):+.1f}] p {r.get('p', 1):.4f} "
              f"seasons+ {r.get('folds_positive', 0)}/{r.get('folds', 0)}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"nba_pub_{name}",
                     "sha": sha, "dataset": DATASET, "hypothesis": f"pre-registered in scripts/nba_public_models.py ({name})",
                     "source": "public NBA models vs close (user 2026-10-21)", "status": "ok", "trial_number": len(shas),
                     "passed": bool(r.get("p", 1) < 0.05 / N_TRIALS and r.get("roi", 0) > 0), **r})
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"appended {len(recs)} trials to {LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
