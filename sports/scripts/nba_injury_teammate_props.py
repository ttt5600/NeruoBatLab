"""(A) injury -> teammate OPENING props, (B) hot/cold streaks -> OPENING points props. Pre-registered in
notes/nba_injury_teammate_props.md (written before any signal was joined to a line or outcome).

    ../.venv_quant/bin/python scripts/nba_injury_teammate_props.py signals    # news side only: reports + nba_sim deltas
    ../.venv_quant/bin/python scripts/nba_injury_teammate_props.py run [--ledger]
    ../.venv_quant/bin/python scripts/nba_injury_teammate_props.py reliability   # hot-streak split-half, box only

DEV = ESPN seasons 2023-2025 (2022-23..2024-25), OPENING line/price only. SEALED 2025-26 is never loaded.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import nba_player_opp_venue as pov  # noqa: E402
import nba_sim as ns  # noqa: E402
from q011_reports import norm  # noqa: E402

DATA = ROOT / "data" / "espn" / "nba"
REPD = ROOT / "data" / "nba_injury_props"
OUT = ROOT / "results" / "nba_injury_teammate_props"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DEV = [2023, 2024, 2025]
ROT_MIN = 25.0
X_PTS, X_REB, X_AST = 2.0, 1.0, 1.0
A3_THR = {"points": 1.0, "rebounds": 0.5, "assists": 0.5}
K_HOT = 5.0
N_TRIALS = 8
BAR = 0.05 / N_TRIALS
NB = 4000
ET = "America/New_York"
NAME_ALIAS = {"jimmybutler": "jimmybutleriii", "carltoncarrington": "bubcarrington", "alexandresarr": "alexsarr",
              "brandonboston": "brandonbostonjr", "reggiebullock": "reggiebullockjr", "cuiyongxi": "yongxicui"}
SMAP = {"points": "pts", "rebounds": "reb", "assists": "ast"}


# ------------------------------------------------------------------------------------------- injury report side
def report_status(g: pd.DataFrame, box: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """-> one row per (event_id, team, athlete_id) listed Out/Doubtful in any pre-tip report:
    final_out (Out/Doubtful in the LAST pre-tip report that has this team's game), t_out (first pub listing him
    Out/Doubtful), plus per team-game coverage (n pre-tip reports with the team submitted)."""
    idx = pd.read_parquet(REPD / "report_index.parquet")
    rows = pd.read_parquet(REPD / "report_rows.parquet")
    rows = rows.merge(idx[["label", "pub"]], on="label", how="left")
    rows = rows[rows.pub.notna() & rows.gdate.notna() & rows.team.notna()]
    rows["gd"] = pd.to_datetime(rows.gdate, format="%m/%d/%Y").dt.date
    rows["tm"] = rows.team.map(lambda t: ns.ALIAS.get(t, t))
    # report game -> ESPN event (ET date, team)
    sides = pd.concat([g[["event_id", "date", "home", "start_utc", "season"]].rename(columns={"home": "tm"}),
                       g[["event_id", "date", "away", "start_utc", "season"]].rename(columns={"away": "tm"})])
    sides = sides.rename(columns={"date": "gd"})
    m = rows.merge(sides, on=["gd", "tm"], how="inner")
    audit = {"report_rows": int(len(rows)), "rows_matched_game": int(len(m))}
    # AMENDMENT (before outcomes): the 2-hourly grid misses the last ~30-90 min before tip (e.g. 17:30 report, 19:00
    # tip); a report published within 90 min AFTER tip still shows the pre-tip status of that game, so it is used.
    # A first OUT listing seen only after tip is class LATE.
    m = m[m.pub < m.start_utc + pd.Timedelta(minutes=90)]
    # name -> athlete id per (season, team) from box
    nm = box[["event_id", "athlete_id", "name", "team"]].merge(g[["event_id", "season"]], on="event_id")
    nm = nm[["athlete_id", "name", "team", "season"]].drop_duplicates()
    nm["pkey"] = nm.name.fillna("").map(norm)
    cnt = nm.groupby(["season", "team", "pkey"]).athlete_id.nunique()
    amb = set(cnt[cnt > 1].index)
    nm = nm.drop_duplicates(["season", "team", "pkey"])
    m["pkey"] = m.pkey.replace(NAME_ALIAS)
    pl = m[m.status != "NOT_SUBMITTED"].merge(nm.rename(columns={"team": "tm"}), on=["season", "tm", "pkey"], how="left")
    audit["player_rows_pre_tip"] = int(len(pl))
    audit["player_rows_unmatched"] = int(pl.athlete_id.isna().sum())
    audit["player_rows_ambiguous"] = int(sum((s, t, k) in amb for s, t, k in zip(pl.season, pl.tm, pl.pkey)))
    pl = pl[pl.athlete_id.notna()]
    # last pre-tip report per team-game in which the team had SUBMITTED (has >= 1 player row, not NOT_SUBMITTED)
    sub = m[m.status != "NOT_SUBMITTED"].groupby(["event_id", "tm"]).agg(last_pub=("pub", "max"), n_rep=("label", "nunique"))
    pl = pl.merge(sub, left_on=["event_id", "tm"], right_index=True)
    pl["out"] = pl.status.isin(["Out", "Doubtful"])
    fin = pl[pl.pub == pl.last_pub].groupby(["event_id", "tm", "athlete_id"]).out.max().rename("final_out")
    tout = pl[pl.out].groupby(["event_id", "tm", "athlete_id"]).pub.min().rename("t_out")
    st = pd.concat([fin, tout], axis=1).reset_index()
    st["final_out"] = st.final_out.fillna(False).astype(bool)
    return st, {"audit": audit, "covered": sub.reset_index()}


def timing_class(t_out: pd.Timestamp, start_utc: pd.Timestamp) -> str:
    gday = start_utc.tz_convert(ET).normalize()
    t = t_out.tz_convert(ET)
    if t < gday + pd.Timedelta(hours=10):
        return "EARLY"
    if t < gday + pd.Timedelta(hours=17):
        return "MID"
    return "LATE"


def build_signals() -> dict:
    g, box = ns.load_hist([2022] + DEV)
    st_tab, cov = report_status(g, box)
    covered = set(zip(cov["covered"].event_id, cov["covered"].tm))
    S = {(e, t, a): (fo, to) for e, t, a, fo, to in st_tab[["event_id", "tm", "athlete_id", "final_out", "t_out"]].itertuples(index=False)}
    listed = {}
    for (e, t, a), (fo, to) in S.items():
        if fo:
            listed.setdefault((e, t), {})[a] = to
    last10: dict = {}
    trows, prow = [], []
    g = g.reset_index(drop=True)

    def on_date(d, idx, st, pr, bx):
        for i in idx:
            r = g.iloc[i]
            if int(r.season) in DEV:
                for side in ("home", "away"):
                    T = r[side]
                    full, prev = st.full_set(T), st.prev_set(T)
                    act = {x.athlete_id for x in bx.get((r.event_id, T), [])}
                    absent = full - act
                    lo = listed.get((r.event_id, T), {})
                    m10 = {a: np.mean(last10[a][-10:]) if a in last10 else 0.0 for a in full}
                    rot = {a for a in absent if a in prev and m10[a] >= ROT_MIN}
                    news = {a for a in rot if a in lo}
                    unl = rot - news
                    row = {"event_id": r.event_id, "team": T, "date": r.date, "start_utc": r.start_utc,
                           "season": int(r.season), "covered": (r.event_id, T) in covered,
                           "n_news": len(news), "n_unlisted": len(unl),
                           "n_carry": len({a for a in absent & set(lo) if a not in prev and m10[a] >= ROT_MIN}),
                           "news_ids": sorted(news), "unlisted_ids": sorted(unl),
                           "news_names": [st.names.get(a, a) for a in sorted(news)]}
                    if news:
                        cls = [timing_class(lo[a], r.start_utc) for a in news]
                        row["cls"] = "LATE" if "LATE" in cls else ("MID" if "MID" in cls else "EARLY")
                        row["t_out_max"] = max(lo[a] for a in news)
                        avail = full - (absent & set(lo))
                        s0 = ns.side_inputs(st, d, pr, full, avail | news, team=T)  # baseline: news players in
                        s1 = ns.side_inputs(st, d, pr, full, avail, team=T)         # tonight: news players out
                        for a in avail:
                            R = s0["R"][a]
                            p = {"event_id": r.event_id, "team": T, "athlete_id": a, "m0": s0["m"][a], "m1": s1["m"][a],
                                 "cls": row["cls"]}
                            for stt, k in SMAP.items():
                                p[f"proj0_{stt}"] = s0["m"][a] * R[k]
                                p[f"d_{stt}"] = (s1["m"][a] - s0["m"][a]) * R[k]
                            prow.append(p)
                    trows.append(row)
            for side in ("home", "away"):
                for x in bx.get((r.event_id, r[side]), []):
                    last10.setdefault(x.athlete_id, []).append(x.minutes)

    ns.walk(g, box, on_date)
    TG, P = pd.DataFrame(trows), pd.DataFrame(prow)
    OUT.mkdir(parents=True, exist_ok=True)
    TG.to_parquet(OUT / "team_games.parquet", index=False)
    P.to_parquet(OUT / "teammate_deltas.parquet", index=False)
    news = TG[TG.n_news > 0]
    summ = {"audit": cov["audit"], "team_games": int(len(TG)), "team_games_report_covered": int(TG.covered.sum()),
            "team_games_with_news": int(len(news)), "news_absences": int(TG.n_news.sum()),
            "unlisted_rotation_dnps": int(TG.n_unlisted.sum()), "carryover_listed_rotation": int(TG.n_carry.sum()),
            "news_team_games_by_class": news.cls.value_counts().to_dict(),
            "teammate_rows": int(len(P)),
            "d_pts_top_per_teamgame": P.groupby(["event_id", "team"]).d_points.max().describe().round(3).to_dict()}
    (OUT / "signals_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))
    return summ


# ------------------------------------------------------------------------------------------- quotes + stats
def load_quotes() -> pd.DataFrame:
    q = pov.quotes(DEV, at="open")
    gm = pd.concat([pd.read_parquet(DATA / f"games_{s}.parquet")[["event_id", "start_utc"]] for s in DEV])
    q = q.merge(gm.drop_duplicates("event_id"), on="event_id", how="left")
    q["date"] = pd.to_datetime(q.start_utc, utc=True).dt.tz_convert(ET).dt.date
    return q


def boot_mean(vals, keys, seed=0):
    bs = np.array([(w * vals).sum() / w.sum() for w in pov._boot_idx(keys, NB, seed)])
    return bs


def roi_trial(d: pd.DataFrame, side: str) -> dict:
    if len(d) == 0:
        return {"bets": 0}
    ov = side == "over"
    won = d.over.values if ov else 1 - d.over.values
    dec = (d.d_o if ov else d.d_u).values
    p_side = (d.p_over if ov else 1 - d.p_over).values
    pr = np.where(won == 1, dec - 1, -1.0)
    keys = d.date.astype(str).values
    bs = boot_mean(pr, keys)
    dates = np.sort(d.date.unique())
    med = dates[len(dates) // 2]
    h1, h2 = pr[(d.date < med).values].mean(), pr[(d.date >= med).values].mean()
    res = {"bets": int(len(d)), "dates": int(len(dates)), "units": float(pr.sum()), "roi": float(pr.mean()),
           "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float((bs <= 0).mean()),
           "win_rate": float(won.mean()), "hit_minus_novig": float(won.mean() - p_side.mean()),
           "halves": [float(h1), float(h2)], "mean_actual_minus_line": float((d.actual - d.L).mean())}
    res["passed"] = bool(res["p"] < BAR and h1 > 0 and h2 > 0 and res["roi"] > 0)
    return res


def slope_trial(d: pd.DataFrame, x: str, y: str) -> dict:
    d = d[[x, y, "date"]].dropna()
    X, Y = d[x].values, d[y].values

    def sl(w):
        xm, ym = (w * X).sum() / w.sum(), (w * Y).sum() / w.sum()
        return (w * (X - xm) * (Y - ym)).sum() / (w * (X - xm) ** 2).sum()
    b = sl(np.ones(len(d)))
    bs = np.array([sl(w) for w in pov._boot_idx(d.date.astype(str).values, NB, 0)])
    dates = np.sort(d.date.unique())
    med = dates[len(dates) // 2]
    h = [sl((d.date < med).values.astype(float)), sl((d.date >= med).values.astype(float))]
    res = {"n": int(len(d)), "dates": int(len(dates)), "slope": float(b),
           "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float((bs <= 0).mean()),
           "halves": [float(h[0]), float(h[1])], "x_sd": float(X.std())}
    res["passed"] = bool(res["p"] < BAR and h[0] > 0 and h[1] > 0 and b > 0)
    return res


def slope_ci(d, x, y):
    r = slope_trial(d, x, y)
    return {k: r[k] for k in ("n", "slope", "ci")}


# ------------------------------------------------------------------------------------------- hot streaks
def hot_box() -> pd.DataFrame:
    p = pov.load_box([2019, 2020, 2021, 2022] + DEV)
    b3 = []
    for s in [2019, 2020, 2021, 2022] + DEV:
        b = pd.read_parquet(DATA / f"box_{s}.parquet", columns=["event_id", "athlete_id", "stat", "value"])
        b = b[b.stat.isin(["threePointFieldGoalsMade", "threePointFieldGoalsAttempted"])]
        b3.append(b.pivot_table(index=["event_id", "athlete_id"], columns="stat", values="value", aggfunc="first").reset_index())
    b3 = pd.concat(b3).rename(columns={"threePointFieldGoalsMade": "tpm", "threePointFieldGoalsAttempted": "tpa"})
    p = p.merge(b3, on=["event_id", "athlete_id"], how="left")
    p = p.sort_values(["athlete_id", "start_utc"]).reset_index(drop=True)
    ga = p.groupby("athlete_id", sort=False)
    p["base20"] = ga.points.transform(lambda x: x.shift(6).rolling(20, min_periods=10).mean())
    p["sum5"] = ga.points.transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())
    p["hot"] = (p.sum5 - 5 * p.base20) / (5 + K_HOT)
    tpm6 = ga.tpm.transform(lambda x: x.shift(6).rolling(20, min_periods=10).sum())
    tpa6 = ga.tpa.transform(lambda x: x.shift(6).rolling(20, min_periods=10).sum())
    pct = tpm6 / tpa6.where(tpa6 > 0)
    m5 = ga.tpm.transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())
    a5 = ga.tpa.transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())
    p["hot3"] = (m5 - a5 * pct) / (5 + K_HOT)
    p["pct3"] = pct
    # minutes-adjusted: points-per-minute residual over the last 5 x this game's... (descriptive only)
    p["ppm20"] = ga.points.transform(lambda x: x.shift(6).rolling(20, min_periods=10).sum()) / \
        ga.minutes.transform(lambda x: x.shift(6).rolling(20, min_periods=10).sum())
    min5 = ga.minutes.transform(lambda x: x.shift(1).rolling(5, min_periods=5).sum())
    p["hot_ppm"] = (p.sum5 - min5 * p.ppm20) / (5 + K_HOT)
    return p


def hot_cutoffs(p):
    out = {}
    for s in DEV:
        v = p[(p.season == s - 1)].hot.dropna()
        v3 = p[(p.season == s - 1)].hot3.dropna()
        out[s] = {"p10": float(v.quantile(.1)), "p90": float(v.quantile(.9)), "sd": float(v.std()), "n": int(len(v)),
                  "sd3": float(v3.std())}
    return out


def reliability(p=None) -> dict:
    if p is None:
        p = hot_box()
    a, b, o, e = [], [], [], []
    for _, x in p.groupby("athlete_id", sort=False):
        pts, base = x.points.values, x.base20.values
        # base20 at game t uses games t-25..t-6; for a block starting at game j use the baseline of game j+5
        # (= games j-20..j-1, all before the block)
        j = 0
        while j + 10 <= len(x):
            bb = base[j + 5] if j + 5 < len(x) else np.nan
            if np.isfinite(bb):
                r = pts[j:j + 10] - bb
                a.append(r[:5].mean()); b.append(r[5:].mean()); o.append(r[0::2].mean()); e.append(r[1::2].mean())
                j += 10
            else:
                j += 1
    a, b, o, e = map(np.array, (a, b, o, e))
    rng = np.random.default_rng(0)
    res = {}
    for nm_, (u, v) in {"first5_vs_next5": (a, b), "odd_vs_even": (o, e)}.items():
        r = float(np.corrcoef(u, v)[0, 1])
        bs = []
        for _ in range(1000):
            k = rng.integers(0, len(u), len(u))
            bs.append(np.corrcoef(u[k], v[k])[0, 1])
        res[nm_] = {"r": r, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "blocks": int(len(u))}
    # same with the market-relevant question: does the residual of the last 5 predict the next game's residual vs
    # a baseline that INCLUDES the last 5 (what a recent-form line uses)?
    x = p.dropna(subset=["hot", "base20", "sum5"]).copy()
    x["b25"] = (x.base20 * 20 + x.sum5) / 25
    res["hot_vs_next_resid_incl_recent"] = float(np.corrcoef(x.hot, x.points - x.b25)[0, 1])
    res["hot_vs_next_resid_excl_recent"] = float(np.corrcoef(x.hot, x.points - x.base20)[0, 1])
    res["n_player_games"] = int(len(x))
    y = x.dropna(subset=["hot3", "pct3", "tpa"])
    res["hot3_vs_next_3pm_resid"] = float(np.corrcoef(y.hot3, y.tpm - y.tpa * y.pct3)[0, 1])
    y = x.dropna(subset=["hot_ppm", "ppm20"])
    res["hot_ppm_vs_next_ppm_resid_x_min"] = float(np.corrcoef(y.hot_ppm, y.points - y.minutes * y.ppm20)[0, 1])
    res["hot_ppm_vs_next_resid_incl_recent"] = float(np.corrcoef(y.hot_ppm, y.points - y.b25)[0, 1])
    return res


# ------------------------------------------------------------------------------------------- run
def run(ledger: bool) -> dict:
    TG = pd.read_parquet(OUT / "team_games.parquet")
    P = pd.read_parquet(OUT / "teammate_deltas.parquet")
    q = load_quotes()
    res = {"bar": BAR, "k": N_TRIALS, "quotes": int(len(q))}
    # ---- A
    qa = q.merge(P, on=["event_id", "athlete_id"], how="inner")
    res["A_join"] = {"teammate_rows": int(len(P)), "teammate_quotes": int(len(qa)),
                     "teammate_quotes_by_stat": qa.stat.value_counts().to_dict()}
    qa = qa[qa.stat.isin(SMAP)].copy()
    qa["delta"] = np.select([qa.stat == s for s in SMAP], [qa[f"d_{s}"] for s in SMAP])
    qa["proj0"] = np.select([qa.stat == s for s in SMAP], [qa[f"proj0_{s}"] for s in SMAP])
    qa["resid"] = qa.actual - qa.L

    def top(stat):
        x = qa[qa.stat == stat].sort_values("delta", ascending=False)
        return x.drop_duplicates(["event_id", "team"])

    t_pts = top("points")
    A1 = t_pts[t_pts.delta >= X_PTS]
    A2 = pd.concat([top("rebounds").query("delta >= @X_REB"), top("assists").query("delta >= @X_AST")])
    A3 = qa[qa.delta >= qa.stat.map(A3_THR)]
    A5 = A1[A1.cls == "EARLY"]
    T = {"A1_pts_top_over": roi_trial(A1, "over"), "A2_reb_ast_top_over": roi_trial(A2, "over"),
         "A3_all_teammates_over": roi_trial(A3, "over"),
         "A4_slope_resid_on_dpts": slope_trial(qa[qa.stat == "points"], "delta", "resid"),
         "A5_pts_top_over_EARLY": roi_trial(A5, "over")}
    # descriptive by class
    desc = {}
    for c in ("EARLY", "MID", "LATE"):
        desc[c] = {"A1": roi_trial(A1[A1.cls == c], "over"), "A3": roi_trial(A3[A3.cls == c], "over"),
                   "A4": slope_ci(qa[(qa.stat == "points") & (qa.cls == c)], "delta", "resid")}
    # incorporation: line_open this game minus his previous quoted open line in the same stat (any game), vs delta
    qs = q[q.stat.isin(SMAP)].sort_values(["athlete_id", "stat", "start_utc"]).copy()
    qs["prevL"] = qs.groupby(["athlete_id", "stat"]).L.shift(1)
    qs["prev_date"] = qs.groupby(["athlete_id", "stat"]).date.shift(1)
    inc = qa.merge(qs[["event_id", "athlete_id", "stat", "prevL", "prev_date"]], on=["event_id", "athlete_id", "stat"], how="left")
    inc["dL"] = inc.L - inc.prevL
    inc = inc[(pd.to_datetime(inc.date) - pd.to_datetime(inc.prev_date)).dt.days.between(1, 10)]
    incorp = {}
    for c in ("ALL", "EARLY", "MID", "LATE"):
        x = inc[(inc.stat == "points") & ((inc.cls == c) if c != "ALL" else True)]
        incorp[c] = slope_ci(x, "delta", "dL")
    # realized: did the sim delta come true? (actual - proj0) ~ delta
    qa["act_minus_proj0"] = qa.actual - qa.proj0_points
    real = {c: slope_ci(qa[(qa.stat == "points") & ((qa.cls == c) if c != "ALL" else True)], "delta", "act_minus_proj0")
            for c in ("ALL", "EARLY", "MID", "LATE")}
    # 2023-24 'current' (pre-game-looking) price: descriptive
    qc = pov.quotes([2024], at="current")
    qc = qc.merge(q[["event_id", "date"]].drop_duplicates("event_id"), on="event_id", how="left")
    qac = qc.merge(P, on=["event_id", "athlete_id"], how="inner")
    qac = qac[qac.stat == "points"].copy()
    qac["delta"] = qac.d_points
    tc = qac.sort_values("delta", ascending=False).drop_duplicates(["event_id", "team"])
    cur = {"A1_2024_current": roi_trial(tc[tc.delta >= X_PTS], "over"),
           "A1_2024_open_same_season": roi_trial(A1[A1.season_q == 2024], "over")}
    # open -> current move (2023-24) of teammate points lines vs delta, by class
    oc = qa[(qa.season_q == 2024) & (qa.stat == "points")].merge(
        qc[qc.stat == "points"][["event_id", "athlete_id", "L"]].rename(columns={"L": "Lcur"}), on=["event_id", "athlete_id"])
    oc["move"] = oc.Lcur - oc.L
    cur["open_to_current_move_slope"] = {c: slope_ci(oc[(oc.cls == c) if c != "ALL" else oc.cls.notna()], "delta", "move")
                                         for c in ("ALL", "EARLY", "MID", "LATE")}
    res["A"] = {"trials": T, "by_class": desc, "incorporation_open_vs_prev_open": incorp,
                "realized_slope_actual_minus_proj0": real, "current_2024": cur,
                "A1_by_season": {int(s): roi_trial(A1[A1.season_q == s], "over") for s in DEV},
                "A1_class_counts": A1.cls.value_counts().to_dict(),
                "news_teamgames_class_counts": TG[TG.n_news > 0].cls.value_counts().to_dict(),
                "news_teamgames_with_points_quote_class": t_pts.cls.value_counts().to_dict()}
    A1.to_csv(OUT / "bets_A1.csv", index=False); A3.to_csv(OUT / "bets_A3.csv", index=False)
    # ---- B
    p = hot_box()
    cut = hot_cutoffs(p)
    qb = q[q.stat == "points"].merge(p[["event_id", "athlete_id", "hot", "hot3", "season"]], on=["event_id", "athlete_id"], how="inner")
    qb = qb[qb.hot.notna()].copy()
    qb["p10"] = qb.season.map({s: c["p10"] for s, c in cut.items()})
    qb["p90"] = qb.season.map({s: c["p90"] for s, c in cut.items()})
    qb["z"] = qb.hot / qb.season.map({s: c["sd"] for s, c in cut.items()})
    qb["z3"] = qb.hot3 / qb.season.map({s: c["sd3"] for s, c in cut.items()})
    qb["resid"] = qb.actual - qb.L
    B1, B2 = qb[qb.hot >= qb.p90], qb[qb.hot <= qb.p10]
    TB = {"B1_hot_over": roi_trial(B1, "over"), "B2_cold_under": roi_trial(B2, "under"),
          "B3_slope_resid_on_zhot": slope_trial(qb, "z", "resid")}
    res["B"] = {"cutoffs": cut, "quotes_with_hot": int(len(qb)), "trials": TB,
                "desc": {"B1_under_side": roi_trial(B1, "under"), "B2_over_side": roi_trial(B2, "over"),
                         "slope_resid_on_z3pct": slope_ci(qb.dropna(subset=["z3"]), "z3", "resid"),
                         "slope_line_minus_base_on_zhot": None}}
    # how much of the hot streak is already in the line: L - base20 ~ z
    p2 = p[["event_id", "athlete_id", "base20"]]
    qb2 = qb.merge(p2, on=["event_id", "athlete_id"])
    qb2["L_minus_base"] = qb2.L - qb2.base20
    res["B"]["desc"]["slope_line_minus_base_on_zhot"] = slope_ci(qb2, "z", "L_minus_base")
    res["B"]["desc"]["slope_actual_minus_base_on_zhot"] = slope_ci(qb2.assign(y=qb2.actual - qb2.base20), "z", "y")
    B1.to_csv(OUT / "bets_B1.csv", index=False); B2.to_csv(OUT / "bets_B2.csv", index=False)
    res["B"]["reliability"] = reliability(p)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    if ledger:
        sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        allt = [("nba_injury_teammate_props", k, v) for k, v in T.items()] + [("nba_hot_streaks", k, v) for k, v in TB.items()]
        with LEDGER.open("a") as f:
            for n, (ds, k, v) in enumerate(allt, 1):
                rec = {"ts": ts, "strategy": f"{ds}_{k}_dev", "sha": sha, "dataset": ds,
                       "hypothesis": "pre-registered in notes/nba_injury_teammate_props.md", "source": "brief 2026-10-30",
                       "status": "ok", "split": "DEV", "trial_number": n, "study_trials": N_TRIALS, "bonferroni_bar": BAR,
                       "trial": k, "price": "ESPN open", **{kk: vv for kk, vv in v.items()}}
                f.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({"A": T, "B": TB}, indent=1, default=str))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["signals", "run", "reliability"])
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    if a.cmd == "signals":
        build_signals()
    elif a.cmd == "run":
        run(a.ledger)
    else:
        print(json.dumps(reliability(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
