"""Player x opponent / venue / former-team history vs NBA player prop lines.

    python scripts/nba_player_opp_venue.py run [--ledger]
    python scripts/nba_player_opp_venue.py reliability       # box scores only, no lines
    python scripts/nba_player_opp_venue.py example --player "Nikola Jokic" --opp NO

Pre-registration: notes/nba_player_opp_venue.md (written before any signal was joined to a line/outcome).
DEV = ESPN seasons 2023-2025 (2022-23..2024-25). SEALED 2026 is never loaded by `run`.
Decision price = ESPN OPENING line/price ('current' is contaminated by in-game lines in 2023 and 2025).
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
DATA = ROOT / "data" / "espn" / "nba"
LEDGER = ROOT / "lab" / "ledger.jsonl"
OUT = ROOT / "results" / "nba_player_opp_venue"
DATASET = "nba_player_opp_venue"
HIST = [2019, 2020, 2021, 2022]
DEV = [2023, 2024, 2025]
BOOK = {2023: "DraftKings", 2024: "DraftKings", 2025: "ESPN BET", 2026: "ESPN BET"}
MARKETS = {"total points": "points", "total rebounds": "rebounds", "total assists": "assists",
           "total points, rebounds, and assists": "pra"}
STATS = ["points", "rebounds", "assists", "pra"]
K, MIN_BASE, BASE_N, MIN_MEET, REV_G = 10.0, 10, 20, 3, 20
N_TRIALS = 5
BAR = 0.05 / N_TRIALS
NB = 4000


# ------------------------------------------------------------------ box + signals

def load_box(seasons) -> pd.DataFrame:
    rows = []
    for s in seasons:
        g = pd.read_parquet(DATA / f"games_{s}.parquet")
        g = g[(g.season == s) & (g.season_type != "preseason") & (g.status == "STATUS_FINAL")]
        nba = g.home_team_id.value_counts()
        nba = set(nba[nba >= 20].index)  # drops All-Star / exhibition teams (2025 All-Star mini-tournament)
        g = g[g.home_team_id.isin(nba) & g.away_team_id.isin(nba)]
        b = pd.read_parquet(DATA / f"box_{s}.parquet")
        b = b[b.stat.isin(["minutes", "points", "rebounds", "assists"])]
        w = b.pivot_table(index=["event_id", "athlete_id", "team_id"], columns="stat", values="value",
                          aggfunc="first").reset_index()
        w["name"] = w.athlete_id.map(b.drop_duplicates("athlete_id").set_index("athlete_id")["name"])
        w = w.merge(g[["event_id", "season", "season_type", "start_utc", "home_team_id", "away_team_id",
                       "home", "away", "venue_id", "venue"]], on="event_id", how="inner")
        rows.append(w)
    p = pd.concat(rows, ignore_index=True)
    p = p[p.minutes > 0].copy()
    p["home_g"] = p.team_id == p.home_team_id
    p["opp_id"] = np.where(p.home_g, p.away_team_id, p.home_team_id)
    p["opp"] = np.where(p.home_g, p.away, p.home)
    p["pra"] = p.points + p.rebounds + p.assists
    p["date"] = pd.to_datetime(p.start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    p = p.sort_values(["athlete_id", "start_utc"]).reset_index(drop=True)
    dup = p.duplicated(["athlete_id", "date"], keep=False)
    if dup.any():  # ESPN 2019-20 has GS games on 2020-03-01 twice (14 rows); drop, they are history only
        print(f"dropping {int(dup.sum())} player-rows with two games on one date "
              f"(seasons {sorted(p[dup].season.unique().tolist())})", file=sys.stderr)
        p = p[~dup].reset_index(drop=True)
    return p


def add_signals(p: pd.DataFrame) -> pd.DataFrame:
    """All signals use rows strictly earlier in the player's own chronological sequence (one game per date)."""
    ga = p.groupby("athlete_id", sort=False)
    p["n_prior"] = ga.cumcount()
    for st in STATS:
        base = ga[st].transform(lambda x: x.shift(1).rolling(BASE_N, min_periods=MIN_BASE).mean())
        p[f"base_{st}"] = base
        p[f"r_{st}"] = p[st] - base
    for key, tag in (("opp_id", "opp"), ("venue_id", "ven")):
        g2 = p.groupby(["athlete_id", key], sort=False)
        for st in STATS:
            r = p[f"r_{st}"]
            csum = r.fillna(0).groupby([p.athlete_id, p[key]]).cumsum() - r.fillna(0)
            ccnt = r.notna().astype(int).groupby([p.athlete_id, p[key]]).cumsum() - r.notna().astype(int)
            p[f"S_{tag}_{st}"] = csum / (ccnt + K)
            p[f"n_{tag}_{st}"] = ccnt
        p[f"meet_{tag}"] = g2.cumcount()
    # revenge: >= REV_G regular-season games for this opponent in a PRIOR season, not on it now
    rs = p[p.season_type == "regular-season"].groupby(["athlete_id", "team_id", "season"]).size().rename("g").reset_index()
    rs = rs[rs.g >= REV_G]
    first_stint = rs.groupby(["athlete_id", "team_id"]).season.min().rename("stint_season")
    p = p.merge(first_stint, left_on=["athlete_id", "opp_id"], right_index=True, how="left")
    p["revenge"] = (p.stint_season < p.season) & (p.team_id != p.opp_id)
    # last date he played FOR the opponent, before this game
    rev_rank = np.full(len(p), -1)
    for aid, g in p.groupby("athlete_id", sort=False):
        meets_since = {}
        for i, team, opp, rv in zip(g.index, g.team_id.values, g.opp_id.values, g.revenge.values):
            if rv:
                rev_rank[i] = meets_since.get(opp, 0)
                meets_since[opp] = meets_since.get(opp, 0) + 1
            meets_since[team] = 0  # playing for a team resets "meetings since leaving"
    p["rev_meet"] = rev_rank  # 0 = first meeting after leaving
    return p


def cutoffs(p: pd.DataFrame) -> dict:
    """Per scored season s: p10/p90/SD of each signal among eligible rows of season s-1."""
    out = {}
    for s in sorted(p.season.unique()):
        prev = p[p.season == s - 1]
        if prev.empty:
            continue
        c = {}
        for tag in ("opp", "ven"):
            for st in STATS:
                e = prev[(prev[f"n_{tag}_{st}"] >= MIN_MEET) & prev[f"base_{st}"].notna()]
                if tag == "ven":
                    e = e[~e.home_g]
                v = e[f"S_{tag}_{st}"].dropna()
                if len(v) > 200:
                    c[(tag, st)] = (float(v.quantile(.1)), float(v.quantile(.9)), float(v.std()), int(len(v)))
        out[int(s)] = c
    return out


# ------------------------------------------------------------------ quotes

def quotes(seasons, at="open") -> pd.DataFrame:
    lc, dc = ("line_open", "dec_open") if at == "open" else ("line", "dec")
    out = []
    for s in seasons:
        g = pd.read_parquet(DATA / f"graded_{s}.parquet")
        g = g[g.market_n.isin(MARKETS) & (g.provider == BOOK[s]) & g.athlete_id.notna() & g[lc].notna()]
        key = ["event_id", "athlete_id", "market_n", lc]
        o = g[g.side == "over"].drop_duplicates(key).set_index(key)
        u = g[g.side == "under"].drop_duplicates(key).set_index(key)
        j = o[[dc, "actual"]].rename(columns={dc: "d_o"}).join(u[[dc]].rename(columns={dc: "d_u"}), how="inner").reset_index()
        j = j[(j.d_o > 1) & (j.d_u > 1) & j.actual.notna()].rename(columns={lc: "L"})
        io, iu = 1 / j.d_o, 1 / j.d_u
        j["p_over"] = io / (io + iu)
        j["hold"] = io + iu - 1
        j["dist"] = (j.p_over - .5).abs()
        j = j.sort_values("dist").drop_duplicates(["event_id", "athlete_id", "market_n"])
        j["stat"] = j.market_n.map(MARKETS)
        j["season_q"] = s
        out.append(j)
    q = pd.concat(out, ignore_index=True)
    q = q[q.actual != q.L].copy()  # pushes leave the sample
    q["over"] = (q.actual > q.L).astype(float)
    return q


def join(q: pd.DataFrame, p: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    cols = ["event_id", "athlete_id", "name", "team_id", "opp_id", "opp", "venue_id", "home_g", "season", "date",
            "revenge", "rev_meet"] + STATS + [c for c in p.columns if c.startswith(("S_", "n_opp_", "n_ven_", "base_"))]
    m = q.merge(p[cols], on=["event_id", "athlete_id"], how="left", indicator=True)
    audit = {"quotes": int(len(q)), "matched_box": int((m._merge == "both").sum())}
    m = m[m._merge == "both"].drop(columns="_merge")
    m["stat_box"] = m.apply(lambda r: r[r["stat"]], axis=1)
    audit["actual_eq_box"] = float((m.stat_box == m.actual).mean())
    return m, audit


# ------------------------------------------------------------------ stats

def _boot_idx(keys, n, seed):
    rng = np.random.default_rng(seed)
    codes, uniq = pd.factorize(pd.Series(keys))
    G = len(uniq)
    for _ in range(n):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        yield w[codes]


def bets(df, side_col):
    d = df.copy()
    ov = d[side_col] == "over"
    d["won"] = np.where(ov, d.over, 1 - d.over)
    d["dec"] = np.where(ov, d.d_o, d.d_u)
    d["p_side"] = np.where(ov, d.p_over, 1 - d.p_over)
    d["profit"] = np.where(d.won == 1, d.dec - 1, -1.0)
    return d


def roi_test(d: pd.DataFrame, seed=0) -> dict:
    if len(d) == 0:
        return {"bets": 0}
    pr = d.profit.values
    res = {"bets": int(len(d)), "units": float(pr.sum()), "roi": float(pr.mean()), "win_rate": float(d.won.mean()),
           "mkt_p_mean": float(d.p_side.mean()), "hit_minus_mkt": float(d.won.mean() - d.p_side.mean())}
    for name, keys in (("pg", d.event_id.astype(str) + "_" + d.athlete_id.astype(str)), ("date", d.date.astype(str))):
        bs = np.array([(w * pr).sum() / w.sum() for w in _boot_idx(keys.values, NB, seed)])
        res[f"p_{name}"] = float((bs <= 0).mean())
        res[f"ci_{name}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    med = np.sort(d.date.unique())[len(d.date.unique()) // 2]
    res["half1"], res["half2"] = float(d[d.date < med].profit.mean()), float(d[d.date >= med].profit.mean())
    res["by_season"] = {int(s): {"n": int(len(t)), "roi": float(t.profit.mean())} for s, t in d.groupby("season")}
    res["p"] = max(res["p_pg"], res["p_date"])
    res["ci_lo"], res["ci_hi"] = res["ci_date"]
    res["passed"] = bool(res["p"] < BAR and res["half1"] > 0 and res["half2"] > 0)
    return res


def logit_fit(X, y, w=None, it=25):
    w = np.ones(len(y)) if w is None else w
    b = np.zeros(X.shape[1])
    for _ in range(it):
        mu = 1 / (1 + np.exp(-(X @ b)))
        g = X.T @ (w * (y - mu))
        H = (X * (w * mu * (1 - mu))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def logit_test(d: pd.DataFrame, xcol: str, seed=0) -> dict:
    lp = np.log(d.p_over / (1 - d.p_over)).values
    X = np.column_stack([np.ones(len(d)), lp, d[xcol].values])
    y = d.over.values
    b = logit_fit(X, y)
    res = {"n": int(len(d)), "coef_const": float(b[0]), "coef_logit_p": float(b[1]), "coef_z": float(b[2])}
    for name, keys in (("pg", d.event_id.astype(str) + "_" + d.athlete_id.astype(str)), ("date", d.date.astype(str))):
        bs = np.array([logit_fit(X, y, w)[2] for w in _boot_idx(keys.values, NB, seed)])
        res[f"p_{name}"] = float((bs <= 0).mean())
        res[f"ci_{name}"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    med = np.sort(d.date.unique())[len(d.date.unique()) // 2]
    h = [d[d.date < med], d[d.date >= med]]
    res["half1"], res["half2"] = [float(logit_fit(np.column_stack([np.ones(len(t)), np.log(t.p_over / (1 - t.p_over)),
                                                                     t[xcol]]), t.over.values)[2]) for t in h]
    res["p"] = max(res["p_pg"], res["p_date"])
    res["ci_lo"], res["ci_hi"] = res["ci_date"]
    res["passed"] = bool(res["p"] < BAR and res["half1"] > 0 and res["half2"] > 0)
    # what 1 SD of z is worth in hit rate at p = .5
    res["dhit_per_sd_at_half"] = float(1 / (1 + np.exp(-b[2])) - .5)
    return res


def decile_bets(m: pd.DataFrame, cut: dict, tag: str, stats, away_only=False) -> pd.DataFrame:
    out = []
    for st in stats:
        d = m[(m.stat == st) & (m[f"n_{tag}_{st}"] >= MIN_MEET) & m[f"base_{st}"].notna()].copy()
        if away_only:
            d = d[~d.home_g]
        parts = []
        for s, t in d.groupby("season"):
            c = cut.get(int(s), {}).get((tag, st))
            if c is None:
                continue
            lo, hi = c[0], c[1]
            sig = t[f"S_{tag}_{st}"]
            parts.append(t[sig >= hi].assign(side="over"))
            parts.append(t[sig <= lo].assign(side="under"))
        if parts:
            out.append(pd.concat(parts))
    return bets(pd.concat(out, ignore_index=True), "side") if out else pd.DataFrame()


# ------------------------------------------------------------------ commands

def build(seasons_q, at="open"):
    p = add_signals(load_box(HIST + DEV))
    cut = cutoffs(p)
    q = quotes(seasons_q, at)
    m, audit = join(q, p)
    return p, cut, m, audit


def run(ledger: bool, at: str = "open") -> dict:
    p, cut, m, audit = build(DEV, at)
    res = {"price": at, "audit": audit, "bar": BAR, "k_shrink": K}
    res["base_rates"] = {st: {"n": int(len(t)), "over_hit": float(t.over.mean()), "p_over_mean": float(t.p_over.mean()),
                              "hold_mean": float(t.hold.mean())} for st, t in m.groupby("stat")}
    res["cutoffs"] = {str(s): {f"{a}_{b}": v for (a, b), v in c.items()} for s, c in cut.items()}

    t1 = decile_bets(m, cut, "opp", ["points"])
    t2 = decile_bets(m, cut, "ven", ["points"], away_only=True)
    t3 = decile_bets(m, cut, "opp", STATS)
    res["T1_points_Sopp_decile"] = roi_test(t1, 1)
    res["T1_by_side"] = {s: roi_test(t, 11) for s, t in t1.groupby("side")}
    res["T2_points_Svenue_away_decile"] = roi_test(t2, 2)
    res["T2_by_side"] = {s: roi_test(t, 12) for s, t in t2.groupby("side")}
    res["T3_allstats_Sopp_decile"] = roi_test(t3, 3)
    res["T3_by_stat"] = {s: roi_test(t, 13) for s, t in t3.groupby("stat")}

    e = m[(m.stat == "points") & (m.n_opp_points >= MIN_MEET) & m.base_points.notna()].copy()
    e["z"] = [m_ / cut[int(s)][("opp", "points")][2] for m_, s in zip(e.S_opp_points, e.season)]
    res["T4_logit_points_zopp"] = logit_test(e, "z", 4)
    # mechanism check: does S_opp predict the raw residual (actual - baseline)? and actual - line?
    e["res_line"] = e.actual - e.L
    e["res_base"] = e.points - e.base_points
    res["T4_desc"] = {"corr_z_res_base": float(np.corrcoef(e.z, e.res_base)[0, 1]),
                      "corr_z_res_line": float(np.corrcoef(e.z, e.res_line)[0, 1]),
                      "corr_z_line_minus_base": float(np.corrcoef(e.z, e.L - e.base_points)[0, 1])}

    r = m[(m.stat == "points") & m.revenge].assign(side="over")
    t5 = bets(r, "side")
    res["T5_revenge_points_over"] = roi_test(t5, 5)
    res["T5_desc"] = {
        "first_meeting": roi_test(t5[t5.rev_meet == 0], 15),
        "later_meetings": roi_test(t5[t5.rev_meet > 0], 16),
        "mean_actual_minus_line": float((t5.actual - t5.L).mean()),
        "mean_actual_minus_line_first": float((t5[t5.rev_meet == 0].actual - t5[t5.rev_meet == 0].L).mean()),
        "nonrevenge_mean_actual_minus_line": float((m[(m.stat == "points") & ~m.revenge].actual
                                                    - m[(m.stat == "points") & ~m.revenge].L).mean()),
        "pra_over": roi_test(bets(m[(m.stat == "pra") & m.revenge].assign(side="over"), "side"), 17),
    }
    res["verdicts"] = {k: res[k]["passed"] for k in ("T1_points_Sopp_decile", "T2_points_Svenue_away_decile",
                                                       "T3_allstats_Sopp_decile", "T4_logit_points_zopp",
                                                       "T5_revenge_points_over")}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"results_{at}.json").write_text(json.dumps(res, indent=1, default=float))
    for name, d in (("T1", t1), ("T2", t2), ("T3", t3), ("T5", t5)):
        d.to_csv(OUT / f"bets_{name}_{at}.csv", index=False)
    if ledger and at == "open":
        sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        hyp = {"T1_points_Sopp_decile": "points: over top / under bottom decile of shrunk player-vs-opponent residual",
               "T2_points_Svenue_away_decile": "points, away games: same on player-at-arena residual",
               "T3_allstats_Sopp_decile": "pts/reb/ast/PRA pooled: S_opp deciles per stat",
               "T4_logit_points_zopp": "logit over_hit ~ logit(no-vig p_open) + z(S_opp), coef on z > 0",
               "T5_revenge_points_over": "over points vs former team (>=20 g in a prior season)"}
        with LEDGER.open("a") as f:
            for n, k in enumerate(hyp, 1):
                r_ = {kk: vv for kk, vv in res[k].items() if kk != "passed"}
                f.write(json.dumps({"ts": ts, "strategy": f"npov_{k}", "sha": sha, "dataset": DATASET,
                                    "hypothesis": f"pre-registered in notes/nba_player_opp_venue.md; {hyp[k]}",
                                    "source": "user hypothesis 10-29", "status": "ok",
                                    "split": "DEV 2022-23..2024-25, opening line", "trial_number": n,
                                    "study_trials": N_TRIALS, "bonferroni_bar": BAR, **r_,
                                    "passed": res[k]["passed"], "verdict": "PASS" if res[k]["passed"] else "FAIL"},
                                   default=float) + "\n")
    return res


def reliability() -> dict:
    p = add_signals(load_box(HIST + DEV))  # sealed 2026 not loaded
    out = {}
    for key, tag in (("opp_id", "opp"), ("venue_id", "ven")):
        d = p[p.r_points.notna()]
        if tag == "ven":
            d = d[~d.home_g]
        d = d.assign(half=d.season % 2)
        a = d.groupby(["athlete_id", key, "half"]).r_points.agg(["mean", "count"]).unstack("half")
        a = a[(a[("count", 0)] >= MIN_MEET) & (a[("count", 1)] >= MIN_MEET)]
        x, y = a[("mean", 0)].values, a[("mean", 1)].values
        rng = np.random.default_rng(0)
        bs = [np.corrcoef(x[i], y[i])[0, 1] for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
        cov = float(np.cov(x, y)[0, 1])
        sig2 = float(d.r_points.var())
        out[tag] = {"pairs": int(len(x)), "r": float(np.corrcoef(x, y)[0, 1]),
                    "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    "cov_halves_tau2": cov, "sigma2_game": sig2,
                    "k_implied": float(sig2 / cov) if cov > 0 else None,
                    "mean_meetings_per_half": float(a[("count", 0)].mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reliability.json").write_text(json.dumps(out, indent=1))
    return out


def example(player: str, opp: str) -> pd.DataFrame:
    p = add_signals(load_box(HIST + DEV))
    q = quotes(DEV, "open")
    q = q[q.stat == "points"][["event_id", "athlete_id", "L", "p_over", "over"]]
    d = p[(p.name == player) & (p.opp == opp)][["date", "season", "season_type", "home_g", "minutes", "points",
                                                "base_points", "r_points", "S_opp_points", "n_opp_points",
                                                "event_id", "athlete_id"]]
    return d.merge(q, on=["event_id", "athlete_id"], how="left").drop(columns=["event_id", "athlete_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "reliability", "example"])
    ap.add_argument("--ledger", action="store_true")
    ap.add_argument("--at", default="open", choices=["open", "current"])
    ap.add_argument("--player", default="Nikola Jokic")
    ap.add_argument("--opp", default="NO")
    a = ap.parse_args()
    if a.cmd == "run":
        print(json.dumps(run(a.ledger, a.at), indent=1, default=float))
    elif a.cmd == "reliability":
        print(json.dumps(reliability(), indent=1))
    else:
        pd.set_option("display.width", 200)
        print(example(a.player, a.opp).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
