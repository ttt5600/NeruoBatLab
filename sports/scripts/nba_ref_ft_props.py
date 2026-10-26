"""L6b: does the referee crew's free-throw tendency move FT-dependent player props vs the market?

    python scripts/nba_ref_ft_props.py run [--ledger]

PRE-REGISTERED 2026-10-25, written after checking which prop markets exist (counts and columns only)
and BEFORE any crew rating was joined to a prop line or an outcome.

Data check (2026-10-25). ESPN prop markets in DEV: Total Points, Rebounds, Assists, 3PM, Turnovers,
Steals, Blocks (+ combos in 2025). There is NO free-throws-made prop in any dev season, so only
player POINTS is tested. Points props with opening line+price (line_open/dec_open):
2023 (2022-23) 7 books, 2024 (2023-24) 7 books, 2025 (2024-25) ESPN BET only; 2022 has 750 rows (unused).
'open' is ESPN's opener; its timestamp is unknown, so whether it predates the ~9am ET crew announcement
is NOT verified. 'current' (line/dec) is approximately the close (data.md quirk, not verified).
SEALED 2026 (2025-26) is never loaded. Box scores / crews 2022 serve as burn-in only.

Feature (walk-forward, strictly prior DATES only).
  Game FT residual y_g = (home_fta + away_fta) - mean total FTA of the previous 500 dev games.
  Referee rating r_j = sum(y_g over j's prior games) / (n_j + K), K = 100 (ridge shrinkage; C0 in
  nba_referees.md puts crew SD ~1.5 FTA vs game SD ~9 -> per-ref var ~0.75 -> K ~ 81/0.75 ~ 100).
  Crew score c = sum of the listed referees' r (4-official games: mean * 3; unseen referee = 0).
  Crew z = c / SD(c over all games on earlier dates). Scored games: seasons 2023-2025 only.
High-FTA player: mean FTA over his games (minutes > 0) on earlier dates in the current + previous
  season >= 5.0, with >= 15 such games.
Market quote (one obs per player-game; protocol rule 4): one book per season -- DraftKings 2023 and
  2024 (sharpest per data.md), ESPN BET 2025 (only book). Pair over/under at the OPENING line; when a
  book has several lines for a player-game, keep the pair whose de-vigged open p_over is closest to 0.5.
  De-vig: proportional two-way (two-way prices near -110, method barely matters). Outcome graded
  from the actual points vs line_open; pushes refund and leave the hit-rate sample.

Trials (Bonferroni bar = 0.05 / 4 = 0.0125), all on scored seasons 2023-2025:
  T1 OVER high-FTA player points at the opening line/price when crew z >= +1. Metric ROI at dec_open;
     p one-sided = share of 4000 game-date-cluster bootstrap resamples with units <= 0.
  T2 UNDER, same players, when crew z <= -1. Same metric/p.
  T3 Slope: OLS of (over_hit - p_over_open) on crew z, all high-FTA player-games (non-push).
     p two-sided from a game-date-cluster bootstrap of the slope (4000).
  G2 Game level: OLS of (home+away points - close total from refs_dev.mkt_total) on crew z.
     p two-sided, date-cluster bootstrap.
Pass = p < 0.0125 AND same sign in both time halves (split at the median scored date) AND, for
  T1/T2, ROI > 0 in each half.
Descriptive (no p in the bar):
  G1 positive control: slope of game FT residual (total FTA - previous-500 mean) on crew z. If the
     walk-forward rating carries no signal here, the trials have no power and are uninformative.
  P0 mechanism: slope of (player FTA - his prior FTA/g) and (points - his prior pts/g) on crew z,
     high-FTA players.
  CLV: slope of (close line - open line) on crew z for the same quotes; positive = market moves
     points lines toward the crew after the open (i.e. the crew is priced by the close).
  T3 at the close (line/dec) and T3 on all players (any FTA rate).
Power note written before results: crew SD ~1.5 total FTA -> z=+1 is ~+0.75 FTA per team; a
  high-FTA player takes ~25-30% of team FTA -> ~+0.2 FTA ~ +0.16 points vs a points SD ~7, i.e. an
  expected hit-rate shift of ~1 pp. Prop holds ~4-5% at -110/-115. A pass is a priori unlikely;
  a null cannot rule out a ~1 pp effect.
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
OUT = ROOT / "results" / "nba_ref_ft_props.json"
DATASET = "nba_ref_ft_props"
BURN, SCORED = [2022], [2023, 2024, 2025]
BOOK = {2023: "DraftKings", 2024: "DraftKings", 2025: "ESPN BET"}
K, BASE_N, ZTHR, FTA_MIN, MIN_G = 100.0, 500, 1.0, 5.0, 15
N_TRIALS, BAR = 4, 0.05 / 4
NB = 4000


def crew_z() -> pd.DataFrame:
    d = pd.read_parquet(DATA / "refs_dev.parquet")
    d = d[d.season.isin(BURN + SCORED) & (d.n_refs >= 3)].copy()
    d["tfta"] = d.home_fta + d.away_fta
    d = d[d.tfta.notna()].sort_values("start_utc").reset_index(drop=True)
    d["date"] = (d.start_utc.dt.tz_convert("America/New_York")).dt.date
    d["base"] = d.tfta.shift(1).rolling(BASE_N, min_periods=100).mean()
    d["y"] = d.tfta - d.base
    S, N = {}, {}
    crews, hist = [], []
    zs = np.full(len(d), np.nan)
    cs = np.full(len(d), np.nan)
    for date, idx in d.groupby("date", sort=True).groups.items():
        sd = np.std(hist) if len(hist) > 200 else np.nan
        for i in idx:
            refs = [d.at[i, f"ref{k}"] for k in range(1, 5)]
            refs = [r for r in refs if isinstance(r, str) and r]
            c = np.mean([S.get(r, 0.0) / (N.get(r, 0) + K) for r in refs]) * 3
            cs[i] = c
            zs[i] = c / sd if sd and sd > 0 else np.nan
        for i in idx:                      # update AFTER the whole date
            hist.append(cs[i])
            if np.isfinite(d.at[i, "y"]):
                for k in range(1, 5):
                    r = d.at[i, f"ref{k}"]
                    if isinstance(r, str) and r:
                        S[r] = S.get(r, 0.0) + d.at[i, "y"]
                        N[r] = N.get(r, 0) + 1
    d["crew"], d["z"] = cs, zs
    return d


def players(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in BURN + SCORED:
        b = pd.read_parquet(DATA / f"box_{s}.parquet")
        b = b[b.stat.isin(["minutes", "points", "freeThrowsAttempted"])]
        w = b.pivot_table(index=["event_id", "athlete_id"], columns="stat", values="value", aggfunc="first").reset_index()
        w["season"] = s
        rows.append(w)
    p = pd.concat(rows, ignore_index=True)
    p = p.merge(games[["event_id", "date", "start_utc"]], on="event_id", how="inner")
    p = p[p.minutes > 0].sort_values("start_utc")
    out = []
    for aid, g in p.groupby("athlete_id", sort=False):
        g = g.copy()
        # prior games on earlier dates, current + previous season
        fta_prior, pts_prior, n_prior = [], [], []
        dates, seas = g.date.values, g.season.values
        fta, pts = g.freeThrowsAttempted.values, g.points.values
        for i in range(len(g)):
            m = (dates < dates[i]) & ((seas == seas[i]) | (seas == seas[i] - 1))
            n_prior.append(m.sum())
            fta_prior.append(fta[m].mean() if m.any() else np.nan)
            pts_prior.append(pts[m].mean() if m.any() else np.nan)
        g["n_prior"], g["fta_prior"], g["pts_prior"] = n_prior, fta_prior, pts_prior
        out.append(g)
    return pd.concat(out, ignore_index=True)


def quotes() -> pd.DataFrame:
    out = []
    for s in SCORED:
        g = pd.read_parquet(DATA / f"graded_{s}.parquet")
        g = g[(g.market_n == "total points") & (g.provider == BOOK[s]) & g.athlete_id.notna() & g.line_open.notna()]
        key = ["event_id", "athlete_id", "line_open"]
        o = g[g.side == "over"].drop_duplicates(key).set_index(key)
        u = g[g.side == "under"].drop_duplicates(key).set_index(key)
        j = o[["dec_open", "dec", "line", "actual"]].rename(columns={"dec_open": "do_o", "dec": "dc_o", "line": "line_c"}) \
            .join(u[["dec_open", "dec", "line"]].rename(columns={"dec_open": "do_u", "dec": "dc_u", "line": "line_cu"}), how="inner").reset_index()
        j = j[(j.do_o > 1) & (j.do_u > 1) & j.actual.notna()]
        io, iu = 1 / j.do_o, 1 / j.do_u
        j["p_open"] = io / (io + iu)
        ok = (j.dc_o > 1) & (j.dc_u > 1) & (j.line_c == j.line_cu)
        j["p_close"] = np.where(ok, (1 / j.dc_o) / (1 / j.dc_o + 1 / j.dc_u), np.nan)
        j["dist"] = (j.p_open - 0.5).abs()
        j = j.sort_values("dist").drop_duplicates(["event_id", "athlete_id"])
        j["season"] = s
        out.append(j)
    q = pd.concat(out, ignore_index=True)
    q["push"] = q.actual == q.line_open
    q["over"] = (q.actual > q.line_open).astype(float)
    q["push_c"] = q.actual == q.line_c
    q["over_c"] = (q.actual > q.line_c).astype(float)
    return q


def boot(df, fn, n=NB, seed=0):
    rng = np.random.default_rng(seed)
    keys = df["date"].values
    ud = np.unique(keys)
    groups = {k: np.where(keys == k)[0] for k in ud}
    out = np.empty(n)
    for b in range(n):
        pick = rng.choice(ud, len(ud))
        idx = np.concatenate([groups[k] for k in pick])
        out[b] = fn(df.iloc[idx])
    return out


def slope(df, y, x="z"):
    xv, yv = df[x].values, df[y].values
    xv = xv - xv.mean()
    return float((xv * (yv - yv.mean())).sum() / (xv ** 2).sum())


def slope_test(df, y, seed=0):
    obs = slope(df, y)
    bs = boot(df, lambda t: slope(t, y), seed=seed)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return {"n": int(len(df)), "slope": obs, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": float(min(1, p))}


def bet_test(df, seed=0):
    units = lambda t: t["profit"].sum()
    obs = units(df)
    bs = boot(df, units, seed=seed)
    return {"bets": int(len(df)), "units": float(obs), "roi": float(obs / len(df)), "win": float(df.won.mean()),
            "mkt_p_mean": float(df.p_side.mean()), "p": float((bs <= 0).mean())}


def halves(df, fn):
    med = np.sort(df.date.unique())[len(df.date.unique()) // 2]
    return [fn(df[df.date < med]), fn(df[df.date >= med])]


def run(ledger: bool) -> dict:
    g = crew_z()
    sc = g[g.season.isin(SCORED) & g.z.notna()].copy()
    res = {"games_scored": int(len(sc)), "z_sd_check": float(sc.z.std())}
    # G1 positive control, G2 trial
    res["G1_fta_resid"] = slope_test(sc[sc.y.notna()], "y")
    sc["tot_res"] = sc.home_score + sc.away_score - sc.mkt_total
    g2 = sc[sc.tot_res.notna()]
    res["G2_total_vs_close"] = slope_test(g2, "tot_res", seed=1)
    res["G2_halves"] = halves(g2, lambda t: slope(t, "tot_res"))

    pl = players(g)
    q = quotes().merge(pl, on=["event_id", "athlete_id"], how="inner", suffixes=("", "_b"))
    q = q.merge(sc[["event_id", "z"]], on="event_id", how="inner")
    res["quotes_joined"] = int(len(q))
    hi = q[(q.n_prior >= MIN_G) & (q.fta_prior >= FTA_MIN)].copy()
    res["hi_fta_player_games"] = int(len(hi))
    res["hi_fta_players"] = int(hi.athlete_id.nunique())

    # P0 mechanism
    hi["fta_dev"] = hi.freeThrowsAttempted - hi.fta_prior
    hi["pts_dev"] = hi.points - hi.pts_prior
    res["P0_fta"] = slope_test(hi, "fta_dev", seed=2)
    res["P0_pts"] = slope_test(hi, "pts_dev", seed=3)

    # T3
    h = hi[~hi.push].copy()
    h["res"] = h.over - h.p_open
    res["T3_slope"] = slope_test(h, "res", seed=4)
    res["T3_halves"] = halves(h, lambda t: slope(t, "res"))
    res["T3_by_season"] = {int(s): slope(t, "res") for s, t in h.groupby("season")}

    # T1/T2 bets
    def mk(df, side):
        d = df[~df.push].copy()
        if side == "over":
            d["won"], d["dec"], d["p_side"] = d.over, d.do_o, d.p_open
        else:
            d["won"], d["dec"], d["p_side"] = 1 - d.over, d.do_u, 1 - d.p_open
        d["profit"] = np.where(d.won == 1, d.dec - 1, -1.0)
        return d
    t1 = mk(hi[hi.z >= ZTHR], "over")
    t2 = mk(hi[hi.z <= -ZTHR], "under")
    res["T1_over_hiZ"] = bet_test(t1, seed=5)
    res["T1_halves_roi"] = halves(t1, lambda t: float(t.profit.mean()))
    res["T2_under_loZ"] = bet_test(t2, seed=6)
    res["T2_halves_roi"] = halves(t2, lambda t: float(t.profit.mean()))
    res["T1_hit_minus_mkt"] = float(t1.won.mean() - t1.p_side.mean())
    res["T2_hit_minus_mkt"] = float(t2.won.mean() - t2.p_side.mean())

    # descriptives
    hc = hi[(~hi.push_c) & hi.p_close.notna()].copy()
    hc["res_c"] = hc.over_c - hc.p_close
    res["T3_close"] = slope_test(hc, "res_c", seed=7)
    hi["dline"] = hi.line_c - hi.line_open
    res["CLV_line_move"] = slope_test(hi[hi.dline.notna()], "dline", seed=8)
    a = q[(~q.push) & (q.n_prior >= MIN_G)].copy()
    a["res"] = a.over - a.p_open
    res["T3_all_players"] = slope_test(a, "res", seed=9)
    res["base_over_hit_minus_mkt_hi"] = float(h.over.mean() - h.p_open.mean())

    def passed(p, hv, roi_h=None):
        same = all(np.sign(x) == np.sign(hv[0]) for x in hv)
        ok = p < BAR and same
        if roi_h is not None:
            ok = ok and all(x > 0 for x in roi_h)
        return bool(ok)
    verdict = {
        "T1": passed(res["T1_over_hiZ"]["p"], res["T1_halves_roi"], res["T1_halves_roi"]),
        "T2": passed(res["T2_under_loZ"]["p"], res["T2_halves_roi"], res["T2_halves_roi"]),
        "T3": passed(res["T3_slope"]["p"], res["T3_halves"]),
        "G2": passed(res["G2_total_vs_close"]["p"], res["G2_halves"]),
    }
    res["pass"] = verdict
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1, default=float))
    if ledger:
        sha = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = [
            ("T1_over_hiFTA_crewZ_ge1", res["T1_over_hiZ"], "Over high-FTA player points at open when crew FT z>=+1"),
            ("T2_under_hiFTA_crewZ_le-1", res["T2_under_loZ"], "Under high-FTA player points at open when crew FT z<=-1"),
            ("T3_slope_hiFTA_pts_resid_on_crewZ", res["T3_slope"], "Slope of (over - devig open p) on crew FT z, high-FTA players"),
            ("G2_total_minus_close_on_crewZ", res["G2_total_vs_close"], "Slope of game total - close total on crew FT z"),
        ]
        with LEDGER.open("a") as f:
            for n, (name, r, hyp) in enumerate(rows, 1):
                f.write(json.dumps({"ts": ts, "strategy": name, "sha": sha, "dataset": DATASET, "source": "lead L6b",
                                    "hypothesis": hyp, "status": "ok", "trial_number": n, "study_trials": N_TRIALS,
                                    "bonferroni_bar": BAR, "passed": verdict[name.split("_")[0]], **r}, default=float) + "\n")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--ledger", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.ledger), indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
