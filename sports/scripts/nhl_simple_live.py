"""LIVE path for the NHL simple model (scripts/nhl_simple_model.py, L34). PREDICTION ONLY -- never scores anything.

    ../.venv_quant/bin/python scripts/nhl_simple_live.py fit                 # once: dev coefficients -> results/
    ../.venv_quant/bin/python scripts/nhl_simple_live.py predict [--date YYYY-MM-DD] [--json] [--line 6.5]

    from nhl_simple_live import predict, p_over; rows = predict("2026-10-26")

COEFFICIENTS: `fit` re-runs nhl_simple_model.build_frame() on the DEV ESPN seasons (2021-22 warm-up, 2022-23..2024-25)
  and fits the same OLS stack (goals ~ r_gf + g_opp(ACT) + b2b_t + b2b_o + S1 + S2 + S3) on seasons 2023..2025, i.e.
  the dev walk-forward's last fit plus 2024-25. Style z-scores use those rows' mean/SD (A1 relative-to-league form).
  Written to results/nhl_simple_live_fit.json. No sealed data is read.

LIVE FEATURES: NHL web API (api-web.nhle.com) schedule + boxscores, regular season only (as in dev), games strictly
  before the target date. History = 2024-25 + the current season. **2025-26 is SKIPPED** (NHL 2026 is sealed in
  data/frozen/2026-10-14/MANIFEST.json and L34 never scored it); consequence: ratings early in 2026-27 rest on the
  current season + a faint 2024-25 tail (ridge half-life 90 d, goalie 365 d), noisier than in dev.
  Same feature formulas as dev, evaluated at the target date only:
    r_gf ridge (half-life 90 d, lambda 10), goalie GSAA/shot shrunk with 1,500 shots (half-life 365 d) x 30,
    b2b, S1/S2/S3 decayed 60 d relative to the league's decayed mean, z with the fit's training SDs.
  GOALIES: no probable-starter feed is used (the NHL API publishes none; DailyFaceoff is not scraped). The goalie is
  the pre-registered STRICT projection = most starts in the team's last 10 games (tie -> most recent), which is the
  variant the dev trials T1-T6 scored. Rookies / goalies with no history rate 0 (league average).
Output per game: date, game_id, start_utc, home, away, lam_home, lam_away (regulation goals), p_home (incl. OT/SO,
  ties split 50/50), exp_total (final goals incl. the OT/SO goal), proj starters, warnings. p_over(row, line) gives
  P(over | no push) under the dev Poisson + OT rule.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch

API = "https://api-web.nhle.com/v1"
FIT = ROOT / "results" / "nhl_simple_live_fit.json"
LIVE = ROOT / "data" / "nhl_simple_model" / "live"
HIST = LIVE / "games.parquet"     # one row per finished regular-season game (parsed boxscore)
GOAL = LIVE / "goalies.parquet"   # one row per goalie appearance
SEASONS = [(Date(2024, 9, 15), Date(2025, 7, 1)), (Date(2026, 9, 15), None)]   # 2025-26 sealed -> skipped
H_TEAM, LAM, H_G, K_G, H_S, SHOTS = 90.0, 10.0, 365.0, 1500.0, 60.0, 30.0
FEAT = ["r_gf", "g_opp", "b2b_t", "b2b_o", "S1_shots", "S2_pen", "S3_sh"]
G = np.arange(0, 16)


# ------------------------------------------------------------------------------------------------ coefficients
def cmd_fit() -> int:
    import statsmodels.api as sm
    import nhl_simple_model as m
    t, _ = m.build_frame()
    t = t.dropna(subset=["r_gf", "sf_pg", "o_sa_pg", "pim_pg", "o_pim_pg", "pimd_pg", "shpct"])
    tr = t[(t.season >= 2023) & (t.season <= 2025)]
    zc = m.zfit(tr)
    trs = m.style(tr, zc).assign(g_opp=lambda d: d.g_opp_act)
    f = sm.OLS(trs.goals.to_numpy(float), sm.add_constant(trs[FEAT].to_numpy(float))).fit()
    out = {"fit_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seasons": [2023, 2024, 2025],
           "n": int(f.nobs), "sigma": float(np.sqrt(f.scale)), "names": ["const"] + FEAT,
           "params": [float(x) for x in f.params], "bse": [float(x) for x in f.bse], "zc": zc}
    FIT.write_text(json.dumps(out, indent=1))
    for n_, b, se in zip(out["names"], f.params, f.bse):
        print(f"  {n_:10s} {b:+.4f} ({se:.4f})")
    print(f"wrote {FIT} (n {out['n']})")
    return 0


# ------------------------------------------------------------------------------------------------ data
def _schedule(start: Date, end: Date) -> list[dict]:
    """All games (any state) from the weekly schedule endpoint between start and end inclusive."""
    today = datetime.now(timezone.utc).date()
    out, d = [], start
    while d <= end:
        p = fetch(f"{API}/schedule/{d.isoformat()}", cache=d + timedelta(days=8) < today)
        for day in p.get("gameWeek", []):
            for g in day.get("games", []):
                if start.isoformat() <= day["date"] <= end.isoformat():
                    g["_date"] = day["date"]
                    out.append(g)
        nxt = p.get("nextStartDate")
        if not nxt or Date.fromisoformat(nxt) <= d:
            break
        d = Date.fromisoformat(nxt)
    return list({g["id"]: g for g in out}.values())


def _parse_box(g: dict, b: dict) -> tuple[dict, list[dict]]:
    last = (g.get("gameOutcome") or {}).get("lastPeriodType") or (b.get("gameOutcome") or {}).get("lastPeriodType")
    row = {"game_id": int(g["id"]), "date": g["_date"], "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
           "home_score": int(b["homeTeam"]["score"]), "away_score": int(b["awayTeam"]["score"]),
           "overtime": last in ("OT", "SO")}
    gl = []
    st = b.get("playerByGameStats", {})
    for side, k in (("home", "homeTeam"), ("away", "awayTeam")):
        sk = st.get(k, {})
        row[f"{side}_pim"] = float(sum((p.get("pim") or 0) for grp in ("forwards", "defense") for p in sk.get(grp, [])))
        gs = sk.get("goalies", [])
        for x in gs:
            gl.append({"game_id": int(g["id"]), "date": g["_date"], "team": row[side], "athlete_id": int(x["playerId"]),
                       "name": (x.get("name") or {}).get("default"), "sv": float(x.get("saves") or 0),
                       "shots_ag": float(x.get("shotsAgainst") or 0), "starter": bool(x.get("starter"))})
        row[f"{side}_shots_ag"] = float(sum(float(x.get("shotsAgainst") or 0) for x in gs))
        s = [x for x in gs if x.get("starter")] or sorted(gs, key=lambda x: -float(x.get("shotsAgainst") or 0))
        row[f"{side}_starter"] = int(s[0]["playerId"]) if s else None
    return row, gl


def refresh_history(upto: Date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parsed finished regular-season games before `upto` (incremental parquet cache)."""
    LIVE.mkdir(parents=True, exist_ok=True)
    H = pd.read_parquet(HIST) if HIST.exists() else pd.DataFrame()
    GL = pd.read_parquet(GOAL) if GOAL.exists() else pd.DataFrame()
    have = set(H.game_id) if len(H) else set()
    new_r, new_g = [], []
    for a, b in SEASONS:
        end = min(b or upto, upto - timedelta(days=1))
        if end < a:
            continue
        for g in _schedule(a, end):
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL") or int(g["id"]) in have:
                continue
            try:
                box = fetch(f"{API}/gamecenter/{g['id']}/boxscore", cache=True)
                r, gl = _parse_box(g, box)
            except Exception as e:  # noqa: BLE001
                print(f"box {g['id']} failed: {type(e).__name__}: {e}")
                continue
            new_r.append(r)
            new_g += gl
    if new_r:
        H = pd.concat([H, pd.DataFrame(new_r)], ignore_index=True).drop_duplicates("game_id")
        GL = pd.concat([GL, pd.DataFrame(new_g)], ignore_index=True).drop_duplicates(["game_id", "athlete_id"])
        H.to_parquet(HIST)
        GL.to_parquet(GOAL)
    H = H[pd.to_datetime(H.date) < pd.Timestamp(upto)]
    GL = GL[pd.to_datetime(GL.date) < pd.Timestamp(upto)]
    return H, GL


def team_rows(H: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side, o in (("home", "away"), ("away", "home")):
        win_ot = H.overtime & (H[f"{side}_score"] > H[f"{o}_score"])
        rows.append(pd.DataFrame({
            "game_id": H.game_id, "date": pd.to_datetime(H.date), "team": H[side], "opp": H[o],
            "home_ind": int(side == "home"), "goals": H[f"{side}_score"] - win_ot.astype(int),
            "goals_final": H[f"{side}_score"], "starter": H[f"{side}_starter"],
            "shots_ag": H[f"{side}_shots_ag"], "shots_for": H[f"{o}_shots_ag"],
            "pim": H[f"{side}_pim"], "pim_drawn": H[f"{o}_pim"]}))
    return pd.concat(rows, ignore_index=True).sort_values(["date", "game_id", "home_ind"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ features at d
def features_at(t: pd.DataFrame, GL: pd.DataFrame, d: pd.Timestamp) -> dict:
    dn = (d - t.date).dt.days.to_numpy(float)          # days before d (> 0)
    # ridge
    teams = sorted(set(t.team) | set(t.opp))
    ti = {k: i for i, k in enumerate(teams)}
    nT = len(teams)
    X = np.zeros((len(t), 2 * nT + 2))
    X[np.arange(len(t)), t.team.map(ti).to_numpy()] = 1
    X[np.arange(len(t)), nT + t.opp.map(ti).to_numpy()] = 1
    X[:, -2], X[:, -1] = t.home_ind, 1
    w = 0.5 ** (dn / H_TEAM)
    pen = np.r_[np.full(2 * nT, LAM), 1e-6, 1e-6]
    beta = np.linalg.solve((X * w[:, None]).T @ X + np.diag(pen), (X * w[:, None]).T @ t.goals.to_numpy(float))
    # goalies
    gd = (d - pd.to_datetime(GL.date)).dt.days.to_numpy(float)
    wg = 0.5 ** (gd / H_G)
    lg = (wg * GL.sv).sum() / (wg * GL.shots_ag).sum()
    gs = pd.DataFrame({"a": GL.athlete_id, "g": wg * (GL.sv - GL.shots_ag * lg), "s": wg * GL.shots_ag}).groupby("a").sum()
    grat = (gs.g / (gs.s + K_G)).to_dict()
    # decayed team style + league means (row-weighted == dev's day-grouped form)
    ws = 0.5 ** (dn / H_S)
    lgm = {c: (ws * t[n]).sum() / (ws * (t[den] if den else 1)).sum()
           for c, n, den in (("sf_pg", "shots_for", None), ("sa_pg", "shots_ag", None), ("pim_pg", "pim", None),
                             ("pimd_pg", "pim_drawn", None), ("shpct", "goals_final", "shots_for"))}
    team = {}
    for k, idx in t.groupby("team").indices.items():
        r = t.iloc[np.sort(idx)]
        wv = ws[np.sort(idx)]
        st = list(r.starter)[-10:]
        cnt = {a: st.count(a) for a in st}
        proj = [a for a in reversed(st) if cnt[a] == max(cnt.values())][0] if st else None
        team[k] = {"n": len(r), "last": r.date.max(),
                   "sf_pg": (wv * r.shots_for).sum() / wv.sum() - lgm["sf_pg"],
                   "sa_pg": (wv * r.shots_ag).sum() / wv.sum() - lgm["sa_pg"],
                   "pim_pg": (wv * r.pim).sum() / wv.sum() - lgm["pim_pg"],
                   "pimd_pg": (wv * r.pim_drawn).sum() / wv.sum() - lgm["pimd_pg"],
                   "shpct": (wv * r.goals_final).sum() / (wv * r.shots_for).sum() - lgm["shpct"],
                   "proj": proj, "proj_share": cnt.get(proj, 0) / max(len(st), 1),
                   "att": beta[ti[k]], "def": beta[nT + ti[k]]}
    gname = GL.drop_duplicates("athlete_id", keep="last").set_index("athlete_id").name.to_dict()
    return {"beta_home": beta[-2], "beta_const": beta[-1], "grat": grat, "team": team, "gname": gname, "n_rows": len(t)}


def _lam(F: dict, fit: dict, t: str, o: str, home: int, d: pd.Timestamp) -> float:
    zc, T, O = fit["zc"], F["team"][t], F["team"][o]
    z = lambda v, c: (v - zc[c][0]) / zc[c][1]
    x = {"r_gf": F["beta_const"] + T["att"] - O["def"] + F["beta_home"] * home,
         "g_opp": F["grat"].get(O["proj"], 0.0) * SHOTS,
         "b2b_t": int((d - T["last"]).days == 1), "b2b_o": int((d - O["last"]).days == 1),
         "S1_shots": z(T["sf_pg"], "sf_pg") + z(O["sa_pg"], "sa_pg"),
         "S2_pen": z(O["pim_pg"], "pim_pg") + z(T["pimd_pg"], "pimd_pg"),
         "S3_sh": z(T["shpct"], "shpct")}
    p = fit["params"]
    return max(p[0] + sum(b * x[n] for b, n in zip(p[1:], FEAT)), 0.5), x


def _joint(lh: float, la: float):
    from scipy.stats import poisson
    J = poisson.pmf(G, lh)[:, None] * poisson.pmf(G, la)[None, :]
    H, A = np.meshgrid(G, G, indexing="ij")
    return J, H, A


def p_home(lh: float, la: float) -> float:
    J, H, A = _joint(lh, la)
    return float((J * (H > A)).sum() + 0.5 * (J * (H == A)).sum())


def p_over(row: dict, line: float) -> float:
    """P(final total > line | no push); final = regulation goals + 1 if tied after regulation (dev rule)."""
    J, H, A = _joint(row["lam_home"], row["lam_away"])
    T = H + A + (H == A)
    po, pu = (J * (T > line)).sum(), (J * (T < line)).sum()
    return float(po / (po + pu))


def predict(date_et: str) -> list[dict]:
    fit = json.loads(FIT.read_text())
    d = pd.Timestamp(date_et)
    games = [g for g in _schedule(d.date(), d.date()) if g.get("gameType") == 2]
    if not games:
        return []
    H, GL = refresh_history(d.date())
    t = team_rows(H)
    F = features_at(t, GL, d)
    out = []
    for g in sorted(games, key=lambda g: g["startTimeUTC"]):
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        warn = []
        if h not in F["team"] or a not in F["team"]:
            out.append({"date": date_et, "game_id": g["id"], "start_utc": g["startTimeUTC"], "home": h, "away": a,
                        "warnings": ["team has no history"]})
            continue
        lh, xh = _lam(F, fit, h, a, 1, d)
        la, xa = _lam(F, fit, a, h, 0, d)
        for side, k in (("home", h), ("away", a)):
            T = F["team"][k]
            if T["proj"] not in F["grat"]:
                warn.append(f"{k}: projected goalie has no rating history (rated 0)")
            if T["proj_share"] <= 0.5:
                warn.append(f"{k}: projected goalie started only {T['proj_share']:.0%} of last 10 (tandem)")
        row = {"date": date_et, "game_id": g["id"], "start_utc": g["startTimeUTC"], "home": h, "away": a,
               "lam_home": round(lh, 4), "lam_away": round(la, 4), "p_home": round(p_home(lh, la), 4),
               "goalie_home": F["gname"].get(F["team"][h]["proj"]), "goalie_away": F["gname"].get(F["team"][a]["proj"]),
               "goalie_source": "STRICT last-10 modal starter (no probable-goalie feed)",
               "feat_home": {k: round(float(v), 4) for k, v in xh.items()},
               "feat_away": {k: round(float(v), 4) for k, v in xa.items()}, "warnings": warn}
        J, Hh, Aa = _joint(lh, la)
        row["exp_total"] = round(float((J * (Hh + Aa + (Hh == Aa))).sum()), 3)
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "predict"])
    ap.add_argument("--date", default=pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%d"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--line", type=float, default=6.5)
    a = ap.parse_args()
    if a.cmd == "fit":
        return cmd_fit()
    rows = predict(a.date)
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    for r in rows:
        if "lam_home" not in r:
            print(r)
            continue
        print(f"{r['away']:>4s} @ {r['home']:4s} lam {r['lam_away']:.2f}-{r['lam_home']:.2f} p_home {r['p_home']:.3f} "
              f"E[total] {r['exp_total']:.2f} P(over {a.line}) {p_over(r, a.line):.3f}  G {r['goalie_away']} / "
              f"{r['goalie_home']}  {'; '.join(r['warnings'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
