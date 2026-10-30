"""NBA player-level Monte Carlo game simulator, v1. PRICING AID ONLY -- places no orders, reads no keys.

    ../.venv_quant/bin/python scripts/nba_sim.py game  CLE ATL [--date 2026-10-29] [--out "A,B"] [--in "C"]
                                                   [--minutes "Player:30,Other:12"] [--n 20000] [--seed 7]
                                                   [--spread -3.5] [--total 238.5] [--props props.csv] [--no-refresh]
    ../.venv_quant/bin/python scripts/nba_sim.py delta CLE ATL --out "Jalen Johnson" [--date ...]
    ../.venv_quant/bin/python scripts/nba_sim.py validate --season 2024-25
    ../.venv_quant/bin/python scripts/nba_sim.py fit            # (re)fit the live params on dev seasons 2022-23..2024-25

Design and limitations: notes/nba_sim.md.  Argument order is AWAY HOME (as on a ticket: CLE @ ATL).

MODEL (all player/team state is built from games dated strictly before the game date):
  player state   decayed (half-life 180 d) sums of box stats + decayed minutes; EWMA(0.3) of minutes when he played.
  impact r_i     "BPM-lite" (nba_simple_model): on-court plus-minus/48 ~ per-48 box rates, minutes-weighted OLS on
                 player-seasons of seasons < S (>= 300 min); rates shrunk to replacement with 500 pseudo-minutes.
  stat rates     points/rebounds/assists per minute, shrunk to the player's ROLE prior (G/W/B from his assist:rebound
                 mix) with 150 pseudo-minutes.
  minutes        full rotation = players on the team in its last 10 games (+ ESPN roster adds/drops when live);
                 full minutes = EWMA minutes scaled to 240 (cap 42). OUT players' minutes go to available teammates
                 in proportion to m*(48-m) x 2 if same role; then rescaled to exactly 240; --minutes fixes a player.
  team mean      margin = c0 + c1*mu_ridge + c2*dP_full + c3*dP_absence ; total = k0 + k1*tot_ridge + k2*O_full + k3*O_abs
                 mu_ridge = decayed team ridge with home, b2b, rest diff, travel (nba_player_sim, frozen H=60, lam=5);
                 tot_ridge = decayed off/def points ridge (= recency-weighted pace x efficiency);
                 P = sum m_i r_i / 240 (minutes-weighted impact); O = sum m_i ppm_i (expected points from those players);
                 absence = tonight's set minus the full rotation. Coefficients: OLS on dev seasons (see `fit`).
  game noise     (margin, total) bivariate normal with the training residual SDs and correlation; integer scores;
                 ties go to a 5-minute OT. P(home) = share of sims home wins.
  player stats   per sim: minutes jittered (lognormal, SD fitted) and renormalised to 240; team points from the game
                 sim, team rebounds/assists = expectation x lognormal noise (CV fitted; assists scale with points);
                 split among players by Gamma/Dirichlet shares around minutes x rate (concentration kappa fitted so the
                 10-90 interval covers ~80% on the training season).
  NOT modelled   arena / opponent / revenge / body-clock effects (tested dead: notes/nba_player_opp_venue.md,
                 notes/nba_clock_calendar_arena.md); blowout minutes; foul trouble; defensive matchups.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import date as Date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_simple_model as nsm
from nba_player_sim import LOC, TI, km, ridge_preds

PARAMS = ROOT / "results" / "nba_sim_params.json"
OUTD = ROOT / "results" / "nba_sim"
CACHE = ROOT / "data" / "nba_sim"
STATS = nsm.STATS                     # points, fga, fta, 3pm, orb, drb, ast, tov, stl, blk, pf
I_PTS, I_ORB, I_DRB, I_AST = 0, 4, 5, 6
HP, KSHRINK, KROLE = nsm.HP, nsm.KSHRINK, 150.0
CAP, ROLE_W, DEFAULT_MIN = 42.0, 2.0, 15.0
REG_MIN = 24.0   # EWMA minutes at/above which a player is a regular: his absences are injuries (handled by OUT lists),
                 # below it his expected minutes are EWMA x participation (DNP-coach's-decision risk)
ALIAS = {"NOP": "NO", "GSW": "GS", "NYK": "NY", "SAS": "SA", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX", "BRK": "BKN"}
STAT3 = ("pts", "reb", "ast")


# --------------------------------------------------------------------------------------------------- minutes model
def role_of(ast_pm: float, reb_pm: float) -> str:
    x = ast_pm / max(ast_pm + reb_pm, 1e-9)
    return "G" if x > 0.40 else ("B" if x < 0.20 else "W")


def project_minutes(base: dict, roles: dict, absent: list, fixed: dict | None = None, cap: float = CAP,
                    total: float = 240.0) -> dict:
    """base: id -> minutes before absences; absent: [(minutes, role)] of rotation players who are out.
    Absent minutes go to available non-fixed players in proportion to m*(48-m) x ROLE_W if same role; then the
    non-fixed players are rescaled so the team sums to `total` (additive, weights m*(48-m)) with an individual cap."""
    fixed = {k: float(v) for k, v in (fixed or {}).items() if k in base}
    ids = list(base)
    if not ids:
        return {}
    m = np.array([max(float(base[i]), 0.5) for i in ids])
    fx = np.array([i in fixed for i in ids])
    ro = np.array([roles.get(i, "W") for i in ids])
    for am, ar in absent:
        if am <= 0 or (~fx).sum() == 0:
            continue
        w = m * np.maximum(48 - m, 1.0) * np.where(ro == ar, ROLE_W, 1.0) * ~fx
        m = m + am * w / w.sum()
    for k, i in enumerate(ids):
        if fx[k]:
            m[k] = fixed[i]
    # rescale to `total` ADDITIVELY with weights m*(48-m): mid-rotation minutes absorb most of the slack, stars and
    # deep bench move least (a multiplicative rescale over-projected 34+ mpg players by ~2 min in the 2024-25 backtest)
    rem = total - m[fx].sum()
    free = ~fx
    for _ in range(100):
        diff = rem - m[free].sum()
        if abs(diff) < 1e-9:
            break
        f = free & ((m < cap) if diff > 0 else (m > 0.5))
        if not f.any():
            break
        w = np.where(f, np.maximum(m * (48 - m), 1e-3), 0.0)
        m = np.clip(m + diff * w / w.sum(), 0.5, np.where(free, cap, np.inf))
        m[fx] = [fixed[i] for i, k in zip(ids, fx) if k]
    if free.any() and abs(m[free].sum() - rem) > 1e-6:   # too few players for the cap: spread the rest evenly
        m[free] += (rem - m[free].sum()) / free.sum()
    return dict(zip(ids, m))


# --------------------------------------------------------------------------------------------------- player state
class State:
    """Decayed per-player stat sums, updated date by date (nba_simple_model.player_pass conventions)."""

    def __init__(self):
        self.ST, self.TH, self.names = {}, {}, {}

    def update(self, d: int, team: str, rows) -> None:
        for x in rows:
            v = np.array([getattr(x, c) for c in STATS], float)
            s = self.ST.get(x.athlete_id)
            if s is None:
                s = {"s": np.zeros(len(STATS)), "m": 0.0, "gn": 0.0, "t": d, "em": x.minutes}
            dec = 0.5 ** ((d - s["t"]) / HP)
            s["s"], s["m"], s["gn"], s["t"] = s["s"] * dec + v, s["m"] * dec + x.minutes, s["gn"] * dec + 1, d
            s["em"] = 0.7 * s["em"] + 0.3 * x.minutes
            s["team"] = team
            self.ST[x.athlete_id] = s
            self.names[x.athlete_id] = x.name
        if rows:
            self.TH.setdefault(team, []).append({x.athlete_id for x in rows})

    def full_set(self, T: str) -> set:
        hist = self.TH.get(T, [])
        return {a for a in set().union(*hist[-10:]) if self.ST[a]["team"] == T} if hist else set()

    def participation(self, T: str, a) -> float:
        """Share of T's last 10 games he appeared in, counted from his first appearance for T in that window."""
        hist = self.TH.get(T, [])[-10:]
        idx = [i for i, S in enumerate(hist) if a in S]
        return len(idx) / (len(hist) - idx[0]) if idx else 0.0

    def prev_set(self, T: str) -> set:
        hist = self.TH.get(T, [])
        return set(hist[-1]) if hist else set()


def priors(box: pd.DataFrame, g: pd.DataFrame, season: int) -> dict:
    """BPM-lite betas (nsm.fit_bpm) and per-role per-minute pts/reb/ast priors from seasons < season."""
    ps = nsm.player_seasons(box, g)
    tr = ps[ps.season < season]
    if tr.empty:
        tr = ps[ps.season == season]
    bt = nsm.fit_bpm(tr)
    x = tr[tr.minutes >= 300].copy()
    x["reb"] = x.offensiveRebounds + x.defensiveRebounds
    x["role"] = [role_of(a / m, r / m) for a, r, m in zip(x.assists, x.reb, x.minutes)]
    rp = {}
    for ro, y in x.groupby("role"):
        rp[ro] = [float(y.points.sum() / y.minutes.sum()), float(y.reb.sum() / y.minutes.sum()),
                  float(y.assists.sum() / y.minutes.sum())]
    return {"bpm": bt, "role": rp}


def player_rates(st: State, a, d: int, pr: dict) -> dict:
    """rating r (BPM-lite per 48), ppm for the team-strength O term, role-shrunk pts/reb/ast per minute, em, mpg."""
    bt = pr["bpm"]
    beta, repl, rppm = np.array(bt["beta"]), np.array(bt["repl48"]) / 48, bt["repl_ppm"]
    s = st.ST.get(a)
    if s is None:
        rw = pr["role"]["W"]
        return {"r": bt["b0"] + repl @ beta * 48, "ppm": rppm, "em": DEFAULT_MIN, "mpg": 0.0, "role": "W",
                "pts": rw[0], "reb": rw[1], "ast": rw[2], "known": False}
    dec = 0.5 ** ((d - s["t"]) / HP)
    mm, sv = s["m"] * dec, s["s"] * dec
    rt = (sv + KSHRINK * repl) / (mm + KSHRINK)
    raw_ast, raw_reb = (sv[I_AST] + 20 * repl[I_AST]) / (mm + 20), (sv[I_ORB] + sv[I_DRB] + 20 * (repl[I_ORB] + repl[I_DRB])) / (mm + 20)
    ro = role_of(raw_ast, raw_reb)
    rp = pr["role"].get(ro, pr["role"]["W"])
    return {"r": float(bt["b0"] + rt @ beta * 48), "ppm": float((sv[I_PTS] + KSHRINK * rppm) / (mm + KSHRINK)),
            "em": float(s["em"]), "mpg": float(s["m"] / max(s["gn"], 1e-9)), "role": ro,
            "pts": float((sv[I_PTS] + KROLE * rp[0]) / (mm + KROLE)),
            "reb": float((sv[I_ORB] + sv[I_DRB] + KROLE * rp[1]) / (mm + KROLE)),
            "ast": float((sv[I_AST] + KROLE * rp[2]) / (mm + KROLE)), "known": True}


def side_inputs(st: State, d: int, pr: dict, full: set, avail: set, fixed: dict | None = None, team=None,
                known_play: bool = False) -> dict:
    """Minutes + strength for the full rotation and for tonight's set. Raw expected minutes: EWMA minutes when he
    played; for non-regulars (EWMA < REG_MIN) times his participation share unless known_play (backtest sets of
    players who did play). Absent rotation players' raw minutes are redistributed, then one rescale to 240."""
    R = {a: player_rates(st, a, d, pr) for a in full | avail}
    roles = {a: R[a]["role"] for a in R}
    if team is not None:
        for a in full | avail:
            R[a]["part"] = st.participation(team, a) if a in full else 1.0
    exp_min = lambda a: R[a]["em"] if R[a]["em"] >= REG_MIN or team is None else R[a]["em"] * max(R[a].get("part", 1.0), 0.1)
    m_full = project_minutes({a: exp_min(a) for a in full}, roles, [], None) if full else {}
    base = {a: (R[a]["em"] if known_play else exp_min(a)) for a in avail}
    absent = [(exp_min(a), roles[a]) for a in full - avail]
    m_av = project_minutes(base, roles, absent, fixed)
    P = lambda M: float(sum(M[a] * R[a]["r"] for a in M) / 240) if M else np.nan
    O = lambda M: float(sum(M[a] * R[a]["ppm"] for a in M)) if M else np.nan
    return {"R": R, "m_full": m_full, "m": m_av, "Pfull": P(m_full), "Pav": P(m_av), "Ofull": O(m_full), "Oav": O(m_av)}


# ----------------------------------------------------------------------------------------------- team-level ridges
def ridge_totals(g: pd.DataFrame, H=nsm.H_TEAM, lam=nsm.LAM, only_from=None) -> np.ndarray:
    """nsm.ridge_totals with an only_from switch (live: refit only for the game date)."""
    n = len(g)
    hi, ai = g.home.map(TI).to_numpy(), g.away.map(TI).to_numpy()
    X = np.zeros((2 * n, 62))
    X[:, 0] = 1
    r = np.arange(n)
    X[r, 1 + hi] = 1; X[r, 31 + ai] = 1; X[r, 61] = g.home_ind
    X[n + r, 1 + ai] = 1; X[n + r, 31 + hi] = 1
    y = np.r_[g.home_score.to_numpy(float), g.away_score.to_numpy(float)]
    dn = np.array([d.toordinal() for d in g.date])
    dn2 = np.r_[dn, dn]
    pen = np.r_[1e-6, np.full(60, lam), 1e-6]
    out = np.full(n, np.nan)
    for d in np.unique(dn):
        if only_from is not None and d < only_from:
            continue
        tr = (dn2 < d) & ~np.isnan(y)
        if tr.sum() < 100:
            continue
        w = 0.5 ** ((d - dn2[tr]) / H)
        Xt = X[tr]
        beta = np.linalg.solve((Xt * w[:, None]).T @ Xt + np.diag(pen), (Xt * w[:, None]).T @ y[tr])
        te = np.where(dn == d)[0]
        out[te] = X[te] @ beta + X[n + te] @ beta
    return out


# ------------------------------------------------------------------------------------------------------ simulation
def team_mean(p: dict, mu0, tot0, h: dict, a: dict) -> tuple[float, float]:
    cm, ct = p["margin"], p["total"]
    dPfull, dAbs = h["Pfull"] - a["Pfull"], (h["Pav"] - h["Pfull"]) - (a["Pav"] - a["Pfull"])
    Ofull, oAbs = h["Ofull"] + a["Ofull"], (h["Oav"] - h["Ofull"]) + (a["Oav"] - a["Ofull"])
    mu = cm["const"] + cm["mu0"] * mu0 + cm["dPfull"] * dPfull + cm["dAbs"] * dAbs
    tot = ct["const"] + ct["tot0"] * tot0 + ct["Ofull"] * Ofull + ct["oAbs"] * oAbs
    return float(mu), float(tot)


def sim_scores(mu, tot, p, n, rng):
    z = rng.standard_normal((n, 2))
    margin = mu + p["sigma_m"] * z[:, 0]
    total = tot + p["sigma_t"] * (p["rho"] * z[:, 0] + np.sqrt(1 - p["rho"] ** 2) * z[:, 1])
    hp, ap = np.rint((total + margin) / 2), np.rint((total - margin) / 2)
    tie = hp == ap
    k = int(tie.sum())
    if k:                                                  # 5-minute OT: ~10 pts each, winner by 1-7, coin flip
        o = np.rint(rng.normal(10, 2.5, k))
        w = rng.random(k) < 0.5
        lead = rng.integers(1, 8, k)
        hp[tie] += o + np.where(w, lead, 0)
        ap[tie] += o + np.where(w, 0, lead)
    return hp, ap


def sim_players(si: dict, team_pts, exp_pts_team, p, rng, n):
    """-> dict stat -> (n, k) array, minutes (n, k), ids. Team totals: pts from the game sim; reb/ast lognormal."""
    ids = [a for a in si["m"] if si["m"][a] > 0]
    if not ids:
        return None
    m = np.array([si["m"][a] for a in ids])
    sd = p["min_sd"]
    ms = m * np.exp(sd * rng.standard_normal((n, len(ids))) - sd ** 2 / 2)
    ms *= 240 / ms.sum(1, keepdims=True)
    out = {"ids": ids, "min": ms}
    R = si["R"]
    pts_scale = team_pts / max(exp_pts_team, 1e-9)
    for st in STAT3:
        rate = np.array([R[a][st] for a in ids])
        e = ms * rate
        exp_team = (m * rate).sum()
        if st == "pts":
            T = team_pts
        else:
            cv = p["cv"][st]
            T = exp_team * np.exp(cv * rng.standard_normal(n) - cv ** 2 / 2)
            if st == "ast":
                T = T * pts_scale
        s = e / e.sum(1, keepdims=True)
        gm = rng.gamma(np.maximum(p["kappa"][st] * s, 1e-3))
        out[st] = np.rint(T[:, None] * gm / gm.sum(1, keepdims=True))
    return out


def price(p: dict, mu0, tot0, h: dict, a: dict, n=20000, seed=7, players=True) -> dict:
    rng = np.random.default_rng(seed)
    mu, tot = team_mean(p, mu0, tot0, h, a)
    hp, ap = sim_scores(mu, tot, p, n, rng)
    res = {"mu": mu, "tot": tot, "hp": hp, "ap": ap, "margin": hp - ap, "total": hp + ap,
           "p_home": float((hp > ap).mean())}
    if players:
        eh = (mu + tot) / 2
        res["players_h"] = sim_players(h, hp, eh, p, rng, n)
        res["players_a"] = sim_players(a, ap, tot - eh, p, rng, n)
    return res


# ---------------------------------------------------------------------------------------------- historical pass
def load_hist(seasons, live_date=None):
    g = nsm.load_games(seasons, live_date)
    box = nsm.load_box(g, seasons)
    box["name"] = box["name"].fillna(box.athlete_id)
    return g, box


def walk(g, box, on_date, stop_before=None):
    """Calls on_date(d, idx, state, priors) for each ET date before that date's games update the state."""
    bx = {}
    for r in box.itertuples(index=False):
        bx.setdefault((r.event_id, r.team), []).append(r)
    st = State()
    PR = {}
    dn = np.array([d.toordinal() for d in g.date])
    for d in np.unique(dn):
        if stop_before is not None and d >= stop_before:
            break
        idx = np.where(dn == d)[0]
        s = int(g.season.iloc[idx[0]])
        if s not in PR:
            PR[s] = priors(box, g, s)
        on_date(d, idx, st, PR[s], bx)
        for i in idx:
            r = g.iloc[i]
            for side in ("home", "away"):
                st.update(d, r[side], bx.get((r.event_id, r[side]), []))
    return st, PR


def features(seasons, keep_players=()) -> tuple[pd.DataFrame, dict]:
    """Per scored game: ridge means, P/O for full / strict (previous-game set) / act (who actually played = injury-
    report proxy, LOOKAHEAD on late scratches), absence info for >= 28 mpg players; player projections for seasons in
    keep_players (act set) for the distribution checks."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"features_{min(seasons)}_{max(seasons)}_{'-'.join(map(str, keep_players))}.pkl"
    src = [Path(__file__), Path(nsm.__file__)] + [nsm.ESPN / f"{k}_{y}.parquet" for y in seasons for k in ("games", "box")]
    sig = str([(str(f), f.stat().st_mtime) for f in src])
    if cp.exists():
        c = pickle.loads(cp.read_bytes())
        if c["sig"] == sig:
            return c["F"], c["PL"]
    t0 = time.time()
    g, box = load_hist(seasons)
    g["mu0"] = ridge_preds(g, nsm.H_TEAM, nsm.LAM)
    g["tot0"] = ridge_totals(g)
    rows, PL = [], {}

    def on_date(d, idx, st, pr, bx):
        for i in idx:
            r = g.iloc[i]
            row = {"event_id": r.event_id}
            for side in ("home", "away"):
                T = r[side]
                full, prev = st.full_set(T), st.prev_set(T)
                act = {x.athlete_id for x in bx.get((r.event_id, T), [])}
                for tag, S in (("strict", prev), ("act", act)):
                    si = side_inputs(st, d, pr, full, S, team=T, known_play=True)
                    row[f"Pfull_{side}"], row[f"Ofull_{side}"] = si["Pfull"], si["Ofull"]
                    row[f"P{tag}_{side}"], row[f"O{tag}_{side}"] = si["Pav"], si["Oav"]
                    if tag == "act":
                        A = {x.athlete_id: x for x in bx.get((r.event_id, T), [])}
                        row[f"Rexp_{side}"] = sum(si["m"][a] * si["R"][a]["reb"] for a in si["m"])
                        row[f"Aexp_{side}"] = sum(si["m"][a] * si["R"][a]["ast"] for a in si["m"])
                        row[f"Ract_{side}"] = sum(x.rebounds for x in A.values())
                        row[f"Aact_{side}"] = sum(x.assists for x in A.values())
                        big = [a for a in full - act if si["R"][a]["mpg"] >= 28]
                        row[f"out28_{side}"] = [(st.names.get(a, a), round(si["R"][a]["mpg"], 1), a in prev) for a in big]
                        if int(r.season) in keep_players and r.scored:
                            PL[(r.event_id, side)] = {
                                "m": si["m"], "R": {a: si["R"][a] for a in si["m"]},
                                "act": {a: (x.minutes, x.points, x.rebounds, x.assists) for a, x in A.items()}}
            rows.append(row)

    walk(g, box, on_date)
    F = g.merge(pd.DataFrame(rows), on="event_id", how="left")
    for v in ("strict", "act"):
        F[f"dAbs_{v}"] = (F[f"P{v}_home"] - F.Pfull_home) - (F[f"P{v}_away"] - F.Pfull_away)
        F[f"oAbs_{v}"] = (F[f"O{v}_home"] - F.Ofull_home) + (F[f"O{v}_away"] - F.Ofull_away)
    F["dPfull"] = F.Pfull_home - F.Pfull_away
    F["Ofull"] = F.Ofull_home + F.Ofull_away
    cp.write_bytes(pickle.dumps({"sig": sig, "F": F, "PL": PL}))
    print(f"(features {min(seasons)}-{max(seasons)} built in {time.time() - t0:.0f}s, cached {cp.name})", file=sys.stderr)
    return F, PL


def ols(X: np.ndarray, y: np.ndarray):
    X1 = np.c_[np.ones(len(X)), X]
    b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    return b, y - X1 @ b


def fit_team(F: pd.DataFrame, train_seasons, var="act") -> dict:
    tr = F[F.scored & F.season.isin(train_seasons)].dropna(subset=["mu0", "tot0", "dPfull", f"dAbs_{var}", "Ofull",
                                                                    f"oAbs_{var}"])
    bm, rm = ols(tr[["mu0", "dPfull", f"dAbs_{var}"]].to_numpy(float), tr.margin.to_numpy(float))
    bt, rt = ols(tr[["tot0", "Ofull", f"oAbs_{var}"]].to_numpy(float), tr.total.to_numpy(float))
    # team rebound / assist noise around the minutes x rate expectation (assists also scale with team points,
    # as in sim_players, so their CV is measured after dividing out actual/expected points)
    rel_r = np.log(tr[["Ract_home", "Ract_away"]].to_numpy(float) / tr[["Rexp_home", "Rexp_away"]].to_numpy(float)).ravel()
    mu_hat = tr.margin.to_numpy(float) - rm
    tot_hat = tr.total.to_numpy(float) - rt
    pe = np.c_[(tot_hat + mu_hat) / 2, (tot_hat - mu_hat) / 2]
    pa = tr[["home_score", "away_score"]].to_numpy(float)
    rel_a = np.log(tr[["Aact_home", "Aact_away"]].to_numpy(float) /
                   (tr[["Aexp_home", "Aexp_away"]].to_numpy(float) * pa / pe)).ravel()
    return {"var": var, "train_seasons": [int(s) for s in train_seasons], "n": int(len(tr)),
            "margin": dict(zip(["const", "mu0", "dPfull", "dAbs"], map(float, bm))),
            "total": dict(zip(["const", "tot0", "Ofull", "oAbs"], map(float, bt))),
            "sigma_m": float(rm.std(ddof=4)), "sigma_t": float(rt.std(ddof=4)), "rho": float(np.corrcoef(rm, rt)[0, 1]),
            "cv": {"reb": float(np.nanstd(rel_r[np.isfinite(rel_r)])), "ast": float(np.nanstd(rel_a[np.isfinite(rel_a)]))}}


def minutes_sd(PL: dict, season_keys) -> float:
    v = []
    for k in season_keys:
        x = PL[k]
        for a, mp in x["m"].items():
            if mp >= 15 and a in x["act"] and x["act"][a][0] > 0:
                v.append(np.log(x["act"][a][0] / mp))
    v = np.array(v)
    return float(np.std(v[np.isfinite(v)]))


def player_check(PL: dict, keys, p: dict, n=2000, seed=11, min_proj=15.0) -> dict:
    """Coverage of the simulated 10-90 interval and mean bias, for players who played with projection >= min_proj.
    Uses the team-points distribution N(expected, sigma) as a stand-in for the joint game sim."""
    rng = np.random.default_rng(seed)
    hit = {s: [] for s in STAT3}
    err = {s: [] for s in STAT3}
    for k in keys:
        x = PL[k]
        si = {"m": x["m"], "R": x["R"]}
        exp_pts = sum(x["m"][a] * x["R"][a]["pts"] for a in x["m"])
        tp = exp_pts + p["sigma_t"] / np.sqrt(2) * rng.standard_normal(n)   # one team's points SD ~ sigma_t/sqrt2
        sp = sim_players(si, tp, exp_pts, p, rng, n)
        if sp is None:
            continue
        for j, a in enumerate(sp["ids"]):
            if x["m"][a] < min_proj or a not in x["act"]:
                continue
            for si_, s in enumerate(STAT3):
                y = x["act"][a][1 + si_]
                q10, q90 = np.quantile(sp[s][:, j], [0.1, 0.9])
                hit[s].append(q10 <= y <= q90)
                err[s].append(sp[s][:, j].mean() - y)
    return {s: {"n": len(hit[s]), "cover_10_90": float(np.mean(hit[s])), "mean_bias": float(np.mean(err[s])),
                "mae": float(np.mean(np.abs(err[s])))} for s in STAT3}


def calibrate_kappa(PL, keys, p, grid=(8, 15, 30, 60, 120, 240)) -> dict:
    best = {}
    for s in STAT3:
        sc = []
        for kp in grid:
            q = dict(p, kappa={t: kp for t in STAT3})
            sc.append((abs(player_check(PL, keys, q, n=1000)[s]["cover_10_90"] - 0.80), kp))
        best[s] = float(min(sc)[1])
    return best


def fit_all(train_seasons=(2023, 2024, 2025), feat_seasons=(2021, 2022, 2023, 2024, 2025), save=True) -> dict:
    F, PL = features(list(feat_seasons), keep_players=(max(train_seasons),))
    p = fit_team(F, train_seasons)
    keys = [k for k in PL if F.loc[F.event_id == k[0], "season"].iloc[0] == max(train_seasons)]
    rng = np.random.default_rng(0)
    keys = [keys[i] for i in rng.choice(len(keys), min(500, len(keys)), replace=False)]
    p["min_sd"] = minutes_sd(PL, keys)
    p["kappa"] = calibrate_kappa(PL, keys, dict(p, kappa={}))
    p["kappa_calibrated_on"] = int(max(train_seasons))
    p["fitted"] = pd.Timestamp.now(tz="UTC").isoformat()
    if save:
        PARAMS.write_text(json.dumps(p, indent=1))
    return p


def load_params() -> dict:
    if not PARAMS.exists():
        print("(no results/nba_sim_params.json -> fitting on dev 2022-23..2024-25, one-time, a few minutes)", file=sys.stderr)
        return fit_all()
    return json.loads(PARAMS.read_text())


# ------------------------------------------------------------------------------------------------------- live
def _names_to_ids(names_str: str, cands: dict) -> tuple[set, list]:
    """'A,B' -> athlete ids. cands: id -> name. Case-insensitive exact, else unique substring."""
    ids, miss = set(), []
    for nm in [x.strip() for x in (names_str or "").split(",") if x.strip()]:
        low = nm.lower()
        hit = [a for a, n in cands.items() if (n or "").lower() == low]
        if not hit:
            hit = [a for a, n in cands.items() if low in (n or "").lower()]
        if len(hit) == 1:
            ids.add(hit[0])
        else:
            miss.append(f"{nm} ({'ambiguous' if hit else 'not found'})")
    return ids, miss


def live_setup(away: str, home: str, d: Date, refresh: bool):
    import nba_simple_live as nsl
    cur = nsl.season_of(d)
    if refresh:
        try:
            nsl.refresh_season(cur, d, True)
        except Exception as ex:
            print(f"(ESPN refresh failed: {str(ex)[:60]}; using disk)", file=sys.stderr)
    seasons = [s for s in range(cur - 3, cur + 1) if (nsm.ESPN / f"games_{s}.parquet").exists()]
    g, box = load_hist(seasons, d)
    gm = g[(g.date == d) & (g.home == home) & (g.away == away)]
    warn = []
    if gm.empty:                                           # not on the schedule: synthesize a neutral-free home game
        warn.append(f"{away}@{home} not on ESPN's {d} schedule -> synthetic home game (rest/travel from last games)")
        row = {"event_id": "SYNTH", "season": cur, "season_type": "regular-season", "date": d, "home": home, "away": away,
               "neutral": False, "home_ind": 1.0, "home_score": np.nan, "away_score": np.nan, "margin": np.nan,
               "total": np.nan, "status": "SYNTH", "start_utc": pd.Timestamp(d, tz="US/Eastern").tz_convert("UTC")}
        for side, T in (("home", home), ("away", away)):
            past = g[((g.home == T) | (g.away == T)) & (g.date < d) & (g.season == cur)]
            if len(past):
                lr = past.iloc[-1]
                row[f"rest_{side}"] = min((d - lr.date).days, 4)
                row[f"trav_{side}"] = km(lr["loc"], LOC[home]) / 1000
            else:
                row[f"rest_{side}"], row[f"trav_{side}"] = 4, 0.0
        row["b2b_home"], row["b2b_away"] = float(row["rest_home"] == 1), float(row["rest_away"] == 1)
        row["restdiff"] = min(row["rest_home"], 3) - min(row["rest_away"], 3)
        g = pd.concat([g[g.date < d], pd.DataFrame([row])], ignore_index=True)
        gm = g[g.event_id == "SYNTH"]
    i = gm.index[0]
    mu0 = float(ridge_preds(g, nsm.H_TEAM, nsm.LAM, only_from=d.toordinal())[i])
    tot0 = float(ridge_totals(g, only_from=d.toordinal())[i])
    past = g[g.date < d]
    st, PR = walk(past, box[box.event_id.isin(set(past.event_id))], lambda *a: None)
    pr = priors(box, g, cur)
    rs = past[(past.season == cur) & (past.season_type == "regular-season")]
    n_cur = pd.concat([rs.home, rs.away]).value_counts().to_dict()
    tid = {T: str(gm.iloc[0][f"{s}_team_id"]) for s, T in (("home", home), ("away", away))}
    roster = nsl.rosters(list(tid.values()), True)
    inj, ok = nsl.injuries(True)
    if not ok:
        warn.append("ESPN INJURY FEED DOWN -> nobody removed except --out")
    return {"g": g, "row": gm.iloc[0], "mu0": mu0, "tot0": tot0, "st": st, "pr": pr, "n_cur": n_cur, "tid": tid,
            "roster": roster, "inj": inj, "warn": warn, "d": d}


def live_sides(L: dict, out_str="", in_str="", minutes_str="") -> dict:
    import nba_simple_live as nsl
    st, d = L["st"], L["d"].toordinal()
    sst = {"ST": st.ST, "TH": st.TH, "names": st.names, "n_cur": L["n_cur"]}
    res = {"warn": list(L["warn"])}
    allc = {}
    for side in ("home", "away"):
        T = L["row"][side]
        w = []
        full_pre, prev, full, sel, sel_q, avail, info = nsl._side_sets(T, L["tid"][T], d, sst, L["roster"], L["inj"],
                                                                        set(), w)
        res["warn"] += [x for x in w if "missed the previous game" not in x]
        cands = {a: st.names.get(a, a) for a in (full | set(L["roster"].get(L["tid"][T], set())))}
        res[side] = {"T": T, "full": full, "avail": avail, "out": info["out"], "q": info["questionable"], "cands": cands}
        allc.update({a: (side, n) for a, n in cands.items()})
    # manual --out / --in / --minutes
    flat = {a: n for a, (s, n) in allc.items()}
    out_ids, m1 = _names_to_ids(out_str, flat)
    in_ids, m2 = _names_to_ids(in_str, flat)
    fixed = {}
    for tok in [x for x in (minutes_str or "").split(",") if x.strip()]:
        nm, _, mins = tok.rpartition(":")
        ids, m3 = _names_to_ids(nm, flat)
        m2 += m3
        for a in ids:
            fixed[a] = float(mins)
    for m in m1 + m2:
        res["warn"].append(f"name not matched: {m}")
    for side in ("home", "away"):
        x = res[side]
        mine = lambda S: {a for a in S if allc.get(a, (None,))[0] == side}
        x["avail"] = (x["avail"] - mine(out_ids)) | mine(in_ids)
        x["manual_out"] = sorted(flat[a] for a in mine(out_ids))
        x["manual_in"] = sorted(flat[a] for a in mine(in_ids))
        x["fixed"] = {a: v for a, v in fixed.items() if allc.get(a, (None,))[0] == side}
        x["full"] = x["full"] | mine(in_ids)
        x["si"] = side_inputs(st, d, L["pr"], x["full"], x["avail"], x["fixed"], team=x["T"])
    return res


def kc(p: float) -> str:
    return f"{100 * p:5.1f}c"


def print_game(L, S, p, res, a):
    h, aw = S["home"], S["away"]
    print(f"\n{aw['T']} @ {h['T']}  {L['d']}   (n={a.n:,} sims, seed {a.seed}; params fit on {p['train_seasons']})")
    print(f"  team ridge (home/rest/b2b/travel incl.) home margin {L['mu0']:+.1f}, total {L['tot0']:.1f}")
    for x in (aw, h):
        print(f"  {x['T']}: strength full {x['si']['Pfull']:+.2f} tonight {x['si']['Pav']:+.2f} (pts/48 impact-weighted);"
              f" OUT(feed) {', '.join(x['out']) or '-'}" + (f"; Q(plays) {', '.join(x['q'])}" if x['q'] else "")
              + (f"; --out {', '.join(x['manual_out'])}" if x['manual_out'] else "")
              + (f"; --in {', '.join(x['manual_in'])}" if x['manual_in'] else ""))
    for w in S["warn"]:
        print(f"  ! {w}")
    m, t = res["margin"], res["total"]
    ph = res["p_home"]
    print(f"\n  FAIR  P({h['T']} win) {ph:.3f} = {kc(ph)}   P({aw['T']} win) {1 - ph:.3f} = {kc(1 - ph)}")
    sp = float(np.median(m))
    print(f"  FAIR spread: {h['T']} {-sp:+.1f}  (median home margin {sp:+.1f}; mean {m.mean():+.2f}; model mean {res['mu']:+.2f})")
    print(f"  FAIR total: {np.median(t):.1f}  (mean {t.mean():.1f}; p10/p90 {np.quantile(t, .1):.0f}/{np.quantile(t, .9):.0f})")
    if a.spread is not None:
        v = m + a.spread
        print(f"  at {h['T']} {a.spread:+.1f}: P(home covers) {(v > 0).mean():.3f}  push {(v == 0).mean():.3f}  "
              f"P(away covers) {(v < 0).mean():.3f}")
    if a.total is not None:
        print(f"  at total {a.total}: P(over) {(t > a.total).mean():.3f}  push {(t == a.total).mean():.3f}  "
              f"P(under) {(t < a.total).mean():.3f}")
    props = {}
    if a.props:
        pf = pd.read_csv(a.props)
        smap = {"points": "pts", "pts": "pts", "rebounds": "reb", "reb": "reb", "assists": "ast", "ast": "ast"}
        for r in pf.itertuples():
            props.setdefault(str(r.player).lower(), []).append((smap.get(str(r.stat).lower(), str(r.stat)), float(r.line)))
    for side, x in (("players_a", aw), ("players_h", h)):
        sp_ = res[side]
        if sp_ is None:
            continue
        print(f"\n  {x['T']:<5}{'player':<24}{'role':>4}{'min':>6}  {'pts mean p10/p50/p90':>22}  {'reb':>16}  {'ast':>16}")
        order = np.argsort(-sp_["min"].mean(0))
        for j in order:
            aid = sp_["ids"][j]
            nm = L["st"].names.get(aid) or x["cands"].get(aid, aid)
            cells = []
            for s in STAT3:
                v = sp_[s][:, j]
                q = np.quantile(v, [.1, .5, .9])
                cells.append(f"{v.mean():5.1f} {q[0]:3.0f}/{q[1]:3.0f}/{q[2]:3.0f}")
            print(f"  {'':<5}{nm[:23]:<24}{x['si']['R'][aid]['role']:>4}{sp_['min'][:, j].mean():6.1f}  {cells[0]:>22}  "
                  f"{cells[1]:>16}  {cells[2]:>16}")
            for s, line in props.get((nm or "").lower(), []):
                v = sp_[s][:, j]
                print(f"  {'':<29}prop {s} {line}: P(over) {(v > line).mean():.3f}  P(under) {(v < line).mean():.3f}")


def cmd_game(a) -> int:
    away, home = (ALIAS.get(x.upper(), x.upper()) for x in (a.away, a.home))
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
    p = load_params()
    L = live_setup(away, home, d, not a.no_refresh)
    S = live_sides(L, a.out, a.inn, a.minutes)
    res = price(p, L["mu0"], L["tot0"], S["home"]["si"], S["away"]["si"], a.n, a.seed)
    print_game(L, S, p, res, a)
    return 0


def cmd_delta(a) -> int:
    away, home = (ALIAS.get(x.upper(), x.upper()) for x in (a.away, a.home))
    d = pd.Timestamp(a.date).date() if a.date else pd.Timestamp.now(tz="US/Eastern").date()
    if not a.out:
        print("delta needs --out")
        return 2
    p = load_params()
    L = live_setup(away, home, d, not a.no_refresh)
    S0 = live_sides(L, "", a.inn, a.minutes)
    S1 = live_sides(L, a.out, a.inn, a.minutes)
    r0 = price(p, L["mu0"], L["tot0"], S0["home"]["si"], S0["away"]["si"], a.n, a.seed, players=False)
    r1 = price(p, L["mu0"], L["tot0"], S1["home"]["si"], S1["away"]["si"], a.n, a.seed, players=False)
    gone = S1["home"]["manual_out"] + S1["away"]["manual_out"]
    if not gone:
        print(f"no --out name matched; warnings: {S1['warn']}")
        return 2
    for side in ("home", "away"):
        if S1[side]["manual_out"]:
            x0, x1 = S0[side], S1[side]
            for nm in x1["manual_out"]:
                aid = [k for k, v in x1["cands"].items() if v == nm][0]
                was = x0["si"]["m"].get(aid)
                print(f"{nm} ({x1['T']}): projected {was:.1f} min tonight in the base case" if was else
                      f"{nm} ({x1['T']}): already NOT in the base set (injury feed OUT / not in rotation) -> delta ~0")
            print(f"  {x1['T']} strength tonight {x0['si']['Pav']:+.2f} -> {x1['si']['Pav']:+.2f} pts/48; expected pts "
                  f"{x0['si']['Oav']:.1f} -> {x1['si']['Oav']:.1f}")
            top = sorted(x1["si"]["m"].items(), key=lambda kv: -(kv[1] - x0["si"]["m"].get(kv[0], 0)))[:4]
            print("  minutes go to: " + ", ".join(f"{L['st'].names.get(k, k)} +{v - x0['si']['m'].get(k, 0):.1f}" for k, v in top))
    for w in S1["warn"]:
        print(f"  ! {w}")
    H = L["row"]["home"]
    ph0, ph1 = r0["p_home"], r1["p_home"]
    m0, m1 = float(np.median(r0["margin"])), float(np.median(r1["margin"]))
    print(f"\n{away} @ {H} {d}  (common random numbers, n={a.n:,})")
    print(f"  P({H} win): {kc(ph0)} -> {kc(ph1)}   change {100 * (ph1 - ph0):+.1f}c   "
          f"(P({away}) {kc(1 - ph0)} -> {kc(1 - ph1)})")
    print(f"  fair spread ({H}): {-m0:+.1f} -> {-m1:+.1f}   model mean margin {r0['mu']:+.2f} -> {r1['mu']:+.2f} "
          f"(change {r1['mu'] - r0['mu']:+.2f})")
    print(f"  fair total: {np.median(r0['total']):.1f} -> {np.median(r1['total']):.1f}   model mean {r0['tot']:.1f} -> "
          f"{r1['tot']:.1f} (change {r1['tot'] - r0['tot']:+.2f})")
    return 0


# ----------------------------------------------------------------------------------------------------- validate
def boot_ci(v, n=4000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def cmd_validate(a) -> int:
    from sports.market import devig_two_way
    S = int(a.season.split("-")[0]) + 1 if "-" in a.season else int(a.season)
    if S >= 2026:
        print("2025-26 is SEALED (and 2026-27 in progress): validate refuses; pass a dev season <= 2024-25")
        return 2
    train = [s for s in range(2023, S)]
    if not train:
        print("need at least one training season >= 2022-23 before the validated season")
        return 2
    feat = list(range(2021, S + 1))
    F, PL = features(feat, keep_players=(S - 1, S))
    p = fit_team(F, train)
    trk = [k for k in PL if F.loc[F.event_id == k[0], "season"].iloc[0] == S - 1]
    rng = np.random.default_rng(0)
    trk = [trk[i] for i in rng.choice(len(trk), min(500, len(trk)), replace=False)]
    p["min_sd"] = minutes_sd(PL, trk)
    p["kappa"] = calibrate_kappa(PL, trk, dict(p, kappa={}))
    ps = fit_team(F, train, var="strict")
    print(f"VALIDATE {S - 1}-{str(S)[2:]} (ESPN season {S}); team fit on {train} (n={p['n']} scored games), "
          f"kappa/min-noise calibrated on {S - 1}")
    print(f"  margin = {p['margin']['const']:+.2f} + {p['margin']['mu0']:.3f}*mu_ridge + {p['margin']['dPfull']:.3f}*dP_full"
          f" + {p['margin']['dAbs']:.3f}*dP_absence   sigma {p['sigma_m']:.2f}")
    print(f"  total  = {p['total']['const']:+.2f} + {p['total']['tot0']:.3f}*tot_ridge + {p['total']['Ofull']:.3f}*O_full"
          f" + {p['total']['oAbs']:.3f}*O_absence   sigma {p['sigma_t']:.2f}   rho {p['rho']:+.3f}")
    print(f"  player noise: minutes log-SD {p['min_sd']:.3f}; team CV reb {p['cv']['reb']:.3f} ast {p['cv']['ast']:.3f}; "
          f"kappa {p['kappa']}")

    te = F[F.scored & (F.season == S)].copy()
    L = pd.read_parquet(nsm.ESPN / f"lines_{S}.parquet")
    L = L[~L.provider.str.lower().isin(nsm.NOT_BOOKS)]
    prov = L.provider.value_counts().index[0]
    L = L[L.provider == prov].drop_duplicates("event_id")
    te = te.merge(L[["event_id", "home_spread_close", "home_spread_open", "total_close", "total_open", "home_ml_close",
                     "away_ml_close"]], on="event_id", how="left")
    rows = []
    for v, pp in (("act", p), ("strict", ps)):
        for r in te.itertuples():
            h = {"Pfull": r.Pfull_home, "Pav": getattr(r, f"P{v}_home"), "Ofull": r.Ofull_home, "Oav": getattr(r, f"O{v}_home")}
            aw = {"Pfull": r.Pfull_away, "Pav": getattr(r, f"P{v}_away"), "Ofull": r.Ofull_away, "Oav": getattr(r, f"O{v}_away")}
            if any(pd.isna(x) for x in list(h.values()) + list(aw.values()) + [r.mu0, r.tot0]):
                continue
            res = price(pp, r.mu0, r.tot0, h, aw, n=4000, seed=int(r.event_id) % 2 ** 31, players=False)
            rows.append({"event_id": r.event_id, "var": v, "sim_margin": float(np.median(res["margin"])),
                         "sim_mean": res["mu"], "sim_total": float(np.median(res["total"])), "sim_tmean": res["tot"],
                         "p_home": res["p_home"]})
    R = pd.DataFrame(rows).merge(te, on="event_id")
    R["close_margin"] = -R.home_spread_close
    R["open_margin"] = -R.home_spread_open
    ph_c, _, ov = devig_two_way(R.home_ml_close.to_numpy(float), R.away_ml_close.to_numpy(float))
    R["p_close"] = np.where((ov >= 0) & (ov <= 0.10), ph_c, np.nan)
    R["y"] = (R.margin > 0).astype(float)
    OUTD.mkdir(parents=True, exist_ok=True)
    R.drop(columns=[c for c in R.columns if c.startswith("out28")]).to_csv(OUTD / f"validate_{S}_games.csv", index=False)
    summ = {"season": S, "train": train, "params_act": p, "params_strict": ps, "provider": prov}
    print(f"\n  scored games {len(te)}; lines from {prov} (open/close as recorded by ESPN)")
    for v in ("act", "strict"):
        x = R[(R["var"] == v)].dropna(subset=["close_margin", "total_close"])
        d_sc = (x.sim_margin - x.close_margin).abs()
        d_tc = (x.sim_total - x.total_close).abs()
        out = {"n": len(x),
               "spread_mae_vs_close": boot_ci(d_sc), "spread_mae_vs_actual": boot_ci((x.sim_margin - x.margin).abs()),
               "close_spread_mae_vs_actual": boot_ci((x.close_margin - x.margin).abs()),
               "total_mae_vs_close": boot_ci(d_tc), "total_mae_vs_actual": boot_ci((x.sim_total - x.total).abs()),
               "close_total_mae_vs_actual": boot_ci((x.total_close - x.total).abs()),
               "spread_bias_vs_close": boot_ci(x.sim_margin - x.close_margin),
               "total_bias_vs_close": boot_ci(x.sim_total - x.total_close),
               "share_spread_within_1.5": float((d_sc <= 1.5).mean()), "share_total_within_3": float((d_tc <= 3).mean())}
        y = x.dropna(subset=["p_close"])
        bs, bc = (y.p_home - y.y) ** 2, (y.p_close - y.y) ** 2
        out["brier_sim"], out["brier_close"] = boot_ci(bs), boot_ci(bc)
        out["brier_sim_minus_close"] = boot_ci(bs.to_numpy() - bc.to_numpy())
        out["n_brier"] = len(y)
        out["corr_sim_close_margin"] = float(np.corrcoef(x.sim_margin, x.close_margin)[0, 1])
        b, _ = ols(np.c_[x.close_margin, x.sim_margin], x.margin.to_numpy(float))
        out["stack_actual_margin_on_close_and_sim"] = {"close": float(b[1]), "sim": float(b[2])}
        cal = y.assign(bin=pd.cut(y.p_home, [0, .2, .35, .5, .65, .8, 1])).groupby("bin", observed=True).agg(
            n=("y", "size"), p_sim=("p_home", "mean"), won=("y", "mean"), p_close=("p_close", "mean"))
        out["calibration"] = cal.reset_index().astype({"bin": str}).to_dict("records")
        summ[v] = out
        f = lambda k: f"{out[k][0]:.2f} [{out[k][1]:.2f}, {out[k][2]:.2f}]"
        lab = "ACT (who actually played = injury-report-at-tip proxy; lookahead on late scratches)" if v == "act" else \
              "STRICT (previous game's players; no injury info)"
        print(f"\n  == {lab}, n={out['n']}")
        print(f"   spread MAE vs close  {f('spread_mae_vs_close')}   bias sim-close {f('spread_bias_vs_close')}   "
              f"within 1.5 pts: {out['share_spread_within_1.5']:.1%}   corr {out['corr_sim_close_margin']:.3f}")
        print(f"   spread MAE vs actual {f('spread_mae_vs_actual')}   (close vs actual {f('close_spread_mae_vs_actual')})")
        print(f"   total  MAE vs close  {f('total_mae_vs_close')}   bias {f('total_bias_vs_close')}   "
              f"within 3 pts: {out['share_total_within_3']:.1%}")
        print(f"   total  MAE vs actual {f('total_mae_vs_actual')}   (close vs actual {f('close_total_mae_vs_actual')})")
        print(f"   Brier sim {out['brier_sim'][0]:.4f}  de-vig close {out['brier_close'][0]:.4f}  diff sim-close "
              f"{out['brier_sim_minus_close'][0]:+.4f} [{out['brier_sim_minus_close'][1]:+.4f}, "
              f"{out['brier_sim_minus_close'][2]:+.4f}]  (n={out['n_brier']})")
        print(f"   actual margin ~ close + sim: coef close {out['stack_actual_margin_on_close_and_sim']['close']:.3f}, "
              f"sim {out['stack_actual_margin_on_close_and_sim']['sim']:.3f}")
        print("   calibration: " + "; ".join(f"{c['bin']} n{c['n']} sim {c['p_sim']:.2f} won {c['won']:.2f} close "
                                          f"{c['p_close']:.2f}" for c in out["calibration"]))

    # injury delta check: >= 28 mpg rotation player absent; news-type = he played the team's previous game
    x = R[R["var"] == "act"].merge(R[R["var"] == "strict"][["event_id", "sim_mean"]].rename(columns={"sim_mean": "strict_mean"}),
                                   on="event_id")
    full_mu = lambda r: (p["margin"]["const"] + p["margin"]["mu0"] * r.mu0 + p["margin"]["dPfull"] * r.dPfull)
    x["full_mean"] = [full_mu(r) for r in x.itertuples()]
    x["move"] = x.close_margin - x.open_margin
    inj = []
    for r in x.itertuples():
        o_h, o_a = r.out28_home or [], r.out28_away or []
        if not (o_h or o_a) or pd.isna(r.move):
            continue
        news = any(t[2] for t in o_h + o_a)
        inj.append({"event_id": r.event_id, "news": news, "move": r.move, "d_full": r.sim_mean - r.full_mean,
                    "d_strict": r.sim_mean - r.strict_mean, "who": "; ".join(f"{t[0]}({t[1]})" for t in o_h + o_a)})
    I = pd.DataFrame(inj)
    print(f"\n  == injury delta check (>= 28 mpg rotation player did not play; delta in home margin; ESPN open->close move)")
    summ["injury"] = {}
    for tag, y, col in (("news (played previous game -> news after it): sim delta vs STRICT", I[I.news], "d_strict"),
                        ("all >=28-mpg absences: sim delta vs FULL rotation", I, "d_full")):
        if len(y) < 5:
            print(f"   {tag}: n={len(y)} (too few)")
            continue
        c = float(np.corrcoef(y[col], y.move)[0, 1])
        b, _ = ols(y[[col]].to_numpy(float), y.move.to_numpy(float))
        sg = float((np.sign(y[col]) == np.sign(y.move))[y.move != 0].mean())
        rb = np.random.default_rng(1)
        bsl = []
        for _ in range(2000):
            k = rb.integers(0, len(y), len(y))
            bb, _ = ols(y[[col]].to_numpy(float)[k], y.move.to_numpy(float)[k])
            bsl.append(bb[1])
        o = {"n": len(y), "mean_abs_sim_delta": float(y[col].abs().mean()), "mean_abs_move": float(y.move.abs().mean()),
             "corr": c, "slope_move_on_delta": float(b[1]), "slope_ci": [float(np.quantile(bsl, .025)), float(np.quantile(bsl, .975))],
             "sign_agree_nonzero_moves": sg, "mae_delta_vs_move": float((y[col] - y.move).abs().mean())}
        summ["injury"][col] = o
        print(f"   {tag}: n={o['n']}  mean|delta| {o['mean_abs_sim_delta']:.2f}  mean|move| {o['mean_abs_move']:.2f}  "
              f"corr {c:.3f}  slope move~delta {o['slope_move_on_delta']:.2f} [{o['slope_ci'][0]:.2f}, {o['slope_ci'][1]:.2f}]  "
              f"sign agree {sg:.1%}  MAE(delta-move) {o['mae_delta_vs_move']:.2f}")
    if len(I):
        I.to_csv(OUTD / f"validate_{S}_injury.csv", index=False)

    tek = [k for k in PL if F.loc[F.event_id == k[0], "season"].iloc[0] == S]
    rng = np.random.default_rng(5)
    tek = [tek[i] for i in rng.choice(len(tek), min(800, len(tek)), replace=False)]
    pc = player_check(PL, tek, p)
    summ["players"] = pc
    print(f"\n  == player distributions ({S - 1}-{str(S)[2:]}, {len(tek)} random team-games, ACT set, players with "
          f"projected >= 15 min who played; team pts ~ N(exp, sigma_t/sqrt2) stand-in)")
    for s in STAT3:
        print(f"   {s}: n={pc[s]['n']}  10-90 coverage {pc[s]['cover_10_90']:.1%} (target 80%)  mean bias "
              f"{pc[s]['mean_bias']:+.2f}  MAE {pc[s]['mae']:.2f}")
    (OUTD / f"validate_{S}.json").write_text(json.dumps(summ, indent=1, default=str))
    v = summ["act"]["spread_mae_vs_close"][0]
    print(f"\n  VERDICT (pre-stated bar: mean |sim spread - close| <= ~1.5 pts = usable as a fast repricer): "
          f"ACT {v:.2f} -> {'WITHIN' if v <= 1.5 else 'NOT within'} 1.5")
    print(f"  wrote {OUTD / f'validate_{S}.json'}, validate_{S}_games.csv, validate_{S}_injury.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("game", "delta"):
        x = sub.add_parser(c)
        x.add_argument("away")
        x.add_argument("home")
        x.add_argument("--date")
        x.add_argument("--out", default="")
        x.add_argument("--in", dest="inn", default="")
        x.add_argument("--minutes", default="")
        x.add_argument("--n", type=int, default=20000)
        x.add_argument("--seed", type=int, default=7)
        x.add_argument("--spread", type=float)
        x.add_argument("--total", type=float)
        x.add_argument("--props")
        x.add_argument("--no-refresh", action="store_true")
    v = sub.add_parser("validate")
    v.add_argument("--season", default="2024-25")
    sub.add_parser("fit")
    a = ap.parse_args(argv)
    if a.cmd == "game":
        return cmd_game(a)
    if a.cmd == "delta":
        return cmd_delta(a)
    if a.cmd == "validate":
        return cmd_validate(a)
    if a.cmd == "fit":
        p = fit_all()
        print(json.dumps(p, indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
