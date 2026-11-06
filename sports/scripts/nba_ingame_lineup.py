"""NBA IN-GAME WP WITH PLAYERS: does who is on the floor / available add to pregame strength + margin + time,
and does it add anything to the Kalshi KXNBAGAME in-game price? DESCRIPTIVE GATE FIRST, no money.

User idea (2026-11-05): in-game EV from pregame odds + minutes left + score margin, PLUS who the teams/players are,
location etc. The simple version (spread + margin + time) is DEAD vs Kalshi (late_game_calibration.md, L37 sealed
fail). The untested part is the player-aware version.

    python scripts/nba_ingame_lineup.py ratings   # point-in-time player ratings per game (nba_sim v2.2 walk)
    python scripts/nba_ingame_lineup.py states    # ESPN pbp -> per-play lineup/foul/availability features
    python scripts/nba_ingame_lineup.py oos       # GATE A: pre-Kalshi holdout log loss, base vs player-aware
    python scripts/nba_ingame_lineup.py resid     # GATE B: outcome ~ logit(Kalshi mid) + (model - Kalshi) on DEV

PRE-REGISTRATION (written 2026-11-05 BEFORE any feature was built or any Kalshi price joined to these features)
-------------------------------------------------------------------------------------------------------------
Data: ESPN summary pbp (cached) for the 6,602 games in data/nba_late_states.parquet (2021-22 .. 2025-26, reg+post).
  Lineups: period-1 starters from the boxscore; later periods inferred (a player's first event in the period that is
  not "enters the game" => he started it; short sets filled from the previous period's end lineup); then
  Substitution plays (participants[0] enters, [1] leaves). Personal fouls = types containing "Foul" without
  "Technical"/"Turnover"/"Free Throw" (participants each get one). Ejection plays and 6th fouls => player gone.
Player values: nba_sim v2.2 (shrunk BPM-lite + on/off) rating r and full-strength expected minutes m_full, from the
  walk() state BEFORE the game date (point-in-time). Pfull = sum(m_full * r)/240 over the last-10-games rotation.
Snapshot rows for fitting: the state at each elapsed game minute 1..47 (+ each OT minute), last play <= the mark.
  tau = (regulation seconds left + 4)/2880 (OT: (period clock + 4)/2880), m = home margin, z0 = Phi^-1(ESPN pregame
  home WP), poss = heuristic as nba_late_wp.py.
BASE (refit, logistic): [1, m/sqrt(tau), z0*sqrt(tau), poss/sqrt(tau), m].
FULL = BASE + (all home-minus-away):
  L   = on-floor mean r - Pfull                 as  L * min(tau, 4/48)/sqrt(tau)  and  L * sqrt(tau)
  G   = gone value: sum over gone players (6 PF, ejected, or m_full>=24 player who has played and has been off the
        floor >= 15 game-minutes) of (r - (-2)) * m_full/48                 as  G * sqrt(tau)
  F   = foul trouble: count of the team's top-2-by-r (m_full>=24) players with PF >= period+1 (Q4/OT: >= 5)
                                                                             as  F * sqrt(tau)
  M   = starter minutes surplus: top-5 by m_full, minutes played - m_full * elapsed/2880, in minutes /48
                                                                             as  M / sqrt(tau)
  team identity: +1 home team, -1 away team dummies (30) x sqrt(tau); home: sqrt(tau) term.
  Team dummies carry an L2 penalty (C = 1.0 on standardized-free dummies); all else unpenalized (C large).
GATE A (descriptive): fit on 2021-22 + 2022-23 (reg+post), score 2023-24 + 2024-25 REGULAR season snapshots.
  Report log loss BASE / BASE+player terms (L,G,F,M) / FULL; paired game-cluster bootstrap 95% CI of (FULL - BASE).
GATE B (descriptive, the one that decides): refit BASE and FULL on all games dated < 2025-04-15. Kalshi KXNBAGAME
  home-side 1-min candles (data/nba_ingame.parquet + data/nba_late_candles_2025po.parquet), DEV window ONLY =
  games dated 2025-04-15 .. 2026-01-14; two-sided quote with ask-bid <= 0.05, 0.03 <= mid <= 0.97; state = last
  play with wallclock <= candle end - 60 s. Logit: y = home won ~ 1 + logit(mid) + (logit(FULL) - logit(mid)).
  PRIMARY = coefficient on (FULL - Kalshi), 95% CI by game-cluster bootstrap (2000). Secondary (reported, not a
  gate): the same with (logit FULL - logit BASE) = the player increment only.
  Money trials (<= 2, edgelab family kalshi_ingame, date split) are registered ONLY IF the PRIMARY CI excludes 0 on
  the positive side. Otherwise: dead, no trials.

VERDICT (2026-11-05): DEAD, no trials registered. results/nba_ingame_lineup/gate_{a_oos,b_resid,b_posthoc_player}.json
  Gate A (test 2,462 games / 116,369 minute rows): log loss base .43639, +player .43604 (diff -.00035
  [-.00080, +.00011], not distinguishable), full (+team FE) .44876 (+.0124 [+.0075, +.0173], worse).
  Gate B (DEV 676 games / 81,491 candles): Kalshi .5524 < base .5595 < full .5632. PRIMARY coef on (FULL - Kalshi)
  +0.116 [-0.179, +0.434] -> crosses 0. Secondary (FULL - BASE) -0.071 [-0.80, +0.62]. Post-hoc (not pre-reg)
  PLAYER-only model: (PLAYER - Kalshi) +0.171 [-0.138, +0.501]; (PLAYER - BASE) -0.089 [-2.05, +2.16].
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, os.path.expanduser("~/agent-harness/lib"))

import numpy as np
import pandas as pd
from scipy.stats import norm

from sports.http import fetch

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
RAT = ROOT / "data" / "nba_ingame_lineup_ratings.pkl"
ST = ROOT / "data" / "nba_ingame_lineup_states.parquet"
OUT = ROOT / "results" / "nba_ingame_lineup"
LATE = ROOT / "data" / "nba_late_states.parquet"
TRAIN_END = pd.Timestamp("2025-04-15")
DEV_START, DEV_END = pd.Timestamp("2025-04-15"), pd.Timestamp("2026-01-14")
R_REPL = -2.0
LONG_OFF = 15 * 60


# ------------------------------------------------------------------------------------------------------ ratings
def cmd_ratings() -> int:
    import nba_sim as ns
    ns.set_impact("v22")
    g, box = ns.load_hist(list(range(2021, 2027)))
    out = {}

    def on_date(d, idx, st, pr, bx):
        for i in idx:
            r = g.iloc[i]
            rec = {}
            for side in ("home", "away"):
                T = r[side]
                full = st.full_set(T)
                act = {x.athlete_id for x in bx.get((r.event_id, T), [])}
                si = ns.side_inputs(st, d, pr, full, full, team=T)
                R = {a: ns.player_rates(st, a, d, pr)["r"] for a in full | act}
                rec[side] = {"team": T, "r": R, "m": dict(si["m_full"]), "Pfull": si["Pfull"]}
            out[str(r.event_id)] = rec

    ns.walk(g, box, on_date)
    RAT.write_bytes(pickle.dumps(out))
    print(f"-> {RAT}: {len(out)} games")
    return 0


# ------------------------------------------------------------------------------------------------------- states
def secs(c):
    try:
        if ":" in c:
            mm, ss = c.split(":")
            return int(mm) * 60 + float(ss)
        return float(c)
    except Exception:
        return np.nan


def is_pf(t: str) -> bool:
    return "Foul" in t and not any(k in t for k in ("Technical", "Turnover", "Free Throw"))


def parse(eid: str, rat: dict) -> list[dict]:
    j = fetch(f"{ESPN}/summary?event={eid}")
    comp = j["header"]["competitions"][0]
    side_of_tid = {c["id"]: c["homeAway"] for c in comp["competitors"]}
    team_of, starters = {}, {"home": set(), "away": set()}
    for t in j["boxscore"].get("players", []):
        sd = side_of_tid.get(t["team"]["id"])
        for a in t["statistics"][0]["athletes"]:
            aid = a["athlete"].get("id")
            if aid is None:
                continue
            team_of[aid] = sd
            if a.get("starter"):
                starters[sd].add(aid)
    plays = j.get("plays") or []
    if not plays or len(starters["home"]) != 5 or len(starters["away"]) != 5:
        return []
    RR = {sd: {str(k): v for k, v in rat[sd]["r"].items()} for sd in ("home", "away")}
    MF = {sd: {str(k): v for k, v in rat[sd]["m"].items()} for sd in ("home", "away")}
    Pf = {sd: rat[sd]["Pfull"] for sd in ("home", "away")}
    rot = {sd: sorted(MF[sd], key=lambda a: -MF[sd][a]) for sd in ("home", "away")}
    top5 = {sd: rot[sd][:5] for sd in ("home", "away")}
    stars = {sd: sorted([a for a in MF[sd] if MF[sd][a] >= 24], key=lambda a: -RR[sd].get(a, R_REPL))[:2] for sd in ("home", "away")}
    rv = lambda sd, a: RR[sd].get(a, R_REPL)

    # period start lineups
    per_plays = {}
    for p in plays:
        per_plays.setdefault(p.get("period", {}).get("number", 0), []).append(p)
    on = {sd: set(starters[sd]) for sd in ("home", "away")}
    pf, played, last_off, gone_hard = {}, {}, {}, set()
    rows, poss = [], 0
    prev_el = 0.0
    for pn in sorted(per_plays):
        pp = per_plays[pn]
        if pn > 1:
            first = {}
            for p in pp:
                ids = [x["athlete"]["id"] for x in p.get("participants", []) if "athlete" in x]
                if p["type"]["text"] == "Substitution" and len(ids) == 2:
                    first.setdefault(ids[0], "in")
                    first.setdefault(ids[1], "start")
                else:
                    for a in ids:
                        first.setdefault(a, "start")
            new = {"home": set(), "away": set()}
            for a, v in first.items():
                sd = team_of.get(a)
                if v == "start" and sd:
                    new[sd].add(a)
            for sd in ("home", "away"):
                if len(new[sd]) < 5:
                    for a in sorted(on[sd], key=lambda a: -MF[sd].get(a, 0)):
                        if len(new[sd]) >= 5:
                            break
                        if a not in first and a not in gone_hard:
                            new[sd].add(a)
                if len(new[sd]) > 5:
                    new[sd] = set(sorted(new[sd], key=lambda a: -MF[sd].get(a, 0))[:5])
                for a in on[sd] - new[sd]:
                    last_off[a] = prev_el
                on[sd] = new[sd]
            poss = 0
        plen = 720 if pn <= 4 else 300
        base = (pn - 1) * 720 if pn <= 4 else 2880 + (pn - 5) * 300
        for p in pp:
            cs = secs(p.get("clock", {}).get("displayValue", ""))
            if not np.isfinite(cs):
                continue
            el = base + plen - cs
            dt = max(el - prev_el, 0.0)
            for sd in ("home", "away"):
                for a in on[sd]:
                    played[a] = played.get(a, 0.0) + dt
            prev_el = max(prev_el, el)
            typ = p.get("type", {}).get("text", "")
            ids = [x["athlete"]["id"] for x in p.get("participants", []) if "athlete" in x]
            if typ == "Substitution" and len(ids) == 2:
                sd = team_of.get(ids[0]) or team_of.get(ids[1])
                if sd:
                    on[sd].discard(ids[1])
                    on[sd].add(ids[0])
                    last_off[ids[1]] = el
            elif is_pf(typ):
                for a in ids[:2] if "Double" in typ else ids[:1]:
                    pf[a] = pf.get(a, 0) + 1
                    if pf[a] >= 6:
                        gone_hard.add(a)
            elif "Ejection" in typ and ids:
                gone_hard.add(ids[0])
            tid = (p.get("team") or {}).get("id")
            s = side_of_tid.get(tid)
            sgn = 1 if s == "home" else -1 if s == "away" else 0
            made = bool(p.get("scoringPlay"))
            if sgn:
                if "Rebound" in typ:
                    poss = sgn
                elif "Turnover" in typ:
                    poss = -sgn
                elif "Free Throw" in typ:
                    parts = typ.replace("Free Throw - ", "").split(" of ")
                    if made and len(parts) == 2 and parts[0] == parts[1]:
                        poss = -sgn
                elif made and p.get("shootingPlay"):
                    poss = -sgn
            f = {}
            for sd in ("home", "away"):
                o = list(on[sd])
                f[f"L_{sd}"] = (np.mean([rv(sd, a) for a in o]) - Pf[sd]) if o else 0.0
                f[f"n_{sd}"] = len(o)
                G = 0.0
                for a in MF[sd]:
                    hard = a in gone_hard
                    soft = (MF[sd][a] >= 24 and played.get(a, 0) > 0 and a not in on[sd]
                            and el - last_off.get(a, el) >= LONG_OFF)
                    if hard or soft:
                        G += (rv(sd, a) - R_REPL) * MF[sd][a] / 48
                f[f"G_{sd}"] = G
                thr = pn + 1 if pn <= 3 else 5
                f[f"F_{sd}"] = sum(1 for a in stars[sd] if pf.get(a, 0) >= thr and a not in gone_hard)
                f[f"M_{sd}"] = sum(played.get(a, 0) / 60 - MF[sd][a] * min(el, 2880) / 2880 for a in top5[sd]) / 48
            rows.append({"eid": eid, "seq": int(p.get("sequenceNumber", 0)), "period": pn, "clock_s": cs, "el": el,
                         "hs": p.get("homeScore"), "as": p.get("awayScore"), "poss": poss, "wall": p.get("wallclock"),
                         "home_t": rat["home"]["team"], "away_t": rat["away"]["team"], **f})
    return rows


def cmd_states() -> int:
    rat = pickle.loads(RAT.read_bytes())
    meta = pd.read_parquet(LATE, columns=["eid", "p0", "home_won", "date", "season", "gtype"]).drop_duplicates("eid")
    rows, miss, bad = [], 0, 0
    for i, eid in enumerate(meta.eid):
        if str(eid) not in rat:
            miss += 1
            continue
        try:
            rr = parse(str(eid), rat[str(eid)])
        except Exception as ex:          # noqa: BLE001
            print("skip", eid, ex)
            bad += 1
            continue
        if not rr:
            bad += 1
        rows += rr
        if i % 500 == 0:
            print(i, len(rows), flush=True)
    s = pd.DataFrame(rows)
    s["eid"] = s.eid.astype(str)
    meta["eid"] = meta.eid.astype(str)
    s = s.merge(meta, on="eid", how="inner", validate="many_to_one")
    s["wall"] = pd.to_datetime(s.wall, utc=True, format="ISO8601", errors="coerce")
    s.to_parquet(ST, index=False)
    print(f"-> {ST}: {s.eid.nunique()} games, {len(s)} plays; no rating {miss}, unparsed {bad}")
    return 0


# ---------------------------------------------------------------------------------------------------- modelling
def prep(s: pd.DataFrame) -> pd.DataFrame:
    s = s.dropna(subset=["clock_s", "hs", "as"]).copy()
    s["m"] = (s.hs - s["as"]).astype(float)
    s["z0"] = norm.ppf(np.clip(s.p0, 0.01, 0.99))
    reg_left = np.where(s.period <= 4, (4 - s.period) * 720 + s.clock_s, s.clock_s)
    s["tau"] = (reg_left + 4) / 2880
    return s


def minute_rows(s: pd.DataFrame) -> pd.DataFrame:
    """Last play at or before each elapsed minute mark (1..47 regulation, every OT minute)."""
    s = s.sort_values(["eid", "el", "seq"])
    s["mk"] = np.floor(s.el / 60).astype(int)
    x = s.groupby(["eid", "mk"]).tail(1)
    return x[(x.mk >= 1) & (x.mk != 48)]


TEAMS: list[str] = []


def design(x: pd.DataFrame, which: str) -> np.ndarray:
    st = np.sqrt(x.tau.to_numpy())
    cols = [x.m / st, x.z0 * st, x.poss / st, x.m]
    if which in ("player", "full"):
        d = lambda k: (x[f"{k}_home"] - x[f"{k}_away"]).to_numpy(float)
        cols += [d("L") * np.minimum(x.tau, 4 / 48) / st, d("L") * st, d("G") * st, d("F") * st, d("M") / st]
    if which == "full":
        cols += [st]
        for t in TEAMS:
            cols.append(((x.home_t == t).astype(float) - (x.away_t == t).astype(float)).to_numpy() * st)
    return np.column_stack([np.asarray(c, float) for c in cols])


def n_unpen(which: str) -> int:
    return {"base": 4, "player": 9, "full": 10}[which]


def fit(x: pd.DataFrame, which: str):
    """Logistic regression; team dummies L2-penalised (lambda=1), the rest unpenalised. Newton / IRLS."""
    X = np.column_stack([np.ones(len(x)), design(x, which)])
    y = x.home_won.to_numpy(float)
    k = X.shape[1]
    pen = np.zeros(k)
    pen[1 + n_unpen(which):] = 1.0
    b = np.zeros(k)
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - pen * b
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def predict(b, x, which):
    X = np.column_stack([np.ones(len(x)), design(x, which)])
    return 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))


def llv(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_diff(eids, d, n=2000, seed=0):
    df = pd.DataFrame({"e": eids, "d": d}).groupby("e").d.agg(["sum", "count"])
    s, c = df["sum"].to_numpy(), df["count"].to_numpy()
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        bs.append(s[i].sum() / c[i].sum())
    return float(s.sum() / c.sum()), np.percentile(bs, [2.5, 97.5]).tolist()


def load_states():
    global TEAMS
    s = prep(pd.read_parquet(ST))
    s["date"] = pd.to_datetime(s.date)
    TEAMS = sorted(set(s.home_t) | set(s.away_t))[1:]   # drop one for identifiability
    return s


def cmd_oos() -> int:
    s = load_states()
    x = minute_rows(s)
    tr = x[x.season.isin([2021, 2022])]
    te = x[x.season.isin([2023, 2024]) & (x.gtype == "regular")]
    y = te.home_won.to_numpy(float)
    res = {"n_train_rows": len(tr), "n_train_games": int(tr.eid.nunique()), "n_test_rows": len(te),
           "n_test_games": int(te.eid.nunique()), "ll": {}, "coef": {}}
    L = {}
    for w in ("base", "player", "full"):
        b = fit(tr, w)
        L[w] = llv(y, predict(b, te, w))
        res["ll"][w] = float(L[w].mean())
        res["coef"][w] = [round(float(v), 4) for v in b[: 1 + n_unpen(w) + (0 if w != "full" else 0)]]
    for w in ("player", "full"):
        res[f"d_{w}_minus_base"] = boot_diff(te.eid.to_numpy(), L[w] - L["base"])
    # by phase
    ph = pd.cut(te.el, [0, 1440, 2160, 2880, 1e9], labels=["H1", "Q3", "Q4", "OT"])
    res["by_phase"] = {str(k): {"n": int(m.sum()), "base": float(L["base"][m].mean()), "player": float(L["player"][m].mean()),
                                "full": float(L["full"][m].mean())} for k in ph.cat.categories for m in [(ph == k).to_numpy()] if m.sum()}
    res["feature_sd_test"] = {k: float((te[f"{k}_home"] - te[f"{k}_away"]).std()) for k in "LGFM"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_a_oos.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


def kalshi_snaps(s: pd.DataFrame, lag: int = 60) -> pd.DataFrame:
    from traps import check_epoch_seconds, epoch_seconds
    old = pd.read_parquet(ROOT / "data" / "nba_ingame.parquet")
    new = pd.read_parquet(ROOT / "data" / "nba_late_candles_2025po.parquet")
    cd = pd.concat([old[["end", "bid", "ask", "eid", "side", "result", "date"]], new[["end", "bid", "ask", "eid", "side", "result", "date"]]],
                   ignore_index=True)
    cd["date"] = pd.to_datetime(cd.date)
    cd = cd[(cd.date >= DEV_START) & (cd.date <= DEV_END) & (cd.side == "home")].copy()
    cd["eid"] = cd.eid.astype(str)
    check_epoch_seconds(cd.end.to_numpy(float), "kalshi end")
    s = s.dropna(subset=["wall"])
    s = s[s.eid.isin(set(cd.eid))].copy()
    s["wt"] = epoch_seconds(s.wall).to_numpy()
    check_epoch_seconds(s.wt.to_numpy(), "espn wall")
    out = []
    for eid, gp in s.groupby("eid"):
        gp = gp.sort_values(["wt", "seq"])
        wt = gp.wt.to_numpy()
        c = cd[cd.eid == eid]
        c = c[(c.end >= wt.min() + lag) & (c.end <= wt.max() + 120)]
        idx = np.searchsorted(wt, c.end.to_numpy() - lag, side="right") - 1
        ok = idx >= 0
        c, idx = c[ok], idx[ok]
        st = gp.iloc[idx].reset_index(drop=True)
        assert (st.wt.to_numpy() <= c.end.to_numpy() - lag).all(), "lookahead: state after candle end - lag"
        st["end"], st["bid"], st["ask"], st["kres"] = c.end.to_numpy(), c.bid.to_numpy(), c.ask.to_numpy(), c.result.to_numpy()
        out.append(st)
    x = pd.concat(out, ignore_index=True)
    x["mid"] = (x.bid + x.ask) / 2
    x = x[(x.ask - x.bid <= 0.05) & (x.bid > 0) & (x.mid >= 0.03) & (x.mid <= 0.97)]
    return x


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def logit_fit(X, y):
    import statsmodels.api as sm
    return sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit().params


def cmd_resid() -> int:
    s = load_states()
    x = minute_rows(s)
    tr = x[x.date < TRAIN_END]
    bb, bf = fit(tr, "base"), fit(tr, "full")
    k = kalshi_snaps(s)
    agree = float(((k.kres == "yes").astype(int) == k.home_won).mean())
    k["lb"], k["lf"], k["lk"] = logit(predict(bb, k, "base")), logit(predict(bf, k, "full")), logit(k.mid.to_numpy())
    y = k.home_won.to_numpy(float)
    X1 = np.column_stack([k.lk, k.lf - k.lk])
    X2 = np.column_stack([k.lk, k.lf - k.lb])
    p1, p2 = logit_fit(X1, y), logit_fit(X2, y)
    g = k.eid.to_numpy()
    ug = np.unique(g)
    gi = {e: np.where(g == e)[0] for e in ug}
    rng = np.random.default_rng(0)
    B1, B2 = [], []
    for _ in range(2000):
        ii = np.concatenate([gi[e] for e in rng.choice(ug, len(ug))])
        B1.append(logit_fit(X1[ii], y[ii])[2])
        B2.append(logit_fit(X2[ii], y[ii])[2])
    res = {"n_snaps": len(k), "n_games": int(k.eid.nunique()), "dates": [str(k.date.min().date()), str(k.date.max().date())],
           "kalshi_result_agrees_espn": agree,
           "ll_kalshi": float(llv(y, k.mid.to_numpy()).mean()), "ll_full": float(llv(y, predict(bf, k, "full")).mean()),
           "ll_base": float(llv(y, predict(bb, k, "base")).mean()),
           "primary_coef_full_minus_kalshi": float(p1[2]), "primary_ci": np.percentile(B1, [2.5, 97.5]).tolist(),
           "kalshi_coef": float(p1[1]),
           "secondary_coef_full_minus_base": float(p2[2]), "secondary_ci": np.percentile(B2, [2.5, 97.5]).tolist(),
           "sd_full_minus_kalshi_logit": float(np.std(k.lf - k.lk)), "sd_full_minus_base_logit": float(np.std(k.lf - k.lb))}
    (OUT / "gate_b_resid.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"ratings": cmd_ratings, "states": cmd_states, "oos": cmd_oos, "resid": cmd_resid}[cmd]())
