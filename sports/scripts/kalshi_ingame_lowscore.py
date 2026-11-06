"""Kalshi IN-GAME low-scoring sports (soccer 1X2, NHL): does a Poisson/Skellam in-game model beat Kalshi's live prices?

    python scripts/kalshi_ingame_lowscore.py prior-soccer      # ESPN 2023-24 big-5 goal minutes + stoppage (pre-Kalshi)
    python scripts/kalshi_ingame_lowscore.py build-soccer [--sealed]   # Kalshi events -> ESPN keyEvents + 1-min candles
    python scripts/kalshi_ingame_lowscore.py frame-soccer [--sealed]   # in-play snapshot frame (model p, quotes) -- no outcomes read
    python scripts/kalshi_ingame_lowscore.py lagcheck          # ESPN goal wallclock vs Kalshi jump (no outcomes)
    python scripts/kalshi_ingame_lowscore.py prior-nhl         # api-web.nhle.com 2023-24 pbp: goal rate + empty-net regime
    python scripts/kalshi_ingame_lowscore.py frame-nhl         # NHL dev (Apr-Jun 2025, the only non-sealed NHL Kalshi games)
    python scripts/kalshi_ingame_lowscore.py brier             # diagnostics AFTER edgelab registration: Brier by minutes-left / volume

Edgelab: dataset `kalshi_soccer_ingame` (vault.yaml), rule module edgelab/rules/ingame_lowscore.py, family kalshi_ingame.
Trials are registered BEFORE `edgelab evaluate` touches outcomes. This script never joins outcomes except in `brier`.

SOCCER MODEL (all parameters fixed from data before any Kalshi in-game outcome is read):
  pregame: last 1-min candle before ESPN kickoff where all 3 markets are two-sided (spread <= 5c); mids normalised ->
           (pH, pD, pA); (lam_h, lam_a) = least squares fit of an independent-Poisson 90-min model to (pH, pD, pA).
  time:    real elapsed minutes since ESPN half start (kickoff / start-2nd-half wallclock). Remaining-goal fraction
           R(half, elapsed) from the 2023-24 ESPN big-5 goal-minute profile (per displayed minute) + stoppage-goal rate
           and stoppage-length distribution (EPL/LaLiga 2023-24 summaries). Halftime -> R = second-half share.
  reds:    per red card, carded team remaining rate x0.75, opponent x1.25 (literature-sized constants, not fitted).
  state:   goals/reds with ESPN wallclock <= t. Probabilities via Skellam(lam_h*R*, lam_a*R*) shifted by the margin.
  anchor:  logit(p_model_t) += R_t * (logit(p_mkt_pre) - logit(p_model_pre)) per outcome, then renormalise -- the
           pregame residual (Poisson draw under-statement) decays with the remaining fraction.
SNAPSHOTS: every 1-min candle end t with KO+60s <= t <= FT-60s (FT = ESPN end-regular-time wallclock -- used ONLY as the
  settlement bound / event_start, never as a model input: the model uses the prior stoppage distribution).
  Flags (PIT): event_recent = a goal/red with wallclock in (t-180s, t]; jump = any of the 3 mids moved >= 8c over the
  last 3 candles (catches goals the ESPN feed has not posted yet).
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson, skellam

import sports.http as http
from sports.http import fetch

DATA = ROOT / "data"
API = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/soccer"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
PRIOR_SOC = DATA / "ingame_lowscore_soccer_prior.json"
SOC_GAMES = DATA / "ingame_lowscore_soccer_games.parquet"
SOC_CAND = DATA / "ingame_lowscore_soccer_candles.parquet"
SOC_FRAME = DATA / "ingame_lowscore_soccer_frame.parquet"
PRIOR_NHL = DATA / "ingame_lowscore_nhl_prior.json"
NHL_FRAME = DATA / "ingame_lowscore_nhl_frame.parquet"
DEV_HI = pd.Timestamp("2026-02-28")
SEALED_HI = pd.Timestamp("2026-10-31")
LEAGUE = {"KXEPLGAME": "eng.1", "KXLALIGAGAME": "esp.1", "KXBUNDESLIGAGAME": "ger.1", "KXSERIEAGAME": "ita.1",
          "KXLIGUE1GAME": "fra.1", "KXMLSGAME": "usa.1", "KXUCLGAME": "uefa.champions", "KXEFLCHAMPIONSHIPGAME": "eng.2",
          "KXEREDIVISIEGAME": "ned.1", "KXSCOTTISHPREMGAME": "sco.1", "KXBRASILEIROGAME": "bra.1",
          "KXLALIGA2GAME": "esp.2", "KXBUNDESLIGA2GAME": "ger.2", "KXEFLL1GAME": "eng.3", "KXLIGUE2GAME": "fra.2"}
RED_SELF, RED_OPP = 0.75, 1.25
GOAL_TYPES = ("goal", "own-goal", "penalty---scored")


def esec(t) -> np.ndarray:
    return ((pd.to_datetime(t, utc=True) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()


def pmap(fn, xs, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, xs))


def safe(fn):
    def g(*a, **k):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            return {"_err": str(e)[:200]}
    return g


sfetch = safe(fetch)


def _fast_kalshi(iv: float = 0.1) -> None:
    """Kalshi public reads allow ~20/s; keep ESPN (and every other host) at the module's polite 0.25 s."""
    import time

    def thr(host: str) -> None:
        gap = iv if "kalshi" in host else http.MIN_INTERVAL
        with http._lock:
            now = time.monotonic()
            slot = max(now, http._last_hit.get(host, 0.0) + gap)
            http._last_hit[host] = slot
        if slot > now:
            time.sleep(slot - now)
    http._throttle = thr


# ------------------------------------------------------------------------------------------------ ESPN parsing
def parse_clock(disp: str):
    """ESPN display "11'" / "45'+2'" / "90'+10'" -> (minute, stoppage)."""
    m = re.match(r"(\d+)'(?:\+(\d+)')?", disp or "")
    if not m:
        return None
    return int(m.group(1)), int(m.group(2) or 0)


def key_events(s: dict) -> dict | None:
    ke = s.get("keyEvents") or []
    by = {}
    goals, reds = [], []
    comp = (s.get("header") or {}).get("competitions", [{}])[0]
    teams = {x["team"]["id"]: x["homeAway"] for x in comp.get("competitors", [])}
    score = {x["homeAway"]: int(x.get("score") or 0) for x in comp.get("competitors", [])}
    for k in ke:
        ty = k["type"]["type"]
        w = k.get("wallclock")
        if ty in ("kickoff", "halftime", "start-2nd-half", "end-regular-time") and ty not in by:
            by[ty] = (w, (k.get("clock") or {}).get("displayValue", ""))
        side = teams.get((k.get("team") or {}).get("id"))
        if k.get("scoringPlay") and int((k.get("period") or {}).get("number", 9)) <= 2:
            goals.append({"side": side, "w": w, "disp": (k.get("clock") or {}).get("displayValue", ""), "type": ty})
        if ty in ("red-card", "yellow-red-card", "second-yellow") or "red" in ty:
            reds.append({"side": side, "w": w, "disp": (k.get("clock") or {}).get("displayValue", ""), "type": ty})
    return {"by": by, "goals": goals, "reds": reds, "score": score, "teams": teams}


# ------------------------------------------------------------------------------------------------ soccer prior (2023-24)
def cmd_prior_soccer() -> int:
    seasons = {"eng.1": ("2023-08-11", "2024-05-19"), "esp.1": ("2023-08-11", "2024-05-26"),
               "ger.1": ("2023-08-18", "2024-05-18"), "ita.1": ("2023-08-19", "2024-06-02"),
               "fra.1": ("2023-08-11", "2024-05-19")}
    jobs = [(lg, d.strftime("%Y%m%d")) for lg, (a, b) in seasons.items() for d in pd.date_range(a, b)]
    sbs = pmap(lambda j: sfetch(f"{ESPN}/{j[0]}/scoreboard", {"dates": j[1]}), jobs)
    games, goals = {}, []
    for (lg, _), sb in zip(jobs, sbs):
        for e in (sb or {}).get("events", []):
            c = e["competitions"][0]
            if not c.get("status", {}).get("type", {}).get("completed") or e["id"] in games:
                continue
            games[e["id"]] = lg
            for d in c.get("details", []):
                if d.get("scoringPlay") and not d.get("shootout"):
                    pc = parse_clock((d.get("clock") or {}).get("displayValue"))
                    if pc:
                        goals.append({"eid": e["id"], "min": pc[0], "st": pc[1]})
    g = pd.DataFrame(goals)
    n = len(games)
    # summaries (EPL + LaLiga) for stoppage lengths: halftime "45'+x'" and end-regular-time "90'+y'"
    ids = [k for k, v in games.items() if v in ("eng.1", "esp.1")]
    ss = pmap(lambda i: sfetch(f"{ESPN}/{games[i]}/summary", {"event": i}), ids)
    st1, st2 = [], []
    for s in ss:
        ke = key_events(s) if s and "_err" not in s else None
        if not ke:
            continue
        a = parse_clock(ke["by"].get("halftime", (None, ""))[1])
        b = parse_clock(ke["by"].get("end-regular-time", (None, ""))[1])
        if a and a[0] == 45:
            st1.append(a[1])
        if b and b[0] == 90:
            st2.append(b[1])
    h1 = g[g["min"] <= 45]
    h2 = g[g["min"] > 45]
    reg1 = np.array([(h1["min"].eq(m) & h1.st.eq(0)).sum() for m in range(1, 46)]) / n
    reg2 = np.array([(h2["min"].eq(m) & h2.st.eq(0)).sum() for m in range(46, 91)]) / n
    s1g, s2g = (h1.st > 0).sum() / n, (h2.st > 0).sum() / n
    # stoppage minutes: displayed "+x" at the whistle means x full-ish minutes; length L = x + 0.5
    L1, L2 = np.array(st1) + 0.5, np.array(st2) + 0.5
    prior = {"n_games": n, "n_goals": int(len(g)), "goals_per_game": len(g) / n, "reg1": reg1.tolist(),
             "reg2": reg2.tolist(), "stop1_goals_pg": s1g, "stop2_goals_pg": s2g, "L1": L1.tolist(), "L2": L2.tolist(),
             "q1": s1g / L1.mean(), "q2": s2g / L2.mean(), "n_stoppage_games": [len(L1), len(L2)],
             "source": "ESPN scoreboard details, big-5 2023-24; stoppage from EPL+LaLiga 2023-24 summaries"}
    PRIOR_SOC.write_text(json.dumps(prior))
    print(f"prior: {n} games, {len(g)} goals ({len(g)/n:.3f}/game); 1H reg {reg1.sum():.3f} + stop {s1g:.3f}; "
          f"2H reg {reg2.sum():.3f} + stop {s2g:.3f}; mean stoppage 1H {L1.mean():.2f} 2H {L2.mean():.2f} min "
          f"(n {len(L1)}/{len(L2)})")
    return 0


class SoccerTime:
    """Remaining expected goals (share of a whole game's goals) from (half, real elapsed minutes in the half)."""

    def __init__(self, prior: dict):
        self.reg = {1: np.array(prior["reg1"]), 2: np.array(prior["reg2"])}
        self.q = {1: prior["q1"], 2: prior["q2"]}
        self.L = {1: np.array(prior["L1"]), 2: np.array(prior["L2"])}
        self.full = {k: self.reg[k].sum() + self.q[k] * self.L[k].mean() for k in (1, 2)}
        self.total = self.full[1] + self.full[2]

    def remaining_half(self, k: int, e: float) -> float:
        if e < 45:
            i = int(np.floor(e))
            frac = 1 - (e - i)
            reg = self.reg[k][i] * frac + self.reg[k][i + 1:].sum()
            return reg + self.q[k] * self.L[k].mean()
        s = e - 45
        rest = self.L[k][self.L[k] > s] - s
        return self.q[k] * (rest.mean() if len(rest) else 0.5)

    def R(self, k: int, e: float | None) -> float:
        """k = 1, 2; e = elapsed real minutes in that half; k = 1.5 -> halftime."""
        if k == 1.5:
            return self.full[2] / self.total
        r = self.remaining_half(int(k), e)
        if k == 1:
            r += self.full[2]
        return r / self.total


# ------------------------------------------------------------------------------------------------ Poisson / Skellam
def probs_1x2(lh, la, margin=0):
    """P(home wins, draw, away wins) at 90' when the home side leads by `margin` and lh/la goals remain expected."""
    lh, la = max(lh, 1e-9), max(la, 1e-9)
    pd_ = skellam.pmf(-margin, lh, la)
    ph = skellam.sf(-margin, lh, la)          # P(D > -margin)
    return ph, pd_, max(0.0, 1 - ph - pd_)


def fit_lams(p):
    p = np.asarray(p, float)

    def loss(x):
        lh, la = np.exp(x)
        return float(np.sum((np.array(probs_1x2(lh, la)) - p) ** 2))
    best = None
    for x0 in ([0.3, 0.1], [0.1, 0.3], [0.5, -0.2]):
        r = minimize(loss, np.array(x0), method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    return np.exp(best.x), best.fun


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------------------------------------ soccer build
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    stop = {"fc", "afc", "cf", "sc", "ac", "the", "club", "de", "and", "hove", "albion", "united", "city", "real",
            "sporting", "athletic", "atletico", "cd", "ud", "sv", "vfb", "vfl", "tsg", "fsv", "rc", "as", "ss", "us",
            "1", "04", "05", "1899", "1846", "hotspur", "wanderers", "rovers", "town", "county"}
    return " ".join(w for w in s.split() if w not in stop)


def name_match(k: str, names: list[str]) -> bool:
    a = _norm(k)
    if not a:
        return False
    for n in names:
        b = _norm(n)
        if not b:
            continue
        if a == b or a in b or b in a or (set(a.split()) & set(b.split())):
            return True
    return False


def espn_names(c):
    return {x["homeAway"]: [x["team"].get(f, "") for f in ("displayName", "shortDisplayName", "name", "location",
                                                           "abbreviation")] for x in c["competitors"]}


def kalshi_events(sealed: bool) -> pd.DataFrame:
    d = pd.read_parquet(DATA / "kalshi_soccer_prices.parquet")
    hi = SEALED_HI if sealed else DEV_HI
    lo = DEV_HI + pd.Timedelta(days=1) if sealed else pd.Timestamp("2025-04-01")
    d = d[(d.event_date >= lo) & (d.event_date <= hi)]
    w = d.pivot_table(index=["series", "event_ticker", "event_date", "decision_ts"], columns="side",
                      values=["ticker", "yes_sub_title"], aggfunc="first")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index().dropna(subset=["ticker_H", "ticker_D", "ticker_A"])


def kcandles(series: str, tk: str, s: int, e: int):
    for path in (f"{API}/historical/markets/{tk}/candlesticks", f"{API}/series/{series}/markets/{tk}/candlesticks"):
        j = sfetch(path, {"start_ts": s, "end_ts": e, "period_interval": 1})
        c = (j or {}).get("candlesticks") or []
        if c:
            out = []
            for x in c:
                def g(k):
                    v = (x.get(k) or {})
                    v = v.get("close_dollars", v.get("close"))
                    return np.nan if v is None else float(v)
                out.append((x["end_period_ts"], g("yes_bid"), g("yes_ask"), float(x.get("volume_fp", x.get("volume")) or 0)))
            return out
    return []


def cmd_build_soccer(sealed: bool) -> int:
    _fast_kalshi()
    ev = kalshi_events(sealed)
    print(f"Kalshi events ({'sealed' if sealed else 'dev'}): {len(ev)}", flush=True)
    jobs = sorted({(LEAGUE[s], (d + pd.Timedelta(days=o)).strftime("%Y%m%d"))
                   for s, d in zip(ev.series, ev.event_date) for o in (-1, 0, 1)})
    sbs = dict(zip(jobs, pmap(lambda j: sfetch(f"{ESPN}/{j[0]}/scoreboard", {"dates": j[1]}), jobs)))
    rows, nomatch = [], 0
    for _, r in ev.iterrows():
        lg = LEAGUE[r.series]
        ko_est = pd.Timestamp(int(r.decision_ts) + 4 * 3600 - 115 * 60, unit="s", tz="UTC")  # close-4h -> ~kickoff
        cands = []
        for o in (-1, 0, 1):
            for e in (sbs.get((lg, (r.event_date + pd.Timedelta(days=o)).strftime("%Y%m%d"))) or {}).get("events", []):
                c = e["competitions"][0]
                nm = espn_names(c)
                if name_match(r.yes_sub_title_H, nm["home"]) and name_match(r.yes_sub_title_A, nm["away"]):
                    dt = abs((pd.Timestamp(e["date"]) - ko_est).total_seconds())
                    cands.append((dt, e["id"]))
        cands = sorted(set(cands))
        if not cands or cands[0][0] > 3 * 3600:
            nomatch += 1
            continue
        rows.append({**r.to_dict(), "league": lg, "eid": cands[0][1]})
    m = pd.DataFrame(rows)
    print(f"ESPN matched {len(m)}/{len(ev)} (no match {nomatch})", flush=True)
    sums = pmap(lambda x: sfetch(f"{ESPN}/{x[0]}/summary", {"event": x[1]}), list(zip(m.league, m.eid)))
    keep, why = [], {}
    for (_, r), s in zip(m.iterrows(), sums):
        ke = key_events(s) if s and "_err" not in s else None
        bad = None
        if not ke:
            bad = "no_summary"
        else:
            by = ke["by"]
            if not all(by.get(k, (None,))[0] for k in ("kickoff", "halftime", "start-2nd-half", "end-regular-time")):
                bad = "missing_phase_wallclock"
            elif ke["score"].get("home", -1) != sum(g["side"] == "home" for g in ke["goals"]) or \
                    ke["score"].get("away", -1) != sum(g["side"] == "away" for g in ke["goals"]):
                bad = "goal_count_mismatch"   # extra time / shootout games or missing goal events
            elif any(not g["w"] for g in ke["goals"] + ke["reds"]):
                bad = "event_without_wallclock"
            else:
                ko, ht, h2, ft = (pd.Timestamp(by[k][0]) for k in ("kickoff", "halftime", "start-2nd-half", "end-regular-time"))
                if not (ko < ht < h2 < ft) or not (40 <= (ht - ko).total_seconds() / 60 <= 60) or \
                        not (40 <= (ft - h2).total_seconds() / 60 <= 70):
                    bad = "phase_order"
                else:
                    # goal wallclock vs displayed minute (+/- 4 min) -- the alignment check
                    for g in ke["goals"] + ke["reds"]:
                        pc = parse_clock(g["disp"])
                        if not pc:
                            bad = "unparsed_clock"
                            break
                        mm, st = pc
                        if mm <= 45:
                            el = (pd.Timestamp(g["w"]) - ko).total_seconds() / 60
                            exp = mm - 0.5 + st
                        else:
                            el = (pd.Timestamp(g["w"]) - h2).total_seconds() / 60
                            exp = mm - 45 - 0.5 + st
                        if abs(el - exp) > 4:
                            bad = "wallclock_vs_clock"
                            break
        if bad:
            why[bad] = why.get(bad, 0) + 1
            continue
        keep.append({**r.to_dict(), "ko": by["kickoff"][0], "ht": by["halftime"][0], "h2": by["start-2nd-half"][0],
                     "ft": by["end-regular-time"][0], "goals": json.dumps(ke["goals"]), "reds": json.dumps(ke["reds"])})
    G = pd.DataFrame(keep)
    for c in ("ko", "ht", "h2", "ft"):
        G[c] = pd.to_datetime(G[c], utc=True)
    print(f"games with clean timeline: {len(G)}; dropped {why}", flush=True)
    # candles: 3 markets x [KO-30min, FT+10min]
    jobs = []
    for _, r in G.iterrows():
        s0, e0 = int(r.ko.timestamp()) - 1800, int(r.ft.timestamp()) + 600
        for side in "HDA":
            jobs.append((r.series, r[f"ticker_{side}"], s0, e0, r.event_ticker, side))
    cs = pmap(lambda j: kcandles(*j[:4]), jobs, workers=8)
    out = []
    for j, c in zip(jobs, cs):
        for t, b, a, v in c:
            out.append((j[4], j[1], j[5], t, b, a, v))
    C = pd.DataFrame(out, columns=["event_ticker", "ticker", "side", "end", "bid", "ask", "vol"])
    print(f"candles: {len(C)} rows, {C.ticker.nunique()} markets of {len(jobs)}", flush=True)
    suf = "_sealed" if sealed else ""
    G.to_parquet(SOC_GAMES.with_name(SOC_GAMES.stem + suf + ".parquet"), index=False)
    C.to_parquet(SOC_CAND.with_name(SOC_CAND.stem + suf + ".parquet"), index=False)
    return 0


# ------------------------------------------------------------------------------------------------ soccer frame
def soccer_frame(G: pd.DataFrame, C: pd.DataFrame, T: SoccerTime) -> pd.DataFrame:
    res = pd.read_parquet(DATA / "kalshi_soccer_prices.parquet", columns=["ticker", "result"]).drop_duplicates("ticker")
    rmap = dict(zip(res.ticker, res.result))
    out = []
    for _, g in G.iterrows():
        c = C[C.event_ticker == g.event_ticker]
        w = c.pivot_table(index="end", columns="side", values=["bid", "ask", "vol"], aggfunc="last")
        if w.empty or not all(("bid", s) in w.columns for s in "HDA"):
            continue
        w = w.sort_index()
        bid = w["bid"][list("HDA")]
        ask = w["ask"][list("HDA")]
        vol = w["vol"][list("HDA")].fillna(0)
        mid = (bid + ask) / 2
        ko, ht, h2, ft = (int(g[k].timestamp()) for k in ("ko", "ht", "h2", "ft"))
        # pregame anchor: last candle ending <= KO with all three two-sided, spread <= 5c
        ok = ((bid > 0) & (ask < 1) & (ask - bid <= 0.05)).all(axis=1)
        pre = ok[(ok.index <= ko)]
        pre = pre[pre]
        if pre.empty:
            continue
        tp = pre.index[-1]
        pm = mid.loc[tp].to_numpy()
        pm = pm / pm.sum()
        (lh, la), fl = fit_lams(pm)
        p0 = np.array(probs_1x2(lh, la))
        resid = logit(pm) - logit(p0)
        pre_vol = float(vol[(vol.index > ko - 1800) & (vol.index <= ko)].sum().sum())
        game_vol = float(vol[(vol.index > ko) & (vol.index <= ft)].sum().sum())
        goals = json.loads(g.goals)
        reds = json.loads(g.reds)
        gw = [(int(pd.Timestamp(x["w"]).timestamp()), x["side"]) for x in goals]
        rw = [(int(pd.Timestamp(x["w"]).timestamp()), x["side"]) for x in reds]
        ev_times = np.array(sorted([t for t, _ in gw + rw]))
        mids = mid.to_numpy()
        idx = mid.index.to_numpy()
        cumv = vol.sum(axis=1).cumsum()
        for i, t in enumerate(idx):
            if t < ko + 60 or t > ft - 60:
                continue
            # state at t (events with wallclock <= t)
            sh = sum(1 for tt, s in gw if tt <= t and s == "home")
            sa = sum(1 for tt, s in gw if tt <= t and s == "away")
            rh = sum(1 for tt, s in rw if tt <= t and s == "home")
            ra = sum(1 for tt, s in rw if tt <= t and s == "away")
            # time: halves from ESPN half-start wallclocks; the half-end (ht) wallclock is used only to know we are in
            # the break, which is public at the time (whistle) -- t > ht means the first half has ended.
            if t <= ht:
                k, e = 1, (t - ko) / 60
            elif t < h2:
                k, e = 1.5, None
            else:
                k, e = 2, (t - h2) / 60
            R = T.R(k, e)
            mh = (RED_SELF ** rh) * (RED_OPP ** ra)
            ma = (RED_SELF ** ra) * (RED_OPP ** rh)
            p = np.array(probs_1x2(lh * R * mh, la * R * ma, sh - sa))
            pa = 1 / (1 + np.exp(-(logit(p) + R * resid)))
            pa = pa / pa.sum()
            recent = bool(((ev_times > t - 180) & (ev_times <= t)).any())
            lo3 = max(0, i - 3)
            jump = bool(np.nanmax(np.abs(mids[i] - mids[lo3])) >= 0.08) if i > 0 else False
            t10 = t + 600
            j10 = np.searchsorted(idx, min(t10, ft - 60), side="right") - 1
            minutes_left = (90 - (e if k == 1 else 45 if k == 1.5 else 45 + e)) if k != 1.5 else 45.0
            for si, side in enumerate("HDA"):
                b, a = bid.iloc[i, si], ask.iloc[i, si]
                tk = g[f"ticker_{side}"]
                out.append({"event_ticker": g.event_ticker, "series": g.series, "date": g.event_date, "ticker": tk,
                            "side": side, "t": int(t), "ko": ko, "ft": ft, "half": k,
                            "elapsed": np.nan if e is None else e, "minutes_left": float(minutes_left),
                            "score_h": sh, "score_a": sa, "reds_h": rh, "reds_a": ra, "R": R,
                            "lam_h": lh, "lam_a": la, "fit_loss": fl, "p_pre_mkt": pm[si], "p_model": pa[si],
                            "p_model_raw": p[si], "bid": b, "ask": a, "mid_t10_bid": bid.iloc[j10, si],
                            "mid_t10_ask": ask.iloc[j10, si], "t10": int(idx[j10]), "pre_vol": pre_vol,
                            "game_vol": game_vol, "cum_vol": float(cumv.iloc[i]), "event_recent": recent,
                            "jump": jump, "result": rmap.get(tk)})
    return pd.DataFrame(out)


def cmd_frame_soccer(sealed: bool) -> int:
    suf = "_sealed" if sealed else ""
    G = pd.read_parquet(SOC_GAMES.with_name(SOC_GAMES.stem + suf + ".parquet"))
    C = pd.read_parquet(SOC_CAND.with_name(SOC_CAND.stem + suf + ".parquet"))
    T = SoccerTime(json.loads(PRIOR_SOC.read_text()))
    # timestamp trap check: candle ends are unix seconds near the ESPN kickoff, not ms/us
    ko = esec(G.ko)
    assert 1.6e9 < ko.min() < 1.9e9 and 1.6e9 < C["end"].min() < 1.9e9, "epoch unit trap"
    F = soccer_frame(G, C, T)
    out = SOC_FRAME.with_name(SOC_FRAME.stem + suf + ".parquet")
    if sealed:
        old = pd.read_parquet(SOC_FRAME) if SOC_FRAME.exists() else None
        F = pd.concat([old, F], ignore_index=True) if old is not None else F
        out = SOC_FRAME
    F.to_parquet(out, index=False)
    print(f"frame: {len(F)} rows, {F.event_ticker.nunique()} events, dates {F.date.min()}..{F.date.max()} -> {out}")
    print(F[["p_model", "bid", "ask", "R", "minutes_left"]].describe().T.to_string())
    return 0


def cmd_lagcheck() -> int:
    """Price-only alignment check: for each goal, minute the home-market mid first moves >= 8c vs ESPN wallclock."""
    G = pd.read_parquet(SOC_GAMES)
    C = pd.read_parquet(SOC_CAND)
    lags = []
    for _, g in G.sample(min(400, len(G)), random_state=0).iterrows():
        c = C[(C.event_ticker == g.event_ticker) & (C.side == "H")].sort_values("end")
        m = ((c.bid + c.ask) / 2).to_numpy()
        t = c["end"].to_numpy()
        for x in json.loads(g.goals):
            w = int(pd.Timestamp(x["w"]).timestamp())
            i0 = np.searchsorted(t, w - 600)
            base = np.searchsorted(t, w - 300)
            if base >= len(t) or base < 1:
                continue
            ref = m[base - 1]
            mv = np.where(np.abs(m[base:] - ref) >= 0.08)[0]
            if len(mv):
                lags.append((t[base + mv[0]] - w) / 60)
    lags = np.array(lags)
    print(f"goals {len(lags)}: Kalshi first >=8c move minus ESPN wallclock (min): quantiles "
          f"{np.round(np.quantile(lags, [.05, .25, .5, .75, .95]), 2)}; share moved >1 min BEFORE ESPN wallclock "
          f"{np.mean(lags < -1):.3f}")
    return 0


# ------------------------------------------------------------------------------------------------ NHL
def cmd_prior_nhl() -> int:
    """2023-24 regular season pbp (api-web.nhle.com): regulation goals/60 per team, goal rates in the last 3 min of
    regulation by margin state (empty-net regime) as multipliers of the base rate."""
    ids = [f"2023020{i:03d}" if i < 1000 else f"202302{i:04d}" for i in range(1, 1313)]
    pbps = pmap(lambda g: sfetch(f"https://api-web.nhle.com/v1/gamecenter/{g}/play-by-play"), ids)
    base_goals, base_min = 0, 0.0
    late = {}   # key (abs margin 1/2, role trailer/leader) -> goals, team-minutes
    for p in pbps:
        if not p or "_err" in p:
            continue
        home = p.get("homeTeam", {}).get("id")
        sc = {"h": 0, "a": 0}
        evs = []
        for pl in p.get("plays", []):
            per = pl.get("periodDescriptor", {}).get("number", 0)
            if per > 3:
                break
            tm, ss = pl.get("timeInPeriod", "00:00").split(":")
            el = (per - 1) * 20 + int(tm) + int(ss) / 60
            if pl.get("typeDescKey") == "goal":
                side = "h" if pl.get("details", {}).get("eventOwnerTeamId") == home else "a"
                evs.append((el, side))
        # walk the timeline in 1/6-minute steps over regulation
        gi = 0
        step = 1 / 6
        for k in range(int(60 / step)):
            t0, t1 = k * step, (k + 1) * step
            m = sc["h"] - sc["a"]
            new = [s for (el, s) in evs[gi:] if t0 <= el < t1]
            if t0 >= 57 and abs(m) in (1, 2):
                lead = "h" if m > 0 else "a"
                for role, side in (("leader", lead), ("trailer", "a" if lead == "h" else "h")):
                    key = f"{abs(m)}_{role}"
                    gk, mk = late.get(key, (0, 0.0))
                    late[key] = (gk + sum(1 for s in new if s == side), mk + step)
            elif t0 < 57:
                base_goals += len(new)
                base_min += 2 * step
            for s in new:
                sc[s] += 1
            gi += len(new)
    base = base_goals / base_min          # goals per team-minute, minutes 0-57
    mult = {k: (g / m) / base if m else None for k, (g, m) in late.items()}
    prior = {"base_per_team_min": base, "late_mult": mult, "late_counts": late, "n_games": sum(1 for p in pbps if p and "_err" not in p),
             "source": "api-web.nhle.com 2023-24 regular season pbp"}
    PRIOR_NHL.write_text(json.dumps(prior))
    print(json.dumps({k: (round(v, 3) if v else v) for k, v in mult.items()}), f"base {base*60:.3f} goals/60/team")
    return 0


def nhl_probs(lh_rate, la_rate, margin, t_left, prior, ot_home):
    """P(home win) incl. OT/shootout. Rates in goals per minute at the pregame strength. Last 3 min with |margin| 1-2:
    each team's rate is multiplied by its empty-net-regime multiplier (fit 2023-24)."""
    lm = prior["late_mult"]
    late_t = min(t_left, 3.0)
    early_t = max(0.0, t_left - 3.0)
    # early part: Skellam over early_t then late regime conditional on the margin at minute 57 -> enumerate
    eh, ea = lh_rate * early_t, la_rate * early_t
    pw = 0.0
    ks = np.arange(0, 12)
    ph_k, pa_k = poisson.pmf(ks, eh), poisson.pmf(ks, ea)
    for i, pi in enumerate(ph_k):
        for j, pj in enumerate(pa_k):
            w = pi * pj
            if w < 1e-9:
                continue
            m = margin + i - j
            if abs(m) in (1, 2) and late_t > 0:
                lead_home = m > 0
                mh = lm.get(f"{abs(m)}_{'leader' if lead_home else 'trailer'}") or 1.0
                ma = lm.get(f"{abs(m)}_{'trailer' if lead_home else 'leader'}") or 1.0
            else:
                mh = ma = 1.0
            lh2, la2 = lh_rate * late_t * mh, la_rate * late_t * ma
            if late_t <= 0:
                pw += w * (1.0 if m > 0 else ot_home if m == 0 else 0.0)
                continue
            p_up = skellam.sf(-m, max(lh2, 1e-9), max(la2, 1e-9))
            p_tie = skellam.pmf(-m, max(lh2, 1e-9), max(la2, 1e-9))
            pw += w * (p_up + p_tie * ot_home)
    return float(pw)


def cmd_frame_nhl() -> int:
    prior = json.loads(PRIOR_NHL.read_text())
    base = prior["base_per_team_min"]
    C = pd.read_parquet(DATA / "ingame_longshot_nhl_candles.parquet")
    G = pd.read_parquet(DATA / "ingame_longshot_nhl_games.parquet")
    out = []
    for _, g in G.iterrows():
        s = fetch(f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/summary", {"event": g.eid})
        comp = s["header"]["competitions"][0]
        tid = {x["team"]["id"]: x["homeAway"] for x in comp["competitors"]}
        goals, phase = [], []
        for pl in s.get("plays", []):
            per = pl.get("period", {}).get("number", 0)
            w = pl.get("wallclock")
            if not w:
                continue
            cl = pl.get("clock", {}).get("displayValue", "20:00")
            mm, ss = (cl.split(":") + ["0"])[:2]
            left_in_per = max(0.0, 20 - (int(mm) + float(ss) / 60))   # ESPN NHL clock counts ELAPSED time
            phase.append((int(pd.Timestamp(w).timestamp()), per, left_in_per))
            if pl.get("scoringPlay"):
                goals.append((int(pd.Timestamp(w).timestamp()), tid.get((pl.get("team") or {}).get("id"))))
        phase.sort()
        if not phase:
            continue
        c = C[C.gid == g.eid].pivot_table(index="end", columns="side", values=["bid", "ask"], aggfunc="last").sort_index()
        if c.empty or ("bid", "home") not in c.columns or ("bid", "away") not in c.columns:
            continue
        hb, ha = c["bid"]["home"], c["ask"]["home"]
        start = phase[0][0]
        pre = c[(c.index <= start) & (hb > 0) & (ha < 1) & (ha - hb <= 0.05)]
        if pre.empty:
            continue
        pm = float(((pre["bid"]["home"] + pre["ask"]["home"]) / 2).iloc[-1])
        # pregame strengths: total = 2 x base x 60 (2023-24 league avg); split so P(home win) at t=0 matches pm
        tot = 2 * base

        def f(x):
            r = 1 / (1 + np.exp(-x))
            return nhl_probs(tot * r, tot * (1 - r), 0, 60.0, prior, r) - pm
        lo_, hi_ = -3.0, 3.0
        for _ in range(40):
            mid_ = (lo_ + hi_) / 2
            if f(mid_) > 0:
                hi_ = mid_
            else:
                lo_ = mid_
        r = 1 / (1 + np.exp(-(lo_ + hi_) / 2))
        lh, la = tot * r, tot * (1 - r)
        res = C[(C.gid == g.eid) & (C.side == "home")].result.iloc[0]
        for t in c.index:
            if t < start + 60:
                continue
            past = [x for x in phase if x[0] <= t]
            if not past:
                continue
            _, per, left = past[-1]
            if per > 3:
                break
            t_left = (3 - per) * 20 + left
            sh = sum(1 for w, s in goals if w <= t and s == "home")
            sa = sum(1 for w, s in goals if w <= t and s == "away")
            p = nhl_probs(lh, la, sh - sa, t_left, prior, r)
            b, a = c["bid"]["home"].get(t), c["ask"]["home"].get(t)
            out.append({"gid": g.eid, "date": g.start.date(), "t": int(t), "minutes_left": t_left, "margin": sh - sa,
                        "p_model": p, "bid": b, "ask": a, "p_pre_mkt": pm, "result": res})
    F = pd.DataFrame(out)
    F.to_parquet(NHL_FRAME, index=False)
    print(f"NHL frame: {len(F)} rows, {F.gid.nunique()} games")
    return 0


# ------------------------------------------------------------------------------------------------ diagnostics (post-registration)
def brier_tables(F: pd.DataFrame, label: str, games_col: str) -> dict:
    F = F[(F.bid > 0) & (F.ask < 1) & F.result.isin(["yes", "no"])].copy()
    F["y"] = (F.result == "yes").astype(float)
    F["mid"] = (F.bid + F.ask) / 2
    F["ml_bucket"] = pd.cut(F.minutes_left, [-1, 5, 15, 30, 45, 60, 75, 200],
                            labels=["0-5", "5-15", "15-30", "30-45", "45-60", "60-75", "75+"])
    out = {}

    def tab(g):
        return pd.Series({"n_rows": len(g), "games": g[games_col].nunique(), "brier_kalshi": np.mean((g.mid - g.y) ** 2),
                          "brier_model": np.mean((g.p_model - g.y) ** 2)})
    t1 = F.groupby("ml_bucket", observed=True).apply(tab)
    t1["model_minus_kalshi"] = t1.brier_model - t1.brier_kalshi
    out["by_minutes_left"] = t1
    if "game_vol" in F.columns:
        gv = F.drop_duplicates(games_col).set_index(games_col).game_vol
        q = gv.quantile([1 / 3, 2 / 3]).to_numpy()
        F["vol_tercile"] = np.where(F.game_vol <= q[0], "low", np.where(F.game_vol <= q[1], "mid", "high"))
        t2 = F.groupby("vol_tercile").apply(tab)
        t2["model_minus_kalshi"] = t2.brier_model - t2.brier_kalshi
        out["by_volume_tercile"] = t2
        out["vol_cuts"] = q.tolist()
    print(f"== {label} ==")
    for k, v in out.items():
        print(k)
        print(v.round(4).to_string() if hasattr(v, "round") else v)
    return out


def cmd_brier() -> int:
    F = pd.read_parquet(SOC_FRAME)
    F = F[pd.to_datetime(F.date) <= DEV_HI]           # dev only
    F = F[~F.event_recent & ~F.jump]
    o = brier_tables(F, "SOCCER dev (excl. 3 min after goals/reds and price jumps)", "event_ticker")
    o2 = brier_tables(F[F.side == "D"], "SOCCER dev, Tie market only", "event_ticker")
    res = {"soccer": {k: (v.reset_index().to_dict("records") if hasattr(v, "reset_index") else v) for k, v in o.items()},
           "soccer_tie": {k: (v.reset_index().to_dict("records") if hasattr(v, "reset_index") else v) for k, v in o2.items()}}
    if NHL_FRAME.exists():
        N = pd.read_parquet(NHL_FRAME)
        o3 = brier_tables(N, "NHL dev Apr-Jun 2025 (home market)", "gid")
        res["nhl"] = {k: (v.reset_index().to_dict("records") if hasattr(v, "reset_index") else v) for k, v in o3.items()}
    p = ROOT / "results" / "kalshi_ingame_lowscore_brier.json"
    p.write_text(json.dumps(res, default=str, indent=1))
    print(f"-> {p}")
    return 0


def main() -> int:
    a = sys.argv[1:]
    cmd = a[0] if a else ""
    sealed = "--sealed" in a
    return {"prior-soccer": cmd_prior_soccer, "build-soccer": lambda: cmd_build_soccer(sealed),
            "frame-soccer": lambda: cmd_frame_soccer(sealed), "lagcheck": cmd_lagcheck, "prior-nhl": cmd_prior_nhl,
            "frame-nhl": cmd_frame_nhl, "brier": cmd_brier}[cmd]()


if __name__ == "__main__":
    sys.exit(main())
