"""NHL + NFL OPENING-LINE LOGGER (forward paper for L34 NHL and L33 NFL). Paper only -- no orders, ever.

    python scripts/open_lines_logger.py tick   --league nhl|nfl   # one pass (launchd, every 15 min)
    python scripts/open_lines_logger.py report --league nhl|nfl   # picks, W-L, ROI (no-vig / Kalshi bid+ask), CLV, kill
    python scripts/open_lines_logger.py show   --league nhl|nfl   # tracked games: open vs latest vs model

Why: the simple NHL model (notes/nhl_simple_model.md, L34) and NFL model (notes/nfl_simple_model.md, L33) both LEAD
the opener on dev (NHL ML CLV +1.04 prob pts, totals +0.073 goals; NFL spread/total CLV +0.6..+0.8 pts, p<.001) but
neither pays at the offered opening price (vig). The only route to money is near-opener prices at low vig (no-vig /
Kalshi maker). Same design as scripts/nba_open_totals.py (L32): catch the first Pinnacle number we see, paper-bet the
model gap, record what Kalshi offered at that moment, and watch CLV.

PRE-REGISTERED 2026-10-25 (before the first tick of either league; nothing here was tuned on forward data):
  OPEN       = first Pinnacle full-game main line seen for a game/market (period 0, not alternate, pre-start), with
               prices. 15-min polling, so "open" = our first sighting, not the book's true open.
  FIRST RUN  = games already listed at the league's first tick are marked pre-existing and never picked.
  DECISION   = first tick at which (a) the market has an OPEN and (b) the model returns a number for the game, and
               the game starts >= 30 min later. First sightings < 30 min before start are logged, never picked.
               One pick per game per market; flat 1u; model and inputs logged at the decision tick.
  NHL (L34), model = nhl_simple_live.predict(date) (dev-fit stack, live NHL-API features, STRICT projected goalie =
               last-10 modal starter; no probable-goalie feed -- see that file; NHL 2025-26 sealed -> not used):
    ML       : gap = p_home_model - p_home_open_novig (proportional de-vig of the open ML); |gap| >= 0.03 -> bet
               home if gap > 0 else away. (= dev T1-T3 threshold ML_X .03)
    TOTAL    : gap = P_model(over OPEN line | no push) - p_over_open_novig; |gap| >= 0.04 -> over if gap > 0 else
               under. (= dev T4-T6 threshold TOT_X .04)
    CLV unit : probability points of the bet side at the OPEN line. ML: novig(last pre-start) - novig(open).
               TOTAL: market mean mu solved from (line, novig over) at open and at last pre-start (dev implied_mu,
               lh = la = mu/2), CLV = P_close(side at open line) - P_open(side); goals shift logged alongside.
  NFL (L33), model = nfl_simple_model.py `week --json --week N` (2020-25 refit; spread = model B, total = model C,
               the `week` default and the note's live rule after the 10-25 refit; total B logged, never picked on):
    SPREAD   : gap = model home margin - (-open home line); |gap| >= 2.0 pts -> home side of the open line if
               gap > 0 else away.
    TOTAL    : gap = model total - open total; |gap| >= 2.5 pts -> over if gap > 0 else under.
    CLV unit : points of line movement in the bet's favour, open -> last pre-start Pinnacle line.
  PRICES recorded at the decision tick (the money metrics, primary first):
    (a) Kalshi MAKER at the bid of the matching contract (same strike as the open line; NHL ML = the team's
        KXNHLGAME market; NFL spread = "<fav> wins by over s" with s = |open line|, YES for the fav, NO for the dog;
        totals = KX*TOTAL strike == open line, YES over / NO under). ROI assumes a fill at the bid, maker fee 0 --
        an upper bound (fills are adversely selected; see K-FAV).
    (b) Pinnacle OPEN no-vig price of the bet side (proxy for a resting maker order at fair).
    (c) Kalshi TAKER: ask + 0.07 p(1-p) fee at the same contract; (d) Pinnacle open offered price; (e) Pinnacle line
        and prices at the decision tick. Integer lines (no Kalshi .5 strike) have no matching contract -> (a)/(c) blank;
        the half-point-WORSE .5 strike is recorded as kal_alt_* and reported separately (secondary).
  GRADING    = vs the OPEN line (push on exact integer = 0); NHL ML includes OT/SO (as Pinnacle and Kalshi).
  JUDGE      = CLV is the leading indicator; KILL a league if its mean CLV (all its markets pooled, in the units
               above) is <= 0 after 100 settled picks. ROI needs ~1,000+ picks; report it, do not act on it early.
  ADDENDUM Q004 (2026-10-28 15:30Z, written and deployed BEFORE any of these fields were collected; the rules above
  are unchanged -- this only ADDS fields and a separate paper pick type; helpers in scripts/kalshi_tight.py):
    (a) LISTING  = per Kalshi event of every configured series (ML, spread, total; NFL ML is tracked for timing/depth
                   only): Kalshi open_time (min over the event's markets), our first sighting, and the FIRST TWO-SIDED
                   anchor quote (anchor = two-sided market whose YES mid is nearest 50c: at-the-money strike for
                   ladders, the closer team market for ML).
    (b) TIGHT    = first tick at which the anchor's YES spread <= 4c: time, ticker/strike/team, bid/ask, and displayed
                   depth at the best price from the orderbook endpoint (contracts and $ for YES ask, NO ask, YES bid).
                   Censored (no pick_tight, excluded from timing stats) when seen tight/two-sided at our FIRST sighting
                   of an event whose open_time is > 35 min older. 15-min polling -> times are upper bounds (~15 min).
    (c) DEPTH    = at the main rule's decision tick, for a pick with a matching Kalshi contract (kal_ or kal_alt_):
                   contracts and $ on our side at the best ask and within 1c of it (kal_depth_* on the pick row).
    (d) pick_tight = separate paper pick per game x logger market (type pick_tight / settle_tight in the picks file;
                   state in data/open_lines_<league>_tight.json). Decided at the first tick >= the event's uncensored
                   first tight quote at which (i) the game is in our Pinnacle state (for home/away), (ii) the model
                   returns the game, (iii) start is >= 30 min away, (iv) the anchor is CURRENTLY tight. SAME
                   thresholds as the main rule, with the Kalshi contract replacing the Pinnacle open:
      NHL ML     : gap = p_home_model - home-team market YES mid; |gap| >= 0.03 -> buy that team's market YES at its ask.
      NHL TOTAL  : S = anchor strike; gap = P_model(over S) - YES mid; |gap| >= 0.04 -> YES (over) / NO (under) at ask.
      NFL SPREAD : anchor "<fav> wins by over s" -> home line = -s if fav is home else +s; gap = model home margin +
                   home line; |gap| >= 2.0 -> home if gap > 0 else away; YES if backing the fav else NO, at the ask.
      NFL TOTAL  : S = anchor strike; gap = model total - S; |gap| >= 2.5 -> over (YES) / under (NO) at the ask.
                   Price = ask + exact taker fee (100-contract order, rounded up to the cent). Settled as the binary
                   contract (no pushes; ML includes OT/SO). CLV vs the last pre-start Pinnacle line, signed for the bet:
                   NFL points (close line - S, or home line - close home line); NHL probability (Pinnacle no-vig close
                   P(side) - Kalshi entry mid; totals via implied mu at strike S). Main-rule picks are unaffected.
    JUDGE (Q004) = after >= 100 settled pick_tight (pooled across the NBA/NHL/NFL loggers in QUESTIONS.md Q004, also
                   read per sport): mean CLV > 0 AND ROI at ask+fee > 0 -> write a sealed ask; otherwise no ask.
Files (per league): data/open_lines_<league>_state.json, data/open_lines_<league>_picks.jsonl,
                    data/open_lines_<league>_tight.json (Q004).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import live
from sports.live import KALSHI, _get, am_to_dec, devig, kalshi_fee
import kalshi_tight as KT

TIGHT_SLACK = pd.Timedelta(minutes=35)

MIN_LEAD = pd.Timedelta(minutes=30)
ET = "America/New_York"
NHL_ML_X, NHL_TOT_X = 0.03, 0.04
NFL_SPR_X, NFL_TOT_X = 2.0, 2.5
NFL_MODEL_TTL = pd.Timedelta(hours=3)
NFL_REFRESH_H = 12

NHL_TEAMS = {  # canonical = NHL API abbrev; first alias = nickname used for substring matching
    "ANA": ["ducks", "anaheim"], "BOS": ["bruins", "boston"], "BUF": ["sabres", "buffalo"],
    "CGY": ["flames", "calgary"], "CAR": ["hurricanes", "carolina"], "CHI": ["blackhawks", "chicago"],
    "COL": ["avalanche", "colorado"], "CBJ": ["blue jackets", "columbus", "clb"], "DAL": ["stars", "dallas"],
    "DET": ["red wings", "detroit"], "EDM": ["oilers", "edmonton"], "FLA": ["panthers", "florida"],
    "LAK": ["kings", "la", "los angeles"], "MIN": ["wild", "minnesota"], "MTL": ["canadiens", "montreal", "mon"],
    "NSH": ["predators", "nashville"], "NJD": ["devils", "nj", "new jersey"], "NYI": ["islanders", "ny islanders"],
    "NYR": ["rangers", "ny rangers"], "OTT": ["senators", "ottawa"], "PHI": ["flyers", "philadelphia"],
    "PIT": ["penguins", "pittsburgh"], "SJS": ["sharks", "sj", "san jose"], "SEA": ["kraken", "seattle"],
    "STL": ["blues", "st louis", "st. louis"], "TBL": ["lightning", "tb", "tampa bay"],
    "TOR": ["maple leafs", "toronto"], "UTA": ["mammoth", "utah", "utah hockey club", "utah mammoth"],
    "VAN": ["canucks", "vancouver"], "VGK": ["golden knights", "vgs", "veg", "vegas"],
    "WSH": ["capitals", "was", "washington"], "WPG": ["jets", "winnipeg"],
}
NFL_TEAMS = {  # canonical = nflverse abbrev
    "ARI": ["cardinals", "arizona"], "ATL": ["falcons", "atlanta"], "BAL": ["ravens", "baltimore"],
    "BUF": ["bills", "buffalo"], "CAR": ["panthers", "carolina"], "CHI": ["bears", "chicago"],
    "CIN": ["bengals", "cincinnati"], "CLE": ["browns", "cleveland"], "DAL": ["cowboys", "dallas"],
    "DEN": ["broncos", "denver"], "DET": ["lions", "detroit"], "GB": ["packers", "gnb", "green bay"],
    "HOU": ["texans", "houston"], "IND": ["colts", "indianapolis"], "JAX": ["jaguars", "jac", "jacksonville"],
    "KC": ["chiefs", "kan", "kansas city"], "LA": ["rams", "lar", "la rams"], "LAC": ["chargers", "la chargers"],
    "LV": ["raiders", "lvr", "oak", "las vegas"], "MIA": ["dolphins", "miami"], "MIN": ["vikings", "minnesota"],
    "NE": ["patriots", "nwe", "new england"], "NO": ["saints", "nor", "new orleans"], "NYG": ["giants", "ny giants"],
    "NYJ": ["jets", "ny jets"], "PHI": ["eagles", "philadelphia"], "PIT": ["steelers", "pittsburgh"],
    "SEA": ["seahawks", "seattle"], "SF": ["49ers", "sfo", "san francisco"], "TB": ["buccaneers", "tam", "tampa bay"],
    "TEN": ["titans", "tennessee"], "WAS": ["commanders", "wsh", "washington"],
}
CFG = {
    "nhl": {"teams": NHL_TEAMS, "markets": ("ml", "total"),
            "kalshi": {"ml": "KXNHLGAME", "total": "KXNHLTOTAL", "spread": "KXNHLSPREAD"},
            "espn": "hockey/nhl"},
    "nfl": {"teams": NFL_TEAMS, "markets": ("spread", "total"),
            "kalshi": {"ml": "KXNFLGAME", "total": "KXNFLTOTAL", "spread": "KXNFLSPREAD"},
            "espn": "football/nfl"},
}


def paths(lg: str) -> tuple[Path, Path]:
    return ROOT / "data" / f"open_lines_{lg}_state.json", ROOT / "data" / f"open_lines_{lg}_picks.jsonl"


def canon(lg: str, name: str, code_only: bool = False) -> str | None:
    teams = CFG[lg]["teams"]
    s = re.sub(r"\s+", " ", str(name).lower().strip())
    if s.upper() in teams:
        return s.upper()
    for k, al in teams.items():
        if s in al:
            return k
    if code_only:
        return None
    for k, al in sorted(teams.items(), key=lambda kv: -len(kv[1][0])):
        if al[0] in s:
            return k
    return None


def now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


def key(date_et: str, home: str, away: str) -> str:
    return f"{date_et}|{away}@{home}"


def novig_first(p1_am: float, p2_am: float) -> float:
    return devig(1 / am_to_dec(p1_am), 1 / am_to_dec(p2_am))[0]


# ---------------------------------------------------------------- sources

def pin_lines(lg: str) -> list[dict]:
    """One dict per game: start, home, away, and {ml, spread, total} snapshots (price1 = home / over)."""
    d = live.pinnacle(lg)
    if d.empty:
        return []
    d = d[(d.market.isin(["moneyline", "spread", "total"])) & (~d.live)]
    out = []
    for pid, g in d.groupby("pin_id"):
        r0 = g.iloc[0]
        h, a = canon(lg, r0.home), canon(lg, r0.away)
        if not h or not a:
            print(f"unmatched pinnacle teams: {r0.home} / {r0.away}")
            continue
        start = pd.Timestamp(r0.start)
        rec = {"key": key(start.tz_convert(ET).strftime("%Y-%m-%d"), h, a), "start": start.isoformat(),
               "home": h, "away": a, "mk": {}}
        for mk, name in (("ml", "moneyline"), ("spread", "spread"), ("total", "total")):
            x = g[g.market == name]
            if x.empty:
                continue
            r = x.iloc[x.hold.argsort()].iloc[0]
            rec["mk"][mk] = {"line": None if pd.isna(r.line) else float(r.line), "p1": float(r.price1),
                             "p2": float(r.price2), "fair1": round(float(r.fair1), 5)}
        out.append(rec)
    return out


def _event_teams(lg: str, letters: str) -> tuple[str, str] | None:
    for i in range(2, len(letters) - 1):
        a, b = canon(lg, letters[:i], True), canon(lg, letters[i:], True)
        if a and b and a != b:
            return a, b
    return None


def kalshi_markets(lg: str) -> dict:
    """(date_et, frozenset(teams)) -> {mk: [ {team, strike, yes_bid, yes_ask, no_bid, no_ask, ticker} ]}"""
    out = {}
    for mk, series in CFG[lg]["kalshi"].items():
        cursor = None
        while True:
            p = {"series_ticker": series, "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = _get(f"{KALSHI}/markets", params=p)
            for m in d.get("markets", []):
                tail = m["event_ticker"].split("-")[1]
                try:
                    date = datetime.strptime(tail[:7], "%y%b%d").strftime("%Y-%m-%d")
                except ValueError:
                    continue
                tm = _event_teams(lg, tail[7:])
                if tm is None:
                    continue
                suf = re.sub(r"\d+$", "", m["ticker"].split("-")[-1])
                f = lambda k: float(m[k]) if m.get(k) not in (None, "") else np.nan
                out.setdefault((date, frozenset(tm)), {}).setdefault(mk, []).append({
                    "ticker": m["ticker"], "event": m["event_ticker"], "team": canon(lg, suf, True) if suf else None,
                    "strike": None if m.get("floor_strike") is None else float(m["floor_strike"]),
                    "yes_bid": f("yes_bid_dollars"), "yes_ask": f("yes_ask_dollars"),
                    "no_bid": f("no_bid_dollars"), "no_ask": f("no_ask_dollars"), "open_time": m.get("open_time")})
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    return out


def _strike(rows: list, strike: float, team: str | None = None) -> dict | None:
    c = [r for r in rows if r["strike"] is not None and abs(r["strike"] - strike) < 1e-9
         and (team is None or r["team"] == team)]
    return c[0] if c else None


def _q(r: dict, yes: bool, pre: str) -> dict:
    bid, ask = (r["yes_bid"], r["yes_ask"]) if yes else (r["no_bid"], r["no_ask"])
    q = {f"{pre}ticker": r["ticker"], f"{pre}side": "yes" if yes else "no"}
    if np.isfinite(ask):
        q.update({f"{pre}ask": ask, f"{pre}fee": round(kalshi_fee(ask), 4)})
    if np.isfinite(bid) and bid > 0:
        q[f"{pre}bid"] = bid
    return q


def kalshi_quote(km: dict, mk: str, side: str, g: dict, line: float | None) -> dict:
    """Matching Kalshi contract for a pick -> kal_{ticker,side,bid,ask,fee}. When the open line is an integer (Kalshi
    lists only .5 strikes) the half-point-WORSE strike is recorded instead as kal_alt_* (+ kal_alt_strike, and
    kal_alt_fav for spreads); it is graded on its own terms in `report`, never mixed into the main metric."""
    rows = km.get(mk, [])
    if mk == "ml":
        team = g["home"] if side == "home" else g["away"]
        c = [r for r in rows if r["team"] == team]
        return _q(c[0], True, "kal_") if c else {}
    if line is None:
        return {}
    integer = abs(line - round(line)) < 1e-9
    if mk == "total":
        yes = side == "over"
        r = _strike(rows, line)
        if r:
            return _q(r, yes, "kal_")
        if integer:
            s = line + 0.5 if yes else line - 0.5
            r = _strike(rows, s)
            return {**_q(r, yes, "kal_alt_"), "kal_alt_strike": s} if r else {}
        return {}
    if line == 0:  # spread pick'em: no fav contract
        return {}
    fav = g["home"] if line < 0 else g["away"]
    yes = (side == "home") == (fav == g["home"])   # backing the favourite = YES "fav wins by over s"
    r = _strike(rows, abs(line), fav)
    if r:
        return _q(r, yes, "kal_")
    if integer:
        s = abs(line) + 0.5 if yes else abs(line) - 0.5
        r = _strike(rows, s, fav) if s > 0 else None
        return {**_q(r, yes, "kal_alt_"), "kal_alt_strike": s, "kal_alt_fav": fav} if r else {}
    return {}


# ---------------------------------------------------------------- models

def nhl_models(dates: list[str]) -> dict:
    from nhl_simple_live import predict
    out = {}
    for d in dates:
        try:
            for r in predict(d):
                if "lam_home" in r:  # key by the ET date of the start (a post-midnight-ET start keys to the next day)
                    de = pd.Timestamp(r["start_utc"]).tz_convert(ET).strftime("%Y-%m-%d")
                    out[key(de, r["home"], r["away"])] = r
        except Exception as e:  # noqa: BLE001 -- logged, never fatal
            print(f"nhl model unavailable for {d}: {type(e).__name__}: {e}")
    return out


def nfl_models(keys: list[str], lg_state: dict) -> dict:
    sched_p = ROOT / "data" / "nfl_schedule.csv"
    pbp = ROOT / "data" / "pbp" / "play_by_play_2026.parquet"
    stale = (not pbp.exists()) or (datetime.now().timestamp() - pbp.stat().st_mtime > NFL_REFRESH_H * 3600)
    s = pd.read_csv(sched_p, usecols=["season", "week", "gameday", "home_team", "away_team"], low_memory=False)
    s = s[s.season == 2026]
    wk = {key(r.gameday, r.home_team, r.away_team): int(r.week) for r in s.itertuples()}
    cache = lg_state.setdefault("_model_cache", {})
    out, t = {}, now()
    for w in sorted({wk[k] for k in keys if k in wk}):
        c = cache.get(str(w))
        if c is None or t - pd.Timestamp(c["t"]) > NFL_MODEL_TTL:
            cmd = [sys.executable, str(ROOT / "scripts" / "nfl_simple_model.py"), "week", "--json", "--week", str(w)]
            if stale:
                cmd.append("--refresh")
                stale = False
            err = ""
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
                err = p.stderr[-300:]
                rows = json.loads(p.stdout[p.stdout.index("["):])
            except Exception as e:  # noqa: BLE001
                print(f"nfl model unavailable for week {w}: {type(e).__name__}: {e} {err}")
                continue
            c = cache[str(w)] = {"t": t.isoformat(), "rows": rows}
        for r in c["rows"]:
            d = pd.Timestamp(r["kickoff_utc"]).tz_convert(ET).strftime("%Y-%m-%d")
            out[key(d, r["home"], r["away"])] = {**r, "model_t": c["t"], "week": w}
    missing = [k for k in keys if k not in wk]
    if missing:
        print(f"nfl: {len(missing)} games not in nflverse 2026 schedule (date/team mismatch?): {missing[:4]}")
    return out


# ---------------------------------------------------------------- NHL totals helpers (dev implied-mu)

def _implied_mu(line: float, p_over: float) -> float:
    from nhl_simple_live import p_over as pov
    lo, hi = 2.0, 11.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if pov({"lam_home": mid / 2, "lam_away": mid / 2}, line) < p_over:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------- decide

def decide(lg: str, mk: str, g: dict, m: dict, t: pd.Timestamp, km: dict) -> dict:
    o = g["pin"][mk]["open"]
    dec = {"t": t.isoformat(), "market": mk, "open_line": o["line"], "open_p1": o["p1"], "open_p2": o["p2"],
           "open_t": o["t"], "pick": None}
    if lg == "nhl":
        from nhl_simple_live import p_over
        dec.update(model_lam_home=m["lam_home"], model_lam_away=m["lam_away"], model_exp_total=m["exp_total"],
                   goalie_home=m.get("goalie_home"), goalie_away=m.get("goalie_away"), model_warnings=m.get("warnings"))
        if mk == "ml":
            q_home = novig_first(o["p1"], o["p2"])
            dec.update(model=m["p_home"], open_novig=round(q_home, 5), gap=round(m["p_home"] - q_home, 4))
            if abs(dec["gap"]) >= NHL_ML_X:
                dec["pick"] = "home" if dec["gap"] > 0 else "away"
        else:
            q_over = novig_first(o["p1"], o["p2"])
            pm = p_over(m, o["line"])
            dec.update(model=round(pm, 4), open_novig=round(q_over, 5), gap=round(pm - q_over, 4),
                       open_mu=round(_implied_mu(o["line"], q_over), 3))
            if abs(dec["gap"]) >= NHL_TOT_X:
                dec["pick"] = "over" if dec["gap"] > 0 else "under"
    else:
        dec.update(model_week=m.get("week"), model_t=m.get("model_t"), qb_home=m.get("qb_home"),
                   qb_away=m.get("qb_away"), model_warnings=m.get("warnings"), all_models=m.get("all_models"))
        q1 = novig_first(o["p1"], o["p2"])
        if mk == "spread":
            dec.update(model=m["model_spread_home"], open_novig=round(q1, 5),
                       gap=round(m["model_spread_home"] + o["line"], 2))
            if abs(dec["gap"]) >= NFL_SPR_X:
                dec["pick"] = "home" if dec["gap"] > 0 else "away"
        else:
            dec.update(model=m["model_total"], open_novig=round(q1, 5), gap=round(m["model_total"] - o["line"], 2))
            if abs(dec["gap"]) >= NFL_TOT_X:
                dec["pick"] = "over" if dec["gap"] > 0 else "under"
    if dec["pick"]:
        first = dec["pick"] in ("home", "over")
        dec["nv_prob_open"] = round(dec["open_novig"] if first else 1 - dec["open_novig"], 5)
        dec["price_open_am"] = o["p1"] if first else o["p2"]
        cur = g["pin"][mk]["last"]
        dec.update(pin_now_line=cur["line"], pin_now_am=cur["p1"] if first else cur["p2"])
        dec.update(kalshi_quote(km, mk, dec["pick"], g, o["line"] if mk != "ml" else None))
        pre = "kal_" if dec.get("kal_ticker") else "kal_alt_" if dec.get("kal_alt_ticker") else None
        if pre:  # Q004 (c): our side's depth at the decision tick
            dd = KT.buy_depth(KT.book(_get, KALSHI, dec[f"{pre}ticker"]), dec[f"{pre}side"] == "yes")
            dec.update({f"kal_depth_{x}": v for x, v in dd.items()})
    return dec


def _pin_q(pin: dict, side_first: bool) -> float:
    q = novig_first(pin["p1"], pin["p2"])
    return q if side_first else 1 - q


def decide_tight(lg: str, mk: str, g: dict, m: dict, rows: list, t: pd.Timestamp) -> dict | None:
    """Q004 (d) pick_tight at the current Kalshi quotes; None = not decidable this tick (anchor not tight now)."""
    a = KT.anchor(rows)
    if a is None or a["yes_ask"] - a["yes_bid"] > KT.TIGHT_MAX + 1e-9:
        return None
    dec = {"t": t.isoformat(), "market": mk, "home": g["home"], "anchor": a["ticker"], "anchor_bid": a["yes_bid"],
           "anchor_ask": a["yes_ask"], "pick": None}
    if mk == "ml":
        c = {r["team"]: r for r in rows if r["team"] in (g["home"], g["away"]) and KT.two_sided(r)}
        if g["home"] not in c:
            return None
        h = c[g["home"]]
        mid = (h["yes_bid"] + h["yes_ask"]) / 2
        dec.update(model=m["p_home"], kal_mid_home=mid, gap=round(m["p_home"] - mid, 4))
        if abs(dec["gap"]) >= NHL_ML_X:
            side = "home" if dec["gap"] > 0 else "away"
            r = c.get(g[side])
            if r is None:
                return dec
            dec.update(pick=side, ticker=r["ticker"], yes=True, ask=r["yes_ask"], entry_mid=(r["yes_bid"] + r["yes_ask"]) / 2)
    else:
        if a["strike"] is None:
            return None
        S, mid = a["strike"], (a["yes_bid"] + a["yes_ask"]) / 2
        dec.update(strike=S, kal_mid=mid)
        if mk == "total":
            if lg == "nhl":
                from nhl_simple_live import p_over
                pm = p_over(m, S)
                dec.update(model=round(pm, 4), gap=round(pm - mid, 4))
                hit = abs(dec["gap"]) >= NHL_TOT_X
            else:
                dec.update(model=m["model_total"], gap=round(m["model_total"] - S, 2))
                hit = abs(dec["gap"]) >= NFL_TOT_X
            if hit:
                side = "over" if dec["gap"] > 0 else "under"
                yes = side == "over"
                dec.update(pick=side, yes=yes)
        else:  # NFL spread
            if a["team"] not in (g["home"], g["away"]):
                return None
            hl = -S if a["team"] == g["home"] else S
            dec.update(fav=a["team"], home_line=hl, model=m["model_spread_home"],
                       gap=round(m["model_spread_home"] + hl, 2))
            if abs(dec["gap"]) >= NFL_SPR_X:
                side = "home" if dec["gap"] > 0 else "away"
                dec.update(pick=side, yes=(side == "home") == (a["team"] == g["home"]))
        if dec["pick"]:
            dec.update(ticker=a["ticker"], ask=a["yes_ask"] if dec["yes"] else a["no_ask"],
                       entry_mid=mid if dec["yes"] else 1 - mid)
    if dec["pick"]:
        if not np.isfinite(dec["ask"]):
            dec["pick"] = None
            return dec
        dec["fee"] = KT.fee_exact(dec["ask"])
        dd = KT.buy_depth(KT.book(_get, KALSHI, dec["ticker"]), dec["yes"])
        dec.update({f"depth_{x}": v for x, v in dd.items()})
    return dec


def settle_tight(lg: str, mk: str, p: dict, hs: int, as_: int, close: dict) -> dict:
    side = p["pick"]
    if mk == "ml":
        res = "win" if (hs > as_) == (side == "home") else "loss"
        qc = _pin_q(close, side == "home")
        clv = round(qc - p["entry_mid"], 5)
    elif mk == "total":
        yes_wins = hs + as_ > p["strike"]
        res = "win" if yes_wins == p["yes"] else "loss"
        if lg == "nhl":
            from nhl_simple_live import p_over
            mu_c = _implied_mu(close["line"], novig_first(close["p1"], close["p2"]))
            pc = p_over({"lam_home": mu_c / 2, "lam_away": mu_c / 2}, p["strike"])
            clv = round((pc if side == "over" else 1 - pc) - p["entry_mid"], 5)
        else:
            clv = (close["line"] - p["strike"]) * (1 if side == "over" else -1)
    else:
        margin = hs - as_ if p["fav"] == p["home"] else as_ - hs
        res = "win" if (margin > p["strike"]) == p["yes"] else "loss"
        clv = (p["home_line"] - close["line"]) * (1 if side == "home" else -1)
    return {"result": res, "home_score": hs, "away_score": as_, "roi": KT.roi(res, p["ask"], p["fee"]),
            "close_line": close.get("line"), "close_p1": close["p1"], "close_p2": close["p2"], "clv": clv}


# ---------------------------------------------------------------- settle

def finals(lg: str, date_et: str) -> dict:
    d = _get(f"https://site.api.espn.com/apis/site/v2/sports/{CFG[lg]['espn']}/scoreboard",
             params={"dates": date_et.replace("-", "")})
    out = {}
    for e in d.get("events", []):
        if not e["status"]["type"].get("completed"):
            continue
        c = {x["homeAway"]: x for x in e["competitions"][0]["competitors"]}
        h, a = canon(lg, c["home"]["team"]["displayName"]), canon(lg, c["away"]["team"]["displayName"])
        out[key(date_et, h, a)] = (int(c["home"]["score"]), int(c["away"]["score"]))
    return out


def grade(lg: str, mk: str, dec: dict, hs: int, as_: int, close: dict) -> dict:
    side, L = dec["pick"], dec["open_line"]
    if mk == "ml":
        res = "win" if (hs > as_) == (side == "home") else "loss"
    elif mk == "total":
        diff = (hs + as_ - L) * (1 if side == "over" else -1)
        res = "push" if diff == 0 else ("win" if diff > 0 else "loss")
    else:
        diff = (hs - as_ + L) * (1 if side == "home" else -1)
        res = "push" if diff == 0 else ("win" if diff > 0 else "loss")
    first = side in ("home", "over")
    out = {"result": res, "home_score": hs, "away_score": as_, "close_line": close["line"],
           "close_p1": close["p1"], "close_p2": close["p2"], "close_t": close["t"]}
    qc = novig_first(close["p1"], close["p2"])
    if lg == "nhl":
        if mk == "ml":
            out["clv"] = round((qc - dec["open_novig"]) * (1 if first else -1), 5)
        else:
            from nhl_simple_live import p_over
            mu_c = _implied_mu(close["line"], qc)
            pc = p_over({"lam_home": mu_c / 2, "lam_away": mu_c / 2}, L)
            out["clv"] = round((pc - dec["open_novig"]) * (1 if first else -1), 5)
            out["clv_goals"] = round((mu_c - dec["open_mu"]) * (1 if first else -1), 4)
    else:
        if mk == "spread":
            out["clv"] = (L - close["line"]) * (1 if side == "home" else -1)
        else:
            out["clv"] = (close["line"] - L) * (1 if side == "over" else -1)
    return out


# ---------------------------------------------------------------- tick

def load_state(lg: str) -> dict:
    p = paths(lg)[0]
    return json.loads(p.read_text()) if p.exists() else {}


def cmd_tick(lg: str) -> int:
    STATE, PICKS = paths(lg)
    t = now()
    st = load_state(lg)
    first_run = not st
    games = st.setdefault("games", {})
    errs = []
    try:
        pins = pin_lines(lg)
    except Exception as e:  # noqa: BLE001
        pins, _ = [], errs.append(f"pinnacle {type(e).__name__}")
    try:
        kal = kalshi_markets(lg)
    except Exception as e:  # noqa: BLE001
        kal, _ = {}, errs.append(f"kalshi {type(e).__name__}")

    if first_run and not pins:
        print(f"tick {t:%F %H:%M}Z {lg} FIRST RUN with no Pinnacle games ({errs}); state not written")
        return 1
    if first_run:
        st["first_run_at"] = t.isoformat()
    for p in pins:
        g = games.setdefault(p["key"], {"home": p["home"], "away": p["away"], "start": p["start"], "pin": {},
                                        "decision": {}, "first_seen": t.isoformat()})
        if first_run:
            g["pre_existing"] = True
        g["start"] = p["start"]
        if pd.Timestamp(p["start"]) <= t:
            continue
        for mk, s in p["mk"].items():
            snap = {"t": t.isoformat(), **s}
            m = g["pin"].setdefault(mk, {})
            m.setdefault("open", snap)
            m["last"] = snap
    for (date, tm), mks in kal.items():
        for k, g in games.items():
            if k.startswith(date) and {g["home"], g["away"]} == set(tm):
                for mk in mks:
                    g.setdefault("kal_first_seen", {}).setdefault(mk, t.isoformat())
                g["_kal"] = mks
    TP = ROOT / "data" / f"open_lines_{lg}_tight.json"
    T = json.loads(TP.read_text()) if TP.exists() else {"markets": {}, "picks": {}}
    gkey = lambda date, tm: next((k for k, g in games.items()  # noqa: E731
                                  if k.startswith(date) and {g["home"], g["away"]} == set(tm)), None)
    for (date, tm), mks in kal.items():  # Q004 (a)/(b): listing + first tight per event, independent of Pinnacle
        for mk, rows in mks.items():
            ev = rows[0]["event"]
            T["markets"][ev] = {**KT.track(T["markets"].get(ev), rows, t, TIGHT_SLACK, _get, KALSHI),
                                "mk": mk, "date": date, "teams": sorted(tm)}
    tight_due = []
    for ev, r in T["markets"].items():
        if r["mk"] not in CFG[lg]["markets"] or not r.get("first_tight") or r["first_tight"]["censored"]:
            continue
        k = gkey(r["date"], r["teams"])
        if k is not None and f"{k}|{r['mk']}" not in T["picks"]:
            tight_due.append((k, r["mk"], ev))

    # decisions
    todo = []
    for k, g in games.items():
        for mk in CFG[lg]["markets"]:
            if mk not in g["pin"] or mk in g["decision"]:
                continue
            if g.get("pre_existing"):
                g["decision"][mk] = {"t": t.isoformat(), "pick": None, "why": "pre-existing at first logger run"}
            elif pd.Timestamp(g["start"]) - t < MIN_LEAD:
                g["decision"][mk] = {"t": t.isoformat(), "pick": None, "why": "first seen < 30 min before start"}
            else:
                todo.append((k, mk))
    models = {}
    ks = sorted({k for k, _ in todo} | {k for k, _, _ in tight_due if pd.Timestamp(games[k]["start"]) - t >= MIN_LEAD})
    if ks:
        models = nhl_models(sorted({k.split("|")[0] for k in ks})) if lg == "nhl" else nfl_models(ks, st)
    new = 0
    for k, mk in todo:
        g, m = games[k], models.get(k)
        if m is None:
            continue
        try:
            dec = decide(lg, mk, g, m, t, g.get("_kal", {}))
        except Exception as e:  # noqa: BLE001
            errs.append(f"decide {k} {mk} {type(e).__name__}: {e}")
            continue
        g["decision"][mk] = dec
        if dec["pick"]:
            with PICKS.open("a") as f:
                f.write(json.dumps({"type": "pick", "league": lg, "key": k, "start": g["start"], **dec},
                                   default=str) + "\n")
            new += 1
    for k, mk, ev in tight_due:  # Q004 (d): pick_tight
        g, m, pk = games[k], models.get(k), f"{k}|{mk}"
        if pd.Timestamp(g["start"]) - t < MIN_LEAD:
            T["picks"][pk] = {"t": t.isoformat(), "market": mk, "event": ev, "pick": None,
                              "why": "not decidable >= 30 min before start"}
            continue
        rows = g.get("_kal", {}).get(mk, [])
        if m is None or not rows:
            continue
        try:
            dec = decide_tight(lg, mk, g, m, rows, t)
        except Exception as e:  # noqa: BLE001
            errs.append(f"decide_tight {k} {mk} {type(e).__name__}: {e}")
            continue
        if dec is None:
            continue
        dec.update(event=ev, first_tight_t=T["markets"][ev]["first_tight"]["t"])
        T["picks"][pk] = dec
        if dec["pick"]:
            with PICKS.open("a") as f:
                f.write(json.dumps({"type": "pick_tight", "league": lg, "key": k, "start": g["start"], **dec},
                                   default=str) + "\n")
            new += 1
    for g in games.values():
        g.pop("_kal", None)

    # settle (3.5 h NHL / 4 h NFL after start)
    wait = pd.Timedelta(hours=3.5 if lg == "nhl" else 4)
    pend = [(k, mk) for k, g in games.items() for mk, d in g["decision"].items()
            if d.get("pick") and "result" not in d and t - pd.Timestamp(g["start"]) > wait]
    tpend = [(pk.rsplit("|", 1)[0], p["market"], pk) for pk, p in T["picks"].items() if p.get("pick")
             and "result" not in p and pk.rsplit("|", 1)[0] in games
             and t - pd.Timestamp(games[pk.rsplit("|", 1)[0]]["start"]) > wait]
    fin = {}
    for d in sorted({k.split("|")[0] for k, _ in pend} | {k.split("|")[0] for k, _, _ in tpend}):
        try:
            fin.update(finals(lg, d))
        except Exception as e:  # noqa: BLE001
            errs.append(f"espn {type(e).__name__}")
    for k, mk in pend:
        if k not in fin:
            continue
        g = games[k]
        dec = g["decision"][mk]
        r = grade(lg, mk, dec, *fin[k], g["pin"][mk]["last"])
        dec.update(result=r["result"])
        with PICKS.open("a") as f:
            f.write(json.dumps({"type": "settle", "league": lg, "key": k, "market": mk, **r}, default=str) + "\n")
    for k, mk, pk in tpend:
        if k not in fin or mk not in games[k]["pin"]:
            continue
        p = T["picks"][pk]
        r = settle_tight(lg, mk, p, *fin[k], games[k]["pin"][mk]["last"])
        p.update(r)
        with PICKS.open("a") as f:
            f.write(json.dumps({"type": "settle_tight", "league": lg, "key": k, "market": mk, **r}, default=str) + "\n")
    TP.write_text(json.dumps(T, indent=1, default=str))
    tight_open = {pk.rsplit("|", 1)[0] for pk, p in T["picks"].items() if p.get("pick") and "result" not in p}
    # drop finished non-pick games after 3 days to keep the file small
    for k in [k for k, g in games.items() if t - pd.Timestamp(g["start"]) > pd.Timedelta(days=3) and k not in tight_open
              and not any(d.get("pick") and "result" not in d for d in g["decision"].values())]:
        st.setdefault("archived", {})[k] = {mk: d.get("gap") for mk, d in games[k]["decision"].items()}
        del games[k]
    STATE.write_text(json.dumps(st, indent=1, default=str))
    msg = (f"tick {t:%F %H:%M}Z {lg} games {len(games)} pins {len(pins)} kalshi_events {len(kal)} "
           f"decisions_due {len(todo)} new_picks {new}{' FIRST RUN (all pre-existing)' if first_run else ''}")
    if errs:
        msg += " ERR " + ",".join(errs)
    if new or errs or first_run:
        print(msg)
    return 0


# ---------------------------------------------------------------- report

def _rows(lg: str) -> pd.DataFrame:
    PICKS = paths(lg)[1]
    if not PICKS.exists():
        return pd.DataFrame()
    ev = [json.loads(l) for l in PICKS.read_text().splitlines() if l.strip()]
    p = pd.DataFrame([e for e in ev if e["type"] == "pick"])
    if p.empty:
        return p
    p = p.drop_duplicates(["key", "market"])
    s = pd.DataFrame([e for e in ev if e["type"] == "settle"])
    return p.merge(s.drop(columns=["type", "league"]), on=["key", "market"], how="left", suffixes=("", "_s")) \
        if not s.empty else p


def _roi(res: pd.Series, win_pay: np.ndarray) -> np.ndarray:
    return np.where(res == "win", win_pay, np.where(res == "loss", -1.0, 0.0))


def _alt_result(r) -> str:
    """Result of the half-point-worse Kalshi contract (no pushes at .5 strikes)."""
    home = r.key.split("@")[1]
    if r.market == "total":
        yes_wins = r.home_score + r.away_score > r.kal_alt_strike
    else:
        margin = r.home_score - r.away_score if r.kal_alt_fav == home else r.away_score - r.home_score
        yes_wins = margin > r.kal_alt_strike
    return "win" if yes_wins == (r.kal_alt_side == "yes") else "loss"


def cmd_report(lg: str) -> int:
    st = load_state(lg)
    games = st.get("games", {})
    decs = [(mk, d) for g in games.values() for mk, d in g["decision"].items()]
    print(f"{lg.upper()}: games tracked {len(games)}; decided {len(decs)} "
          f"(pre-existing {sum(d.get('why', '').startswith('pre') for _, d in decs)}); "
          f"picks {sum(bool(d.get('pick')) for _, d in decs)}")
    for mk in CFG[lg]["markets"]:
        gaps = [d["gap"] for m, d in decs if m == mk and d.get("gap") is not None]
        gaps += [v[mk] for v in st.get("archived", {}).values() if v.get(mk) is not None]
        if gaps:
            u = "prob" if lg == "nhl" else "pts"
            print(f"  {mk:6s} model - open ({u}): mean {np.mean(gaps):+.3f} sd {np.std(gaps):.3f} n {len(gaps)} "
                  f"(a mean far from 0 = level problem, not edge)")
    r = _rows(lg)
    TP = ROOT / "data" / f"open_lines_{lg}_tight.json"
    T = json.loads(TP.read_text()) if TP.exists() else {"markets": {}, "picks": {}}
    dd = [p.get("depth_usd_1c") for p in T["picks"].values() if p.get("pick")]
    if not r.empty and "kal_depth_usd_1c" in r:
        dd += r.kal_depth_usd_1c.dropna().tolist()
    KT.report(T, [p for p in T["picks"].values() if "result" in p], dd, lg.upper())
    if r.empty:
        print("no picks yet")
        return 0
    s = r[r["result"].notna()] if "result" in r else r.iloc[0:0]
    for mk, x in [("ALL", s)] + [(mk, s[s.market == mk]) for mk in CFG[lg]["markets"]]:
        if x.empty:
            continue
        w, l_, pu = (x.result == "win").sum(), (x.result == "loss").sum(), (x.result == "push").sum()
        nv = _roi(x.result, 1 / x.nv_prob_open - 1)
        off = _roi(x.result, np.array([am_to_dec(a) - 1 for a in x.price_open_am]))
        line = (f"  {mk:6s} settled {len(x)}: {w}-{l_}-{pu}  ROI no-vig {nv.mean():+.1%}  offered {off.mean():+.1%}  "
                f"CLV {x.clv.mean():+.4f} (sd {x.clv.std():.4f})")
        if "kal_bid" in x and x.kal_bid.notna().any():
            k = x[x.kal_bid.notna()]
            line += f"  Kalshi maker@bid {len(k)}: {_roi(k.result, (1 - k.kal_bid) / k.kal_bid).mean():+.1%}"
        if "kal_ask" in x and x.kal_ask.notna().any():
            k = x[x.kal_ask.notna()]
            line += (f"  taker {len(k)}: "
                     f"{_roi(k.result, (1 - k.kal_ask - k.kal_fee) / (k.kal_ask + k.kal_fee)).mean():+.1%}")
        if "kal_alt_ticker" in x and x.kal_alt_ticker.notna().any():
            k = x[x.kal_alt_ticker.notna()].copy()
            k["alt_res"] = [_alt_result(r) for r in k.itertuples()]
            line += f"\n         half-point-worse Kalshi strike (integer open lines) {len(k)}:"
            if k.kal_alt_bid.notna().any():
                b = k[k.kal_alt_bid.notna()]
                line += f" maker@bid {_roi(b.alt_res, (1 - b.kal_alt_bid) / b.kal_alt_bid).mean():+.1%}"
            if k.kal_alt_ask.notna().any():
                b = k[k.kal_alt_ask.notna()]
                line += (f" taker "
                         f"{_roi(b.alt_res, (1 - b.kal_alt_ask - b.kal_alt_fee) / (b.kal_alt_ask + b.kal_alt_fee)).mean():+.1%}")
        print(line)
    if len(s) >= 100:
        print("  KILL RULE: mean CLV <= 0 after 100 picks -> KILL" if s.clv.mean() <= 0 else
              "  kill rule: CLV > 0 after 100+ picks -> keep running")
    else:
        print(f"  kill rule check at 100 settled picks ({len(s)} so far)")
    cols = [c for c in ["key", "market", "pick", "open_line", "model", "gap", "nv_prob_open", "kal_bid", "kal_ask",
                        "close_line", "result", "clv"] if c in r]
    print(r[cols].tail(25).to_string(index=False))
    return 0


def cmd_show(lg: str) -> int:
    rows = []
    for k, g in sorted(load_state(lg).get("games", {}).items()):
        row = {"game": k}
        for mk in ("ml", "spread", "total"):
            pm = g["pin"].get(mk, {})
            o, l_ = pm.get("open", {}), pm.get("last", {})
            row[f"{mk}_open"] = o.get("line") if mk != "ml" else o.get("p1")
            row[f"{mk}_last"] = l_.get("line") if mk != "ml" else l_.get("p1")
        for mk in CFG[lg]["markets"]:
            d = g["decision"].get(mk, {})
            row[f"{mk}_model"], row[f"{mk}_pick"] = d.get("model"), d.get("pick") or (d.get("why") or "")[:12]
        rows.append(row)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no games")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "report", "show"])
    ap.add_argument("--league", required=True, choices=["nhl", "nfl"])
    a = ap.parse_args()
    raise SystemExit({"tick": cmd_tick, "report": cmd_report, "show": cmd_show}[a.cmd](a.league))
