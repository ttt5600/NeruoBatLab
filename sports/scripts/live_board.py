"""UNIFIED LIVE GAME BOARD (NBA / NFL / NHL): decision support + calibration recorder. NO ORDERS, ever.

    python scripts/live_board.py show      # one table: every live game, state, pregame line, model P(home), Kalshi, gap, note
    python scripts/live_board.py tick      # launchd every 60 s: append live states + prices + model to data/live_board.jsonl
    python scripts/live_board.py report    # calibration: Kalshi mid vs our model vs realised, by sport x period
    (deployed: cd ~/.claude/quant-kalshi && .venv/bin/python scripts/live_board.py show)

User 2026-10-27: "given how lines adjust and change within live events we should have some way to monitor it and put
in real time predictions based on what the models run."

Models (all frozen, dev data only):
  NBA  results/nba_midgame_fit.json  (L37 probit: deficit x time x pregame spread; scripts/nba_midgame_scan.py). Fit
       range el 600..2640 s; outside it the probit extrapolates (flagged). Tie: mean of both 'trailer' views.
  NFL  results/nfl_ingame_fit.json   (nflfastR vegas_wp-style logistic; scripts/live_board_models.py). Live state from
       ESPN situation (possession, down, distance, yard line, timeouts). No possession (kickoff/break) -> mean of
       'home ball own 25' and 'away ball own 25'. 2nd-half kickoff receiver unknown live -> 0.5.
  NHL  results/nhl_ingame_fit.json   (logistic: goal diff, time, pregame prob, man advantage, empty nets). Live state
       from NHL api-web /v1/score/now (situationCode -> PP / empty net).
  OT / shootout: no model (blank).
Pregame line: ESPN pregame spread (NBA/NFL) or de-vigged moneyline (NHL), cached while the game is `pre`; at first
  live sight ESPN summary pickcenter; fallback = last pregame Kalshi home mid (-> spread via the dev probit sigma).
Kalshi: KXNBAGAME / KXNFLGAME / KXNHLGAME YES bid/ask per team; matched by team-code set and ET date (+-1 day).
gap = model P(side) - ask(side) - 0.07 ask (1-ask) (taker fee, unrounded); the better side is shown.
READ THE NOTE COLUMN: sealed L28 and L37 (2026-10-26) showed in-game Kalshi is SHARPER than our NBA model (Kalshi Brier
  .1284 vs model .1319; model edges >= 5c lost -3.9c/contract). A big gap usually means the market knows something
  (injury, foul trouble, goalie pulled, QB out). This board is for watching and for building a calibration record --
  it is NOT a trade signal. No trial is registered here.
Files: data/live_board.jsonl (ticks), data/live_board_results.jsonl (finals), data/live_board_pregame.json (lines),
  data/live_board_open.json (games awaiting settlement).
"""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and sys.argv[1:2] == ["game"] and "--fresh" not in sys.argv:
    # fast path for `lb`: render data/live_board_latest.json with the stdlib-only view (no pandas/scipy import)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import live_board_view as _v
    _rc = _v.main(sys.argv[2:])
    if _rc is not None:
        raise SystemExit(_rc)
    sys.argv.append("--fresh")  # no cache / stale / no match -> live pull below

import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import live_board_models as lbm  # noqa: E402
import live_board_view as view  # noqa: E402
import nba_midgame_scan as nba  # noqa: E402

DATA = Path(os.environ.get("LIVE_BOARD_DATA", ROOT / "data"))
TICKS = DATA / "live_board.jsonl"
RESULTS = DATA / "live_board_results.jsonl"
PRE = DATA / "live_board_pregame.json"
OPEN = DATA / "live_board_open.json"
FITS = {"nba": ROOT / "results" / "nba_midgame_fit.json", "nfl": lbm.NFL_FIT, "nhl": lbm.NHL_FIT}
ET = ZoneInfo("America/New_York")
H = {"User-Agent": "Mozilla/5.0"}  # api-web returns 429 to the default python-requests UA
ESPN = {"nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
        "nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
        "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl"}
NHLAPI = "https://api-web.nhle.com/v1"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = {"nba": "KXNBAGAME", "nfl": "KXNFLGAME", "nhl": "KXNHLGAME"}
ALIAS = {"nba": nba.ESPN2K, "nfl": {"WSH": "WAS", "JAX": "JAC"},
         "nhl": {"NJD": "NJ", "SJS": "SJ", "TBL": "TB", "LAK": "LA", "UTAH": "UTA"}}
BIG_GAP = 0.08
WHY = {"nba": "injury / foul trouble / rest", "nfl": "injury (QB?) / weather / last play not in feed",
       "nhl": "goalie pulled or changed / injury / PP just ended"}


# ------------------------------------------------------------------------------------------------ utils
def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get(url: str, **params) -> dict:
    r = requests.get(url, params=params or None, headers=H, timeout=15)
    r.raise_for_status()
    return r.json()


def jl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for x in path.read_text().splitlines():
        try:
            out.append(json.loads(x))
        except json.JSONDecodeError:
            pass
    return out


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, default=lambda v: None if v != v else float(v)) + "\n")


def jload(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return {}


def jsave(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(path)


def fee(p: float) -> float:
    return 0.07 * p * (1 - p)


def am_prob(a) -> float | None:
    try:
        a = float(str(a).replace("+", ""))
    except (TypeError, ValueError):
        return None
    if a == 0 or a != a:
        return None
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def ok(x) -> bool:
    return x is not None and x == x


def kcode(sport: str, abbr: str) -> str:
    return ALIAS[sport].get(abbr, abbr)


# ------------------------------------------------------------------------------------------------ feeds
def espn_clock(st: dict) -> float:
    c = st.get("clock")
    if c is not None:
        return float(c)
    dc = st.get("displayClock", "0:00")
    mm, ss = dc.split(":") if ":" in dc else ("0", dc)
    return float(mm) * 60 + float(ss)


def espn_pre_odds(c: dict) -> dict:
    o = (c.get("odds") or [{}])[0]
    out = {"home_spread": o.get("spread"), "details": o.get("details"), "total": o.get("overUnder")}
    ml = o.get("moneyline") or {}
    hm = ((ml.get("home") or {}).get("close") or {}).get("odds") or (o.get("homeTeamOdds") or {}).get("moneyLine")
    am = ((ml.get("away") or {}).get("close") or {}).get("odds") or (o.get("awayTeamOdds") or {}).get("moneyLine")
    out.update(home_ml=hm, away_ml=am)
    return out


def espn_names(c: dict) -> dict:
    t = {x["homeAway"]: x["team"] for x in c["competitors"]}
    return {f"names_{s}": sorted({str(t[s][k]).lower() for k in ("abbreviation", "displayName", "shortDisplayName",
                                                                    "location", "name") if t[s].get(k)})
            for s in ("home", "away")}


def games_nba() -> list[dict]:
    out = []
    for e in get(f"{ESPN['nba']}/scoreboard").get("events", []):
        g = nba.game_state(e)
        c = e["competitions"][0]
        out.append({"sport": "nba", "gid": g["eid"], "eid": g["eid"], "game": f"{g['away']}@{g['home']}",
                    "state": g["state"], "start": g["start"], "home": g["home"], "away": g["away"],
                    "hs": g["hs"], "as": g["as"], "period": g["period"], "clock_s": g["clock_s"], "el": g["el"],
                    "winner_home": None if g["winner"] is None else g["winner"] == g["home"],
                    "completed": g["completed"], **espn_pre_odds(c), **espn_names(c)})
    return out


def _nfl_situation(c: dict, home: str, away: str, home_id: str) -> dict:
    s = c.get("situation") or {}
    out = {"pos": None, "down": None, "dist": None, "yl100": None,
           "home_to": s.get("homeTimeouts"), "away_to": s.get("awayTimeouts")}
    pid = s.get("possession")
    if pid is None or not s.get("down") or s.get("down", 0) < 1:
        return out
    pos = "home" if str(pid) == str(home_id) else "away"
    posabbr = home if pos == "home" else away
    out.update(pos=pos, down=int(s["down"]), dist=int(s.get("distance") or 10))
    txt = (s.get("possessionText") or "").strip()  # e.g. "KC 35" or "50"
    parts = txt.split()
    try:
        if len(parts) == 2:
            n = int(parts[1])
            out["yl100"] = 100 - n if kcode("nfl", parts[0]) == posabbr else n
        elif len(parts) == 1:
            out["yl100"] = int(parts[0])
    except ValueError:
        pass
    return out


def games_nfl() -> list[dict]:
    out = []
    for e in get(f"{ESPN['nfl']}/scoreboard").get("events", []):
        c = e["competitions"][0]
        st = c["status"]
        t = {x["homeAway"]: x for x in c["competitors"]}
        home, away = (kcode("nfl", t[s]["team"]["abbreviation"]) for s in ("home", "away"))
        per = int(st.get("period") or 0)
        clk = espn_clock(st)
        win = next((s for s in ("home", "away") if t[s].get("winner")), None)
        out.append({"sport": "nfl", "gid": e["id"], "eid": e["id"], "game": f"{away}@{home}",
                    "state": st["type"]["state"], "start": datetime.fromisoformat(e["date"].replace("Z", "+00:00")),
                    "home": home, "away": away, "hs": int(t["home"].get("score") or 0),
                    "as": int(t["away"].get("score") or 0), "period": per, "clock_s": clk,
                    "el": (per - 1) * 900 + (900 - clk) if 1 <= per <= 4 else None,
                    "winner_home": None if win is None else win == "home", "completed": bool(st["type"].get("completed")),
                    **espn_pre_odds(c), **_nfl_situation(c, home, away, t["home"]["team"]["id"]), **espn_names(c)})
    return out


def games_nhl() -> list[dict]:
    d = get(f"{NHLAPI}/score/now")
    try:  # ESPN board only for the pregame moneyline (keyed by Kalshi codes)
        esp = {}
        for e in get(f"{ESPN['nhl']}/scoreboard").get("events", []):
            c = e["competitions"][0]
            t = {x["homeAway"]: x for x in c["competitors"]}
            k = (kcode("nhl", t["away"]["team"]["abbreviation"]), kcode("nhl", t["home"]["team"]["abbreviation"]))
            esp[k] = {"eid": e["id"], **espn_pre_odds(c), **espn_names(c)}
    except Exception:  # noqa: BLE001
        esp = {}
    out = []
    for g in d.get("games", []):
        gs = g.get("gameState")
        state = "pre" if gs in ("FUT", "PRE") else "in" if gs in ("LIVE", "CRIT") else "post"
        home, away = kcode("nhl", g["homeTeam"]["abbrev"]), kcode("nhl", g["awayTeam"]["abbrev"])
        per = int(g.get("period") or 0)
        ck = g.get("clock") or {}
        clk = 0.0 if ck.get("inIntermission") else float(ck.get("secondsRemaining") or 0)
        sc = (g.get("situation") or {}).get("situationCode") or "1551"
        ag_, ask, hsk, hg_ = (int(x) for x in sc) if len(sc) == 4 and sc.isdigit() else (1, 5, 5, 1)
        hs, as_ = int(g["homeTeam"].get("score") or 0), int(g["awayTeam"].get("score") or 0)
        fin = gs in ("OFF", "FINAL")
        e = esp.get((away, home), {})
        out.append({"sport": "nhl", "gid": str(g["id"]), "eid": e.get("eid"), "game": f"{away}@{home}", "state": state,
                    "start": datetime.fromisoformat(g["startTimeUTC"].replace("Z", "+00:00")), "home": home,
                    "away": away, "hs": hs, "as": as_, "period": per, "clock_s": clk,
                    "intermission": bool(ck.get("inIntermission")),
                    "el": (per - 1) * 1200 + (1200 - clk) if 1 <= per <= 3 else None,
                    "pp": hsk - ask, "en_home": int(hg_ == 0), "en_away": int(ag_ == 0),
                    "winner_home": (hs > as_) if fin and hs != as_ else None, "completed": fin,
                    "home_spread": None, "home_ml": e.get("home_ml"), "away_ml": e.get("away_ml"),
                    "details": e.get("details"), "total": e.get("total"),
                    "names_home": sorted(set(e.get("names_home", [])) | _nhl_names(g["homeTeam"])),
                    "names_away": sorted(set(e.get("names_away", [])) | _nhl_names(g["awayTeam"]))})
    return out


def _nhl_names(t: dict) -> set:
    out = {str(t.get("abbrev", "")).lower()}
    for k in ("name", "commonName", "placeName"):
        v = t.get(k)
        v = v.get("default") if isinstance(v, dict) else v
        if v:
            out.add(str(v).lower())
    return out - {""}


FEEDS = {"nba": games_nba, "nfl": games_nfl, "nhl": games_nhl}


def all_games(msgs: list) -> list[dict]:
    out = []
    for sp, f in FEEDS.items():
        try:
            out += f()
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"{sp} feed error: {str(ex)[:150]}")
    return out


def kalshi_books(sports) -> dict:
    """sport -> list of {date, event, q: {code: {bid, ask}}}."""
    out = {}
    for sp in sports:
        evs: dict = {}
        cursor = None
        while True:
            p = {"series_ticker": SERIES[sp], "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = get(f"{KALSHI}/markets", **p)
            for m in d.get("markets", []):
                f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None
                ev = evs.setdefault(m["event_ticker"], {"event": m["event_ticker"], "q": {}})
                ev["q"][m["ticker"].split("-")[-1]] = {"bid": f("yes_bid_dollars"), "ask": f("yes_ask_dollars")}
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
        for ev in evs.values():
            try:
                ev["date"] = datetime.strptime(ev["event"].split("-")[1][:7], "%y%b%d").date()
            except (ValueError, IndexError):
                ev["date"] = None
        out[sp] = list(evs.values())
    return out


def match_kalshi(g: dict, books: dict) -> dict | None:
    d0 = g["start"].astimezone(ET).date()
    best = None
    for ev in books.get(g["sport"], []):
        if set(ev["q"]) != {g["home"], g["away"]} or ev["date"] is None:
            continue
        dd = abs((ev["date"] - d0).days)
        if dd <= 1 and (best is None or dd < best[0]):
            best = (dd, ev)
    return best[1] if best else None


# ------------------------------------------------------------------------------------------------ pregame
def update_pregame(g: dict, cache: dict, ev: dict | None, fits: dict) -> dict:
    key = f"{g['sport']}:{g['gid']}"
    c = cache.setdefault(key, {"game": g["game"]})
    sp = g["sport"]
    if g["state"] == "pre":  # pregame odds only ever read while the game is `pre`
        if sp in ("nba", "nfl") and g.get("home_spread") is not None:
            c.update(hm=-float(g["home_spread"]), src="espn_scoreboard", details=g.get("details"), at=now())
        if g.get("total") not in (None, ""):
            c.update(total=float(g["total"]), total_src="ESPN scoreboard (pre-tip)")
        if sp == "nhl":
            ph, pa = am_prob(g.get("home_ml")), am_prob(g.get("away_ml"))
            if ph and pa:
                c.update(p_pre=ph / (ph + pa), src="espn_scoreboard_ml", details=g.get("details"), at=now())
        if ev:
            hq = ev["q"].get(g["home"], {})
            if ok(hq.get("bid")) and ok(hq.get("ask")) and hq["bid"] > 0 and hq["ask"] - hq["bid"] <= 0.06:
                c.update(k_pre_mid_home=(hq["bid"] + hq["ask"]) / 2, k_pre_at=now())
    need = c.get("hm") is None if sp in ("nba", "nfl") else c.get("p_pre") is None
    if need and g["state"] == "in" and g.get("eid") and not c.get("summary_tried"):
        c["summary_tried"] = True
        try:
            pc = (get(f"{ESPN[sp]}/summary", event=g["eid"]).get("pickcenter") or [{}])[0]
            if sp in ("nba", "nfl") and pc.get("spread") is not None:
                c.update(hm=-float(pc["spread"]), src="espn_pickcenter", details=pc.get("details"), at=now())
            if sp == "nhl":
                ph = am_prob((pc.get("homeTeamOdds") or {}).get("moneyLine"))
                pa = am_prob((pc.get("awayTeamOdds") or {}).get("moneyLine"))
                if ph and pa:
                    c.update(p_pre=ph / (ph + pa), src="espn_pickcenter_ml", details=pc.get("details"), at=now())
        except Exception as ex:  # noqa: BLE001
            c["summary_err"] = str(ex)[:150]
    need = c.get("hm") is None if sp in ("nba", "nfl") else c.get("p_pre") is None
    if need and c.get("k_pre_mid_home"):
        p = min(max(c["k_pre_mid_home"], 0.02), 0.98)
        if sp == "nhl":
            c.update(p_pre=p, src="kalshi_pregame_mid")
        else:
            c.update(hm=fits[sp]["pregame_sigma"] * float(norm.ppf(p)), src="kalshi_pregame_mid")
    return c


# ------------------------------------------------------------------------------------------------ models
_HIST: dict = {}


def hist(sp: str) -> pd.DataFrame:
    if sp not in _HIST:
        path = {"nba": nba.STATES, "nfl": lbm.NFL_STATES, "nhl": lbm.NHL_HIST}[sp]
        _HIST[sp] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _HIST[sp]


def model_home(g: dict, pre: dict, fits: dict) -> tuple[float | None, int | None, float | None, str]:
    """-> (P(home), history n, history home win rate, flag)."""
    sp, el = g["sport"], g.get("el")
    if sp not in fits:
        return None, None, None, "model fit file missing"
    if el is None:
        return None, None, None, "OT/not in regulation: no model"
    mar = g["hs"] - g["as"]
    if sp == "nba":
        hm = pre.get("hm")
        if hm is None:
            return None, None, None, "no pregame line"
        f = fits["nba"]

        def ptr(d, s):
            return float(nba.p_trailer(f, [d], [s], [el])[0])
        if mar == 0:
            p = 0.5 * (ptr(0, hm) + 1 - ptr(0, -hm))  # home as 'trailer' by 0, away as 'trailer' by 0
            flag, n, ph = "", None, None
        else:
            d, s = abs(mar), (hm if mar < 0 else -hm)
            pt = ptr(d, s)
            p = pt if mar < 0 else 1 - pt
            pe, n, _ = nba.empirical(hist("nba"), d, s, el) if len(hist("nba")) else (float("nan"), 0, None)
            ph = None if pe != pe else (pe if mar < 0 else 1 - pe)
            flag = ""
        if el < 600 or el > 2640:
            flag = "model extrapolating (fit el 600-2640)"
        return p, n, ph, flag
    if sp == "nfl":
        hm = pre.get("hm")
        if hm is None:
            return None, None, None, "no pregame line"
        gsr = 3600 - el
        half = gsr - 1800 if g["period"] <= 2 else gsr
        hto, ato = (g.get("home_to") if ok(g.get("home_to")) else 3), (g.get("away_to") if ok(g.get("away_to")) else 3)

        def row(pos, yl, down, dist):
            sd = mar if pos == "home" else -mar
            return {"score_differential": sd, "game_seconds_remaining": gsr, "half_seconds_remaining": half,
                    "posteam_spread": hm if pos == "home" else -hm, "yardline_100": yl, "down": down, "ydstogo": dist,
                    "pos_to": hto if pos == "home" else ato, "def_to": ato if pos == "home" else hto,
                    "receive_2h_ko": 0.5, "home": int(pos == "home"), "qtr": g["period"]}
        flag = ""
        if g.get("pos") and ok(g.get("yl100")):
            pp = float(lbm.nfl_predict_pos(fits["nfl"], pd.DataFrame([row(g["pos"], g["yl100"], g["down"], g["dist"])]))[0])
            p = pp if g["pos"] == "home" else 1 - pp
        else:
            r = pd.DataFrame([row("home", 75, 1, 10), row("away", 75, 1, 10)])
            pp = lbm.nfl_predict_pos(fits["nfl"], r)
            p = float(0.5 * (pp[0] + 1 - pp[1]))
            flag = "no possession in feed: averaged"
        h = hist("nfl")
        n = ph = None
        if len(h):
            q = h[(h.gsr - gsr).abs().le(120) & (h.sd_home == mar) & (h.hm - hm).abs().le(3)].drop_duplicates("game_id")
            n = len(q)
            ph = float(q.y_home.mean()) if n else None
        return p, n, ph, flag
    if sp == "nhl":
        pp_ = pre.get("p_pre")
        if pp_ is None:
            return None, None, None, "no pregame line"
        x = pd.DataFrame([{"p_pre": pp_, "gd": mar, "el": el, "pp": g.get("pp", 0), "en_home": g.get("en_home", 0),
                           "en_away": g.get("en_away", 0)}])
        p = float(lbm.nhl_predict_home(fits["nhl"], x)[0])
        h = hist("nhl")
        n = ph = None
        if len(h) and "p_pre" in h:
            q = h[(h.el - el).abs().le(120) & (h.gd == mar) & (h.p_pre - pp_).abs().le(0.08)].drop_duplicates("gameId")
            n = len(q)
            ph = float(q.y.mean()) if n else None
        flag = "PP" if g.get("pp") else ""
        if g.get("en_home") or g.get("en_away"):
            flag = "empty net"
        return p, n, ph, flag
    return None, None, None, ""


def evaluate(g: dict, pre: dict, ev: dict | None, fits: dict) -> dict:
    p, n, ph, flag = model_home(g, pre, fits)
    q = (ev or {}).get("q", {})
    hq, aq = q.get(g["home"], {}), q.get(g["away"], {})
    hb, ha, ab, aa = hq.get("bid"), hq.get("ask"), aq.get("bid"), aq.get("ask")
    mid = (hb + ha) / 2 if ok(hb) and ok(ha) and ha > 0 else None
    r = {"ts": now(), "sport": g["sport"], "gid": g["gid"], "game": g["game"], "home": g["home"], "away": g["away"],
         "period": g["period"], "clock_s": g["clock_s"], "el": g.get("el"), "hs": g["hs"], "as": g["as"],
         "pre_hm": pre.get("hm"), "pre_p": pre.get("p_pre"), "pre_src": pre.get("src"),
         "p_home": None if p is None else round(p, 4), "hist_n": n, "hist_p": None if ph is None else round(ph, 4),
         "h_bid": hb, "h_ask": ha, "a_bid": ab, "a_ask": aa, "mid_home": None if mid is None else round(mid, 4),
         "kalshi_event": (ev or {}).get("event"), "flag": flag}
    for k in ("pos", "down", "dist", "yl100", "home_to", "away_to", "pp", "en_home", "en_away", "intermission"):
        if k in g:
            r[k] = g[k]
    gaps = {}
    if p is not None:
        for side, ps, a in (("home", p, ha), ("away", 1 - p, aa)):
            if ok(a) and 0.01 <= a <= 0.99:
                gaps[side] = round(ps - a - fee(a), 4)
    r["gap_home"], r["gap_away"] = gaps.get("home"), gaps.get("away")
    notes = []
    if flag:
        notes.append(flag)
    if p is not None and mid is not None and abs(p - mid) >= BIG_GAP:
        notes.append(f"BIG GAP {p - mid:+.2f}: market likely knows something -- check {WHY[g['sport']]}")
    elif gaps and max(gaps.values()) >= 0.05:
        notes.append("gap >= 5c: sealed L28/L37 say such gaps lost money; check news first")
    if ok(hb) and ok(ha) and ha - hb >= 0.06:
        notes.append(f"wide book {ha - hb:.2f}")
    if ev is None:
        notes.append("no Kalshi market matched")
    r["note"] = "; ".join(notes)
    return r


def load_fits() -> dict:
    out = {}
    for sp, p in FITS.items():
        if p.exists():
            out[sp] = json.loads(p.read_text())
    return out


def state_str(r: dict, g: dict) -> str:
    sp = g["sport"]
    m, s = int(g["clock_s"] // 60), int(g["clock_s"] % 60)
    pre = {"nba": "Q", "nfl": "Q", "nhl": "P"}[sp]
    per = g["period"]
    lim = {"nba": 4, "nfl": 4, "nhl": 3}[sp]
    ps = f"{pre}{per}" if per <= lim else f"OT{per - lim if per - lim > 1 else ''}"
    st = f"{g['away']} {g['as']}-{g['hs']} {g['home']} {ps} {m}:{s:02d}"
    if sp == "nfl" and g.get("pos"):
        st += f" {g['home'] if g['pos'] == 'home' else g['away']} ball {g['down']}&{g['dist']} @{g.get('yl100')}"
    if sp == "nhl":
        if g.get("intermission"):
            st += " INT"
        if g.get("pp"):
            st += f" {'home' if g['pp'] > 0 else 'away'} PP"
    return st


# ------------------------------------------------------------------------------------------------ per-game records
DEPLOY = Path.home() / ".claude" / "quant-kalshi"
LATEST = DATA / "live_board_latest.json"       # tick -> every game's record; `game` / `lb` read it
ALERT_STATE = DATA / "live_board_alerts.json"  # dedupe keys + positions cache
ALERT_LOG = DATA / "live_board_alerts.jsonl"
WATCH = DEPLOY / "data" / "watchlist.txt"      # team codes, one per line
OPEN_TOTALS = DEPLOY / "data" / "open_totals_state.json"  # L32 logger: Pinnacle / Kalshi pregame NBA totals
MANUAL = DEPLOY / "data" / "manual_bets.jsonl"
CORE = {"nba": "basketball/leagues/nba", "nfl": "football/leagues/nfl", "nhl": "hockey/leagues/nhl"}
SPORT_OF = {"KXNBA": "nba", "KXNFL": "nfl", "KXNHL": "nhl"}
MOVE_ALERT = 0.25
POS_MAX_AGE = 300


def game_key(g: dict) -> str:
    """Kalshi event middle segment, e.g. 26OCT27PORLAL (ET date + away + home)."""
    return f"{g['start'].astimezone(ET):%y%b%d}".upper() + g["away"] + g["home"]


def totals_info(g: dict, c: dict) -> dict:
    """Pregame total into the cache (ESPN core odds, first non-live book, once per game) + NBA Pinnacle/Kalshi
    last pre-tip totals from the L32 open-totals logger."""
    if c.get("total") is None and g.get("eid") and g["state"] != "pre" and not c.get("core_tried"):
        c["core_tried"] = True
        try:
            eid = g["eid"]
            items = get(f"https://sports.core.api.espn.com/v2/sports/{CORE[g['sport']]}/events/{eid}/competitions/"
                        f"{eid}/odds", limit=100).get("items", [])
            for it in items:
                name = (it.get("provider") or {}).get("name", "")
                if "Live" in name:
                    continue  # in-play feed, not a pregame price
                tot = ((it.get("close") or {}).get("total") or {}).get("american") or it.get("overUnder")
                if tot not in (None, ""):
                    c.update(total=float(tot), total_src=f"ESPN {name} close")
                    break
        except Exception as ex:  # noqa: BLE001
            c["core_err"] = str(ex)[:120]
    out = {}
    if g["sport"] == "nba":
        try:
            st = json.loads(OPEN_TOTALS.read_text())
            gk = game_key(g)
            for v in st.values():
                if ((v.get("kal_last") or v.get("kal_open") or {}).get("event") or "").endswith(gk):
                    out = {"pin_last": (v.get("pin_last") or v.get("pin_open") or {}).get("line"),
                           "kal_pre": (v.get("kal_last") or {}).get("implied")}
                    break
        except Exception:  # noqa: BLE001
            pass
    return out


def our_positions(max_age: float = 0) -> tuple[list[dict], str | None]:
    """Our manual_bets orders on NBA/NFL/NHL tickers (weather is a separate executor, ignored): filled + resting
    contracts per order. Read-only GET /portfolio/orders via the deployed signed client; cached max_age s."""
    st = jload(ALERT_STATE)
    pc = st.get("positions") or {}
    if max_age and pc.get("at") and (datetime.now(timezone.utc) - datetime.fromisoformat(pc["at"])).total_seconds() < max_age:
        return pc.get("rows", []), pc.get("err")
    jr = [r for r in jl(MANUAL) if r.get("status") == "placed" and r.get("ticker", "")[:5] in SPORT_OF]
    rows, err = [], None
    if jr:
        try:
            import kalshi_exec as K
            K.set_root(DEPLOY)
            e = K.env()
            cl = K.Kalshi(e, K.config(e)["env"])
            mine = {r["client_order_id"]: r for r in jr}
            for o in cl.orders():
                r = mine.get(o.get("client_order_id"))
                if not r:
                    continue
                f = lambda k: float(o.get(f"{k}_fp") or o.get(k) or 0)  # noqa: E731
                filled = f("fill_count")
                resting = f("remaining_count") if o.get("status") == "resting" else 0.0
                if filled or resting:
                    rows.append({"ticker": r["ticker"], "side": r["side"], "filled": filled, "resting": resting,
                                 "entry": round(r["max_cost"] / r["count"], 4) if r.get("taker") else r["price"]})
        except Exception as ex:  # noqa: BLE001
            err = str(ex)[:150]
            rows = pc.get("rows", [])
    st = jload(ALERT_STATE)
    st["positions"] = {"at": now(), "rows": rows, "err": err}
    jsave(ALERT_STATE, st)
    return rows, err


def value_positions(g: dict, rows: list[dict]) -> list[dict]:
    gk = game_key(g)
    out = []
    for p in rows:
        t = p["ticker"]
        if SPORT_OF.get(t[:5]) != g["sport"] or t.split("-")[1] != gk:
            continue
        q = dict(p)
        try:
            m = get(f"{KALSHI}/markets/{t}").get("market") or {}
        except Exception:  # noqa: BLE001
            m = {}
        team = t.split("-")[-1]
        q["label"] = f"{team} win" if "GAME-" in t else (m.get("yes_sub_title") or t.split("-", 2)[-1])
        res = m.get("result") or ""
        if res in ("yes", "no"):
            q.update(settled=True, now=1.0 if res == p["side"] else 0.0)
        else:
            f = lambda k: float(m[k]) if m.get(k) not in (None, "") else None  # noqa: E731
            yb, ya = f("yes_bid_dollars"), f("yes_ask_dollars")
            mid = (yb + ya) / 2 if yb is not None and ya and ya > 0 else None
            if mid is not None:
                q["now"] = round(mid if p["side"] == "yes" else 1 - mid, 4)
            if "GAME-" in t and g.get("completed") and g.get("winner_home") is not None:  # final, market not settled yet
                yes_won = team == (g["home"] if g["winner_home"] else g["away"])
                q.update(settled=True, now=1.0 if yes_won == (p["side"] == "yes") else 0.0, by_score=True)
        if ok(q.get("now")):
            q["pnl"] = round(q["filled"] * (q["now"] - q["entry"]), 2)
        out.append(q)
    return out


def ser_g(g: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in g.items()
            if k not in ("home_ml", "away_ml", "home_spread", "details", "total")}


def make_rec(g: dict, ev: dict | None, pre: dict, fits: dict, pos_rows: list, pos_err: str | None) -> dict:
    r = evaluate(g, pre, ev, fits)
    tot = totals_info(g, pre)
    return {"key": f"{g['sport']}:{g['gid']}", "ts": now(), "g": ser_g(g), "r": r,
            "pre": {k: pre.get(k) for k in ("hm", "p_pre", "src", "details", "total", "total_src")},
            "tot": tot, "pos": value_positions(g, pos_rows), "pos_err": pos_err}


def watchlist() -> set[str]:
    try:
        return {x.strip().upper() for x in WATCH.read_text().splitlines() if x.strip() and not x.startswith("#")}
    except Exception:  # noqa: BLE001
        return set()


def alerts(recs: dict, msgs: list) -> None:
    """macOS notifications (kalshi_exec.notify), deduped per game per event in data/live_board_alerts.json."""
    from kalshi_exec import notify
    st = jload(ALERT_STATE)
    sent = st.setdefault("sent", {})
    watch = watchlist()

    def fire(key: str, title: str, msg: str, block: list | None = None) -> None:
        if key in sent:
            return
        sent[key] = now()
        notify(title, msg)
        append(ALERT_LOG, {"ts": now(), "key": key, "title": title, "msg": msg})
        msgs.append(f"ALERT [{key}] {title}: {msg}" + ("\n  " + "\n  ".join(block) if block else ""))

    for k, rec in recs.items():
        g, r = rec["g"], rec["r"]
        if g["state"] == "pre":
            continue
        lines, summ = view.render(rec)
        # (a) our positions
        for q in rec.get("pos") or []:
            if not q.get("filled"):
                continue
            if q.get("settled") and g["state"] == "post":
                won = q["now"] >= 0.5
                fire(f"final:{q['ticker']}", f"{'WON' if won else 'LOST'} {q['label']} ({g['game']})",
                     f"{q['side'].upper()} x{q['filled']:g} @ {q['entry']:.2f}: P&L {q['pnl']:+.2f}"
                     f"{' (by final score; Kalshi not settled yet)' if q.get('by_score') else ''} | {lines[0]}")
            elif g["state"] == "in" and ok(q.get("now")) and abs(q["now"] - q["entry"]) >= MOVE_ALERT:
                d = "up" if q["now"] > q["entry"] else "down"
                fire(f"move:{q['ticker']}:{d}", f"Our {q['label']} {d} {abs(q['now'] - q['entry']) * 100:.0f}c",
                     f"entry {q['entry']:.2f} -> mid {q['now']:.2f} (P&L {q['pnl']:+.2f}) | {summ}")
        # (b) model vs market gap
        if g["state"] == "in" and ok(r.get("p_home")) and ok(r.get("mid_home")) and \
                abs(r["p_home"] - r["mid_home"]) >= BIG_GAP:
            fire(f"gap:{k}", f"Likely news: {g['game']}",
                 f"model {r['p_home']:.0%} vs Kalshi {r['mid_home']:.0%} home -- check {WHY[g['sport']]}; not value", lines)
        # (c) watchlist
        if watch & {g["home"], g["away"]}:
            if g["state"] == "in":
                fire(f"w:{k}:start", f"Underway: {g['game']}", summ, lines)
                if g["sport"] in ("nba", "nfl") and g["period"] == 2 and float(g["clock_s"]) == 0:
                    fire(f"w:{k}:half", f"Halftime: {g['game']}", summ, lines)
                if g["sport"] == "nhl" and g.get("intermission"):
                    fire(f"w:{k}:int{g['period']}", f"Intermission {g['period']}: {g['game']}", summ, lines)
            elif g["state"] == "post":
                fire(f"w:{k}:final", f"Final: {g['game']}", summ, lines)
    cut = datetime.now(timezone.utc) - timedelta(days=4)
    st = {**jload(ALERT_STATE), "sent": {k: v for k, v in sent.items() if datetime.fromisoformat(v) > cut}}
    jsave(ALERT_STATE, st)


def game_fresh(args: list[str]) -> int:
    """Live pull of just the queried game: one scoreboard + one Kalshi event + our orders (~1-2 s)."""
    q = " ".join(x for x in args if not x.startswith("--"))
    if not q:
        return view.main([]) or 0
    d = view.load_latest()
    hits = view.match(d.get("games") or {}, q)
    sports = [d["games"][hits[0][0]]["g"]["sport"]] if hits else list(FEEDS)
    msgs: list[str] = []
    games = []
    for sp in sports:
        try:
            games += FEEDS[sp]()
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"{sp} feed error: {str(ex)[:150]}")
    recs0 = {f"{g['sport']}:{g['gid']}": {"g": ser_g(g)} for g in games}
    hits = view.match(recs0, q)
    if not hits:
        print(f"no NBA/NFL/NHL game today matching {q!r}" + ("; " + "; ".join(msgs) if msgs else ""))
        return 1
    live = [h for h in hits if recs0[h[0]]["g"]["state"] == "in"]
    fits = load_fits()
    cache = jload(PRE)
    pos_rows, pos_err = our_positions(0)
    by = {f"{g['sport']}:{g['gid']}": g for g in games}
    for k, focus in (live or hits[:1]):
        g = by[k]
        try:
            et_ = f"{SERIES[g['sport']]}-{game_key(g)}"
            m = get(f"{KALSHI}/markets", event_ticker=et_).get("markets", [])
            fx = lambda x, k_: float(x[k_]) if x.get(k_) not in (None, "") else None  # noqa: E731
            ev = {"event": et_, "q": {x["ticker"].split("-")[-1]: {"bid": fx(x, "yes_bid_dollars"),
                                                                   "ask": fx(x, "yes_ask_dollars")} for x in m}} \
                if m else match_kalshi(g, kalshi_books([g["sport"]]))
        except Exception as ex:  # noqa: BLE001
            ev = None
            msgs.append(f"Kalshi error: {str(ex)[:120]}")
        pre = update_pregame(g, cache, ev, fits)
        rec = make_rec(g, ev, pre, fits, pos_rows, pos_err)
        lines, _ = view.render(rec, focus)
        print(f"[{g['sport'].upper()} {g['game']}] fresh pull")
        print("\n".join(lines))
    jsave(PRE, cache)
    for m_ in msgs:
        print(m_)
    return 0


# ------------------------------------------------------------------------------------------------ commands
def show() -> int:
    msgs: list[str] = []
    games = all_games(msgs)
    fits = load_fits()
    live_ = [g for g in games if g["state"] == "in"]
    nowu = datetime.now(timezone.utc)
    soon = [g for g in games if g["state"] == "pre" and g["start"] - nowu < timedelta(hours=3)]
    books = kalshi_books(sorted({g["sport"] for g in live_ + soon}))
    cache = jload(PRE)
    rows = []
    for g in live_:
        ev = match_kalshi(g, books)
        pre = update_pregame(g, cache, ev, fits)
        r = evaluate(g, pre, ev, fits)
        line = (f"{-pre['hm']:+.1f}" if pre.get("hm") is not None else "") if g["sport"] != "nhl" else \
            (f"P {pre['p_pre']:.2f}" if pre.get("p_pre") is not None else "")
        best = max((("H", r["gap_home"]), ("A", r["gap_away"])), key=lambda x: -9 if x[1] is None else x[1])
        rows.append({"sport": g["sport"].upper(), "state": state_str(r, g), "pre(home)": line,
                     "P_home": r["p_home"], "K_home b/a": f"{r['h_bid']}/{r['h_ask']}", "mid": r["mid_home"],
                     "gap": None if best[1] is None else f"{best[0]} {best[1]:+.3f}",
                     "hist": None if r["hist_n"] is None else f"{r['hist_n']}" + (
                         f" ({r['hist_p']:.2f})" if r["hist_p"] is not None else ""),
                     "note": r["note"][:90]})
    jsave(PRE, cache)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 95)
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        print("no live NBA / NFL / NHL games right now")
    if soon:
        print("\nstarting within 3 h: " + ", ".join(
            f"{g['sport'].upper()} {g['game']} {g['start'].astimezone(ET):%H:%M} ET" for g in sorted(soon, key=lambda x: x["start"])))
    for m in msgs:
        print(m)
    print("\nP_home = our frozen model; gap = model - ask - taker fee (better side, H/A); hist = dev-season neighbour "
          "games (home win rate). Sealed L28/L37: in-game Kalshi beat our NBA model -- big gaps = news, not edge. "
          "No orders.")
    return 0


def tick() -> int:
    t0 = time.time()
    msgs: list[str] = []
    games = all_games(msgs)
    live_ = [g for g in games if g["state"] == "in"]
    nowu = datetime.now(timezone.utc)
    soon = [g for g in games if g["state"] == "pre" and g["start"] - nowu < timedelta(minutes=75)]
    opn = jload(OPEN)
    fits = load_fits()
    cache = jload(PRE)
    for g in games:  # cache ESPN pregame lines (no Kalshi call)
        if g["state"] == "pre":
            update_pregame(g, cache, None, fits)
    books: dict = {}
    if live_ or soon:
        books = kalshi_books(sorted({g["sport"] for g in live_ + soon}))
        for g in soon:
            update_pregame(g, cache, match_kalshi(g, books), fits)
        for g in live_:
            ev = match_kalshi(g, books)
            pre = update_pregame(g, cache, ev, fits)
            try:
                r = evaluate(g, pre, ev, fits)
            except Exception as ex:  # noqa: BLE001
                msgs.append(f"evaluate failed {g['sport']} {g['game']}: {ex}")
                continue
            append(TICKS, r)
            opn[f"{g['sport']}:{g['gid']}"] = {"sport": g["sport"], "gid": g["gid"], "eid": g.get("eid"),
                                               "game": g["game"], "home": g["home"], "last": r["ts"]}
    # per-game records for `game` / `lb` (all of today's games; Kalshi only where a book was pulled)
    pos_rows, pos_err = our_positions(POS_MAX_AGE)
    recs = {}
    for g in games:
        key = f"{g['sport']}:{g['gid']}"
        try:
            ev = match_kalshi(g, books) if g["state"] != "post" else None
            pre = cache.get(key) or update_pregame(g, cache, ev, fits)
            recs[key] = make_rec(g, ev, pre, fits, pos_rows, pos_err)
        except Exception as ex:  # noqa: BLE001
            msgs.append(f"record failed {key} {g.get('game')}: {str(ex)[:150]}")
    jsave(LATEST, {"ts": now(), "games": recs, "tick_s": round(time.time() - t0, 2)})
    jsave(PRE, cache)
    try:
        alerts(recs, msgs)
    except Exception as ex:  # noqa: BLE001
        msgs.append(f"alerts failed: {str(ex)[:150]}")
    settle(games, opn, msgs)
    jsave(OPEN, opn)
    if time.time() - t0 > 15:
        msgs.append(f"slow tick {time.time() - t0:.1f}s")
    if msgs:
        print("\n".join(msgs))
    return 0


def settle(games: list[dict], opn: dict, msgs: list) -> None:
    by = {f"{g['sport']}:{g['gid']}": g for g in games}
    lookups = 0
    for key in list(opn):
        o = opn[key]
        g = by.get(key)
        if g is None:  # off today's board: look it up (a few per tick)
            if lookups >= 4:
                continue
            lookups += 1
            try:
                if o["sport"] == "nhl":
                    d = get(f"{NHLAPI}/gamecenter/{o['gid']}/landing")
                    hs, as_ = d["homeTeam"].get("score"), d["awayTeam"].get("score")
                    fin = d.get("gameState") in ("OFF", "FINAL")
                    g = {"completed": fin, "winner_home": (hs > as_) if fin and hs is not None and hs != as_ else None,
                         "hs": hs, "as": as_}
                else:
                    s = get(f"{ESPN[o['sport']]}/summary", event=o["eid"])
                    hc = s["header"]["competitions"][0]
                    t = {x["homeAway"]: x for x in hc["competitors"]}
                    win = next((k for k in ("home", "away") if t[k].get("winner")), None)
                    g = {"completed": bool(hc["status"]["type"].get("completed")),
                         "winner_home": None if win is None else win == "home",
                         "hs": int(t["home"].get("score") or 0), "as": int(t["away"].get("score") or 0)}
            except Exception as ex:  # noqa: BLE001
                msgs.append(f"settle lookup failed {key}: {str(ex)[:120]}")
                continue
        if not g.get("completed") or g.get("winner_home") is None:
            last = datetime.fromisoformat(o["last"])
            if datetime.now(timezone.utc) - last > timedelta(days=3):
                del opn[key]
                msgs.append(f"dropped unsettled {key} (3 days)")
            continue
        append(RESULTS, {"ts": now(), "sport": o["sport"], "gid": o["gid"], "game": o["game"],
                         "home_won": bool(g["winner_home"]), "final_home": g["hs"], "final_away": g["as"]})
        del opn[key]
        msgs.append(f"SETTLED {o['sport']} {o['game']} {g['as']}-{g['hs']} home_won {g['winner_home']}")


def report() -> int:
    t = pd.DataFrame(jl(TICKS))
    res = pd.DataFrame(jl(RESULTS))
    if t.empty:
        print("no ticks recorded yet (data/live_board.jsonl)")
        return 0
    print(f"ticks {len(t)} over {t.groupby(['sport', 'gid']).ngroups} games; settled games {len(res)}")
    if res.empty:
        return 0
    res = res.drop_duplicates(["sport", "gid"], keep="last")
    t = t.merge(res[["sport", "gid", "home_won"]], on=["sport", "gid"])
    t = t[t.mid_home.notna()].copy()
    t["y"] = t.home_won.astype(float)
    t["mid"] = t.mid_home.astype(float)
    t["pm"] = pd.to_numeric(t.p_home, errors="coerce")
    lim = {"nba": 4, "nfl": 4, "nhl": 3}
    t["bucket"] = [f"{'P' if s == 'nhl' else 'Q'}{p}" if p <= lim[s] else "OT" for s, p in zip(t.sport, t.period)]
    rows = []
    for (sp, b), g in t.groupby(["sport", "bucket"]):
        both = g[g.pm.notna()]
        rows.append({"sport": sp, "period": b, "games": g.gid.nunique(), "ticks": len(g),
                     "brier_kalshi": round(float(((g.mid - g.y) ** 2).mean()), 4),
                     "brier_kalshi_same": round(float(((both.mid - both.y) ** 2).mean()), 4) if len(both) else None,
                     "brier_model_same": round(float(((both.pm - both.y) ** 2).mean()), 4) if len(both) else None,
                     "mean|model-mid|": round(float((both.pm - both.mid).abs().mean()), 4) if len(both) else None})
    print("\nBrier by sport x period (all ticks; 'same' = ticks where both exist). Ticks are serially correlated: "
          "trust games, not ticks.")
    print(pd.DataFrame(rows).to_string(index=False))
    for sp, g in t.groupby("sport"):
        g = g.assign(bin=pd.cut(g.mid, [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1], include_lowest=True))
        out = []
        for b, x in g.groupby("bin", observed=True):
            x1 = x.sort_values("ts").drop_duplicates("gid")  # one tick per game per bin (first)
            lo, hi = lbm.wilson(int(x1.y.sum()), len(x1))
            out.append({"home_mid": str(b), "games": len(x1), "mean_mid": round(x1.mid.mean(), 3),
                        "mean_model": round(x1.pm.mean(), 3) if x1.pm.notna().any() else None,
                        "realised": round(x1.y.mean(), 3), "ci95": f"{lo:.3f}-{hi:.3f}"})
        print(f"\n{sp.upper()} calibration by Kalshi home mid (one tick per game per bin):")
        print(pd.DataFrame(out).to_string(index=False))
        big = g[(g.pm - g.mid).abs() >= BIG_GAP].sort_values("ts").drop_duplicates("gid")
        if len(big):
            mb = float(((big.mid - big.y) ** 2).mean())
            pb = float(((big.pm - big.y) ** 2).mean())
            print(f"  big-gap games (|model-mid| >= {BIG_GAP}, first tick): n {len(big)}  Brier Kalshi {mb:.4f} "
                  f"vs model {pb:.4f}  -> {'Kalshi' if mb < pb else 'model'} closer")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "game":  # reached only with --fresh (or when the cache could not answer)
        raise SystemExit(game_fresh(sys.argv[2:]))
    raise SystemExit({"show": show, "tick": tick, "report": report}[cmd]())
