"""ESPN public APIs: games, multi-book game lines, player/team props, box scores.

No key. One source covers NFL, college football, NBA, WNBA, NHL and MLB, and
the props and box scores share ESPN athlete ids, so every prop can be graded
without a name-matching step.

What exists (probed 2026-10-10, one slate per league/season):

  game lines  2022-23: ~10-15 books (DraftKings, Caesars, MGM, PointsBet,
              Bet365, ...), current line + prices. 2024+: ESPN BET with
              open / close / current.
  props       2022-23: DraftKings, Caesars, MGM, PointsBet (NFL, NBA, CFB;
              not NHL/MLB). 2024+: ESPN BET, ~1,000+ per NFL/NBA slate, NHL.
              MLB props only from 2025.
  box scores  every league, every season (summary endpoint).

Prop records: a main line is a PAIR of records (one carries the over price,
one the under) at a .5 line, each with open and current values. Records with
a line but no price are alternate lines the API lists without odds: dropped.
Scorer markets (anytime TD etc.) have a price and no line: side = "yes".

``current`` is the last price ESPN stored, NOT verified to be the close.
Measured on NFL 2024 (57k prop sides): ``last_updated`` is a median 6.4 h
AFTER kickoff (when ESPN closed the market), and 5% are stamped weeks
later -- so the timestamp says nothing about when the price was set. Treat
``current`` as approximately-closing; `collect.py snapshot` records true
pre-game prices with their own timestamps going forward.
Sanity check on the same data: over 48.6% / under 51.1% / push 0.3%, ROI
-6.2% on both sides = the book's hold, as an efficient market should give.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .http import fetch

SITE = "https://site.api.espn.com/apis/site/v2/sports/{path}"
CORE = "https://sports.core.api.espn.com/v2/sports/{sport}/leagues/{league}"

LEAGUES = {
    "nfl": "football/nfl",
    "cfb": "football/college-football",
    "nba": "basketball/nba",
    "wnba": "basketball/wnba",
    "nhl": "hockey/nhl",
    "mlb": "baseball/mlb",
}
# ESPN season label Y -> calendar window to scan (generous; the event's own season field decides)
WINDOWS = {
    "nfl": lambda y: (date(y, 8, 1), date(y + 1, 2, 20)),
    "cfb": lambda y: (date(y, 8, 20), date(y + 1, 1, 25)),
    "nba": lambda y: (date(y - 1, 10, 1), date(y, 6, 30)),
    "nhl": lambda y: (date(y - 1, 10, 1), date(y, 6, 30)),
    "mlb": lambda y: (date(y, 3, 15), date(y, 11, 10)),
    "wnba": lambda y: (date(y, 5, 1), date(y, 10, 31)),
}
# finished-day responses never change; anything newer is re-fetched
SETTLED_AFTER = timedelta(days=2)


def _core(league: str) -> str:
    sport, lg = LEAGUES[league].split("/")
    return CORE.format(sport=sport, league=lg)


def _settled(day: date) -> bool:
    return day < datetime.now(timezone.utc).date() - SETTLED_AFTER


def _num(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("+", "")
    if s.upper() in ("EVEN", "EV", "PK", "PICK"):
        return 100.0 if s.upper().startswith("EV") else 0.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def _ref_id(ref: str | None, kind: str) -> str | None:
    m = re.search(rf"/{kind}/(\d+)", ref or "")
    return m.group(1) if m else None


# ---------------------------------------------------------------- games

def scoreboard(league: str, day: date) -> list[dict]:
    params = {"dates": day.strftime("%Y%m%d"), "limit": 1000}
    if league == "cfb":
        params["groups"] = 80  # all FBS games, not just the featured ones
    return fetch(SITE.format(path=LEAGUES[league]) + "/scoreboard", params,
                 cache=_settled(day)).get("events", [])


def parse_event(league: str, e: dict) -> dict:
    c = e["competitions"][0]
    side = {t["homeAway"]: t for t in c["competitors"]}
    v = c.get("venue", {}) or {}
    a = v.get("address", {}) or {}
    row = {"league": league, "event_id": e["id"], "season": e.get("season", {}).get("year"),
           "season_type": e.get("season", {}).get("slug"),
           "start_utc": pd.Timestamp(e["date"]), "status": e["status"]["type"]["name"],
           "neutral": bool(c.get("neutralSite", False)), "venue_id": v.get("id"), "venue": v.get("fullName"),
           "city": a.get("city"), "state": a.get("state"), "country": a.get("country"),
           "indoor": v.get("indoor")}
    for s in ("home", "away"):
        t = side.get(s, {})
        row[f"{s}"] = t.get("team", {}).get("abbreviation")
        row[f"{s}_team_id"] = t.get("team", {}).get("id")
        row[f"{s}_score"] = _num(t.get("score")) if row["status"] == "STATUS_FINAL" else np.nan
    return row


def games(league: str, seasons, progress=None) -> pd.DataFrame:
    rows, seen = [], set()
    for y in seasons:
        lo, hi = WINDOWS[league](y)
        hi = min(hi, datetime.now(timezone.utc).date() + timedelta(days=7))
        d = lo
        while d <= hi:
            for e in scoreboard(league, d):
                if e["id"] not in seen:
                    seen.add(e["id"])
                    rows.append(parse_event(league, e))
            d += timedelta(days=1)
            if progress:
                progress(d)
    g = pd.DataFrame(rows)
    return g[g["season"].isin(list(seasons))].sort_values("start_utc").reset_index(drop=True) if len(g) else g


# ---------------------------------------------------------------- game lines

def _odds_list(league: str, eid: str, cache: bool) -> list[dict]:
    try:
        return fetch(f"{_core(league)}/events/{eid}/competitions/{eid}/odds", {"limit": 100},
                     cache=cache).get("items", [])
    except Exception:
        return []


def parse_line(item: dict) -> dict:
    """One provider's game line. close/open where the book reports them, else current."""
    h, a = item.get("homeTeamOdds", {}) or {}, item.get("awayTeamOdds", {}) or {}
    row = {"provider_id": item["provider"]["id"], "provider": item["provider"]["name"],
           "spread": _num(item.get("spread")), "total": _num(item.get("overUnder")),
           "over_odds": _num(item.get("overOdds")), "under_odds": _num(item.get("underOdds")),
           "home_ml": _num(h.get("moneyLine")), "away_ml": _num(a.get("moneyLine")),
           "home_spread_odds": _num(h.get("spreadOdds")), "away_spread_odds": _num(a.get("spreadOdds"))}
    for when in ("open", "close"):
        tot = item.get(when, {}) or {}
        row[f"total_{when}"] = _num(tot.get("total", {}).get("american"))
        row[f"over_odds_{when}"] = _num(tot.get("over", {}).get("american"))
        row[f"under_odds_{when}"] = _num(tot.get("under", {}).get("american"))
        for s, t in (("home", h), ("away", a)):
            w = t.get(when, {}) or {}
            row[f"{s}_spread_{when}"] = _num(w.get("pointSpread", {}).get("american"))
            row[f"{s}_spread_odds_{when}"] = _num(w.get("spread", {}).get("american"))
            row[f"{s}_ml_{when}"] = _num(w.get("moneyLine", {}).get("american"))
    return row


def lines(league: str, eid: str, cache: bool = True) -> list[dict]:
    out = []
    for it in _odds_list(league, eid, cache):
        if "provider" not in it or "Live" in it["provider"]["name"]:
            continue  # in-play feeds are not pre-game prices
        if it.get("spread") is None and it.get("overUnder") is None and not it.get("homeTeamOdds"):
            continue  # model/consensus providers with no price
        out.append({"event_id": eid, **parse_line(it)})
    return out


# ---------------------------------------------------------------- props

def parse_prop(item: dict) -> list[dict]:
    """0, 1 or 2 rows (over and/or under) for one propBets record; unpriced alt lines -> []."""
    cur, opn = item.get("current", {}) or {}, item.get("open", {}) or {}
    rows = []
    for side in ("over", "under"):
        if side not in cur:
            continue
        line = cur.get("target", {}).get("value")
        rows.append({
            "provider_id": _ref_id(item.get("provider", {}).get("$ref"), "casinos"),
            "market": item["type"]["name"], "market_id": item["type"]["id"],
            "athlete_id": _ref_id(item.get("athlete", {}).get("$ref"), "athletes"),
            "team_id": _ref_id(item.get("team", {}).get("$ref"), "teams"),
            "side": side if line is not None else "yes",
            "line": _num(line), "line_open": _num(opn.get("target", {}).get("value")),
            "dec": _num(cur[side].get("decimal")), "dec_open": _num((opn.get(side) or {}).get("decimal")),
            "last_updated": pd.Timestamp(item["lastUpdated"]) if item.get("lastUpdated") else pd.NaT})
    return rows


def props(league: str, eid: str, cache: bool = True) -> list[dict]:
    out = []
    for it in _odds_list(league, eid, cache):
        pid = it.get("provider", {}).get("id")
        if not pid or "Live" in it["provider"]["name"]:
            continue
        page = 1
        while True:
            try:
                d = fetch(f"{_core(league)}/events/{eid}/competitions/{eid}/odds/{pid}/propBets",
                          {"limit": 1000, "page": page}, cache=cache)
            except Exception:
                break  # 404 = this book has no props for the game
            for r in (x for i in d.get("items", []) for x in parse_prop(i)):
                out.append({"event_id": eid, "provider": it["provider"]["name"], **r})
            if page >= d.get("pageCount", 1):
                break
            page += 1
    return out


# ---------------------------------------------------------------- box scores

def summary(league: str, eid: str, cache: bool = True) -> dict:
    return fetch(SITE.format(path=LEAGUES[league]) + "/summary", {"event": eid}, cache=cache)


def box(league: str, eid: str, cache: bool = True) -> list[dict]:
    """Long player stats: one row per (athlete, stat). 'a/b' values split into two stats."""
    try:
        s = summary(league, eid, cache)
    except Exception:
        return []
    out = []
    for team in s.get("boxscore", {}).get("players", []) or []:
        tid = team.get("team", {}).get("id")
        for grp in team.get("statistics", []):
            keys = grp.get("keys") or grp.get("labels") or []
            gname = grp.get("name") or grp.get("type")  # MLB: type = batting / pitching
            for ath in grp.get("athletes", []):
                a = ath.get("athlete", {})
                vals: dict[str, float] = {}
                for k, v in zip(keys, ath.get("stats", [])):
                    if k == "fullInnings.partInnings":  # 6.1 IP = 19 outs
                        ip = _num(v)
                        vals["outs"] = np.floor(ip) * 3 + round((ip % 1) * 10) if np.isfinite(ip) else np.nan
                        continue
                    sep = "/" if "/" in k else "-" if ("-" in k and "-" in str(v)[1:]) else None
                    if sep:  # compound "made/att": a separate key of the same name wins
                        for kk, vv in zip(k.split(sep), str(v).split(sep, 1)):
                            vals.setdefault(kk, _num(vv))
                    else:
                        vals[k] = _num(v)
                for kk, vv in vals.items():
                    out.append({"event_id": eid, "team_id": tid, "athlete_id": a.get("id"),
                                "name": a.get("displayName"), "group": gname, "stat": kk, "value": vv})
    return out
