"""NHL from the league's public web API (api-web.nhle.com, no key).

Schedules come a week per request (~35 a season). The starting goalie -- the
single most important lineup variable in hockey -- needs the per-game
boxscore, about 1,400 requests a season, so it is a separate, optional pull.

The boxscore's ``starter`` flag records who actually started. Starting goalies
are normally confirmed at the morning skate, so for a closing-line bet this is
close to what was knowable; for a bet placed the night before it is not.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .http import fetch, fetch_many
from .schema import conform

BASE = "https://api-web.nhle.com/v1"
_TYPES = {2: "regular", 3: "post"}
_FINAL = {"OFF", "FINAL"}


def _row(g: dict) -> dict | None:
    if g.get("gameType") not in _TYPES or g.get("gameState") not in _FINAL:
        return None
    h, a = g["homeTeam"], g["awayTeam"]
    last = (g.get("gameOutcome") or {}).get("lastPeriodType")
    return {
        "league": "NHL",
        "game_id": str(g["id"]),
        "season": int(g["season"]) // 10000,
        "game_type": _TYPES[g["gameType"]],
        "start_utc": g["startTimeUTC"],
        "home": h["abbrev"],
        "away": a["abbrev"],
        "neutral": bool(g.get("neutralSite", False)),
        "venue": (g.get("venue") or {}).get("default"),
        "roof": "indoor",
        "home_score": h.get("score"),
        "away_score": a.get("score"),
        "overtime": last in ("OT", "SO") if last else pd.NA,
    }


def from_schedule_json(payloads: list[dict]) -> pd.DataFrame:
    """Parse weekly schedule responses -> common schema. Pure; tested offline."""
    rows = []
    for p in payloads:
        for day in p.get("gameWeek", []):
            for g in day.get("games", []):
                r = _row(g)
                if r is not None:
                    r["date"] = day["date"]  # the league's own (local) game date
                    rows.append(r)
    df = pd.DataFrame(rows)
    return conform(df.drop_duplicates("game_id")) if not df.empty else df


def games(seasons, cache: bool = True, goalies: bool = False) -> pd.DataFrame:
    today = date.today()
    payloads = []
    for s in seasons:
        d, end = date(s, 9, 15), min(date(s + 1, 6, 30), today)
        while d <= end:
            p = fetch(f"{BASE}/schedule/{d.isoformat()}",
                      cache=cache and d + timedelta(days=7) < today)
            payloads.append(p)
            nxt = p.get("nextStartDate")
            if not nxt or date.fromisoformat(nxt) <= d:
                break
            d = date.fromisoformat(nxt)
    df = from_schedule_json(payloads)
    if goalies and not df.empty:
        df = attach_goalies(df, cache=cache)
    return df


def parse_starting_goalies(box: dict) -> dict:
    """Boxscore -> {home_starter, home_starter_id, away_...}. Pure; tested offline."""
    out = {}
    stats = box.get("playerByGameStats", {})
    for side, key in (("home", "homeTeam"), ("away", "awayTeam")):
        gs = stats.get(key, {}).get("goalies", [])
        st = next((g for g in gs if g.get("starter")), None)
        out[f"{side}_starter"] = (st or {}).get("name", {}).get("default")
        out[f"{side}_starter_id"] = (st or {}).get("playerId")
    return out


def attach_goalies(df: pd.DataFrame, cache: bool = True) -> pd.DataFrame:
    boxes = fetch_many([f"{BASE}/gamecenter/{gid}/boxscore" for gid in df["game_id"]],
                       cache=cache)
    rows = [parse_starting_goalies(b) for b in boxes]
    g = pd.DataFrame(rows, index=df.index)
    out = df.copy()
    for c in g.columns:
        out[c] = g[c]
    return out
