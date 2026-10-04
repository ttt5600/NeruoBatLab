"""MLB from the official Stats API (statsapi.mlb.com, no key).

One hydrated schedule request per month returns, per game: venue coordinates,
elevation and roof type, weather at first pitch, the announced probable
pitchers, the umpiring crew and day/night. That is most of what a game-level
MLB model needs, in about eight requests a season.

The probable pitcher is the one genuinely pre-game starter field in this
package: it is what was announced, not who actually threw. Openers and late
scratches mean the two differ occasionally, and the announced one is what a
bettor knew.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd

from .http import fetch
from .schema import conform

SCHEDULE = "https://statsapi.mlb.com/api/v1/schedule"
HYDRATE = "team,venue(location,fieldInfo),weather,probablePitcher,officials,linescore"
_POST = {"F", "D", "L", "W"}          # wild card, division, LCS, World Series
_KEEP = {"R"} | _POST

_WIND = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)


def parse_wind(s) -> tuple[float | None, str | None]:
    """'12 mph, Out To CF' -> (12.0, 'Out To CF'); 'Calm'/'0 mph, None' -> (0.0, 'None')."""
    if not isinstance(s, str) or not s.strip():
        return None, None
    m = _WIND.match(s.strip())
    if not m:
        return (0.0, "Calm") if "calm" in s.lower() else (None, s.strip())
    return float(m.group(1)), (m.group(2).strip() or None)


def _month_windows(season: int):
    for m in range(3, 12):  # March (openers abroad) through November (World Series)
        yield date(season, m, 1), date(season, m + 1, 1) - timedelta(days=1)


def _row(g: dict) -> dict | None:
    if g.get("gameType") not in _KEEP:
        return None
    st = g.get("status", {})
    if st.get("abstractGameState") != "Final" or st.get("codedGameState") not in ("F", "O"):
        return None  # postponed, suspended, cancelled, in progress
    h, a = g["teams"]["home"], g["teams"]["away"]
    v = g.get("venue", {})
    loc, field = v.get("location", {}), v.get("fieldInfo", {})
    coords = loc.get("defaultCoordinates", {})
    w = g.get("weather", {}) or {}
    mph, wdir = parse_wind(w.get("wind"))
    ls = g.get("linescore", {}) or {}
    ump = next((o["official"]["fullName"] for o in g.get("officials", [])
                if o.get("officialType") == "Home Plate"), None)

    def pp(side, key):
        return (side.get("probablePitcher") or {}).get(key)

    return {
        "league": "MLB",
        "game_id": str(g["gamePk"]),
        "season": int(g["season"]),
        "game_type": "post" if g["gameType"] in _POST else "regular",
        "date": g["officialDate"],
        "start_utc": g["gameDate"],
        "home": h["team"].get("abbreviation", h["team"]["name"]),
        "away": a["team"].get("abbreviation", a["team"]["name"]),
        "neutral": False,
        "venue": v.get("name"),
        "venue_lat": coords.get("latitude"),
        "venue_lon": coords.get("longitude"),
        "elevation_ft": loc.get("elevation"),
        "roof": (field.get("roofType") or "").lower() or None,
        "surface": (field.get("turfType") or "").lower() or None,
        "day_night": g.get("dayNight"),
        "home_starter": pp(h, "fullName"),
        "away_starter": pp(a, "fullName"),
        "home_starter_id": pp(h, "id"),
        "away_starter_id": pp(a, "id"),
        "officials": ump,
        "temp_f": pd.to_numeric(w.get("temp"), errors="coerce"),
        "wind_mph": mph,
        "wind_dir": wdir,
        "condition": w.get("condition"),
        "home_score": h.get("score"),
        "away_score": a.get("score"),
        "overtime": (ls.get("currentInning") or 0) > (g.get("scheduledInnings") or 9),
    }


def from_schedule_json(payloads: list[dict]) -> pd.DataFrame:
    """Parse schedule responses -> common schema. Pure; tested offline."""
    rows = [r for p in payloads for d in p.get("dates", []) for g in d.get("games", [])
            if (r := _row(g)) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # A resumed suspended game can appear twice under one gamePk; keep the final.
    df = df.drop_duplicates("game_id", keep="last")
    return conform(df)


def games(seasons, cache: bool = True) -> pd.DataFrame:
    today = date.today()
    payloads = []
    for s in seasons:
        for lo, hi in _month_windows(s):
            if lo > today:
                break
            payloads.append(fetch(SCHEDULE, {
                "sportId": 1, "startDate": lo.isoformat(), "endDate": hi.isoformat(),
                "hydrate": HYDRATE}, cache=cache and hi < today))
    return from_schedule_json(payloads)
