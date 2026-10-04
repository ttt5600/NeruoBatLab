"""NBA from ESPN's public scoreboard API.

stats.nba.com is the richer source, but it fingerprints clients and refused
every request from this machine, with or without browser headers. ESPN's site
API answers plainly. It does not accept date ranges for the NBA, so a season
is one request per game day (~200), each cached forever once the day is over.

Season labels follow the convention used across this package: the year the
season STARTS (2024 = 2024-25), matching the NHL loader. ESPN itself labels by
the year it ends.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .http import fetch, fetch_many
from .schema import conform

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
_TYPES = {2: "regular", 3: "post", 5: "post"}  # 5 = play-in


def _row(e: dict, day: str) -> dict | None:
    stype = (e.get("season") or {}).get("type")
    if stype not in _TYPES:
        return None
    c = e["competitions"][0]
    # All-Star events are filed as regular season with invented teams; ESPN
    # marks them by competition type, which is the only reliable tell.
    if (c.get("type") or {}).get("abbreviation") == "ALLSTAR":
        return None
    st = c.get("status") or e.get("status") or {}
    if not (st.get("type") or {}).get("completed"):
        return None
    side = {t["homeAway"]: t for t in c["competitors"]}
    v = c.get("venue") or {}
    return {
        "league": "NBA",
        "game_id": str(e["id"]),
        "season": int(e["season"]["year"]) - 1,
        "game_type": _TYPES[stype],
        "date": day,
        "start_utc": e["date"],
        "home": side["home"]["team"]["abbreviation"],
        "away": side["away"]["team"]["abbreviation"],
        "neutral": bool(c.get("neutralSite", False)),
        "venue": v.get("fullName"),
        "roof": "indoor",
        "home_score": side["home"].get("score"),
        "away_score": side["away"].get("score"),
        "overtime": (st.get("period") or 4) > 4,
        "attendance": c.get("attendance") or pd.NA,
    }


def from_scoreboard_json(days: list[tuple[str, dict]]) -> pd.DataFrame:
    """[(YYYY-MM-DD, scoreboard json)] -> common schema. Pure; tested offline."""
    rows = [r for day, p in days for e in p.get("events", []) if (r := _row(e, day))]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return conform(df.drop_duplicates("game_id"))


def games(seasons, cache: bool = True) -> pd.DataFrame:
    today = date.today()
    settled = today - timedelta(days=1)
    dates = []
    for s in seasons:
        # Windows tile the calendar (Oct 1 - Sep 30). Seasons do not respect
        # a June end: the 2020 bubble ran to October, the 2021 Finals into
        # July. Each game's season label comes from ESPN, not the window.
        d, end = date(s, 10, 1), min(date(s + 1, 9, 30), today)
        while d <= end:
            dates.append(d)
            d += timedelta(days=1)
    url = lambda d: f"{SCOREBOARD}?dates={d.strftime('%Y%m%d')}"
    old = [d for d in dates if d < settled]
    new = [d for d in dates if d >= settled]
    pays = fetch_many([url(d) for d in old], cache=cache) + [fetch(url(d), cache=False) for d in new]
    return from_scoreboard_json([(d.isoformat(), p) for d, p in zip(old + new, pays)])


def box_score(game_id: str, cache: bool = True) -> pd.DataFrame:
    """Per-player box score for one game (POST-game; lag before use as a feature)."""
    return parse_box(fetch(SUMMARY, {"event": game_id}, cache=cache), game_id)


def box_scores(game_ids: list[str], cache: bool = True) -> pd.DataFrame:
    urls = [f"{SUMMARY}?event={g}" for g in game_ids]
    return pd.concat([parse_box(j, g) for j, g in zip(fetch_many(urls, cache=cache), game_ids)],
                     ignore_index=True)


def parse_box(j: dict, game_id: str) -> pd.DataFrame:
    rows = []
    for team in (j.get("boxscore") or {}).get("players", []):
        abbr = team["team"]["abbreviation"]
        for block in team.get("statistics", []):
            labels = block.get("labels", [])
            for a in block.get("athletes", []):
                rec = {"game_id": game_id, "team": abbr,
                       "player_id": a["athlete"]["id"],
                       "player": a["athlete"]["displayName"],
                       "starter": bool(a.get("starter")),
                       "dnp": bool(a.get("didNotPlay"))}
                rec.update(dict(zip(labels, a.get("stats", []))))
                rows.append(rec)
    return pd.DataFrame(rows)
