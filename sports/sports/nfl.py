"""NFL from nflverse: games since 1999 and weekly player stats.

nflverse is the best free source in any of the four leagues. Its games table
already carries stadium, roof, surface, weather, coaches, referee, starting
QBs *and* closing spread / total / moneylines, so the NFL is the one league
where a market benchmark is available before any odds work is done.

Conventions inherited from nflverse and kept as-is:
  * ``spread_line`` is points the HOME team is favoured by (positive = home fav).
  * ``gametime`` is US Eastern; it is converted to UTC here.
  * ``temp``/``wind`` are NA for domes and closed roofs, and also for some
    outdoor games -- see ``coverage`` in the pull script before trusting them.
"""
from __future__ import annotations

import io

import pandas as pd

from .http import fetch
from .schema import conform

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PLAYER_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
              "stats_player/stats_player_week_{season}.parquet")

_POST = {"WC", "DIV", "CON", "SB"}


def raw_games(cache: bool = True) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(fetch(GAMES_URL, cache=cache, binary=True)))


def from_raw(g: pd.DataFrame) -> pd.DataFrame:
    """nflverse games table -> common schema. Pure; tested offline."""
    local = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"),
                           format="%Y-%m-%d %H:%M")
    start = local.dt.tz_localize("America/New_York", ambiguous="NaT",
                                 nonexistent="shift_forward").dt.tz_convert("UTC")
    out = pd.DataFrame({
        "league": "NFL",
        "game_id": g["game_id"],
        "season": g["season"],
        "game_type": g["game_type"].map(lambda t: "post" if t in _POST else "regular"),
        "date": g["gameday"],
        "start_utc": start,
        "home": g["home_team"],
        "away": g["away_team"],
        "neutral": g["location"].eq("Neutral"),
        "venue": g["stadium"],
        "roof": g["roof"],
        "surface": g["surface"].replace("", pd.NA),
        "day_night": pd.NA,
        "home_coach": g["home_coach"],
        "away_coach": g["away_coach"],
        "home_starter": g["home_qb_name"],
        "away_starter": g["away_qb_name"],
        "home_starter_id": g["home_qb_id"],
        "away_starter_id": g["away_qb_id"],
        "officials": g["referee"],
        "temp_f": g["temp"],
        "wind_mph": g["wind"],
        "home_ml": g["home_moneyline"],
        "away_ml": g["away_moneyline"],
        "spread_line": g["spread_line"],
        "total_line": g["total_line"],
        "home_score": g["home_score"],
        "away_score": g["away_score"],
        "overtime": g["overtime"].map({1: True, 0: False}),
    })
    return conform(out)


def games(seasons=None, cache: bool = True) -> pd.DataFrame:
    df = from_raw(raw_games(cache=cache))
    return df if seasons is None else df[df["season"].isin(list(seasons))].reset_index(drop=True)


def player_week(seasons, cache: bool = True) -> pd.DataFrame:
    """Per-player per-week box stats (150 columns: passing, rushing, receiving,
    defence, kicking, EPA). Every row carries ``game_id``, so it joins to
    :func:`games` directly. These are POST-game numbers: use them only as
    lagged history, never as features of the game they describe."""
    frames = []
    for s in seasons:
        b = fetch(PLAYER_URL.format(season=s), cache=cache, binary=True)
        frames.append(pd.read_parquet(io.BytesIO(b)))
    return pd.concat(frames, ignore_index=True)
