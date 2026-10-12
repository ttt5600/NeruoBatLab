"""nflverse play-by-play joined to defensive charting, cached per season.

    plays(2024)  -> every play with game script (score diff, win prob, time),
                    field position, play type, EPA, kicks, and -- where charted --
                    the DEFENSE it faced (coverage, man/zone, box count, rushers,
                    blitzers) and the offense's look (formation, personnel,
                    play action, motion, screen).

Sources (nflverse-data GitHub releases, free):
  play_by_play_YYYY        1999-   every play
  pbp_participation_YYYY   2016-2025  coverage / man-zone / box / rushers / formation / personnel
  ftn_charting_YYYY        2022-   blitzers, play action, motion, screen, rpo (FTN Data)

Charting columns are NaN for seasons/plays the provider did not chart; nothing
is imputed.
"""
from __future__ import annotations

import io

import pandas as pd

from .http import fetch

REL = "https://github.com/nflverse/nflverse-data/releases/download/{tag}/{name}_{y}.parquet"

PBP_COLS = [
    "game_id", "play_id", "season", "week", "season_type", "game_date", "home_team", "away_team",
    "posteam", "defteam", "qtr", "game_seconds_remaining", "half_seconds_remaining", "down", "ydstogo",
    "yardline_100", "drive", "fixed_drive", "fixed_drive_result", "score_differential", "posteam_score",
    "defteam_score", "wp", "vegas_wp", "spread_line", "total_line", "roof", "surface", "temp", "wind",
    "play_type", "pass", "rush", "shotgun", "no_huddle", "qb_dropback", "air_yards", "yards_gained",
    "epa", "success", "touchdown", "pass_touchdown", "rush_touchdown", "interception", "sack",
    "field_goal_attempt", "field_goal_result", "kick_distance", "kicker_player_id", "kicker_player_name",
    "extra_point_result", "fourth_down_converted", "fourth_down_failed", "penalty",
]
PART_COLS = ["offense_formation", "offense_personnel", "defense_personnel", "defenders_in_box",
             "number_of_pass_rushers", "defense_man_zone_type", "defense_coverage_type",
             "was_pressure", "time_to_throw"]
FTN_COLS = ["n_blitzers", "n_pass_rushers", "n_offense_backfield", "is_play_action", "is_motion",
            "is_screen_pass", "is_rpo", "is_no_huddle", "is_qb_out_of_pocket", "is_trick_play"]


def _parquet(tag: str, name: str, y: int, cols=None) -> pd.DataFrame:
    try:
        b = fetch(REL.format(tag=tag, name=name, y=y), binary=True, retries=3)
    except Exception:
        return pd.DataFrame()
    d = pd.read_parquet(io.BytesIO(b))
    return d[[c for c in cols if c in d.columns]] if cols else d


def plays(season: int) -> pd.DataFrame:
    p = _parquet("pbp", "play_by_play", season, PBP_COLS)
    if p.empty:
        return p
    part = _parquet("pbp_participation", "pbp_participation", season)
    if not part.empty:
        part = part.rename(columns={"nflverse_game_id": "game_id"})
        keep = ["game_id", "play_id"] + [c for c in PART_COLS if c in part.columns]
        p = p.merge(part[keep], on=["game_id", "play_id"], how="left")
    ftn = _parquet("ftn_charting", "ftn_charting", season)
    if not ftn.empty:
        ftn = ftn.rename(columns={"nflverse_game_id": "game_id", "nflverse_play_id": "play_id"})
        keep = ["game_id", "play_id"] + [c for c in FTN_COLS if c in ftn.columns]
        p = p.merge(ftn[keep].drop_duplicates(["game_id", "play_id"]), on=["game_id", "play_id"], how="left",
                    suffixes=("", "_ftn"))
    return p
