"""One game-level schema for all four leagues, with a timing tag on every column.

The tags order the moments a bettor passes through before a game:

    PRE      known days ahead: schedule, venue, roof, coach, rest
    LINEUP   known roughly an hour before: starting QB / pitcher / goalie,
             officials, weather observed at the start (a forecast proxy)
    CLOSE    known only at the closing line: the market's final price
    POST     known after the game: scores, attendance

``pregame(df, as_of=...)`` returns only the columns knowable at that moment.
Build features through it and a final score cannot leak into a pre-game model
by accident -- the same job the execution lag does inside ``quant``.

Two honest caveats are recorded here rather than buried:

* Weather in LINEUP is the *observed* reading at the start, not the forecast a
  bettor actually saw. For a closing-line bet the two are close; for a bet
  placed days ahead they are not.
* Starters are LINEUP, but what each source stores differs: MLB gives the
  announced *probable* pitcher (genuinely pre-game); NFL and NHL give who
  *actually* started, which is usually but not always the announced one.
"""
from __future__ import annotations

import pandas as pd

PRE, LINEUP, CLOSE, POST = "pre", "lineup", "close", "post"
_ORDER = {PRE: 0, LINEUP: 1, CLOSE: 2, POST: 3}

COLUMNS: dict[str, str] = {
    # identity and schedule
    "league": PRE, "game_id": PRE, "season": PRE, "game_type": PRE,
    "date": PRE, "start_utc": PRE, "home": PRE, "away": PRE, "neutral": PRE,
    # venue
    "venue": PRE, "venue_lat": PRE, "venue_lon": PRE, "elevation_ft": PRE,
    "roof": PRE, "surface": PRE, "day_night": PRE,
    # people
    "home_coach": PRE, "away_coach": PRE,
    "home_starter": LINEUP, "away_starter": LINEUP,
    "home_starter_id": LINEUP, "away_starter_id": LINEUP,
    "officials": LINEUP,
    # conditions at the start
    "temp_f": LINEUP, "wind_mph": LINEUP, "wind_dir": LINEUP, "condition": LINEUP,
    # market (closing)
    "home_ml": CLOSE, "away_ml": CLOSE, "spread_line": CLOSE, "total_line": CLOSE,
    # outcome
    "home_score": POST, "away_score": POST, "overtime": POST, "attendance": POST,
}

GAME_TYPES = ("regular", "post")


def empty_games() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})


def conform(df: pd.DataFrame) -> pd.DataFrame:
    """Give a loader's output the full schema, in order, with sane dtypes.

    Missing columns are added as NA rather than dropped, so "this league has no
    coach data" is visible as an all-NA column instead of a KeyError later.
    Unknown columns are an error: an untagged column has no timing guarantee.
    """
    extra = set(df.columns) - set(COLUMNS)
    if extra:
        raise ValueError(f"untagged columns {sorted(extra)}; add them to schema.COLUMNS")
    out = df.reindex(columns=list(COLUMNS))
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["start_utc"] = pd.to_datetime(out["start_utc"], utc=True)
    for c in ("venue_lat", "venue_lon", "elevation_ft", "temp_f", "wind_mph",
              "home_ml", "away_ml", "spread_line", "total_line",
              "home_score", "away_score", "attendance"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["season"] = out["season"].astype(int)
    bad = set(out["game_type"].dropna()) - set(GAME_TYPES)
    if bad:
        raise ValueError(f"game_type must be one of {GAME_TYPES}, got {sorted(bad)}")
    if out["game_id"].duplicated().any():
        dup = out.loc[out["game_id"].duplicated(), "game_id"].head().tolist()
        raise ValueError(f"duplicate game_id, e.g. {dup}")
    return out.sort_values(["start_utc", "game_id"]).reset_index(drop=True)


def columns_known_at(as_of: str) -> list[str]:
    if as_of not in _ORDER:
        raise ValueError(f"as_of must be one of {list(_ORDER)}")
    lim = _ORDER[as_of]
    return [c for c, tag in COLUMNS.items() if _ORDER[tag] <= lim]


def pregame(df: pd.DataFrame, as_of: str = LINEUP) -> pd.DataFrame:
    """The columns of ``df`` that a bettor could have known at ``as_of``."""
    keep = [c for c in columns_known_at(as_of) if c in df.columns]
    return df[keep]
