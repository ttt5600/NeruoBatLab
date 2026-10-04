"""Capture small real API responses as offline test fixtures.

    python scripts/capture_fixtures.py

Parsers are tested against what the APIs actually return, not what their
(unofficial, undocumented) docs say they return. Re-run this when an endpoint
changes shape, and the parser tests will say what broke.
"""
from __future__ import annotations

import gzip
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sports import mlb, nba, nfl, nhl
from sports.http import fetch

FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def dump(name: str, obj) -> None:
    with gzip.open(FIX / f"{name}.json.gz", "wt") as f:
        json.dump(obj, f)
    print("wrote", name)


def main() -> int:
    FIX.mkdir(parents=True, exist_ok=True)

    # MLB: July 4 2025 (outdoor, roof and weather variety) + 2024 World Series G5.
    dump("mlb_schedule_2025-07-04", fetch(mlb.SCHEDULE, {
        "sportId": 1, "startDate": "2025-07-04", "endDate": "2025-07-04", "hydrate": mlb.HYDRATE}))
    dump("mlb_schedule_2024-10-30", fetch(mlb.SCHEDULE, {
        "sportId": 1, "startDate": "2024-10-30", "endDate": "2024-10-30", "hydrate": mlb.HYDRATE}))

    # NHL: one regular-season week, one boxscore.
    week = fetch(f"{nhl.BASE}/schedule/2025-01-13")
    dump("nhl_schedule_2025-01-13", week)
    gid = week["gameWeek"][0]["games"][0]["id"]
    dump("nhl_boxscore", fetch(f"{nhl.BASE}/gamecenter/{gid}/boxscore"))

    # NBA: a regular day and an All-Star Sunday (fake teams must be dropped).
    dump("nba_scoreboard_2025-01-15", fetch(nba.SCOREBOARD, {"dates": "20250115"}))
    dump("nba_scoreboard_2025-02-16", fetch(nba.SCOREBOARD, {"dates": "20250216"}))

    # NFL: the 2024 season slice of the nflverse games table.
    g = nfl.raw_games()
    buf = io.StringIO()
    g[g["season"] == 2024].to_csv(buf, index=False)
    with gzip.open(FIX / "nfl_games_2024.csv.gz", "wt") as f:
        f.write(buf.getvalue())
    print("wrote nfl_games_2024")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
