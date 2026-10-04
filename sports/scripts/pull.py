"""Pull game tables for all four leagues to data/{league}_games.parquet.

    python scripts/pull.py                       # all leagues, 2015 on
    python scripts/pull.py --leagues NHL --goalies
    python scripts/pull.py --coverage            # report only, no network

Every response for a finished day is cached, so re-running is cheap and an
interrupted pull resumes where it stopped.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sports import mlb, nba, nfl, nhl

DATA = Path(__file__).resolve().parents[1] / "data"
THIS_YEAR = date.today().year


def out_path(league: str) -> Path:
    return DATA / f"{league.lower()}_games.parquet"


def pull(league: str, first: int, goalies: bool) -> pd.DataFrame:
    seasons = range(first, THIS_YEAR + 1)
    if league == "NFL":
        return nfl.games(seasons)
    if league == "MLB":
        return mlb.games(seasons)
    if league == "NHL":
        return nhl.games(seasons, goalies=goalies)
    if league == "NBA":
        return nba.games(seasons)
    raise ValueError(league)


def pull_nfl_players(first: int) -> None:
    df = nfl.player_week(range(first, THIS_YEAR + 1))
    p = DATA / "nfl_player_week.parquet"
    df.to_parquet(p, index=False)
    print(f"NFL players: {len(df)} player-weeks, {df['player_id'].nunique()} players -> {p.name}")


def pull_nba_boxes(seasons: list[int]) -> None:
    """One request per game; ~1,300 a season. Cached, so safe to interrupt."""
    games = pd.read_parquet(out_path("NBA"))
    for s in seasons:
        ids = games.loc[games["season"] == s, "game_id"].tolist()
        t = time.time()
        box = nba.box_scores(ids)
        p = DATA / f"nba_box_{s}.parquet"
        box.to_parquet(p, index=False)
        print(f"NBA box {s}: {len(ids)} games, {len(box)} player rows in "
              f"{time.time() - t:.0f}s -> {p.name}", flush=True)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Share of non-missing values per column per season -- read before trusting."""
    cols = [c for c in df.columns if c not in ("league", "season")]
    return df.groupby("season")[cols].agg(lambda s: s.notna().mean()).round(2)


def report(league: str) -> None:
    p = out_path(league)
    if not p.exists():
        print(f"{league}: not pulled")
        return
    df = pd.read_parquet(p)
    reg = df[df["game_type"] == "regular"]
    print(f"\n=== {league}: {len(df)} games ({len(reg)} regular), "
          f"seasons {df['season'].min()}-{df['season'].max()}")
    print(reg.groupby("season").size().rename("regular_games").to_frame().T.to_string())
    cov = coverage(df)
    informative = [c for c in cov.columns if 0 < cov[c].max() and cov[c].min() < 1]
    if informative:
        print("partially-covered columns (share present by season):")
        print(cov[informative].T.to_string())
    empty = [c for c in cov.columns if cov[c].max() == 0]
    print("not available for this league:", ", ".join(empty) or "none")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="+", default=["NFL", "MLB", "NHL", "NBA"])
    ap.add_argument("--first", type=int, default=2015)
    ap.add_argument("--goalies", action="store_true", help="NHL starting goalies (~1,400 requests/season)")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--nfl-players", action="store_true", help="weekly NFL player stats (1 file/season)")
    ap.add_argument("--nba-boxes", nargs="*", type=int, default=None,
                    help="NBA box scores for these seasons (~1,300 requests each)")
    args = ap.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)
    if args.nfl_players:
        pull_nfl_players(args.first)
    if args.nba_boxes:
        pull_nba_boxes(args.nba_boxes)
    if args.nfl_players or args.nba_boxes:
        return 0
    for lg in args.leagues:
        if not args.coverage:
            t = time.time()
            df = pull(lg, args.first, args.goalies)
            df.to_parquet(out_path(lg), index=False)
            print(f"{lg}: {len(df)} games in {time.time() - t:.0f}s -> {out_path(lg).name}", flush=True)
        report(lg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
