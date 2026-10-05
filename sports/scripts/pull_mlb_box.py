"""Pull MLB boxscores -> data/mlb_pitching.parquet, data/mlb_team_batting.parquet.

    python scripts/pull_mlb_box.py --first 2019

Needs data/mlb_games.parquet (scripts/pull.py) for the list of game ids.
~2,430 requests per season, cached, so an interrupted pull resumes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sports import mlb_box

DATA = Path(__file__).resolve().parents[1] / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2019)
    args = ap.parse_args()
    games = pd.read_parquet(DATA / "mlb_games.parquet")
    games = games[games["season"] >= args.first]
    keep = ["game_id", "season", "game_type", "date", "start_utc", "venue", "temp_f",
            "wind_mph", "condition", "officials", "home", "away"]
    P, B = [], []
    for s, g in games.groupby("season"):
        t = time.time()
        p, b = mlb_box.boxscores(g["game_id"].astype(int).tolist())
        P.append(p)
        B.append(b)
        print(f"{s}: {len(g)} games, {int(p['started'].sum())} starts in {time.time() - t:.0f}s", flush=True)
    meta = games[keep].assign(game_pk=games["game_id"].astype(int)).drop(columns="game_id")
    pit = pd.concat(P, ignore_index=True).merge(meta, on="game_pk", how="left")
    bat = pd.concat(B, ignore_index=True).merge(meta, on="game_pk", how="left")
    pit.to_parquet(DATA / "mlb_pitching.parquet", index=False)
    bat.to_parquet(DATA / "mlb_team_batting.parquet", index=False)
    print(f"pitching rows {len(pit)}, team-batting rows {len(bat)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
