"""One command to collect backtest data: games, game lines, props, box scores, weather.

    python scripts/collect.py all     --league nfl --seasons 2022 2025
    python scripts/collect.py props   --league nba --seasons 2025
    python scripts/collect.py weather --league cfb --seasons 2023 2025
    python scripts/collect.py grade   --league nfl --seasons 2024     # props x box -> graded
    python scripts/collect.py status                                   # what is on disk
    python scripts/collect.py snapshot --league nfl                    # today's lines, timestamped

Writes data/espn/<league>/<kind>_<season>.parquet. Kinds:
  games    one row per game: teams, scores, venue, indoor flag, season type
  lines    one row per (game, book): spread / total / moneyline, open + close where reported
  props    one row per (game, book, player or team, market, side): line, price, open line/price
  box      long player box score: (game, athlete, group, stat, value)
  weather  outdoor games: observed (ERA5, hindsight) + forecast issued 1 and 3 days before
  graded   props joined to box scores: actual, won, push, profit per 1u, hours_before

Every HTTP response for a finished game is cached under data/cache, so a rerun
or an interrupted pull resumes for free. Seasons use ESPN's label (NBA/NHL
2025 = the 2024-25 season). `snapshot` is uncached and appends: run it a few
times a day to build open->close histories ESPN itself does not keep.
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow.parquet as pq

from sports import espn, gamewx, settle

DATA = Path(__file__).resolve().parents[1] / "data" / "espn"
KINDS = ["games", "lines", "props", "box", "weather"]


def path(league: str, kind: str, season: int) -> Path:
    return DATA / league / f"{kind}_{season}.parquet"


def save(df: pd.DataFrame, p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    for c in df.columns:  # ids as strings so parquet types are stable across seasons
        if c.endswith("_id") or c == "event_id":
            df[c] = df[c].astype("string")
    df.to_parquet(p, index=False)
    print(f"  -> {p.relative_to(DATA.parent.parent)}  ({len(df):,} rows)")


def load_games(league: str, season: int, refresh: bool = False) -> pd.DataFrame:
    p = path(league, "games", season)
    if p.exists() and not refresh:
        return pd.read_parquet(p)
    t = time.time()
    g = espn.games(league, [season])
    print(f"{league} {season}: {len(g)} games ({time.time() - t:.0f}s)")
    save(g, p)
    return g


def per_game(fn, league: str, g: pd.DataFrame, workers: int = 6) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    done = (pd.to_datetime(g["start_utc"], utc=True) < now - espn.SETTLED_AFTER).to_list()
    ids = g["event_id"].astype(str).to_list()
    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(lambda a: fn(league, a[0], cache=a[1]), zip(ids, done))):
            rows.extend(r)
            if (i + 1) % 100 == 0:
                print(f"    {i + 1}/{len(ids)} games, {len(rows):,} rows, {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def collect(kind: str, league: str, season: int, refresh: bool) -> None:
    g = load_games(league, season, refresh and kind == "games")
    if kind == "games" or g.empty:
        return
    print(f"{league} {season} {kind}:")
    if kind == "lines":
        save(per_game(espn.lines, league, g), path(league, kind, season))
    elif kind == "props":
        save(per_game(espn.props, league, g), path(league, kind, season))
    elif kind == "box":
        fin = g[g["status"] == "STATUS_FINAL"]
        save(per_game(espn.box, league, fin), path(league, kind, season))
    elif kind == "weather":
        save(gamewx.for_games(g, league), path(league, kind, season))


def grade(league: str, season: int) -> None:
    need = {k: path(league, k, season) for k in ("games", "props", "box")}
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        print(f"{league} {season}: collect {missing} first")
        return
    g, p, b = (pd.read_parquet(need[k]) for k in ("games", "props", "box"))
    out = settle.grade(p, b, g)
    save(out, path(league, "graded", season))
    ok = out["profit"].notna()
    print(f"  graded {ok.sum():,}/{len(out):,} prop sides; unmapped markets (top 10):")
    print(settle.unmapped(p, league).head(10).to_string())


def status() -> None:
    rows = []
    for p in sorted(DATA.glob("*/*.parquet")):
        kind, season = p.stem.rsplit("_", 1)
        rows.append({"league": p.parent.name, "season": season, "kind": kind,
                     "rows": pq.ParquetFile(p).metadata.num_rows})
    if not rows:
        print("nothing collected yet")
        return
    t = pd.DataFrame(rows).pivot_table(index=["league", "season"], columns="kind", values="rows", aggfunc="sum")
    print(t.fillna(0).astype(int).to_string())


def snapshot(league: str) -> None:
    today = pd.Timestamp.now(tz="UTC").date()
    ev = [espn.parse_event(league, e) for d in range(0, 4)
          for e in espn.scoreboard(league, today + pd.Timedelta(days=d))]
    g = pd.DataFrame(ev)
    if g.empty:
        print("no games in the next 4 days")
        return
    g = g[g["status"] == "STATUS_SCHEDULED"]
    ts = pd.Timestamp.now(tz="UTC")
    for kind, fn in (("lines", espn.lines), ("props", espn.props)):
        df = per_game(fn, league, g.assign(start_utc=ts + pd.Timedelta(days=30)))  # uncached
        df["snapshot_utc"] = ts
        save(df, DATA / league / "snapshots" / f"{kind}_{ts:%Y%m%dT%H%M}.parquet")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=KINDS + ["all", "grade", "status", "snapshot"])
    ap.add_argument("--league", choices=list(espn.LEAGUES))
    ap.add_argument("--seasons", type=int, nargs="+", help="one season, or FIRST LAST")
    ap.add_argument("--refresh", action="store_true", help="rebuild the games table")
    a = ap.parse_args()
    if a.kind == "status":
        status()
        return 0
    if not a.league:
        ap.error("--league is required")
    if a.kind == "snapshot":
        snapshot(a.league)
        return 0
    if not a.seasons:
        ap.error("--seasons is required")
    seasons = range(a.seasons[0], a.seasons[-1] + 1)
    for s in seasons:
        if a.kind == "grade":
            grade(a.league, s)
        else:
            for k in (KINDS if a.kind == "all" else [a.kind]):
                collect(k, a.league, s, a.refresh)
            if a.kind == "all":
                grade(a.league, s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
