"""Fetch NBA.com per-game player x defender matchups (boxscorematchupsv3) into data/nba_matchups/ (gitignored).

    ../.venv_quant/bin/python scripts/nba_matchup_fetch.py probe GAMEID [GAMEID ...]
    ../.venv_quant/bin/python scripts/nba_matchup_fetch.py games SEASON          # 2023-24 -> game id list (leaguegamelog)
    ../.venv_quant/bin/python scripts/nba_matchup_fetch.py fetch SEASON [SEASON ...]  # all regular-season games, <=1 req/s

One raw JSON per game: data/nba_matchups/raw/{GAMEID}.json.gz. Empty/failed games logged in data/nba_matchups/fail.txt.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "nba_matchups"
RAW = OUT / "raw"
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/129.0.0.0 Safari/537.36")
HDR = {"User-Agent": UA, "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
       "x-nba-stats-origin": "stats", "x-nba-stats-token": "true", "Accept": "application/json, text/plain, */*",
       "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "gzip, deflate, br", "Connection": "keep-alive",
       "Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty"}
S = requests.Session()
S.headers.update(HDR)


def get(url: str, params: dict, tries: int = 6) -> dict | None:
    wait = 3.0
    for k in range(tries):
        try:
            r = S.get(url, params=params, timeout=40)
            if r.status_code == 200:
                return r.json()
            print(f"  http {r.status_code} {params}", flush=True)
        except Exception as e:  # timeouts, resets
            print(f"  err {type(e).__name__} {params}", flush=True)
        time.sleep(wait)
        wait = min(wait * 2, 90)
    return None


def matchups(gid: str) -> dict | None:
    return get("https://stats.nba.com/stats/boxscorematchupsv3",
               {"GameID": gid, "LeagueID": "00", "endPeriod": 0, "endRange": 31800, "rangeType": 0,
                "startPeriod": 0, "startRange": 0})


def games(season: str) -> list[tuple[str, str]]:
    cache = OUT / f"games_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    d = get("https://stats.nba.com/stats/leaguegamelog",
            {"Counter": 1000, "DateFrom": "", "DateTo": "", "Direction": "ASC", "LeagueID": "00",
             "PlayerOrTeam": "T", "Season": season, "SeasonType": "Regular Season", "Sorter": "DATE"})
    rs = d["resultSets"][0]
    h = rs["headers"]
    gi, di = h.index("GAME_ID"), h.index("GAME_DATE")
    seen = {}
    for row in rs["rowSet"]:
        seen[row[gi]] = row[di]
    out = sorted(seen.items(), key=lambda x: (x[1], x[0]))
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def n_pairs(d: dict) -> int:
    b = d.get("boxScoreMatchups", {})
    return sum(len(p.get("matchups", [])) for t in ("homeTeam", "awayTeam") for p in b.get(t, {}).get("players", []))


def main() -> None:
    cmd = sys.argv[1]
    RAW.mkdir(parents=True, exist_ok=True)
    if cmd == "probe":
        for gid in sys.argv[2:]:
            d = matchups(gid)
            print(gid, None if d is None else n_pairs(d), flush=True)
            time.sleep(1.1)
    elif cmd == "games":
        g = games(sys.argv[2])
        print(len(g), g[:2], g[-2:])
    elif cmd == "fetch":
        for season in sys.argv[2:]:
            g = games(season)
            time.sleep(1.1)
            todo = [x for x in g if not (RAW / f"{x[0]}.json.gz").exists()]
            print(season, len(g), "games;", len(todo), "to fetch", flush=True)
            for i, (gid, date) in enumerate(todo):
                t0 = time.time()
                d = matchups(gid)
                if d is None or n_pairs(d) == 0:
                    with open(OUT / "fail.txt", "a") as f:
                        f.write(f"{season}\t{gid}\t{date}\t{'none' if d is None else 'empty'}\n")
                else:
                    with gzip.open(RAW / f"{gid}.json.gz", "wt") as f:
                        json.dump(d, f)
                if i % 100 == 0:
                    print(f"  {season} {i}/{len(todo)} {gid} {date}", flush=True)
                time.sleep(max(0.0, 1.05 - (time.time() - t0)))


if __name__ == "__main__":
    main()
