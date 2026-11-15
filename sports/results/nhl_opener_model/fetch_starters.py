"""NHL API (api-web.nhle.com): 2026-27 regular-season games 10-01..11-14 -> starting goalies (boxscore 'starter' flag)."""
import json, time, urllib.request
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
OUT = Path(__file__).parent
C = OUT / "nhlapi"; C.mkdir(exist_ok=True)


def get(u, f):
    if f.exists():
        return json.loads(f.read_text())
    for i in range(5):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=30).read()
            f.write_bytes(b); time.sleep(0.2)
            return json.loads(b)
        except Exception as e:  # noqa: BLE001
            print("retry", u, e); time.sleep(3 * (i + 1))
    raise RuntimeError(u)


rows, d = [], date(2026, 10, 1)
while d <= date(2026, 11, 14):
    s = get(f"https://api-web.nhle.com/v1/schedule/{d}", C / f"sched_{d}.json")
    for wk in s.get("gameWeek", []):
        if wk["date"] != str(d):
            continue
        for g in wk["games"]:
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                continue
            b = get(f"https://api-web.nhle.com/v1/gamecenter/{g['id']}/boxscore", C / f"box_{g['id']}.json")
            r = {"gameId": g["id"], "date": str(d), "start_utc": g["startTimeUTC"],
                 "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                 "hs": g["homeTeam"].get("score"), "as": g["awayTeam"].get("score")}
            for side in ("home", "away"):
                gl = b["playerByGameStats"][f"{side}Team"]["goalies"]
                st = [x for x in gl if x.get("starter")]
                r[f"{side}_starter_id"] = st[0]["playerId"] if st else None
            rows.append(r)
    d += timedelta(days=1)
df = pd.DataFrame(rows)
df.to_parquet(OUT / "starters_2027.parquet")
print(len(df), df.home_starter_id.isna().sum(), df.away_starter_id.isna().sum())
