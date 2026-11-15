"""Download MoneyPuck team + goalie game-by-game CSVs (seasons 2021-22..2026-27). Results only, no lines."""
import csv, io, sys, time, urllib.request
from pathlib import Path
OUT = Path(__file__).parent / "mp"; OUT.mkdir(exist_ok=True)
B = "https://moneypuck.com/moneypuck/playerData"


def get(u):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
        except Exception as e:  # noqa: BLE001
            print("retry", u, e); time.sleep(3 * (i + 1))
    raise RuntimeError(u)


teams, goalies = set(), set()
for s in range(2021, 2027):
    for kind in ("teams", "goalies"):
        f = OUT / f"summary_{kind}_{s}.csv"
        if not f.exists():
            f.write_bytes(get(f"{B}/seasonSummary/{s}/regular/{kind}.csv"))
        for r in csv.DictReader(io.StringIO(f.read_text())):
            (teams.add(r["team"]) if kind == "teams" else goalies.add(r["playerId"]))
print(len(teams), sorted(teams), len(goalies))
ref = "--refresh" in sys.argv
for t in sorted(teams):
    f = OUT / f"team_{t}.csv"
    if not f.exists() or ref:
        f.write_bytes(get(f"{B}/careers/gameByGame/regular/teams/{t}.csv")); time.sleep(0.3)
for g in sorted(goalies):
    f = OUT / f"goalie_{g}.csv"
    if not f.exists() or ref:
        f.write_bytes(get(f"{B}/careers/gameByGame/regular/goalies/{g}.csv")); time.sleep(0.3)
print("done")
