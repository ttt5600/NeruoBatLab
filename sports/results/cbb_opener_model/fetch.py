"""CBB opener model data: ESPN scoreboard finals (scores) + ESPN core odds (DK total open/close).

    python results/cbb_opener_model/fetch.py games 2023 2024 2025 2026 2027
    python results/cbb_opener_model/fetch.py odds 2024 2025 2027      # 2026 (2025-26) lines are SEALED: never fetched
Writes results/cbb_opener_model/games_<s>.parquet, odds_<s>.parquet. Read-only public GETs, cached (sports.http).
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

from sports.http import fetch

OUT = Path(__file__).resolve().parent
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"
SEASONS = {2023: ("2022-11-07", "2023-04-03"), 2024: ("2023-11-06", "2024-04-08"), 2025: ("2024-11-04", "2025-04-07"),
           2026: ("2025-11-03", "2026-04-06"), 2027: ("2026-11-02", "2026-11-12")}


def games(season):
    a, b = SEASONS[season]
    rows = []
    for d in pd.date_range(a, b):
        sb = fetch(f"{SITE}/scoreboard", {"dates": d.strftime("%Y%m%d"), "groups": 50, "limit": 400},
                   cache=d < pd.Timestamp.now() - pd.Timedelta(days=2))
        for e in sb.get("events", []):
            c = e["competitions"][0]
            if c.get("status", {}).get("type", {}).get("completed") is not True:
                continue
            tm = {x["homeAway"]: x for x in c["competitors"]}
            try:
                hs, as_ = float(tm["home"]["score"]), float(tm["away"]["score"])
            except Exception:  # noqa: BLE001
                continue
            rows.append({"eid": str(e["id"]), "date_et": d, "start": e.get("date"), "neutral": bool(c.get("neutralSite")),
                         "home_id": str(tm["home"]["team"]["id"]), "away_id": str(tm["away"]["team"]["id"]),
                         "home": tm["home"]["team"].get("abbreviation"), "away": tm["away"]["team"].get("abbreviation"),
                         "home_score": hs, "away_score": as_, "n_periods": c.get("status", {}).get("period"),
                         "stype": e.get("season", {}).get("type"), "season": season})
    df = pd.DataFrame(rows).drop_duplicates("eid")
    df.to_parquet(OUT / f"games_{season}.parquet")
    print(season, len(df))


def _num(x):
    try:
        return float(str(x).replace("o", "").replace("u", "").replace("+", ""))
    except Exception:  # noqa: BLE001
        return None


def odds_one(eid):
    try:
        j = fetch(f"{CORE}/events/{eid}/competitions/{eid}/odds", {"limit": 100})
    except Exception:  # noqa: BLE001
        return []
    out = []
    for it in j.get("items", []):
        prov = (it.get("provider") or {}).get("name")
        r = {"eid": eid, "provider": prov}
        for when in ("open", "close"):
            w = it.get(when) or {}
            t = w.get("total") or {}
            r[f"total_{when}"] = _num(t.get("american") or t.get("value") or t.get("alternateDisplayValue"))
            r[f"over_{when}"] = _num((w.get("over") or {}).get("american"))
            r[f"under_{when}"] = _num((w.get("under") or {}).get("american"))
        r["ou_current"] = it.get("overUnder")
        # 2026-27 "new format": prices under item['odds']
        od = it.get("odds") or {}
        for when in ("open", "close"):
            for s in ("over", "under"):
                if r[f"{s}_{when}"] is None:
                    v = ((od.get(s) or {}).get(when) or {}).get("american")
                    r[f"{s}_{when}"] = _num(v)
        out.append(r)
    return out


def odds(season):
    g = pd.read_parquet(OUT / f"games_{season}.parquet")
    with ThreadPoolExecutor(4) as ex:
        rows = [r for rs in ex.map(odds_one, g.eid.tolist()) for r in rs]
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / f"odds_{season}.parquet")
    print(season, len(df), df.provider.value_counts().to_dict() if len(df) else {})


if __name__ == "__main__":
    cmd, seasons = sys.argv[1], [int(s) for s in sys.argv[2:]]
    for s in seasons:
        if cmd == "odds" and s == 2026:
            raise SystemExit("2025-26 lines are sealed")
        (games if cmd == "games" else odds)(s)
