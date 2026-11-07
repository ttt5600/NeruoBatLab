"""Census of NON-weather Kalshi series that settle on a physical / mechanical process (2026-11-06).

Read-only public API, no orders. Counts and volumes only -- no forecasts or outcomes are joined here (the census
happens before any trial is registered).

    python scripts/phys_census.py series     # full /series list -> data/phys_census/series_all.json
    python scripts/phys_census.py markets    # per candidate series: settled/open counts, volume -> markets.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "data" / "phys_census"
H = {"User-Agent": "Mozilla/5.0"}

# Candidates picked from the /series list by keyword (launch, power, sea ice, geomagnetic, CO2, disease counts, ...).
# Weather/climate series are in data/wx_census (scripts/wx_census_all.py) and are not repeated here.
CANDIDATES = [
    # ISO power prices (hub day-ahead/real-time averages -> partially observed within the week/month)
    "KXPJMWESTW", "KXPJMWESTM", "KXPJMDOMW", "KXPJMDOMM", "KXPJMAEPDAYW", "KXPJMAEPDAYM",
    "KXMISOINDW", "KXMISOINDM", "KXERCOTHOUW", "KXERCOTHOUM", "KXERCOTNORTHW", "KXERCOTNORTHM",
    "KXERCOTWESTW", "KXERCOTWESTM", "KXPJMEMERGENCY", "KXPOWERKWH", "KXCPIELECYOY",
    # natural gas / energy inventories
    "KXNATGASD", "KXNATGASW", "KXNATGASMON", "KXNGASW", "KXNGAS", "KXEIAGASINVM", "KXEIAREFUTILM",
    "KXEIACRUDEXPM", "KXSPRLVL", "KXOILRIGS",
    # space launches
    "KXROCKET", "ROCKET", "KXSPACEXCOUNT", "KXLAUNCHCOUNTM", "KXOTHERLAUNCH", "KXLAUNCHES", "KXRKLBCOUNT",
    "KXSPACEX", "KXSPCXLAUNCH", "KXSTARSHIPLAUNCH", "KXSPACEXREUSE", "KXNEXTVULCAN", "KXNEXTLAUNCHSMALL",
    "KXDRAGONISS", "KXSTARSHIP",
    # space weather / earth system
    "KXGSTORM", "GSTORM", "KXSEAICE", "SEAICE", "KXCO2LEVEL",
    # disease counts (surveillance -> partially observed cumulative counts)
    "KXMEASLES", "MEASLES", "KXCASEW", "KXCASED", "KXCASE7D", "KXH5N1COUNT", "KXH5N1CASES", "KXFLDENGUE",
    "KXCHIKCOUNT", "KXSCREWWORMCOUNT", "KXMPOXCOUNT", "KXPERTUSSIS", "KXTBCOUNT", "KXPOLIO", "KXCDCSTATFLU",
    "KXVAXX", "KXBOOSTER",
    # misc mechanical
    "KXEVMKT", "KXSOLAR", "KXLOUDOUNWATERDC", "KXILNUCLEAR",
]


def _get(path, **p):
    for i in range(5):
        try:
            r = requests.get(f"{API}{path}", params=p, headers=H, timeout=60)
        except requests.RequestException:
            time.sleep(2 + 2 * i)
            continue
        if r.status_code == 429:
            time.sleep(2 + 2 * i)
            continue
        return r.json()
    return {}


def _all(path, **p):
    out, cur = [], None
    while True:
        q = dict(p, limit=1000)
        if cur:
            q["cursor"] = cur
        j = _get(path, **q)
        out += j.get("markets", [])
        cur = j.get("cursor")
        if not cur or len(out) > 50000:
            return out


def series():
    OUT.mkdir(parents=True, exist_ok=True)
    r = _get("/series", limit=20000)
    json.dump(r["series"], open(OUT / "series_all.json", "w"))
    print(len(r["series"]), "series")


def markets():
    OUT.mkdir(parents=True, exist_ok=True)
    ser = {x["ticker"]: x for x in json.load(open(OUT / "series_all.json"))}
    rows = {}
    for s in CANDIDATES:
        st = _all("/markets", series_ticker=s, status="settled")
        hi = _all("/historical/markets", series_ticker=s)
        op = _all("/markets", series_ticker=s, status="open")
        ms = {m["ticker"]: m for m in st + hi}
        vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in ms.values())
        evs = sorted({m["event_ticker"] for m in ms.values()})
        x = ser.get(s, {})
        rows[s] = {"cat": x.get("category"), "freq": x.get("frequency"), "title": x.get("title"),
                   "settle_src": [d.get("name") for d in (x.get("settlement_sources") or [])],
                   "fee_type": x.get("fee_type"),
                   "settled_mkts": len(ms), "settled_events": len(evs), "open_mkts": len(op), "vol": vol,
                   "open_vol24": sum(float(m.get("volume_24h_fp") or m.get("volume_24h") or 0) for m in op),
                   "first": min((m.get("close_time") or "" for m in ms.values()), default=None),
                   "last": max((m.get("close_time") or "" for m in ms.values()), default=None),
                   "rules": (op or list(ms.values()) or [{}])[0].get("rules_primary")}
        print(s, rows[s]["settled_events"], round(vol), rows[s]["first"], rows[s]["last"], flush=True)
    json.dump(rows, open(OUT / "markets.json", "w"), indent=1)


if __name__ == "__main__":
    {"series": series, "markets": markets}[sys.argv[1]]()
