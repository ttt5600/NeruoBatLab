"""Scan Kalshi (public API, no key) for earnings-related markets: earnings-call MENTION series and company KPI series.

    python scripts/earnings_kalshi_scan.py open      # open events -> data/earnings/kalshi_open_events.json
    python scripts/earnings_kalshi_scan.py settled   # settled mention markets -> data/earnings/kalshi_settled_mentions.json

Kalshi has NO "beats EPS consensus" contract (checked the full /series list 2026-10-21: 15,336 series; earnings content =
~190 KXEARNINGSMENTION* word-mention series + company annual-KPI series). Data only, no orders.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
D = Path(__file__).resolve().parents[1] / "data" / "earnings"
KPI_EXTRA = {"NETFLIXSUBS", "TESLA", "SPOTIFYSUBS", "SOFIMEMBERS", "FBDAU", "METADAP", "NYTSUBS", "CBVOLUME", "KXUBERKPI"}


def get(path: str, **params) -> dict:
    for _ in range(5):
        r = requests.get(f"{API}{path}", params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(2)
            continue
        r.raise_for_status()
        return r.json()
    return {}


def tickers() -> list[str]:
    s = json.load(open(D / "series.json"))["series"]
    out = []
    for x in s:
        t = x["ticker"]
        if ("MENTION" in t and "EARN" in t) or t == "KXBRKEM":
            out.append(t)
        elif x.get("category") == "Companies" and (t.endswith("A") or "KPI" in t or t in KPI_EXTRA):
            out.append(t)
    return out


def scan(status: str) -> list[dict]:
    out = []
    for t in tickers():
        cursor = None
        while True:
            kw = {"series_ticker": t, "status": status, "with_nested_markets": "true", "limit": 200}
            if cursor:
                kw["cursor"] = cursor
            j = get("/events", **kw)
            out += j.get("events", [])
            cursor = j.get("cursor")
            if not cursor or status == "open":
                break
            time.sleep(0.1)
        time.sleep(0.1)
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "open"
    ev = scan(mode)
    fn = D / ("kalshi_open_events.json" if mode == "open" else "kalshi_settled_mentions.json")
    json.dump(ev, open(fn, "w"))
    print(mode, len(tickers()), "series ->", len(ev), "events ->", fn)
