"""E1-pattern family scan (2026-11-01): for candidate Kalshi series, print the series meta, one open ladder
(far-rung bids/asks), and the number of settled markets/events (live + historical tier).

Read-only public API. Paced <= ~1.6 req/s with 5-60 s backoff on 429 (Kalshi API is IP-shared).
Usage: python scripts/e1_family_scan.py KXTSAW KXTSAD ...
"""
from __future__ import annotations

import sys
import time

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"


def get(u: str, p: dict | None = None) -> dict:
    for a in range(10):
        time.sleep(0.6)
        try:
            r = requests.get(API + u, params=p, timeout=30)
        except requests.RequestException:
            time.sleep(5)
            continue
        if r.status_code == 429:
            time.sleep(min(5 * (a + 1), 60))
            continue
        if r.status_code == 404:
            return {}
        r.raise_for_status()
        return r.json()
    raise RuntimeError(u)


def scan(s: str) -> None:
    se = get(f"/series/{s}").get("series", {})
    print("=====", s, se.get("frequency"), se.get("fee_type"), se.get("settlement_sources"))
    op = get("/markets", {"series_ticker": s, "status": "open", "limit": 200}).get("markets", [])
    if op:
        print("RULES:", op[0]["rules_primary"][:500])
        ev = op[0]["event_ticker"]
        for m in sorted([m for m in op if m["event_ticker"] == ev], key=lambda m: (m.get("floor_strike") or 0)):
            print(f"  {m['ticker']:42s} {m.get('strike_type')} fl={m.get('floor_strike')} cap={m.get('cap_strike')} "
                  f"yb={m.get('yes_bid_dollars')} ya={m.get('yes_ask_dollars')} vol={m.get('volume_fp')} "
                  f"oi={m.get('open_interest_fp')} close={m['close_time']}")
    n, evs, first = 0, set(), None
    for ep in ("/markets", "/historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000}
            if ep == "/markets":
                p["status"] = "settled"
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            ms = r.get("markets", [])
            n += len(ms)
            evs |= {m["event_ticker"] for m in ms}
            for m in ms:
                if first is None or m["close_time"] < first:
                    first = m["close_time"]
            cur = r.get("cursor")
            if not cur or not ms:
                break
    print(f"settled markets {n} events {len(evs)} first close {first}")


if __name__ == "__main__":
    for s in sys.argv[1:]:
        scan(s)
