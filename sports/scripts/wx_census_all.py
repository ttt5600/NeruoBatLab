"""Census of ALL Kalshi Climate-and-Weather series (any frequency), 2026-11-06. Read-only public API, no orders.

    python scripts/wx_census_all.py series      # list series + frequency -> data/wx_census/series.json
    python scripts/wx_census_all.py markets     # per non-traded series: settled/open counts, volume -> markets.json

Counts and volumes only; no forecasts or outcomes are joined here (census happens before any trial is registered).
"""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sports import wx  # noqa: E402

OUT = ROOT / "data" / "wx_census"
H = {"User-Agent": "Mozilla/5.0"}


def _get(path, **p):
    for i in range(4):
        r = requests.get(f"{wx.API}{path}", params=p, headers=H, timeout=60)
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
        if not cur or len(out) > 20000:
            return out


def series():
    OUT.mkdir(parents=True, exist_ok=True)
    r = _get("/series", limit=5000)
    ser = [x for x in r["series"] if x.get("category") == "Climate and Weather"]
    json.dump(ser, open(OUT / "series.json", "w"))
    print(len(r["series"]), "series total;", len(ser), "Climate and Weather")
    print(collections.Counter(x.get("frequency") for x in ser))


def markets(prefix_skip=("KXHIGHT", "KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHLAX",
                         "KXHIGHPHIL", "KXLOWT")):
    ser = json.load(open(OUT / "series.json"))
    rows = {}
    for x in sorted(ser, key=lambda x: x["ticker"]):
        s = x["ticker"]
        if s.startswith(prefix_skip):
            continue
        st = _all("/markets", series_ticker=s, status="settled")
        hi = _all("/historical/markets", series_ticker=s)
        op = _all("/markets", series_ticker=s, status="open")
        ms = {m["ticker"]: m for m in st + hi}
        vol = sum(float(m.get("volume_fp") or m.get("volume") or 0) for m in ms.values())
        evs = sorted({m["event_ticker"] for m in ms.values()})
        rows[s] = {"freq": x.get("frequency"), "title": x.get("title"), "settled_mkts": len(ms),
                   "settled_events": len(evs), "open_mkts": len(op), "vol": vol,
                   "open_vol24": sum(float(m.get("volume_24h_fp") or m.get("volume_24h") or 0) for m in op),
                   "first": min((m.get("close_time") or "" for m in ms.values()), default=None),
                   "last": max((m.get("close_time") or "" for m in ms.values()), default=None)}
        print("..", s, file=sys.stderr, flush=True)
        if ms or op:
            print(s, json.dumps(rows[s]), flush=True)
    json.dump(rows, open(OUT / "markets.json", "w"), indent=1)


if __name__ == "__main__":
    {"series": series, "markets": markets}[sys.argv[1]]()
