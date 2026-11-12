"""Read-only timing logger (forward rows), 11-12 night. Every 10 min until --until (UTC):
Kalshi: all open markets in the 7 ladder series -> per event per tick: anchor strike/bid/ask/sizes, implied line, open_time.
Pinnacle: main total/spread lines for nfl/nhl/cfb/cbb. No orders. Appends to ticks.jsonl; first sightings in first_seen.json.
Usage: nohup python tonight_logger.py --until 2026-11-13T07:55Z &"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
sys.path.insert(0, str(Path.home() / ".claude/quant-kalshi"))
from kx import SERIES, get, implied, side_x  # noqa: E402
from sports import live  # noqa: E402

live.LEAGUES.setdefault("cbb", (493, "KXNCAAMBGAME", "cbb"))
F = lambda v: float(v) if v not in (None, "") else float("nan")  # noqa: E731


def kalshi_tick(t):
    rows = []
    for (lg, mk), s in SERIES.items():
        ms, cur = [], None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = get("/markets", p)
            ms += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
        ev = defaultdict(list)
        for m in ms:
            ev[m["event_ticker"]].append(m)
        for e, mm in ev.items():
            pts, best = [], None
            for m in mm:
                if m.get("floor_strike") is None:
                    continue
                b, a = F(m.get("yes_bid_dollars")), F(m.get("yes_ask_dollars"))
                if not (b > 0 and a < 1):
                    continue
                x, sg = side_x(e, m["ticker"], m["floor_strike"], mk)
                mid = (a + b) / 2
                pts.append((x, mid if sg > 0 else 1 - mid))
                if best is None or abs(mid - .5) < abs(best[0] - .5):
                    best = (mid, m, b, a)
            r = {"t": t, "lg": lg, "mk": mk, "event": e, "n_mk": len(mm), "n_2s": len(pts),
                 "open_time": min(m["open_time"] for m in mm), "occ": mm[0].get("occurrence_datetime"),
                 "exp": mm[0].get("expected_expiration_time"), "implied": implied(pts) if pts else None}
            if best:
                _, m, b, a = best
                r.update(anchor=m["ticker"], strike=float(m["floor_strike"]), bid=b, ask=a,
                         bid_sz=F(m.get("yes_bid_size_fp")), ask_sz=F(m.get("yes_ask_size_fp")))
            rows.append(r)
    return rows


def pin_tick(t):
    rows = []
    for lg in ("nfl", "nhl", "cfb", "cbb"):
        try:
            d = live.pinnacle(lg)
        except Exception as e:  # noqa: BLE001
            print("pin err", lg, type(e).__name__, flush=True)
            continue
        if d.empty:
            continue
        d = d[d.market.isin(["spread", "total"]) & ~d.live]
        for (pid, mk), g in d.groupby(["pin_id", "market"]):
            r = g.iloc[g.hold.argsort()].iloc[0]
            rows.append({"t": t, "lg": lg, "src": "pin", "pin_id": int(pid), "mk": mk, "home": r.home, "away": r.away,
                         "start": pd.Timestamp(r.start).isoformat(), "line": None if pd.isna(r.line) else float(r.line),
                         "p1": float(r.price1), "p2": float(r.price2), "fair1": round(float(r.fair1), 5)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--until", required=True)
    ap.add_argument("--every", type=int, default=600)
    a = ap.parse_args()
    until = pd.Timestamp(a.until)
    fs_path = OUT / "first_seen.json"
    fs = json.loads(fs_path.read_text()) if fs_path.exists() else {}
    while pd.Timestamp.now(tz="UTC") < until:
        t = pd.Timestamp.now(tz="UTC").isoformat()
        try:
            rows = kalshi_tick(t) + pin_tick(t)
        except Exception as e:  # noqa: BLE001
            print(t, "tick err", type(e).__name__, e, flush=True)
            rows = []
        with open(OUT / "ticks.jsonl", "a") as f:
            for r in rows:
                f.write(json.dumps(r, default=str) + "\n")
                k = r["event"] if "event" in r else f"pin|{r['lg']}|{r['pin_id']}|{r['mk']}"
                if k not in fs:
                    fs[k] = r
        fs_path.write_text(json.dumps(fs, default=str))
        print(t, "rows", len(rows), "first_seen", len(fs), flush=True)
        time.sleep(max(0, min(a.every, (until - pd.Timestamp.now(tz="UTC")).total_seconds())))
    print("done", flush=True)


if __name__ == "__main__":
    main()
