"""READ-ONLY order-book snapshot of every open KXAAAGASCD<ST> state gas-change event (notes/ops_state_gas_book.md).

Why: order-book depth is the only state-gas data that is not retroactive (settlements, candles and RBOB can be pulled
later; notes/state_gas.md). This logs it twice a day so a maker variant of GS1 can later be judged filled-vs-unfilled.

    python scripts/state_gas_books.py snap --slot am|close   # one snapshot, appended to OUT as one gzip member
    python scripts/state_gas_books.py check                  # summarise what has been logged

Per snapshot, per state: GET /events (series KXAAAGASCD<ST>, open, nested markets = top of book, sizes, volume, OI)
then at once GET /markets/orderbooks (that state's tickers) for the full book of all 5 buckets. ~100 public GETs through
kx_throttle (shared <= ~3.3 req/s slot). No keys, no orders. One JSON row per market:
    ts, slot, state, event, ticker, bucket, status, close_time, yes_bid/ask (+size), no_bid/ask, last, volume, oi,
    yes_book / no_book (full [[price, qty], ...] ascending, as Kalshi returns them), yes_top3 / no_top3 (best first).
On a YES/NO book a YES ask at p is a NO bid at 1-p, so yes_top3 = resting YES bids and no_top3 = resting NO bids.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

QS = Path(__file__).resolve().parents[1]
OUT = QS / "data" / "kx_forward" / "state_gas_books.jsonl.gz"
STATES = ("AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK "
          "OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()
BATCH = 25


def _book(ob: dict, side: str) -> list:
    return [[float(p), float(q)] for p, q in (ob.get(f"{side}_dollars") or [])]


def snap(slot: str) -> int:
    t0 = time.time()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    mkts, missing, books = [], [], {}
    for st in STATES:
        d = kget("/events", {"series_ticker": f"KXAAAGASCD{st}", "status": "open", "with_nested_markets": "true"})
        evs = (d or {}).get("events") or []
        got = [m for e in evs for m in (e.get("markets") or []) if m.get("status") == "active"]
        if not got:
            missing.append(st)
        for m in got:
            m["_state"] = st
        mkts += got
        tick = [m["ticker"] for m in got]   # book right after its event (< 1 s skew vs the market fields)
        for i in range(0, len(tick), BATCH):
            d = kget("/markets/orderbooks", {"tickers": tick[i:i + BATCH]}) or {}
            for ob in d.get("orderbooks") or []:
                books[ob["ticker"]] = ob.get("orderbook_fp") or {}
    rows, nobook = [], 0
    for m in mkts:
        ob = books.get(m["ticker"])
        if ob is None:
            nobook += 1
            ob = {}
        yb, nb = _book(ob, "yes"), _book(ob, "no")
        rows.append({
            "ts": ts, "slot": slot, "state": m["_state"], "event": m["event_ticker"], "ticker": m["ticker"],
            "bucket": m["ticker"].rsplit("-", 1)[-1], "status": m.get("status"), "close_time": m.get("close_time"),
            "yes_bid": m.get("yes_bid_dollars"), "yes_ask": m.get("yes_ask_dollars"),
            "yes_bid_size": m.get("yes_bid_size_fp"), "yes_ask_size": m.get("yes_ask_size_fp"),
            "no_bid": m.get("no_bid_dollars"), "no_ask": m.get("no_ask_dollars"),
            "last": m.get("last_price_dollars"), "volume": m.get("volume_fp"), "oi": m.get("open_interest_fp"),
            "book_ok": m["ticker"] in books, "yes_book": yb, "no_book": nb,
            "yes_top3": yb[::-1][:3], "no_top3": nb[::-1][:3],
        })
    if not rows:
        print(f"{ts} slot={slot} NO open KXAAAGASCD markets (missing {len(missing)} states); nothing written")
        return 3
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        g.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows).encode())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "ab") as f:          # one complete gzip member per snapshot; gzip.open reads them all in order
        f.write(buf.getvalue())
    nst = len({r["state"] for r in rows})
    print(f"{ts} slot={slot} states={nst}/50 markets={len(rows)} events={len({r['event'] for r in rows})} "
          f"no_book={nobook} missing={','.join(missing) or '-'} bytes+={len(buf.getvalue())} {time.time() - t0:.0f}s")
    return 0 if nobook == 0 else 4


def check() -> int:
    if not OUT.exists():
        print("no file", OUT)
        return 1
    snaps: dict = {}
    with gzip.open(OUT, "rt") as f:
        for line in f:
            r = json.loads(line)
            k = (r["ts"], r["slot"])
            s = snaps.setdefault(k, {"states": set(), "n": 0, "events": set()})
            s["states"].add(r["state"]); s["n"] += 1; s["events"].add(r["event"])
    for (ts, slot), s in sorted(snaps.items()):
        print(f"{ts} {slot:5s} states={len(s['states'])} markets={s['n']} events={len(s['events'])}")
    print(f"{len(snaps)} snapshots, {OUT.stat().st_size} bytes {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("snap"); a.add_argument("--slot", default="manual")
    sp.add_parser("check")
    x = ap.parse_args()
    sys.exit(snap(x.slot) if x.cmd == "snap" else check())
