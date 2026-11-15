"""READ-ONLY order-book logger for the Kalshi KX*EVENT macro-event ladders (notes/ops_econ_event_logger.md;
proposal 6 of notes/econ_event_ladders.md). Order-book state around a release is not retroactive, so it is logged now.

    python scripts/econ_event_books.py tick              # launchd every 60 s: refresh schedule (<= every 6 h), snap due slots
    python scripts/econ_event_books.py schedule          # print upcoming releases / slot times (refreshes the cache)
    python scripts/econ_event_books.py snap --event EV [--slot manual] [--deribit]   # one manual snapshot -> *_manual files
    python scripts/econ_event_books.py check             # summarise what has been logged

Timing. obs = event strike_date (the settlement observation time). release = obs - offset, offset parsed from the event
title/sub_title ('N hour(s)/min after' explicit; RATEEVENT series = 15 min (8:45 obs for an 8:30 print); named central-bank
decisions -> their fixed ET clock time; US data prints (CPI, PPI, jobs, claims, retail, PCE, GDP, ADP, JOLTS, ...) = 60 min;
anything else -> 60 min with release_src='default' -- obs is logged too, so release can be recomputed later).
Slots per event: pre = release - 5 min, p5 = release + 5 min, p30 = release + 30 min. A slot is taken by the first tick in
[slot, slot + 10 min) and only if slot < obs (books close at obs). Events whose slots fall together are snapped together.

Per snapshot per event: GET /markets?event_ticker= (top of book, sizes, volume, OI) then GET /markets/orderbooks in
batches of 25 (full depth; rows keep top-3 levels per side + full book). At p5, for releases with a KXBTCEVENT/KXETHEVENT
event, Deribit BTC/ETH option trades in [t - 60 min, t) + index price are fetched FIRST (OT1 ordering: forecaster inputs
strictly older than the Kalshi quote; deribit_trades() reused from scripts/ot1_forward.py).
All Kalshi GETs go through kx_throttle (shared <= ~3.3 req/s slot). No keys, no orders, no portfolio endpoints.

Output (append-only; one complete gzip member per write; gzip.open reads them all):
  ~/.claude/quant-sports/data/kx_forward/econ_events/books.jsonl.gz    one row per market per slot
  ~/.claude/quant-sports/data/kx_forward/econ_events/deribit.jsonl.gz  one row per currency per p5 slot (trades + index)
  ~/.claude/quant-sports/data/kx_forward/econ_events/schedule.jsonl.gz one row per event each time the schedule refreshes
Manual runs write *_manual.jsonl.gz so the scheduled files only ever hold on-time slots.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402

ET = ZoneInfo("America/New_York")
QS = Path.home() / ".claude" / "quant-sports"
OUT = QS / "data" / "kx_forward" / "econ_events"
STATE = QS / "state"
SCHED = STATE / "econ_events_schedule.json"
DONE = STATE / "econ_events_done.json"
FAM = ["KXGOLDEVENT", "KXSILVEREVENT", "KXCOPPEREVENT", "KXBTCEVENT", "KXETHEVENT", "KXSOLEVENT",
       "KXEURUSDEVENT", "KXGBPUSDEVENT", "KXUSDJPYEVENT", "KXAUDUSDEVENT", "KXUSDCADEVENT", "KXUSDMXNEVENT",
       "KX2YRRATEEVENT", "KX5YRRATEEVENT", "KX10YRRATEEVENT", "KX30YRRATEEVENT"]
SLOTS = {"pre": -5, "p5": 5, "p30": 30}          # minutes relative to release
GRACE_MIN = 10
SCHED_MAX_AGE = 6 * 3600
BATCH = 25
DERIBIT = {"KXBTCEVENT": "BTC", "KXETHEVENT": "ETH"}
# named decisions -> release clock time (ET) on the obs date; used only when no explicit 'N after' offset is given
FIXED = [(r"\bfed\b|fomc", (14, 0)), (r"\becb\b", (8, 15)), (r"\bboe\b|bank of england", (7, 0)),
         (r"\bboc\b|bank of canada", (9, 45)), (r"banxico", (14, 0))]
US_DATA = r"cpi|ppi|jobs|payroll|claims|retail|pce|gdp|adp|jolts|ism|sentiment|confidence|housing|durable"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_of(series: str, title: str, sub: str, obs: datetime) -> tuple[datetime, str]:
    txt = f"{title} · {sub}".lower()
    m = re.search(r"(\d+|an|one)\s*hours?\s+after", txt)
    if m:
        n = 1 if m.group(1) in ("an", "one") else int(m.group(1))
        return obs - timedelta(hours=n), f"explicit_{n}h"
    m = re.search(r"(\d+)\s*min(?:ute)?s?\s+after", txt)
    if m:
        return obs - timedelta(minutes=int(m.group(1))), f"explicit_{m.group(1)}m"
    if re.search(r"\bauction\b", txt) and "RATEEVENT" not in series:   # Treasury auctions print at 1:00pm ET
        return obs - timedelta(minutes=30), "auction_30m"
    if "RATEEVENT" in series:
        return obs - timedelta(minutes=15), "rates_15m"
    o = obs.astimezone(ET)
    for pat, (h, mi) in FIXED:
        if re.search(pat, txt):
            r = o.replace(hour=h, minute=mi, second=0, microsecond=0)
            if r < o:
                return r.astimezone(timezone.utc), f"fixed_{h:02d}{mi:02d}ET"
    if re.search(US_DATA, txt):
        return obs - timedelta(minutes=60), "us_data_60m"
    return obs - timedelta(minutes=60), "default"


def append_gz(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        g.write("".join(json.dumps(r, separators=(",", ":"), default=str) + "\n" for r in rows).encode())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(buf.getvalue())
    return len(buf.getvalue())


def load(p: Path, default):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return default


def save(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


# ------------------------------------------------------------------ schedule
def refresh_schedule(manual: bool = False) -> dict:
    t = now_utc()
    evs, rows = {}, []
    for s in FAM:
        cur = None
        while True:
            p = {"series_ticker": s, "status": "open", "limit": 200}
            if cur:
                p["cursor"] = cur
            d = kget("/events", p) or {}
            for e in d.get("events") or []:
                if not e.get("strike_date"):
                    continue
                obs = parse_ts(e["strike_date"])
                rel, src = release_of(s, e.get("title") or "", e.get("sub_title") or "", obs)
                ev = {"event": e["event_ticker"], "series": s, "obs": iso(obs), "release": iso(rel), "release_src": src,
                      "title": e.get("title"), "sub_title": e.get("sub_title")}
                evs[e["event_ticker"]] = ev
                rows.append({"ts": iso(t), **ev})
            cur = d.get("cursor")
            if not cur:
                break
    sched = {"ts": t.timestamp(), "events": evs}
    if not manual:
        save(SCHED, sched)
        append_gz(OUT / "schedule.jsonl.gz", rows)
    return sched


def get_schedule() -> dict:
    sc = load(SCHED, None)
    if not sc or time.time() - sc.get("ts", 0) > SCHED_MAX_AGE:
        sc = refresh_schedule()
    return sc


def slot_time(ev: dict, slot: str) -> datetime:
    return parse_ts(ev["release"]) + timedelta(minutes=SLOTS[slot])


# ------------------------------------------------------------------ snapshots
def _book(ob: dict, side: str) -> list:
    return [[float(p), float(q)] for p, q in (ob.get(f"{side}_dollars") or [])]


def snap_event(ev: dict, slot: str, ts: str) -> list[dict]:
    d = kget("/markets", {"event_ticker": ev["event"], "limit": 200}) or {}
    mkts = d.get("markets") or []
    books, nobook = {}, 0
    tick = [m["ticker"] for m in mkts]
    for i in range(0, len(tick), BATCH):
        b = kget("/markets/orderbooks", {"tickers": tick[i:i + BATCH]}) or {}
        for ob in b.get("orderbooks") or []:
            books[ob["ticker"]] = ob.get("orderbook_fp") or {}
    t_book = iso(now_utc())
    rows = []
    for m in mkts:
        ob = books.get(m["ticker"])
        nobook += ob is None
        yb, nb = _book(ob or {}, "yes"), _book(ob or {}, "no")
        rows.append({
            "ts": ts, "t_book": t_book, "slot": slot, "series": ev["series"], "event": ev["event"],
            "obs": ev["obs"], "release": ev["release"], "release_src": ev["release_src"],
            "ticker": m["ticker"], "status": m.get("status"), "strike_type": m.get("strike_type"),
            "floor": m.get("floor_strike"), "cap": m.get("cap_strike"), "close_time": m.get("close_time"),
            "yes_bid": m.get("yes_bid_dollars"), "yes_ask": m.get("yes_ask_dollars"),
            "yes_bid_size": m.get("yes_bid_size_fp"), "yes_ask_size": m.get("yes_ask_size_fp"),
            "no_bid": m.get("no_bid_dollars"), "no_ask": m.get("no_ask_dollars"),
            "last": m.get("last_price_dollars"), "volume": m.get("volume_fp"), "oi": m.get("open_interest_fp"),
            "book_ok": ob is not None, "yes_top3": yb[::-1][:3], "no_top3": nb[::-1][:3],
            "yes_book": yb, "no_book": nb,
        })
    if not mkts:
        rows.append({"ts": ts, "slot": slot, "series": ev["series"], "event": ev["event"], "obs": ev["obs"],
                     "release": ev["release"], "release_src": ev["release_src"], "ticker": None, "note": "no markets"})
    return rows


def deribit_snap(curs: list[str], slot: str, ts_dt: datetime) -> list[dict]:
    rows = []
    try:
        from ot1_forward import deribit_trades, _S  # noqa: E402  (same public-API code as the OT1 forward paper)
    except Exception as e:  # noqa: BLE001
        return [{"ts": iso(ts_dt), "slot": slot, "currency": c, "error": f"import: {e!r}"} for c in curs]
    t1 = int(ts_dt.timestamp() * 1000)
    for c in curs:
        r = {"ts": iso(ts_dt), "slot": slot, "currency": c, "window_ms": [t1 - 3_600_000, t1]}
        try:
            ix = _S.get("https://www.deribit.com/api/v2/public/get_index_price",
                        params={"index_name": f"{c.lower()}_usd"}, timeout=20).json()["result"]
            r["index_price"], r["t_index"] = ix.get("index_price"), iso(now_utc())
        except Exception as e:  # noqa: BLE001
            r["index_error"] = repr(e)
        try:
            tr, host = deribit_trades(c, t1 - 3_600_000, t1)
            r["host"], r["n_trades"] = host, len(tr)
            r["trades"] = [{k: x.get(k) for k in ("timestamp", "trade_id", "instrument_name", "price", "iv", "amount",
                                                   "direction", "index_price", "mark_price")} for x in tr]
        except Exception as e:  # noqa: BLE001
            r["error"] = repr(e)
        rows.append(r)
    return rows


def run_slot(evs: list[dict], slot: str, suffix: str = "", deribit: bool | None = None) -> str:
    t0 = now_utc()
    curs = sorted({DERIBIT[e["series"]] for e in evs if e["series"] in DERIBIT})
    if deribit is None:
        deribit = slot == "p5"
    nb_d = 0
    if deribit and curs:            # forecaster inputs first, strictly older than the Kalshi quotes
        drows = deribit_snap(curs, slot, t0)
        nb_d = append_gz(OUT / f"deribit{suffix}.jsonl.gz", drows)
        dmsg = " ".join(f"{r['currency']}:{r.get('n_trades', 'ERR')}tr/ix={r.get('index_price')}" for r in drows)
    else:
        dmsg = "-"
    ts = iso(now_utc())
    rows = []
    for ev in evs:
        rows += snap_event(ev, slot, ts)
    nb = append_gz(OUT / f"books{suffix}.jsonl.gz", rows)
    mk = [r for r in rows if r.get("ticker")]
    two = sum(1 for r in mk if (r.get("yes_top3") and r.get("no_top3")))
    return (f"{ts} slot={slot} events={len(evs)} markets={len(mk)} two_sided={two} "
            f"no_book={sum(1 for r in mk if not r.get('book_ok'))} deribit[{dmsg}] bytes+={nb + nb_d} "
            f"{(now_utc() - t0).total_seconds():.0f}s")


def tick() -> int:
    sc = get_schedule()
    done = load(DONE, {})
    t = now_utc()
    first = parse_ts(done.setdefault("_first_tick", iso(t)))   # slots before the logger existed are never 'missed'
    due: dict[str, list] = {}
    for ev in sc["events"].values():
        obs = parse_ts(ev["obs"])
        for slot in SLOTS:
            st = slot_time(ev, slot)
            key = f"{ev['event']}|{slot}"
            if key in done or st >= obs:
                continue
            if st <= t < st + timedelta(minutes=GRACE_MIN):
                due.setdefault(slot, []).append(ev)
    save(DONE, done)
    rc = 0
    for slot, evs in due.items():
        try:
            print(run_slot(evs, slot))
            for ev in evs:
                done[f"{ev['event']}|{slot}"] = iso(t)
            save(DONE, done)
        except Exception as e:  # noqa: BLE001
            print(f"{iso(now_utc())} slot={slot} ERROR {e!r}")
            rc = 1
    # missed slots (Mac asleep): record once so the gap is visible in the log
    for ev in sc["events"].values():
        obs = parse_ts(ev["obs"])
        for slot in SLOTS:
            st, key = slot_time(ev, slot), f"{ev['event']}|{slot}"
            if key not in done and first <= st < obs and t >= st + timedelta(minutes=GRACE_MIN):
                done[key] = "missed"
                print(f"{iso(t)} MISSED {key} (slot {iso(st)})")
                save(DONE, done)
    return rc


def schedule_cmd() -> int:
    sc = refresh_schedule()
    t = now_utc()
    for ev in sorted(sc["events"].values(), key=lambda e: e["release"]):
        if parse_ts(ev["obs"]) < t:
            continue
        sl = " ".join(f"{s}={slot_time(ev, s).astimezone(ET):%m-%d %H:%M}" for s in SLOTS if slot_time(ev, s) < parse_ts(ev["obs"]))
        print(f"{ev['event']:34s} obs={parse_ts(ev['obs']).astimezone(ET):%m-%d %H:%M}ET {ev['release_src']:14s} {sl} | {ev['sub_title']}")
    print(f"{len(sc['events'])} open events")
    return 0


def snap_cmd(event: str, slot: str, deribit: bool) -> int:
    sc = load(SCHED, None) or refresh_schedule(manual=True)
    ev = sc["events"].get(event)
    if not ev:
        print("event not in schedule:", event)
        return 2
    print(run_slot([ev], slot, suffix="_manual", deribit=deribit))
    return 0


def check() -> int:
    for name in ("books", "deribit", "books_manual", "deribit_manual"):
        p = OUT / f"{name}.jsonl.gz"
        if not p.exists():
            print(f"{name}: no file")
            continue
        agg: dict = {}
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                k = (r["ts"], r["slot"])
                a = agg.setdefault(k, {"n": 0, "ev": set(), "two": 0, "tr": 0})
                a["n"] += 1
                a["ev"].add(r.get("event") or r.get("currency"))
                a["two"] += bool(r.get("yes_top3") and r.get("no_top3"))
                a["tr"] += r.get("n_trades") or 0
        for (ts, slot), a in sorted(agg.items()):
            print(f"{name:15s} {ts} {slot:6s} rows={a['n']} events={len(a['ev'])} two_sided={a['two']} trades={a['tr']}")
        print(f"{name}: {len(agg)} snapshots, {p.stat().st_size} bytes {p}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("tick")
    sp.add_parser("schedule")
    a = sp.add_parser("snap"); a.add_argument("--event", required=True); a.add_argument("--slot", default="manual")
    a.add_argument("--deribit", action="store_true")
    sp.add_parser("check")
    x = ap.parse_args()
    if x.cmd == "tick":
        sys.exit(tick())
    if x.cmd == "schedule":
        sys.exit(schedule_cmd())
    if x.cmd == "snap":
        sys.exit(snap_cmd(x.event, x.slot, x.deribit))
    sys.exit(check())
