"""QM1 / QM2 KXBIGGESTQUAKEMONTH forward paper logger (READ-ONLY; prereg notes/quake_monthly.md, ops notes there under 'Ops').

    python scripts/quake_monthly_logger.py tick [--force-now]   # launchd every 1800 s (job quake-monthly via qs_job.sh)
    python scripts/quake_monthly_logger.py settle               # fills for QM2 posts + market results (tick runs it hourly)
    python scripts/quake_monthly_logger.py status

Each tick (order fixed: every USGS read happens BEFORE any Kalshi GET of the same snapshot):
  1. USGS ComCat M>=6 worldwide over the last 120 d (1 GET) -> merged into the frozen base catalog
     results/quake_monthly/usgs_m6.parquet (by event id, current values). Month-to-date max (mag, magType, status, id).
  2. Snapshot slots (one each; nothing else touches the network on idle ticks except the USGS GET):
       daily   first tick of the UTC day at/after 15:00 UTC. DECISION slot iff the tick starts in [15:00, 15:45) UTC;
               a later first tick still snapshots (status decision_missed, no picks/posts).
       late    first tick at/after 23:00 UTC on the month's last 5 UTC days (descriptive snapshot only).
       quake   any USGS M>=6.5 whose origin is <= 6 h old and not yet snapshotted (captures maker repricing; descriptive).
     A snapshot = /events?series_ticker=KXBIGGESTQUAKEMONTH&status=open (nested markets) + per market full /orderbook +
     /markets/trades since that ticker's previous snapshot; frozen model P per rung (results/quake_monthly/scripts/model.py,
     p_model unchanged; RUNGS set to the event's floor strikes).
  3. Decision (daily slot in window, guard not triggered): QM1 pick per (ticker, side) with p_S >= .85 and
     p_S - ask_S - .07 ask_S (1 - ask_S) >= .03 (first fire per ticker-side, 1 contract). QM2 post per (ticker, side)
     with p_S >= .85: q = best bid_S + .01, needs q < ask_S and p_S - q >= .03 (every decision day; 1 contract each),
     valid until the next day's 15:00 UTC (or market close).
  4. settle (at most hourly): a post whose window ended gets a `fill` row from the public tape: filled iff a print in
     (posted, valid_until] with taker_side opposite to S and S-price <= q (fill_lt = strictly below q, also logged).
     Markets settled on Kalshi -> `settle` rows (result, expiration_value, USGS month max at settle time).
Output (append-only) ~/.claude/quant-sports/data/kx_forward/quake_monthly.jsonl:
  day | pick (QM1) | post (QM2) | fill | settle  rows;  quake_monthly_snaps.jsonl: one row per snapshot (books, trades, p).
--force-now: snapshot + decision now, written to *_manual.jsonl only. No orders, no keys, no portfolio endpoints.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                            # sports/ or ~/.claude/quant-sports
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402

SERIES = "KXBIGGESTQUAKEMONTH"
QDIR = ROOT / "results" / "quake_monthly"
OUT = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"
STATE = Path.home() / ".claude" / "quant-sports" / "state"
DECIDE_H, DECIDE_WINDOW_MIN, LATE_H, BIG = 15, 45, 23, 6.5
P_MIN, MARGIN = 0.85, 0.03
SUFFIX = ""
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
_S = requests.Session()
_S.headers["User-Agent"] = "quant-sports-readonly-paper"
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


def load_model():
    spec = importlib.util.spec_from_file_location("quake_month_model", QDIR / "scripts" / "model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def path(name: str) -> Path:
    return OUT / f"{name}{SUFFIX}.jsonl"


def _js(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    raise TypeError(type(x))


def append(p: Path, rows: list[dict]) -> None:
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_js) + "\n")


def read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def sget(name: str, default):
    try:
        return json.loads((STATE / f"quake_monthly_{name}{SUFFIX}.json").read_text())
    except Exception:  # noqa: BLE001
        return default


def sput(name: str, v) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"quake_monthly_{name}{SUFFIX}.json").write_text(json.dumps(v))


# ------------------------------------------------------------------ USGS
def usgs_recent(now: float, days: int = 120) -> pd.DataFrame:
    for k in range(4):
        try:
            r = _S.get(USGS, params={"format": "csv", "starttime": iso(now - days * 86400)[:19], "minmagnitude": 6.0,
                                     "eventtype": "earthquake", "orderby": "time-asc", "limit": 20000}, timeout=60)
            if r.status_code == 200:
                d = pd.read_csv(io.StringIO(r.text))
                d["time"] = pd.to_datetime(d["time"], utc=True, format="ISO8601")
                return d
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3 * (k + 1))
    raise RuntimeError("USGS fetch failed")


def catalog(recent: pd.DataFrame, base_cat: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(QDIR / "usgs_m6.parquet")
    c = pd.concat([base, recent]).drop_duplicates("id", keep="last")   # recent = current USGS values
    return c.sort_values("time").reset_index(drop=True)


def month_max(cat: pd.DataFrame, ms: pd.Timestamp, me: pd.Timestamp, t: pd.Timestamp) -> dict:
    m = cat[(cat.time >= ms) & (cat.time < me) & (cat.time <= t)]
    if m.empty:
        return {"mag": None, "note": "no M>=6.0 this month"}
    r = m.loc[m.mag.idxmax()]
    return {"mag": float(r.mag), "magType": r.get("magType"), "status": r.get("status"), "id": r.get("id"),
            "time": r.time.isoformat(), "place": r.get("place"), "updated": r.get("updated")}


# ------------------------------------------------------------------ Kalshi (public, throttled)
def month_of(event_ticker: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})$", event_ticker)
    if not m:
        raise ValueError(f"cannot parse month from {event_ticker}")
    ms = pd.Timestamp(year=2000 + int(m.group(3)), month=MON[m.group(2)], day=1, tz="UTC")
    return ms, ms + pd.offsets.MonthBegin(1)


def book(t: str) -> dict:
    ob = (kget(f"/markets/{t}/orderbook") or {})
    fp = ob.get("orderbook_fp") or {}
    lv = lambda k: sorted([(float(p), float(s)) for p, s in (fp.get(k) or [])], key=lambda x: -x[0])  # noqa: E731
    yes, no = lv("yes_dollars"), lv("no_dollars")
    by = next(((p, s) for p, s in yes if s >= 1), (None, 0.0))
    bn = next(((p, s) for p, s in no if s >= 1), (None, 0.0))
    return {"yes": yes, "no": no, "yes_bid": by[0], "yes_bid_sz": by[1], "no_bid": bn[0], "no_bid_sz": bn[1],
            "yes_ask": None if bn[0] is None else round(1 - bn[0], 4), "no_ask": None if by[0] is None else round(1 - by[0], 4)}


def trades(t: str, t0: float | None, t1: float | None = None) -> list[dict]:
    out, cur = [], None
    while True:
        p = {"ticker": t, "limit": 1000}
        if t0:
            p["min_ts"] = int(t0)
        if t1:
            p["max_ts"] = int(t1)
        if cur:
            p["cursor"] = cur
        d = kget("/markets/trades", p) or {}
        out += d.get("trades", [])
        cur = d.get("cursor")
        if not cur:
            break
    keep = ("created_time", "taker_side", "yes_price_dollars", "no_price_dollars", "count_fp", "trade_id")
    return [{k: x.get(k) for k in keep} for x in out]


def fee(a: float) -> float:
    return 0.07 * a * (1 - a)


# ------------------------------------------------------------------ snapshot + decision
def snapshot(now: float, slot: str, cat: pd.DataFrame, usgs_t: float, decide: bool, M) -> dict:
    t = pd.Timestamp(now, unit="s", tz="UTC")
    k_start = time.time()
    evs = (kget("/events", {"series_ticker": SERIES, "status": "open", "with_nested_markets": "true"}) or {}).get("events", [])
    last_ts = sget("last_trade_ts", {})
    snap = {"kind": "snap", "slot": slot, "t": iso(now), "usgs_t": iso(usgs_t), "kalshi_start": iso(k_start), "events": []}
    picks, posts = [], []
    prior = {(r["ticker"], r["side"]) for r in read(path("quake_monthly")) if r.get("kind") == "pick"}
    for ev in evs:
        et = ev["event_ticker"]
        ms, me = month_of(et)
        mks = sorted([m for m in ev.get("markets", []) if m.get("floor_strike") is not None], key=lambda m: m["floor_strike"])
        M.RUNGS = [float(m["floor_strike"]) for m in mks]
        pm = M.p_model(t, ms, me, catalog=cat[["time", "mag"]])
        e = {"event": et, "month_start": ms.isoformat(), "month_end": me.isoformat(), "T_days": pm["T_days"],
             "obs_max": pm["obs_max"], "guard_skip": pm["guard_skip"], "n_windows": pm["n_windows"],
             "usgs_month_max": month_max(cat, ms, me, t), "markets": []}
        for m in mks:
            tk, K = m["ticker"], float(m["floor_strike"])
            b = book(tk)
            tr = trades(tk, last_ts.get(tk))
            last_ts[tk] = k_start
            py = float(pm["p"][K])
            row = {"ticker": tk, "K": K, "status": m.get("status"), "close_time": m.get("close_time"),
                   "volume": m.get("volume_fp"), "oi": m.get("open_interest_fp"), "p_yes": py, **b,
                   "trades_since_last": tr, "book_t": iso(time.time())}
            e["markets"].append(row)
            if not decide or pm["guard_skip"] or m.get("status") != "active":
                continue
            for side, ps, ask, bid in (("yes", py, b["yes_ask"], b["yes_bid"]), ("no", 1 - py, b["no_ask"], b["no_bid"])):
                if ps < P_MIN:
                    continue
                base = {"date": t.strftime("%Y-%m-%d"), "event": et, "ticker": tk, "K": K, "side": side, "p_side": ps,
                        "p_yes": py, "T_days": pm["T_days"], "obs_max": pm["obs_max"], "yes_bid": b["yes_bid"],
                        "yes_ask": b["yes_ask"], "decided_at": iso(time.time()), "inputs_as_of": iso(usgs_t),
                        "quote_time": row["book_t"], "month_end": me.isoformat(), "cluster": f"quake_month|{ms:%Y-%m}"}
                if ask is not None and ps - ask - fee(ask) >= MARGIN and (tk, side) not in prior:
                    picks.append({**base, "kind": "pick", "rule": "QM1", "ask": ask, "fee": fee(ask),
                                  "edge": ps - ask - fee(ask), "status": "bet"})
                    prior.add((tk, side))
                q = None if bid is None else round(bid + 0.01, 2)
                if q is not None and ask is not None and q < ask - 1e-9 and ps - q >= MARGIN:
                    vu = min(pd.Timestamp(t.strftime("%Y-%m-%d"), tz="UTC") + pd.Timedelta(days=1, hours=DECIDE_H),
                             pd.Timestamp(m["close_time"]))
                    posts.append({**base, "kind": "post", "rule": "QM2", "post_id": f"{tk}|{side}|{base['date']}",
                                  "q": q, "best_bid": bid, "ask": ask, "edge": ps - q, "valid_until": vu.isoformat()})
        snap["events"].append(e)
    sput("last_trade_ts", last_ts)
    append(path("quake_monthly_snaps"), [snap])
    append(path("quake_monthly"), picks + posts)
    return {"snap": snap, "picks": picks, "posts": posts}


def summary(s: dict) -> str:
    out = []
    for e in s["snap"]["events"]:
        mm = e["usgs_month_max"]
        out.append(f"{e['event']} T={e['T_days']:.2f}d obs_max={mm.get('mag')} {mm.get('magType') or ''} guard={e['guard_skip']}")
        for r in e["markets"]:
            out.append(f"  {r['K']:.1f} p={r['p_yes']:.3f} yes {r['yes_bid']}/{r['yes_ask']} (sz {r['yes_bid_sz']:.0f}/"
                       f"{r['no_bid_sz']:.0f}) trades+{len(r['trades_since_last'])}")
    out.append(f"QM1 picks {len(s['picks'])}: " + " ".join(f"{p['ticker'].split('-')[-1]}:{p['side']}@{p['ask']}" for p in s["picks"]))
    out.append(f"QM2 posts {len(s['posts'])}: " + " ".join(f"{p['ticker'].split('-')[-1]}:{p['side']}@{p['q']}" for p in s["posts"]))
    return "\n".join(out)


# ------------------------------------------------------------------ settle
def settle(now: float, cat: pd.DataFrame | None) -> list[str]:
    rows = read(path("quake_monthly"))
    msgs, new = [], []
    filled = {r["post_id"] for r in rows if r.get("kind") == "fill"}
    for p in [r for r in rows if r.get("kind") == "post" and r["post_id"] not in filled]:
        vu = pd.Timestamp(p["valid_until"]).timestamp()
        if now < vu + 120:
            continue
        t0 = pd.Timestamp(p["decided_at"]).timestamp()
        tr = sorted(trades(p["ticker"], t0, vu), key=lambda x: x["created_time"])
        opp = "yes" if p["side"] == "no" else "no"
        px = f"{p['side']}_price_dollars"
        cross = [x for x in tr if x["taker_side"] == opp and pd.Timestamp(x["created_time"]).timestamp() > t0
                 and float(x[px]) <= p["q"] + 1e-9]
        lt = [x for x in cross if float(x[px]) < p["q"] - 1e-9]
        new.append({"kind": "fill", "post_id": p["post_id"], "ticker": p["ticker"], "side": p["side"], "q": p["q"],
                    "filled": bool(cross), "fill_ts": cross[0]["created_time"] if cross else None,
                    "fill_lt": bool(lt), "n_trades_window": len(tr), "n_cross": len(cross),
                    "cross_contracts": sum(float(x["count_fp"]) for x in cross), "checked_at": iso(time.time())})
        msgs.append(f"fill {p['post_id']} q={p['q']} filled={bool(cross)}")
    done = {r["ticker"] for r in rows if r.get("kind") == "settle"}
    need = sorted({r["ticker"] for r in rows if r.get("kind") in ("pick", "post")} - done)
    for tk in need:
        _, me = month_of(tk.rsplit("-", 1)[0])
        if now < me.timestamp():
            continue
        m = (kget(f"/markets/{tk}") or {}).get("market") or {}
        if m.get("result") in ("yes", "no"):
            ms = me - pd.offsets.MonthBegin(1)
            mm = month_max(cat, ms, me, pd.Timestamp(now, unit="s", tz="UTC")) if cat is not None else None
            new.append({"kind": "settle", "ticker": tk, "result": m["result"], "expiration_value": m.get("expiration_value"),
                        "status_mkt": m.get("status"), "usgs_month_max_now": mm, "settled_seen": iso(time.time())})
            msgs.append(f"settle {tk} {m['result']} ({m.get('expiration_value')})")
    append(path("quake_monthly"), new)
    return msgs


# ------------------------------------------------------------------ tick
def tick(force: bool) -> int:
    M = load_model()
    now = time.time()
    t = pd.Timestamp(now, unit="s", tz="UTC")
    rec = usgs_recent(now)
    usgs_t = time.time()
    cat = catalog(rec, M.cat)
    uday = t.strftime("%Y-%m-%d")
    slots = []
    if force:
        slots.append(("manual", True))
    else:
        if t.hour >= DECIDE_H and sget("daily", "") != uday:
            late_min = (now - pd.Timestamp(uday, tz="UTC").timestamp() - DECIDE_H * 3600) / 60
            slots.append(("daily", late_min < DECIDE_WINDOW_MIN))
        me = t.normalize() + pd.offsets.MonthBegin(1)
        if t.hour >= LATE_H and (me - t.normalize()).days <= 5 and sget("late", "") != uday:
            slots.append(("late", False))
        seen = set(sget("big_seen", []))
        big = rec[(rec.mag >= BIG) & (rec.time >= t - pd.Timedelta(hours=6)) & (~rec.id.isin(seen))]
        if len(big):
            slots.append(("quake:" + ",".join(big.id), False))
    for slot, decide in slots:
        s = snapshot(now, slot, cat, usgs_t, decide, M)
        day = {"kind": "day", "slot": slot, "date": uday, "t": iso(now), "decision": decide,
               "status": "decided" if decide else ("decision_missed" if slot == "daily" else "descriptive"),
               "n_picks": len(s["picks"]), "n_posts": len(s["posts"]),
               "events": [{k: e[k] for k in ("event", "T_days", "obs_max", "guard_skip", "usgs_month_max")} for e in s["snap"]["events"]]}
        if t.day <= 7:                                   # USGS value of LAST month after its 12-01 10:00 ET print (revisions)
            pe = t.normalize().replace(day=1)
            day["usgs_prev_month_max"] = month_max(cat, pe - pd.offsets.MonthBegin(1), pe, t)
        append(path("quake_monthly"), [day])
        print(f"=== {iso(now)} slot {slot} decision={decide}\n{summary(s)}")
        if slot == "daily":
            sput("daily", uday)
        elif slot == "late":
            sput("late", uday)
        elif slot.startswith("quake:"):
            sput("big_seen", sorted(set(sget("big_seen", [])) | set(slot[6:].split(","))))
    last = sget("settle_last", 0)
    if force or now - last >= 3600:
        for msg in settle(now, cat):
            print(msg)
        sput("settle_last", now)
    return 0


def status() -> int:
    rows = read(path("quake_monthly"))
    kinds = pd.Series([r.get("kind") for r in rows]).value_counts().to_dict() if rows else {}
    days = [r for r in rows if r.get("kind") == "day"]
    print(f"{path('quake_monthly')}: {kinds}")
    for d in days[-5:]:
        print(" ", d["t"], d["slot"], d["status"], "picks", d["n_picks"], "posts", d["n_posts"],
              [(e["event"], e["usgs_month_max"].get("mag")) for e in d["events"]])
    print(f"snaps: {len(read(path('quake_monthly_snaps')))}")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:] or ["tick"]
    if "--force-now" in a:
        SUFFIX = "_manual"
    if a[0] == "tick":
        sys.exit(tick("--force-now" in a))
    if a[0] == "settle":
        M_ = load_model()
        sys.exit(0 if print("\n".join(settle(time.time(), catalog(usgs_recent(time.time()), M_.cat)))) is None else 0)
    if a[0] == "status":
        sys.exit(status())
    print(__doc__)
    sys.exit(2)
