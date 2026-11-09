"""READ-ONLY measurement of the real queue behind trial lock_t2_nba_maker99 (scripts/lock_in.py, edgelab/rules/lock_in.py).
No orders, no keys, public data only. Measurement only: not an edgelab trial, never reads the registry or the sealed window.

    python scripts/lock_queue_logger.py tick      # launchd quant-sports-lock-queue, every 60 s (notes/HARNESS.md)
    python scripts/lock_queue_logger.py report [--all] [--file PATH]

Why: T2 posts a 99c YES bid on NBA markets whose outcome is already locked. Dev fills depend wholly on queue position and
N_QUEUE = 5000 was a guess. This logs what a bid joining at t would actually have stood behind.

Locks (ESPN public scoreboard):
  moneyline  ESPN game status STATUS_FINAL + completed, ticker KXNBAGAME-<event>-<winner>. NOT the last play (3/78
             sampled games had a bad "final" row), so only the game's own FINAL status counts.
  over rung  KXNBATOTAL strike_type "greater" with score sum >= K + 3.5 (one reversed 3 cannot undo it).
late_detect = the game was first observed near its end with the rung already over (true lock time unknown, earlier).

Snapshots at lock + 0/15/30/60/120/300 s: resting YES bids at 99c and 98c, best bid, best ask (= 1 - best NO bid), market
status, and every public trade since lock - 120 s not already logged. Tape-only rows at +900 and +2400 s so later prints
are visible to a joiner. A market that is no longer active stops book snapshots (tape rows continue).

Cost control: launchd fires every 60 s. Outside [start + 75 min, start + 6 h] of the cached schedule the run makes no
network call. Inside it, one scoreboard call; only if a game is in period >= 4 does the run stay up, polling ESPN every
15 s (<= 55 min per process, state kept across runs). Idle runs print nothing.

Output: data/live/lock_queue.jsonl (kind lock | snapshot | tape), data/live/lock_queue_state.json.
Timestamps: decided_ts = when this logger saw the lock; as_of_ts = receipt of the book; trades carry Kalshi created_time.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KAPI = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
LIVE = ROOT / "data" / "live"
OUT = LIVE / "lock_queue.jsonl"
STATE = LIVE / "lock_queue_state.json"
ET = ZoneInfo("America/New_York")
KCODE = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}   # same map as lock_in.py
MON = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
SNAP_S = (0, 15, 30, 60, 120, 300)
TAPE_S = (900, 2400)
MARGIN = 3.5
POLL_S, MAX_RUN_S, SCHED_S = 15, 3300, 1800
WIN_LO, WIN_HI = 75 * 60, 6 * 3600
LOOKBACK_S = 120
KEEP_S = 3 * 86400


def _get(url, params=None):
    from sports import http as H
    H.MIN_INTERVAL = 0.06
    return H.fetch(url, params, cache=False, retries=3, timeout=10.0)


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds")


def epoch(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def load_state() -> dict:
    try:
        return json.loads(STATE.read_text()) if STATE.exists() else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st))
    os.replace(tmp, STATE)


def append(rows: list[dict]) -> None:
    if rows:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%SZ} {msg}", flush=True)


def k(x: str) -> str:
    return KCODE.get(x, x)


def parse_games(sb: dict) -> list[dict]:
    out = []
    for e in sb.get("events") or []:
        c = (e.get("competitions") or [{}])[0]
        side = {x.get("homeAway"): x for x in c.get("competitors") or []}
        if set(side) != {"home", "away"}:
            continue
        status = e.get("status") or {}
        typ = status.get("type") or {}
        start = datetime.fromisoformat(e["date"].replace("Z", "+00:00"))
        d = start.astimezone(ET)
        home, away = side["home"]["team"]["abbreviation"], side["away"]["team"]["abbreviation"]
        hs, as_ = int(side["home"].get("score") or 0), int(side["away"].get("score") or 0)
        final = typ.get("name") == "STATUS_FINAL" and bool(typ.get("completed")) and typ.get("state") == "post"
        winner = None
        if final and hs != as_:
            by_score = "home" if hs > as_ else "away"
            flags = [s for s in ("home", "away") if side[s].get("winner") is True]
            if not flags or flags == [by_score]:      # ESPN's winner flag, when present, must agree with the score
                winner = home if by_score == "home" else away
        out.append({"id": str(e["id"]), "start": int(start.timestamp()), "state": typ.get("state"), "final": final,
                    "period": int(status.get("period") or 0), "clock": status.get("displayClock"), "home": home,
                    "away": away, "hs": hs, "as": as_, "winner": winner, "espn_date": d.strftime("%Y%m%d"),
                    "ev": f"{d:%y}{MON[d.month - 1]}{d:%d}{k(away)}{k(home)}"})
    return out


def near_end(g: dict) -> bool:
    return g["state"] == "in" and g["period"] >= 4


def et_dates(now: float) -> list[str]:
    d = datetime.fromtimestamp(now, ET).date()
    return [(d - timedelta(days=1)).strftime("%Y%m%d"), d.strftime("%Y%m%d")]


def refresh_schedule(st: dict, get, now: float) -> None:
    sch = st.get("sched") or {}
    if sch and now - sch.get("fetched", 0) < SCHED_S:
        return
    games = {}
    for day in et_dates(now):
        for g in parse_games(get(ESPN, {"dates": day})):
            games[g["id"]] = {"start": g["start"], "espn_date": g["espn_date"]}
    st["sched"] = {"fetched": now, "games": games}
    upcoming = [g["start"] + WIN_LO for g in games.values() if g["start"] + WIN_HI >= now]
    log(f"schedule refreshed: {len(games)} games on {'/'.join(et_dates(now))}; "
        + (f"first watch window opens {iso(min(upcoming))}" if upcoming else "no window ahead"))


def in_window(st: dict, now: float) -> list[str]:
    games = (st.get("sched") or {}).get("games") or {}
    return sorted({g["espn_date"] for g in games.values() if g["start"] + WIN_LO <= now <= g["start"] + WIN_HI})


def event_markets(series: str, ev: str, get) -> list[dict]:
    r = get(f"{KAPI}/markets", {"event_ticker": f"{series}-{ev}", "limit": 200})
    return r.get("markets") or []


def new_lock(g: dict, ticker: str, series: str, ltype: str, K, late: bool, status, now: float) -> tuple[dict, dict]:
    row = {"kind": "lock", "decided_at": iso(now), "decided_ts": now, "ticker": ticker, "series": series, "lock_type": ltype,
           "K": K, "game_id": g["id"], "ev": g["ev"], "espn_date": g["espn_date"], "home": g["home"], "away": g["away"],
           "hs": g["hs"], "as": g["as"], "sum": g["hs"] + g["as"], "period": g["period"], "clock": g["clock"],
           "game_final": g["final"], "late_detect": late, "market_status_at_lock": status}
    lock = {"ticker": ticker, "series": series, "ltype": ltype, "K": K, "ts": now, "late": late, "game": g["id"], "ev": g["ev"],
            "snaps": list(SNAP_S), "tape": list(TAPE_S), "seen": []}
    return lock, row


def detect(st: dict, games: list[dict], get, now: float) -> list[dict]:
    locks, per_game, rows = st.setdefault("locks", {}), st.setdefault("games", {}), []
    for g in games:
        if not near_end(g) and not g["final"]:
            continue
        gs = per_game.get(g["id"])
        first = gs is None
        if first:
            gs = per_game[g["id"]] = {"first": now, "ev": g["ev"], "final_done": False}
        if gs["final_done"]:
            continue
        if "rungs" not in gs:
            ms = event_markets("KXNBATOTAL", g["ev"], get)
            gs["rungs"] = [[m["ticker"], float(m["floor_strike"]), m.get("status")] for m in ms
                           if m.get("strike_type") == "greater" and m.get("floor_strike") is not None]
            gs["ml"] = {m["ticker"]: m.get("status") for m in event_markets("KXNBAGAME", g["ev"], get)}
            if not gs["ml"]:
                log(f"WARN no KXNBAGAME markets for {g['ev']} (ESPN {g['away']}@{g['home']})")
            log(f"tracking {g['away']}@{g['home']} {g['ev']}: {len(gs['rungs'])} over rungs, {len(gs['ml'])} ml tickers")
        total = g["hs"] + g["as"]
        for tk, K, status in gs["rungs"]:
            if tk not in locks and total >= K + MARGIN:
                lock, row = new_lock(g, tk, "KXNBATOTAL", "over", K, first, status, now)
                locks[tk] = lock
                rows.append(row)
        if g["final"]:
            gs["final_done"] = True
            if g["winner"] is None:
                log(f"WARN final {g['ev']} with no clean winner (score {g['hs']}-{g['as']}); ml not logged")
                continue
            tk = f"KXNBAGAME-{g['ev']}-{k(g['winner'])}"
            if tk in gs["ml"]:
                lock, row = new_lock(g, tk, "KXNBAGAME", "final", None, first, gs["ml"][tk], now)
                locks[tk] = lock
                rows.append(row)
            else:
                log(f"WARN winner ticker {tk} not listed; have {sorted(gs['ml'])}")
    return rows


def book_summary(book: dict) -> dict:
    """Kalshi books hold bids only: YES bids and NO bids; a YES ask at p is a NO bid at 1 - p."""
    b = book.get("orderbook_fp") or {}
    yes = {round(float(p) * 100): float(n) for p, n in b.get("yes_dollars") or []}
    no = {round(float(p) * 100): float(n) for p, n in b.get("no_dollars") or []}
    bid = max(yes) if yes else None
    nb = max(no) if no else None
    return {"yes_99": yes.get(99, 0.0), "yes_98": yes.get(98, 0.0), "best_bid_c": bid, "best_bid_sz": yes.get(bid) if yes else None,
            "best_ask_c": (100 - nb) if no else None, "best_ask_sz": no.get(nb) if no else None, "no_1c": no.get(1, 0.0),
            "book_empty": not yes and not no}


def tape_new(lock: dict, get) -> list[list]:
    seen, out, cur = set(lock["seen"]), [], None
    for _ in range(20):
        p = {"ticker": lock["ticker"], "min_ts": int(lock["ts"]) - LOOKBACK_S, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = get(f"{KAPI}/markets/trades", p)
        for t in r.get("trades") or []:
            if t["trade_id"] in seen:
                continue
            seen.add(t["trade_id"])
            out.append([epoch(t["created_time"]), round(float(t["yes_price_dollars"]) * 100), float(t["count_fp"]),
                        t.get("taker_side"), t["trade_id"]])
        cur = r.get("cursor")
        if not cur:
            break
    lock["seen"] = sorted(seen)
    return sorted(out)


def take(lock: dict, delay: int, kind: str, get, clock) -> dict:
    tk = lock["ticker"]
    row = {"kind": kind, "ticker": tk, "series": lock["series"], "lock_type": lock["ltype"], "K": lock["K"], "game_id": lock["game"],
           "ev": lock["ev"], "delay_s": delay, "decided_at": iso(lock["ts"]), "decided_ts": lock["ts"], "late_detect": lock["late"]}
    try:
        if kind == "snapshot":
            t0 = clock()                               # the book first: its timing is the measurement
            book = get(f"{KAPI}/markets/{tk}/orderbook", {"depth": 0})
            as_of = clock()
            row.update(book_summary(book), as_of=iso(as_of), as_of_ts=as_of, book_latency_s=round(as_of - t0, 3),
                       lag_s=round(as_of - lock["ts"] - delay, 3))
            row["market_status"] = (get(f"{KAPI}/markets/{tk}").get("market") or {}).get("status")
            if row["market_status"] != "active":
                lock["snaps"] = []                     # a closed book holds no queue; only the tape rows remain
        else:
            row["as_of_ts"] = clock()
            row["as_of"] = iso(row["as_of_ts"])
        row["new_trades"] = tape_new(lock, get)
    except Exception as ex:  # noqa: BLE001
        row["error"] = f"{type(ex).__name__}: {ex}"[:200]
        row.setdefault("as_of_ts", clock())
        row.setdefault("as_of", iso(row["as_of_ts"]))
    return row


def due(st: dict, now: float) -> list[tuple[dict, int, str]]:
    out = []
    for lock in (st.get("locks") or {}).values():
        if lock["snaps"] and now >= lock["ts"] + lock["snaps"][0]:
            out.append((lock, lock["snaps"].pop(0), "snapshot"))
        elif lock["tape"] and now >= lock["ts"] + lock["tape"][0]:
            out.append((lock, lock["tape"].pop(0), "tape"))
    return out


def pending(st: dict) -> bool:
    return any(l["snaps"] or l["tape"] for l in (st.get("locks") or {}).values())


def books_owed(st: dict) -> bool:
    """A book snapshot is owed: stay up for those. Tape rows alone are picked up by later launchd runs."""
    return any(l["snaps"] for l in (st.get("locks") or {}).values())


def prune(st: dict, now: float) -> None:
    locks = st.get("locks") or {}
    for t in [t for t, l in locks.items() if not l["snaps"] and not l["tape"] and now - l["ts"] > KEEP_S]:
        del locks[t]
    gs = st.get("games") or {}
    for g in [g for g, v in gs.items() if now - v.get("first", now) > KEEP_S]:
        del gs[g]


def fetch_games(dates: list[str], get) -> list[dict]:
    out = []
    for day in dates:
        out += parse_games(get(ESPN, {"dates": day}))
    return out


def snapshot_round(st: dict, get, clock, pool) -> None:
    todo = due(st, clock())
    if not todo:
        return
    rows = list(pool.map(lambda x: take(x[0], x[1], x[2], get, clock), todo))
    append(rows)
    bad = sum("error" in r for r in rows)
    log(f"snapshots {len(rows)} " + " ".join(f"{r['ticker'].split('-', 1)[1]}+{r['delay_s']}" for r in rows[:4])
        + (" ..." if len(rows) > 4 else "") + (f" ERRORS {bad}" if bad else ""))


def tick(get=_get, clock=time.time, sleep=time.sleep, max_s: float = MAX_RUN_S) -> int:
    st = load_state()
    now = clock()
    rc = 0
    try:
        refresh_schedule(st, get, now)
    except Exception as ex:  # noqa: BLE001
        log(f"espn schedule error: {type(ex).__name__}: {ex}")
        rc = 1
    if not in_window(st, now) and not pending(st):
        save_state(st)
        return rc
    t0, next_poll, polling = now, 0.0, False
    with ThreadPoolExecutor(4) as pool:
        while True:
            now = clock()
            dates = in_window(st, now)
            if dates and now >= next_poll:
                next_poll = now + POLL_S
                try:
                    games = fetch_games(dates, get)
                    rows = detect(st, games, get, now)
                    append(rows)
                    for r in rows:
                        log(f"LOCK {r['lock_type']} {r['ticker']} sum={r['sum']} p{r['period']} {r['clock']}"
                            f"{' late_detect' if r['late_detect'] else ''} status={r['market_status_at_lock']}")
                    polling = any(near_end(g) for g in games)
                except Exception as ex:  # noqa: BLE001
                    log(f"poll error: {type(ex).__name__}: {ex}")
                    rc = 1
                    polling = True
            elif not dates:
                polling = False
            snapshot_round(st, get, clock, pool)
            save_state(st)
            now = clock()
            if not polling and not books_owed(st):
                break
            if now - t0 >= max_s:
                log(f"run cap {max_s:.0f}s reached; state kept, next launchd run continues")
                break
            sleep(0.5)
    prune(st, clock())
    save_state(st)
    return rc


def report(path: Path = OUT, all_locks: bool = False) -> None:
    import numpy as np

    with path.open() as f:
        rows = [json.loads(line) for line in f]
    prints: dict[str, dict] = {}
    for r in rows:
        for ts, px, n, side, tid in r.get("new_trades") or []:
            prints.setdefault(r["ticker"], {})[tid] = (ts, px, n, side)
    locks = [r for r in rows if r["kind"] == "lock"]
    print(f"{path}: {len(rows)} rows, {len(locks)} locks ({sum(r['late_detect'] for r in locks)} late_detect), "
          f"{len({r['game_id'] for r in locks})} games; "
          f"{'all locks' if all_locks else 'late_detect excluded (--all to include)'}")
    print("queue = resting YES at 99c when the bid joins; V = later 99c prints that hit bids (taker_side no, ts > as_of).")
    print("share = sum max(V - queue, 0) / sum V (FIFO, no cancels ahead: conservative; own size ignored); V>q = share of tickers reached.")
    print(f"{'series':<12}{'delay':>6}{'n':>5}{'q_med':>9}{'q_p90':>10}{'ask<=99':>9}{'V_sum':>10}{'share':>8}{'V>q':>7}")
    snaps = [r for r in rows if r["kind"] == "snapshot" and "error" not in r and r.get("market_status") == "active"
             and (all_locks or not r["late_detect"])]
    for key in sorted({(r["series"], r["delay_s"]) for r in snaps}):
        grp = [r for r in snaps if (r["series"], r["delay_s"]) == key]
        q = np.array([r["yes_99"] for r in grp], float)
        vs = np.array([sum(n for ts, px, n, side in prints.get(r["ticker"], {}).values()
                           if px == 99 and side == "no" and ts > r["as_of_ts"]) for r in grp], float)
        fills = np.maximum(vs - q, 0)
        ask = [r["best_ask_c"] for r in grp if r["best_ask_c"] is not None]
        share = f"{fills.sum() / vs.sum():.3f}" if vs.sum() > 0 else "n/a"
        ask_le = f"{np.mean([a <= 99 for a in ask]):.2f}" if ask else "n/a"
        print(f"{key[0]:<12}{key[1]:>6}{len(grp):>5}{np.median(q):>9.0f}{np.percentile(q, 90):>10.0f}{ask_le:>9}"
              f"{vs.sum():>10.0f}{share:>8}{np.mean(fills > 0):>7.2f}")
    errs = sum("error" in r for r in rows)
    if errs:
        print(f"rows with errors: {errs}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        sys.exit(tick())
    if cmd == "report":
        args = sys.argv[2:]
        report(Path(args[args.index("--file") + 1]) if "--file" in args else OUT, "--all" in args)
    else:
        sys.exit(f"unknown command {cmd}")
