"""OT1 crypto-taker FORWARD paper logger (READ-ONLY; notes/ops_ot1_forward.md; trial ot1_crypto_taker-18d38e5c).

    python scripts/ot1_forward.py tick [--force-now]   # decide once/day from 11:00 ET (KXBTCD/KXETHD 17:00 event); settle
    python scripts/ot1_forward.py report               # descriptive summary (bets, ROI variants, one-bet-per-day)

Decide: Deribit option trades in [t_d - 60 min, t_d) (public API, fetched FIRST), then ONE Kalshi GET /markets per event
at t_k > t_d (live bid/ask + size at best), options-implied P(YES) with the frozen DEV code (options_implied_tails.py),
frame restricted to DEV amendment 2 (tail_candidates), frozen rule edgelab/rules/opt_tails.py:taker. Picks get an
orderbook GET for depth. Settle after 17:05 ET: one GET /markets?event_ticker= per event -> result of every rung.

Output (append-only): ~/.claude/quant-sports/data/kx_forward/ot1.jsonl (close dates >= 2026-11-16; earlier dates go to
ot1_shakedown.jsonl) + ot1_quotes.jsonl (every rung's quote + implied p, for calibration). Rows:
  day     one per ET date: status (decided | partial | late_* | forced_* | missed), timing, counts, best_of_day ticker
  pick    one per rule-passing market: ticker, side, ask, sizes, fees, edge, in_frame, status (bet | outside_frame | off_time)
  settle  one per event once final: results {ticker: yes|no} for every rung
No orders, no keys, no portfolio endpoints.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from kx_throttle import kget  # noqa: E402
import options_implied_tails as oit  # noqa: E402  (frozen DEV math: smile, p_above_crypto, p_yes, tail_candidates)
from edgelab.rules.opt_tails import taker  # noqa: E402  (frozen rule)

ET = ZoneInfo("America/New_York")
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH"}
FWD_START = "2026-11-16"
OUT = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"
STATE = Path.home() / ".claude" / "quant-sports" / "state"
DERIBIT_HOSTS = ["https://www.deribit.com", "https://history.deribit.com"]
PATH_TR = "/api/v2/public/get_last_trades_by_currency_and_time"
BET_WINDOW_MIN = 20       # snapshot must start in [11:00, 11:20) ET to count as a bet
MISSED_AFTER_H = 12       # no decision by 12:00 ET -> missed
_S = requests.Session()


def utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


SUFFIX = ""               # "--force-now" (manual test) -> "_manual": never touches the scheduled logs


def log_path(date: str, name: str = "ot1") -> Path:
    return OUT / (f"{name}{SUFFIX}.jsonl" if date >= FWD_START and not SUFFIX else f"{name}{SUFFIX or '_shakedown'}.jsonl")


def append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_js) + "\n")


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    return str(x)


def read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def tag(d: pd.Timestamp) -> str:
    return d.strftime("%y") + d.strftime("%b").upper() + d.strftime("%d")


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def fee_exact(p):
    return 0.07 * p * (1 - p)


def fee_cent(p, n):
    return math.ceil(round(0.07 * n * p * (1 - p) * 100, 6)) / 100 / n


# ------------------------------------------------------------------ Deribit (live public)
def deribit_trades(cur: str, t0_ms: int, t1_ms: int) -> tuple[list[dict], str]:
    for host in DERIBIT_HOSTS:
        out, start, ok = [], t0_ms, True
        for _ in range(100):
            r = None
            for i in range(4):
                try:
                    r = _S.get(host + PATH_TR, params={"currency": cur, "kind": "option", "start_timestamp": start,
                                                       "end_timestamp": t1_ms - 1, "count": 1000,
                                                       "include_old": "true", "sorting": "asc"}, timeout=30)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    r = None
                time.sleep(2 * (i + 1))
            if r is None or r.status_code != 200:
                ok = False
                break
            res = r.json()["result"]
            tr = res["trades"]
            out += tr
            if not res.get("has_more") or not tr:
                break
            start = tr[-1]["timestamp"]
        if ok:
            seen, ded = set(), []
            for x in out:
                if x["trade_id"] not in seen and x["timestamp"] < t1_ms:
                    seen.add(x["trade_id"])
                    ded.append(x)
            return ded, host
    raise RuntimeError(f"Deribit {cur}: no host answered")


# ------------------------------------------------------------------ decide
def decide(now: datetime, force: bool = False) -> None:
    d = pd.Timestamp(now.astimezone(ET).date())
    date = str(d.date())
    t11 = datetime(d.year, d.month, d.day, 11, 0, tzinfo=ET)
    t_d = time.time()
    trades = {}
    hosts = {}
    for s, cur in SERIES.items():
        trades[s], hosts[s] = deribit_trades(cur, int((t_d - 3600) * 1000), int(t_d * 1000))
    snaps, picks, quotes = [], [], []
    for s, cur in SERIES.items():
        ev = f"{s}-{tag(d)}17"
        t_k = time.time()
        js = kget("/markets", {"event_ticker": ev, "limit": 1000}) or {}
        t_k1 = time.time()
        ms = js.get("markets", [])
        tr = trades[s]
        last_tr = max((x["timestamp"] for x in tr), default=None)
        snap = {"series": s, "event": ev, "t_deribit": utc_iso(t_d), "t_kalshi": utc_iso(t_k),
                "t_kalshi_end": utc_iso(t_k1), "deribit_host": hosts[s], "deribit_trades": len(tr),
                "last_trade": utc_iso(last_tr / 1000) if last_tr else None,
                "input_age_s": round(t_k - last_tr / 1000, 3) if last_tr else None,
                "inputs_older_than_quote": bool(last_tr is not None and last_tr / 1000 < t_k), "markets": len(ms)}
        if not ms:
            snap["status"] = "no markets"
            snaps.append(snap)
            continue
        close_ts = int(pd.Timestamp(ms[0]["close_time"]).timestamp())
        closes = {int(pd.Timestamp(m["close_time"]).timestamp()) for m in ms}
        T = int(t_k)            # integer second <= t_k; every trade used is < t_d < t_k (smile() also filters < T)
        sm = oit.smile(tr, close_ts, T)
        snap.update({"close": utc_iso(close_ts), "close_times_distinct": len(closes),
                     "smile_points": None if sm is None else len(sm[2]), "spot": None if sm is None else sm[0],
                     "tauD_h": None if sm is None else round(sm[1] * 365 * 24, 3)})
        if sm is None:
            snap["status"] = "no smile"
            snaps.append(snap)
            continue
        tauK = (close_ts - T) / (365 * 86400)
        rows = []
        for m in ms:
            r = {"series": s, "ticker": m["ticker"], "event_ticker": ev, "strike_type": m.get("strike_type"),
                 "floor": num(m.get("floor_strike")), "cap": num(m.get("cap_strike")),
                 "close_ts": int(pd.Timestamp(m["close_time"]).timestamp()),
                 "yes_bid": num(m.get("yes_bid_dollars")), "yes_ask": num(m.get("yes_ask_dollars")),
                 "yes_bid_size": num(m.get("yes_bid_size_fp")), "yes_ask_size": num(m.get("yes_ask_size_fp")),
                 "status_k": m.get("status")}
            r["p_yes"] = oit.p_yes(r, lambda k: oit.p_above_crypto(k, sm, tauK))
            rows.append(r)
        df = pd.DataFrame(rows)
        df["spot"] = sm[0]
        df["date"] = d
        df["asset_class"] = "crypto"
        df["T"] = T
        df["inputs_as_of"] = int(last_tr // 1000)
        df["quote_ts"] = T
        df["mark_yes_bid"] = np.nan
        df["mark_yes_ask"] = np.nan
        for r in df.itertuples():
            quotes.append({"kind": "quote", "date": date, "ticker": r.ticker, "event": ev, "T": utc_iso(T),
                           "strike_type": r.strike_type, "floor": r.floor, "cap": r.cap, "p_yes": r.p_yes,
                           "yes_bid": r.yes_bid, "yes_ask": r.yes_ask, "yes_bid_size": r.yes_bid_size,
                           "yes_ask_size": r.yes_ask_size, "spot": sm[0]})
        frame_tk = set(oit.tail_candidates(df)["ticker"])
        allp = taker(df, {"asset_class": "crypto"})          # rule on ALL rungs; in_frame marks the DEV amendment-2 set
        snap.update({"status": "ok", "p_finite": int(np.isfinite(df["p_yes"]).sum()), "tail_candidates": len(frame_tk),
                     "rule_picks_all_rungs": len(allp)})
        snaps.append(snap)
        dt = datetime.fromtimestamp(t_k, ET) - t11
        on_time = (not force) and timedelta(0) <= dt < timedelta(minutes=BET_WINDOW_MIN)
        for b in allp.itertuples():
            m = df[df["ticker"] == b.bet_id].iloc[0]
            pF = (1 - m.p_yes) if b.side == "no" else m.p_yes
            ask = float(b.price)
            size_best = m.yes_bid_size if b.side == "no" else m.yes_ask_size
            depth = book_depth(b.bet_id, b.side, ask)
            inf = b.bet_id in frame_tk
            picks.append({"kind": "pick", "date": date, "key": b.bet_id, "ticker": b.bet_id, "event": ev, "series": s,
                          "status": ("bet" if inf else "outside_frame") if on_time else "off_time", "in_frame": inf,
                          "side": b.side, "ask": ask, "pF": float(pF), "p_yes": float(m.p_yes),
                          "edge": float(pF - ask - fee_exact(ask)), "yes_bid": m.yes_bid, "yes_ask": m.yes_ask,
                          "size_at_best": size_best, **depth,
                          "fee_exact": fee_exact(ask), "fee_cent_1": fee_cent(ask, 1), "fee_cent_100": fee_cent(ask, 100),
                          "decided_at": utc_iso(t_k1), "quote_time": utc_iso(t_k), "inputs_as_of": utc_iso(last_tr / 1000),
                          "event_start": utc_iso(int(m.close_ts) - 1), "spot": sm[0], "strike_type": m.strike_type,
                          "floor": m.floor, "cap": m.cap})
    bets = [p for p in picks if p["status"] == "bet"]
    best = max(bets, key=lambda p: p["edge"])["ticker"] if bets else None
    st = "decided" if all(x.get("status") == "ok" for x in snaps) else "partial"
    if force:
        st = "forced_" + st
    elif any(not (timedelta(0) <= datetime.fromtimestamp(pd.Timestamp(x["t_kalshi"]).timestamp(), ET) - t11
                  < timedelta(minutes=BET_WINDOW_MIN)) for x in snaps):
        st = "late_" + st
    day = {"kind": "day", "date": date, "status": st, "forced": force, "decided_at": utc_iso(time.time()),
           "snapshots": snaps, "picks_all": len(picks), "bets": len(bets), "best_of_day": best,
           "best_of_day_edge": max((p["edge"] for p in bets), default=None)}
    append(log_path(date, "ot1_quotes"), quotes)
    append(log_path(date), picks + [day])
    print(f"{date} {st}: {len(bets)} bets ({len(picks)} rule picks all rungs), best {best}; "
          + "; ".join(f"{x['series']} {x.get('status')} trades {x['deribit_trades']} smile {x.get('smile_points')} "
                      f"age {x.get('input_age_s')}s" for x in snaps))


def book_depth(ticker: str, side: str, ask: float) -> dict:
    """Size available to BUY `side` at ask, ask+1c, ask+2c (cumulative) from the live orderbook (bids of the other side)."""
    try:
        ob = (kget(f"/markets/{ticker}/orderbook") or {})
        ob = ob.get("orderbook_fp") or ob.get("orderbook") or {}
        lv = ob.get("yes_dollars" if side == "no" else "no_dollars") or []
        lv = [(1 - float(p), float(q)) for p, q in lv]       # our side's ask = 1 - other side's bid
        best = min((p for p, _ in lv), default=float("nan"))
        cum = {f"depth_{k}c": sum(q for p, q in lv if p <= ask + k / 100 + 1e-9) for k in (0, 1, 2)}
        return {"book_best_ask": best, **cum, "book_time": utc_iso(time.time())}
    except Exception as e:  # noqa: BLE001 -- depth is descriptive; never block the decision
        return {"book_error": str(e)[:200]}


# ------------------------------------------------------------------ settle
def settle() -> None:
    now = time.time()
    for name in ("ot1", "ot1_shakedown", "ot1_manual"):
        p = OUT / f"{name}.jsonl"
        rows = read(p)
        done = {r["event"] for r in rows if r.get("kind") == "settle"}
        evs = {}
        for r in rows:
            if r.get("kind") == "day":
                for sn in r.get("snapshots", []):
                    if sn.get("close") and sn["event"] not in done:
                        evs[sn["event"]] = (r["date"], pd.Timestamp(sn["close"]).timestamp())
        new = []
        for ev, (date, close) in sorted(evs.items()):
            if now < close + 300:
                continue
            ms = (kget("/markets", {"event_ticker": ev, "limit": 1000}) or {}).get("markets", [])
            res = {m["ticker"]: m.get("result") for m in ms}
            if not ms or any(v not in ("yes", "no") for v in res.values()):
                print(f"{ev}: not final yet ({sum(v in ('yes', 'no') for v in res.values())}/{len(res)})")
                continue
            new.append({"kind": "settle", "date": date, "event": ev, "settled_at": utc_iso(now), "results": res,
                        "expiration_value": ms[0].get("expiration_value")})
            print(f"{ev}: settled {len(res)} rungs, expiration_value {ms[0].get('expiration_value')}")
        append(p, new)


def tick(force_now: bool = False) -> None:
    now = datetime.now(timezone.utc)
    et = now.astimezone(ET)
    date = str(et.date())
    rows = read(log_path(date))
    decided = any(r.get("kind") == "day" and r["date"] == date for r in rows)
    t11 = et.replace(hour=11, minute=0, second=0, microsecond=0)
    if not decided:
        if force_now:
            decide(now, force=True)
        elif t11 <= et < t11.replace(hour=MISSED_AFTER_H):
            decide(now)
        elif et >= t11.replace(hour=MISSED_AFTER_H):
            append(log_path(date), [{"kind": "day", "date": date, "status": "missed", "decided_at": utc_iso(time.time()),
                                     "snapshots": [], "picks_all": 0, "bets": 0, "best_of_day": None}])
            print(f"{date} missed (first run at {et:%H:%M} ET)")
    settle()


# ------------------------------------------------------------------ report
def report(name: str = "ot1") -> None:
    rows = read(OUT / f"{name}.jsonl")
    days = [r for r in rows if r.get("kind") == "day"]
    res = {}
    for r in rows:
        if r.get("kind") == "settle":
            res.update(r["results"])
    picks = [r for r in rows if r.get("kind") == "pick"]
    print(f"{OUT / (name + '.jsonl')}: days {len(days)} {pd.Series([d['status'] for d in days]).value_counts().to_dict()}")
    best = {d["best_of_day"] for d in days if d.get("best_of_day")}
    for lab, sel in [("bets (in frame, on time)", lambda p: p["status"] == "bet"),
                     ("one bet per day (max edge)", lambda p: p["status"] == "bet" and p["ticker"] in best),
                     ("outside frame (descr.)", lambda p: p["status"] == "outside_frame"),
                     ("off time / forced (descr.)", lambda p: p["status"] == "off_time")]:
        ps = [p for p in picks if sel(p)]
        st = [p for p in ps if res.get(p["ticker"]) in ("yes", "no")]
        line = f"  {lab}: {len(ps)} logged, {len(st)} settled"
        for fk in ("fee_exact", "fee_cent_1", "fee_cent_100"):
            cost = np.array([p["ask"] + p[fk] for p in st])
            won = np.array([res[p["ticker"]] == p["side"] for p in st])
            if len(st):
                line += f" | ROI@{fk} {((won - cost).sum() / cost.sum()):+.2%}"
        if st:
            line += f" | W {int(sum(res[p['ticker']] == p['side'] for p in st))}/{len(st)}"
            line += f" | mean ask {np.mean([p['ask'] for p in st]):.3f}, median size@best {np.median([p['size_at_best'] for p in st]):.0f}"
        print(line)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if cmd == "tick":
        if "--force-now" in sys.argv:
            SUFFIX = "_manual"
        tick("--force-now" in sys.argv)
    elif cmd == "report":
        report(sys.argv[2] if len(sys.argv) > 2 else "ot1")
    else:
        raise SystemExit(__doc__)
