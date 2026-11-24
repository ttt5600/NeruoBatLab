"""E4C cold-band forward PAPER arm (READ-ONLY: public GETs; no keys, no orders). Agent e4_coldmargin, prereg
sports/notes/e4_coldmargin.md; edgelab trial e4c_cold1_maker_fwd (family kalshi_weather_temp).

Hypothesis under test: the frozen E4 rule with the COLD margin narrowed 3F -> 1F (warm margin 3F) adds money. This arm
logs only the EXTRA picks: tradeable brackets whose top edge is 1-3F below the coldest of the 5 live members and that E4
itself does not pick. Hypothetical execution = E4's live maker: post-only NO bid at q = min(cent_floor(NO mid), NO ask - 1c),
skip if NO ask - q > 5c, resting to the market close; FILLED iff a public print at no_price < q (trade-through) after the
decision; unfilled = no bet.

Inputs come from the e4p logger's decision rows (~/.claude/quant-sports/data/kx_forward/e4p.jsonl; override E4C_E4P):
  city_day: the 5 live member values (the same /v1/forecast endpoint + params as kalshi_paper.py), decided_at
  bracket : every tradeable bracket at that decision with its live yes_bid / yes_ask, e4_own, e4_live_at_T, close_time
So this arm adds no new forecast or quote fetch at decision time and cannot look ahead: it only re-selects from rows
e4p wrote at 22:00-22:50 station-local D-1. A night e4p missed is missed here too (counted in `status`).

    python e4c_cold_paper.py tick      # select new decided city-days + settle pending picks (idempotent; hourly is fine)
    python e4c_cold_paper.py status    # picks, fills, losses, units, date-clustered CI

Output (append-only): ~/.claude/quant-sports/data/kx_forward/e4c_cold.jsonl (override E4C_LOG)
  kind city_day  one per (series, D) processed: n_tradeable, n_extra, nmod
  kind pick      one per EXTRA bracket: quote at decision, q, status bid | skip_maker_spread, coldest member, cold gap
  kind settle    one per pick once the market settled: result, tt_fill, fill_ts, n10, pnl_unit, e4_live_final
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
QS = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"
E4P = Path(os.environ.get("E4C_E4P", str(QS / "e4p.jsonl")))
LOG = Path(os.environ.get("E4C_LOG", str(QS / "e4c_cold.jsonl")))
E4_LIVE = Path(os.environ.get("E4C_E4_LIVE", str(Path.home() / ".claude" / "quant-kalshi" / "data" / "kalshi_paper.jsonl")))
FORWARD_FROM = os.environ.get("E4C_FORWARD_FROM", "2026-11-25")   # first event day whose decisions all follow registration (11-24 03:11Z)
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
WARM_M, COLD_M, E4_M = 3.0, 1.0, 3.0
MAX_IMPROVE = 0.05
H = {"User-Agent": "quant-sports-readonly-paper"}


# ------------------------------------------------------------------ pure rule (tested)
def rejected(st: str, floor, cap, vals: list[float], warm: float, cold: float) -> bool:
    """kalshi_paper.rejected() with separate warm / cold margins (warm = cold = 3 is the frozen E4 rule)."""
    if len(vals) < 3:
        return False
    hi, lo = max(vals), min(vals)
    f = float(floor) if floor is not None else math.nan
    c = float(cap) if cap is not None else math.nan
    if st == "greater":
        return hi + warm < f + 1
    if st == "less":
        return lo - cold > c - 1
    if st == "between":
        return hi + warm < f or lo - cold > c
    return False


def cold_extra(st: str, floor, cap, vals: list[float]) -> bool:
    """Picked by the cold-1F variant but not by frozen E4."""
    return rejected(st, floor, cap, vals, WARM_M, COLD_M) and not rejected(st, floor, cap, vals, E4_M, E4_M)


def cent_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def maker_q(yes_bid: float, yes_ask: float) -> tuple[float, str]:
    ask, bid = 1 - yes_bid, 1 - yes_ask
    q = round(min(cent_floor((ask + bid) / 2), ask - 0.01), 4)
    if q <= 0 or ask - q > MAX_IMPROVE + 1e-9:
        return q, "skip_maker_spread"
    return q, "bid"


def tt_fill(trades: list[tuple], t0: int, t1: int, q: float) -> tuple[bool, int | None, bool]:
    """trades = (ts, no_price, count[, taker_side]). (trade-through filled, first fill ts, >= 10 contracts at <= q)."""
    w = [(t[0], t[1], t[2]) for t in trades if t0 < t[0] <= t1]
    thr = sorted(ts for ts, p, _ in w if p < q - 1e-9)
    n10 = sum(c for _, p, c in w if p <= q + 1e-9) >= 10
    return bool(thr), (thr[0] if thr else None), bool(n10)


def close_no(trades: list[tuple], t0: int, hours: float = 10) -> float | None:
    """edgelab E4 CLV close: last print with taker_side 'no' in (t0, t0 + 10 h]."""
    w = sorted(t for t in trades if len(t) > 3 and t[3] == "no" and t0 < t[0] <= t0 + hours * 3600)
    return float(w[-1][1]) if w else None


def pnl_unit(result: str, q: float, filled: bool) -> float:
    if not filled:
        return 0.0
    return 1 / q - 1 if result == "no" else -1.0


# ------------------------------------------------------------------ io
def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue   # a half-written last line from a concurrent writer; it is re-read next tick
    return out


def append(rows: list[dict], p: Path) -> None:
    if not rows:
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def epoch(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def kget(path: str, params: dict | None = None, tries: int = 5):
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from kx_throttle import kget as kt   # shared cross-process throttle when available
        return kt(path, params)
    except ImportError:
        pass
    for i in range(tries):
        try:
            r = requests.get(API + path, params=params, headers=H, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 * (i + 1) + random.random())
                continue
            return r.json() if r.ok else None
        except requests.RequestException:
            time.sleep(2 * (i + 1))
    return None


def tape(tk: str, t0: int, t1: int) -> list[tuple]:
    out = []
    for path in ("/markets/trades", "/historical/trades"):
        cur = None
        while True:
            p = {"ticker": tk, "min_ts": t0, "max_ts": t1, "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = kget(path, p) or {}
            out += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        if out:
            break
    rows = []
    for r in out:
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append((epoch(r["created_time"]), npx, cnt, r.get("taker_side")))
    return sorted(set(rows))


# ------------------------------------------------------------------ select + settle
def select(e4p_rows: list[dict], done: set, forward_from: str = FORWARD_FROM) -> list[dict]:
    """New city_day + pick rows from e4p's decided city-days not yet processed."""
    cds = {r["event"]: r for r in e4p_rows if r.get("kind") == "city_day" and r.get("status") == "decided"}
    br: dict[tuple, list] = {}
    for r in e4p_rows:
        if r.get("kind") == "bracket":
            br.setdefault((r["series"], r["event_day"]), []).append(r)
    out = []
    for ev, cd in sorted(cds.items()):
        if cd["event_day"] < forward_from or ev in done:
            continue
        vals = [float(v) for v in (cd.get("forecasts") or {}).values() if v is not None]
        rows = br.get((cd["series"], cd["event_day"]), [])
        picks = []
        for b in rows:
            if b.get("decided_at") != cd.get("decided_at"):
                continue
            if not cold_extra(b.get("strike_type"), b.get("floor"), b.get("cap"), vals):
                continue
            if b.get("e4_own") or b.get("e4_live_at_T"):
                continue  # already an E4 pick (own recompute or live log at decision) -> not EXTRA
            q, st = maker_q(float(b["yes_bid"]), float(b["yes_ask"]))
            gap = min(vals) - (float(b["cap"]) if b.get("strike_type") == "between" else float(b["cap"]) - 1)
            picks.append({"kind": "pick", "ticker": b["ticker"], "series": cd["series"], "event": ev,
                          "event_day": cd["event_day"], "decided_at": cd["decided_at"], "strike_type": b.get("strike_type"),
                          "floor": b.get("floor"), "cap": b.get("cap"), "yes_bid": b["yes_bid"], "yes_ask": b["yes_ask"],
                          "no_ask": round(1 - float(b["yes_bid"]), 4), "q": q, "status": st, "close_time": b.get("close_time"),
                          "coldest": min(vals), "cold_gap": round(gap, 2), "nmod": len(vals)})
        out.append({"kind": "city_day", "event": ev, "series": cd["series"], "event_day": cd["event_day"],
                    "decided_at": cd["decided_at"], "nmod": len(vals), "n_tradeable": len(rows), "n_extra": len(picks)})
        out += picks
    return out


def settle(rows: list[dict], now: datetime, get_market=None, get_tape=None, e4_live: set | None = None) -> list[dict]:
    get_market = get_market or (lambda tk: ((kget(f"/markets/{tk}") or {}).get("market") or {}))
    get_tape = get_tape or tape
    e4_live = e4_live if e4_live is not None else {r.get("ticker") for r in jl(E4_LIVE)}
    settled = {r["ticker"] for r in rows if r.get("kind") == "settle"}
    out = []
    for p in rows:
        if p.get("kind") != "pick" or p["ticker"] in settled:
            continue
        if p.get("close_time") and epoch(p["close_time"]) > now.timestamp():
            continue
        m = get_market(p["ticker"])
        if m.get("result") not in ("yes", "no"):
            continue
        t0, t1 = epoch(p["decided_at"]), epoch(m.get("close_time") or p["close_time"])
        filled, fts, n10, cno = False, None, False, None
        if p["status"] == "bid":
            tr = get_tape(p["ticker"], t0 - 60, t1)
            filled, fts, n10 = tt_fill(tr, t0, t1, float(p["q"]))
            cno = close_no(tr, t0)
        out.append({"kind": "settle", "ticker": p["ticker"], "event_day": p["event_day"], "settled_at": iso(now),
                    "result": m["result"], "expiration_value": m.get("expiration_value"), "tt_fill": filled,
                    "fill_ts": fts, "n10": n10, "close_no": cno, "close_time": m.get("close_time") or p["close_time"], "pnl_unit": round(pnl_unit(m["result"], float(p["q"]), filled), 6),
                    "e4_live_final": p["ticker"] in e4_live})
    return out


def cmd_tick() -> int:
    mine = jl(LOG)
    done = {r["event"] for r in mine if r.get("kind") == "city_day"}
    new = select(jl(E4P), done, FORWARD_FROM)
    append(new, LOG)
    st = settle(mine + new, datetime.now(timezone.utc))
    append(st, LOG)
    print(f"e4c_cold: {sum(r['kind'] == 'city_day' for r in new)} city-days, {sum(r['kind'] == 'pick' for r in new)} picks, "
          f"{len(st)} settled -> {LOG}")
    return 0


def cmd_status() -> int:
    rs = jl(LOG)
    picks = {r["ticker"]: r for r in rs if r.get("kind") == "pick"}
    st = {r["ticker"]: r for r in rs if r.get("kind") == "settle"}
    bets = [st[t] for t in st if st[t]["tt_fill"] and not st[t]["e4_live_final"] and picks.get(t, {}).get("status") == "bid"]
    days: dict[str, float] = {}
    for b in bets:
        days[b["event_day"]] = days.get(b["event_day"], 0.0) + b["pnl_unit"]
    v = list(days.values())
    ci = [None, None]
    if len(v) >= 5:
        rng = random.Random(0)
        bs = sorted(sum(rng.choice(v) for _ in v) for _ in range(4000))
        ci = [round(bs[100], 3), round(bs[3899], 3)]
    cds = [r for r in rs if r.get("kind") == "city_day"]
    print(json.dumps({"city_days": len(cds), "picks": len(picks), "bid": sum(p["status"] == "bid" for p in picks.values()),
                      "settled": len(st), "fills": len(bets), "losses": sum(b["result"] == "yes" for b in bets),
                      "units": round(sum(b["pnl_unit"] for b in bets), 3), "units_ci95_fill_days": ci,
                      "first_day": min((r["event_day"] for r in cds), default=None)}, indent=1))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    sys.exit({"tick": cmd_tick, "status": cmd_status}[cmd]())
