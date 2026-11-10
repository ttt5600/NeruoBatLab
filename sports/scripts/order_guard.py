"""Exchange-truth guards for every Kalshi order path (after the 2026-11-09 NFLF runaway exit, sports/STATE.md).

What went wrong: ingame_exec's NFLF exit sent an IOC sell, could not find the order by client_order_id afterwards, so
it assumed nothing filled and sent ANOTHER sell with a NEW client_order_id every tick. Every one executed. On Kalshi,
selling YES you do not hold == buying NO, so 14 sells of 17 opened 325 NO on one market. The journal was the only
check, and the journal was wrong. Rules here: trust the EXCHANGE POSITION, not the journal or an order lookup.

  signed_position(client, t)     + = YES held, - = NO held (GET /portfolio/positions, position_fp). Raises on failure.
  guarded_create(client, body, strategy=, cap=, on_trip=)
                                 the ONLY way a bot should POST an order:
                                 (b) per-ticker exposure cap from the exchange position: refuse if |position after| x
                                     price > cap and the order increases exposure (a sell that would flip YES -> NO
                                     is an increase; a pure reduction is always allowed);
                                 (c) circuit breaker: > MAX_ORDERS distinct client_order_ids on one ticker in WINDOW_S
                                     -> on_trip(reason) (write the strategy's kill file + notify) and refuse.
                                 Refusals return (0, {"error": "order_guard: ..."}) -- nothing was sent.
  safe_reduce(client, t, side_held, want, own_price, cid, ...)
                                 one IOC reduce of min(want, exchange-held) contracts; filled = position change.
  resolve_attempt(...)           did a prior reduce attempt fill? order-by-cid lookup, else the position change;
                                 None = unconfirmed (caller must NOT send a new client_order_id).
Retries of one order must reuse the same client_order_id (kalshi_exec.Kalshi.request re-sends the same body; a
duplicate is a 409, never a second order).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import kalshi_exec as kx

WINDOW_S = 600
MAX_ORDERS = 4          # distinct client_order_ids per ticker per WINDOW_S; the 5th trips the breaker
TOL = 1e-9


def log_path() -> Path:
    return kx.DATA / "order_guard.jsonl"


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


# ---------------------------------------------------------------- exchange position
def signed_position(client, ticker: str) -> float:
    """Contracts held on `ticker`: + YES, - NO. Raises kx.TransportError if the read fails (callers must refuse)."""
    rows = client.positions(ticker=ticker)
    return sum(float(p.get("position_fp") or p.get("position") or 0) for p in rows
               if p.get("ticker", ticker) == ticker)


def held(client, ticker: str, side: str) -> float:
    """Contracts of `side` ('yes'|'no') held per the exchange (>= 0)."""
    pos = signed_position(client, ticker)
    return max(pos if side == "yes" else -pos, 0.0)


def _delta(body: dict) -> float:
    """Signed YES-contract change if `body` fills fully: bid = buy YES (+), ask = sell YES / buy NO (-)."""
    n = float(body["count"])
    return n if body["side"] == "bid" else -n


def exposure(pos: float, yes_px: float) -> float:
    """Dollar exposure of a signed position at a YES price (NO contracts priced at 1 - yes_px)."""
    return abs(pos) * (yes_px if pos > 0 else 1 - yes_px)


# ---------------------------------------------------------------- circuit breaker
def _recent(ticker: str, now: datetime) -> set[str]:
    p = log_path()
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("ticker") != ticker or not r.get("cid"):
            continue
        try:
            age = (now - datetime.fromisoformat(str(r["ts"]).replace("Z", "+00:00"))).total_seconds()
        except (KeyError, ValueError):
            continue
        if 0 <= age <= WINDOW_S:
            out.add(r["cid"])
    return out


def _record(row: dict) -> None:
    p = log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def guarded_create(client, body: dict, *, strategy: str, cap: float, on_trip: Callable[[str], None],
                   now: datetime | None = None):
    """POST `body` only if the breaker and the exchange-position exposure cap allow it. Returns (code, out) like
    client.create_order; code 0 = refused here, nothing sent."""
    now = _now(now)
    t, cid = body["ticker"], body["client_order_id"]
    prior = _recent(t, now) - {cid}            # a re-send of the same cid is idempotent (409), never counted twice
    if len(prior) >= MAX_ORDERS:
        why = (f"order_guard breaker: {len(prior) + 1} orders on {t} within {WINDOW_S // 60} min ({strategy}); "
               f"refused and stopped")
        _record({"ts": now.isoformat(), "event": "breaker", "strategy": strategy, "ticker": t, "cid": None, "why": why})
        on_trip(why)
        return 0, {"error": why}
    try:
        pos = signed_position(client, t)
    except Exception as ex:  # noqa: BLE001 -- no position, no order
        return 0, {"error": f"order_guard: position read failed ({str(ex)[:120]}); refused"}
    px = float(body["price"])
    after = pos + _delta(body)
    x0, x1 = exposure(pos, px), exposure(after, px)
    if x1 > cap + 0.005 and x1 > x0 + TOL:
        why = (f"order_guard: {strategy} {t} position {pos:+g} -> {after:+g} = ${x1:.2f} exposure > per-ticker cap "
               f"${cap:.2f}; refused")
        _record({"ts": now.isoformat(), "event": "refused", "strategy": strategy, "ticker": t, "cid": None, "why": why})
        return 0, {"error": why}
    _record({"ts": now.isoformat(), "event": "order", "strategy": strategy, "ticker": t, "cid": cid,
             "side": body["side"], "count": body["count"], "price": body["price"], "pos_before": pos})
    return client.create_order(body)


# ---------------------------------------------------------------- reducing a position (exits)
def _find(client, ticker: str, cid: str) -> dict | None:
    m = [x for x in client.orders(ticker=ticker) if x.get("client_order_id") == cid]
    return m[0] if m else None


def safe_reduce(client, ticker: str, side_held: str, want: float, own_price: float, cid: str, *, strategy: str,
                cap: float, on_trip: Callable[[str], None], now: datetime | None = None) -> dict:
    """ONE IOC that sells at most min(want, exchange-held) contracts of `side_held` at `own_price` (that side's
    price). Never sells what we do not hold, so it can never open the opposite side. `filled` comes from the
    position change (None if the re-read failed or showed no change yet -- resolve it with resolve_attempt)."""
    h0 = held(client, ticker, side_held)
    n = int(math.floor(min(float(want), h0) + TOL))
    out = {"held_before": h0, "count": n, "cid": cid, "sent": False, "filled": 0.0}
    if n < 1:
        return {**out, "reason": "nothing_held"}
    yes_px = own_price if side_held == "yes" else 1 - own_price
    body = {"ticker": ticker, "side": "ask" if side_held == "yes" else "bid", "count": str(n),
            "price": f"{yes_px:.4f}", "time_in_force": "immediate_or_cancel",
            "self_trade_prevention_type": "taker_at_cross", "client_order_id": cid}
    code, resp = guarded_create(client, body, strategy=strategy, cap=cap, on_trip=on_trip, now=now)
    out.update(sent=code != 0, http=code, response=resp if code not in (200, 201) else None)
    if code == 0:
        return {**out, "reason": resp.get("error")}
    try:
        h1 = held(client, ticker, side_held)
    except Exception:  # noqa: BLE001
        return {**out, "filled": None, "held_after": None}
    f = min(max(h0 - h1, 0.0), float(n))
    return {**out, "held_after": h1, "filled": f if f > 0 else None}


def resolve_attempt(client, ticker: str, cid: str, side_held: str, held_before: float, count: float,
                    allow_zero: bool) -> float | None:
    """Filled contracts of a previous reduce attempt, or None if unconfirmed.
    1) the order by client_order_id (if the exchange lists it and it is not resting): its fill_count;
    2) else the exchange position change since the attempt (> 0 = that many filled);
    3) else 0.0 only when allow_zero (a LATER tick, so the position had time to update), otherwise None."""
    x = _find(client, ticker, cid)
    if x is not None and x.get("status") != "resting":
        return min(float(x.get("fill_count_fp") or 0), float(count))
    try:
        h = held(client, ticker, side_held)
    except Exception:  # noqa: BLE001
        return None
    f = min(max(float(held_before) - h, 0.0), float(count))
    if f > 0:
        return f
    return 0.0 if allow_zero else None
