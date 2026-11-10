"""order_guard + every bot's order path against the fake exchange (no network, no key, no real orders).

Regression for the 2026-11-09 NFLF runaway exit (sports/STATE.md): the exit IOC filled but the order lookup by
client_order_id returned nothing, so each tick sent another sell with a NEW client_order_id and the account sold YES
it did not hold (= bought NO): 325 NO on one market.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_exec as kx  # noqa: E402
import lead_exec as lx  # noqa: E402
import ingame_exec as ix  # noqa: E402
import order_guard as og  # noqa: E402
from test_lead_exec import FakeExchange, FakeClient, write_env  # noqa: E402
from test_ingame_exec import ARM, NF, HT, NOW, book, nfl_sig, nhl_sig, posts, run  # noqa: E402


@pytest.fixture
def ex(tmp_path, monkeypatch):
    kx.set_root(tmp_path, deploy=tmp_path / "deploy")
    (tmp_path / "deploy").mkdir()
    monkeypatch.setattr(kx, "notify", lambda *a, **k: None)
    e = FakeExchange()
    e.fee_type = "quadratic_with_maker_fees"
    for t in (HT, NF):
        e.markets[t] = {"ticker": t, "status": "active", "close_time": "2026-11-10T00:00:00Z"}
    book(e, HT, 0.80, 0.82)
    book(e, NF, 0.40, 0.42)
    return e


def realistic(e, hide_exits=True, exit_fill=True):
    """Fills move the exchange position (a sell of YES not held goes negative = NO), and optionally the order
    lookup by client_order_id cannot see exit (ask) orders -- the incident's failure mode."""
    send, hidden = e.send, set()

    def send2(method, url, headers, params, data, timeout):
        b = json.loads(data) if data else None
        pre = len(e.orders)
        code, out = send(method, url, headers, params, data, timeout)
        if method == "POST" and b and len(e.orders) > pre:
            o = e.orders[-1]
            if b["side"] == "ask" and not exit_fill:
                o.update(fill_count_fp="0", status="canceled", taker_fill_cost_dollars="0")
                out = {**out, "fill_count": "0.00", "remaining_count": "0.00"}   # POST response agrees with the order
            f = float(o["fill_count_fp"])
            e.positions[b["ticker"]] = e.positions.get(b["ticker"], 0) + (f if b["side"] == "bid" else -f)
            if b["side"] == "ask" and hide_exits:
                hidden.add(b["client_order_id"])
        if method == "GET" and url.endswith("/portfolio/orders"):
            out = {**out, "orders": [o for o in out["orders"] if o["client_order_id"] not in hidden]}
        return code, out
    e.send = send2
    return e


def filled_nflf(e, tmp_path, n=4):
    write_env(tmp_path, **ARM, LEAD_LIVE_NFLF="1")
    sig = nfl_sig()
    run(e, [sig])
    e.orders[0].update(status="executed", fill_count_fp=str(n), taker_fill_cost_dollars=f"{0.40 * n:.2f}")
    e.positions[NF] = n
    book(e, NF, 0.43, 0.45)
    return sig


def at(sig, s):
    return datetime.fromtimestamp(sig["t_play"] + s, timezone.utc)


def sells(e):
    return [c[3] for c in posts(e) if c[3]["side"] == "ask"]


# ---------------------------------------------------------------- 1. the incident
def test_incident_exit_never_oversells(ex, tmp_path):
    """Pre-fix code: 6 ticks -> 6 sells of 4, position 4 -> -20 (20 NO opened). Post-fix: one sell, position 0."""
    realistic(ex)
    sig = filled_nflf(ex, tmp_path)
    for k in range(6):
        run(ex, [], at=at(sig, 485 + 15 * k))
    assert ex.positions[NF] == 0, f"{len(sells(ex))} sells, position {ex.positions[NF]}"
    assert len(sells(ex)) == 1
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["settled"] and r["exit_filled"] == 4 and r["result"] == "exited"


def test_exit_caps_at_exchange_position(ex, tmp_path):
    realistic(ex, hide_exits=False)
    sig = filled_nflf(ex, tmp_path)
    ex.positions[NF] = 2                    # journal says 4, exchange says 2 (e.g. sold by hand)
    run(ex, [], at=at(sig, 485))
    (b,) = sells(ex)
    assert b["count"] == "2" and ex.positions[NF] == 0


def test_exit_nothing_held_sends_nothing_and_closes(ex, tmp_path):
    realistic(ex)
    sig = filled_nflf(ex, tmp_path)
    ex.positions[NF] = 0
    for k in range(3):
        run(ex, [], at=at(sig, 485 + 15 * k))
    assert sells(ex) == []
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["settled"] and r["result"] == "no_position" and r.get("needs_reconcile")


def test_exit_held_as_no_is_never_sold_as_yes(ex, tmp_path):
    realistic(ex)
    sig = filled_nflf(ex, tmp_path)
    ex.positions[NF] = -325                 # the incident's end state: NO held, no YES
    run(ex, [], at=at(sig, 485))
    assert sells(ex) == [] and ex.positions[NF] == -325


def test_unconfirmed_attempt_never_retried(ex, tmp_path):
    realistic(ex, hide_exits=True, exit_fill=False)       # IOC killed unfilled, lookup also blind
    sig = filled_nflf(ex, tmp_path)
    run(ex, [], at=at(sig, 485))
    assert len(sells(ex)) == 1
    r = lx.fold(lx.journal(), "live")[sig["key"]]
    assert r["exit_pending"] and r["exit_attempts"] == 1
    run(ex, [], at=at(sig, 500))   # critic 11-10: MAX_EXIT_ATTEMPTS = 1 -> no second sell, hold to settlement
    assert len(sells(ex)) == 1
    assert ex.positions[NF] == 4


def test_unreadable_position_blocks_exit(ex, tmp_path):
    realistic(ex)
    sig = filled_nflf(ex, tmp_path)
    send = ex.send

    def broken(method, url, *a):
        if url.endswith("/portfolio/positions"):
            return 500, {}
        return send(method, url, *a)
    ex.send = broken
    run(ex, [], at=at(sig, 485))
    assert sells(ex) == []


# ---------------------------------------------------------------- 2. safe_reduce / resolve_attempt directly
def test_safe_reduce_unit(ex):
    c = FakeClient(realistic(ex))
    ex.positions[NF] = 3
    r = og.safe_reduce(c, NF, "yes", 10, 0.43, "cid-1", strategy="T", cap=5, on_trip=lambda w: None)
    assert r["count"] == 3 and r["filled"] == 3 and ex.positions[NF] == 0
    r = og.safe_reduce(c, NF, "yes", 10, 0.43, "cid-2", strategy="T", cap=5, on_trip=lambda w: None)
    assert r["count"] == 0 and not r["sent"] and r["reason"] == "nothing_held"
    ex.positions[NF] = -5                    # NO held: reduce = buy YES (bid) at 1 - NO price
    r = og.safe_reduce(c, NF, "no", 2, 0.60, "cid-3", strategy="T", cap=5, on_trip=lambda w: None)
    b = posts(ex)[-1][3]
    assert b["side"] == "bid" and b["price"] == "0.4000" and b["count"] == "2" and ex.positions[NF] == -3


def test_resolve_attempt(ex):
    c = FakeClient(ex)
    ex.positions[NF] = 4
    assert og.resolve_attempt(c, NF, "nope", "yes", 4, 4, allow_zero=False) is None
    assert og.resolve_attempt(c, NF, "nope", "yes", 4, 4, allow_zero=True) == 0.0
    ex.positions[NF] = 1
    assert og.resolve_attempt(c, NF, "nope", "yes", 4, 4, allow_zero=False) == 3
    ex.orders.append({"order_id": "z", "client_order_id": "seen", "ticker": NF, "status": "canceled",
                      "fill_count_fp": "2"})
    assert og.resolve_attempt(c, NF, "seen", "yes", 4, 4, allow_zero=False) == 3   # position shows 3 sold > lookup 2


# ---------------------------------------------------------------- 3b. exposure cap from the exchange position
def test_exposure_cap_refuses_flip_and_growth(ex):
    c, trips = FakeClient(ex), []
    ex.positions[NF] = 3
    body = {"ticker": NF, "side": "ask", "count": "20", "price": "0.4300", "client_order_id": "a"}
    code, out = og.guarded_create(c, body, strategy="T", cap=2.0, on_trip=trips.append)
    assert code == 0 and "per-ticker cap" in out["error"] and posts(ex) == []     # 3 YES -> 17 NO = $9.69
    body = {**body, "count": "3", "client_order_id": "b"}                         # pure reduction: allowed
    code, _ = og.guarded_create(c, body, strategy="T", cap=0.5, on_trip=trips.append)
    assert code == 201
    ex.positions[NF] = 4
    body = {"ticker": NF, "side": "bid", "count": "2", "price": "0.4000", "client_order_id": "c"}
    code, _ = og.guarded_create(c, body, strategy="T", cap=2.0, on_trip=trips.append)   # 6 x 0.40 = 2.40 > 2
    assert code == 0 and not trips


def test_position_read_failure_refuses(ex):
    send = ex.send
    ex.send = lambda m, u, *a: (500, {}) if u.endswith("/portfolio/positions") else send(m, u, *a)
    c = FakeClient(ex)                       # the client binds ex.send at construction
    body = {"ticker": NF, "side": "bid", "count": "1", "price": "0.4000", "client_order_id": "c"}
    code, out = og.guarded_create(c, body, strategy="T", cap=2.0, on_trip=lambda w: None)
    assert code == 0 and "position read failed" in out["error"] and posts(ex) == []


# ---------------------------------------------------------------- 3c. circuit breaker
def test_breaker_trips_on_fifth_cid_not_on_resend(ex):
    c, trips = FakeClient(ex), []
    t0 = NOW
    for k in range(og.MAX_ORDERS):
        b = {"ticker": NF, "side": "bid", "count": "1", "price": "0.0100", "client_order_id": f"k{k}"}
        assert og.guarded_create(c, b, strategy="T", cap=100, on_trip=trips.append, now=t0)[0] == 201
    b0 = {"ticker": NF, "side": "bid", "count": "1", "price": "0.0100", "client_order_id": "k0"}
    assert og.guarded_create(c, b0, strategy="T", cap=100, on_trip=trips.append, now=t0)[0] == 409   # same cid
    assert not trips
    b = {**b0, "client_order_id": "k9"}
    code, out = og.guarded_create(c, b, strategy="T", cap=100, on_trip=trips.append, now=t0)
    assert code == 0 and trips and "breaker" in trips[0]
    later = t0 + timedelta(seconds=og.WINDOW_S + 1)                  # window passed: allowed again
    assert og.guarded_create(c, {**b, "client_order_id": "k10"}, strategy="T", cap=100,
                             on_trip=trips.append, now=later)[0] == 201


def test_breaker_in_ingame_writes_disable(ex, tmp_path):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    for k in range(og.MAX_ORDERS):
        og._record({"ts": NOW.isoformat(), "event": "order", "ticker": HT, "cid": f"x{k}"})
    run(ex, [nhl_sig()])
    assert posts(ex) == [] and lx.disable_path("NHL").exists()


# ---------------------------------------------------------------- each bot routes through the guard
def test_nhl_entry_refused_over_cap(ex, tmp_path, monkeypatch):
    write_env(tmp_path, **ARM, LEAD_LIVE_NHL="1")
    monkeypatch.setitem(ix.STRATS["NHL"], "ticker_cap", 0.50)
    run(ex, [nhl_sig()])
    assert posts(ex) == []
    o = lx.fold(lx.journal(), "live")["NHL|g1|home"]
    assert o["status"] == "rejected" and o["cost"] == 0.0


def test_lead_and_weather_use_guard(ex, monkeypatch):
    seen = []
    real = og.guarded_create

    def spy(client, body, **kw):
        seen.append((kw["strategy"], kw["cap"]))
        return real(client, body, **kw)
    monkeypatch.setattr(og, "guarded_create", spy)
    c = FakeClient(ex)
    rec = {"mode": "live", "ticker": NF, "strategy": "DUL", "tag": "lead_DUL", "side": "no", "count": 2,
           "avg_price": 0.50, "cost": 1.04}
    lx.place(c, rec, {"notify": False}, NOW)
    wrec = {"mode": "live", "ticker": HT, "count": 2, "limit_yes_price": "0.2000", "no_ask_now": 0.80,
            "cost": 1.64}
    kx.place(c, wrec, {"notify": False, "max_order": 6.20}, NOW)
    assert seen == [("lead_DUL", 2.00), ("weather", 6.20)]


# critic 11-10: blind order lookup + stale position reads on ticks seconds apart must not open NO
def lagged(e, stale_reads):
    send, st = e.send, {"left": 0, "snap": None}
    def s2(method, url, headers, params, data, timeout):
        if method == "POST":
            st["snap"] = dict(e.positions)
            code, out = send(method, url, headers, params, data, timeout)
            st["left"] = stale_reads
            return code, out
        if url.endswith("/portfolio/positions") and st["left"] > 0:
            st["left"] -= 1
            return 200, {"market_positions": [{"ticker": t, "position_fp": str(q)} for t, q in st["snap"].items()]}
        return send(method, url, headers, params, data, timeout)
    e.send = s2
    return e

def test_lag_blind_lookup_ticks_seconds_apart_never_oversells(ex, tmp_path):
    realistic(ex, hide_exits=True)
    sig = filled_nflf(ex, tmp_path)
    lagged(ex, stale_reads=6)
    run(ex, [], at=at(sig, 485))
    run(ex, [], at=at(sig, 487))
    run(ex, [], at=at(sig, 600))
    assert ex.positions[NF] >= 0
