"""kalshi_exec.py against a mocked Kalshi HTTP layer: no network, no real key, no real orders.

Covers: live gating, idempotency (re-run, ambiguous POST, 409 duplicate, exchange-side existing order),
caps (order / day / open / bankroll tiers), kill switch, partial fills, price + depth + staleness guards,
loss stops, the fill-price convention kill, retry on 429, stale resting-order cancel, and request signing.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import kalshi_exec as kx  # noqa: E402

NOW = datetime(2026, 10, 22, 3, 10, tzinfo=timezone.utc)
T1 = "KXHIGHNY-26OCT22-T80"     # orig7
T2 = "KXHIGHTBOS-26OCT22-T75"   # x1
NOTES: list = []  # notifications captured per test


class FakeExchange:
    """Routes (method, path) like the Trade API v2; records every call."""

    def __init__(self):
        self.calls = []
        self.books = {}        # ticker -> [[yes_bid_price, qty], ...]
        self.markets = {}      # ticker -> market dict
        self.orders = []       # exchange-side Order objects
        self.positions = {}    # ticker -> position_fp (negative = NO)
        self.post_script = []  # queued responses / exceptions for POST create order
        self.balance = 1000.0
        self.trading = True
        self.no_books = {}     # ticker -> [[no_bid_price, qty], ...]
        self.fee_type = "quadratic"

    def send(self, method, url, headers, params, data, timeout):
        path = urlparse(url).path.replace(kx.BASE, "", 1)
        self.calls.append((method, path, dict(params or {}), json.loads(data) if data else None, dict(headers)))
        if path == "/exchange/status":
            return 200, {"exchange_active": True, "trading_active": self.trading}
        if path.startswith("/markets/") and path.endswith("/orderbook"):
            t = path.split("/")[2]
            return ((200, {"orderbook_fp": {"yes_dollars": self.books[t], "no_dollars": self.no_books.get(t, [])}})
                    if t in self.books else (404, {}))
        if path.startswith("/series/"):
            return 200, {"series": {"ticker": path.split("/")[2], "fee_type": self.fee_type, "fee_multiplier": 1}}
        if path.startswith("/markets/"):
            t = path.split("/")[2]
            return (200, {"market": self.markets[t]}) if t in self.markets else (404, {})
        if path == "/portfolio/balance":
            return 200, {"balance": int(self.balance * 100), "balance_dollars": f"{self.balance:.4f}"}
        if path == "/portfolio/orders" and method == "GET":
            rows = [o for o in self.orders if (not params.get("ticker") or o["ticker"] == params["ticker"])
                    and (not params.get("status") or o["status"] == params["status"])]
            return 200, {"orders": rows, "cursor": ""}
        if path == "/portfolio/positions":
            rows = [{"ticker": t, "position_fp": f"{q:.2f}", "market_exposure_dollars": "0"} for t, q in self.positions.items()
                    if q and (not params.get("ticker") or t == params["ticker"])]
            return 200, {"market_positions": rows, "event_positions": [], "cursor": ""}
        if path == "/portfolio/events/orders" and method == "POST":
            body = json.loads(data)
            if any(o["client_order_id"] == body["client_order_id"] for o in self.orders):
                return 409, {"code": "conflict", "message": "duplicate client_order_id"}
            nxt = self.post_script.pop(0) if self.post_script else ("rest" if body.get("post_only") else "fill", None)
            kind, arg = nxt
            if kind == "rest":
                return 201, self._rest(body)
            if kind == "raise":
                raise requests.ConnectionError("boom")
            if kind == "raise_after_accept":  # exchange accepted it, the response got lost
                self._accept(body, arg)
                raise requests.ConnectionError("reset after send")
            if kind == "http":
                return arg
            return 201, self._accept(body, arg)
        if path.startswith("/portfolio/events/orders/") and method == "DELETE":
            oid = path.rsplit("/", 1)[1]
            for o in self.orders:
                if o["order_id"] == oid:
                    rem = o["remaining_count_fp"]
                    o.update(status="canceled", remaining_count_fp="0.00")
                    return 200, {"order_id": oid, "reduced_by": rem, "ts_ms": 1}
            return 404, {}
        return 404, {"error": f"unrouted {method} {path}"}

    def _accept(self, body, fill_n):
        n = float(body["count"]) if fill_n is None else float(fill_n)
        yes = float(body["price"])
        no = round(1 - yes, 4)
        fee = round(0.07 * no * (1 - no), 6)
        oid = f"oid-{len(self.orders) + 1}"
        self.orders.append({"order_id": oid, "client_order_id": body["client_order_id"], "ticker": body["ticker"],
                            "status": "executed" if n else "canceled", "outcome_side": "no", "book_side": "ask",
                            "fill_count_fp": f"{n:.2f}", "remaining_count_fp": "0.00",
                            "initial_count_fp": f"{float(body['count']):.2f}",
                            "taker_fill_cost_dollars": f"{n * no:.4f}", "maker_fill_cost_dollars": "0",
                            "taker_fees_dollars": f"{n * fee:.4f}", "maker_fees_dollars": "0",
                            "created_time": NOW.isoformat()})
        if n:
            self.positions[body["ticker"]] = self.positions.get(body["ticker"], 0) - n
        out = {"order_id": oid, "client_order_id": body["client_order_id"], "fill_count": f"{n:.2f}",
               "remaining_count": "0.00", "ts_ms": 1}
        if n:
            out.update(average_fill_price=f"{yes:.4f}", average_fee_paid=f"{fee:.6f}")
        return out

    def _rest(self, body):
        """A post-only bid that rests untouched."""
        oid = f"oid-{len(self.orders) + 1}"
        self.orders.append({"order_id": oid, "client_order_id": body["client_order_id"], "ticker": body["ticker"],
                            "status": "resting", "outcome_side": "no", "book_side": "ask",
                            "fill_count_fp": "0.00", "remaining_count_fp": f"{float(body['count']):.2f}",
                            "initial_count_fp": f"{float(body['count']):.2f}",
                            "taker_fill_cost_dollars": "0", "maker_fill_cost_dollars": "0",
                            "taker_fees_dollars": "0", "maker_fees_dollars": "0",
                            "created_time": NOW.isoformat()})
        return {"order_id": oid, "client_order_id": body["client_order_id"], "fill_count": "0.00",
                "remaining_count": f"{float(body['count']):.2f}", "ts_ms": 1}

    def maker_fill(self, ticker, n, no_price):
        """Someone sells into our resting NO bid: n contracts at no_price, maker fee 0."""
        o = next(x for x in self.orders if x["ticker"] == ticker and x["status"] == "resting")
        f = float(o["fill_count_fp"]) + n
        o.update(fill_count_fp=f"{f:.2f}", remaining_count_fp=f"{float(o['initial_count_fp']) - f:.2f}",
                 maker_fill_cost_dollars=f"{float(o['maker_fill_cost_dollars']) + n * no_price:.4f}")
        if f >= float(o["initial_count_fp"]):
            o["status"] = "executed"
        self.positions[ticker] = self.positions.get(ticker, 0) - n

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


def _pick(ticker, city, cohort, no_price=0.96, age_min=5, floor=80.0):
    return {"logged_at": (NOW - timedelta(minutes=age_min)).isoformat(timespec="seconds"), "city": city,
            "event": ticker.rsplit("-", 1)[0], "cohort": cohort, "ticker": ticker, "strike_type": "greater",
            "floor": floor, "cap": None, "forecasts": {"gfs": 70.0, "ecmwf": 71.0, "icon": 72.0, "gem": None, "nbm": 70.5},
            "yes_bid": round(1 - no_price, 4), "yes_ask": round(1 - no_price + 0.01, 4), "no_price": no_price,
            "fee_per_contract": 0.0027, "depth_1c": 500, "depth_2c": 800, "late_min": 3, "result": None}


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    (tmp_path / "deploy").mkdir()
    kx.set_root(tmp_path, deploy=tmp_path / "deploy")
    NOTES.clear()
    monkeypatch.setattr(kx, "notify", lambda title, msg, cfg=None: NOTES.append((title, msg)))
    yield tmp_path
    kx.set_root(Path(kx.__file__).resolve().parents[1])


def _key(root):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519
    p = root / "k.pem"
    p.write_bytes(ed25519.Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    os.chmod(p, 0o600)
    return p


def setup(root, picks, env_lines=(), live=True, env_name="demo"):
    lines = list(env_lines)
    if live:
        lines += ["EXEC_MODE=live", f"EXEC_ENV={env_name}", "KALSHI_KEY_ID=test-key-id-123",
                  f"KALSHI_PRIVATE_KEY_PATH={_key(root)}", "KALSHI_BANKROLL_USD=1000"]
    (root / ".env").write_text("\n".join(lines) + "\n")
    (root / "data" / "kalshi_paper.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks))
    ex = FakeExchange()
    for p in picks:
        ex.books[p["ticker"]] = [[f"{1 - p['no_price']:.4f}", "400.00"], [f"{1 - p['no_price'] - 0.01:.4f}", "300.00"]]
        ex.markets[p["ticker"]] = {"ticker": p["ticker"], "status": "active",
                                   "close_time": (NOW + timedelta(hours=20)).isoformat()}
    return ex


def run(root, ex, argv=("--live",), now=NOW):
    c = kx.Kalshi(kx.env(), kx.config(kx.env())["env"], transport=ex.send, sleep=lambda s: None)
    return kx.cmd_run(list(argv), client=c, now=now)


def orders_of(root, mode="demo"):
    return kx.fold(kx.journal(), mode)


# ------------------------------------------------------------------------------------------- gating
def test_dry_unless_both_flag_and_env(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")], live=False)
    run(root, ex)                                   # --live without EXEC_MODE=live
    assert not ex.posts()
    st = orders_of(root, "dry")
    assert st[T1]["status"] == "dry" and st[T1]["count"] > 0
    assert not any("KALSHI-ACCESS-SIGNATURE" in c[4] for c in ex.calls)  # no signed request in dry mode


def test_env_live_but_no_flag_is_dry(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex, argv=())
    assert not ex.posts() and orders_of(root, "dry")[T1]["status"] == "dry"


def test_demo_uses_demo_host_and_prod_uses_prod(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    hosts = []
    send = ex.send
    ex_send = lambda m, u, *a: (hosts.append(urlparse(u).netloc), send(m, u, *a))[1]  # noqa: E731
    c = kx.Kalshi(kx.env(), "demo", transport=ex_send, sleep=lambda s: None)
    kx.cmd_run(["--live"], client=c, now=NOW)
    assert set(hosts) == {"external-api.demo.kalshi.co"}
    assert kx.resolve_mode({"EXEC_MODE": "live", "EXEC_ENV": "prod"}, True) == "live"
    assert kx.resolve_mode({"EXEC_MODE": "live", "EXEC_ENV": "demo"}, True) == "demo"
    assert kx.resolve_mode({"EXEC_MODE": "dry", "EXEC_ENV": "prod"}, True) == "dry"


def test_live_refuses_without_bankroll(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    (root / ".env").write_text((root / ".env").read_text().replace("KALSHI_BANKROLL_USD=1000", ""))
    assert run(root, ex) == 2 and not ex.posts()


# ------------------------------------------------------------------------------------------- idempotency
def test_rerun_never_double_orders(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    run(root, ex, now=NOW + timedelta(minutes=60))
    assert len(ex.posts()) == 1
    body = ex.posts()[0][3]
    assert body["side"] == "ask" and body["time_in_force"] == "immediate_or_cancel" and body["price"] == "0.0400"
    assert body["client_order_id"] == kx.client_order_id("demo", T1)
    assert orders_of(root)[T1]["status"] == "filled"


def test_ambiguous_post_reconciles_without_reordering(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("raise_after_accept", None)] + [("raise", None)] * 3  # first send lands, then retries 409
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] in ("filled", "unknown")      # 409 on retry -> read back, or unknown
    run(root, ex, now=NOW + timedelta(minutes=60))
    o = orders_of(root)[T1]
    assert o["status"] == "filled" and o["filled"] == 1
    assert sum(1 for x in ex.orders if x["ticker"] == T1) == 1


def test_transport_failure_is_unknown_and_counts_against_caps(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("raise", None)] * 4
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] == "unknown" and kx.committed(o) > 0
    assert any("UNKNOWN" in t for t, _ in NOTES)
    # 40 min later nothing appeared on the exchange -> not_placed, but the ticker is still never re-ordered
    run(root, ex, now=NOW + timedelta(minutes=40))
    assert orders_of(root)[T1]["status"] == "not_placed"
    assert len(ex.posts()) == 4  # the 4 attempts of the single original POST, nothing new


def test_existing_exchange_order_blocks_even_with_lost_journal(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.positions[T1] = -3.0
    run(root, ex)
    assert not ex.posts() and orders_of(root)[T1]["status"] == "skip_exchange_has_order"


def test_retry_on_429_then_success(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("http", (429, {"error": "too many requests"})), ("fill", None)]
    run(root, ex)
    assert len(ex.posts()) == 2 and orders_of(root)[T1]["status"] == "filled"
    assert ex.posts()[0][3]["client_order_id"] == ex.posts()[1][3]["client_order_id"]


def test_rejected_order_costs_nothing(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("http", (400, {"code": "invalid_price"}))]
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] == "rejected" and kx.committed(o) == 0
    assert any("REJECTED" in t for t, _ in NOTES)


# ------------------------------------------------------------------------------------------- caps
def test_bankroll_tiers_and_caps_in_dry(root):
    picks = [_pick(T1, "KXHIGHNY", "orig7"), _pick(T2, "KXHIGHTBOS", "x1")]
    ex = setup(root, picks, env_lines=["KALSHI_BANKROLL_USD=1000"], live=False)
    run(root, ex)
    st = orders_of(root, "dry")
    assert st[T1]["cost"] <= 10.0 + 1e-9 and st[T1]["cost"] > 9.0     # 1% of B
    assert st[T2]["cost"] <= 5.0 + 1e-9 and st[T2]["cost"] > 4.0      # 0.5% of B


def test_daily_cap_binds(root):
    tickers = [f"KXHIGHNY-26OCT22-T{80 + i}" for i in range(4)]
    picks = [_pick(t, "KXHIGHNY", "orig7", floor=80.0 + i) for i, t in enumerate(tickers)]
    ex = setup(root, picks, env_lines=["KALSHI_MAX_ORDER_USD=25", "KALSHI_MAX_DAILY_USD=60"], live=False)
    run(root, ex)
    st = orders_of(root, "dry")
    spent = sum(kx.committed(o) for o in st.values())
    assert spent <= 60 + 1e-9
    assert [st[t]["status"] for t in tickers][-1] == "skip_cap"


def test_open_exposure_cap_counts_unsettled_days(root):
    p_old = _pick("KXHIGHNY-26OCT21-T80", "KXHIGHNY", "orig7", age_min=60 * 24)
    ex = setup(root, [p_old, _pick(T1, "KXHIGHNY", "orig7")],
               env_lines=["KALSHI_MAX_ORDER_USD=25", "KALSHI_MAX_OPEN_USD=30"], live=False)
    run(root, ex, now=NOW - timedelta(days=1) + timedelta(minutes=10))   # yesterday: old pick fresh then
    run(root, ex)                                                        # today: only 30 - ~25 left open
    st = orders_of(root, "dry")
    assert kx.committed(st[T1]) <= 30 - kx.committed(st["KXHIGHNY-26OCT21-T80"]) + 1e-9


def test_live_max_contracts_and_depth(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")], env_lines=["KALSHI_MAX_CONTRACTS=50"])
    ex.books[T1] = [["0.0400", "6.00"]]   # depth 6 -> at most 3 at 50% of depth
    run(root, ex)
    assert float(ex.posts()[0][3]["count"]) == 3


def test_settlement_frees_open_exposure(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")], live=False)
    run(root, ex)
    picks = [json.loads(l) for l in (root / "data" / "kalshi_paper.jsonl").read_text().splitlines()]
    picks[0]["result"] = "no"
    (root / "data" / "kalshi_paper.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks))
    run(root, ex, argv=(), now=NOW + timedelta(hours=30))
    o = orders_of(root, "dry")[T1]
    assert o["settled"] and o["pnl"] == pytest.approx(o["filled"] - o["cost"])
    assert kx.spend_and_open(orders_of(root, "dry"), NOW + timedelta(hours=30))[1] == 0


# ------------------------------------------------------------------------------------------- kill / loss
def test_kill_switch_blocks_everything(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    (root / "deploy" / "KILL").write_text("stop\n")
    run(root, ex)
    run(root, ex)
    assert ex.calls == [] and not orders_of(root)
    assert sum("KILLED" in t for t, _ in NOTES) == 1   # one notification per KILL file


def test_kill_command_cancels_our_resting_orders(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    coid = kx.client_order_id("demo", T1)
    kx.append({"event": "order", "ts": NOW.isoformat(), "mode": "demo", "ticker": T1, "status": "resting",
               "client_order_id": coid, "count": 1, "filled": 0, "cost": 0.04})
    ex.orders.append({"order_id": "r1", "client_order_id": coid, "ticker": T1, "status": "resting",
                      "remaining_count_fp": "1.00", "fill_count_fp": "0.00"})
    ex.orders.append({"order_id": "r2", "client_order_id": "someone-else", "ticker": T1, "status": "resting",
                      "remaining_count_fp": "1.00", "fill_count_fp": "0.00"})
    c = kx.Kalshi(kx.env(), "demo", transport=ex.send, sleep=lambda s: None)
    kx.cmd_kill(["test"], client=c)
    assert kx.killed() is not None
    assert [x["status"] for x in ex.orders] == ["canceled", "resting"]


def test_daily_loss_stop(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    kx.append({"event": "order", "ts": (NOW - timedelta(hours=30)).isoformat(), "mode": "demo", "ticker": "KXHIGHNY-26OCT20-T70",
               "status": "filled", "filled": 60, "cost": 57.6, "avg_no": 0.96})
    kx.append({"event": "settle", "ts": (NOW - timedelta(hours=2)).isoformat(), "mode": "demo",
               "ticker": "KXHIGHNY-26OCT20-T70", "result": "yes", "held": 60, "pnl": -57.6})
    run(root, ex)
    assert not ex.posts() and kx.killed() is None   # demo: loss stop blocks, never writes KILL
    assert any(r.get("event") == "loss_stop" for r in kx.journal())


def test_weekly_loss_writes_kill_in_live(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")], env_name="prod")
    for i in range(3):
        t = f"KXHIGHNY-26OCT1{i}-T70"
        kx.append({"event": "order", "ts": (NOW - timedelta(days=4)).isoformat(), "mode": "live", "ticker": t,
                   "status": "filled", "filled": 40, "cost": 38.4, "avg_no": 0.96})
        kx.append({"event": "settle", "ts": (NOW - timedelta(days=3 - i)).isoformat(), "mode": "live", "ticker": t,
                   "result": "yes", "held": 40, "pnl": -38.4})
    ex.positions = {}
    run(root, ex)
    assert kx.killed() is not None and not ex.posts()


# ------------------------------------------------------------------------------------------- fills / guards
def test_partial_fill_records_actual_cost_and_alerts(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")], env_lines=["KALSHI_MAX_CONTRACTS=5"])
    ex.post_script = [("fill", 3)]
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] == "partial" and o["filled"] == 3 and o["count"] == 5
    assert o["cost"] == pytest.approx(3 * 0.96 + o["fee"], abs=1e-4)
    assert any("PARTIAL" in t for t, _ in NOTES)
    # reconcile next run agrees with the exchange position -> no mismatch alert
    run(root, ex, now=NOW + timedelta(minutes=60))
    assert not any(r.get("event") == "reconcile_mismatch" for r in kx.journal())


def test_position_mismatch_alerts(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    ex.positions[T1] = -7.0           # someone traded by hand
    run(root, ex, now=NOW + timedelta(minutes=60))
    assert any(r.get("event") == "reconcile_mismatch" for r in kx.journal())
    assert any("mismatch" in t for t, _ in NOTES)


def test_create_response_convention_off_but_order_object_ok_no_kill(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    orig = ex._accept
    ex._accept = lambda body, n: {**orig(body, n), "average_fill_price": "0.9600"}  # as if NO priced at 0.04
    run(root, ex)
    assert kx.killed() is None
    up = [r for r in kx.journal() if r.get("event") == "update" and r.get("ticker") == T1][-1]
    assert up["source"] == "get_order" and abs(up["avg_no"] - 0.96) < 1e-9


def test_fill_price_convention_violation_kills(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    orig = ex._accept

    def bad(body, n):
        out = {**orig(body, n), "average_fill_price": "0.9600"}
        ex.orders[-1]["taker_fill_cost_dollars"] = "0.0400"     # the Order object disagrees with the ask too
        return out
    ex._accept = bad
    run(root, ex)
    assert kx.killed() is not None


@pytest.mark.parametrize("ask_yes_bid,status", [("0.0200", "skip_price_moved"), ("0.0900", "skip_price_dropped")])
def test_price_sanity(root, ask_yes_bid, status):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.books[T1] = [[ask_yes_bid, "500.00"]]   # NO ask 0.98 (+2c) or 0.91 (-5c) vs pick 0.96
    run(root, ex)
    assert orders_of(root)[T1]["status"] == status and not ex.posts()


def test_stale_pick_and_closed_market_and_rule_check(root):
    stale = _pick(T1, "KXHIGHNY", "orig7", age_min=180)
    bad_rule = _pick(T2, "KXHIGHTBOS", "x1", floor=72.0)   # forecasts 70-72 within 3F -> not rejected
    closing = _pick("KXHIGHNY-26OCT22-T85", "KXHIGHNY", "orig7", floor=85.0)
    ex = setup(root, [stale, bad_rule, closing])
    ex.markets["KXHIGHNY-26OCT22-T85"]["close_time"] = (NOW + timedelta(minutes=20)).isoformat()
    run(root, ex)
    st = orders_of(root)
    assert T1 not in st
    assert st[T2]["status"] == "skip_rule_check"
    assert st["KXHIGHNY-26OCT22-T85"]["status"] == "skip_market_not_active"
    assert not ex.posts()


def test_unknown_series_is_never_traded(root):
    p = _pick("KXHIGHLOWNY-26OCT22-T80", "KXHIGHLOWNY", "orig7")
    ex = setup(root, [p])
    run(root, ex)
    assert not ex.posts() and not orders_of(root)


def test_stale_resting_order_cancelled(root):
    ex = setup(root, [])
    coid = kx.client_order_id("demo", T1)
    kx.append({"event": "order", "ts": (NOW - timedelta(minutes=30)).isoformat(), "mode": "demo", "ticker": T1,
               "status": "resting", "client_order_id": coid, "count": 2, "filled": 0, "cost": 1.92})
    ex.orders.append({"order_id": "r1", "client_order_id": coid, "ticker": T1, "status": "resting",
                      "remaining_count_fp": "2.00", "fill_count_fp": "0.00", "initial_count_fp": "2.00",
                      "created_time": (NOW - timedelta(minutes=30)).isoformat()})
    ex.markets[T1] = {"status": "active"}
    run(root, ex)
    assert ex.orders[0]["status"] == "canceled"
    assert any(r.get("event") == "cancel" for r in kx.journal())


# ------------------------------------------------------------------------------------------- signing
def test_signature_verifies_and_secrets_never_logged(root):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    kp = root / "rsa.pem"
    kp.write_bytes(k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                   serialization.NoEncryption()))
    c = kx.Kalshi({"KALSHI_KEY_ID": "kid-secret-XYZ", "KALSHI_PRIVATE_KEY_PATH": str(kp)}, "demo")
    h = c._sign("GET", "/trade-api/v2/portfolio/orders?limit=5")
    msg = (h["KALSHI-ACCESS-TIMESTAMP"] + "GET" + "/trade-api/v2/portfolio/orders").encode()
    k.public_key().verify(base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"]), msg,
                          padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                          hashes.SHA256())
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    log = (root / "data" / "kalshi_exec.jsonl").read_text()
    pem = Path(kx.env()["KALSHI_PRIVATE_KEY_PATH"]).read_text()
    assert "test-key-id-123" not in log and "PRIVATE KEY" not in log and pem[40:80] not in log
    assert "KALSHI-ACCESS" not in log


def test_fee_rounds_up_to_cent():
    assert kx.fee_total(25, 0.96) == 0.07          # 0.07*25*.96*.04 = 0.0672 -> 0.07
    assert kx.fee_total(1, 0.99) == 0.01
    assert kx.order_cost(10, 0.96) == pytest.approx(9.6 + 0.03)


def test_legacy_dry_rows_fold(root):
    kx.append({"ts": "2026-10-21T03:03:21+00:00", "mode": "dry", "ticker": T1, "pick_no_price": 0.96,
               "no_ask_now": 0.96, "depth_1c": 500, "status": "dry", "count": 25, "cost": 24.0, "limit_cents": 96})
    o = kx.fold(kx.journal(), "dry")[T1]
    assert o["filled"] == 25 and kx.committed(o) == 24.0


# ------------------------------------------------------------------------------------------- maker (M3_N10)
MAKER = ["KALSHI_EXEC_STYLE=maker"]


def maker_setup(root, picks, env_lines=(), **kw):
    ex = setup(root, picks, env_lines=list(MAKER) + list(env_lines), **kw)
    for p in picks:  # NO ask 0.96 (YES bid 0.04), NO bid 0.93 -> mid 0.945 -> floor 0.94 (<= ask - 1c)
        ex.no_books[p["ticker"]] = [["0.9300", "100.00"]]
    return ex


def _settle_paper(root, result):
    picks = [json.loads(l) for l in (root / "data" / "kalshi_paper.jsonl").read_text().splitlines()]
    for p in picks:
        p["result"] = result
    (root / "data" / "kalshi_paper.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks))


def test_taker_path_unchanged_when_style_unset(root):
    ex = setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    body = ex.posts()[0][3]
    assert set(body) == {"ticker", "side", "count", "price", "time_in_force", "self_trade_prevention_type",
                         "client_order_id"} and body["time_in_force"] == "immediate_or_cancel"
    assert not any(c[1].startswith("/series/") for c in ex.calls)            # no extra request on the taker path
    assert not any("exec_style" in r or "maker_q" in r for r in kx.journal())  # journal rows unchanged
    assert kx.config({})["exec_style"] == "taker" and not kx.config({})["ab"]


def test_bad_style_value_refuses():
    with pytest.raises(ValueError):
        kx.config({"KALSHI_EXEC_STYLE": "mkr"})


def test_maker_body_shape_and_resting_row(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    assert len(ex.posts()) == 1
    body = ex.posts()[0][3]
    close = int(datetime.fromisoformat(ex.markets[T1]["close_time"]).timestamp())
    assert body == {"ticker": T1, "side": "ask", "count": "1", "price": "0.0600", "time_in_force": "good_till_canceled",
                    "post_only": True, "expiration_time": close, "self_trade_prevention_type": "taker_at_cross",
                    "client_order_id": kx.client_order_id("demo", T1)}
    o = orders_of(root)[T1]
    assert o["exec_style"] == "maker" and o["maker_q"] == 0.94 and o["fee_type"] == "quadratic"
    assert o["status"] == "resting" and o["filled"] == 0 and kx.committed(o) == pytest.approx(0.94)  # fee 0
    assert kx.held(o) == 0


def test_maker_fee_type_not_quadratic_sizes_with_taker_fee(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.fee_type = "quadratic_with_maker_fees"
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["fee_type"] == "quadratic_with_maker_fees"
    assert kx.committed(o) == pytest.approx(0.94 + kx.fee_total(1, 0.94))


def test_maker_wide_spread_or_empty_no_side_skips(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.no_books[T1] = []   # no NO bids -> mid 0.48 -> 48c below the ask -> skip
    run(root, ex)
    assert orders_of(root)[T1]["status"] == "skip_maker_spread" and not ex.posts()


def test_maker_price_is_mid_floored_and_capped_at_ask_minus_1c(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.no_books[T1] = [["0.9500", "10.00"]]   # 1c spread: mid 0.955 -> 0.95 -> capped at ask-1c = 0.95
    run(root, ex)
    assert orders_of(root)[T1]["maker_q"] == 0.95 and ex.posts()[0][3]["price"] == "0.0500"


def test_resting_maker_not_cancelled_by_stale_sweep(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    run(root, ex, now=NOW + timedelta(hours=3))                # far past KALSHI_STALE_ORDER_MIN, market active
    assert ex.orders[0]["status"] == "resting"
    assert not any(r.get("event") == "cancel" for r in kx.journal())
    o = orders_of(root)[T1]
    assert o["status"] == "resting" and kx.committed(o) == pytest.approx(0.94)  # reserve kept through reconcile
    assert len(ex.posts()) == 1                                # per-ticker 'already ordered' skip still holds
    ex.markets[T1]["status"] = "closed"                        # market no longer active -> now it may be cancelled
    run(root, ex, now=NOW + timedelta(hours=21))
    assert ex.orders[0]["status"] == "canceled"
    assert any(r.get("event") == "cancel" and r.get("why") == "maker_market_closed" for r in kx.journal())
    run(root, ex, now=NOW + timedelta(hours=22))               # next reconcile reads the 0-fill cancel
    o = orders_of(root)[T1]
    assert o["status"] == "no_fill" and kx.committed(o) == 0


def test_taker_resting_order_still_cancelled_when_maker_enabled(root):
    ex = setup(root, [], env_lines=MAKER)
    coid = kx.client_order_id("demo", T1)
    kx.append({"event": "order", "ts": (NOW - timedelta(minutes=30)).isoformat(), "mode": "demo", "ticker": T1,
               "status": "resting", "client_order_id": coid, "count": 2, "filled": 0, "cost": 1.92})
    ex.orders.append({"order_id": "r1", "client_order_id": coid, "ticker": T1, "status": "resting",
                      "remaining_count_fp": "2.00", "fill_count_fp": "0.00", "initial_count_fp": "2.00",
                      "created_time": (NOW - timedelta(minutes=30)).isoformat()})
    ex.markets[T1] = {"status": "active"}
    run(root, ex)
    assert ex.orders[0]["status"] == "canceled"


def test_resting_cost_counted_in_caps(root):
    t2 = "KXHIGHNY-26OCT22-T81"
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7"), _pick(t2, "KXHIGHNY", "orig7", floor=81.0)],
                     env_lines=["KALSHI_MAX_DAILY_USD=1.5"])
    run(root, ex)
    st = orders_of(root)
    assert st[T1]["status"] == "resting" and st[t2]["status"] == "skip_cap"   # 1.5 - 0.94 < 0.94
    assert kx.spend_and_open(st, NOW) == (pytest.approx(0.94), pytest.approx(0.94))
    run(root, ex, now=NOW + timedelta(hours=1))
    assert kx.spend_and_open(orders_of(root), NOW + timedelta(hours=1))[1] == pytest.approx(0.94)


def test_post_only_reject_no_retry_no_fallback(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("http", (400, {"error": {"code": "invalid_order", "message": "post only order would cross"}}))]
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] == "rejected_post_only" and kx.committed(o) == 0
    assert len(ex.posts()) == 1 and ex.posts()[0][3]["post_only"] is True
    run(root, ex, now=NOW + timedelta(minutes=60))
    assert len(ex.posts()) == 1                                 # never re-sent, never re-sent as a taker


def test_post_only_cancelled_at_entry_is_rejected(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.post_script = [("http", (201, {"order_id": "x", "fill_count": "0.00", "remaining_count": "0.00", "ts_ms": 1}))]
    run(root, ex)
    o = orders_of(root)[T1]
    assert o["status"] == "rejected_post_only" and kx.committed(o) == 0


def test_partial_maker_fill_then_cancel_then_settle(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")], env_lines=["KALSHI_MAX_CONTRACTS=5"])
    run(root, ex)
    assert orders_of(root)[T1]["count"] == 5 and kx.committed(orders_of(root)[T1]) == pytest.approx(4.70)
    ex.maker_fill(T1, 2, 0.94)
    run(root, ex, now=NOW + timedelta(hours=1))
    o = orders_of(root)[T1]
    assert o["status"] == "resting" and o["filled"] == 2 and o["avg_no"] == pytest.approx(0.94)
    assert kx.committed(o) == pytest.approx(2 * 0.94 + 3 * 0.94)   # filled cost + remaining reserve
    assert kx.killed() is None and any("fill" in t.lower() for t, _ in NOTES)
    assert not any(r.get("event") == "reconcile_mismatch" for r in kx.journal())
    ex.markets[T1]["status"] = "closed"
    run(root, ex, now=NOW + timedelta(hours=21))                 # cancels the rest
    run(root, ex, now=NOW + timedelta(hours=22))                 # reads the final state
    o = orders_of(root)[T1]
    assert o["status"] == "partial" and o["filled"] == 2 and kx.committed(o) == pytest.approx(1.88)
    _settle_paper(root, "no")
    run(root, ex, now=NOW + timedelta(hours=30))
    o = orders_of(root)[T1]
    assert o["settled"] and o["pnl"] == pytest.approx(2 - 1.88)


def test_resting_row_does_not_settle_before_reconcile(root):
    kx.append({"event": "order", "ts": NOW.isoformat(), "mode": "demo", "ticker": T1, "status": "resting",
               "exec_style": "maker", "maker_q": 0.94, "count": 5, "filled": 2, "cost": 4.70})
    assert kx.settle(kx.journal(), [{"ticker": T1, "result": "no"}], NOW + timedelta(hours=30)) == 0


def test_maker_fill_convention_vs_q_kills(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    ex.maker_fill(T1, 1, 0.06)                                   # NO cost 0.06 vs our bid 0.94 -> wrong convention
    run(root, ex, now=NOW + timedelta(hours=1))
    assert kx.killed() is not None
    assert kx.committed(orders_of(root)[T1]) == pytest.approx(1.0)   # worst case booked


def test_maker_fill_at_q_does_not_kill_even_far_from_ask(root):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    ex.no_books[T1] = [["0.9100", "10.00"]]   # 5c spread: q = floor(0.935) = 0.93, ask 0.96
    run(root, ex)
    ex.maker_fill(T1, 1, 0.93)
    run(root, ex, now=NOW + timedelta(hours=1))
    o = orders_of(root)[T1]
    assert kx.killed() is None and o["status"] == "filled" and o["cost"] == pytest.approx(0.93)


def test_ab_parity_split(root):
    import hashlib
    tickers = [f"KXHIGHNY-26OCT22-T{80 + i}" for i in range(12)]
    picks = [_pick(t, "KXHIGHNY", "orig7", floor=80.0 + i) for i, t in enumerate(tickers)]
    ex = setup(root, picks, env_lines=["KALSHI_AB=1", "KALSHI_MAX_DAILY_USD=100"])
    for t in tickers:
        ex.no_books[t] = [["0.9300", "100.00"]]
    run(root, ex)
    st = orders_of(root)
    want = {t: "maker" if int(hashlib.sha1(t.encode()).hexdigest()[:8], 16) % 2 == 0 else "taker" for t in tickers}
    assert {"maker", "taker"} <= set(want.values())
    for t in tickers:
        assert st[t]["exec_style"] == want[t] == kx.exec_style(kx.config(kx.env()), t)
    for b in (c[3] for c in ex.posts()):
        assert b["time_in_force"] == ("good_till_canceled" if want[b["ticker"]] == "maker" else "immediate_or_cancel")
        assert ("post_only" in b) == (want[b["ticker"]] == "maker")


def test_status_shows_style(root, capsys):
    ex = maker_setup(root, [_pick(T1, "KXHIGHNY", "orig7")])
    run(root, ex)
    capsys.readouterr()
    kx.cmd_status(client=kx.Kalshi(kx.env(), "demo", transport=ex.send, sleep=lambda s: None), now=NOW)
    out = capsys.readouterr().out
    assert "exec style: maker" in out and "maker: 1 placed, 0 filled (0%), 1 resting" in out
