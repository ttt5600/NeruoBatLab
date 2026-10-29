"""scripts/forecastex_exec.py (E5 executor) against a mocked IBKR gateway: no network, no real orders.

Covers: dry-run prints and sends/writes nothing; side + price at the side's best bid; skip on no bid / cross / cost>=1;
sizing under FX caps + tier B + remaining daily cap; live placement journals to exec + bet.py (venue forecastex, tier B,
strategy E5) with a BUY LMT GTC body on the bought side's conid; one decision per ticker; KILL; client gate; verify
mismatch -> cancel + KILL; daily loss stop; reconcile fill -> settle; cutoff expiry; stale picks.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bet  # noqa: E402
import forecastex_exec as fe  # noqa: E402
from venues import forecastex_client as fxc  # noqa: E402

NOW = datetime(2026, 10, 29, 3, 0, tzinfo=timezone.utc)
LTT = int(datetime(2026, 10, 30, 3, 59, tzinfo=timezone.utc).timestamp())
LIVE_ENV = {"IBKR_ACCOUNT_ID": "U1234567", "IBKR_MODE": "live", "FX_ALLOW_LIVE_ACCOUNT": "1"}
TREE = {"categories": {"g": {"markets": [{"symbol": "UHLGA", "exchange": "FORECASTX", "conid": 1}]}}}


def cc(strike):
    return {"strike": strike, "measured_period": "Oct29'26 11:59 PM ET", "open_interest": 0,
            "contracts": [{"conid_yes": 1000 + strike, "conid_no": 2000 + strike, "last_trade_time": LTT}]}


class Gateway:
    def __init__(self):
        self.calls = []
        # conid -> (bid, ask, bid_size, ask_size)
        self.quotes = {1063: (0.05, None, 10, None), 2063: (0.94, None, 10, None),
                       1070: (0.88, 0.92, 5, 5), 2070: (0.08, 0.12, 5, 5)}
        self.status = {}          # order_id -> status dict override
        self.trades = []
        self.next_id = 500

    def __call__(self, method, url, headers, params, data, timeout):
        p = urlparse(url).path
        body = json.loads(data) if data else None
        self.calls.append((method, p, body))
        if p.endswith("/category/tree"):
            return 200, TREE
        if p.endswith("/prediction/contract/list"):
            return 200, {"complex_contracts": [cc(63), cc(70)]}
        if p.endswith("/iserver/auth/status"):
            return 200, {"authenticated": True}
        if p.endswith("/marketdata/snapshot"):
            out = []
            for c in params["conids"].split(","):
                b, a, bs, as_ = self.quotes.get(int(c), (None, None, None, None))
                out.append({"conid": int(c), "84": b, "86": a, "88": bs, "85": as_})
            return 200, out
        if method == "POST" and p.endswith("/orders"):
            self.next_id += 1
            o = body["orders"][0]
            self.status[str(self.next_id)] = {"order_status": "Submitted", "side": "BUY", "conid": o["conid"],
                                              "limit_price": o["price"], "size": o["quantity"], "cum_fill": 0}
            return 200, [{"order_id": str(self.next_id), "order_status": "Submitted"}]
        if method == "GET" and "/order/status/" in p:
            return 200, self.status[p.rsplit("/", 1)[1]]
        if p.endswith("/iserver/account/trades"):
            return 200, self.trades
        if p.endswith("/iserver/account/orders"):
            return 200, {"orders": []}
        if method == "DELETE":
            oid = p.rsplit("/", 1)[1]
            if oid in self.status:
                self.status[oid]["order_status"] = "Cancelled"
            return 200, {"msg": "Request was submitted", "order_id": oid}
        if p.endswith("/positions/0"):
            return 200, []
        return 404, {"error": "not mocked"}

    def sent(self, method, frag):
        return [c for c in self.calls if c[0] == method and frag in c[1]]


def pick(ticker="UHLGA_102926_63", side="NO", ts="2026-10-29T02:02:00+00:00", **kw):
    return {"kind": "pick", "trial": "FX1-fwd", "ts": ts, "code": "LGA", "event_date": "2026-10-29",
            "ticker": ticker, "side": side, **kw}


@pytest.fixture
def home(tmp_path):
    (tmp_path / "data").mkdir()
    bet.set_root(tmp_path)
    return tmp_path


def write_picks(home, *picks):
    (home / "data" / "forecastex_paper.jsonl").write_text("".join(json.dumps(p) + "\n" for p in picks))


def make(home, gw, live=False, env=None):
    c = fxc.ForecastEx(env=env if env is not None else dict(LIVE_ENV), live=live, transport=gw,
                       sleep=lambda s: None, deploy=home, journal=home / "data" / "forecastex_orders.jsonl")
    lines = []
    return fe.Exec(live=live, client=c, home=home, bet_mod=bet if live else None, out=lines.append), lines


def test_dry_run_prints_and_sends_nothing(home):
    write_picks(home, pick())
    gw = Gateway()
    ex, out = make(home, gw)
    assert ex.run(NOW) == 0
    would = [l for l in out if "WOULD" in l]
    assert len(would) == 1 and "BUY NO" in would[0] and "@ 0.94" in would[0] and " 5 x UHLGA_102926_63" in would[0]
    assert "$4.75" in would[0]
    assert not gw.sent("POST", "/orders") and not gw.sent("DELETE", "")
    assert not (home / "data" / "forecastex_exec.jsonl").exists() and not (home / "data" / "bets.jsonl").exists()


@pytest.mark.parametrize("quotes,reason", [
    ({2063: (None, None, None, None), 1063: (None, None, None, None)}, "no bid"),
    ({2063: (0.94, 0.94, 1, 1), 1063: (0.06, 0.06, 1, 1)}, "would cross"),
    ({2063: (0.99, None, 1, None), 1063: (None, None, None, None)}, ">= 1"),
])
def test_skips(home, quotes, reason):
    write_picks(home, pick())
    gw = Gateway()
    gw.quotes.update(quotes)
    ex, out = make(home, gw)
    ex.run(NOW)
    assert any("SKIP" in l and reason in l for l in out), out


def test_yes_pick_uses_yes_side_and_maxes_contracts(home):
    write_picks(home, pick("UHLGA_102926_70", side="YES"))
    gw = Gateway()
    gw.quotes[1070] = (0.30, 0.35, 5, 5)
    gw.quotes[2070] = (0.65, 0.70, 5, 5)
    ex, out = make(home, gw)
    ex.run(NOW)
    w = [l for l in out if "WOULD" in l][0]
    assert "BUY YES" in w and "@ 0.30" in w and "10 x" in w and "$3.10" in w  # 10-contract cap binds, not $5


def test_daily_cap_shrinks_then_stops(home):
    ex, _ = make(home, Gateway())
    assert ex.size(0.94, 0.0) == (5, 4.75)
    assert ex.size(0.94, 23.0) == (2, 1.9)
    assert ex.size(0.94, 24.5) == (0, 0.0)


def test_live_places_verifies_and_journals_once(home):
    write_picks(home, pick())
    gw = Gateway()
    ex, out = make(home, gw, live=True)
    assert ex.run(NOW) == 0
    posts = gw.sent("POST", "/orders")
    assert len(posts) == 1
    o = posts[0][2]["orders"][0]
    assert o == {"conid": 2063, "orderType": "LMT", "side": "BUY", "price": 0.94, "quantity": 5, "tif": "GTC",
                 "cOID": "E5X-UHLGA_102926_63"}
    assert any("verified" in l for l in out)
    b = [json.loads(l) for l in (home / "data" / "bets.jsonl").read_text().splitlines()]
    assert b[0]["venue"] == "forecastex" and b[0]["tier"] == "B" and b[0]["strategy"] == "E5"
    assert b[0]["stake_usd"] == 4.75 and b[0]["status"] == "placed"
    kinds = [json.loads(l)["kind"] for l in (home / "data" / "forecastex_exec.jsonl").read_text().splitlines()]
    assert kinds == ["decision", "order", "verify"]
    ex2, _ = make(home, gw, live=True)
    ex2.run(NOW)
    assert len(gw.sent("POST", "/orders")) == 1     # decided ticker never re-ordered


def test_kill_file_blocks(home):
    write_picks(home, pick())
    (home / "KILL").write_text("x")
    gw = Gateway()
    ex, out = make(home, gw, live=True)
    ex.run(NOW)
    assert not gw.sent("POST", "/orders") and "KILL" in out[0]


def test_client_gate_paper_mode_blocks_live_account(home):
    write_picks(home, pick())
    gw = Gateway()
    ex, out = make(home, gw, live=True, env={"IBKR_ACCOUNT_ID": "U1234567"})  # IBKR_MODE defaults to paper
    assert ex.run(NOW) == 1 and "NOT TRADING" in out[0]
    assert not gw.sent("POST", "/orders")


def test_verify_mismatch_cancels_and_kills(home):
    write_picks(home, pick())
    gw = Gateway()
    orig = gw.__call__

    def tamper(method, url, headers, params, data, timeout):
        code, out = orig(method, url, headers, params, data, timeout)
        if "/order/status/" in url:
            out = dict(out, limit_price=0.95)
        return code, out
    ex, out = make(home, gw, live=True)
    ex.c.send = tamper
    assert ex.run(NOW) == 2
    assert gw.sent("DELETE", "/order/") and (home / "KILL").exists()
    assert any("VERIFY FAILED" in l for l in out)


def test_loss_stop_halts(home):
    write_picks(home, pick())
    ex, out = make(home, Gateway(), live=True)
    ex.log({"kind": "order", "ts": "2026-10-28T03:00:00+00:00", "order_id": "1", "tag": "E5X-x", "ticker": "x",
            "side": "no", "price": 0.94, "qty": 10, "cost": 9.5, "cutoff": "2026-10-29T03:44:00+00:00"})
    ex.log({"kind": "settle", "ts": "2026-10-28T20:00:00+00:00", "order_id": "1", "result": "loss", "pnl_usd": -10.0})
    gw = ex.c.send
    ex.run(NOW)
    assert any("HALT" in l for l in out) and not gw.sent("POST", "/orders")


def test_reconcile_fill_then_settle_win(home):
    write_picks(home, pick())
    gw = Gateway()
    ex, out = make(home, gw, live=True)
    ex.run(NOW)
    oid = next(iter(ex.orders()))
    gw.status[oid].update(order_status="Filled", cum_fill=5, average_price=0.94)
    gw.trades = [{"order_ref": "E5X-UHLGA_102926_63", "size": "5", "price": "0.94", "commission": "0.05"}]
    assert ex.reconcile(NOW) == 1
    write_picks(home, pick(yes_won=0.0))
    assert ex.settle(NOW) == 1
    o = ex.orders()[oid]
    assert o["result"] == "win" and o["pnl_usd"] == round(5 - 5 * 0.94 - 0.05, 2)
    bb = bet.bets()[f"fx-{oid}"]
    assert bb["settled"] and bb["pnl_usd"] == 0.25 and bb["stake_usd"] == 4.75


def test_unfilled_cancelled_order_settles_void(home):
    write_picks(home, pick())
    gw = Gateway()
    ex, _ = make(home, gw, live=True)
    ex.run(NOW)
    later = datetime(2026, 10, 30, 3, 50, tzinfo=timezone.utc)   # past cutoff (03:44)
    assert ex.expire(later) == 1 and gw.sent("DELETE", "/order/")
    ex.reconcile(later)
    assert ex.settle(later) == 1
    assert list(ex.orders().values())[0]["result"] == "void-unfilled"


def test_stale_and_past_cutoff_picks(home):
    write_picks(home, pick(ts="2026-10-28T18:00:00+00:00"))
    ex, out = make(home, Gateway())
    ex.run(NOW)
    assert "0 due picks" in out[0]
    write_picks(home, pick(ts="2026-10-30T01:00:00+00:00"))
    ex.run(datetime(2026, 10, 30, 3, 50, tzinfo=timezone.utc))
    assert any("past order cutoff" in l for l in out)
