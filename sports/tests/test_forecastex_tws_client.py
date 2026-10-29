"""venues/forecastex_tws_client.py (IB Gateway + ib_async backend) against a fake IB and a mocked public feed.
No network, no socket, no real orders.

Covers: event date -> lastTradeDateOrContractMonth (IB expiration = event day + 1) and strike; YES = right C, NO = right
P; conId cross-check against the public feed; the never-sell rule (order factory is BUY-only, _place refuses SELL, every
placed order is BUY LMT, an exit is a BUY of the opposite right); gating (no --live, Gateway down, wrong account, KILL,
caps); TWS snapshot book incl. complement fill and public fallback; permId as order id + write-ahead journal; order
status normalisation; cancel; fills/positions shape; forecastex_exec backend switch + a full live executor run.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

pytest.importorskip("ib_async")
from ib_async import Contract, Order, OrderStatus  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bet  # noqa: E402
import forecastex_exec as fe  # noqa: E402
from venues import get_venue  # noqa: E402
from venues import forecastex_client as fxc  # noqa: E402
from venues import forecastex_tws_client as tws  # noqa: E402
from venues.base import TradingDisabled, Venue, VenueError  # noqa: E402

MKT = "UHLGA_102926_63"
ACCT = "U1234567"
LIVE_ENV = {"IBKR_ACCOUNT_ID": ACCT, "IBKR_MODE": "live", "FX_ALLOW_LIVE_ACCOUNT": "1", "FX_TWS_QUOTE_WAIT": "0.05",
            "FX_TWS_ACK_WAIT": "0.05"}
NOW = datetime(2026, 10, 29, 3, 0, tzinfo=timezone.utc)
LTT = int(datetime(2026, 10, 30, 3, 59, tzinfo=timezone.utc).timestamp())
TREE = {"categories": {"g": {"markets": [{"symbol": "UHLGA", "exchange": "FORECASTX", "conid": 1}]}}}


def cc(strike, expiration="20261030", by=None, ay=None, bn=None, an=None):
    d = {"strike": strike, "expiration": expiration, "measured_period": "Oct29'26 11:59 PM ET", "open_interest": 0,
         "contracts": [{"conid_yes": 1000 + strike, "conid_no": 2000 + strike, "last_trade_time": LTT}]}
    for k, v in (("bid_yes", by), ("ask_yes", ay), ("bid_no", bn), ("ask_no", an)):
        if v is not None:
            d[k] = v
    return d


class Public:
    """Key-free ForecastTrader feed only; any Client Portal call is a test failure."""
    def __init__(self, rows=None):
        self.rows = rows or [cc(63, by=0.04, ay=0.07, bn=0.93, an=0.96), cc(70)]

    def __call__(self, method, url, headers, params, data, timeout):
        p = urlparse(url).path
        assert "/tws.proxy/public/" in url, f"TWS backend must not call the Client Portal API: {url}"
        if p.endswith("/category/tree"):
            return 200, TREE
        if p.endswith("/prediction/contract/list"):
            return 200, {"complex_contracts": self.rows}
        return 404, {}


class FakeIB:
    def __init__(self, down=False, accounts=(ACCT,)):
        self.down, self.accounts = down, list(accounts)
        self.connected = False
        self.placed = []          # (contract, order)
        self._trades = []
        self._fills = []
        self.portfolio_items = []
        self.next_perm = 900001
        self.assign_perm = True
        self.reject = False
        # (symbol, lastTradeDate, strike, right) -> conId, the Gateway's view of the listing
        self.listing = {("UHLGA", "20261029", 63.0, "C"): 1063, ("UHLGA", "20261029", 63.0, "P"): 2063,
                        ("UHLGA", "20261029", 70.0, "C"): 1070, ("UHLGA", "20261029", 70.0, "P"): 2070}
        self.quotes = {1063: (0.05, 0.08, 7, 9), 2063: (0.92, 0.95, 9, 7), 1070: (0.88, 0.92, 5, 5),
                       2070: (0.08, 0.12, 5, 5)}
        self.detail_requests = []

    def connect(self, host, port, clientId, timeout):
        if self.down:
            raise ConnectionRefusedError(61, "Connect call failed")
        assert (host, port, clientId) == ("127.0.0.1", 4001, 17)
        self.connected = True

    def isConnected(self):
        return self.connected

    def disconnect(self):
        self.connected = False

    def managedAccounts(self):
        return self.accounts

    def reqMarketDataType(self, t):
        self.mdt = t

    def sleep(self, s):
        pass

    def reqContractDetails(self, c):
        self.detail_requests.append(c)
        cid = self.listing.get((c.symbol, c.lastTradeDateOrContractMonth, float(c.strike), c.right))
        if cid is None:
            return []
        return [SimpleNamespace(contract=Contract(conId=cid, symbol=c.symbol, secType="OPT", exchange="FORECASTX",
                                                  currency="USD", strike=c.strike, right=c.right,
                                                  lastTradeDateOrContractMonth=c.lastTradeDateOrContractMonth,
                                                  localSymbol=f"UHLGA_102926_{c.strike:g}_{'YES' if c.right == 'C' else 'NO'}"))]

    def reqMktData(self, c, gen, snapshot, reg):
        assert snapshot is True
        b, a, bs, as_ = self.quotes.get(c.conId, (None, None, None, None))
        nan = math.nan
        return SimpleNamespace(bid=nan if b is None else b, ask=-1.0 if a is None else a,
                               bidSize=nan if bs is None else bs, askSize=nan if as_ is None else as_)

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        order.orderId = len(self.placed)
        st = OrderStatus(orderId=order.orderId, status="Inactive" if self.reject else "Submitted")
        if self.assign_perm and not self.reject:
            order.permId = self.next_perm
            self.next_perm += 1
        t = SimpleNamespace(contract=contract, order=order, orderStatus=st,
                            log=[SimpleNamespace(message="rejected: test" if self.reject else "")])
        self._trades.append(t)
        return t

    def cancelOrder(self, order):
        for t in self._trades:
            if t.order is order:
                t.orderStatus.status = "Cancelled"

    def trades(self):
        return list(self._trades)

    def openTrades(self):
        return [t for t in self._trades if t.orderStatus.status in ("Submitted", "PreSubmitted")]

    def reqAllOpenOrders(self):
        return self.openTrades()

    def reqCompletedOrders(self, apiOnly):
        return [t for t in self._trades if t.orderStatus.status not in ("Submitted", "PreSubmitted")]

    def fills(self):
        return list(self._fills)

    def portfolio(self, account):
        return list(self.portfolio_items)

    def accountSummary(self, account):
        return [SimpleNamespace(tag="NetLiquidation", value="123.45", currency="USD")]


def make(tmp_path, ib=None, live=True, env=None, public=None):
    ib = ib or FakeIB()
    c = tws.ForecastExTWS(env=dict(LIVE_ENV if env is None else env), live=live, transport=public or Public(),
                          sleep=lambda s: None, deploy=tmp_path, journal=tmp_path / "data" / "forecastex_orders.jsonl",
                          ib_factory=lambda: ib)
    return c, ib


# ---------------------------------------------------------------------- contract mapping
def test_event_date_maps_to_expiration_and_rights(tmp_path):
    c, _ = make(tmp_path)
    y, n = c.tws_contract(MKT, "yes"), c.tws_contract(MKT, "no")
    for k in (y, n):
        assert (k.symbol, k.secType, k.exchange, k.currency) == ("UHLGA", "OPT", "FORECASTX", "USD")
        assert k.lastTradeDateOrContractMonth == "20261029" and k.strike == 63.0   # IB lastTradeDate = event day (verified live 10-29)
    assert (y.right, n.right) == ("C", "P")                                       # YES = Call, NO = Put
    with pytest.raises(VenueError):
        c.tws_contract(MKT, "sell")


def test_expiration_fallback_is_event_day_plus_one(tmp_path):
    c, _ = make(tmp_path, public=Public([cc(63, expiration=None)]))
    assert c.tws_contract(MKT, "no").lastTradeDateOrContractMonth == "20261029"


def test_resolve_crosschecks_public_conid(tmp_path):
    c, ib = make(tmp_path)
    assert c.resolve(MKT, "yes").conId == 1063 and c.resolve(MKT, "no").conId == 2063
    c2, ib2 = make(tmp_path / "b")
    ib2.listing[("UHLGA", "20261029", 63.0, "P")] = 9999          # Gateway disagrees with the public feed
    with pytest.raises(VenueError, match="!= public conid"):
        c2.resolve(MKT, "no")
    c3, ib3 = make(tmp_path / "c")
    ib3.listing.clear()
    with pytest.raises(VenueError, match="0 matches"):
        c3.resolve(MKT, "yes")


# ---------------------------------------------------------------------- never sell
def test_order_factory_is_buy_limit_only():
    o = tws.ForecastExTWS.make_order(0.94, 5, tif="GTC", tag="t", account=ACCT)
    assert (o.action, o.orderType, o.totalQuantity, o.lmtPrice, o.tif, o.orderRef, o.account) == \
        ("BUY", "LMT", 5, 0.94, "GTC", "t", ACCT)
    for bad in [dict(price=0.5, qty=1.5), dict(price=0.5, qty=0), dict(price=0.5, qty=1, tif="GTD")]:
        with pytest.raises(VenueError):
            tws.ForecastExTWS.make_order(**bad)
    import inspect
    assert "action" not in inspect.signature(tws.ForecastExTWS.make_order).parameters


def test_place_refuses_sell_and_market(tmp_path):
    c, ib = make(tmp_path)
    con = c.resolve(MKT, "yes")
    with pytest.raises(VenueError, match="BUY only"):
        c._place(con, Order(action="SELL", orderType="LMT", totalQuantity=1, lmtPrice=0.5))
    with pytest.raises(VenueError, match="LMT only"):
        c._place(con, Order(action="BUY", orderType="MKT", totalQuantity=1))
    assert ib.placed == []


def test_live_place_buy_no_then_exit_by_buying_yes(tmp_path):
    c, ib = make(tmp_path)
    r1 = c.place_limit(MKT, "no", 0.92, 5, post_only=True, tag="E5X-a")
    r2 = c.place_limit(MKT, "yes", 0.05, 5, post_only=True, tag="E5X-b")   # exit/hedge = BUY the opposite side
    assert (r1.status, r2.status) == ("placed", "placed")
    assert r1.order_id == "900001" and r2.order_id == "900002"               # permId, not the per-client orderId
    assert [(k.right, k.conId) for k, _ in ib.placed] == [("P", 2063), ("C", 1063)]
    assert all(o.action == "BUY" and o.orderType == "LMT" for _, o in ib.placed)
    assert ib.placed[0][1].tif == "DAY" and ib.placed[0][1].orderRef == "E5X-a" and ib.placed[0][1].account == ACCT
    rows = [json.loads(l) for l in (tmp_path / "data" / "forecastex_orders.jsonl").read_text().splitlines()]
    assert [r["event"] for r in rows] == ["intent", "response", "intent", "response"]
    assert rows[0]["body"]["right"] == "P" and rows[0]["body"]["action"] == "BUY"
    again = c.place_limit(MKT, "no", 0.92, 1, tag="E5X-a")
    assert again.status == "rejected" and "already used" in again.detail and len(ib.placed) == 2


def test_post_only_and_caps(tmp_path):
    c, ib = make(tmp_path)
    r = c.place_limit(MKT, "no", 0.95, 1, post_only=True)          # NO ask 0.95 -> would cross
    assert r.status == "rejected" and "post-only" in r.detail
    r = c.place_limit(MKT, "no", 0.92, 11)                         # > FX_MAX_CONTRACTS 10
    assert r.status == "rejected" and "FX_MAX_CONTRACTS" in r.detail
    assert ib.placed == []


def test_rejected_and_missing_permid(tmp_path):
    c, ib = make(tmp_path)
    ib.reject = True
    r = c.place_limit(MKT, "no", 0.92, 1, tag="x1")
    assert r.status == "rejected" and "Inactive" in r.detail
    c2, ib2 = make(tmp_path / "b")
    ib2.assign_perm = False
    r = c2.place_limit(MKT, "no", 0.92, 1, tag="x2")
    assert r.status == "ambiguous"


# ---------------------------------------------------------------------- gating
def test_gates(tmp_path):
    c, ib = make(tmp_path, live=False)
    with pytest.raises(TradingDisabled, match="--live"):
        c.place_limit(MKT, "no", 0.92, 1)
    assert c.place_limit(MKT, "no", 0.92, 1, dry=True).status == "dry" and ib.placed == []
    c, ib = make(tmp_path / "down", ib=FakeIB(down=True))
    assert c.session_ok() is False
    with pytest.raises(TradingDisabled, match="no IB Gateway API session"):
        c.place_limit(MKT, "no", 0.92, 1)
    c, ib = make(tmp_path / "acct", ib=FakeIB(accounts=["U7654321"]))
    assert c.session_ok() is False and "not among" in c.connect_error
    c, ib = make(tmp_path / "paper", env={**LIVE_ENV, "IBKR_MODE": "paper"})
    with pytest.raises(TradingDisabled, match="paper"):
        c.place_limit(MKT, "no", 0.92, 1)
    (tmp_path / "kill").mkdir()
    (tmp_path / "kill" / "KILL").write_text("x")
    c, ib = make(tmp_path / "kill")
    with pytest.raises(TradingDisabled, match="KILL"):
        c.place_limit(MKT, "no", 0.92, 1)
    assert ib.placed == []


# ---------------------------------------------------------------------- quotes / account
def test_book_from_tws_snapshot_and_fallback(tmp_path):
    c, ib = make(tmp_path)
    b = c.book(MKT)
    assert b.best("yes") == (0.05, 0.08) and b.best("no") == (0.92, 0.95)
    assert b.bids[0][1] == 7 and b.asks[0][1] == 9
    ib.quotes[1063] = (None, None, None, None)                     # YES side empty: complement of NO
    b = c.book(MKT)
    assert b.best("yes") == (0.05, 0.08) and b.bids[0][1] == 7     # YES bid size = NO ask size
    ib.quotes[2063] = (None, None, None, None)                     # nothing from TWS: public top of book
    b = c.book(MKT)
    assert b.best("yes") == (0.04, 0.07) and math.isnan(b.bids[0][1])
    c2, _ = make(tmp_path / "d", ib=FakeIB(down=True))
    assert c2.book(MKT).best("no") == (0.93, 0.96)


def test_status_cancel_fills_positions(tmp_path):
    c, ib = make(tmp_path)
    r = c.place_limit(MKT, "no", 0.92, 5, tag="E5X-UHLGA_102926_63")
    s = c.order_status(r.order_id)
    assert s == {"status": "submitted", "side": "BUY", "conid": 2063, "price": 0.92, "size": 5.0, "filled": 0.0,
                 "avg_price": None}
    with pytest.raises(VenueError):
        c.order_status("123")
    ok, st = c.cancel_order(r.order_id)
    assert ok and st == "cancelled" and c.order_status(r.order_id)["status"] == "cancelled"
    con = c.resolve(MKT, "no")
    ib._fills = [SimpleNamespace(contract=con, execution=SimpleNamespace(
        orderRef="E5X-UHLGA_102926_63", shares=5.0, price=0.92, side="BOT", execId="e1", permId=900001, time=NOW),
        commissionReport=SimpleNamespace(commission=0.05))]
    f = c.fills()
    assert f[0]["order_ref"] == "E5X-UHLGA_102926_63" and f[0]["size"] == 5.0 and f[0]["commission"] == 0.05
    assert c.fills(market="UHLGA_102926_70") == []
    ib.portfolio_items = [SimpleNamespace(contract=con, position=5.0, marketValue=4.6, averageCost=0.93,
                                          unrealizedPNL=-0.05)]
    p = c.positions()
    assert p[0]["conid"] == 2063 and p[0]["side"] == "no" and p[0]["position"] == 5.0
    assert c.balance()["NetLiquidation"] == "123.45"
    assert isinstance(c, Venue) and isinstance(get_venue("forecastex_tws", env={}), tws.ForecastExTWS)


# ---------------------------------------------------------------------- executor backend switch + live run
def test_backend_selection(monkeypatch):
    monkeypatch.delenv("FX_BACKEND", raising=False)
    assert fe.backend_name({}) == "webapi"
    assert fe.backend_name({"FX_BACKEND": "tws"}) == "tws"
    monkeypatch.setenv("FX_BACKEND", "webapi")
    assert fe.backend_name({"FX_BACKEND": "tws"}) == "webapi"            # process env wins
    monkeypatch.setenv("FX_BACKEND", "bogus")
    with pytest.raises(SystemExit):
        fe.backend_name({})
    monkeypatch.setenv("FX_BACKEND", "tws")
    assert isinstance(fe.make_client(False, env={}), tws.ForecastExTWS)
    monkeypatch.setenv("FX_BACKEND", "webapi")
    c = fe.make_client(False, env={})
    assert type(c) is fxc.ForecastEx


def test_executor_live_run_on_tws_backend(tmp_path):
    (tmp_path / "data").mkdir()
    bet.set_root(tmp_path)
    pick = {"kind": "pick", "trial": "FX1-fwd", "ts": "2026-10-29T02:02:00+00:00", "code": "LGA",
            "event_date": "2026-10-29", "ticker": MKT, "side": "NO"}
    (tmp_path / "data" / "forecastex_paper.jsonl").write_text(json.dumps(pick) + "\n")
    c, ib = make(tmp_path)
    out = []
    ex = fe.Exec(live=True, client=c, home=tmp_path, bet_mod=bet, out=out.append)
    assert ex.run(NOW) == 0, out
    assert len(ib.placed) == 1
    k, o = ib.placed[0]
    assert (k.right, k.conId, o.action, o.orderType, o.lmtPrice, o.totalQuantity, o.tif) == \
        ("P", 2063, "BUY", "LMT", 0.92, 5, "GTC")
    assert any("verified" in l for l in out), out
    oid = next(iter(ex.orders()))
    assert oid == "900001"
    assert ex.cancel_all(NOW) == 1 and ib._trades[0].orderStatus.status == "Cancelled"
    assert not (tmp_path / "KILL").exists()
