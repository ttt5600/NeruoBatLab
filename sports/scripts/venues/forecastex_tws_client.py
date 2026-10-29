"""ForecastEx via the TWS API (IB Gateway + IBC + ib_async): the low-maintenance alternative to the Client Portal
Web API client (forecastex_client.py). Same venues/base.py interface, same gating, caps, journal and market ids.

    python scripts/venues/forecastex_tws_client.py preflight            # socket / account / mode / caps / KILL
    python scripts/venues/forecastex_tws_client.py resolve UHLGA_102926_63
    python scripts/venues/forecastex_tws_client.py book UHLGA_102926_63
    python scripts/venues/forecastex_tws_client.py orders | fills | positions

WHY: the Client Portal Gateway needs a browser login + 2FA EVERY DAY. IB Gateway run by IBC logs in by itself
(credentials in ~/ibkr/ibc/config.ini, mode 600), auto-restarts nightly WITHOUT re-authentication (AutoRestartTime) and
needs one IBKR Mobile push approval per WEEK (IBKR expires the session on Sunday; ColdRestartTime). Setup and the
user's steps: notes/ibkr_ibc_setup.md. Selected in forecastex_exec.py by FX_BACKEND=tws (default webapi).

TRANSPORT: ib_async socket to IB Gateway at FX_TWS_HOST:FX_TWS_PORT (127.0.0.1:4001 = Gateway live), fixed client id
FX_TWS_CLIENT_ID (17) so orders placed by an earlier run stay cancellable by a later one (IB only lets the placing client
id cancel). One connection per process, closed at exit.

CONTRACTS (IBKR TWS API docs, event contracts): secType OPT, exchange FORECASTX, currency USD, symbol = the product code
(UHLGA), strike = the threshold, right C = YES and P = NO, lastTradeDateOrContractMonth = the IB EXPIRATION date, which
for the UH* daily-high stations is the day AFTER the event date (public feed, checked 2026-10-29 on LGA/DFW/LAX/PHX).
Discovery stays on the key-free public ForecastTrader feed (inherited from ForecastEx: strikes, conids, expiration,
last trade time); each contract is then resolved with reqContractDetails from (symbol, expiration, strike, right) and
the returned conId MUST equal the public conid of that side, else VenueError (wrong instrument = no order).

ORDERS: ForecastEx has NO sell orders. Every order this client builds is BUY / LMT / whole contracts / TIF DAY|GTC;
the single order factory refuses anything else and _place() re-checks the action before it reaches the socket. An exit
is a BUY of the opposite side (IB nets YES against NO). Prices are in the bought side's own terms (base.py).
order_id returned = IB permId (global, survives reconnects); orderRef = our tag. Post-only is emulated exactly as in
the Web API client (re-read the ask; refuse if crossing).
"""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from venues.base import Book, OrderResult, TradingDisabled, VenueError
    from venues.forecastex_client import DEFAULT_CAPS, DEPLOY, EXCHANGE, ForecastEx, parse_ticker
else:
    from .base import Book, OrderResult, TradingDisabled, VenueError
    from .forecastex_client import DEFAULT_CAPS, DEPLOY, EXCHANGE, ForecastEx, parse_ticker

RIGHT = {"yes": "C", "no": "P"}          # IBKR: Call = YES, Put = NO
SIDE_OF_RIGHT = {"C": "yes", "P": "no"}
ALLOWED_TIF = ("DAY", "GTC", "IOC")
STATUS = {"presubmitted": "submitted", "submitted": "submitted", "pendingsubmit": "submitted",
          "apipending": "submitted", "pendingcancel": "pendingcancel", "apicancelled": "cancelled",
          "cancelled": "cancelled", "filled": "filled", "inactive": "inactive"}


def _px(v) -> float | None:
    """A usable ForecastEx price from a Ticker field (nan / -1 / 0 / >= 1 mean 'no quote')."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if (not math.isnan(f) and 0 < f < 1) else None


def _sz(v) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return math.nan
    return f if (not math.isnan(f) and f >= 0) else math.nan


def _default_ib():
    from ib_async import IB
    return IB()


class ForecastExTWS(ForecastEx):
    name = "forecastex"
    backend = "tws"

    def __init__(self, env: dict | None = None, live: bool = False, transport=None, sleep=time.sleep,
                 deploy: Path = DEPLOY, journal: Path | None = None, ib_factory=None):
        super().__init__(env=env, live=live, transport=transport, sleep=sleep, deploy=deploy, journal=journal)
        self.host = self.env.get("FX_TWS_HOST", "127.0.0.1")
        self.port = int(self.env.get("FX_TWS_PORT", "4001"))
        self.client_id = int(self.env.get("FX_TWS_CLIENT_ID", "17"))
        self.quote_wait = float(self.env.get("FX_TWS_QUOTE_WAIT", "4"))
        self.ack_wait = float(self.env.get("FX_TWS_ACK_WAIT", "10"))
        self.mkt_data_type = int(self.env.get("FX_TWS_MKT_DATA_TYPE", "1"))
        self._ib_factory = ib_factory or _default_ib
        self._ib = None
        self._resolved: dict[tuple[str, str], object] = {}
        self.connect_error: str | None = None

    # ------------------------------------------------------------------ connection
    @property
    def ib(self):
        if self._ib is None or not self._ib.isConnected():
            ib = self._ib or self._ib_factory()
            try:
                ib.connect(self.host, self.port, clientId=self.client_id, timeout=8)
            except Exception as e:  # noqa: BLE001 -- refused / timeout / API disabled all mean "no session"
                self.connect_error = f"{type(e).__name__}: {e}"
                raise TradingDisabled(f"IB Gateway not reachable at {self.host}:{self.port} ({self.connect_error}); "
                                      "is the IBC agent running and logged in? (~/ibkr/install_ibc_agent.sh status)")
            self._ib = ib
            try:
                ib.reqMarketDataType(self.mkt_data_type)
            except Exception:  # noqa: BLE001
                pass
            atexit.register(self.close)
        return self._ib

    def close(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def session_ok(self, refresh: bool = False) -> bool:
        if self._session_ok is None or refresh:
            try:
                ib = self.ib
                accts = ib.managedAccounts() or []
                self._session_ok = bool(ib.isConnected() and accts)
                if self._session_ok and self.account and self.account not in accts:
                    self.connect_error = f"IBKR_ACCOUNT_ID {self.account} not among the Gateway's accounts"
                    self._session_ok = False
            except TradingDisabled:
                self._session_ok = False
        return self._session_ok

    def _require_session(self):
        if not self.session_ok():
            raise TradingDisabled(f"no IB Gateway API session ({self.connect_error or 'not logged in'})")
        if not self.account:
            raise TradingDisabled("IBKR_ACCOUNT_ID not set in .env")

    # ------------------------------------------------------------------ contracts
    @staticmethod
    def expiration_for(c: dict) -> str:
        """lastTradeDateOrContractMonth for a public contract row: its IB expiration, else event day + 1."""
        exp = c.get("expiration")
        if exp:
            return str(exp)
        return (c["day"] + timedelta(days=1)).strftime("%Y%m%d")

    def tws_contract(self, market: str, side: str):
        """The un-qualified ib_async Contract for one side of our market id (no network)."""
        from ib_async import Contract
        if side not in RIGHT:
            raise VenueError(f"side must be yes/no, got {side!r}")
        code, day, strike = parse_ticker(market)
        c = self.contract(market)
        if c is None:
            raise VenueError(f"{market} not listed on the public feed")
        return Contract(symbol=f"UH{code}", secType="OPT", exchange=EXCHANGE, currency="USD",
                        lastTradeDateOrContractMonth=self.expiration_for(c), strike=float(strike), right=RIGHT[side])

    def resolve(self, market: str, side: str):
        """Qualified Contract via reqContractDetails; conId must equal the public conid of that side."""
        key = (market, side)
        if key in self._resolved:
            return self._resolved[key]
        want = self.tws_contract(market, side)
        pub = self.contract(market) or {}
        pub_id = pub.get("conid_yes" if side == "yes" else "conid_no")
        det = self.ib.reqContractDetails(want) or []
        if len(det) != 1:
            raise VenueError(f"reqContractDetails {market} {side}: {len(det)} matches for "
                             f"{want.symbol} {want.lastTradeDateOrContractMonth} {want.strike} {want.right}")
        con = det[0].contract
        if con.right != want.right or abs(float(con.strike) - float(want.strike)) > 1e-9:
            raise VenueError(f"{market} {side}: resolved {con.right} {con.strike} != {want.right} {want.strike}")
        if pub_id and int(con.conId) != int(pub_id):
            raise VenueError(f"{market} {side}: TWS conId {con.conId} != public conid {pub_id}")
        con.exchange = EXCHANGE
        self._resolved[key] = con
        return con

    # ------------------------------------------------------------------ quotes
    def quotes(self, contracts: list) -> list[dict]:
        """Snapshot bid/ask/sizes per contract; waits up to quote_wait s for the first values."""
        ib = self.ib
        tks = [ib.reqMktData(c, "", True, False) for c in contracts]
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.quote_wait:
            if all(_px(t.bid) is not None or _px(t.ask) is not None for t in tks):
                break
            ib.sleep(0.25)
        return [{"bid": _px(t.bid), "ask": _px(t.ask), "bid_size": _sz(t.bidSize), "ask_size": _sz(t.askSize)}
                for t in tks]

    def book(self, market: str) -> Book | None:
        c = self.contract(market)
        if c is None:
            return None
        if self.session_ok():
            y, n = self.quotes([self.resolve(market, "yes"), self.resolve(market, "no")])
            if any(v is not None for v in (y["bid"], y["ask"], n["bid"], n["ask"])):
                b = self.book_from_quotes(market, y["bid"], y["ask"], n["bid"], n["ask"])
                if b.bids:
                    b.bids = [(b.bids[0][0], y["bid_size"] if y["bid"] is not None else n["ask_size"])]
                if b.asks:
                    b.asks = [(b.asks[0][0], y["ask_size"] if y["ask"] is not None else n["bid_size"])]
                return b
        # no session or no TWS quote yet: the public top of book (no sizes)
        return self.book_from_quotes(market, c["bid_yes"], c["ask_yes"], c["bid_no"], c["ask_no"])

    # ------------------------------------------------------------------ account (read-only)
    def _ours(self, contract) -> bool:
        return (getattr(contract, "exchange", "") == EXCHANGE or getattr(contract, "primaryExchange", "") == EXCHANGE
                or str(getattr(contract, "symbol", "")).startswith("UH"))

    def _ids(self, market: str | None) -> set | None:
        if not market:
            return None
        c = self.contract(market) or {}
        return {int(x) for x in (c.get("conid_yes"), c.get("conid_no")) if x}

    def balance(self) -> dict:
        self._require_session()
        return {v.tag: v.value for v in self.ib.accountSummary(self.account) if v.currency in ("USD", "BASE", "")}

    def positions(self, market: str | None = None) -> list[dict]:
        self._require_session()
        ids = self._ids(market)
        out = []
        for p in self.ib.portfolio(self.account):
            if not self._ours(p.contract) or (ids is not None and int(p.contract.conId) not in ids):
                continue
            out.append({"conid": int(p.contract.conId), "contractDesc": p.contract.localSymbol or p.contract.symbol,
                        "side": SIDE_OF_RIGHT.get(p.contract.right), "position": p.position,
                        "avgCost": p.averageCost, "mktValue": p.marketValue, "unrealizedPnl": p.unrealizedPNL})
        return out

    @staticmethod
    def _trade_row(t) -> dict:
        o, s, c = t.order, t.orderStatus, t.contract
        return {"orderId": str(o.permId or ""), "client_order_id": o.orderId, "order_ref": o.orderRef or "",
                "conid": int(c.conId or 0), "ticker": c.localSymbol or c.symbol, "side": o.action,
                "right": c.right, "price": o.lmtPrice, "size": o.totalQuantity, "tif": o.tif,
                "status": STATUS.get(str(s.status).lower(), str(s.status).lower()), "filled": s.filled,
                "avg_price": s.avgFillPrice or None}

    def open_orders(self, market: str | None = None) -> list[dict]:
        self._require_session()
        ids = self._ids(market)
        try:
            trades = self.ib.reqAllOpenOrders()
        except Exception:  # noqa: BLE001
            trades = self.ib.openTrades()
        return [self._trade_row(t) for t in trades
                if self._ours(t.contract) and (ids is None or int(t.contract.conId) in ids)]

    def fills(self, market: str | None = None, limit: int = 100) -> list[dict]:
        self._require_session()
        ids = self._ids(market)
        rows = []
        for f in self.ib.fills():
            if not self._ours(f.contract) or (ids is not None and int(f.contract.conId) not in ids):
                continue
            e, cr = f.execution, f.commissionReport
            rows.append({"order_ref": e.orderRef or "", "size": e.shares, "price": e.price,
                         "commission": cr.commission if cr and cr.commission and cr.commission < 1e9 else 0.0,
                         "conid": int(f.contract.conId), "side": e.side, "exec_id": e.execId,
                         "perm_id": str(e.permId), "time": str(e.time)})
        return rows[:limit]

    def _find_trade(self, order_id: str):
        pid = int(order_id)
        for t in self.ib.trades():
            if int(t.order.permId or 0) == pid:
                return t
        for fetch in (self.ib.reqAllOpenOrders, lambda: self.ib.reqCompletedOrders(False)):
            try:
                for t in fetch() or []:
                    if int(t.order.permId or 0) == pid:
                        return t
            except Exception:  # noqa: BLE001
                continue
        return None

    def order_status(self, order_id: str) -> dict:
        """Normalised {status, side, conid, price, size, filled, avg_price} (forecastex_exec.verify/reconcile)."""
        self._require_session()
        t = self._find_trade(order_id)
        if t is None:
            raise VenueError(f"order permId {order_id} not found on the Gateway")
        r = self._trade_row(t)
        return {"status": r["status"], "side": str(r["side"]).upper(), "conid": r["conid"],
                "price": float(r["price"]) if r["price"] is not None else None, "size": float(r["size"]),
                "filled": float(r["filled"] or 0.0), "avg_price": r["avg_price"]}

    def cancel_order(self, order_id: str) -> tuple[bool, str]:
        """Cancel one order by permId. Needs a session only (cancelling removes risk; no --live / KILL gate)."""
        self._require_session()
        t = self._find_trade(order_id)
        if t is None:
            return False, f"permId {order_id} not found"
        st = STATUS.get(str(t.orderStatus.status).lower(), "")
        if st in ("cancelled", "filled", "inactive"):
            return st == "cancelled", f"already {st}"
        self.ib.cancelOrder(t.order)
        for _ in range(20):
            self.ib.sleep(0.25)
            st = STATUS.get(str(t.orderStatus.status).lower(), "")
            if st in ("cancelled", "pendingcancel", "filled", "inactive"):
                break
        return st in ("cancelled", "pendingcancel"), st

    # ------------------------------------------------------------------ orders
    @staticmethod
    def make_order(price: float, qty: float, tif: str = "DAY", tag: str | None = None, account: str | None = None):
        """THE order factory: BUY / LMT / whole contracts. There is no parameter for the action on purpose."""
        from ib_async import Order
        if tif not in ALLOWED_TIF:
            raise VenueError(f"tif {tif} not in {ALLOWED_TIF}")
        if qty < 1 or int(qty) != qty:
            raise VenueError(f"qty {qty} must be a whole number of contracts >= 1")
        o = Order(action="BUY", orderType="LMT", totalQuantity=int(qty), lmtPrice=round(float(price), 2), tif=tif,
                  orderRef=tag or "", transmit=True)
        if account:
            o.account = account
        return o

    def _place(self, contract, order):
        if order.action != "BUY":  # ForecastEx: never sell -- exit by buying the opposite side
            raise VenueError(f"refusing {order.action}: ForecastEx orders are BUY only")
        if order.orderType != "LMT":
            raise VenueError(f"refusing orderType {order.orderType}: LMT only")
        return self.ib.placeOrder(contract, order)

    def place_limit(self, market: str, side: str, price: float, qty: float, post_only: bool = True,
                    expire: datetime | None = None, dry: bool = False, tag: str | None = None) -> OrderResult:
        now = datetime.now(timezone.utc)
        base = dict(venue=self.name, market=market, side=side, price=price, qty=qty, ts=now)
        if not dry:
            self._require_trading()
        if tag and any(r.get("tag") == tag and r.get("event") == "intent" for r in self._journal_rows()):
            return OrderResult(**base, status="rejected", detail=f"tag {tag} already used (one decision per tag)")
        try:
            chk = self.check_order(market, side, price, qty, post_only, now)
            con = self.resolve(market, side) if not dry or self.session_ok() else self.tws_contract(market, side)
            order = self.make_order(price, qty, tif="GTC" if expire else "DAY", tag=tag, account=self.account)
        except (VenueError, TradingDisabled) as e:
            return OrderResult(**base, status="rejected", detail=str(e))
        body = {"backend": "tws", "symbol": con.symbol, "lastTradeDateOrContractMonth": con.lastTradeDateOrContractMonth,
                "strike": con.strike, "right": con.right, "conid": getattr(con, "conId", 0) or chk["conid"],
                "action": order.action, "orderType": order.orderType, "lmtPrice": order.lmtPrice,
                "totalQuantity": order.totalQuantity, "tif": order.tif, "orderRef": order.orderRef}
        if dry:
            return OrderResult(**base, status="dry", detail=json.dumps(body))
        self._append({"event": "intent", "ts": now.isoformat(), "tag": tag, "market": market, "side": side,
                      "price": price, "qty": qty, "usd": chk["usd"], "mode": self.mode, "account": self.account,
                      "body": body})
        try:
            trade = self._place(con, order)
        except VenueError as e:
            self._append({"event": "response", "ts": now.isoformat(), "tag": tag, "error": str(e)})
            return OrderResult(**base, status="rejected", detail=str(e))
        except Exception as e:  # noqa: BLE001 -- socket died mid-send: outcome unknown, never re-sent
            self._append({"event": "ambiguous", "ts": now.isoformat(), "tag": tag, "detail": str(e)})
            return OrderResult(**base, status="ambiguous", detail=str(e))
        t0 = time.monotonic()
        while time.monotonic() - t0 < self.ack_wait:  # permId arrives with the first openOrder/orderStatus
            st = STATUS.get(str(trade.orderStatus.status).lower(), "")
            if trade.order.permId or st in ("cancelled", "inactive"):
                break
            self.ib.sleep(0.25)
        st = STATUS.get(str(trade.orderStatus.status).lower(), "")
        msgs = "; ".join(str(getattr(e, "message", "")) for e in (trade.log or []) if getattr(e, "message", ""))
        self._append({"event": "response", "ts": now.isoformat(), "tag": tag, "perm_id": trade.order.permId,
                      "order_id": trade.order.orderId, "status": trade.orderStatus.status, "log": msgs[:300]})
        if st in ("cancelled", "inactive"):
            return OrderResult(**base, status="rejected", detail=f"{trade.orderStatus.status}: {msgs}"[:300])
        if not trade.order.permId:
            return OrderResult(**base, status="ambiguous", detail=f"no permId after {self.ack_wait:g} s (status "
                               f"{trade.orderStatus.status}); order may be live: check the Gateway")
        return OrderResult(**base, status="placed", order_id=str(trade.order.permId), detail=trade.orderStatus.status)

    def cancel(self, order_id: str, market: str) -> bool:
        self._require_trading()
        ok, detail = self.cancel_order(order_id)
        self._append({"event": "cancel", "ts": datetime.now(timezone.utc).isoformat(), "order_id": order_id,
                      "ok": ok, "detail": detail})
        return ok


# ---------------------------------------------------------------------- CLI (read-only)
def main(argv: list[str]) -> int:
    c = ForecastExTWS()
    args = [a for a in argv if not a.startswith("--")]
    cmd = args[0] if args else ""
    try:
        if cmd == "preflight":
            ok = c.session_ok()
            print(f"IB Gateway {c.host}:{c.port} clientId {c.client_id}: "
                  f"{'CONNECTED' if ok else 'DOWN (' + str(c.connect_error) + ')'}; mode {c.mode}; account "
                  f"{(c.account or 'unset')[:2]}***; KILL {c.killed() or 'none'}")
            if ok:
                print("Gateway accounts:", ", ".join(a[:2] + "***" for a in c.ib.managedAccounts()))
            print("caps: " + ", ".join(f"{k}={c.cap(k):g}" for k in DEFAULT_CAPS))
            return 0 if ok else 1
        if cmd == "resolve" and len(args) > 1:
            for side in ("yes", "no"):
                print(side, c.tws_contract(args[1], side), "->", c.resolve(args[1], side).conId)
            return 0
        if cmd == "book" and len(args) > 1:
            b = c.book(args[1])
            print(b, "\nYES bid/ask", b.best("yes") if b else None, " NO bid/ask", b.best("no") if b else None)
            return 0
        if cmd in ("orders", "fills", "positions"):
            rows = {"orders": c.open_orders, "fills": c.fills, "positions": c.positions}[cmd]()
            for r in rows:
                print(r)
            print(f"{len(rows)} {cmd}")
            return 0
    except TradingDisabled as e:
        print(f"DOWN: {e}")
        return 1
    finally:
        c.close()
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
