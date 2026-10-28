"""ForecastEx (Interactive Brokers) event contracts behind the venues/base.py interface.

    python scripts/venues/forecastex_client.py stations              # our 24 UH* daily-high markets + underlying conids (public)
    python scripts/venues/forecastex_client.py book UHLGA_102926_63  # live YES/NO top of book (public, or session if up)
    python scripts/venues/forecastex_client.py preflight             # session / account / mode / caps / KILL status

WHICH API AND WHY: IBKR's **Web API (Client Portal API)** -- it is the only IBKR API with a dedicated forecast-contract
endpoint family (`/forecast/contract/market`, `/forecast/contract/details`, `/forecast/category/tree`; docs:
interactivebrokers.com/docs/web-api/.../event-contracts), it is plain REST (same `requests` stack as our other venue
clients, no Java socket client, no ibapi/ib_async dependency), and the same `/forecasttrader/...` routes are served
key-free under `www.interactivebrokers.com/tws.proxy/public/forecasttrader/...`, which ForecastTrader's own page uses.
TWS API (ibapi / ib_async + IB Gateway) also supports ForecastEx (secType OPT, exchange FORECASTX, right C=YES / P=NO),
but needs a running TWS/Gateway socket session and a heavier client for no extra capability here.
Transport target: the Client Portal Gateway on this Mac (`https://localhost:5000/v1/api`, self-signed cert; the user logs
in through its browser page, incl. 2FA). Base URL overridable with IBKR_API_BASE in .env.

FORECASTEX MODEL (IBKR docs): every strike has two separately-conid'd instruments, YES (=Call) and NO (=Put). There are
NO sell orders: you only BUY YES or BUY NO, and a position is reduced by buying the opposite side (IB nets them).
Limit orders only, TIF DAY/GTC/IOC. There is no exchange post-only flag; `post_only=True` is EMULATED here: the order is
refused if its price would cross the side's current ask (re-read immediately before sending). A race between that read
and the order arriving can still make it a taker; stated, not corrected.

MARKET ID (ours) = the ForecastEx daily-file ticker `UH<CODE>_<MMDDYY>_<X>`, e.g. UHLGA_102926_63 = "will the LGA high on
2026-10-29 EXCEED 63F". Prices are in the bought side's own terms (base.py convention); YES and NO are separate
instruments so no translation is needed on the wire, only the conid differs.

BOOK: base.Book is the YES side. Public data gives top-of-book bid/ask for YES and NO but NO SIZE (qty = nan). The two
instruments are pair-matched: NO ask = 1 - YES bid and NO bid = 1 - YES ask (verified on the public feed 2026-10-28), so a
missing YES level is filled from the NO side's complement. With a session, /iserver/marketdata/snapshot gives sizes.

GATING (nothing here can move money unless ALL hold):
  * the caller passed live=True (CLI: --live),
  * a Client Portal session is authenticated (POST /iserver/auth/status -> authenticated true),
  * IBKR_ACCOUNT_ID is set in ~/.claude/quant-kalshi/.env (or FX_ENV_FILE), and the account matches the mode:
    IBKR_MODE=paper (DEFAULT) requires a paper id (starts with "D", e.g. DU1234567); IBKR_MODE=live requires a non-paper
    id AND FX_ALLOW_LIVE_ACCOUNT=1. So the default configuration can only ever reach a paper account.
  * no KILL file (~/.claude/quant-kalshi/KILL, shared with the other venues),
  * caps: FX_MAX_ORDER_USD (5), FX_MAX_CONTRACTS (10), FX_MAX_DAILY_USD (25) from the order journal.
IBKR order "reply" prompts (precautionary warnings) are NOT auto-confirmed: the order comes back status=rejected with the
message, unless FX_AUTO_CONFIRM=1. Every order is journaled before it is sent (data/forecastex_orders.jsonl) and an order
POST is never retried (an unknown outcome is "ambiguous").
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from venues.base import Book, OrderResult, TradingDisabled, VenueError
else:
    from .base import Book, OrderResult, TradingDisabled, VenueError

DEPLOY = Path.home() / ".claude" / "quant-kalshi"
PUBLIC = "https://www.interactivebrokers.com/tws.proxy/public/forecasttrader"
GATEWAY = "https://localhost:5000/v1/api"
EXCHANGE = "FORECASTX"
FEE = 0.01  # $/contract, same as the sealed backtest (scripts/forecastex_wx.py FEE)
DEFAULT_CAPS = {"FX_MAX_ORDER_USD": 5.0, "FX_MAX_CONTRACTS": 10.0, "FX_MAX_DAILY_USD": 25.0}
# the 24 daily-HIGH stations of the sealed test (data/forecastex_wx.parquet codes); WU station = K<code>
STATIONS = ["ATL", "AUS", "BKF", "BNA", "BOS", "CLT", "DCA", "DFW", "DTW", "HOU", "JAX", "LAS", "LAX", "LGA", "MDW",
            "MIA", "MSP", "MSY", "OKC", "PHL", "PHX", "SAT", "SEA", "SFO"]
TICKER = re.compile(r"^UH([A-Z]{3})_(\d{6})_(-?\d+(?:\.\d+)?)$")
SNAP_FIELDS = "31,84,85,86,88"  # last, bid, ask size, ask, bid size


class TransportError(RuntimeError):
    pass


def _requests_send(method, url, headers, params, data, timeout, verify=True):
    import requests
    try:
        r = requests.request(method, url, headers=headers, params=params, data=data, timeout=timeout, verify=verify)
    except requests.RequestException as e:
        raise TransportError(str(e)) from e
    try:
        return r.status_code, r.json()
    except ValueError:
        return r.status_code, {"text": r.text[:500]}


def read_env(path: Path) -> dict:
    out = {}
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def parse_ticker(ticker: str) -> tuple[str, date, float]:
    m = TICKER.match(ticker)
    if not m:
        raise ValueError(f"not a ForecastEx daily-high ticker: {ticker!r}")
    return m.group(1), datetime.strptime(m.group(2), "%m%d%y").date(), float(m.group(3))


def make_ticker(code: str, day: date, strike: float) -> str:
    s = f"{strike:g}"
    return f"UH{code}_{day:%m%d%y}_{s}"


def event_day(c: dict) -> date | None:
    """Event (measured) date of a contract row: time_specifier '2026.10.28' or measured_period "Oct28'26 11:59 PM ET"."""
    ts = c.get("time_specifier")
    if ts:
        y, m, d = (int(x) for x in ts.split("."))
        return date(y, m, d)
    mp = c.get("measured_period") or ""
    m = re.match(r"([A-Z][a-z]{2})(\d{1,2})'(\d{2})", mp)
    if m:
        return datetime.strptime(f"{m.group(1)} {m.group(2)} 20{m.group(3)}", "%b %d %Y").date()
    return None


def _num(v) -> float | None:
    """Snapshot values arrive as strings ('0.91', 'C0.90' for a close, '1.2K', '1,200')."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).replace(",", "").lstrip("CH")
    mult = 1.0
    if s[-1:] in "KM":
        mult = 1e3 if s[-1] == "K" else 1e6
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


class ForecastEx:
    name = "forecastex"

    def __init__(self, env: dict | None = None, live: bool = False, transport=None, sleep=time.sleep,
                 deploy: Path = DEPLOY, journal: Path | None = None):
        self.env_path = Path(os.environ.get("FX_ENV_FILE") or (Path(deploy) / ".env"))
        self.env = read_env(self.env_path) if env is None else dict(env)
        self.live = bool(live)
        self.sleep = sleep
        self.base = self.env.get("IBKR_API_BASE", GATEWAY).rstrip("/")
        verify = self.env.get("IBKR_VERIFY_TLS", "0" if self.base.startswith("https://localhost") else "1") == "1"
        self.send = transport or (lambda m, u, h, p, d, t: _requests_send(m, u, h, p, d, t, verify=verify if u.startswith(self.base) else True))
        self.kills = [Path(deploy) / "KILL"]
        self.journal = Path(journal) if journal else Path(deploy) / "data" / "forecastex_orders.jsonl"
        self._underlying: dict[str, int] = {}
        self._list_cache: dict[str, tuple[float, dict]] = {}
        self._session_ok: bool | None = None

    # ------------------------------------------------------------------ plumbing
    def _call(self, method, path, params=None, body=None, public=False):
        url = (PUBLIC if public else self.base) + path
        h = {"User-Agent": "Mozilla/5.0 (quant-trader forecastex-client)", "Content-Type": "application/json"}
        code, out = self.send(method, url, h, params, json.dumps(body) if body is not None else None, 30)
        return code, out

    def _pub(self, path, params=None):
        code, out = self._call("GET", path, params=params, public=True)
        if code != 200:
            raise VenueError(f"public {path} -> HTTP {code}: {str(out)[:200]}")
        return out

    @property
    def mode(self) -> str:
        return (self.env.get("IBKR_MODE") or "paper").lower()

    @property
    def account(self) -> str | None:
        return self.env.get("IBKR_ACCOUNT_ID") or None

    def session_ok(self, refresh: bool = False) -> bool:
        """True iff the Client Portal gateway answers and the brokerage session is authenticated."""
        if self._session_ok is None or refresh:
            try:
                code, out = self._call("POST", "/iserver/auth/status")
                self._session_ok = bool(code == 200 and isinstance(out, dict) and out.get("authenticated"))
            except TransportError:
                self._session_ok = False
        return self._session_ok

    # ------------------------------------------------------------------ discovery (public, no login)
    def underlying_conids(self) -> dict[str, int]:
        """UH<code> symbol -> underlying conid for our 24 stations, from the public category tree."""
        if not self._underlying:
            tree = self._pub("/category/tree")["categories"]
            for cat in tree.values():
                for m in cat.get("markets", []):
                    sym = m.get("symbol", "")
                    if m.get("exchange") == EXCHANGE and sym.startswith("UH") and sym[2:] in STATIONS:
                        self._underlying[sym] = int(m["conid"])
        return self._underlying

    def strikes(self, code: str, max_age: float = 30.0) -> dict:
        """Public strike list with top-of-book: {ticker: {conid_yes, conid_no, strike, day, bid_yes, ask_yes, bid_no,
        ask_no, open_interest}} for every listed day of this station."""
        sym = f"UH{code}"
        hit = self._list_cache.get(sym)
        if hit and time.monotonic() - hit[0] < max_age:
            return hit[1]
        und = self.underlying_conids().get(sym)
        if und is None:
            raise VenueError(f"{sym} not in the public category tree")
        d = self._pub("/prediction/contract/list", {"conid": str(und), "exchange": EXCHANGE})
        out = {}
        for cc in d.get("complex_contracts", []):
            k = (cc.get("contracts") or [{}])[0]
            day = event_day(cc) or event_day(k)
            if day is None or cc.get("strike") is None:
                continue
            t = make_ticker(code, day, float(cc["strike"]))
            out[t] = {"ticker": t, "code": code, "day": day, "strike": float(cc["strike"]),
                      "conid_yes": k.get("conid_yes"), "conid_no": k.get("conid_no"),
                      "bid_yes": cc.get("bid_yes"), "ask_yes": cc.get("ask_yes"),
                      "bid_no": cc.get("bid_no"), "ask_no": cc.get("ask_no"),
                      "open_interest": cc.get("open_interest"), "restricted": cc.get("restricted"),
                      "last_trade_time": k.get("last_trade_time")}
        self._list_cache[sym] = (time.monotonic(), out)
        return out

    def contract(self, market: str) -> dict | None:
        code, day, x = parse_ticker(market)
        return self.strikes(code).get(make_ticker(code, day, x))

    def search(self, query: str) -> list[dict]:
        """`query` = a station code (LGA), a symbol (UHLGA) or a full ticker."""
        q = query.upper()
        if TICKER.match(q):
            c = self.contract(q)
            return [c] if c else []
        code = q[2:] if q.startswith("UH") else q
        return list(self.strikes(code).values()) if code in STATIONS else []

    # ------------------------------------------------------------------ quotes
    @staticmethod
    def book_from_quotes(market: str, by: float | None, ay: float | None, bn: float | None, an: float | None,
                         sizes: tuple | None = None) -> Book:
        """YES-side Book; a missing YES level is the complement of the NO side (pair-matched instruments)."""
        yb = by if by is not None else (round(1 - an, 6) if an is not None else None)
        ya = ay if ay is not None else (round(1 - bn, 6) if bn is not None else None)
        bsz, asz = sizes if sizes else (math.nan, math.nan)
        return Book(market, bids=[(float(yb), bsz)] if yb is not None else [],
                    asks=[(float(ya), asz)] if ya is not None else [], state="open")

    def snapshot(self, conids: list[int]) -> dict[int, dict]:
        """Session market data: {conid: {bid, ask, bid_size, ask_size, last}}. First call may be empty (IB warms the
        subscription), so one retry."""
        for attempt in range(2):
            code, out = self._call("GET", "/iserver/marketdata/snapshot",
                                   params={"conids": ",".join(str(c) for c in conids), "fields": SNAP_FIELDS})
            if code != 200 or not isinstance(out, list):
                raise VenueError(f"snapshot HTTP {code}: {str(out)[:200]}")
            res = {int(r["conid"]): {"bid": _num(r.get("84")), "ask": _num(r.get("86")), "bid_size": _num(r.get("88")),
                                     "ask_size": _num(r.get("85")), "last": _num(r.get("31"))} for r in out if "conid" in r}
            if any(v["bid"] is not None or v["ask"] is not None for v in res.values()) or attempt:
                return res
            self.sleep(1.0)
        return {}

    def book(self, market: str) -> Book | None:
        c = self.contract(market)
        if c is None:
            return None
        if self.session_ok():
            s = self.snapshot([c["conid_yes"], c["conid_no"]])
            y, n = s.get(int(c["conid_yes"]), {}), s.get(int(c["conid_no"]), {})
            b = self.book_from_quotes(market, y.get("bid"), y.get("ask"), n.get("bid"), n.get("ask"))
            # sizes: YES bid size = YES bid size, or (if from complement) NO ask size; same for asks
            if b.bids:
                b.bids = [(b.bids[0][0], y.get("bid_size") if y.get("bid") is not None else n.get("ask_size"))]
            if b.asks:
                b.asks = [(b.asks[0][0], y.get("ask_size") if y.get("ask") is not None else n.get("bid_size"))]
            return b
        return self.book_from_quotes(market, c["bid_yes"], c["ask_yes"], c["bid_no"], c["ask_no"])

    def fee(self, price: float, qty: float, taker: bool, market: str | None = None) -> float:
        return round(FEE * qty, 6)

    # ------------------------------------------------------------------ account (session, read-only)
    def _require_session(self):
        if not self.session_ok():
            raise TradingDisabled("no authenticated IBKR Client Portal session (start the gateway and log in)")
        if not self.account:
            raise TradingDisabled("IBKR_ACCOUNT_ID not set in .env")

    def _get(self, path, params=None):
        code, out = self._call("GET", path, params=params)
        if code != 200:
            raise VenueError(f"GET {path} -> HTTP {code}: {str(out)[:200]}")
        return out

    def balance(self) -> dict:
        self._require_session()
        return self._get(f"/portfolio/{self.account}/summary")

    def positions(self, market: str | None = None) -> list[dict]:
        self._require_session()
        rows = self._get(f"/portfolio/{self.account}/positions/0") or []
        rows = [r for r in rows if r.get("listingExchange", EXCHANGE) == EXCHANGE or "FORECASTX" in str(r)]
        if market:
            c = self.contract(market) or {}
            ids = {c.get("conid_yes"), c.get("conid_no")}
            rows = [r for r in rows if r.get("conid") in ids]
        return rows

    def open_orders(self, market: str | None = None) -> list[dict]:
        self._require_session()
        rows = (self._get("/iserver/account/orders") or {}).get("orders", [])
        if market:
            c = self.contract(market) or {}
            rows = [r for r in rows if r.get("conid") in {c.get("conid_yes"), c.get("conid_no")}]
        return rows

    def fills(self, market: str | None = None, limit: int = 100) -> list[dict]:
        self._require_session()
        rows = self._get("/iserver/account/trades") or []
        if market:
            c = self.contract(market) or {}
            rows = [r for r in rows if r.get("conid") in {c.get("conid_yes"), c.get("conid_no")}]
        return rows[:limit]

    # ------------------------------------------------------------------ gating
    def killed(self) -> Path | None:
        return next((k for k in self.kills if k.exists()), None)

    def cap(self, k: str) -> float:
        return float(self.env.get(k, DEFAULT_CAPS[k]))

    def _require_trading(self) -> None:
        if not self.live:
            raise TradingDisabled("live trading not enabled (pass --live)")
        k = self.killed()
        if k:
            raise TradingDisabled(f"KILL file present ({k})")
        self._require_session()
        acct = self.account
        if self.mode == "paper":
            if not acct.upper().startswith("D"):
                raise TradingDisabled(f"IBKR_MODE=paper but {acct} is not a paper account id (DU...)")
        elif self.mode == "live":
            if acct.upper().startswith("D"):
                raise TradingDisabled(f"IBKR_MODE=live but {acct} looks like a paper account")
            if self.env.get("FX_ALLOW_LIVE_ACCOUNT") != "1":
                raise TradingDisabled("IBKR_MODE=live needs FX_ALLOW_LIVE_ACCOUNT=1 in .env")
        else:
            raise TradingDisabled(f"unknown IBKR_MODE {self.mode!r}")

    def _journal_rows(self) -> list[dict]:
        if not self.journal.exists():
            return []
        return [json.loads(l) for l in self.journal.read_text().splitlines() if l.strip()]

    def _append(self, rec: dict) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def spent_today(self, now: datetime) -> float:
        d = now.date().isoformat()
        return sum(r.get("usd", 0.0) for r in self._journal_rows()
                   if r.get("event") == "intent" and str(r.get("ts", "")).startswith(d))

    def check_order(self, market: str, side: str, price: float, qty: float, post_only: bool, now: datetime) -> dict:
        """Validate and resolve; returns {'conid', 'book'}. Raises VenueError on anything wrong."""
        if side not in ("yes", "no"):
            raise VenueError(f"side must be yes/no, got {side!r}")
        if not (0.01 <= price <= 0.99) or round(price, 2) != round(price, 6):
            raise VenueError(f"price {price} must be a cent in [0.01, 0.99]")
        if qty < 1 or int(qty) != qty:
            raise VenueError(f"qty {qty} must be a whole number of contracts >= 1")
        c = self.contract(market)
        if c is None:
            raise VenueError(f"{market} not listed")
        usd = price * qty + self.fee(price, qty, taker=False)
        if qty > self.cap("FX_MAX_CONTRACTS"):
            raise VenueError(f"qty {qty} > FX_MAX_CONTRACTS {self.cap('FX_MAX_CONTRACTS'):g}")
        if usd > self.cap("FX_MAX_ORDER_USD"):
            raise VenueError(f"order ${usd:.2f} > FX_MAX_ORDER_USD {self.cap('FX_MAX_ORDER_USD'):g}")
        if self.spent_today(now) + usd > self.cap("FX_MAX_DAILY_USD"):
            raise VenueError(f"daily cap FX_MAX_DAILY_USD {self.cap('FX_MAX_DAILY_USD'):g} would be exceeded")
        b = self.book(market)
        if post_only and b is not None:
            _, ask = b.best(side)
            if ask is not None and price >= ask - 1e-9:
                raise VenueError(f"post-only: {side} {price} would cross the ask {ask}")
        return {"conid": int(c["conid_yes"] if side == "yes" else c["conid_no"]), "book": b, "usd": usd}

    def order_body(self, conid: int, price: float, qty: float, tif: str = "DAY", tag: str | None = None) -> dict:
        o = {"conid": conid, "orderType": "LMT", "side": "BUY", "price": round(price, 2), "quantity": int(qty),
             "tif": tif}
        if tag:
            o["cOID"] = tag
        return {"orders": [o]}

    # ------------------------------------------------------------------ trading
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
        except VenueError as e:
            return OrderResult(**base, status="rejected", detail=str(e))
        # ForecastEx contracts expire end of event day; GTC rests until then. `expire` is advisory (no GTD here).
        body = self.order_body(chk["conid"], price, qty, tif="GTC" if expire else "DAY", tag=tag)
        if dry:
            return OrderResult(**base, status="dry", detail=json.dumps(body))
        self._append({"event": "intent", "ts": now.isoformat(), "tag": tag, "market": market, "side": side,
                      "price": price, "qty": qty, "usd": chk["usd"], "mode": self.mode, "account": self.account,
                      "body": body})
        try:
            code, out = self._call("POST", f"/iserver/account/{self.account}/orders", body=body)
        except TransportError as e:
            self._append({"event": "ambiguous", "ts": now.isoformat(), "tag": tag, "detail": str(e)})
            return OrderResult(**base, status="ambiguous", detail=str(e))
        for _ in range(3):  # IB precautionary "reply" prompts
            if code == 200 and isinstance(out, list) and out and "id" in out[0] and "message" in out[0]:
                if self.env.get("FX_AUTO_CONFIRM") != "1":
                    self._append({"event": "needs_confirm", "ts": now.isoformat(), "tag": tag, "reply": out})
                    return OrderResult(**base, status="rejected", http=code,
                                       detail=f"IB asks to confirm: {out[0]['message']} (FX_AUTO_CONFIRM!=1)")
                code, out = self._call("POST", f"/iserver/reply/{out[0]['id']}", body={"confirmed": True})
                continue
            break
        self._append({"event": "response", "ts": now.isoformat(), "tag": tag, "http": code, "out": out})
        if code == 200 and isinstance(out, list) and out and out[0].get("order_id"):
            return OrderResult(**base, status="placed", order_id=str(out[0]["order_id"]), http=code,
                               detail=out[0].get("order_status"))
        return OrderResult(**base, status="rejected", http=code, detail=str(out)[:300])

    def cancel(self, order_id: str, market: str) -> bool:
        self._require_trading()
        code, out = self._call("DELETE", f"/iserver/account/{self.account}/order/{order_id}")
        self._append({"event": "cancel", "ts": datetime.now(timezone.utc).isoformat(), "order_id": order_id,
                      "http": code, "out": out})
        return code == 200 and isinstance(out, dict) and "error" not in out


# ---------------------------------------------------------------------- CLI
def main(argv: list[str]) -> int:
    c = ForecastEx(live="--live" in argv)
    args = [a for a in argv if not a.startswith("--")]
    cmd = args[0] if args else ""
    if cmd == "stations":
        u = c.underlying_conids()
        print(f"{len(u)}/{len(STATIONS)} stations: " + ", ".join(f"{k}={v}" for k, v in sorted(u.items())))
        missing = sorted(set(STATIONS) - {k[2:] for k in u})
        if missing:
            print("missing:", missing)
        return 0
    if cmd == "book" and len(args) > 1:
        b = c.book(args[1])
        print(b, "\nYES bid/ask", b.best("yes") if b else None, " NO bid/ask", b.best("no") if b else None)
        return 0
    if cmd == "preflight":
        print(f"gateway {c.base}: session {'AUTHENTICATED' if c.session_ok() else 'none'}; mode {c.mode}; "
              f"account {c.account or 'unset'}; live flag {c.live}; KILL {c.killed() or 'none'}")
        print("caps: " + ", ".join(f"{k}={c.cap(k):g}" for k in DEFAULT_CAPS))
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
