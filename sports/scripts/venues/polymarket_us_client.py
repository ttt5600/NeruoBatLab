"""Polymarket US (CFTC DCM, QCEX) client behind the venues.base interface. READ-ONLY UNLESS --live.

    python scripts/venues/polymarket_us_client.py search "highest temperature"
    python scripts/venues/polymarket_us_client.py book <slug>
    python scripts/venues/polymarket_us_client.py preflight          # env, key file + perms, signed GET balance
    python scripts/venues/polymarket_us_client.py account            # balance, positions, open orders, last fills
    python scripts/venues/polymarket_us_client.py place <slug> yes|no PRICE QTY [--expire ISO] [--live]
    python scripts/venues/polymarket_us_client.py cancel <order_id> <slug> --live
    python scripts/venues/polymarket_us_client.py smoke <slug> --live   # 1 contract, post-only, unfillable; read back; cancel

Without --live, `place` prints the exact request body it would send and sends nothing.

API (docs.polymarket.us, fetched 2026-10-27; SDK github.com/Polymarket/polymarket-us-python client.py/auth.py):
  public, no key : https://gateway.polymarket.us  /v1/markets, /v1/market/slug/{slug}, /v1/markets/{slug}/book,
                   /v1/markets/{slug}/bbo, /v1/search?query=, /v1/events/slug/{slug}
  key required   : https://api.polymarket.us      POST /v1/orders, POST /v1/order/{id}/cancel {marketSlug},
                   POST /v1/orders/open/cancel, POST /v1/order/preview {request: ...}, GET /v1/orders/open?slugs=,
                   GET /v1/portfolio/positions?market=, GET /v1/portfolio/activities?types=ACTIVITY_TYPE_TRADE,
                   GET /v1/account/balances
  auth           : Ed25519. message = ts_ms + METHOD + path (path WITH /v1, WITHOUT query string; the SDK signs the
                   path it passes and sends params separately). Secret = base64; first 32 bytes = seed (SDK does the
                   same for 64-byte secrets). Headers X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature; ts within 30 s.
  order body     : {marketSlug, type: ORDER_TYPE_LIMIT, price: {value, currency: USD}, quantity, tif,
                   intent: ORDER_INTENT_BUY_LONG (YES) | ORDER_INTENT_BUY_SHORT (NO), participateDontInitiate (post-only:
                   rejected if it would match on entry), manualOrderIndicator: MANUAL_ORDER_INDICATOR_AUTOMATIC,
                   goodTillTime for TIME_IN_FORCE_GOOD_TILL_DATE}. No client order id exists -> idempotency is ours.
  rate limits    : 25 req/s per IP shared; 429 -> wait >= 1 s, exponential backoff. "Global Rate Limit Exceeded" on an
                   order = 5-second latency stopgap reject (not a real limit; the order did not go in).
  fees           : taker 0.0695 * C * p * (1-p) (table tennis 0.10; per-market `feeCoefficient` used when present);
                   makers pay 0 and get no rebate below $10M/month maker volume (docs fees.md, eff. 2026-10-19).

PRICE CONVENTION FOR NO ORDERS -- UNVERIFIED. docs concepts/orders.md: "the price always refers to the YES side. If you
want to buy NO at $0.40, you're really selling YES at $0.60"; but the create-order schema does not say what `price`
means with intent BUY_SHORT. Default PMUS_PRICE_CONVENTION=yes (send 1 - q for a NO buy at q); `nominal` sends q.
Either mistake is harmless for a post-only order (the wrong price either rests far from the market or is rejected for
crossing), which is what `smoke` relies on: it rests 1 NO contract at a price that cannot fill, reads it back from
/v1/orders/open, and cancels. Run smoke once before any real order; set the convention from what it shows.

Gates (all enforced here): trading methods need live=True (CLI --live) AND POLYMARKET_US_KEY_ID AND a readable
POLYMARKET_US_SECRET_PATH (chmod 600) in ~/.claude/quant-kalshi/.env (or PMUS_ENV_FILE), AND no KILL file
(~/.claude/quant-kalshi/KILL, shared with kalshi_exec.py). Caps (.env): PMUS_MAX_ORDER_USD (5), PMUS_MAX_CONTRACTS (10),
PMUS_MAX_DAILY_USD (25, UTC day, from the local journal). Every order is written to the journal BEFORE the POST; a POST
that fails without an answer is marked `ambiguous` and never re-sent -- reconcile via open_orders(). Order POSTs are never
retried (same as the official SDK). The secret is read only to sign; it is never printed or logged.
"""
from __future__ import annotations

import base64
import json
import math
import os
import stat
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

try:  # works both as package module (venues.polymarket_us_client) and as a script
    from .base import Book, OrderResult, TradingDisabled, VenueError
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from base import Book, OrderResult, TradingDisabled, VenueError  # type: ignore

GATEWAY = "https://gateway.polymarket.us"
API = "https://api.polymarket.us"
DEPLOY = Path.home() / ".claude" / "quant-kalshi"
TAKER_THETA = 0.0695
RETRY_CODES = {429, 500, 502, 503, 504}
MIN_INTERVAL = 0.06          # ~16 req/s, under the 25 req/s documented limit
DEFAULT_CAPS = {"PMUS_MAX_ORDER_USD": 5.0, "PMUS_MAX_CONTRACTS": 10.0, "PMUS_MAX_DAILY_USD": 25.0}


class TransportError(RuntimeError):
    pass


def _requests_send(method, url, headers, params, data, timeout):
    try:
        r = requests.request(method, url, headers=headers, params=params, data=data, timeout=timeout)
    except requests.RequestException as e:
        raise TransportError(str(e)) from e
    try:
        body = r.json() if r.text.strip() else {}
    except ValueError:
        body = {"_text": r.text[:300]}
    return r.status_code, body


def read_env(path: Path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def taker_fee(price: float, qty: float, theta: float = TAKER_THETA) -> float:
    return theta * qty * price * (1 - price)


class PolymarketUS:
    name = "polymarket_us"

    def __init__(self, env: dict | None = None, live: bool = False, transport=None, sleep=time.sleep,
                 deploy: Path = DEPLOY, journal: Path | None = None, retries: int = 4):
        self.env_path = Path(os.environ.get("PMUS_ENV_FILE") or (Path(deploy) / ".env"))
        self.env = read_env(self.env_path) if env is None else dict(env)
        self.live = bool(live)
        self.send = transport or _requests_send
        self.sleep = sleep
        self.retries = retries
        self.kills = [Path(deploy) / "KILL"]
        self.journal = Path(journal) if journal else Path(deploy) / "data" / "pmus_orders.jsonl"
        self._last = 0.0
        self._key = None
        self._market_cache: dict[str, dict] = {}

    # ------------------------------------------------------------------ plumbing
    @property
    def key_id(self) -> str | None:
        return self.env.get("POLYMARKET_US_KEY_ID") or None

    def has_key(self) -> bool:
        p = self.env.get("POLYMARKET_US_SECRET_PATH")
        return bool(self.key_id and p and Path(os.path.expanduser(p)).is_file())

    def _signing_key(self):
        if self._key is None:
            from cryptography.hazmat.primitives.asymmetric import ed25519
            p = Path(os.path.expanduser(self.env["POLYMARKET_US_SECRET_PATH"]))
            raw = base64.b64decode(p.read_text().strip())
            self._key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
        return self._key

    def sign(self, method: str, path: str, ts_ms: int | None = None) -> dict:
        ts = str(ts_ms if ts_ms is not None else int(time.time() * 1000))
        sig = self._signing_key().sign(f"{ts}{method.upper()}{path.split('?')[0]}".encode())
        return {"X-PM-Access-Key": self.key_id, "X-PM-Timestamp": ts,
                "X-PM-Signature": base64.b64encode(sig).decode()}

    def request(self, method: str, path: str, params=None, body=None, auth: bool = False):
        """(status, json). Retries 429/5xx with backoff for GETs and cancels only; order POSTs are sent ONCE."""
        base = API if auth else GATEWAY
        retry_ok = method == "GET" or path.endswith("/cancel")
        for attempt in range(self.retries if retry_ok else 1):
            wait = self._last + MIN_INTERVAL - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self._last = time.monotonic()
            h = {"Content-Type": "application/json", "User-Agent": "quant-trader/pmus-client"}
            if auth:
                if not self.has_key():
                    raise TradingDisabled("no POLYMARKET_US_KEY_ID / POLYMARKET_US_SECRET_PATH in .env")
                h.update(self.sign(method, path))
            data = json.dumps(body) if body is not None else None
            code, out = self.send(method, base + path, h, params, data, 30)
            if code in RETRY_CODES and retry_ok and attempt < self.retries - 1:
                self.sleep(max(1.0, 2 ** attempt))   # docs: stop, wait >= 1 s, exponential backoff
                continue
            return code, out
        return code, out

    def _get(self, path, params=None, auth=False):
        code, out = self.request("GET", path, params=params, auth=auth)
        if code == 404:
            return None
        if code != 200:
            raise VenueError(f"GET {path} -> {code}: {str(out)[:200]}")
        return out

    # ------------------------------------------------------------------ public market data (no key)
    def search(self, query: str) -> list[dict]:
        out = self._get("/v1/search", {"query": query}) or {}
        evs = out.get("events") or []
        mk = [m for e in evs for m in e.get("markets", [])] + list(out.get("markets") or [])
        return mk or evs

    def markets(self, **params) -> list[dict]:
        return (self._get("/v1/markets", params) or {}).get("markets", [])

    def market(self, slug: str) -> dict | None:
        if slug not in self._market_cache:
            d = self._get(f"/v1/market/slug/{slug}")
            if d is None:
                return None
            self._market_cache[slug] = d.get("market", d)
        return self._market_cache[slug]

    def event(self, slug: str) -> dict | None:
        d = self._get(f"/v1/events/slug/{slug}")
        return d.get("event", d) if d else None

    def book(self, market: str) -> Book | None:
        d = self._get(f"/v1/markets/{market}/book")
        if not d:
            return None
        md = d.get("marketData", d)
        lv = lambda xs: [(float(x["px"]["value"]), float(x["qty"])) for x in xs or []]  # noqa: E731
        return Book(market, sorted(lv(md.get("bids")), reverse=True), sorted(lv(md.get("offers"))), md.get("state"))

    def bbo(self, market: str) -> dict | None:
        return self._get(f"/v1/markets/{market}/bbo")

    def theta(self, market: str | None) -> float:
        m = self.market(market) if market else None
        try:
            return float(m.get("feeCoefficient")) if m and m.get("feeCoefficient") not in (None, "") else TAKER_THETA
        except (TypeError, ValueError):
            return TAKER_THETA

    def fee(self, price: float, qty: float, taker: bool, market: str | None = None) -> float:
        return taker_fee(price, qty, self.theta(market)) if taker else 0.0

    # ------------------------------------------------------------------ account (key, read-only)
    def balance(self) -> dict:
        return self._get("/v1/account/balances", auth=True) or {}

    def positions(self, market: str | None = None) -> list[dict]:
        out = self._get("/v1/portfolio/positions", {"market": market} if market else None, auth=True) or {}
        pos = out.get("positions") or {}
        return [dict(v, slug=k) for k, v in pos.items()] if isinstance(pos, dict) else list(pos)

    def open_orders(self, market: str | None = None) -> list[dict]:
        out = self._get("/v1/orders/open", {"slugs": [market]} if market else None, auth=True) or {}
        return out.get("orders", [])

    def fills(self, market: str | None = None, limit: int = 100) -> list[dict]:
        p = {"types": ["ACTIVITY_TYPE_TRADE"], "limit": limit, "sortOrder": "SORT_ORDER_DESCENDING"}
        if market:
            p["marketSlug"] = market
        out = self._get("/v1/portfolio/activities", p, auth=True) or {}
        return [a.get("trade", a) for a in out.get("activities", [])]

    # ------------------------------------------------------------------ trading (key + live)
    def killed(self) -> Path | None:
        return next((k for k in self.kills if k.exists()), None)

    def _require_trading(self) -> None:
        if not self.live:
            raise TradingDisabled("live trading not enabled (pass --live)")
        if not self.has_key():
            raise TradingDisabled("POLYMARKET_US_KEY_ID + POLYMARKET_US_SECRET_PATH missing from .env")
        p = Path(os.path.expanduser(self.env["POLYMARKET_US_SECRET_PATH"]))
        if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise TradingDisabled(f"secret file is readable by group/others -> chmod 600 {p}")
        k = self.killed()
        if k:
            raise TradingDisabled(f"KILL file present ({k})")

    def cap(self, k: str) -> float:
        return float(self.env.get(k, DEFAULT_CAPS[k]))

    def _journal_rows(self) -> list[dict]:
        if not self.journal.exists():
            return []
        return [json.loads(x) for x in self.journal.read_text().splitlines() if x.strip()]

    def _append(self, rec: dict) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def spent_today(self, now: datetime) -> float:
        day = now.astimezone(timezone.utc).date().isoformat()
        sent = {r["tag"]: r for r in self._journal_rows() if r.get("event") == "intent" and r["ts"][:10] == day}
        dead = {r["tag"] for r in self._journal_rows() if r.get("event") == "result" and r.get("status") == "rejected"}
        return sum(r["cost"] for t, r in sent.items() if t not in dead)

    def wire_price(self, side: str, price: float) -> float:
        """Price as sent on the wire (see PRICE CONVENTION in the module docstring)."""
        conv = self.env.get("PMUS_PRICE_CONVENTION", "yes").lower()
        if conv not in ("yes", "nominal"):
            raise ValueError(f"PMUS_PRICE_CONVENTION must be yes or nominal, got {conv!r}")
        return round(1 - price, 6) if side == "no" and conv == "yes" else price

    def order_body(self, market: str, side: str, price: float, qty: float, post_only: bool = True,
                   expire: datetime | None = None) -> dict:
        if side not in ("yes", "no"):
            raise ValueError("side must be yes or no")
        if not 0 < price < 1:
            raise ValueError(f"price {price} outside (0, 1)")
        if qty <= 0:
            raise ValueError("qty must be > 0")
        b = {"marketSlug": market, "type": "ORDER_TYPE_LIMIT",
             "price": {"value": f"{self.wire_price(side, price):.4f}".rstrip("0").rstrip("."), "currency": "USD"},
             "quantity": qty, "intent": "ORDER_INTENT_BUY_LONG" if side == "yes" else "ORDER_INTENT_BUY_SHORT",
             "participateDontInitiate": bool(post_only),
             "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC"}
        if expire is not None:
            b["tif"] = "TIME_IN_FORCE_GOOD_TILL_DATE"
            b["goodTillTime"] = expire.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            b["tif"] = "TIME_IN_FORCE_GOOD_TILL_CANCEL" if post_only else "TIME_IN_FORCE_IMMEDIATE_OR_CANCEL"
        return b

    def check_order(self, market: str, side: str, price: float, qty: float, post_only: bool,
                    now: datetime) -> tuple[float, list[str]]:
        """(cost incl. fee, reasons to refuse). Uses the live public book + market metadata."""
        why = []
        m = self.market(market)
        if not m:
            return 0.0, [f"market {market} not found"]
        if m.get("closed") or m.get("active") is False:
            why.append("market closed / inactive")
        tick = float(m.get("orderPriceMinTickSize") or 0.01)
        if abs(round(price / tick) * tick - price) > 1e-9:
            why.append(f"price {price} is not a multiple of tick {tick}")
        minq = float(m.get("minimumTradeQty") or 1)
        if qty < minq or abs(round(qty / minq) * minq - qty) > 1e-9:
            why.append(f"qty {qty} not a multiple of minimumTradeQty {minq}")
        b = self.book(market)
        bid, ask = b.best(side) if b else (None, None)
        if post_only and ask is not None and price >= ask - 1e-9:
            why.append(f"post-only {side} @ {price} would cross the {side} ask {ask}")
        cost = price * qty + (0.0 if post_only else self.fee(price, qty, True, market))
        if cost > self.cap("PMUS_MAX_ORDER_USD") + 1e-9:
            why.append(f"cost ${cost:.2f} > PMUS_MAX_ORDER_USD {self.cap('PMUS_MAX_ORDER_USD')}")
        if qty > self.cap("PMUS_MAX_CONTRACTS"):
            why.append(f"qty {qty} > PMUS_MAX_CONTRACTS {self.cap('PMUS_MAX_CONTRACTS'):g}")
        if self.spent_today(now) + cost > self.cap("PMUS_MAX_DAILY_USD") + 1e-9:
            why.append(f"daily cap: ${self.spent_today(now):.2f} spent + ${cost:.2f} > {self.cap('PMUS_MAX_DAILY_USD')}")
        return cost, why

    def place_limit(self, market: str, side: str, price: float, qty: float, post_only: bool = True,
                    expire: datetime | None = None, dry: bool = False, tag: str | None = None,
                    now: datetime | None = None) -> OrderResult:
        now = now or datetime.now(timezone.utc)
        body = self.order_body(market, side, price, qty, post_only, expire)
        res = OrderResult(self.name, market, side, price, qty, "dry", ts=now)
        if dry or not self.live:
            res.detail = json.dumps(body)
            return res
        self._require_trading()
        cost, why = self.check_order(market, side, price, qty, post_only, now)
        if why:
            res.status, res.detail = "rejected", "; ".join(why)
            return res
        tag = tag or uuid.uuid4().hex[:12]
        if any(r.get("tag") == tag for r in self._journal_rows()):
            res.status, res.detail = "rejected", f"tag {tag} already in journal (one decision per tag)"
            return res
        self._append({"event": "intent", "ts": now.isoformat(), "tag": tag, "market": market, "side": side,
                      "price": price, "qty": qty, "cost": round(cost, 4), "body": body})
        try:
            code, out = self.request("POST", "/v1/orders", body=body, auth=True)
        except TransportError as e:
            self._append({"event": "result", "ts": now.isoformat(), "tag": tag, "status": "ambiguous", "detail": str(e)[:200]})
            res.status, res.detail = "ambiguous", f"no answer ({e}); check open_orders('{market}') before anything else"
            return res
        res.http = code
        if code == 200 and (out or {}).get("id"):
            res.status, res.order_id = "placed", out["id"]
        elif code in RETRY_CODES:
            res.status, res.detail = "ambiguous", str(out)[:300]
        else:
            res.status, res.detail = "rejected", str(out)[:300]
        self._append({"event": "result", "ts": now.isoformat(), "tag": tag, "status": res.status,
                      "order_id": res.order_id, "http": code, "detail": res.detail})
        return res

    def preview(self, market: str, side: str, price: float, qty: float, post_only: bool = True) -> dict:
        self._require_trading()
        code, out = self.request("POST", "/v1/order/preview",
                                 body={"request": self.order_body(market, side, price, qty, post_only)}, auth=True)
        if code != 200:
            raise VenueError(f"preview -> {code}: {str(out)[:200]}")
        return out

    def cancel(self, order_id: str, market: str) -> bool:
        if not self.live:
            raise TradingDisabled("live trading not enabled (pass --live)")
        if not self.has_key():
            raise TradingDisabled("no key in .env")
        code, out = self.request("POST", f"/v1/order/{order_id}/cancel", body={"marketSlug": market}, auth=True)
        self._append({"event": "cancel", "ts": datetime.now(timezone.utc).isoformat(), "order_id": order_id,
                      "market": market, "http": code})
        return code == 200

    def cancel_all(self, markets: list[str] | None = None) -> dict:
        if not self.live:
            raise TradingDisabled("live trading not enabled (pass --live)")
        code, out = self.request("POST", "/v1/orders/open/cancel",
                                 body={"slugs": markets} if markets else {}, auth=True)
        return {"http": code, **(out or {})}


# ---------------------------------------------------------------------- CLI
def _parse_expire(argv: list[str]) -> datetime | None:
    if "--expire" in argv:
        return datetime.fromisoformat(argv[argv.index("--expire") + 1].replace("Z", "+00:00"))
    return None


def cmd_preflight(c: PolymarketUS) -> int:
    print(f".env: {c.env_path} ({'present' if c.env_path.exists() else 'MISSING'})")
    print(f"POLYMARKET_US_KEY_ID: {'set' if c.key_id else 'MISSING'}")
    p = c.env.get("POLYMARKET_US_SECRET_PATH")
    if not p:
        print("POLYMARKET_US_SECRET_PATH: MISSING")
    else:
        pp = Path(os.path.expanduser(p))
        if not pp.is_file():
            print("POLYMARKET_US_SECRET_PATH: file does not exist")
        else:
            mode = stat.S_IMODE(pp.stat().st_mode)
            print(f"secret file perms {oct(mode)} {'OK' if mode & 0o077 == 0 else '-> chmod 600'}")
    print(f"convention PMUS_PRICE_CONVENTION={c.env.get('PMUS_PRICE_CONVENTION', 'yes')} (UNVERIFIED until smoke)")
    print("caps: " + ", ".join(f"{k}={c.cap(k):g}" for k in DEFAULT_CAPS))
    print(f"KILL: {c.killed() or 'none'}")
    t0 = time.time()
    ms = c.markets(limit=1, closed="false")
    print(f"public gateway: {'OK' if ms else 'FAIL'} ({time.time() - t0:.1f}s)")
    if c.has_key():
        try:
            b = c.balance()
            bl = (b.get("balances") or [{}])[0]
            print(f"signed GET /v1/account/balances: OK (buying power {bl.get('buyingPower')})")
        except Exception as e:  # noqa: BLE001
            print(f"signed GET failed: {str(e)[:200]}")
            return 1
    return 0


def cmd_smoke(c: PolymarketUS, slug: str) -> int:
    """1 NO contract at NO price = max(tick, NO bid - 10c): rests far below the market, cannot fill. Read back, cancel."""
    b = c.book(slug)
    nb, na = b.best("no") if b else (None, None)
    if nb is None or na is None:
        print("need a two-sided book")
        return 1
    m = c.market(slug)
    tick = float(m.get("orderPriceMinTickSize") or 0.01)
    q = round(max(tick, math.floor((nb - 0.10) / tick) * tick), 4)
    r = c.place_limit(slug, "no", q, float(m.get("minimumTradeQty") or 1), post_only=True, tag=f"smoke-{uuid.uuid4().hex[:6]}")
    print(f"sent NO {q} (wire price {c.wire_price('no', q)}) -> {r.status} {r.order_id or ''} {r.detail or ''}")
    if r.status != "placed":
        return 1
    time.sleep(1.0)
    mine = [o for o in c.open_orders(slug) if o.get("id") == r.order_id]
    print("read back:", json.dumps(mine)[:800])
    print("CHECK: the app/website must show this as 'Buy No' at", q, "-- if it shows", round(1 - q, 4),
          "flip PMUS_PRICE_CONVENTION")
    print("cancel:", c.cancel(r.order_id, slug))
    return 0


def main(argv: list[str]) -> int:
    live = "--live" in argv
    args = [a for a in argv if not a.startswith("--")]
    if "--expire" in argv:
        args.remove(argv[argv.index("--expire") + 1])
    c = PolymarketUS(live=live)
    cmd = args[0] if args else ""
    if cmd == "search":
        for m in c.search(" ".join(args[1:]))[:40]:
            print(m.get("slug"), "|", m.get("question") or m.get("title"))
    elif cmd == "book":
        b = c.book(args[1])
        print(b and {"state": b.state, "bids": b.bids[:5], "asks": b.asks[:5], "no(bid,ask)": b.best("no")})
    elif cmd == "preflight":
        return cmd_preflight(c)
    elif cmd == "account":
        print(json.dumps(c.balance(), indent=1)[:1500])
        print("positions:", json.dumps(c.positions())[:1500])
        print("open orders:", json.dumps(c.open_orders())[:1500])
        print("fills:", json.dumps(c.fills(limit=20))[:1500])
    elif cmd == "place":
        slug, side, price, qty = args[1], args[2], float(args[3]), float(args[4])
        r = c.place_limit(slug, side, price, qty, post_only=True, expire=_parse_expire(argv), dry=not live)
        print(r)
    elif cmd == "cancel":
        print(c.cancel(args[1], args[2]))
    elif cmd == "smoke":
        return cmd_smoke(c, args[1])
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except TradingDisabled as e:
        print("REFUSED:", e)
        raise SystemExit(2)
