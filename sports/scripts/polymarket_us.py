"""Read-only client for Polymarket US's PUBLIC market-data gateway (no key, no account, no orders).

    python scripts/polymarket_us.py test                     # live smoke test of every fetcher
    python scripts/polymarket_us.py book <market-slug>        # best bid/ask + depth for one market
    python scripts/polymarket_us.py temps [YYYY-MM-DD]        # that day's daily-high temperature brackets, all cities

Sources (fetched 2026-10-22): docs.polymarket.us/api-reference/introduction.md (public API = https://gateway.polymarket.us,
no key; trading API = https://api.polymarket.us, key required), /api-reference/rate-limits.md (25 req/s per IP; the
price-history page says 20 req/s for that endpoint), /api-reference/price-history/get-price-history.md.

This module never authenticates and never places orders. Trading needs the user's own Key ID + Secret (Ed25519,
headers X-PM-Access-Key / X-PM-Timestamp / X-PM-Signature) created at polymarket.us/developer -- see notes/venues.md.
Only the US-regulated product (polymarket.us); the international site (polymarket.com / gamma-api) is not used here.

Price conventions (from the price-history doc): longPrice = YES display price (normally the best ask);
shortPrice = NO display price (normally 1 - best YES bid). So yes_ask = longPrice, yes_bid = 1 - shortPrice.
These are book-derived display prices, not trades.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import date, datetime, timezone

import requests

GATEWAY = "https://gateway.polymarket.us/v1"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}
_last = [0.0]
MIN_INTERVAL = 0.06  # ~16 req/s, under the documented 20-25 req/s public limit


def _get(path: str, params: dict | None = None, retries: int = 5):
    for attempt in range(retries):
        wait = _last[0] + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        r = requests.get(f"{GATEWAY}{path}", params=params, headers=H, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)  # doc: stop, wait >= 1 s, exponential backoff
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up on {path} after {retries} tries")


# ---------------------------------------------------------------- fetchers
def markets(closed: bool | None = None, limit: int = 500, max_pages: int = 400):
    """Every market (paged by offset). closed=False -> open markets only."""
    out, off = [], 0
    for _ in range(max_pages):
        p = {"limit": limit, "offset": off}
        if closed is not None:
            p["closed"] = str(closed).lower()
        ms = (_get("/markets", p) or {}).get("markets", [])
        out += ms
        if len(ms) < limit:
            break
        off += limit
    return out


def market(slug: str):
    d = _get(f"/market/slug/{slug}")  # singular 'market', per docs get-market-by-slug
    return d.get("market", d) if d else None


def event(slug: str):
    d = _get(f"/events/slug/{slug}")
    return d.get("event", d) if d else None


def book(slug: str) -> dict | None:
    """Order book. Returns {'bids': [(px, qty)...] best first, 'offers': [...], 'state': ...} for the YES side."""
    d = _get(f"/markets/{slug}/book")
    if not d:
        return None
    md = d.get("marketData", d)
    lv = lambda xs: [(float(x["px"]["value"]), float(x["qty"])) for x in xs or []]
    bids = sorted(lv(md.get("bids")), reverse=True)
    offers = sorted(lv(md.get("offers")))
    return {"slug": slug, "bids": bids, "offers": offers, "state": md.get("state"), "raw_stats": md.get("stats")}


def bbo(slug: str):
    return _get(f"/markets/{slug}/bbo")


def settlement(slug: str):
    return _get(f"/markets/{slug}/settlement")


def search(query: str):
    return _get("/search", {"query": query})


def price_history(slug: str, start_ts: int | None = None, end_ts: int | None = None,
                  interval: str = "INTERVAL_ALL") -> list[dict]:
    """Display-price history. With start/end (unix s, window <= 24 h): ~1-minute stored observations.
    Otherwise a fixed profile (INTERVAL_ALL = 3-hour or daily points). Rows: timestamp, longPrice, shortPrice."""
    if start_ts is not None:
        p = {"symbol": slug, "timestamp.startTimestamp": int(start_ts), "timestamp.endTimestamp": int(end_ts), "fidelity": 1}
    else:
        fid = {"INTERVAL_1H": 1, "INTERVAL_6H": 1, "INTERVAL_LIVE": 1, "INTERVAL_1D": 5}.get(interval, 180)
        p = {"symbol": slug, "fixedInterval": interval, "fidelity": fid}
    return (_get("/price-history", p) or {}).get("history", [])


def quote_at(slug: str, ts: int, lookback_s: int = 3600) -> dict | None:
    """Last display quote at or before ts (unix s), searching back lookback_s. yes_bid/yes_ask in dollars."""
    h = [x for x in price_history(slug, ts - lookback_s, ts) if x["timestamp"] <= ts]
    if not h:
        return None
    x = h[-1]
    return {"ts": x["timestamp"], "yes_ask": float(x["longPrice"]), "yes_bid": round(1 - float(x["shortPrice"]), 4)}


# ---------------------------------------------------------------- temperature helpers
# Cities with daily-HIGH series on Polymarket US (live gateway 2026-10-22). Station per the market text
# ("recorded at Central Park (KNYC) ... NWS Climatological Report (Daily)") and docs/faqs/weather-faqs.md.
TEMP_HIGH = {"nychigh": ("KNYC", "America/New_York"), "mdwhigh": ("KMDW", "America/Chicago"),
             "miahigh": ("KMIA", "America/New_York"), "laxhigh": ("KLAX", "America/Los_Angeles"),
             "sfohigh": ("KSFO", "America/Los_Angeles"), "phlhigh": ("KPHL", "America/New_York")}

_BR = re.compile(r"-(?:(lt)(\d+)f|(gte)(\d+)lt(\d+)f|(gte)(\d+)f)$")


def parse_bracket(slug: str) -> tuple[str, float, float] | None:
    """Market slug -> (Kalshi-style strike_type, floor, cap) so the frozen rule applies unchanged.
    Polymarket text: 'lt65f' = '<= 64F' ; 'gte65lt66f' = 'between 65F and 66F' (inclusive) ; 'gte73f' = '>= 73F'.
    Kalshi encoding: less cap=c means value < c ; between [floor, cap] inclusive ; greater floor=f means value > f.
    So lt65f -> ('less', nan, 65); gte65lt66f -> ('between', 65, 66); gte73f -> ('greater', 72, nan)."""
    m = _BR.search(slug)
    if not m:
        return None
    if m.group(1):
        return ("less", float("nan"), float(m.group(2)))
    if m.group(3):
        return ("between", float(m.group(4)), float(m.group(5)))
    return ("greater", float(m.group(7)) - 1, float("nan"))


def temp_event(city: str, day: date) -> dict | None:
    return event(f"temp-{city}-{day.isoformat()}")


def winner(mk: dict) -> str | None:
    """'yes'/'no' for a settled bracket from marketSides price ('1' on the side that paid); None if unsettled."""
    if not mk.get("closed"):
        return None
    for s in mk.get("marketSides", []):
        if s.get("long") is True and s.get("price") in ("0", "1"):
            return "yes" if s["price"] == "1" else "no"
    return None


# ---------------------------------------------------------------- CLI
def _test() -> int:
    ok = True
    t0 = time.time()
    ms = (_get("/markets", {"closed": "false", "limit": 5}) or {}).get("markets", [])
    print(f"markets(limit 5): {len(ms)} rows, e.g. {ms[0]['slug'] if ms else None}")
    ok &= bool(ms)
    today = datetime.now(timezone.utc).date()
    ev = temp_event("nychigh", today)
    mks = (ev or {}).get("markets", [])
    print(f"event temp-nychigh-{today}: {len(mks)} brackets " + str([parse_bracket(m['slug']) for m in mks]))
    ok &= bool(mks)
    if mks:
        b = book(mks[len(mks) // 2]["slug"])
        print(f"book {b['slug']}: state {b['state']} best bid {b['bids'][:1]} best ask {b['offers'][:1]} "
              f"levels {len(b['bids'])}/{len(b['offers'])}")
        print(f"bbo: {json.dumps(bbo(mks[0]['slug']))[:300]}")
        now = int(time.time())
        q = quote_at(mks[0]["slug"], now)
        print(f"quote_at(now) {mks[0]['slug']}: {q}")
        ok &= b is not None
    old = temp_event("nychigh", date(2026, 10, 20))
    if old:
        r = [(m["slug"].split("-")[-1], winner(m)) for m in old["markets"]]
        print(f"settled 2026-10-20 NYC: {r}")
        print(f"settlement endpoint: {json.dumps(settlement(old['markets'][0]['slug']))[:300]}")
        ok &= sum(w == "yes" for _, w in r) == 1
    s = search("highest temperature")
    print(f"search: keys {list((s or {}).keys())}")
    print(f"{'PASS' if ok else 'FAIL'} in {time.time() - t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "test":
        raise SystemExit(_test())
    if cmd == "book":
        print(json.dumps(book(sys.argv[2]), indent=1)[:3000])
    elif cmd == "cmp":  # live book top vs the price-history display quote, to see how empty sides are displayed
        for s in sys.argv[2:]:
            b = book(s)
            print(s, "book bid", b["bids"][:1] if b else None, "ask", b["offers"][:1] if b else None,
                  "| display", quote_at(s, int(time.time())))
    elif cmd == "temps":
        d = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else datetime.now(timezone.utc).date()
        for c in TEMP_HIGH:
            ev = temp_event(c, d)
            for m in (ev or {}).get("markets", []):
                b = book(m["slug"])
                bb = b["bids"][0] if b and b["bids"] else None
                aa = b["offers"][0] if b and b["offers"] else None
                print(f"{m['slug']:42s} {parse_bracket(m['slug'])}  bid {bb}  ask {aa}  settled {winner(m)}")
    else:
        print(__doc__)
