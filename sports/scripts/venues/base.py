"""Minimal venue interface so the unified router (scripts/bet.py) can talk to more than Kalshi.

Conventions every venue client follows (so the router never has to know venue quirks):
  * `market` is the venue's own id (Kalshi ticker, Polymarket US slug, ...).
  * `side` is "yes" or "no"; `price` is ALWAYS in the side's own terms (buying NO at 0.96 -> price=0.96).
    The client translates to the venue's wire convention.
  * Books are returned for the YES side: bids/asks as [(price, qty), ...] best first; `best(book, side)` gives the
    (bid, ask) for either side (NO bid = 1 - YES ask, NO ask = 1 - YES bid).
  * Read-only methods must work without credentials where the venue allows it; anything that can move money must
    raise TradingDisabled unless the caller explicitly enabled live trading AND credentials are present.

A router integration is three lines:
    from venues import get_venue
    v = get_venue("polymarket_us", live=args.live)
    res = v.place_limit(slug, "no", 0.96, 3, post_only=True, expire=start_time)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable


class TradingDisabled(RuntimeError):
    """Raised when a trading method is called without --live, without keys, or with a KILL file present."""


class VenueError(RuntimeError):
    """Venue answered with an error the caller must look at (not retried)."""


@dataclass
class Book:
    market: str
    bids: list[tuple[float, float]] = field(default_factory=list)   # YES bids, best (highest) first
    asks: list[tuple[float, float]] = field(default_factory=list)   # YES asks, best (lowest) first
    state: str | None = None

    def best(self, side: str = "yes") -> tuple[float | None, float | None]:
        yb = self.bids[0][0] if self.bids else None
        ya = self.asks[0][0] if self.asks else None
        if side == "yes":
            return yb, ya
        return (None if ya is None else round(1 - ya, 6)), (None if yb is None else round(1 - yb, 6))

    def depth(self, side: str, within: float = 0.0101) -> float:
        """Contracts available to BUY `side` at the best ask..+within (YES asks for yes, YES bids for no)."""
        lv = self.asks if side == "yes" else self.bids
        if not lv:
            return 0.0
        b = lv[0][0]
        return sum(q for p, q in lv if abs(p - b) <= within)


@dataclass
class OrderResult:
    venue: str
    market: str
    side: str
    price: float
    qty: float
    status: str                     # placed | dry | rejected | ambiguous
    order_id: str | None = None
    http: int | None = None
    detail: str | None = None
    ts: datetime | None = None


@runtime_checkable
class Venue(Protocol):
    name: str

    # read-only
    def search(self, query: str) -> list[dict]: ...
    def book(self, market: str) -> Book | None: ...
    def fee(self, price: float, qty: float, taker: bool, market: str | None = None) -> float: ...

    # account (credentials, read-only)
    def balance(self) -> dict: ...
    def positions(self, market: str | None = None) -> list[dict]: ...
    def open_orders(self, market: str | None = None) -> list[dict]: ...
    def fills(self, market: str | None = None, limit: int = 100) -> list[dict]: ...

    # trading (credentials + live)
    def place_limit(self, market: str, side: str, price: float, qty: float, post_only: bool = True,
                    expire: datetime | None = None, dry: bool = False, tag: str | None = None) -> OrderResult: ...
    def cancel(self, order_id: str, market: str) -> bool: ...
