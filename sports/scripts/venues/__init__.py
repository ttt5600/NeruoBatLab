"""Venue clients behind one small interface (venues/base.py). Kalshi stays in kalshi_exec.py for now."""
from __future__ import annotations

from .base import Book, OrderResult, TradingDisabled, Venue, VenueError  # noqa: F401


def get_venue(name: str, **kw):
    if name in ("polymarket_us", "pmus", "polymarket-us"):
        from .polymarket_us_client import PolymarketUS
        return PolymarketUS(**kw)
    raise KeyError(f"unknown venue {name!r} (known: polymarket_us; Kalshi lives in kalshi_exec.py)")
