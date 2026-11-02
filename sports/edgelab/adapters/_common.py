"""Shared helpers for forward adapters (hashed together with every adapter module; see vault.forward_adapter).

Read-only by construction: files are opened for reading only; the only network calls are PUBLIC Kalshi market-data
GETs (trade tape, market status/result) -- no keys, no orders, no portfolio endpoints.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA, QS_DATA, SPORTS

QK_DATA = Path.home() / ".claude" / "quant-kalshi" / "data"
API = "https://api.elections.kalshi.com/trade-api/v2"
BET_COLS = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "close_price", "fill_deadline"]


class AdapterError(RuntimeError):
    pass


def imp_scripts() -> None:
    for p in (str(SPORTS), str(SPORTS / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r") as f:                       # read-only
        return [json.loads(l) for l in f if l.strip()]


def first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    return paths[0]


def qs_or_repo(rel: str) -> Path:
    """The launchd copy (~/.claude/quant-sports/data, the source of truth for game-day logs) else the repo copy."""
    return first_existing(QS_DATA / rel, DATA / rel)


def empty_bets() -> pd.DataFrame:
    return pd.DataFrame(columns=BET_COLS)


def with_report(b: pd.DataFrame, report: dict) -> pd.DataFrame:
    b.attrs["adapter_report"] = report
    return b


def date_range(s: pd.Series) -> list:
    s = pd.to_datetime(s).dropna()
    return [str(s.min())[:10], str(s.max())[:10]] if len(s) else [None, None]


# ------------------------------------------------------------------ public Kalshi market data
def kalshi_get(path: str, params: dict | None = None, cache: bool = False) -> dict:
    imp_scripts()
    from sports.http import fetch                   # repo HTTP helper (throttle, retries, on-disk cache)
    return fetch(f"{API}/{path}", params, cache=cache)


def tape_last_no(ticker: str, t0: int, t1: int) -> float:
    """Last print with taker_side == 'no' in (t0, t1] (unix s) -> its no_price; NaN if none. The window must be in the
    past (immutable), so the response is cached. This is edgelab's E4 close for CLV (rules/kalshi_wx.py _close_no)."""
    if t1 > time.time():
        return np.nan
    best_ts, best_px = -1, np.nan
    for path in ("markets/trades", "historical/trades"):
        cursor, got = None, False
        while True:
            p = {"ticker": ticker, "min_ts": int(t0), "max_ts": int(t1), "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = kalshi_get(path, p, cache=True)
            for r in d.get("trades", []):
                got = True
                ts = int(pd.Timestamp(r["created_time"]).timestamp())
                if r.get("taker_side") != "no" or not (t0 < ts <= t1):
                    continue
                npx = r.get("no_price_dollars")
                npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
                if ts > best_ts:
                    best_ts, best_px = ts, npx
            cursor = d.get("cursor")
            if not cursor or not d.get("trades"):
                break
        if got:
            break
    return best_px


def market(ticker: str) -> dict:
    """Public market record (status, result, close_time). Not cached: a market can still change until finalized."""
    return kalshi_get(f"markets/{ticker}", None, cache=False).get("market", {})
