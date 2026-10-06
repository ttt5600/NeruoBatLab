"""Market data: fetch real daily bars from Yahoo's public chart endpoint, cache to disk.

No API key. One file per symbol in ``data_cache/``. Adjusted close only --
splits and dividends are already folded in, which is what a backtest must use.
Using raw close instead is the single most common way to invent returns that
never existed (a 2:1 split reads as a -50% day).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[1] / "data_cache"
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
_NY = "America/New_York"
_SETTLED = pd.Timedelta(hours=17)
"""A session's bar is treated as final from 17:00 New York time. Before that,
Yahoo serves today's row with the *current* price, and caching it would store
an intraday quote as if it were a close."""


def last_completed_session(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The most recent weekday whose bar is settled as of ``now``.

    Exchange holidays are not modelled, so on a holiday this names a session
    that has no bar. Callers treat it as an upper bound, never as a promise.
    """
    now = _in_ny(now)
    day = now.normalize()
    if now - day < _SETTLED:
        day -= pd.Timedelta(days=1)
    while day.weekday() >= 5:
        day -= pd.Timedelta(days=1)
    return day.tz_localize(None)


def _in_ny(now: pd.Timestamp | None) -> pd.Timestamp:
    """``now`` as an aware New York timestamp; naive input is read as New York."""
    now = pd.Timestamp.now(tz=_NY) if now is None else pd.Timestamp(now)
    return now.tz_localize(_NY) if now.tzinfo is None else now.tz_convert(_NY)


def _settled_at(session: pd.Timestamp) -> pd.Timestamp:
    return (pd.Timestamp(session).tz_localize(_NY) + _SETTLED).tz_convert("UTC")


def _fetch_one(symbol: str, start: str, end: str, retries: int = 3) -> pd.Series:
    p1 = int(pd.Timestamp(start).timestamp())
    p2 = int(pd.Timestamp(end).timestamp())
    url = (
        _CHART.format(sym=symbol)
        + f"?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"
    )
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                payload = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    else:
        raise RuntimeError(f"{symbol}: download failed after {retries} tries: {last_err}")

    res = payload["chart"]["result"]
    if not res:
        raise RuntimeError(f"{symbol}: empty result ({payload['chart'].get('error')})")
    res = res[0]
    ts = res["timestamp"]
    ind = res["indicators"]
    if "adjclose" in ind and ind["adjclose"][0].get("adjclose"):
        px = ind["adjclose"][0]["adjclose"]
    else:  # some symbols only return the quote block
        px = ind["quote"][0]["close"]

    idx = (
        pd.to_datetime(ts, unit="s", utc=True)
        .tz_convert("America/New_York")
        .normalize()
        .tz_localize(None)
    )
    s = pd.Series(px, index=idx, name=symbol, dtype="float64")
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.dropna()


def load_prices(
    symbols: list[str],
    start: str = "1990-01-01",
    end: str = "2100-01-01",
    refresh: bool = False,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Adjusted-close panel, one column per symbol.

    Cached per symbol, and refetched when the cache is behind the last settled
    session -- a cache that only ever grows backwards serves the same final bar
    forever, and a daily job reading it does nothing without raising. Bars from
    an unsettled session are dropped before caching. Interior gaps are forward-filled, but never before a
    symbol's first real print -- that would fabricate pre-IPO prices and hand
    a backtest free history it could not have traded.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    settled = last_completed_session(now)
    wanted = min(pd.Timestamp(end), settled)
    now_utc = _in_ny(now).tz_convert("UTC")
    cols = {}
    for sym in symbols:
        f = CACHE / f"{sym}.csv"
        meta_f = CACHE / f"{sym}.meta.json"

        # The cache is keyed by SYMBOL, so it must also record the window it was
        # fetched over. Without this, a cheap early call like
        # load_prices(["SPY"], start="2015-01-01") poisons every later call that
        # asks for more history: the short file is reused, dropna() truncates the
        # whole panel, and nothing raises. Always fetch maximal history and slice
        # on read, so a cached file is guaranteed to be a superset.
        stale = True
        if f.exists() and not refresh:
            try:
                meta = json.loads(meta_f.read_text())
                stale = pd.Timestamp(meta["fetched_from"]) > pd.Timestamp(start)
                # Forward staleness. A cache missing ``wanted`` is refetched
                # unless it was already fetched after that session settled, in
                # which case the bar does not exist (a holiday) and refetching
                # on every call would not produce it.
                behind = pd.Timestamp(meta.get("last_print", "1900-01-01")) < wanted
                fetched = meta.get("fetched_at")
                checked = fetched is not None and pd.Timestamp(fetched) >= _settled_at(wanted)
                stale = stale or (behind and not checked)
            except (OSError, ValueError, KeyError):
                stale = True

        if stale:
            fetch_from = min(pd.Timestamp(start), pd.Timestamp("1990-01-01"))
            s = _fetch_one(sym, fetch_from.strftime("%Y-%m-%d"), end)
            s = s[s.index <= settled]
            s.to_frame().to_csv(f)
            meta_f.write_text(json.dumps({
                "fetched_from": fetch_from.strftime("%Y-%m-%d"),
                "first_print": s.index[0].strftime("%Y-%m-%d"),
                "last_print": s.index[-1].strftime("%Y-%m-%d"),
                "fetched_at": now_utc.isoformat(timespec="seconds"),
            }))
        else:
            s = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            s.name = sym
        cols[sym] = s

    df = pd.DataFrame(cols).sort_index()
    df = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    # interior gaps only: mask ffill to each column's own live span
    notna = df.notna()
    live = notna.cummax() & notna[::-1].cummax()[::-1]
    return df.ffill().where(live)


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns. First row drops: no prior close to diff against."""
    return prices.pct_change().iloc[1:]


def synthetic_prices(
    n_days: int = 5000,
    n_assets: int = 5,
    mu: float = 0.07,
    sigma: float = 0.16,
    seed: int = 0,
) -> pd.DataFrame:
    """GBM panel: the null world, where no timing signal exists by construction.

    Any timing strategy that "works" here is fitting noise. Used as the control
    in scripts/run_demo.py -- if the harness reports an edge on this, the
    harness is broken, not the strategy.
    """
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    shocks = rng.normal(
        (mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), (n_days, n_assets)
    )
    px = 100 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.bdate_range("2000-01-03", periods=n_days)
    return pd.DataFrame(px, index=idx, columns=[f"SYN{i}" for i in range(n_assets)])
