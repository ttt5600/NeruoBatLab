"""P1m: L39 earlier-in-month climatology NO, maker (builder scripts/l39_p1m.py; pre-registration
notes/l39_expansion.md, written before any early-window quote / climatology / outcome pull).

Frame (loader kalshi_wx_monthly_early): one row per (settled monthly rain/snow `greater` ticker, decision day d) with
7 <= days_left <= 20. Columns: ticker, event_ticker, kind, d, T (22:00 station-local d), close_time, X, obs_d (ACIS
days 1..d-1), acis_missing, q995 (99.5th pct of climatological totals over the same remaining-window length, ACIS
1991..year-1, +-10 day start shift), yes_bid / yes_ask / price_ts (last hourly candle ending <= T).
attrs['tape'] = public trades (ts, ticker, no_price, count) after T for input-true tickers.

INPUT: acis_missing == 0, X - obs_d > q995. QUOTE: yes_bid > 0, yes_ask < 1, spread <= 0.10, yes_ask > 0.01,
market open at T. First qualifying d per ticker.
maker: q = min(floor_cent(NO mid), NO ask - 0.01); filled iff >= 1 contract prints at no_price <= q in (T, close]
(the W1m / E4 M3 rule); fill time = first such print. Maker fee 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _close(s):
    return ((pd.to_datetime(s, format="ISO8601", utc=True) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy("int64")


def select(frame: pd.DataFrame, qcol: str = "q995") -> pd.DataFrame:
    f = frame.copy()
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok = (f["acis_missing"].to_numpy() == 0) & np.isfinite(f[qcol].to_numpy(float))
    ok &= np.isfinite(yb) & np.isfinite(ya) & (yb > 0) & (ya < 1) & (ya - yb <= 0.10 + 1e-9) & (ya > 0.01)
    ok &= f["T"].to_numpy("int64") < _close(f["close_time"])
    ok &= (f["X"].to_numpy(float) - f["obs_d"].to_numpy(float)) > f[qcol].to_numpy(float)
    s = f[ok].sort_values("d").drop_duplicates("ticker", keep="first").reset_index(drop=True)
    yb, ya = s["yes_bid"].to_numpy(float), s["yes_ask"].to_numpy(float)
    s["q"] = np.round(np.minimum(np.floor(np.round((1 - (yb + ya) / 2) * 100, 6)) / 100, (1 - yb) - 0.01), 4)
    return s


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    s = select(frame, params.get("qcol", "q995"))
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    E = _close(s["close_time"])
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, qq) in enumerate(zip(s["ticker"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts, px = d["ts"].to_numpy("int64"), d["no_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= E[i]) & (px <= qq + 1e-9) & (d["count"].to_numpy(float) > 0)
        if m.any():
            fill_t[i] = ts[m][0]
    b = pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": s["event_ticker"].to_numpy(),
        "event_start": _ts(E),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(T),
        "side": "no",
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(E),
    })
    return b[fill_t >= 0].reset_index(drop=True)
