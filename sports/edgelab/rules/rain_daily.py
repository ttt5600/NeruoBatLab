"""RD1/RD2: daily rain NO when NWS MOS rules measurable rain out (builder scripts/raind_daily_rain.py;
pre-registration notes/raind_daily_rain.md, written before any price / forecast / outcome pull).

Frame (loader kalshi_rain_daily_nyc): one row per settled daily-rain ticker (KXRAINNYC T0, KXRAINSEA T0, KXRAIN *-NYC).
  ticker, event_ticker, city, date, T (22:00 local D-1), event_start_ts (00:00 local D), close_ts, inputs_ts
  (18Z D-1 MOS runtime + 4 h), p06max_GFS / p06max_NAM (max P06 % over the city's five 6-h periods), yes_bid/yes_ask
  (last hourly candle ending <= T), mark_yes_bid/mark_yes_ask (last candle ending <= event_start).
attrs['tape'] = public trades (T, close] for input-true tickers.

INPUT: p06max_GFS present and every available model's p06max <= 5.
QUOTE: 0.01 <= yes_bid < yes_ask <= 1, spread <= 0.10.
taker: NO at 1 - yes_bid in [0.50, 0.99]; taker fee (evaluator).
maker: q = min(floor_cent(NO mid), NO ask - 0.01) in [0.50, 0.99]; filled iff >= 10 contracts print at no_price < q
       (strictly) in (T, close]; fill time = 10th contract. Maker fee 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

THRESH = 5.0
N_MIN = 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def select(frame: pd.DataFrame, thresh: float = THRESH) -> pd.DataFrame:
    f = frame.copy()
    g, n = f["p06max_GFS"].to_numpy(float), f["p06max_NAM"].to_numpy(float)
    ok = np.isfinite(g) & (g <= thresh) & (~np.isfinite(n) | (n <= thresh))
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok &= np.isfinite(yb) & np.isfinite(ya) & (yb >= 0.01 - 1e-9) & (yb < ya) & (ya <= 1) & (ya - yb <= 0.10 + 1e-9)
    f["no_ask"] = 1 - yb
    f["q"] = np.round(np.minimum(_cent_floor(1 - (yb + ya) / 2), (1 - yb) - 0.01), 4)
    return f[ok].reset_index(drop=True)


def _rows(s, price, source, price_time, deadline):
    return pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": s["date"].astype(str).to_numpy(),
        "event_start": _ts(s["event_start_ts"]),
        "decided_at": _ts(s["T"]),
        "inputs_as_of": _ts(s["inputs_ts"]),
        "side": "no",
        "stake": 1.0,
        "price": np.asarray(price, float),
        "price_source": source,
        "price_time": price_time,
        "fill_deadline": deadline,
        "close_price": 1 - s["mark_yes_bid"].to_numpy(float),
    })


def taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    s = select(frame, float(params.get("thresh", THRESH)))
    s = s[(s["no_ask"] >= 0.50 - 1e-9) & (s["no_ask"] <= 0.99 + 1e-9)].reset_index(drop=True)
    pt = _ts(np.minimum(s["price_ts"].to_numpy("int64"), s["T"].to_numpy("int64")))
    return _rows(s, s["no_ask"], "taker", pt, _ts(s["event_start_ts"]))


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    n_min = float(params.get("n_min", N_MIN))
    s = select(frame, float(params.get("thresh", THRESH)))
    s = s[(s["q"] >= 0.50 - 1e-9) & (s["q"] <= 0.99 + 1e-9)].reset_index(drop=True)
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    E = s["close_ts"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, qq) in enumerate(zip(s["ticker"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= E[i]) & (px < qq - 1e-9)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    b = _rows(s, s["q"], "maker_tape", _ts(np.where(fill_t >= 0, fill_t, T)), _ts(E))
    return b[fill_t >= 0].reset_index(drop=True)
