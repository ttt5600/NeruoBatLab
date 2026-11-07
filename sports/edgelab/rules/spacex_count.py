"""SX1: Kalshi monthly SpaceX launch count (KXSPACEXCOUNT-<YY><MON>) -- unreachable-rung NO maker
(builder scripts/phys_ladders.py spacex; notes/phys_ladders.md).

Contract: "If SpaceX has more than K launches in <month>, YES" (strike_type greater, floor_strike K). Markets open on
day 2 of the month 00:00Z and close day 1 of the next month 14:00Z. A launch is a physical event with a public
timestamp, and the pace is capped mechanically (pads, turnaround), so a high rung can become unreachable mid-month.

Frame (loader kalshi_spacex_count): one row per (ticker, daily decision time T = 16:00Z on each day of the month while
the market is open and T < month end). Inputs only:
  ticker, event_ticker, date (1st of the event month = vault split column), K, strike_type, close_ts, month_end_ts
  (first instant of next month, America/New_York, as unix), T (unix)
  c      SpaceX launches (Launch Library 2, lsp SpaceX, any outcome) with liftoff in [month start ET, T)
  L      ceil((month_end_ts - T) / 86400): whole days of the month still to come
  cap    max SpaceX launches in any L-day window (rolling, start at each launch) among liftoffs in [T - 365 d, T)
  last_launch_ts   liftoff time of the newest launch used (<= T) -> inputs_as_of
  yes_bid, yes_ask             last hourly candle ending <= T
  mark_yes_bid, mark_yes_ask   last hourly candle ending <= month_end_ts - 1 h (CLV close mark only)
attrs['tape'] = public trades (ticker, ts, yes_price, no_price, count) after the first T at which the input-only
condition below holds, through month_end_ts, for every ticker where it holds at some T.

RULE (frozen 2026-11-06, before any launch history, quote, tape or outcome was pulled):
  input condition: strike_type 'greater' and need = K + 1 - c >= cap + 2   (needs >= 2 more launches than the
  fastest L-day burst of the past year).
  quote filter: 0 <= yes_bid < yes_ask <= 1, yes_ask - yes_bid <= 0.10.
  For each ticker take the FIRST T (chronological) where the input condition AND the quote filter hold, and the maker
  price q = min(floor_cent(NO mid), NO ask - 0.01) is in [0.50, 0.99] (NO mid = 1 - (yes_bid + yes_ask)/2,
  NO ask = 1 - yes_bid). One post-only NO bid of 10 contracts at q, never cancelled, never re-priced.
  Filled iff >= 10 contracts print at no_price <= q in (T, month_end_ts]; fill time = when the 10th prints.
  Unfilled -> no bet. Hold to settlement. Maker fee 0 (fee_type quadratic).
  decided_at = T; inputs_as_of = last_launch_ts; event_start = month_end_ts (the count is final from then);
  fill_deadline = month_end_ts. close_price (CLV) = NO ask on the mark candle. cluster = event_ticker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN = 2
N_MIN = 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def select(frame: pd.DataFrame, margin: int = MARGIN) -> pd.DataFrame:
    f = frame.sort_values(["ticker", "T"], kind="stable")
    K, c, cap = (f[x].to_numpy(float) for x in ("K", "c", "cap"))
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok = (f["strike_type"].to_numpy() == "greater") & np.isfinite(cap) & (K + 1 - c >= cap + margin)
    ok &= np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya) & (ya - yb <= 0.10 + 1e-9)
    q = np.round(np.minimum(_cent_floor(1 - (yb + ya) / 2), (1 - yb) - 0.01), 4)
    ok &= (q >= 0.50 - 1e-9) & (q <= 0.99 + 1e-9)
    out = f[ok].copy()
    out["q"] = q[ok]
    return out.groupby("ticker", sort=False).head(1).reset_index(drop=True)


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    n_min = float(params.get("n_min", N_MIN))
    s = select(frame, int(params.get("margin", MARGIN)))
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    E = s["month_end_ts"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, qq) in enumerate(zip(s["ticker"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= E[i]) & (px <= qq + 1e-9)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    b = pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": s["event_ticker"].to_numpy(),
        "event_start": _ts(E),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(s["last_launch_ts"].to_numpy("int64")),
        "side": "no",
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(E),
        "close_price": 1 - s["mark_yes_bid"].to_numpy(float),
    })
    return b[fill_t >= 0].reset_index(drop=True)
