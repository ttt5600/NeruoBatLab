"""TC1: Kalshi monthly preliminary tornado count (KXTORNADO-<YY><MON>) -- unreachable-rung NO maker
(builder scripts/tornado_ladder.py; notes/phys_ladders.md).

Contract: "If the preliminary number of tornadoes in <month> is above K, YES" (strike_type greater). Expiration value =
SPC preliminary count as documented at the first 10:00 ET after month end (later revisions ignored). Every listed rung
is 'greater', so a rung the count has already passed is a sure YES -- there is no NO position on a passed rung and this
trial takes none. Only the unreachable leg exists.

Inputs (SPC, public): summary/<year>/ruf/NAT/NAT_<MON>.json -> daily[MMDD]['torn'], where day D = the SPC report day
12Z D -> 12Z D+1 (checked 2011-04-26/27/28 and 2023-04-04/05 against the daily CSVs: pre-filter years = raw file,
2023 = *_filtered_torn file).

Frame (loader kalshi_tornado_count): one row per (ticker, T = 16:00Z on each day D of the event month while open).
  ticker, event_ticker, date (1st of event month = vault split column), K, strike_type, close_ts, month_end_ts, T
  c    tornado reports in report days <= D-1 of the month (day D-1 closed at 12Z D, 4 h before T)
  L    report days still to come = last_day_of_month - D + 1
  cap  max sum of `torn` over any L consecutive report days inside this calendar month, over 2005..2022 (all
       pre-Kalshi; KXTORNADO listed 2023-07). Fixed table, no in-sample years.
  inputs_ts    12:00Z on day D (input as-of)
  yes_bid, yes_ask             last hourly candle ending <= T
  mark_yes_bid, mark_yes_ask   last hourly candle ending <= month_end_ts - 1 h (CLV mark only)
attrs['tape'] = public trades from the first input-true T through close, for every ticker where the input holds.

GATE (pre-registered, run BEFORE any scoring, scripts/tornado_ladder.py gate, DEV months only): SPC JSON month total as
it stands today vs Kalshi expiration_value. Match = |diff| <= max(3, 3% of expiration_value). Require match on >= 90% of
dev months. Below 90% -> do not evaluate; the trial is retired as gate fail. This is the first-print vs revised check.

RULE (frozen 2026-11-09, before any Kalshi expiration value, candle or tape for a settled month was pulled):
  input: strike_type 'greater' and need = K + 1 - c >= cap + MARGIN (MARGIN = 5, allowance for first-print differences).
  quote: 0 <= yes_bid < yes_ask <= 1, spread <= 0.10.
  First T (chronological) per ticker where input + quote hold and q = min(floor_cent(NO mid), NO ask - 0.01) is in
  [0.50, 0.99]: one post-only NO bid, 10 contracts, never cancelled or re-priced. Filled iff >= 10 contracts print at
  no_price <= q in (T, min(close_ts, month_end_ts)]; fill time = when the 10th prints. Unfilled -> no bet. Hold to
  settlement; maker fee 0. decided_at = T; inputs_as_of = inputs_ts; event_start = fill_deadline = min(close, month_end).
  close_price (CLV) = NO ask on the mark candle. cluster = event_ticker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN = 5
N_MIN = 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def input_true(K, c, cap, strike_type, margin: int = MARGIN):
    return (np.asarray(strike_type) == "greater") & np.isfinite(cap) & (K + 1 - c >= cap + margin)


def select(frame: pd.DataFrame, margin: int = MARGIN) -> pd.DataFrame:
    f = frame.sort_values(["ticker", "T"], kind="stable")
    K, c, cap = (f[x].to_numpy(float) for x in ("K", "c", "cap"))
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok = input_true(K, c, cap, f["strike_type"].to_numpy(), margin)
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
    E = np.minimum(s["close_ts"].to_numpy("int64"), s["month_end_ts"].to_numpy("int64"))
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
        "inputs_as_of": _ts(s["inputs_ts"].to_numpy("int64")),
        "side": "no",
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(E),
        "close_price": 1 - s["mark_yes_bid"].to_numpy(float),
    })
    return b[fill_t >= 0].reset_index(drop=True)
