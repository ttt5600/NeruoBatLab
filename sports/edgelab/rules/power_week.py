"""PW1: Kalshi weekly ISO hub power price (KXMISOINDW, KXERCOTNORTHW) -- unreachable-rung maker, FORWARD PAPER only
(logger scripts/power_week.py; notes/phys_ladders.md lead B).

Settlement (rules_secondary, read 2026-11-09): "If the all-hours average real-time power price at <hub> is above $K/MWh
for the week", Friday 5 PM ET to Friday 5 PM ET; simple average of hourly (MISO, PJM) or 15-minute (ERCOT) REAL-TIME
prices, first published. Day-ahead prices are NOT in the settlement. The outcome is partly determined during the week
because realised real-time intervals enter the average; day-ahead prices only forecast the hours not yet realised.
PJM is excluded: its settlement feed needs an API key (401 without one).

Tick rows (loader kalshi_power_week): one per (ticker, tick T):
  ticker, event_ticker, date (Friday the week starts = vault split column), iso, K, strike_type, close_ts, week_end_ts, T
  H          intervals in the settlement week; n_done, a_sum: realised RT intervals ended <= T and their price sum
  da_mean    mean DA price over the not-yet-started hours with a published DA price (NaN if none)
  m14        mean RT price over the 14 complete days before the week started
  inputs_ts  end of the newest realised interval used
  yes_bid, yes_ask   live quote at T; mark_yes_bid, mark_yes_ask  last hourly candle ending <= close - 1 h (CLV only)
attrs['tape'] = public trades on these tickers.

RULE (frozen 2026-11-09, before the first logged tick; no settled power-market outcome read):
  r = (K * H - a_sum) / (H - n_done): average the remaining intervals need for the week to reach K.
  hi = 2.0 * max(da_mean, m14) + 20; lo = 0.25 * min(da_mean, m14) - 10 (NaN ignored in max/min).
  Requires n_done / H >= 0.20, strike_type 'greater', H > n_done.
  NO if r > hi; YES if r < lo.
  quote: 0 <= yes_bid < yes_ask <= 1, spread <= 0.10.
  First T per ticker where these hold and q = min(floor_cent(side mid), side ask - 0.01) is in [0.50, 0.99]: one post-only
  bid, 10 contracts, never cancelled. Filled iff >= 10 contracts print at side price <= q in (T, close_ts]. Unfilled ->
  no bet. Hold to settlement; maker fee 0. decided_at = T; inputs_as_of = inputs_ts; event_start = fill_deadline =
  close_ts; close_price (CLV) = side ask on the mark candle; cluster = event_ticker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_DONE = 0.20
N_MIN = 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def sides(f: pd.DataFrame) -> np.ndarray:
    K, H, nd, a = (f[x].to_numpy(float) for x in ("K", "H", "n_done", "a_sum"))
    da, m14 = f["da_mean"].to_numpy(float), f["m14"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (K * H - a) / (H - nd)
        stack = np.vstack([da, m14])
        any_ok = np.isfinite(stack).any(axis=0)
        hi = np.full(len(f), np.nan)
        lo = np.full(len(f), np.nan)
        hi[any_ok] = 2.0 * np.nanmax(stack[:, any_ok], axis=0) + 20
        lo[any_ok] = 0.25 * np.nanmin(stack[:, any_ok], axis=0) - 10
        ok = (f["strike_type"].to_numpy() == "greater") & (H > nd) & (nd / H >= MIN_DONE) & any_ok & np.isfinite(r)
        return np.where(ok & (r > hi), "no", np.where(ok & (r < lo), "yes", ""))


def select(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.sort_values(["ticker", "T"], kind="stable").reset_index(drop=True)
    side = sides(f)
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok = (side != "") & np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya) & (ya - yb <= 0.10 + 1e-9)
    mid = np.where(side == "no", 1 - (yb + ya) / 2, (yb + ya) / 2)
    ask = np.where(side == "no", 1 - yb, ya)
    q = np.round(np.minimum(_cent_floor(mid), ask - 0.01), 4)
    ok &= (q >= 0.50 - 1e-9) & (q <= 0.99 + 1e-9)
    out = f[ok].copy()
    out["side"], out["q"] = side[ok], q[ok]
    return out.groupby("ticker", sort=False).head(1).reset_index(drop=True)


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    n_min = float(params.get("n_min", N_MIN))
    s = select(frame)
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    E = s["close_ts"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, sd, qq) in enumerate(zip(s["ticker"], s["side"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price" if sd == "no" else "yes_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= E[i]) & (px <= qq + 1e-9)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    mb, ma = s["mark_yes_bid"].to_numpy(float), s["mark_yes_ask"].to_numpy(float)
    b = pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": s["event_ticker"].to_numpy(),
        "event_start": _ts(E),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(s["inputs_ts"].to_numpy("int64")),
        "side": s["side"].to_numpy(),
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(E),
        "close_price": np.where(s["side"].to_numpy() == "no", 1 - mb, ma),
    })
    return b[fill_t >= 0].reset_index(drop=True)
