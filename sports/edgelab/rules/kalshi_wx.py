"""E1 / E2 (taker) and E4 (maker) on the frozen consensus-reject picks.

Selection is NOT re-done here: the frame is the frozen rule's output (lab/strategies/kalshi_reject_safe_check.py for
E1, scripts/wx_expand.py picks() for X1/E2), built by scripts/kalshi_temp_maker.py. These functions only turn picks into
priced, time-stamped bet rows.

Timestamps (UTC): T = decision candle end (~22:00 station-local the night before the observation day).
  decided_at = T; inputs_as_of = T - 4 h (the leak-free 'fcsafe' members use model hours <= 18:00 local);
  price_time = T (the candle-end quote); event_start = T + 2 h (local midnight = first observation of the day).
  maker fill time = time the cumulative printed size at no_price <= q reaches N (tape, (T, close]).
  close for CLV = last print with taker_side == 'no' (= an executed NO ASK, the same side and the same kind of price
  a taker pays) in (T, T + 10 h] (~08:00 local on the event day, before the daily max forms); NaN when none printed
  (coverage reported; < 50% coverage -> CLV null -> rule 5e fails closed). Mixing bid-side prints into the close
  would make every taker's CLV negative by half the spread, which is a measurement artefact, not information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _close_no(frame: pd.DataFrame, t_end: np.ndarray) -> np.ndarray:
    tr = frame.attrs.get("tape")
    out = np.full(len(frame), np.nan)
    if tr is None or tr.empty:
        return out
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}
    for i, (tk, t0, t1) in enumerate(zip(frame["ticker"], frame["T"], t_end)):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy()
        m = (ts > t0) & (ts <= t1) & (d["taker_side"].to_numpy() == "no")
        if m.any():
            out[i] = float(d["no_price"].to_numpy()[m][-1])
    return out


def _base(frame: pd.DataFrame) -> pd.DataFrame:
    T = frame["T"].to_numpy("int64")
    return pd.DataFrame({
        "bet_id": frame["ticker"].to_numpy(),
        "cluster": pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(T + 2 * 3600),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(T - 4 * 3600),
        "side": "no",
        "stake": 1.0,
    }), T


def taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Buy NO at the NO ask = 1 - yes_bid at T (what the live taker pays), fee per the spec's schedule."""
    b, T = _base(frame)
    b["price"] = np.round(1 - frame["yes_bid"].to_numpy(float), 4)
    b["price_source"] = "taker"
    b["price_time"] = _ts(T)
    b["close_price"] = _close_no(frame, T + 10 * 3600)
    return b


def _cent_floor(x):
    return np.floor(np.round(x * 100, 6)) / 100


def maker_m3(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """E4 = kalshi_temp_maker M3_N{n}: post-only NO bid at the mid floored to the cent, capped at ask - 1c; filled iff
    >= n contracts print at no_price <= q in (T, close]; unfilled -> no bet. Fill time = when the n-th contract prints."""
    n_min = float((params or {}).get("n_min", 10))
    b, T = _base(frame)
    ask = 1 - frame["yes_bid"].to_numpy(float)
    q = np.round(np.minimum(_cent_floor(1 - (frame["yes_bid"].to_numpy(float) + frame["yes_ask"].to_numpy(float)) / 2),
                            ask - 0.01), 4)
    close = frame["close_ts"].to_numpy("int64")
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None else {}
    fill_t = np.full(len(frame), -1, "int64")
    for i, (tk, t0, t1) in enumerate(zip(frame["ticker"], T, close)):
        d = g.get(tk)
        if d is None:
            continue
        ts, px, ct = d["ts"].to_numpy(), d["no_price"].to_numpy(), d["count"].to_numpy()
        m = (ts > t0) & (ts <= t1) & (px <= q[i] + 1e-9)
        if not m.any():
            continue
        cum = np.cumsum(ct[m])
        hit = np.nonzero(cum >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    filled = fill_t >= 0
    b["price"] = q
    b["price_source"] = "maker_tape"
    b["price_time"] = _ts(np.where(filled, fill_t, T))
    b["fill_deadline"] = _ts(close)
    b["close_price"] = _close_no(frame, T + 10 * 3600)
    return b[filled].reset_index(drop=True)
