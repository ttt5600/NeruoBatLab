"""E4 at other decision times (agent e4_timing 2026-11-16, prereg sports/notes/e4_timing.md).

Selection is done by the builder (scripts/e4_timing.py dev -> results/e4_timing/edgelab_T10.parquet); the frame's T,
yes_bid, yes_ask are at the decision time. taker = kalshi_wx.taker. maker_tt = M3 price (mid floored to the cent, capped
at NO ask - 1c), filled iff ANY print at no_price < q in (T, close] (trade-through: the level cleared, so queue position
does not matter). Fill time = first such print.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalshi_wx import _base, _close_no, _cent_floor, _ts, taker  # noqa: F401  (taker re-exported for the spec)


def maker_tt(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
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
        ts, px = d["ts"].to_numpy(), d["no_price"].to_numpy()
        m = (ts > t0) & (ts <= t1) & (px < q[i] - 1e-9)
        if m.any():
            fill_t[i] = ts[m].min()
    filled = fill_t >= 0
    b["price"] = q
    b["price_source"] = "maker_tape"
    b["price_time"] = _ts(np.where(filled, fill_t, T))
    b["fill_deadline"] = _ts(close)
    b["close_price"] = _close_no(frame, T + 10 * 3600)
    return b[filled].reset_index(drop=True)
