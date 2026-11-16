"""E4 + calibrated probabilistic selector (agent e4_probabilistic 2026-11-16, prereg sports/notes/e4_probabilistic.md).

The frame (built by scripts/e4_probabilistic.py dev -> results/e4_probabilistic/edgelab_new.parquet) holds every
TEST-period E1 bracket that the frozen 5-model E4 rule does NOT pick, has a tradeable 22:00 quote (E4 filter) and a
walk-forward P(hit) (`phit`, Student-t EMOS on NBM NBS 19Z txn/xnd + fcsafe models). The rules below select from it:
  c  : phit <= 0.02
  pa : phit <= 0.5 * (1 - cost); cost = NO ask + 0.07 p(1-p) (taker) or q (maker)
taker = NO at 1 - yes_bid at T. maker_tt = q = min(floor_cent(NO mid), ask - 1c), filled iff ANY print at no_price < q
in (T, close] (trade-through), fill time = first such print. Self-contained copy of kalshi_wx_timing's maker_tt so the
registered hash covers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalshi_wx import _base, _close_no, _cent_floor, _ts

PSTAR = 0.02


def _q(frame):
    ask = 1 - frame["yes_bid"].to_numpy(float)
    return np.round(np.minimum(_cent_floor(1 - (frame["yes_bid"].to_numpy(float) + frame["yes_ask"].to_numpy(float)) / 2),
                               ask - 0.01), 4)


def _taker_cost(frame):
    a = 1 - frame["yes_bid"].to_numpy(float)
    return a + 0.07 * a * (1 - a)


def _taker(frame):
    b, T = _base(frame)
    b["price"] = np.round(1 - frame["yes_bid"].to_numpy(float), 4)
    b["price_source"] = "taker"
    b["price_time"] = _ts(T)
    b["close_price"] = _close_no(frame, T + 10 * 3600)
    return b


def _maker(frame):
    b, T = _base(frame)
    q = _q(frame)
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


def _sub(frame, m):
    s = frame[m].reset_index(drop=True)
    s.attrs = frame.attrs
    return s


def taker_c(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(_sub(frame, frame["phit"].to_numpy(float) <= PSTAR))


def maker_c(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _maker(_sub(frame, frame["phit"].to_numpy(float) <= PSTAR))


def taker_pa(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(_sub(frame, frame["phit"].to_numpy(float) <= 0.5 * (1 - _taker_cost(frame))))


def maker_pa(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _maker(_sub(frame, frame["phit"].to_numpy(float) <= 0.5 * (1 - _q(frame))))
