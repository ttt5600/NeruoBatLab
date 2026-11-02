"""Cheater rules for the self-test. Each one tries to use information it could not have had; edgelab must refuse.

peek_tape: E1 picks, but keeps only brackets whose NO price ROSE on the tape after the decision (a look at the
  future), and honestly stamps inputs_as_of = the time of the tape print it used -> PIT 'inputs_after_decision'.
late_price: E1 taker, but takes the best NO print in the next 10 h as its price, stamped at that print's time
  -> PIT 'quote_after_decision'.
read_outcome: tries to read the hidden outcome column -> KeyError (the vault stripped it).
set_outcome: returns its own `outcome` column -> refused by the vault.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import kalshi_wx


def peek_tape(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    b = kalshi_wx.taker(frame)
    tr = frame.attrs["tape"]
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}
    keep, seen = [], []
    for tk, t0, px in zip(frame["ticker"], frame["T"], b["price"]):
        d = g.get(tk)
        later = d[(d.ts > t0)] if d is not None else None
        if later is not None and len(later) and later.no_price.iloc[-1] > px:
            keep.append(True)
            seen.append(int(later.ts.iloc[-1]))
        else:
            keep.append(False)
            seen.append(int(t0))
    b["inputs_as_of"] = pd.to_datetime(np.array(seen), unit="s", utc=True)
    return b[np.array(keep)].reset_index(drop=True)


def late_price(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    b = kalshi_wx.taker(frame)
    tr = frame.attrs["tape"]
    g = {k: v for k, v in tr.groupby("ticker")}
    pt, px = [], []
    for tk, t0, p0 in zip(frame["ticker"], frame["T"], b["price"]):
        d = g.get(tk)
        w = d[(d.ts > t0) & (d.ts <= t0 + 36000)] if d is not None else None
        if w is not None and len(w):
            j = w.no_price.idxmin()
            pt.append(int(w.loc[j, "ts"]))
            px.append(float(w.loc[j, "no_price"]))
        else:
            pt.append(int(t0))
            px.append(float(p0))
    b["price"] = px
    b["price_time"] = pd.to_datetime(np.array(pt), unit="s", utc=True)
    return b


def read_outcome(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    b = kalshi_wx.taker(frame)
    return b[frame["won"].to_numpy() == 1]


def set_outcome(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    b = kalshi_wx.taker(frame)
    b["outcome"] = 1.0
    return b
