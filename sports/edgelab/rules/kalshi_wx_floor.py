"""E4 variant (e4_live_audit 2026-11-09, post hoc from the live fill-price slice): M3 maker restricted to picks whose NO
ask at T is >= min_ask. Selection otherwise identical to kalshi_wx.maker_m3."""
from __future__ import annotations

import pandas as pd

from .kalshi_wx import maker_m3


def maker_m3_floor(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    min_ask = float((params or {}).get("min_ask", 0.90))
    keep = (1 - frame["yes_bid"].astype(float)) >= min_ask - 1e-9
    sub = frame[keep.to_numpy()].reset_index(drop=True)
    sub.attrs = frame.attrs
    return maker_m3(sub, params)
