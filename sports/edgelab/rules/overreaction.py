"""OVR-T1 -- Polymarket overreaction fade: resting YES bid at the level of a sharp DOWN-jump (user hypothesis 11-06).

Frame (edgelab.loaders.pm_jumps, built by scripts/pm_jumps.py): one row per jump = the first print whose YES-equivalent
price moved >= 5c vs the last print at or before 15 min earlier (start price 0.03-0.97, one jump per market per 6 h),
with p1 = that print's YES price, decided_at = jump print + 60 s, S = min(closedTime, endDate, gameStartTime),
cat (smallmarket_calibration.pm_category), p_close = last YES print <= S - 1 h; attrs['trades'] = YES-equivalent tape.

Rule (fixed 2026-11-06 after the DEV descriptive study notes/overreaction_fade.md; sealed never pulled):
  bet iff sgn < 0 (YES fell), cat in {politics_world, finance_econ, sports, other}, 0.01 <= a <= 0.99, a = round(p1, 3).
  At decided_at post a YES bid at a; no cancel. FILL = first print strictly after decided_at and before S whose YES
  price is < a (traded THROUGH our level; prints at a ignored = back of queue). Unfilled = no bet.
  Hold to resolution; $1 stake; maker fee 0 (Polymarket charges takers only). CLV close = p_close.
Timestamps: decided_at, inputs_as_of = t_jump (newest print used), event_start = S, price_time = fill time,
  fill_deadline = S; cluster = UTC date of S.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ("politics_world", "finance_econ", "sports", "other")


def t1_down_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    f = frame[(frame["sgn"] < 0) & frame["cat"].isin(CATS)].copy()
    f["a"] = np.round(f["p1"].astype(float), 3)
    f = f[f["a"].between(0.01, 0.99)]
    tr = frame.attrs.get("trades", {})
    fills = []
    for tk, dec, S, a in zip(f["ticker"], f["decided_at"], f["S"], f["a"]):
        x = tr.get(tk)
        ft = np.nan
        if x is not None and len(x):
            ts = x["ts"].to_numpy(np.int64); p = x["p"].to_numpy(float)
            hit = np.flatnonzero((ts > dec) & (ts < S) & (p < a - 1e-9))
            if len(hit):
                ft = float(ts[hit[0]])
        fills.append(ft)
    f["fill_ts"] = fills
    f = f[f["fill_ts"].notna()]

    def sec(c):
        return pd.to_datetime(f[c].astype("int64"), unit="s", utc=True)

    S = sec("S")
    return pd.DataFrame({
        "bet_id": (f["ticker"].astype(str) + "|" + f["t_jump"].astype("int64").astype(str)).to_numpy(),
        "cluster": S.dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": S.to_numpy(),
        "decided_at": sec("decided_at").to_numpy(),
        "inputs_as_of": sec("t_jump").to_numpy(),
        "price": f["a"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": sec("fill_ts").to_numpy(),
        "fill_deadline": S.to_numpy(),
        "side": "yes",
        "stake": 1.0,
        "close_price": f["p_close"].astype(float).to_numpy(),
    }).reset_index(drop=True)
