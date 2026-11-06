"""E1X-M YouTube (KXYTVIEWSW) consensus-reject, maker NO. Prereg: notes/e1x_media.md. Selection = scripts/e1x_media.py
yt_rungs(): rungs listed and still trading at T (Sunday 16:00Z of the Mon..Sun week) with K >= U * exp(r_max + 0.02),
U = lowest still-trading rung at T (rungs early-close publicly once the weekly max passes them), r_max = max over all
artist-weeks ended >= 7 d before of ln(weekly max / U at that week's T).

Price: post NO at q = floor_cent(min(NO mid, NO ask - 0.01)) from the last hourly candle ending <= T
(NO mid = 1 - (yes_bid + yes_ask)/2, NO ask = 1 - yes_bid); requires both quotes and 0.50 <= q <= 0.99.
Fill = first tape trade after T that prints THROUGH q (taker_side yes and yes_price > 1 - q), no cancel, held to
settlement; fill_deadline = close_time. Maker fee 0 (series fee_type quadratic).
Timestamps: decided_at = T; inputs_as_of = T (Kalshi early-close states <= T, prior weeks' settlements);
event_start = Monday 00:00Z after the week (Sunday's views cannot be published before then);
close_price (CLV) = NO ask on the last candle ending <= event_start.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _px(c, side):
    s = c.get(side, {})
    v = s.get("close_dollars", s.get("close"))
    return np.nan if v is None else float(v)


def _yp(t):
    v = float(t.get("yes_price_dollars", t.get("yes_price")))
    return v / 100 if v > 1.5 else v


def _quotes(frame, t_unix):
    cs = frame.attrs["candles"]
    yb, ya, te = [], [], []
    for tk, ts in zip(frame["ticker"], t_unix):
        c = [k for k in cs.get(tk, []) if k["end_period_ts"] <= ts]
        yb.append(_px(c[-1], "yes_bid") if c else np.nan)
        ya.append(_px(c[-1], "yes_ask") if c else np.nan)
        te.append(c[-1]["end_period_ts"] if c else np.nan)
    return np.array(yb), np.array(ya), np.array(te, float)


def _unix(s):
    return ((s - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()


def m1_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    T = pd.to_datetime(frame["T"], utc=True)
    wend = pd.to_datetime(frame["wend"]).dt.tz_localize("UTC")
    ev_start = wend + pd.Timedelta(days=1)
    yb, ya, te = _quotes(frame, _unix(T))
    cb, _, _ = _quotes(frame, _unix(ev_start))
    q = np.array([math.floor(round(min(1 - (b + a) / 2, 1 - b - 0.01) * 100, 6)) / 100
                  if not (np.isnan(b) or np.isnan(a)) else np.nan for b, a in zip(yb, ya)])
    ok = ~np.isnan(q) & (q >= 0.50) & (q <= 0.99)
    tr = frame.attrs["trades"]
    fill_t = [pd.NaT] * len(frame)
    for i, (tk, t0) in enumerate(zip(frame["ticker"], T)):
        if not ok[i]:
            continue
        for k in sorted(tr.get(tk, []), key=lambda k: k["created_time"]):
            ct = pd.Timestamp(k["created_time"])
            if ct <= t0:
                continue
            if k["taker_side"] == "yes" and _yp(k) > 1 - q[i] + 1e-9:
                fill_t[i] = ct
                break
    b = pd.DataFrame({"bet_id": frame["ticker"].to_numpy(), "cluster": frame["wend"].astype(str).to_numpy(),
                      "event_start": ev_start.to_numpy(), "decided_at": T.to_numpy(), "inputs_as_of": T.to_numpy(),
                      "price": q, "price_source": "maker_tape",
                      "price_time": pd.to_datetime(pd.Series(fill_t), utc=True).to_numpy(),
                      "side": "no", "stake": 1.0, "contracts": 1.0, "close_price": 1 - cb,
                      "fill_deadline": pd.to_datetime(frame["close_time"], utc=True).to_numpy()})
    filled = ok & b["price_time"].notna().to_numpy()
    return b[filled].reset_index(drop=True)
