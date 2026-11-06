"""DL-PW KXHORMUZ/PANAMA/SUEZ/BABELMANDEB WEEKLY far-bracket maker (notes/data_ladders_far.md). Selection = the
frozen scripts/dl_portwatch.py rungs() (K outside (F+1)e^(r_min/max -+ .05) - 1 at T = Wednesday 00:00Z of the week,
F = last published PortWatch week). Outcome columns are hidden; the vault settles each bet.

Timestamps (UTC): decided_at = T; quote = last hourly candle ending <= T; inputs_as_of = Tuesday 15:00Z (PortWatch
weekly publication ~14:13Z); event_start = Tuesday after week end 13:00Z (settlement data published ~14:13Z);
maker fill = first tape trade after T printing strictly through q, fill_deadline = close_time.
close for CLV = our side's ASK on the last candle ending <= Monday 00:00Z after the week.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _px(c, side):
    s = c.get(side, {})
    v = s.get("close_dollars", s.get("close"))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1.5 else v


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
    return np.array(yb, float), np.array(ya, float), np.array(te, float)


def t1_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    T = pd.to_datetime(frame["T"], utc=True)
    wend = pd.to_datetime(frame["wend"]).dt.tz_localize("UTC")
    ev_start = wend + pd.Timedelta(days=2, hours=13)
    t_unix = ((T - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
    yb, ya, te = _quotes(frame, t_unix)
    yb = np.where(np.isnan(yb) & ~np.isnan(ya), 0.0, yb)
    cl_unix = ((wend + pd.Timedelta(days=1) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
    cb, ca, _ = _quotes(frame, cl_unix)
    side = frame["side"].to_numpy()
    close_ask = np.where(side == "no", 1 - np.nan_to_num(cb, nan=0.0), ca)
    mid = (yb + ya) / 2
    qraw = np.where(side == "no", np.minimum(1 - mid, (1 - yb) - 0.01), np.minimum(mid, ya - 0.01))
    q = np.array([math.floor(v * 100 + 1e-6) / 100 if not np.isnan(v) else np.nan for v in qraw])
    ok = (~np.isnan(ya) & ((ya - yb) >= 0.02 - 1e-9) & np.where(side == "no", ya <= 0.99 + 1e-9, yb >= 0.01 - 1e-9)
          & (q >= 0.01) & (q <= 0.99))
    tr = frame.attrs["trades"]
    fill_t = [pd.NaT] * len(frame)
    for i, (tk, s, t0) in enumerate(zip(frame["ticker"], side, T)):
        if not ok[i]:
            continue
        for k in sorted(tr.get(tk, []), key=lambda k: k["created_time"]):
            ct = pd.Timestamp(k["created_time"])
            if ct <= t0:
                continue
            if (s == "no" and k["taker_side"] == "yes" and _yp(k) > 1 - q[i] + 1e-9) or \
               (s == "yes" and k["taker_side"] == "no" and _yp(k) < q[i] - 1e-9):
                fill_t[i] = ct
                break
    b = pd.DataFrame({"bet_id": frame["ticker"].to_numpy(), "cluster": frame["event_ticker"].to_numpy(),
                      "event_start": ev_start.to_numpy(), "decided_at": T.to_numpy(),
                      "inputs_as_of": (T - pd.Timedelta(hours=9)).to_numpy(), "side": side, "stake": 1.0,
                      "contracts": 1.0, "close_price": close_ask, "price": q, "price_source": "maker_tape"})
    b["price_time"] = pd.to_datetime(pd.Series(fill_t), utc=True)
    b["fill_deadline"] = pd.to_datetime(frame["close_time"], utc=True).to_numpy()
    filled = ok & b["price_time"].notna().to_numpy()
    return b[filled].reset_index(drop=True)
