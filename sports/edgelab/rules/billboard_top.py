"""TA: Kalshi KXTOPALBUM (Billboard 200 #1) vs HitsDailyDouble midweek-20 / hits-top-50 vintages.
Prereg: notes/billboard_top.md (incl. AMENDMENT 1). Builder: scripts/billboard_top.py (selection = selected()).

Frame (loader kalshi_top_album): one row per (market, stage) selected on inputs only. Leader L = HDD doc's top row;
YES on L if F_L >= m F2, NO on rivals with F <= F_L/m or absent from the full chart with F_L >= m U; m = 1.30 early
(T = Wed 12:00Z inside the Fri..Thu week, midweek-20), 1.15 late (T = Sat 12:00Z after the week, hits-top-50).
decided_at = T; inputs_as_of = doc _updatedAt (< T); event_start = Sunday 12:00Z after the week (before Billboard's
reveal); close_price = our side's ask on the last candle <= event_start; cluster = week end.
Taker: ask from the last hourly candle ending <= T, ask <= 0.98. Maker (early): q = floor_cent(min(mid, ask-1c)),
0.50 <= q <= 0.98, filled at the first tape trade through q after T and <= Kalshi close.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

ASK_MAX = 0.98
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _px(c, side):
    s = c.get(side) or {}
    v = s.get("close_dollars", s.get("close"))
    return np.nan if v is None else float(v)


def _quote(cs, ts):
    c = [k for k in cs if k["end_period_ts"] <= ts]
    if not c:
        return np.nan, np.nan, np.nan
    return _px(c[-1], "yes_bid"), _px(c[-1], "yes_ask"), float(c[-1]["end_period_ts"])


def _u(s):
    return ((pd.to_datetime(s, utc=True) - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()


def _priced(frame, stage):
    f = frame[frame["stage"] == stage].reset_index(drop=True)
    cs = frame.attrs["candles"]
    ev = pd.to_datetime(f["wend"]).dt.tz_localize("UTC") + pd.Timedelta(days=3, hours=12)
    tu, eu = _u(f["T"]), _u(ev)
    ask, bid, close, pt = [], [], [], []
    for i, (tk, s) in enumerate(zip(f["ticker"], f["side"])):
        c = cs.get(f"{tk}|{int(tu[i])}", [])
        yb, ya, te = _quote(c, tu[i])
        cb, ca, _ = _quote(c, eu[i])
        ask.append(1 - yb if s == "no" else ya)
        bid.append(1 - ya if s == "no" else yb)
        close.append(1 - cb if s == "no" else ca)
        pt.append(te)
    return f, ev, np.array(ask, float), np.array(bid, float), np.array(close, float), np.array(pt, float)


def _base(f, ev, close):
    return pd.DataFrame({"bet_id": f["ticker"].to_numpy() + "|" + f["stage"].to_numpy(),
                         "cluster": f["wend"].astype(str).to_numpy(), "event_start": ev.to_numpy(),
                         "decided_at": pd.to_datetime(f["T"], utc=True).to_numpy(),
                         "inputs_as_of": pd.to_datetime(f["doc_up"], utc=True).to_numpy(),
                         "side": f["side"].to_numpy(), "stake": 1.0, "contracts": 1.0, "close_price": close})


def _taker(frame, stage):
    f, ev, ask, bid, close, pt = _priced(frame, stage)
    b = _base(f, ev, close)
    b["price"], b["price_source"] = np.round(ask, 4), "taker"
    b["price_time"] = pd.to_datetime(pt, unit="s", utc=True)
    keep = np.isfinite(ask) & (ask > 0) & (ask <= ASK_MAX + 1e-9)
    return b[keep].reset_index(drop=True)


def early_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(frame, "early")


def late_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(frame, "late")


def _yp(t):
    v = float(t.get("yes_price_dollars", t.get("yes_price")))
    return v / 100 if v > 1.5 else v


def early_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    f, ev, ask, bid, close, pt = _priced(frame, "early")
    trd = frame.attrs.get("trades", {})
    b = _base(f, ev, close)
    tu, cu = _u(f["T"]), _u(f["close_time"])
    q_, ft, keep = [], [], []
    for i, (tk, s) in enumerate(zip(f["ticker"], f["side"])):
        if not (np.isfinite(ask[i]) and np.isfinite(bid[i])):
            continue
        q = math.floor(round(min((ask[i] + bid[i]) / 2, ask[i] - 0.01) * 100, 6)) / 100
        if not (0.50 <= q <= 0.98):
            continue
        for k in sorted(trd.get(tk, []), key=lambda k: k["created_time"]):
            ct = int((pd.Timestamp(k["created_time"]) - EPOCH) // pd.Timedelta(seconds=1))
            if ct <= tu[i] or ct > cu[i]:
                continue
            yp = _yp(k)
            if (s == "no" and k["taker_side"] == "yes" and yp > 1 - q + 1e-9) or \
               (s == "yes" and k["taker_side"] == "no" and yp < q - 1e-9):
                keep.append(i); q_.append(q); ft.append(ct); break
    b = b.iloc[keep].reset_index(drop=True)
    b["price"], b["price_source"] = np.array(q_, float), "maker_tape"
    b["price_time"] = pd.to_datetime(np.array(ft, "int64"), unit="s", utc=True)
    b["fill_deadline"] = pd.to_datetime(f["close_time"].iloc[keep].to_numpy(), utc=True)
    return b
