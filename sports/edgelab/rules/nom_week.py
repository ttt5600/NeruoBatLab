"""NM1: Kalshi KXTRUMPNOMNUM ("Trump sent at least k nominations to the Senate Sun..Sat", whitehouse.gov dates), taker.
Frame built by results/count_nowcast/nm1.py (prereg notes/count_nowcast.md, 2026-11-21 before any candle).
Row = settled rung: ticker, ws (Sunday), k, C (congress.gov civilian nominations with receivedDate Sun..Wed), T = Thursday
15:00Z, close, date = Saturday. attrs['candles'] hourly (yes_bid/yes_ask close or close_dollars).
RULE (two arms, one trial): quote = last candle in (T-6h, T]. N: if C == 0 and k >= 1, p_yes = 0.065 (Jeffreys, 1/22 fit
weeks 2025-07..2026-06), buy NO at 1 - yes_bid if 0.935 - a - fee(a) >= 0.02 and a >= 0.50. Y: if C >= k, buy YES at ask
if 0.50 <= ask <= 0.98. fee = ceil_cent(0.07 a (1-a)); 1 contract; cluster = week; no bet if close <= T.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
P_LATE = 0.065


def _fee(a):
    return math.ceil(round(0.07 * a * (1 - a) * 100, 6)) / 100


def _v(c, s):
    d = c.get(s) or {}
    v = d.get("close", d.get("close_dollars"))
    return np.nan if v is None else float(v)


def _q(cs, hi, lo=None):
    c = [k for k in cs if k["end_period_ts"] <= hi and (lo is None or k["end_period_ts"] > lo)]
    if not c:
        return np.nan, np.nan, None
    return _v(c[-1], "yes_bid"), _v(c[-1], "yes_ask"), c[-1]["end_period_ts"]


def nm1_nowcast(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    cs_all = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        T, close = pd.Timestamp(r.T), pd.Timestamp(r.close)
        if close <= T:
            continue
        t = int((T - EPOCH) // pd.Timedelta(seconds=1))
        yb, ya, te = _q(cs_all.get(r.ticker, []), t, t - 6 * 3600)
        if not (np.isfinite(yb) and np.isfinite(ya)):
            continue
        pick = None
        if r.C == 0 and r.k >= 1:
            a = 1 - yb
            if a >= 0.50 and (1 - P_LATE) - a - _fee(a) >= 0.02:
                pick = ("no", a, 1 - P_LATE)
        if pick is None and r.C >= r.k and 0.50 <= ya <= 0.98:
            pick = ("yes", ya, 1.0)
        if pick is None:
            continue
        cb, ca, _ = _q(cs_all.get(r.ticker, []), int((close - EPOCH) // pd.Timedelta(seconds=1)))
        out.append({"bet_id": r.ticker, "cluster": str(pd.Timestamp(r.ws).date()), "event_start": close, "decided_at": T,
                    "inputs_as_of": T - pd.Timedelta(hours=15), "price": round(pick[1], 4), "price_source": "taker",
                    "price_time": pd.to_datetime(te, unit="s", utc=True), "side": pick[0], "stake": 1.0, "contracts": 1.0,
                    "close_price": (1 - cb) if pick[0] == "no" else ca, "p_model": pick[2]})
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "close_price", "p_model"]
    return pd.DataFrame(out, columns=cols)
