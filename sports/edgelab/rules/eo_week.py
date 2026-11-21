"""EO1-Y: Kalshi KXEOWEEK ("President signs above k executive orders during <Sun>..<Sat>"), locked-YES taker.
Frame built by results/count_nowcast/eo1.py (prereg notes/count_nowcast.md, written 2026-11-21 before any candle).
Row = one settled rung: ticker, ev, ws/we (week), k, n (FR EO count signed in week, unique EO numbers), Tk = 12:00Z on the
FR publication date of the (k+1)-th EO signed in the week (first moment the settlement source itself shows count > k),
close (market close_time), date = week end. attrs['candles'] = hourly candles (yes_ask/yes_bid close).
RULE: if Tk exists and close > Tk: quote = last candle ending in (Tk-6h, Tk+1h]; buy YES at yes_ask if 0.50 <= ask <= 0.98;
fee = ceil_cent(0.07*a*(1-a)), 1 contract. Cluster = week. CLV close = YES ask on last candle <= close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _q(cs, hi, lo=None):
    c = [k for k in cs if k["end_period_ts"] <= hi and (lo is None or k["end_period_ts"] > lo)]
    if not c:
        return np.nan, None
    v = (c[-1].get("yes_ask") or {}).get("close")
    return (np.nan if v is None else float(v)), c[-1]["end_period_ts"]


def eo1_locked_yes(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    cs_all = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        if pd.isna(r.Tk):
            continue
        Tk, close = pd.Timestamp(r.Tk), pd.Timestamp(r.close)
        if close <= Tk:
            continue
        t = int((Tk - EPOCH) // pd.Timedelta(seconds=1))
        a, te = _q(cs_all.get(r.ticker, []), t + 3600, t - 6 * 3600)
        if not (np.isfinite(a) and 0.50 <= a <= 0.98):
            continue
        ca, _ = _q(cs_all.get(r.ticker, []), int((close - EPOCH) // pd.Timedelta(seconds=1)))
        out.append({"bet_id": r.ticker, "cluster": str(pd.Timestamp(r.we).date()), "event_start": close,
                    "decided_at": Tk + pd.Timedelta(hours=1), "inputs_as_of": Tk - pd.Timedelta(hours=1),
                    "price": round(a, 4), "price_source": "taker", "price_time": pd.to_datetime(te, unit="s", utc=True),
                    "side": "yes", "stake": 1.0, "contracts": 1.0, "close_price": ca, "p_model": 1.0})
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "close_price", "p_model"]
    return pd.DataFrame(out, columns=cols)
