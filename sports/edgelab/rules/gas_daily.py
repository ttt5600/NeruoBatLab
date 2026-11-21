"""G2: Kalshi KXAAAGASD (AAA US regular average on day D, "strictly greater than X"), far-rung certain side, taker.
Frame built by results/gas_daily_v2/build_frame.py (prereg notes/gas_daily_v2.md, 2026-11-21, before any candle).
Row = settled rung: ticker, event_ticker, date (= D), X (strike), T (decision time = 10:00 America/New_York on D-1, UTC),
close (23:59 ET D-1), p_m (model M: expanding AR2 + RB=F t4 predictive, strictly-past inputs), p_rw (random-walk
placebo), p_mom (naive-momentum placebo). attrs['candles'] = {ticker: [hourly candles]}.
RULE: quote = last candle with end_period_ts in (T-6h, T]; yes_bid / yes_ask = candle close (`close` or `close_dollars`).
  NO arm: p <= 0.03 and yes_bid >= 0.01 -> a = 1 - yes_bid; bet if (1 - p) - a - fee(a) >= 0.02.
  YES arm: p >= 0.97 and yes_ask <= 0.99 -> a = yes_ask; bet if p - a - fee(a) >= 0.02.
  fee(a) = ceil_cent(0.07 a (1-a)) (series fee_type quadratic, multiplier 1); 1 contract; cluster = D.
params: pcol (default p_m) selects the probability column (placebos p_rw / p_mom); mode 'base' = model-free favourite
placebo (favourite ask in [0.90, 0.99]; NO if yes_bid <= 0.10, YES if yes_ask >= 0.90).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
P_FAR, MIN_EDGE = 0.03, 0.02


def _fee(a):
    return math.ceil(round(0.07 * a * (1 - a) * 100, 6)) / 100


def _v(c, s):
    d = c.get(s) or {}
    v = d.get("close_dollars", d.get("close"))
    return np.nan if v is None else float(v)


def _q(cs, hi, lo=None):
    c = [k for k in cs if k["end_period_ts"] <= hi and (lo is None or k["end_period_ts"] > lo)]
    if not c:
        return np.nan, np.nan, None
    return _v(c[-1], "yes_bid"), _v(c[-1], "yes_ask"), c[-1]["end_period_ts"]


def _ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def pick(p, yb, ya, mode="model"):
    """-> (side, ask, p_side) or None. yb/ya in dollars; yb 0 / NaN = no bid."""
    if mode == "base":
        if np.isfinite(yb) and 0.01 <= yb <= 0.10 and 0.90 <= 1 - yb <= 0.99:
            return ("no", round(1 - yb, 4), np.nan)
        if np.isfinite(ya) and 0.90 <= ya <= 0.99:
            return ("yes", round(ya, 4), np.nan)
        return None
    if not np.isfinite(p):
        return None
    if p <= P_FAR and np.isfinite(yb) and yb >= 0.01:
        a = round(1 - yb, 4)
        if (1 - p) - a - _fee(a) >= MIN_EDGE:
            return ("no", a, 1 - p)
    if p >= 1 - P_FAR and np.isfinite(ya) and ya <= 0.99:
        a = round(ya, 4)
        if p - a - _fee(a) >= MIN_EDGE:
            return ("yes", a, p)
    return None


def g2_far(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    pcol, mode = params.get("pcol", "p_m"), params.get("mode", "model")
    cs_all = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        T, close = pd.Timestamp(r.T), pd.Timestamp(r.close)
        if close <= T:
            continue
        t = _ts(T)
        yb, ya, te = _q(cs_all.get(r.ticker, []), t, t - 6 * 3600)
        pk = pick(getattr(r, pcol), yb, ya, mode)
        if pk is None:
            continue
        cb, ca, _ = _q(cs_all.get(r.ticker, []), _ts(close))
        out.append({"bet_id": r.ticker, "cluster": str(pd.Timestamp(r.date).date()), "event_start": close, "decided_at": T,
                    "inputs_as_of": T - pd.Timedelta(hours=6), "price": pk[1], "price_source": "taker",
                    "price_time": pd.to_datetime(te, unit="s", utc=True), "side": pk[0], "stake": 1.0, "contracts": 1.0,
                    "close_price": (1 - cb) if pk[0] == "no" else ca, "p_model": pk[2]})
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "close_price", "p_model"]
    return pd.DataFrame(out, columns=cols)
