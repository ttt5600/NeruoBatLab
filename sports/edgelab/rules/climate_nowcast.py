"""CN1: Kalshi NOAA monthly global temperature (KXHMONTHRANGE brackets + KXHMONTH 'hottest M ever'), ERA5 nowcast taker.
Frame built by scripts/climate_nowcast.py build (prereg: notes/climate_nowcast.md, frozen 2026-11-20 before any candle
or ERA5-vs-settlement comparison was looked at).

Frame row = one settled contract: ticker, event_ticker, series, month, strike_type (less|between|greater|None),
lo/hi (Kalshi floor/cap strike), rec (NOAA archive record for the calendar month, years < year(M)), pred (ERA5 month mean
anomaly 1991-2020 + calendar-month offset, fit 2001-2023), pred_pers (ERA5 of M-1 + offset), ladder_centre, sd (fit x 1.25),
T (decision, day 4 of M+1 16:00Z), inputs_as_of (day 3 00:00Z), event_start = min(close_time, day 8 00:00Z), close_time.
attrs['candles'] = {ticker: hourly candles (end_period_ts unix s, yes_ask/yes_bid close)}.

RULE: p_yes from N(pred, sd) (bracket (lo, hi], 'less' = (-inf, hi), 'greater' = (lo, inf); HMONTH: P(value > rec)).
Quote = last candle ending in (T-6h, T]; YES ask = yes_ask close, NO ask = 1 - yes_bid close; null -> no bet.
Buy side s if p(s) - ask(s) - fee(ask) >= EDGE and 0.02 <= ask <= 0.97; fee = ceil_cent(0.07*10*a*(1-a))/10.
No bet if close_time <= T. CLV close = our ask on last candle ending <= event_start. Cluster = month. contracts = 10.
params: pred_col (pred | pred_pers | ladder_centre -> placebos, descriptive only), edge.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

EDGE = 0.05
C = 10
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _fee(a):
    return math.ceil(round(0.07 * C * a * (1 - a) * 100, 6)) / 100 / C


def _px(c, side):
    v = (c.get(side) or {}).get("close")
    return np.nan if v is None else float(v)


def _quote(cs, t, lo=None):
    c = [k for k in cs if k["end_period_ts"] <= t and (lo is None or k["end_period_ts"] > lo)]
    if not c:
        return np.nan, np.nan, np.nan
    k = c[-1]
    return _px(k, "yes_bid"), _px(k, "yes_ask"), k["end_period_ts"]


def p_yes(r, pred):
    if not np.isfinite(pred):
        return np.nan
    if r.series == "KXHMONTH":
        return 1 - norm.cdf((r.rec - pred) / r.sd)
    lo = -np.inf if r.strike_type == "less" or not np.isfinite(r.lo) else r.lo
    hi = np.inf if r.strike_type == "greater" or not np.isfinite(r.hi) else r.hi
    return norm.cdf((hi - pred) / r.sd) - norm.cdf((lo - pred) / r.sd)


def bets(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    col, edge = params.get("pred_col", "pred"), float(params.get("edge", EDGE))
    cs = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        T = pd.Timestamp(r.T)
        if pd.Timestamp(r.close_time) <= T:
            continue
        pred = getattr(r, col)
        if col == "ladder_centre" and r.series != "KXHMONTHRANGE":
            continue
        py = p_yes(r, pred)
        if not np.isfinite(py):
            continue
        t = int((T - EPOCH) // pd.Timedelta(seconds=1))
        yb, ya, te = _quote(cs.get(r.ticker, []), t, t - 6 * 3600)
        if not (np.isfinite(yb) and np.isfinite(ya)):
            continue
        best = None
        for side, p, a in (("yes", py, ya), ("no", 1 - py, 1 - yb)):
            if 0.02 <= a <= 0.97:
                e = p - a - _fee(a)
                if e >= edge and (best is None or e > best[3]):
                    best = (side, p, a, e)
        if best is None:
            continue
        es = pd.Timestamp(r.event_start)
        cb, ca, _ = _quote(cs.get(r.ticker, []), int((es - EPOCH) // pd.Timedelta(seconds=1)))
        close = (1 - cb) if best[0] == "no" else ca
        out.append({"bet_id": r.ticker, "cluster": r.month, "event_start": es, "decided_at": T,
                    "inputs_as_of": pd.Timestamp(r.inputs_as_of), "price": round(best[2], 4), "price_source": "taker",
                    "price_time": pd.to_datetime(te, unit="s", utc=True), "side": best[0], "stake": 1.0,
                    "contracts": float(C), "close_price": close, "p_model": best[1]})
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "close_price", "p_model"]
    return pd.DataFrame(out, columns=cols)


def cn1_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return bets(frame, params)
