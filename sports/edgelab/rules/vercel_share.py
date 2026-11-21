"""VS1 / VS2: Kalshi Vercel AI Gateway daily share ladders (KXOPENSOURCESHARE, KXOSSPEND, 9 lab spend/request series).
Frame built by results/vercel_share/build_frame.py (prereg notes/vercel_share.md, 2026-11-21, before any candle).
Row = settled rung: ticker, event_ticker, series, date (= value day D, UTC), X (strike, YES iff round(V_D,1) > X), T1 (05:00Z on D),
close (Kalshi close_time), V (day-D source value; known after D+1 00:00Z), V1 (day D-1 value), p_m (AR1 on logit, trailing 60 d,
input x_{D-1}), p_persist (lag-2 AR applied to x_{D-2}), p_clim (trailing 60-day level distribution). attrs['candles'].
VS1 (vs1_far): quote = last candle with end_period_ts in (T1-6h, T1]; YES if p >= .85 and p - ask - fee >= .05; NO if p <= .15 and
  (1-p) - (1 - yes_bid) - fee >= .05. params pcol in {p_m, p_persist, p_clim}.
VS2 (vs2_post): candles with end_period_ts in [D+1 01:00Z, close]; YES if X <= V - 0.3, NO if X >= V + 0.3, cost a + fee <= .97;
  first qualifying candle. params vcol in {V, V1} (V1 = placebo).
fee(a) = ceil_cent(0.07 a (1-a)), 1 contract (quadratic, multiplier 1). cluster = D.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
P_FAR, MIN_EDGE, MARGIN, MAX_COST = 0.15, 0.05, 0.3, 0.97
COLS = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
        "side", "stake", "contracts", "close_price", "p_model"]


def fee(a):
    return math.ceil(round(0.07 * a * (1 - a) * 100, 6)) / 100


def _v(c, s):
    d = c.get(s) or {}
    v = d.get("close_dollars", d.get("close"))
    return np.nan if v is None else float(v)


def _ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def _q(cs, hi, lo=None):
    c = [k for k in cs if k["end_period_ts"] <= hi and (lo is None or k["end_period_ts"] > lo)]
    if not c:
        return np.nan, np.nan, None
    return _v(c[-1], "yes_bid"), _v(c[-1], "yes_ask"), c[-1]["end_period_ts"]


def _ok(x):
    return np.isfinite(x) and 0.01 <= x <= 0.99


def pick_far(p, yb, ya):
    if not np.isfinite(p):
        return None
    if p >= 1 - P_FAR and _ok(ya):
        a = round(ya, 4)
        if p - a - fee(a) >= MIN_EDGE:
            return ("yes", a, p)
    if p <= P_FAR and _ok(yb):
        a = round(1 - yb, 4)
        if (1 - p) - a - fee(a) >= MIN_EDGE:
            return ("no", a, 1 - p)
    return None


def _row(r, pk, T, te, cs, close):
    cb, ca, _ = _q(cs, _ts(close))
    return {"bet_id": r.ticker, "cluster": str(pd.Timestamp(r.date).date()), "event_start": close, "decided_at": T,
            "inputs_as_of": T, "price": pk[1], "price_source": "taker",
            "price_time": pd.to_datetime(te, unit="s", utc=True), "side": pk[0], "stake": 1.0, "contracts": 1.0,
            "close_price": (1 - cb) if pk[0] == "no" else ca, "p_model": pk[2]}


def vs1_far(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    pcol = params.get("pcol", "p_m")
    cs_all = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        T, close = pd.Timestamp(r.T1), pd.Timestamp(r.close)
        if close <= T:
            continue
        t = _ts(T)
        cs = cs_all.get(r.ticker, [])
        yb, ya, te = _q(cs, t, t - 6 * 3600)
        pk = pick_far(getattr(r, pcol), yb, ya)
        if pk is not None:
            out.append(_row(r, pk, T, te, cs, close))
    return pd.DataFrame(out, columns=COLS)


def vs2_post(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    vcol = params.get("vcol", "V")
    cs_all = frame.attrs["candles"]
    out = []
    for r in frame.itertuples(index=False):
        close = pd.Timestamp(r.close)
        start = pd.Timestamp(r.date).tz_localize("UTC") if pd.Timestamp(r.date).tzinfo is None else pd.Timestamp(r.date)
        t0 = _ts(start + pd.Timedelta(days=1, hours=1))
        V = getattr(r, vcol)
        if not np.isfinite(V):
            continue
        cs = [k for k in cs_all.get(r.ticker, []) if t0 <= k["end_period_ts"] <= _ts(close)]
        for k in cs:
            yb, ya = _v(k, "yes_bid"), _v(k, "yes_ask")
            pk = None
            if r.X <= V - MARGIN and _ok(ya) and ya + fee(ya) <= MAX_COST:
                pk = ("yes", round(ya, 4), np.nan)
            elif r.X >= V + MARGIN and _ok(yb) and (1 - yb) + fee(1 - yb) <= MAX_COST:
                pk = ("no", round(1 - yb, 4), np.nan)
            if pk is not None:
                T = pd.to_datetime(k["end_period_ts"], unit="s", utc=True)
                out.append(_row(r, pk, T, k["end_period_ts"], cs_all.get(r.ticker, []), close))
                break
    return pd.DataFrame(out, columns=COLS)
