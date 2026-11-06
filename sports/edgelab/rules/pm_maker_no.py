"""H2a -- Polymarket longshot-NO as MAKER on mention + culture/award markets (FUND_BOOK 3c H1/H2).

Frame (edgelab.loaders.pm_maker_no, built by scripts/pm_maker_no.py): one row per binary Yes/No market whose
category (scripts/smallmarket_calibration.pm_category, frozen 10-27) is `mentions` or `culture_tech`, with
  S = min(endDate, gameStartTime) (anchor), t = S - 24 h (decision), p_t = last CLOB display YES price at or before t
  (within 6 h; mid, or last trade if the spread > 10c) and its time p_t_time, closed_time = the market's close,
  p_close = display YES price at S - 1 h if the market was still open then (CLV mark), and
  attrs["trades"] = {ticker: DataFrame[timestamp, price, yes (bool: printed on the YES token)]} -- the public taker
  tape (data-api /v2/trades), every print after t.

Rule (fixed before any data was built for it, 2026-11-06):
  bet iff cat in {mentions, culture_tech}, 0.02 <= p_t <= 0.10, t < closed_time.
  Post a resting NO bid at q = 1 - a, a = p_t rounded UP to the cent (= offering YES at a; at or 1c inside the ask
  when the spread is >= 2c). No cancel; the order rests until the market closes.
  FILL = the first print strictly after t and <= closed_time that trades THROUGH our price: a YES-token print at
  price > a, or a NO-token print at price < q. (By price priority, a print through our level means every order at
  our level was filled first; prints AT our level are ignored = conservative queue model.) Unfilled = no bet.
  Hold to resolution. 1 contract per bet; price = q; maker fee 0 (Polymarket and Polymarket US charge takers only).
  CLV close = 1 - p_close (NO side).
Timestamps: decided_at = t, inputs_as_of = p_t_time, event_start = S, price_time = fill time,
  fill_deadline = closed_time; cluster = the UTC date of S (event-day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATS = ("mentions", "culture_tech")
P_LO, P_HI = 0.02, 0.10


def _ceil_cent(p):
    return np.ceil(np.round(np.asarray(p, float) * 100, 6)) / 100


def candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Every posted order (filled or not) with its modeled fill time -- shared by the rule and the diagnostics."""
    t = pd.to_datetime(frame["t"], utc=True)
    closed = pd.to_datetime(frame["closed_time"], utc=True)
    p = frame["p_t"].astype(float)
    ok = frame["cat"].isin(CATS) & (p >= P_LO - 1e-9) & (p <= P_HI + 1e-9) & (t < closed) & p.notna()
    f = frame[ok.to_numpy()].copy()
    a = _ceil_cent(f["p_t"])
    f["a"], f["q"] = a, np.round(1 - a, 2)
    tr = frame.attrs.get("trades", {})
    fills = []
    for tk, ti, cl, ai, qi in zip(f["ticker"], pd.to_datetime(f["t"], utc=True),
                                  pd.to_datetime(f["closed_time"], utc=True), f["a"], f["q"]):
        x = tr.get(tk)
        ft = pd.NaT
        if x is not None and len(x):
            ts = pd.to_datetime(x["timestamp"], unit="s", utc=True)
            px = x["price"].to_numpy(float)
            yes = x["yes"].to_numpy(bool)
            hit = (ts > ti) & (ts <= cl) & np.where(yes, px > ai + 1e-9, px < qi - 1e-9)
            if hit.any():
                ft = ts[hit].min()
        fills.append(ft)
    f["fill_time"] = pd.to_datetime(pd.Series(fills, index=f.index), utc=True)
    return f


def h2a_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    f = candidates(frame)
    f = f[f["fill_time"].notna()]
    S = pd.to_datetime(f["S"], utc=True)
    b = pd.DataFrame({
        "bet_id": f["ticker"].to_numpy(),
        "cluster": S.dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": S.to_numpy(),
        "decided_at": pd.to_datetime(f["t"], utc=True).to_numpy(),
        "inputs_as_of": pd.to_datetime(f["p_t_time"], utc=True).to_numpy(),
        "price": f["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": f["fill_time"].to_numpy(),
        "fill_deadline": pd.to_datetime(f["closed_time"], utc=True).to_numpy(),
        "side": "no",
        "stake": 1.0,
        "contracts": 1.0,
        "close_price": (1 - f["p_close"].astype(float)).to_numpy(),
    })
    return b.reset_index(drop=True)
