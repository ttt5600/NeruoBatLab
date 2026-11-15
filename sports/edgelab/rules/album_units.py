"""AU: Kalshi KXALBUMEQUIV first-week album units vs HitsDailyDouble chart vintages (consensus-reject, E4 analogue).
Prereg: notes/consensus_reject_census.md (AMENDMENT 1). Builder: scripts/album_units.py (selection = selected()).

Frame (loader kalshi_album_units): one row per (market, stage) already selected on inputs only:
  stage 'early' (T = Wednesday 12:00Z inside the Fri..Thu tracking week, F = HDD midweek-20 with _updatedAt < T) or
  'late' (T = Saturday 12:00Z after the week, F = HDD hits-top-50); side 'no' if K >= m*F (present) or K >= m*U
  (absent from a full chart, U = its smallest units), 'yes' if K <= F/m; m = 1.30 early, 1.15 late.
Taker: buy `side` at its ask from the last hourly candle ending <= T, + quadratic taker fee (evaluator), ask <= 0.98.
decided_at = T; inputs_as_of = the HDD doc's _updatedAt (< T); event_start = Friday 04:00Z after the week end (sales
complete); close_price (CLV) = our side's ask on the last candle ending <= event_start; cluster = week end.
The late trials died at the pre-registered data gate (HDD top-50 within 10% of Luminate on 0.864 < 0.90) -> retired.
"""
from __future__ import annotations

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


def _taker(frame: pd.DataFrame, stage: str) -> pd.DataFrame:
    f = frame[frame["stage"] == stage].reset_index(drop=True)
    cs = frame.attrs["candles"]
    T = pd.to_datetime(f["T"], utc=True)
    ev = pd.to_datetime(f["wend"]).dt.tz_localize("UTC") + pd.Timedelta(days=1, hours=4)
    tu = ((T - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
    eu = ((ev - EPOCH) // pd.Timedelta(seconds=1)).to_numpy()
    ask, close, pt = [], [], []
    for i, (tk, s) in enumerate(zip(f["ticker"], f["side"])):
        c = cs.get(f"{tk}|{int(tu[i])}", [])
        yb, ya, te = _quote(c, tu[i])
        cb, ca, _ = _quote(c, eu[i])
        ask.append(1 - yb if s == "no" else ya)
        close.append(1 - cb if s == "no" else ca)
        pt.append(te)
    ask = np.array(ask, float)
    b = pd.DataFrame({"bet_id": f["ticker"].to_numpy() + "|" + stage, "cluster": f["wend"].astype(str).to_numpy(),
                      "event_start": ev.to_numpy(), "decided_at": T.to_numpy(),
                      "inputs_as_of": pd.to_datetime(f["doc_up"], utc=True).to_numpy(),
                      "price": np.round(ask, 4), "price_source": "taker",
                      "price_time": pd.to_datetime(pt, unit="s", utc=True), "side": f["side"].to_numpy(),
                      "stake": 1.0, "contracts": 1.0, "close_price": np.array(close, float)})
    keep = np.isfinite(ask) & (ask > 0) & (ask <= ASK_MAX + 1e-9)
    return b[keep].reset_index(drop=True)


def early_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(frame, "early")


def late_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _taker(frame, "late")
