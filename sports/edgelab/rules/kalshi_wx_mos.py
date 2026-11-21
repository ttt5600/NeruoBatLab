"""E4 + station max-T MOS data (agent e4_data_v2 2026-11-21, prereg sports/notes/e4_data_v2.md).

Frame (built by scripts/e4_data_v2.py frame -> results/e4_data_v2/edgelab_frame.parquet): E1 orig-7 dev brackets with a
tradeable 22:00 quote (E4 filter), the 5 frozen fcsafe_* members, point-in-time MOS members (nbs = NBM NBS 19Z txn, mav =
GFS MOS 12Z n_x, met = NAM MOS 12Z n_x; all issued on D-1 before the decision), the walk-forward margin m_wf (tail-matched)
and the walk-forward station EMOS-L P(hit) phit_l. Selection is done HERE so the registered hash covers it:
  t1 mosvote      : frozen E4 with 3 extra voters (5 fcsafe + present MOS), >= 3 fcsafe, every member misses by > 3F
  t2 mosconsensus : members {nbs, mav, met}, nbs and mav present, every present member misses by > m_wf
  t3 cl_new       : NOT an E4 pick and phit_l <= 0.02
Execution = maker trade-through (E4 live): q = min(floor_cent(NO mid), ask - 1c), filled iff ANY print at no_price < q in
(T, close]; fill time = first such print. Copy of kalshi_wx_prob._maker so the hash covers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .kalshi_wx import _base, _close_no, _cent_floor, _ts

FC = ["fcsafe_gfs_seamless", "fcsafe_ecmwf_ifs025", "fcsafe_icon_seamless", "fcsafe_gem_seamless", "fcsafe_ncep_nbm_conus"]
MOS = ["nbs", "mav", "met"]
PSTAR = 0.02


def _reject(frame, cols, M):
    m = frame[cols].to_numpy(float)
    ok = np.isfinite(m)
    hi = np.where(ok, m, -np.inf).max(1)
    lo = np.where(ok, m, np.inf).min(1)
    f, c, st = frame["floor_strike"].to_numpy(float), frame["cap_strike"].to_numpy(float), frame["strike_type"].to_numpy()
    r = np.where(st == "greater", hi + M < f + 1, np.where(st == "less", lo - M > c - 1, (hi + M < f) | (lo - M > c)))
    return r & ok.any(1)


def sel_e4(frame):
    return _reject(frame, FC, 3.0) & (frame[FC].notna().sum(axis=1).to_numpy() >= 3)


def sel_t1(frame):
    return _reject(frame, FC + MOS, 3.0) & (frame[FC].notna().sum(axis=1).to_numpy() >= 3)


def sel_t2(frame):
    ok = frame["nbs"].notna().to_numpy() & frame["mav"].notna().to_numpy() & np.isfinite(frame["m_wf"].to_numpy(float))
    return _reject(frame, MOS, frame["m_wf"].to_numpy(float)) & ok


def sel_t3(frame):
    return (~sel_e4(frame)) & (frame["phit_l"].to_numpy(float) <= PSTAR)


def _q(frame):
    ask = 1 - frame["yes_bid"].to_numpy(float)
    return np.round(np.minimum(_cent_floor(1 - (frame["yes_bid"].to_numpy(float) + frame["yes_ask"].to_numpy(float)) / 2),
                               ask - 0.01), 4)


def _maker(frame):
    b, T = _base(frame)
    q = _q(frame)
    close = frame["close_ts"].to_numpy("int64")
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None else {}
    fill_t = np.full(len(frame), -1, "int64")
    for i, (tk, t0, t1) in enumerate(zip(frame["ticker"], T, close)):
        d = g.get(tk)
        if d is None:
            continue
        ts, px = d["ts"].to_numpy(), d["no_price"].to_numpy()
        m = (ts > t0) & (ts <= t1) & (px < q[i] - 1e-9)
        if m.any():
            fill_t[i] = ts[m].min()
    filled = fill_t >= 0
    b["price"] = q
    b["price_source"] = "maker_tape"
    b["price_time"] = _ts(np.where(filled, fill_t, T))
    b["fill_deadline"] = _ts(close)
    b["close_price"] = _close_no(frame, T + 10 * 3600)
    return b[filled].reset_index(drop=True)


def _sub(frame, m):
    s = frame[m].reset_index(drop=True)
    s.attrs = frame.attrs
    return s


def maker_t1(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _maker(_sub(frame, sel_t1(frame)))


def maker_t2(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _maker(_sub(frame, sel_t2(frame)))


def maker_t3(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _maker(_sub(frame, sel_t3(frame)))
