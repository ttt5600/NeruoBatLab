"""HT1: Kalshi HOURLY temperature (KXTEMP*H / *HS) consensus-reject MAKER (scripts/kalshi_hourly_temp.py builds data).

Contract: market for hour H (close_time = H, opens H - 1 h) pays YES iff the series' temperature value at H (Synoptic /
Kalshi Weather Index; `expiration_value`) is > K (strike_type 'greater', floor_strike K). Values settle ~40-70 min after H.

Frame (loader kalshi_temp_hourly): one row per market. Inputs only:
  ticker, series, event_ticker, date (event-ticker local date = vault split column), H (close ts), open_ts, K, strike_type
  T        decision = open_ts + 15 min (= H - 45 min)
  v2, v2_settle_ts   the SAME series' published value for hour H - 2 and when that market settled (public then)
  d_<model>  each Open-Meteo model's forecast change temperature(H) - temperature(H-2) at the city point, from the
             previous-runs 'previous_day1' field (issued ~24 h before each valid hour -> before T); NaN when missing
  yes_bid, yes_ask   last 1-min candle ending <= T (event candlesticks endpoint)
  mark_yes_bid, mark_yes_ask   last 1-min candle ending <= H - 5 min: used ONLY as the CLV close mark
attrs['tape'] = public trades (ticker, ts, yes_price, no_price, count) in (T, H] for every market that passes the
input-only selection below (the builder applies the identical selection to decide which tapes to download).

RULE (frozen 2026-11-06, before any hourly price / outcome history was pulled):
  require >= 3 models with d_m; prediction p_m = v2 + d_m; require v2_settle_ts <= T (else skip -- not yet public).
  NO pick  : max_m p_m < K - M   (every model trajectory misses the strike from below by > M)
  YES pick : min_m p_m > K + M   (every model trajectory clears the strike by > M)   M = 3.0 F
  quote filter: 0 <= yes_bid < yes_ask <= 1, spread <= 0.10.
  Maker (E4 M3 pricing mirrored per side): post-only bid on the picked side at
     q = min(floor_cent(side mid), side ask - 0.01); skip if q < 0.50 or q > 0.99.
     NO side: mid = 1 - (yes_bid + yes_ask)/2, ask = 1 - yes_bid; filled iff >= 10 contracts print at no_price <= q in
     (T, H]. YES side: mid = (yes_bid + yes_ask)/2, ask = yes_ask; filled iff >= 10 contracts print at yes_price <= q.
     Fill time = when the 10th contract prints. Unfilled -> no bet. Hold to settlement. Maker fee 0 (fee_type quadratic).
  decided_at = T; inputs_as_of = v2_settle_ts; event_start = H; fill_deadline = H.
  close_price (CLV) = same side's ASK on the mark candle (<= H - 5 min).
  cluster = event-ticker date (all cities/hours of a day share weather regimes).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M = 3.0
N_MIN = 10.0
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def select(frame: pd.DataFrame, m: float = M) -> pd.DataFrame:
    """Input-only selection -> frame rows with a 'side' column ('yes'/'no') and maker price q."""
    d = frame[[f"d_{x}" for x in MODELS]].to_numpy(float)
    nm = np.isfinite(d).sum(1)
    p = frame["v2"].to_numpy(float)[:, None] + d
    with np.errstate(invalid="ignore"):
        pmax, pmin = np.nanmax(np.where(np.isfinite(p), p, -np.inf), 1), np.nanmin(np.where(np.isfinite(p), p, np.inf), 1)
    K = frame["K"].to_numpy(float)
    ok = (nm >= 3) & (frame["strike_type"].to_numpy() == "greater") & np.isfinite(frame["v2"].to_numpy(float)) \
        & (frame["v2_settle_ts"].to_numpy("int64") <= frame["T"].to_numpy("int64"))
    yb, ya = frame["yes_bid"].to_numpy(float), frame["yes_ask"].to_numpy(float)
    ok &= np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya) & (ya - yb <= 0.10 + 1e-9)
    side = np.where(ok & (pmax < K - m), "no", np.where(ok & (pmin > K + m), "yes", ""))
    mid_side = np.where(side == "no", 1 - (yb + ya) / 2, (yb + ya) / 2)
    ask_side = np.where(side == "no", 1 - yb, ya)
    q = np.round(np.minimum(_cent_floor(mid_side), ask_side - 0.01), 4)
    keep = (side != "") & (q >= 0.50 - 1e-9) & (q <= 0.99 + 1e-9)
    out = frame[keep].copy()
    out["side"] = side[keep]
    out["q"] = q[keep]
    return out


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    n_min = float(params.get("n_min", N_MIN))
    s = select(frame, float(params.get("margin", M)))
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    H = s["H"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, sd, qq) in enumerate(zip(s["ticker"], s["side"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price" if sd == "no" else "yes_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= H[i]) & (px <= qq + 1e-9)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    mb, ma = s["mark_yes_bid"].to_numpy(float), s["mark_yes_ask"].to_numpy(float)
    b = pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(H),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(s["v2_settle_ts"].to_numpy("int64")),
        "side": s["side"].to_numpy(),
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(H),
        "close_price": np.where(s["side"].to_numpy() == "no", 1 - mb, ma),
    })
    return b[fill_t >= 0].reset_index(drop=True)
