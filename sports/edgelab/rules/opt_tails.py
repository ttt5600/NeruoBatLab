"""OT1-OT4: Kalshi DAILY crypto (KXBTCD/KXETHD 17:00 ET) and index (KXINX, KXNASDAQ100, KXINXU, KXNASDAQ100U 16:00 ET)
ladders vs the options market's implied distribution (builder scripts/options_implied_tails.py; prereg
notes/options_implied_tails.md, frozen 2026-11-15 before any quote / outcome was fetched).

Frame (loader kalshi_opt_tails): one row per DEV market whose options-implied P(YES) is <= .10 or >= .90. Inputs only:
  ticker, event_ticker, series, asset_class (crypto|index), date (ET close date = vault split column), T (11:00 ET),
  close_ts, p_yes (options-implied), inputs_as_of, yes_bid, yes_ask, quote_ts (last hourly candle ending <= T),
  mark_yes_bid, mark_yes_ask (last candle ending <= close - 1 h; CLV only)
attrs['tape'] = hourly candles ending in (T, close]: ticker, ts, yes_low, yes_high (trade prices), volume.

RULE: tail side = YES if p_yes <= .10, NO if p_yes >= .90 (p_T = its implied prob); F = the other side.
  quote filter 0 <= yes_bid < yes_ask <= 1, spread <= .05.
  taker: buy F at ask_F if (1 - p_T) - ask_F - 0.07 ask_F (1 - ask_F) >= .03 and ask_F in [.50, .99].
  maker: q = min(floor_cent(mid_F), ask_F - .01) in [.50, .99] and (1 - p_T) - q >= .03; filled iff a tape candle shows
         an F-side trade price <= q - .01 (F = NO: yes_high >= 1 - q + .01; F = YES: yes_low <= q - .01). Maker fee 0.
  cluster = asset_class|date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

K_TAIL = 0.10
EDGE = 0.03
SPREAD = 0.05


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def _base(frame, asset_class):
    f = frame[frame["asset_class"] == asset_class]
    p = f["p_yes"].to_numpy(float)
    yb, ya = f["yes_bid"].to_numpy(float), f["yes_ask"].to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya)
    ok &= (ya - yb <= SPREAD + 1e-9)
    side = np.where(ok & (p <= K_TAIL), "no", np.where(ok & (p >= 1 - K_TAIL), "yes", ""))
    pF = np.where(side == "no", 1 - p, p)  # implied prob of the side we buy
    askF = np.where(side == "no", 1 - yb, ya)
    midF = np.where(side == "no", 1 - (yb + ya) / 2, (yb + ya) / 2)
    return f, side, pF, askF, midF


def _rows(s, side, price, src, ptime):
    mb, ma = s["mark_yes_bid"].to_numpy(float), s["mark_yes_ask"].to_numpy(float)
    return pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": (s["asset_class"] + "|" + pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d")).to_numpy(),
        "event_start": _ts(s["close_ts"].to_numpy("int64") - 1),
        "decided_at": _ts(s["T"].to_numpy("int64")),
        "inputs_as_of": _ts(np.minimum(s["inputs_as_of"].to_numpy("int64"), s["T"].to_numpy("int64"))),
        "side": side,
        "stake": 1.0,
        "price": price,
        "price_source": src,
        "price_time": ptime,
        "close_price": np.where(side == "no", 1 - mb, ma),
    })


def taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    f, side, pF, askF, _ = _base(frame, params["asset_class"])
    edge = pF - askF - 0.07 * askF * (1 - askF)
    keep = (side != "") & (edge >= EDGE - 1e-9) & (askF >= 0.50 - 1e-9) & (askF <= 0.99 + 1e-9)
    s = f[keep]
    return _rows(s, side[keep], np.round(askF[keep], 4), "taker", _ts(s["quote_ts"].to_numpy("int64"))
                 ).reset_index(drop=True)


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    f, side, pF, askF, midF = _base(frame, params["asset_class"])
    q = np.round(np.minimum(_cent_floor(midF), askF - 0.01), 4)
    keep = (side != "") & (q >= 0.50 - 1e-9) & (q <= 0.99 + 1e-9) & (pF - q >= EDGE - 1e-9)
    s, sd, qq = f[keep], side[keep], q[keep]
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, si, qi) in enumerate(zip(s["ticker"], sd, qq)):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        if si == "no":
            hit = d["yes_high"].to_numpy(float) >= 1 - qi + 0.01 - 1e-9
        else:
            hit = d["yes_low"].to_numpy(float) <= qi - 0.01 + 1e-9
        hit &= ts > T[i]
        if hit.any():
            fill_t[i] = ts[hit][0]
    out = _rows(s, sd, qq, "maker_tape", _ts(np.where(fill_t >= 0, fill_t, T)))
    out["fill_deadline"] = _ts(s["close_ts"].to_numpy("int64"))
    out["_filled"] = fill_t >= 0
    if params.get("return_unfilled"):
        return out.reset_index(drop=True)
    return out[out["_filled"]].drop(columns="_filled").reset_index(drop=True)
