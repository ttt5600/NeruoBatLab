"""ND1: KXTEMPNYCHS (Kalshi Weather Index NYC, 8-station 1-min) nowcast at T = H - 20 min (prereg notes/nyc_hourly_dir.md).

Frame (loader kalshi_nyc_dir, builder scripts/nyc_hourly_dir.py): one row per market. Inputs: K, H, T, t0 (= T - 6 min,
latest index minute treated as public), xhat (R(t0) + beta*slope*26 min, beta/residuals fit on station data 08-20..09-09),
p_yes (empirical residual CDF by local-hour bin + N(0, 0.3F) recon noise), yes_bid/yes_ask (last 1-min candle end <= T),
mark_yes_bid/ask (last candle end <= H - 5 min, CLV only). attrs['tape'] = prints in (T, H] on maker-selected tickers.

Rule A taker: side yes if p_yes >= .95, no if p_yes <= .05; a = side ask; bet iff p_side - a - fee(a) >= .03, spread
<= 10c; one bet per event (largest edge). Rule B maker: price a - 1c, fee 0, filled iff >= 10 contracts print <= price.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

P_MIN, MARGIN, N_MIN = 0.95, 0.03, 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _fee(a):
    return 0.07 * a * (1 - a)


def select(frame: pd.DataFrame, maker: bool = False) -> pd.DataFrame:
    p = frame["p_yes"].to_numpy(float)
    yb, ya = frame["yes_bid"].to_numpy(float), frame["yes_ask"].to_numpy(float)
    ok = np.isfinite(p) & np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya) & (ya - yb <= 0.10 + 1e-9)
    ok &= frame["strike_type"].to_numpy() == "greater"
    side = np.where(ok & (p >= P_MIN), "yes", np.where(ok & (p <= 1 - P_MIN), "no", ""))
    ps = np.where(side == "yes", p, 1 - p)
    a = np.where(side == "yes", ya, 1 - yb)
    price = np.round(a - 0.01, 4) if maker else a
    edge = ps - price - (0 if maker else _fee(price))
    keep = (side != "") & (edge >= MARGIN - 1e-9) & (price > 0.0)
    out = frame[keep].copy()
    out["side"], out["price"], out["edge"] = side[keep], price[keep], edge[keep]
    return out.sort_values("edge", ascending=False).drop_duplicates("event_ticker").sort_values("H")


def _bets(s, price_source, fill_t=None):
    mb, ma = s["mark_yes_bid"].to_numpy(float), s["mark_yes_ask"].to_numpy(float)
    T = s["T"].to_numpy("int64")
    return pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(s["H"]),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(s["t0"]),
        "side": s["side"].to_numpy(),
        "stake": 1.0,
        "price": s["price"].to_numpy(float),
        "price_source": price_source,
        "price_time": _ts(T if fill_t is None else np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(s["H"]),
        "close_price": np.where(s["side"].to_numpy() == "no", 1 - mb, ma),
    })


def taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(select(frame), "taker").reset_index(drop=True)


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    s = select(frame, maker=True)
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, sd, q, T, H) in enumerate(zip(s["ticker"], s["side"], s["price"], s["T"], s["H"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price" if sd == "no" else "yes_price"].to_numpy(float)
        m = (ts > T) & (ts <= H) & (px <= q + 1e-9)
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= N_MIN)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    b = _bets(s, "maker_tape", fill_t)
    return b[fill_t >= 0].reset_index(drop=True)
