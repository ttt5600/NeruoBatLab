"""Polymarket time-to-resolution favourite trials (user idea 11-06: "EV within the timeframe of the Polymarket system").

Descriptive basis (scripts/pm_timing.py describe, DEV only, notes/pm_timing.md): at the display mid, the favourite side
earns more per trade the further the market is from its scheduled resolution (fav return +1.7% at S-1h, +3.0% at S-1d,
+4.9% at S-7d, +5.0% at S-30d; event-cluster CIs at 1h and 7d do not overlap). Mid returns ignore the spread; these
trials ask whether a MAKER bid at the display price keeps any of it after adverse selection.

Frame (edgelab.loaders.pm_timing, built by scripts/pm_timing.py frame): one row per (binary market, horizon) with
  cat (smallmarket pm_category, frozen 10-27), S = min(endDate, gameStartTime), h in {7d, 30d}, t = S - h,
  p0 = last CLOB display price of token0 at or before t (within 6 h) and its time p0_time, closed_time,
  p_close = token0 display at S - 1 h if the market was still open then (CLV mark), won0 (hidden) = token0 resolved 1,
  attrs["trades"] = {ticker: DataFrame[timestamp, price, tok0 (bool: print on token0)]} = every public taker print
  (data-api /v2/trades) after t.

Rule (fixed 2026-11-06 before any trade tape was pulled):
  eligible iff h == H, cat NOT in {mentions, culture_tech} (another agent's sealed H2a test owns those), t < closed_time,
  favourite = the token whose display price > 0.5; fav price f = max(p0, 1 - p0) in [0.65, 0.95].
  Post a resting bid on the favourite at q = f rounded DOWN to the cent; no cancel; rests until the market closes.
  FILL = first print strictly after t and <= closed_time trading THROUGH q: a favourite-token print at price < q, or an
  other-token print at price > 1 - q (complementary match). Prints AT q are ignored (conservative queue). Unfilled =
  no bet. Hold to resolution; 1 contract at q; maker fee 0. CLV close = favourite display at S - 1 h.
  bet_id = "<ticker>|<h>"; cluster = UTC date of S (event-day).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EXCL = ("mentions", "culture_tech")
F_LO, F_HI = 0.65, 0.95


def _floor_cent(p):
    return np.floor(np.round(np.asarray(p, float) * 100, 6)) / 100


def candidates(frame: pd.DataFrame, h: str) -> pd.DataFrame:
    t = pd.to_datetime(frame["t"], utc=True)
    closed = pd.to_datetime(frame["closed_time"], utc=True)
    p0 = frame["p0"].astype(float)
    fav = np.maximum(p0, 1 - p0)
    ok = ((frame["h"] == h) & ~frame["cat"].isin(EXCL) & p0.notna() & (t < closed)
          & (fav >= F_LO - 1e-9) & (fav <= F_HI + 1e-9) & (p0 != 0.5))
    f = frame[ok.to_numpy()].copy()
    f["fav0"] = f["p0"].astype(float) > 0.5          # True: the favourite is token0
    f["q"] = _floor_cent(np.maximum(f["p0"], 1 - f["p0"]))
    tr = frame.attrs.get("trades", {})
    fills = []
    for tk, ti, cl, q, fav0 in zip(f["ticker"], pd.to_datetime(f["t"], utc=True),
                                   pd.to_datetime(f["closed_time"], utc=True), f["q"], f["fav0"]):
        x = tr.get(tk)
        ft = pd.NaT
        if x is not None and len(x):
            ts = pd.to_datetime(x["timestamp"], unit="s", utc=True)
            px = x["price"].to_numpy(float)
            on_fav = x["tok0"].to_numpy(bool) == bool(fav0)
            hit = (ts > ti) & (ts <= cl) & np.where(on_fav, px < q - 1e-9, px > 1 - q + 1e-9)
            if hit.any():
                ft = ts[hit].min()
        fills.append(ft)
    f["fill_time"] = pd.to_datetime(pd.Series(fills, index=f.index), utc=True)
    return f


def _bets(frame: pd.DataFrame, h: str) -> pd.DataFrame:
    f = candidates(frame, h)
    f = f[f["fill_time"].notna()]
    S = pd.to_datetime(f["S"], utc=True)
    pc = f["p_close"].astype(float)
    return pd.DataFrame({
        "bet_id": (f["ticker"].astype(str) + "|" + h).to_numpy(),
        "cluster": S.dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": S.to_numpy(),
        "decided_at": pd.to_datetime(f["t"], utc=True).to_numpy(),
        "inputs_as_of": pd.to_datetime(f["p0_time"], utc=True).to_numpy(),
        "price": f["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": f["fill_time"].to_numpy(),
        "fill_deadline": pd.to_datetime(f["closed_time"], utc=True).to_numpy(),
        "side": np.where(f["fav0"], "tok0", "tok1"),
        "stake": 1.0,
        "contracts": 1.0,
        "close_price": np.where(f["fav0"], pc, 1 - pc),
    }).reset_index(drop=True)


def t1_fav7d_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(frame, "7d")


def t2_fav30d_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(frame, "30d")
