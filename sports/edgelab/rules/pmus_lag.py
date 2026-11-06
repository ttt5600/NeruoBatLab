"""PMUS-LAG -- after a Kalshi in-game jump, buy Polymarket US toward it at PM US's real (book-derived) ask.

Frame (edgelab.loaders.pmus_ingame_lag, built by scripts/pmus_lag.py; pre-registration notes/pmus_lag.md):
one row per Kalshi jump on the PM-long team's ticker (>= 4c vs the last trade <= 10 s earlier, in-game, price
0.05-0.95, one per game per 60 s): slug, ticker, t_jump, p_jump, p_ref, sgn, k_close, date; attrs['pm'] = {slug:
DataFrame[ts, long, short]} = PM US price-history minute observations (long = YES ask, short = NO ask).

Rule (fixed 2026-11-06 before any split price was pulled):
  side = yes if sgn > 0 else no; obs = first PM US observation with t_jump + 3 <= ts <= t_jump + 10 (none -> no bet);
  ask = long (yes) / short (no) at obs; 0.03 <= ask <= 0.97; decided_at = price_time = obs ts.
  T1: bet every such jump.  T2: only if k_side - ask - 0.0695 ask (1 - ask) >= 0.02, k_side = p_jump (yes) / 1 - p_jump (no).
  $1 cost, hold to PM US settlement; fee schedule pmus_taker (spec). event_start = Kalshi market close (game end).
  close_price = side mid ((long + 1 - short)/2 for yes; 1 - that for no) at the first obs >= decided_at + 300 s.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LO, HI = 3, 10
CLV_S = 300
FEE = 0.0695
STALE = 0.02


def _bets(frame: pd.DataFrame, stale: bool) -> pd.DataFrame:
    pm = frame.attrs.get("pm", {})
    rows = []
    for r in frame.itertuples():
        h = pm.get(r.slug)
        if h is None or not len(h):
            continue
        ts = h["ts"].to_numpy(np.int64)
        k = np.flatnonzero((ts >= r.t_jump + LO) & (ts <= r.t_jump + HI))
        if not len(k):
            continue
        i = k[0]
        yes = r.sgn > 0
        lng, sht = float(h["long"].iloc[i]), float(h["short"].iloc[i])
        ask = lng if yes else sht
        if not (0.03 <= ask <= 0.97):
            continue
        if ts[i] >= r.k_close_ts:
            continue
        if stale:
            ks = r.p_jump if yes else 1 - r.p_jump
            if ks - ask - FEE * ask * (1 - ask) < STALE:
                continue
        c = np.flatnonzero(ts >= ts[i] + CLV_S)
        close = np.nan
        if len(c):
            mid = (float(h["long"].iloc[c[0]]) + 1 - float(h["short"].iloc[c[0]])) / 2
            close = mid if yes else 1 - mid
        rows.append({"bet_id": f"{r.slug}|{int(r.t_jump * 1000)}", "cluster": str(pd.Timestamp(r.date).date()),
                     "event_start": r.k_close_ts, "decided_at": float(ts[i]), "inputs_as_of": float(r.t_jump),
                     "price": ask, "price_time": float(ts[i]), "side": "yes" if yes else "no", "close_price": close})
    if not rows:
        return pd.DataFrame(columns=["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price",
                                     "price_source", "price_time", "side", "stake", "close_price"])
    b = pd.DataFrame(rows)
    for c in ("event_start", "decided_at", "inputs_as_of", "price_time"):
        b[c] = pd.to_datetime(b[c].astype(float), unit="s", utc=True)
    b["price_source"] = "taker"
    b["stake"] = 1.0
    return b


def t1_all(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(frame, stale=False)


def t2_stale(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(frame, stale=True)
