"""Lock-in (scripts/lock_in.py builds the frame): NBA markets whose YES outcome is already fixed by the public score.

Frame (loader lock_in_nba_total): one row per ticker. kind 'total' = KXNBATOTAL over rung K with hs + as >= K + 3.5 at
ESPN play wallclock input_ts (so one reversed 3 cannot undo it); kind 'ml' = the KXNBAGAME ticker of the team ahead on
the last ESPN play of the game (period >= 4, clock 0, not tied). T = input_ts + 60 s. yes_bid/yes_ask = last 1-min
candle ending <= T. attrs['tape'] = public trades (ticker, ts, yes_price, count) in (T, close] with yes_price <= 0.99.

Filter (both trials): yes_bid >= 0.95 at T, the market itself agrees the game is decided. This excludes rows where the
ESPN feed stopped before the real end (seen on 3 of 78 descriptive games: last play at clock 0, market still 50-80c).

T1 taker : yes_ask <= 0.99 -> buy YES at yes_ask, taker fee. Hold to settlement.
T2 maker : yes_ask > 0.99 -> post YES bid at 0.99 at T. Filled iff >= N_QUEUE contracts print at yes_price <= 0.99 in
           (T, close]. N_QUEUE = 5000 approximates the queue ahead of us: resting depth at 99c on pinned Kalshi sports
           markets sampled live 2026-11-09 was median 2k-120k contracts by series. Fee: 0.0175 p(1-p) (qwmf series).
Both frozen 2026-11-09 after a descriptive look at 20 + 78 dev games (prices only, no result column).
event_start = market close (trading after the horn is the whole point; the outcome is settled from close onward).
CLV close_price = last ask on the tape-window candles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N_QUEUE = 5000.0
MIN_BID = 0.95
Q = 0.99


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _base(s: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": pd.to_datetime(s["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(s["close_ts"]),
        "decided_at": _ts(s["T"]),
        "inputs_as_of": _ts(s["input_ts"]),
        "side": "yes",
        "stake": 1.0,
        "contracts": 1.0,
        "close_price": s["mark_ask"].to_numpy(float),
    })


def _ok(f: pd.DataFrame) -> pd.DataFrame:
    return f[f["yes_bid"].notna() & f["yes_ask"].notna() & (f["yes_bid"] >= MIN_BID) & (f["q_end"] > 0)].copy()


def t1_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    s = _ok(frame)
    s = s[s["yes_ask"] <= Q + 1e-9].reset_index(drop=True)
    b = _base(s)
    b["price"] = s["yes_ask"].to_numpy(float)
    b["price_source"] = "taker"
    b["price_time"] = _ts(s["q_end"])
    return b


def t2_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    n_q = float((params or {}).get("n_queue", N_QUEUE))
    s = _ok(frame)
    s = s[s["yes_ask"] > Q + 1e-9].reset_index(drop=True)
    tr = frame.attrs["tape"]
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")}
    keep, fill = [], []
    for i, (tk, T, close) in enumerate(zip(s["ticker"], s["T"], s["close_ts"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        m = (ts > T) & (ts <= close)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_q)[0]
        if len(hit):
            keep.append(i)
            fill.append(int(ts[m][hit[0]]))
    s = s.iloc[keep].reset_index(drop=True)
    b = _base(s)
    b["price"] = Q
    b["price_source"] = "maker_tape"
    b["price_time"] = _ts(np.asarray(fill, "int64"))
    b["fill_deadline"] = _ts(s["close_ts"])
    return b
