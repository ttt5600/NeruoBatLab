"""PMW -- a skilled Polymarket wallet's large taker fill on a game winner -> buy the same side on Kalshi within 60 s.

Frame (edgelab.loaders.pm_wallet_signal, built by scripts/pm_wallet_signal.py events; pre-registration
notes/pm_wallet_signal.md, fixed 2026-11-18 before any split trade or price was pulled): one row per event = first
fill >= $500 by a wallet that is "skilled" on games dated before the split start (taker-only P&L held to resolution,
>= 20 games, >= $5k risk, two-sided share <= .5, ROI > 0 and z >= 2.5), per (game, direction) per 10 min, fill
time <= Kalshi close - 120 s. Columns: bet_key, ticker (Kalshi team market), date, t (PM block time, epoch s), bull
(toward the Kalshi team), kclose_s, candles (JSON list of Kalshi 1-min candles {t=end, bid, ask}, yes side).

Rule: m0 = YES mid at the last candle ending <= t, must be in [0.05, 0.95]; entry = first candle ending in (t, t+60]
with both quotes; bull -> buy YES at ask, bear -> buy NO at 1 - bid; price in [0.03, 0.97]; $1-ish 1-contract taker
order, Kalshi taker fee; hold to settlement. close_price = side mid at the last candle ending <=
min(t + 120 min, kclose - 60 s). decided_at = price_time = entry candle end; inputs_as_of = t.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

COLS = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
        "side", "stake", "close_price"]


def mid_at(cd: pd.DataFrame, ts: float) -> float:
    x = cd[cd.t <= ts].dropna(subset=["bid", "ask"])
    if not len(x):
        return np.nan
    return float((x.bid.iloc[-1] + x.ask.iloc[-1]) / 2)


def bet_rows(frame: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in frame.itertuples():
        cd = pd.DataFrame(json.loads(r.candles))
        if not len(cd):
            continue
        sgn = 1 if r.bull else -1
        m0 = mid_at(cd, r.t)
        if not (0.05 <= m0 <= 0.95):
            continue
        q = cd[cd.t > r.t].dropna(subset=["bid", "ask"])
        if not len(q) or q.t.iloc[0] > r.t + 60:
            continue
        qe = q.iloc[0]
        ask = float(qe.ask) if r.bull else float(1 - qe.bid)
        if not (0.03 <= ask <= 0.97):
            continue
        mc = mid_at(cd, min(r.t + 7200, r.kclose_s - 60))
        path = {f"d{k}": sgn * (mid_at(cd, r.t + 60 * k) - m0) for k in (-30, -5, -1, 1, 5, 30, 120)}
        out.append({"bet_key": r.bet_key, "ticker": r.ticker, "date": r.date, "t": int(r.t), "t_entry": int(qe.t),
                    "side": "yes" if r.bull else "no", "price": ask, "m0": m0,
                    "d_entry": sgn * (float(qe.bid + qe.ask) / 2 - m0),
                    "close_side_mid": (mc if r.bull else 1 - mc), "kclose_s": int(r.kclose_s), **path})
    return pd.DataFrame(out)


def t1(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    b = bet_rows(frame)
    if not len(b):
        return pd.DataFrame(columns=COLS)
    o = pd.DataFrame({
        "bet_id": b.ticker + "|" + b.t.astype(str) + "|" + b.side,
        "cluster": pd.to_datetime(b.date).dt.strftime("%Y-%m-%d"),
        "event_start": pd.to_datetime(b.kclose_s, unit="s", utc=True),
        "decided_at": pd.to_datetime(b.t_entry, unit="s", utc=True),
        "inputs_as_of": pd.to_datetime(b.t, unit="s", utc=True),
        "price": b.price,
        "price_time": pd.to_datetime(b.t_entry, unit="s", utc=True),
        "side": b.side,
        "close_price": b.close_side_mid,
    })
    o["price_source"] = "taker"
    o["stake"] = 1.0
    return o.drop_duplicates("bet_id")
