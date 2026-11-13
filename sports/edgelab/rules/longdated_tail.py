"""LX2 long-dated tail-NO carry on Kalshi (scripts/lx2_tail_no_carry.py builds the frame; prereg in
notes/lx2_tail_no_carry.md, written 2026-11-12 before any price or outcome was loaded).

Frame: one row per settled Kalshi market x trial (A: t = S - 90 d, B: t = S - 180 d; S = expected_expiration_time),
market open and trading at t, quote = last daily candle ending in (t - 7 d, t].
Rule: yes_bid >= 0.01, yes_ask <= 0.10, yes_ask > yes_bid -> buy 1 NO at the NO ask = 1 - yes_bid (taker, fee by the
evaluator), hold to settlement. Interest on positions is NOT in the evaluator's ROI (reported by the script).
"""
from __future__ import annotations

import pandas as pd


def _rule(frame: pd.DataFrame, trial: str) -> pd.DataFrame:
    f = frame[(frame["trial"] == trial) & (frame["yes_bid"] >= 0.01) & (frame["yes_ask"] <= 0.10)
              & (frame["yes_ask"] > frame["yes_bid"])].copy()
    t = pd.to_datetime(f["t"], utc=True)
    q = pd.to_datetime(f["q_end"].astype(float), unit="s", utc=True)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|" + trial,
        "cluster": f["event_ticker"].astype(str).to_numpy(),
        "event_start": pd.to_datetime(f["expected_expiration_time"], utc=True).to_numpy(),
        "decided_at": t.to_numpy(), "inputs_as_of": q.to_numpy(),
        "price": (1 - f["yes_bid"]).round(4).to_numpy(), "price_source": "taker", "price_time": q.to_numpy(),
        "side": "no", "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def a_t90(frame, params=None):
    return _rule(frame, "A")


def b_t180(frame, params=None):
    return _rule(frame, "B")
