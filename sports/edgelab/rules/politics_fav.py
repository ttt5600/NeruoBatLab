"""L22 Kalshi politics favourites (scripts/pol_favourites.py builds the frame; prereg in notes/pol_favourites.md,
written 2026-11-15 before any price or outcome was loaded).

Frame: one row per settled Kalshi market (category Politics/Elections/World) x trial (T7/T14/T3: t = S - d,
S = expected_expiration_time), market open at t, quote = last daily candle ending in (t - 36 h, t].
Rule (category Politics or Elections only): two-sided quote (yes_bid >= .01, yes_ask <= .99, ask > bid);
favourite = side with mid > .5; buy it at its ask (YES: yes_ask, NO: 1 - yes_bid) when that ask is in [.60, .85],
taker (fee by the evaluator), hold to settlement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LO, HI = 0.60, 0.85
PRIMARY_CATS = ("Politics", "Elections")


def bets_frame(frame: pd.DataFrame, trial: str, cats=PRIMARY_CATS) -> pd.DataFrame:
    """Shared selection (the script imports this too, so both paths pick identical bets)."""
    f = frame[(frame["trial"] == trial) & frame["category"].isin(cats) & (frame["yes_bid"] >= 0.01)
              & (frame["yes_ask"] <= 0.99) & (frame["yes_ask"] > frame["yes_bid"])].copy()
    mid = (f["yes_bid"] + f["yes_ask"]) / 2
    f["side"] = np.where(mid > 0.5, "yes", "no")
    f["price"] = np.where(f["side"] == "yes", f["yes_ask"], 1 - f["yes_bid"]).round(4)
    f = f[(mid != 0.5) & (f["price"] >= LO) & (f["price"] <= HI)]
    return f


def _rule(frame: pd.DataFrame, trial: str) -> pd.DataFrame:
    f = bets_frame(frame, trial)
    t = pd.to_datetime(f["t"], utc=True)
    q = pd.to_datetime(f["q_end"].astype(float), unit="s", utc=True)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|" + trial,
        "cluster": f["event_ticker"].astype(str).to_numpy(),
        "event_start": pd.to_datetime(f["expected_expiration_time"], utc=True).to_numpy(),
        "decided_at": t.to_numpy(), "inputs_as_of": q.to_numpy(),
        "price": f["price"].to_numpy(), "price_source": "taker", "price_time": q.to_numpy(),
        "side": f["side"].to_numpy(), "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def t7(frame, params=None):
    return _rule(frame, "T7")


def t14(frame, params=None):
    return _rule(frame, "T14")


def t3(frame, params=None):
    return _rule(frame, "T3")
