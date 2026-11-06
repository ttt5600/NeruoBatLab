"""SOCCER 1X2 IN-GAME: Poisson/Skellam time-decay model vs Kalshi live prices (scripts/kalshi_ingame_lowscore.py builds
the frame; model + prior documented in that script's docstring, all parameters fixed from 2023-24 ESPN data and the
Kalshi PRE-kickoff quote before any in-game outcome was read).

Frame (loader kalshi_soccer_ingame): one row per game x market (H / D / A) x 1-min candle end t, KO+60 s <= t <= FT-60 s.
Columns used: event_ticker, ticker, side (H/D/A), date, t (unix s), ft (ESPN end-regular-time wallclock = settlement
bound, NOT a model input), half (1 / 1.5 = break / 2), elapsed (real min in the half), p_model (state = goals/reds with
ESPN wallclock <= t), bid / ask (close of the candle ending at t), mid_t10_bid / mid_t10_ask (same market's quote on
the last candle ending <= min(t + 600, FT - 60) -- CLV mark only), event_recent (goal/red wallclock in (t-180, t]),
jump (any of the 3 mids moved >= 8c over the last 3 candles -> the feed may lag the market). Outcome hidden.

Sides: yes = buy YES at the ask; no = buy NO at 1 - bid. Quote filter: bid >= 0.01, ask <= 0.99, price in [0.03, 0.97].
Excluded rows (PIT, before any rule): event_recent or jump.

T1 taker_edge5 : any market, any in-play minute; p_side - price - 0.07 p(1-p) >= 0.05. First trigger per GAME (if
                 several markets/sides trigger in that minute, the largest edge), buy at the ask, hold to settlement.
T2 draw_late   : Tie market only, second half with elapsed >= 15 real minutes (match minute >= 60); edge >= 0.04, either
                 side (draw YES or NO). First trigger per game, taker, hold.
event_start = ft (outcome exists only at the final whistle; in-game state is the point of the bet).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAKER = 0.07


def _ts(x):
    return pd.to_datetime(np.asarray(x, np.int64), unit="s", utc=True)


def _candidates(frame: pd.DataFrame, edge_min: float) -> pd.DataFrame:
    f = frame[~frame["event_recent"].astype(bool) & ~frame["jump"].astype(bool)
              & (frame["bid"] >= 0.01) & (frame["ask"] <= 0.99)].copy()
    out = []
    for side in ("yes", "no"):
        g = f.copy()
        if side == "yes":
            g["price"] = g["ask"]
            g["p_side"] = g["p_model"]
            g["close_price"] = g["mid_t10_ask"]
        else:
            g["price"] = 1 - g["bid"]
            g["p_side"] = 1 - g["p_model"]
            g["close_price"] = 1 - g["mid_t10_bid"]
        g["edge"] = g["p_side"] - g["price"] - TAKER * g["price"] * (1 - g["price"])
        g["bside"] = side
        out.append(g[(g["price"] >= 0.03) & (g["price"] <= 0.97) & (g["edge"] >= edge_min)])
    return pd.concat(out, ignore_index=True)


def _bets(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return pd.DataFrame(columns=["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price",
                                     "price_source", "price_time", "side", "stake", "contracts", "close_price"])
    c = c.sort_values(["event_ticker", "t", "edge"], ascending=[True, True, False]).drop_duplicates("event_ticker")
    t = _ts(c["t"])
    return pd.DataFrame({
        "bet_id": c["ticker"].astype(str) + "|" + c["t"].astype(str) + "|" + c["bside"],
        "cluster": pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(c["ft"]), "decided_at": t, "inputs_as_of": t, "price": c["price"].round(4).to_numpy(),
        "price_source": "taker", "price_time": t, "side": c["bside"].to_numpy(), "stake": 1.0, "contracts": 1.0,
        "close_price": c["close_price"].to_numpy()}).reset_index(drop=True)


def t1_taker_edge5(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    return _bets(_candidates(frame, 0.05))


def t2_draw_late(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    f = frame[(frame["side"] == "D") & (frame["half"] == 2) & (frame["elapsed"] >= 15)]
    return _bets(_candidates(f, 0.04))
