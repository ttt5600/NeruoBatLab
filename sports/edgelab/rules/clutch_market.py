"""NBA in-game "clutch reputation" vs Kalshi KXNBAGAME (scripts/clutch_market.py builds the frame; prereg in
notes/clutch_market.md, written 2026-11-12 before outcomes were joined).

Frame: one row per game x cell (deficit 6-10/11-15/16+ x period 2-3 / period 4), the FIRST 1-min candle in that cell with
a two-sided trailer quote (spread <= 3c); state = last ESPN play with wallclock <= candle end - 60 s.
T_CB = trailer's share of wins in its prior-82-game sample of games where it trailed by >= 10 (dates strictly before).
Tercile cuts frozen from the dev snapshot trait distribution (no outcomes): bottom <= 0.2000 < middle <= 0.3243 < top.
Rules (taker at the ask, fee 0.07 p(1-p) by the evaluator, first qualifying snapshot per game in time order, hold):
  CM1 top tercile    -> buy trailer YES at t_ask      CM2 top tercile    -> buy leader YES at l_ask
  CM3 bottom tercile -> buy trailer YES at t_ask      CM4 bottom tercile -> buy leader YES at l_ask
Side = the team side ('home'/'away') whose YES is bought; outcome = result (winning side) == side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LO, HI = 0.2000, 0.3243


def _ts(x):
    return pd.to_datetime(np.asarray(x, float), unit="s", utc=True)


def _rule(frame: pd.DataFrame, tercile: str, buy: str) -> pd.DataFrame:
    f = frame.dropna(subset=["T_CB"]).copy()
    f.attrs = {}
    f = f[f["T_CB"] > HI] if tercile == "top" else f[f["T_CB"] <= LO]
    other = np.where(f["trailer_side"] == "home", "away", "home")
    if buy == "trailer":
        f["side"], f["px"] = f["trailer_side"].to_numpy(), f["t_ask"].to_numpy()
    else:
        f["side"], f["px"] = other, f["l_ask"].to_numpy()
    f = f[(f["px"] >= 0.03) & (f["px"] <= 0.97)]
    f = f.sort_values(["eid", "end"]).drop_duplicates("eid", keep="first")
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|" + f["end"].astype(int).astype(str) + "|" + f["side"],
        "cluster": pd.to_datetime(f["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": _ts(f["game_end"]), "decided_at": _ts(f["end"]), "inputs_as_of": _ts(f["state_wall"]),
        "price": f["px"].round(4).to_numpy(), "price_source": "taker", "price_time": _ts(f["end"]),
        "side": f["side"].to_numpy(), "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def cm1_top_buy(frame, params=None):
    return _rule(frame, "top", "trailer")


def cm2_top_fade(frame, params=None):
    return _rule(frame, "top", "leader")


def cm3_low_buy(frame, params=None):
    return _rule(frame, "bottom", "trailer")


def cm4_low_fade(frame, params=None):
    return _rule(frame, "bottom", "leader")
