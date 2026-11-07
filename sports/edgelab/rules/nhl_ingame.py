"""NHL IN-GAME LATE (forward-only): Poisson + empty-net model vs Kalshi KXNHLGAME live asks, last 15 min of regulation.

Model + logger: scripts/nhl_ingame_logger.py (launchd com.jonathanwang.quant-sports-nhl-ingame, 60 s ticks), model
scripts/kalshi_ingame_lowscore.py nhl_probs (2023-24 pbp prior; pregame strength from the Kalshi pre-puck-drop mid).
Frame (loader kalshi_nhl_ingame): one row per logged tick x team market (home / away). Columns used: gid, ticker, team,
t (unix s = decided_at), inputs_as_of, price_time (Kalshi fetch), event_start (game end as first seen FINAL), period,
period_type, intermission, minutes_left (regulation), p_team (model P this market's team wins incl. OT/SO), bid / ask
(this market's YES), recent_goal (score changed at a tick within the last 180 s -- the 3-min cooldown of the soccer
rules), jump (home mid moved >= 8c vs any of the previous 3 ticks within 5 min: a goal the feed may not show yet),
mark (this market's mid at the first live tick in [t+300, t+420], else [t+60, t+180]; CLV only, never a decision input).
Outcome (hidden `result` per ticker) is resolved by the vault.

T1 late_taker_edge3: period 3, REG, not intermission, 0 < minutes_left <= 15; quote filter bid >= 0.01, ask <= 0.99,
  price in [0.03, 0.97]; excluded rows: recent_goal or jump. Options per row: buy YES at ask (backs this team), buy NO at
  1 - bid (backs the other team). edge = p_side - price - 0.07 price (1 - price) >= 0.03. First trigger per
  (game, backed team) -- at most one entry per side per game; ties within a tick -> largest edge. Taker, 1 contract,
  hold to settlement (Kalshi settles incl. OT/SO). OT is excluded: the model has no OT clock (constant pregame share).
Threshold 0.03 was fixed from the DEV frame (Apr-Jun 2025, 74 games, last 15 min) before any forward row existed: the
model's Brier edge there is tiny (-0.002 at 0-5 min, -0.0008 at 5-15), |model - mid| median 0.02 / p90 0.062, so 0.03
net of the ask and fee keeps roughly the top 10% of disagreements (~0.4 bets/game -> ~100 forward bets in ~5-6 weeks).
Dev taker ROI at 0.03 was NEGATIVE (-15%, n 28): this is a prospective test of a weak Brier lead, not of a dev pass.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAKER = 0.07
EDGE = 0.03

COLS = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time", "side",
        "stake", "contracts", "close_price"]


def _ts(x):
    return pd.to_datetime(np.asarray(x, np.int64), unit="s", utc=True)


def t1_late_taker_edge3(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=COLS)
    f = frame[(frame["period"] == 3) & (frame["period_type"] == "REG") & ~frame["intermission"].astype(bool)
              & (frame["minutes_left"] > 0) & (frame["minutes_left"] <= 15) & frame["p_team"].notna()
              & ~frame["recent_goal"].astype(bool) & ~frame["jump"].astype(bool)
              & (frame["bid"] >= 0.01) & (frame["ask"] <= 0.99) & (frame["bid"] < frame["ask"])]
    out = []
    for side in ("yes", "no"):
        g = f.copy()
        if side == "yes":
            g["price"], g["p_side"], g["close_price"] = g["ask"], g["p_team"], g["mark"]
            g["backed"] = g["team"]
        else:
            g["price"], g["p_side"], g["close_price"] = 1 - g["bid"], 1 - g["p_team"], 1 - g["mark"]
            g["backed"] = np.where(g["team"] == "home", "away", "home")
        g["edge"] = g["p_side"] - g["price"] - TAKER * g["price"] * (1 - g["price"])
        g["bside"] = side
        out.append(g[(g["price"] >= 0.03) & (g["price"] <= 0.97) & (g["edge"] >= EDGE)])
    c = pd.concat(out, ignore_index=True)
    if c.empty:
        return pd.DataFrame(columns=COLS)
    c = c.sort_values(["gid", "backed", "t", "edge"], ascending=[True, True, True, False])
    c = c.drop_duplicates(["gid", "backed"])
    t = _ts(c["t"])
    return pd.DataFrame({
        "bet_id": c["ticker"].astype(str) + "|" + c["t"].astype(str) + "|" + c["bside"],
        "cluster": pd.to_datetime(c["date"]).dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": pd.to_datetime(c["event_start"], utc=True).to_numpy(), "decided_at": t,
        "inputs_as_of": pd.to_datetime(c["inputs_as_of"], utc=True).to_numpy(),
        "price": c["price"].round(4).to_numpy(), "price_source": "taker",
        "price_time": pd.to_datetime(c["price_time"], utc=True).to_numpy(), "side": c["bside"].to_numpy(),
        "stake": 1.0, "contracts": 1.0, "close_price": c["close_price"].to_numpy()}).reset_index(drop=True)
