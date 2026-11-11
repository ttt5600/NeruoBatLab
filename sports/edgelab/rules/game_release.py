"""Game-release slump (notes/game_release_slump.md): UNDER every in-season game in the window after a mega video-game
launch, at the CLOSING total. Frame: loader game_release_totals. Price = 1/dec_under (real SBR/lab close price for
NFL/NHL; -110 for NBA/CFB where SBR has no total price). SBR has no start times: event_start = game date 16:00 UTC
(earliest slate: CFB noon ET), decided_at = price_time = event_start - 1 min, inputs_as_of = 00:00 ET game date.
Bet at the close -> close_price = price (CLV 0 by construction). Cluster = release EVENT (all leagues together).
params.window: 7 (days 0..6) or 14 (days 0..13).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def unders(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    w = int((params or {}).get("window", 7))
    f = frame[frame["k"].between(0, w - 1)]
    d = pd.to_datetime(f["date"]).dt.normalize()
    start = d.dt.tz_localize("UTC") + pd.Timedelta(hours=16)
    et0 = d.dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    b = pd.DataFrame({"bet_id": (f["game_id"].astype(str) + "|" + f["sel"].astype(str)).to_numpy(),
                      "cluster": f["event"].astype(str).to_numpy(),
                      "event_start": start.to_numpy(), "decided_at": (start - pd.Timedelta(minutes=1)).to_numpy(),
                      "inputs_as_of": et0.to_numpy(), "price_time": (start - pd.Timedelta(minutes=1)).to_numpy(),
                      "price": np.round(1.0 / f["dec_under"].to_numpy(float), 6), "price_source": "book_quote",
                      "side": "under", "stake": 1.0})
    b["close_price"] = b["price"]
    return b.reset_index(drop=True)
