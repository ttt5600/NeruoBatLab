"""X1-X3 (ot1_extension): OT1's taker rule (edgelab/rules/opt_tails.py:taker, UNCHANGED) on other Kalshi crypto ladders.
Prereg notes/ot1_extension.md (2026-11-16, before any quote / outcome was read). Builder scripts/ot1_extension.py.

Frame (loader kalshi_opt_tails_ext): one row per (market, decision) with options-implied P(YES) <= .10 or >= .90 among
the amendment-2 candidates. Columns as kalshi_opt_tails plus `arm` (same_day | ahead4..ahead1), `cur`, `series`.
  x1_range_daily : series KXBTC / KXETH, arm same_day
  x2_weekly_ahead: weekly-listed BTC/ETH events, arms ahead4..ahead1; one bet per market = earliest firing arm
  x3_sol_daily   : series KXSOLD / KXSOLE, arm same_day
"""
from __future__ import annotations

import pandas as pd

from .opt_tails import taker as _ot1_taker

SEL = {
    "x1": (("KXBTC", "KXETH"), ("same_day",)),
    "x2": (("KXBTCD", "KXETHD", "KXBTC", "KXETH"), ("ahead4", "ahead3", "ahead2", "ahead1")),
    "x3": (("KXSOLD", "KXSOLE"), ("same_day",)),
}


def taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    ser, arms = SEL[params["arm_set"]]
    f = frame[frame["series"].isin(ser) & frame["arm"].isin(arms)].copy()
    f.attrs = {}
    if not len(f):
        return _ot1_taker(f, {"asset_class": "crypto"})
    out = _ot1_taker(f, {"asset_class": "crypto"})
    # earliest decision per market (x1/x3 have one decision per market already)
    out = out.sort_values("decided_at", kind="stable").drop_duplicates("bet_id", keep="first")
    return out.sort_values(["decided_at", "bet_id"], kind="stable").reset_index(drop=True)
