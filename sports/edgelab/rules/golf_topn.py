"""Golf top-N 'including ties' (rules_quirks prereg G, notes/rules_quirks.md, written 2026-11-15 before prices).
Kalshi KXPGATOP5/10/20 pay the full $1 to every player tied inside the cutoff; books dead-heat (total payout = N).
Rule: at T_ref = last hourly candle ending <= 11:00 UTC on the first-round Thursday, buy YES at the ask (taker; fee by
the evaluator) on every market whose ask is in the band. One contract per market; cluster = tournament week.
"""
from __future__ import annotations

import pandas as pd


def _rule(frame: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    f = frame[frame["ask"].between(lo, hi)].copy()
    r1 = pd.to_datetime(f["R1"], utc=True)
    tref = r1 + pd.Timedelta(hours=11)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|yes",
        "cluster": r1.dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": tref.to_numpy(), "decided_at": tref.to_numpy(), "inputs_as_of": tref.to_numpy(),
        "price": f["ask"].round(4).to_numpy(), "price_source": "taker", "price_time": tref.to_numpy(),
        "side": "yes", "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def gt1_all(frame, params=None):
    return _rule(frame, 0.03, 0.95)


def gt2_mid(frame, params=None):
    return _rule(frame, 0.10, 0.60)
