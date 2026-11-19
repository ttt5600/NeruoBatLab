"""Golf top-N 'including ties' v2 (rules_quirks prereg G + amendment G3, notes/rules_quirks.md).
Same as golf_topn.py but T_ref = last hourly candle ending <= 10:00 UTC on the first-round Thursday (v1's 11:00 UTC was
pit-rejected: in EDT months the first tee ~6:40 ET = 10:40 UTC). event_start = 10:40 UTC (earliest plausible tee).
Buy YES at ask in the band, taker, fee by the evaluator; cluster = tournament week.
"""
from __future__ import annotations

import pandas as pd


def _rule(frame: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    f = frame[frame["ask"].between(lo, hi)].copy()
    r1 = pd.to_datetime(f["R1"], utc=True)
    tref = r1 + pd.Timedelta(hours=10)
    st = r1 + pd.Timedelta(hours=10, minutes=40)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|yes",
        "cluster": r1.dt.strftime("%Y-%m-%d").to_numpy(),
        "event_start": st.to_numpy(), "decided_at": tref.to_numpy(), "inputs_as_of": tref.to_numpy(),
        "price": f["ask"].round(4).to_numpy(), "price_source": "taker", "price_time": tref.to_numpy(),
        "side": "yes", "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def gt1b_all(frame, params=None):
    return _rule(frame, 0.03, 0.95)


def gt2b_mid(frame, params=None):
    return _rule(frame, 0.10, 0.60)
