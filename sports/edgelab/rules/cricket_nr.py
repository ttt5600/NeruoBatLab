"""Cricket tie/NR/abandoned -> all markets $0.50 on Kalshi (books void). rules_quirks prereg C (notes/rules_quirks.md,
written 2026-11-15 before prices). Frame: scripts/rules_quirks_cricket_px.py (one row per market, quote = last hourly
candle ending <= scheduled start - 1 h). Rule: buy the UNDERDOG (lower mid of the two markets) YES at ask + fee when its
ask is in [0.10, 0.45]; one bet per event. NOTE: the evaluator cannot score a $0.50 settlement (it treats non-0/1
outcomes as a push), so the vault outcome kind is 'none' and the script's ROI is authoritative; registered so the
family bar counts these trials.
"""
from __future__ import annotations

import pandas as pd

C2 = ("KXT20MATCH", "KXODIMATCH")


def _rule(frame: pd.DataFrame, series) -> pd.DataFrame:
    f = frame[frame["ask"].notna()].copy()
    f = f[f.groupby("event_ticker")["ticker"].transform("size") == 2]
    f = f.loc[f.groupby("event_ticker")["mid"].idxmin()]
    f = f[f["ask"].between(0.10, 0.45) & f["series"].isin(series)]
    st = pd.to_datetime(f["start"], utc=True)
    tref = st - pd.Timedelta(hours=1)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str) + "|yes", "cluster": f["event_ticker"].astype(str).to_numpy(),
        "event_start": st.to_numpy(), "decided_at": tref.to_numpy(), "inputs_as_of": tref.to_numpy(),
        "price": f["ask"].round(4).to_numpy(), "price_source": "taker", "price_time": tref.to_numpy(),
        "side": "yes", "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def c1_dog(frame, params=None):
    return _rule(frame, ("KXODIMATCH", "KXT20MATCH", "KXWT20MATCH", "KXIPLGAME"))


def c2_dog_high_nr(frame, params=None):
    return _rule(frame, C2)
