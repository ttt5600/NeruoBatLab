"""HC1/HC2: Kalshi per-storm hurricane category (KXHURCAT/HURCAT, "storm reaches >= X mph") vs the NHC official
forecast (ATCF OFCL). Builder scripts/tropical_nhc.py; prereg notes/tropical_nhc.md (2026-11-16, before any candle,
NHC intensity or result was read).

Frame (loader kalshi_hurcat_nhc): one row per (ticker, decision) with
  trial 'HC1' -> one row per OFCL advisory t0: gap = H_kt - max OFCL Vmax over tau 0..120 h (input at r = t0 + 3 h)
  trial 'HC2' -> one row per ticker at r = t_last + 27 h, gap = H_kt - max OFCL tau-0 Vmax so far (no advisory for 24 h)
  d = decision (HC1 r + 2 h = t0 + 5 h; HC2 r + 2 h); yes_bid, yes_ask, q_end: last hourly candle ending <= d (<= 7 d old;
  Kalshi emits a candle only when the book/trades change, so the last one is the standing quote) [amended pre-data]
Rule: gap >= margin (HC1) / > 0 (HC2), yes_bid >= 0.01, yes_ask > yes_bid -> buy 1 NO at 1 - yes_bid (taker); first
qualifying row per ticker; hold to settlement.
"""
from __future__ import annotations

import pandas as pd


def _bets(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values(["ticker", "r"], kind="stable").groupby("ticker", sort=False).head(1)
    r = pd.to_datetime(f["r"].astype("int64"), unit="s", utc=True)
    q = pd.to_datetime(f["q_end"].astype("int64"), unit="s", utc=True)
    d = pd.to_datetime(f["d"].astype("int64"), unit="s", utc=True)
    return pd.DataFrame({
        "bet_id": f["ticker"].astype(str).to_numpy() + "|" + f["trial"].astype(str).to_numpy(),
        "cluster": f["event_ticker"].astype(str).to_numpy(),
        "event_start": pd.to_datetime(f["close_time"], utc=True, format="ISO8601").to_numpy(),
        "decided_at": d.to_numpy(), "inputs_as_of": r.to_numpy(),
        "price": (1 - f["yes_bid"]).round(4).to_numpy(), "price_source": "taker", "price_time": q.to_numpy(),
        "side": "no", "stake": 1.0, "contracts": 1.0}).reset_index(drop=True)


def _ok(f):
    return (f["yes_bid"] >= 0.01) & (f["yes_ask"] > f["yes_bid"]) & (f["yes_ask"] <= 1)


def hc1(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    m = float((params or {}).get("margin", 20))
    f = frame[(frame["trial"] == "HC1") & (frame["gap"] >= m)]
    return _bets(f[_ok(f)])


def hc2(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    f = frame[(frame["trial"] == "HC2") & (frame["gap"] > 0)]
    return _bets(f[_ok(f)])
