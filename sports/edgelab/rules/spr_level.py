"""SP1: Kalshi weekly Strategic Petroleum Reserve level (KXSPRLVL-<release date>) -- unreachable-rung maker on both
sides (builder scripts/phys_ladders.py spr; notes/phys_ladders.md).

Contract: "If the SPR value in the EIA Weekly Petroleum Status Report (Table 1, Current Week) for the week ending W is
above K million barrels, YES" (strike_type greater, 1-Mbbl rungs). Event opens ~1 h after the previous week's WPSR
release (Wed 14:30Z) and settles on the next release. SPR stock moves only by scheduled DOE sales / exchanges /
purchases at bounded pump rates, so a rung far from last week's level is mechanically out of reach.

Frame (loader kalshi_spr_level): one row per settled market. Inputs only:
  ticker, event_ticker, date (event-ticker release date = vault split column), K, strike_type, open_ts, close_ts
  T          decision = open_ts + 1 h
  rel_ts     scheduled release of the TARGET week = 14:30Z on the event-ticker date (outcome public from then)
  v0         EIA weekly SPR stock (WCSSTUS1, Mbbl) for the week ending W - 7 d (released before open_ts)
  dmax       max |weekly change| over the 104 weekly changes ending at week W - 7 d
  inputs_as_of = open_ts - 1 h (the W - 7 d value's release)
  yes_bid, yes_ask             last hourly candle ending <= T
  mark_yes_bid, mark_yes_ask   last hourly candle ending <= rel_ts - 1 h (CLV mark only)
attrs['tape'] = public trades in (T, rel_ts] for every ticker passing the input-only condition.

RULE (frozen 2026-11-06, before any EIA history, quote, tape or outcome was pulled):
  NO pick : K >= v0 + dmax + 1.0      (needs a rise 1 Mbbl beyond the largest weekly move of the past 2 years)
  YES pick: K <= v0 - dmax - 1.0      (a fall that large would be needed for the value to be <= K)
  quote filter: 0 <= yes_bid < yes_ask <= 1, spread <= 0.10.
  Maker on the picked side at q = min(floor_cent(side mid), side ask - 0.01), q in [0.50, 0.99]; post-only, 10
  contracts, never cancelled. Filled iff >= 10 contracts print at side price <= q in (T, rel_ts]. Unfilled -> no bet.
  Hold to settlement; maker fee 0 (fee_type quadratic). decided_at = T; event_start = fill_deadline = rel_ts;
  close_price (CLV) = same side's ask on the mark candle. cluster = event_ticker (all rungs share one print).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MARGIN = 1.0
N_MIN = 10.0


def _ts(x):
    return pd.to_datetime(np.asarray(x, "int64"), unit="s", utc=True)


def _cent_floor(x):
    return np.floor(np.round(np.asarray(x, float) * 100, 6)) / 100


def select(frame: pd.DataFrame, margin: float = MARGIN) -> pd.DataFrame:
    K, v0, dm = (frame[x].to_numpy(float) for x in ("K", "v0", "dmax"))
    yb, ya = frame["yes_bid"].to_numpy(float), frame["yes_ask"].to_numpy(float)
    ok = (frame["strike_type"].to_numpy() == "greater") & np.isfinite(v0) & np.isfinite(dm)
    ok &= np.isfinite(yb) & np.isfinite(ya) & (yb >= 0) & (ya <= 1) & (yb < ya) & (ya - yb <= 0.10 + 1e-9)
    side = np.where(ok & (K >= v0 + dm + margin - 1e-9), "no", np.where(ok & (K <= v0 - dm - margin + 1e-9), "yes", ""))
    mid_side = np.where(side == "no", 1 - (yb + ya) / 2, (yb + ya) / 2)
    ask_side = np.where(side == "no", 1 - yb, ya)
    q = np.round(np.minimum(_cent_floor(mid_side), ask_side - 0.01), 4)
    keep = (side != "") & (q >= 0.50 - 1e-9) & (q <= 0.99 + 1e-9)
    out = frame[keep].copy()
    out["side"] = side[keep]
    out["q"] = q[keep]
    return out


def maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    params = params or {}
    n_min = float(params.get("n_min", N_MIN))
    s = select(frame, float(params.get("margin", MARGIN)))
    tr = frame.attrs.get("tape")
    g = {k: v.sort_values("ts") for k, v in tr.groupby("ticker")} if tr is not None and len(tr) else {}
    T = s["T"].to_numpy("int64")
    R = s["rel_ts"].to_numpy("int64")
    fill_t = np.full(len(s), -1, "int64")
    for i, (tk, sd, qq) in enumerate(zip(s["ticker"], s["side"], s["q"])):
        d = g.get(tk)
        if d is None:
            continue
        ts = d["ts"].to_numpy("int64")
        px = d["no_price" if sd == "no" else "yes_price"].to_numpy(float)
        m = (ts > T[i]) & (ts <= R[i]) & (px <= qq + 1e-9)
        if not m.any():
            continue
        hit = np.nonzero(np.cumsum(d["count"].to_numpy(float)[m]) >= n_min)[0]
        if len(hit):
            fill_t[i] = ts[m][hit[0]]
    mb, ma = s["mark_yes_bid"].to_numpy(float), s["mark_yes_ask"].to_numpy(float)
    b = pd.DataFrame({
        "bet_id": s["ticker"].to_numpy(),
        "cluster": s["event_ticker"].to_numpy(),
        "event_start": _ts(R),
        "decided_at": _ts(T),
        "inputs_as_of": _ts(s["open_ts"].to_numpy("int64") - 3600),
        "side": s["side"].to_numpy(),
        "stake": 1.0,
        "price": s["q"].to_numpy(float),
        "price_source": "maker_tape",
        "price_time": _ts(np.where(fill_t >= 0, fill_t, T)),
        "fill_deadline": _ts(R),
        "close_price": np.where(s["side"].to_numpy() == "no", 1 - mb, ma),
    })
    return b[fill_t >= 0].reset_index(drop=True)
