"""E1X-T KXTSAW weekly TSA consensus-reject -- pricing of scripts/e1x_tsa.py cmd_score() (T1 taker, T2 maker), applied
to the rungs() frame the vault hands over (selection = the frozen rungs(): far side of rungs outside
F * exp(r_min/max -+ .01) at open + 2 h). Outcome columns are hidden; the vault settles each bet.

Timestamps (UTC): decided_at = T (open + 2 h on the week's Monday); quotes = the last hourly candle ending <= T
(price_time = that candle's end); inputs_as_of = T - 1 h (daily TSA data through the previous day);
event_start = Tuesday 13:00Z of the week (first in-week daily TSA release) -- before it no outcome data is public;
maker fill = first tape trade after T that crosses our bid, fill_deadline = close_time.
close for CLV = our side's ASK on the last candle ending <= event_start.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

M_TAKER_MAX = 0.99


def _px(c, side):
    s = c.get(side, {})
    v = s.get("close_dollars", s.get("close"))
    return np.nan if v is None else float(v)


def _yp(t):
    v = float(t.get("yes_price_dollars", t.get("yes_price")))
    return v / 100 if v > 1.5 else v


def _quotes(frame, t_unix):
    cs = frame.attrs["candles"]
    yb, ya, te = [], [], []
    for tk, ts in zip(frame["ticker"], t_unix):
        c = [k for k in cs.get(tk, []) if k["end_period_ts"] <= ts]
        yb.append(_px(c[-1], "yes_bid") if c else np.nan)
        ya.append(_px(c[-1], "yes_ask") if c else np.nan)
        te.append(c[-1]["end_period_ts"] if c else np.nan)
    return np.array(yb), np.array(ya), np.array(te, float)


def _base(frame):
    T = pd.to_datetime(frame["T"], utc=True)
    wend = pd.to_datetime(frame["wend"]).dt.tz_localize("UTC")
    ev_start = wend - pd.Timedelta(days=5) + pd.Timedelta(hours=13)
    t_unix = (T - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)
    yb, ya, te = _quotes(frame, t_unix.to_numpy())
    es_unix = ((ev_start - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).to_numpy()
    cb, ca, _ = _quotes(frame, es_unix)
    side = frame["side"].to_numpy()
    close_ask = np.where(side == "no", 1 - cb, ca)
    b = pd.DataFrame({"bet_id": frame["ticker"].to_numpy(), "cluster": frame["event_ticker"].to_numpy(),
                      "event_start": ev_start.to_numpy(), "decided_at": T.to_numpy(),
                      "inputs_as_of": (T - pd.Timedelta(hours=1)).to_numpy(), "side": side, "stake": 1.0,
                      "contracts": 1.0,
                      "close_price": close_ask})
    b["price_time"] = pd.to_datetime(te, unit="s", utc=True)
    return b, yb, ya, T


def forward_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """scripts/e1x_tsa_logger.py paper log: `post` rows, later `fill` / `settle` update rows. Only filled posts are
    bets. As of 2026-11-02 the log holds posts only -> zero bets."""
    cols = ["bet_id", "cluster", "event_start", "decided_at", "inputs_as_of", "price", "price_source", "price_time",
            "side", "stake", "contracts", "fill_deadline"]
    if frame.empty or "kind" not in frame.columns or not (frame["kind"] == "fill").any():
        return pd.DataFrame(columns=cols)
    raise NotImplementedError("tsa_paper.jsonl now has fill rows: write the forward adapter (new rule = new trial)")


def t1_taker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if "kind" in frame.columns:
        return forward_rows(frame)
    b, yb, ya, _ = _base(frame)
    ask = np.where(b["side"] == "no", 1 - yb, ya)
    b["price"] = np.round(ask, 4)
    b["price_source"] = "taker"
    keep = ~np.isnan(yb) & (ask <= M_TAKER_MAX + 1e-9)
    return b[keep].reset_index(drop=True)


def t2_maker(frame: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if "kind" in frame.columns:
        return forward_rows(frame)
    b, yb, ya, T = _base(frame)
    side = b["side"].to_numpy()
    ok = ~np.isnan(yb) & ((ya - yb) >= 0.02 - 1e-9) & np.where(side == "no", ya <= 0.99 + 1e-9, yb >= 0.01 - 1e-9)
    q = np.round(np.where(side == "no", 1 - ya + 0.01, yb + 0.01), 2)
    tr = frame.attrs["trades"]
    fill_t = [pd.NaT] * len(frame)
    for i, (tk, s, t0) in enumerate(zip(frame["ticker"], side, T)):
        if not ok[i]:
            continue
        for k in sorted(tr.get(tk, []), key=lambda k: k["created_time"]):
            ct = pd.Timestamp(k["created_time"])
            if ct <= t0:
                continue
            if (s == "no" and k["taker_side"] == "yes" and _yp(k) >= 1 - q[i] - 1e-9) or \
               (s == "yes" and k["taker_side"] == "no" and _yp(k) <= q[i] + 1e-9):
                fill_t[i] = ct
                break
    b["price"] = q
    b["price_source"] = "maker_tape"
    b["price_time"] = pd.to_datetime(pd.Series(fill_t), utc=True)
    b["fill_deadline"] = pd.to_datetime(frame["close_time"], utc=True).to_numpy()
    filled = ok & b["price_time"].notna().to_numpy()
    return b[filled].reset_index(drop=True)
