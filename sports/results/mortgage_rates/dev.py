"""MR1 taker / MR2 maker on KX30YMORTW (prereg notes/mortgage_rates.md). Read-only Kalshi GETs.

    python results/mortgage_rates/dev.py quotes  [dev|forward]   # hourly candles at T for every rung (no outcomes)
    python results/mortgage_rates/dev.py score   [dev|forward]   # joins tape + settlement, MR1/MR2 + unfilled
Splits by release date D: dev 2026-09-17..2026-10-29, sealed 2026-11-05..2026-11-12 (NOT touched), forward >= 2026-11-19.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent
sys.path.insert(0, str(R.parents[1] / "scripts"))
from kx_throttle import kget  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
SPLITS = {"dev": ("2026-09-17", "2026-10-29"), "forward": ("2026-11-19", "2099-01-01")}


def ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))   # traps: never astype(int64)//1e9


def px(c, k):
    d = c.get(k) or {}
    v = d.get("close_dollars", d.get("close"))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1 else v


def fee(p):
    return 0.07 * p * (1 - p)          # exact taker per contract (edgelab rounding: exact)


def events(split):
    a, b = SPLITS[split]
    nc = pd.read_csv(R / "nowcast_events.csv", parse_dates=["D"])
    return nc[(nc.D >= a) & (nc.D <= b)]


def markets(ev):
    return (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])


def quotes(split):
    rows = []
    for _, e in events(split).iterrows():
        D = e.D
        ev = f"KX30YMORTW-{D:%y%b%d}".upper()
        T = pd.Timestamp(D.date()).tz_localize("UTC") + pd.Timedelta(hours=14)
        ms = markets(ev)
        for m in ms:
            if pd.Timestamp(m["open_time"]) > T:
                continue
            j = kget(f"/series/KX30YMORTW/markets/{m['ticker']}/candlesticks",
                     {"start_ts": ts(T) - 6 * 3600, "end_ts": ts(T), "period_interval": 60}) or {}
            cs = [c for c in j.get("candlesticks", []) if c["end_period_ts"] <= ts(T)]
            c = cs[-1] if cs else {}
            rows.append(dict(event=ev, D=D.date(), T=T, ticker=m["ticker"], K=float(m["floor_strike"]),
                             close_time=m["close_time"], q_end=c.get("end_period_ts"),
                             yb=px(c, "yes_bid"), ya=px(c, "yes_ask"), mu=e.mu, lo=e.lo_yes_max, hi=e.hi_no_min))
    q = pd.DataFrame(rows)
    q.to_csv(R / f"quotes_{split}.csv", index=False)
    return q


def picks(q):
    q = q.copy()
    q["side"] = np.where(q.K >= q.hi - 1e-9, "no", np.where(q.K + 0.01 <= q.lo + 1e-9, "yes", ""))
    q = q[(q.side != "") & q.yb.notna() & q.ya.notna()].copy()   # no candle at/before T = no quote = no bet
    yb = q.yb.fillna(0.0)
    ya = q.ya.where(q.ya > 0, 1.0).fillna(1.0)
    q["s_ask"] = np.where(q.side == "yes", ya, 1 - yb)
    q["s_mid"] = np.where(q.side == "yes", (ya + yb) / 2, 1 - (ya + yb) / 2)
    q["q_maker"] = np.minimum(np.floor(np.round(q.s_mid * 100, 6)) / 100, q.s_ask - 0.01)
    q["mr1"] = (q.s_ask >= 0.50) & (q.s_ask <= 0.98)
    q["mr2_post"] = (q.q_maker >= 0.50) & (q.q_maker <= 0.98)
    return q


def trades(ticker, t0, t1):
    tr, cur = [], None
    while True:
        p = {"ticker": ticker, "min_ts": t0, "max_ts": t1, "limit": 1000}
        if cur:
            p["cursor"] = cur
        r = kget("/markets/trades", p) or {}
        tr += r.get("trades", [])
        cur = r.get("cursor")
        if not cur or not r.get("trades"):
            break
    return tr


def score(split):
    q = picks(pd.read_csv(R / f"quotes_{split}.csv"))
    out = []
    for _, r in q.iterrows():
        m = (kget(f"/markets/{r.ticker}") or {}).get("market", {})
        res = m.get("result")
        if res not in ("yes", "no"):
            continue
        tr = trades(r.ticker, ts(r["T"]), ts(m["close_time"]))
        q_ = r.q_maker
        if r.side == "no":     # NO bid at q filled by a taker buying YES at yes_price > 1 - q
            fill = any(t.get("taker_side") == "yes" and float(t.get("yes_price_dollars", t.get("yes_price", 0) / 100)) > 1 - q_ + 1e-9 for t in tr)
        else:                  # YES bid at q filled by a taker buying NO at yes_price < q
            fill = any(t.get("taker_side") == "no" and float(t.get("yes_price_dollars", t.get("yes_price", 0) / 100)) < q_ - 1e-9 for t in tr)
        won = res == r.side
        out.append(dict(r.to_dict(), result=res, won=won, mr2_fill=bool(fill and r.mr2_post),
                        pnl_mr1=(1.0 if won else 0.0) - r.s_ask - fee(r.s_ask) if r.mr1 else np.nan,
                        pnl_mr2=(1.0 if won else 0.0) - q_ if (fill and r.mr2_post) else np.nan,
                        n_tr=len(tr)))
    s = pd.DataFrame(out)
    s.to_csv(R / f"score_{split}.csv", index=False)
    return s


if __name__ == "__main__":
    cmd, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "dev")
    assert split in SPLITS, "sealed is not loadable here"
    if cmd == "quotes":
        q = quotes(split)
        p = picks(q)
        print(q.groupby("event").agg(n=("K", "size"), Kmin=("K", "min"), Kmax=("K", "max"), mu=("mu", "first"),
                                     lo=("lo", "first"), hi=("hi", "first")))
        print("far rungs", len(p), "MR1-eligible", int(p.mr1.sum()), "MR2-postable", int(p.mr2_post.sum()))
        print(p[["event", "K", "side", "yb", "ya", "s_ask", "q_maker", "mr1", "mr2_post"]].to_string())
    else:
        s = score(split)
        print(s[["event", "K", "side", "s_ask", "q_maker", "won", "mr1", "mr2_fill", "pnl_mr1", "pnl_mr2", "n_tr"]].to_string())
