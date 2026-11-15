"""Build the maker_fairfill bet universe (same bets as K50 primary / CBB T1) and fetch trade tapes (read-only).

    ../.venv_quant/bin/python results/maker_fairfill/fetch.py      (from sports/)
-> results/maker_fairfill/bets.csv, tapes/<ticker>.json  (all trades in (T - 1 h, start + 1 h], cached)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "results" / "kalshi_openers_xleague"))
from kx_throttle import kget  # noqa: E402
from kx import side_x  # noqa: E402

OUT = Path(__file__).resolve().parent
TAPES = OUT / "tapes"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def secs(t):
    return int((pd.Timestamp(t) - EPOCH) // pd.Timedelta(seconds=1))   # traps: no datetime64[us] // 1e9


def k50(lg, start_of):
    d = pd.read_csv(ROOT / f"results/k50_strike/rows_{lg}.csv")
    b = d[d.rule.astype(str).str.startswith("k50") & (d.primary == True)].copy()  # noqa: E712
    rows = []
    for r in b.itertuples():
        ev = r.ticker.rsplit("-", 1)[0]
        _, sg = side_x(ev, r.ticker, r.S if r.mk == "total" else abs(r.S), r.mk)
        yes = (r.pick == "up") == (sg > 0)
        rows.append(dict(lg=lg, mk=r.mk, key=r.key, date=r.date, T=r.T, start=start_of(r, ev), ticker=r.ticker,
                         pick=r.pick, yes_side=yes, side_ask=r.ask, spread_c=r.spread_c, taker_cost=r.cost,
                         won=bool(r.won), clv=r.clv))
    return rows


def main():
    q1n = pd.read_csv(ROOT / "results/nba_l32b_exec/q1_timing.csv").set_index("key")["start"]
    occ = {}
    for lg in ("cfb", "nfl"):
        q = pd.read_csv(ROOT / f"results/kalshi_openers_xleague/q1_{lg}_rows.csv")
        occ.update(dict(zip(q.event, q.occ)))
    rows = k50("nba", lambda r, ev: q1n.get(r.key))
    rows += k50("cfb", lambda r, ev: occ.get(ev))
    rows += k50("nfl", lambda r, ev: occ.get(ev))
    c = pd.read_csv(ROOT / "results/cbb_opener_model/kalshi_rows.csv")
    for r in c[c.bet == True].itertuples():  # noqa: E712
        yes = r.side == "over"
        rows.append(dict(lg="cbb", mk="total", key=r.event, date=pd.Timestamp(r.tip).tz_convert("America/New_York")
                         .strftime("%Y-%m-%d"), T=r.ft_t, start=r.tip, ticker=f"{r.event}-{math.ceil(r.S)}",
                         pick="up" if yes else "down", yes_side=yes, side_ask=r.px, spread_c=r.spread_c,
                         taker_cost=r.cost, won=bool(r.won), clv=r.clv))
    B = pd.DataFrame(rows)
    print(B.groupby("lg").size().to_dict(), "missing start:", B.start.isna().sum())
    B.to_csv(OUT / "bets.csv", index=False)
    TAPES.mkdir(exist_ok=True)
    for r in B.itertuples():
        f = TAPES / f"{r.ticker}.json"
        if f.exists() or pd.isna(r.start):
            continue
        p = {"ticker": r.ticker, "limit": 1000, "min_ts": secs(r.T) - 3600, "max_ts": secs(r.start) + 3600}
        tr, cur = [], None
        while True:
            if cur:
                p["cursor"] = cur
            d = kget("/markets/trades", p) or {}
            tr += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        f.write_text(json.dumps({"params": {k: v for k, v in p.items() if k != "cursor"}, "trades": tr}))
        print(r.lg, r.ticker, len(tr), flush=True)


if __name__ == "__main__":
    main()
