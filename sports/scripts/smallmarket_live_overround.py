"""Tradeability check for the Polymarket overround-leg pattern, on TODAY's live books (read-only, no orders).

For active negRisk events (>= 3 legs), S = sum of displayed YES prices (outcomePrices[0]); for legs with display price
in [0.20, 0.50) in events with S > 1.10, compare the displayed price with the real book (bestBid/bestAsk): what would
buying NO actually cost (1 - bestBid) versus the 1 - display the backtest assumed (minus a proxy half-spread)?
-> data/smallmarket/live_overround.parquet"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from smallmarket_calibration import GAMMA, OUT, get, pm_category


def main():
    rows = []
    for off in range(0, 3000, 100):
        r = get(f"{GAMMA}/events", {"closed": "false", "active": "true", "limit": 100, "offset": off}, cache=False)
        if not r:
            break
        for e in r:
            ms = [m for m in (e.get("markets") or []) if m.get("active") and not m.get("closed")]
            if len(ms) < 3 or not any(m.get("negRisk") for m in ms):
                continue
            tags = [t.get("label", "") for t in (e.get("tags") or [])]
            for m in ms:
                try:
                    disp = float(json.loads(m.get("outcomePrices") or "[]")[0])
                except Exception:
                    continue
                rows.append({"event": e["id"], "cat": pm_category(tags), "n": len(ms), "disp": disp,
                             "bid": float(m["bestBid"]) if m.get("bestBid") is not None else np.nan,
                             "ask": float(m["bestAsk"]) if m.get("bestAsk") is not None else np.nan,
                             "last": float(m["lastTradePrice"]) if m.get("lastTradePrice") is not None else np.nan,
                             "volume": float(m.get("volumeNum") or 0), "q": m.get("question")})
        if len(r) < 100:
            break
    d = pd.DataFrame(rows)
    d["S"] = d.groupby("event").disp.transform("sum")
    d["Sbid"] = d.groupby("event").bid.transform("sum")
    d.to_parquet(OUT / "live_overround.parquet", index=False)
    ev = d.drop_duplicates("event")
    print(f"live negRisk events {len(ev)}; S median {ev.S.median():.3f}; share S>1.10 {(ev.S > 1.10).mean():.3f}; "
          f"events with sum of bids > 1 (NO-basket arb before fees): {(ev.Sbid > 1).sum()}")
    x = d[(d.S > 1.10) & (d.disp >= 0.2) & (d.disp < 0.5)].copy()
    x["spread"] = x.ask - x.bid
    x["no_cost_real"] = 1 - x.bid
    x["no_cost_assumed"] = 1 - (x.disp - 0.035)
    print(f"target legs now: {len(x)} in {x.event.nunique()} events")
    print(x[["spread", "no_cost_real", "no_cost_assumed", "volume"]].describe(percentiles=[.25, .5, .75]).round(3).to_string())
    print("share of target legs whose real NO cost exceeds assumed by > 5c:",
          round(float((x.no_cost_real - x.no_cost_assumed > 0.05).mean()), 3))
    print(x.groupby("cat").size().to_dict())
    print(x.sort_values("volume", ascending=False)[["cat", "q", "disp", "bid", "ask", "S", "volume"]].head(15).to_string())


if __name__ == "__main__":
    main()
