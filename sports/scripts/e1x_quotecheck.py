"""Liquidity check only (no outcomes, no forecasts): for a few events of a series, print every rung's yes_bid/yes_ask
from the hourly candle at a given hours-after-open, to see whether far rungs are two-sided at decision time.
Usage: python scripts/e1x_quotecheck.py KXTSAW 2 KXTSAW-26OCT25 KXTSAW-25OCT26 ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data" / "e1x"


def candles(series: str, m: pd.Series, start: int, end: int) -> list:
    p = {"start_ts": start, "end_ts": end, "period_interval": 60}
    if m["_src"] == "/historical/markets":
        r = get(f"/historical/markets/{m.ticker}/candlesticks", p)
    else:
        r = get(f"/series/{series}/markets/{m.ticker}/candlesticks", p)
    return r.get("candlesticks", [])


if __name__ == "__main__":
    s, hrs = sys.argv[1], float(sys.argv[2])
    df = pd.read_parquet(D / f"{s}_markets.parquet")
    for ev in sys.argv[3:]:
        sub = df[df.event_ticker == ev].sort_values("floor_strike")
        op = pd.to_datetime(sub.open_time, format="ISO8601").min()
        t = int((op + pd.Timedelta(hours=hrs)).timestamp())
        print("==", ev, "decision", pd.Timestamp(t, unit="s", tz="UTC"))
        for _, m in sub.iterrows():
            cs = [c for c in candles(s, m, t - 6 * 3600, t) if c["end_period_ts"] <= t]
            if not cs:
                print(f"  {m.ticker:34s} no candle")
                continue
            c = cs[-1]
            yb = c.get("yes_bid", {}).get("close_dollars", c.get("yes_bid", {}).get("close"))
            ya = c.get("yes_ask", {}).get("close_dollars", c.get("yes_ask", {}).get("close"))
            print(f"  {m.ticker:34s} K={m.floor_strike} yb={yb} ya={ya} vol={c.get('volume_fp', c.get('volume'))}")
