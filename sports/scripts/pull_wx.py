"""Pull Kalshi daily-high markets + prices + forecasts for every city.

    python scripts/pull_wx.py [SERIES ...]

Writes data/wx_<CITY>.parquet. Cached and resumable; the archived per-bracket
price requests are the slow part (~9k per city).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sports import wx

DATA = Path(__file__).resolve().parents[1] / "data"


def own_result(df: pd.DataFrame) -> pd.Series:
    """Recompute each bracket's outcome from the official value and its strikes."""
    v = df["expiration_value"]
    return np.select(
        [df["strike_type"] == "greater", df["strike_type"] == "less", df["strike_type"] == "between"],
        [v > df["floor_strike"], v < df["cap_strike"], (v >= df["floor_strike"]) & (v <= df["cap_strike"])],
        default=np.nan)


def main() -> int:
    series = sys.argv[1:] or list(wx.CITIES)
    for s in series:
        t = time.time()
        df = wx.build(s)
        mine = own_result(df).astype(float)
        theirs = (df["result"] == "yes").astype(float)
        bad = (mine != theirs) & ~np.isnan(mine)
        df["result_mismatch"] = bad
        out = DATA / f"wx_{s.replace('KXHIGH', '')}.parquet"
        df.to_parquet(out, index=False)
        print(f"{s}: {len(df)} priced brackets over {df['event_ticker'].nunique()} days "
              f"({df['date'].min().date()} .. {df['date'].max().date()}), "
              f"forecast coverage {df['fc_max_lead1'].notna().mean():.0%}, "
              f"result mismatches {int(bad.sum())}, {time.time() - t:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
