"""Pull every market (live settled + historical tier) for the given Kalshi series into data/e1x/<S>_markets.parquet
and print an event-by-month summary. No prices are joined to outcomes here.
Usage: python scripts/e1x_markets.py KXBIGGESTQUAKE KXTSAW
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data" / "e1x"
D.mkdir(parents=True, exist_ok=True)


def pull(s: str) -> pd.DataFrame:
    ms = []
    for ep in ("/markets", "/historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            ms += [dict(m, _src=ep) for m in r.get("markets", [])]
            cur = r.get("cursor")
            if not cur or not r.get("markets"):
                break
    df = pd.DataFrame(ms).drop_duplicates("ticker")
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any():
            df[c] = df[c].astype(str)
    df.to_parquet(D / f"{s}_markets.parquet")
    return df


if __name__ == "__main__":
    for s in sys.argv[1:]:
        df = pull(s)
        df["close"] = pd.to_datetime(df.close_time, format="ISO8601")
        df["vol"] = pd.to_numeric(df.volume_fp, errors="coerce")
        ev = df.groupby("event_ticker").agg(close=("close", "min"), n=("ticker", "size"), vol=("vol", "sum"))
        print(s, len(df), "markets; events", len(ev))
        print(ev.groupby(ev.close.dt.tz_localize(None).dt.to_period("M")).agg(ev=("n", "size"), mk=("n", "sum"),
                                                                             vol=("vol", "sum")).to_string())
        print(df.status.value_counts().to_dict(), df._src.value_counts().to_dict())
        print("SECONDARY:", str(df.iloc[0].get("rules_secondary", ""))[:900])
