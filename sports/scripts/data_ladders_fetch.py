"""Data-ladder far-bracket study (PortWatch weekly chokepoints, EIA crude weekly): fetch Kalshi market lists
(live + historical tiers). Prereg: notes/data_ladders_far.md. Output data/dl/<SERIES>_markets.parquet."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

D = Path(__file__).resolve().parents[1] / "data" / "dl"
SERIES = ["KXHORMUZWEEKLY", "KXPANAMAWEEKLY", "KXSUEZWEEKLY", "KXBABELMANDEBWEEKLY", "KXEIACRUDEW"]


def pull(s: str) -> pd.DataFrame:
    rows = []
    for ep in ("/markets", "/historical/markets"):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get(ep, p)
            ms = r.get("markets", [])
            for m in ms:
                m["_src"] = ep
            rows += ms
            cur = r.get("cursor")
            if not cur or not ms:
                break
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates("ticker")
    return df


if __name__ == "__main__":
    D.mkdir(parents=True, exist_ok=True)
    for s in (sys.argv[1:] or SERIES):
        se = get(f"/series/{s}").get("series", {})
        (D / f"{s}_series.json").write_text(pd.Series(se, dtype=object).to_json())
        df = pull(s)
        df.to_parquet(D / f"{s}_markets.parquet")
        print(s, se.get("frequency"), se.get("settlement_sources"), "markets", len(df),
              "events", df.event_ticker.nunique() if len(df) else 0)
