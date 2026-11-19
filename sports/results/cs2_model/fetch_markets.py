"""Pull every settled Kalshi KXCS2GAME / KXCS2MAP market (historical + live endpoints). Read-only GETs via kx_throttle."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from kx_throttle import kget
OUT = Path(__file__).resolve().parent
for series in sys.argv[1:] or ["KXCS2GAME", "KXCS2MAP"]:
    rows = []
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur: p["cursor"] = cur
            d = kget(path, p) or {}
            ms = d.get("markets", [])
            for m in ms: m["_src"] = path
            rows += ms
            cur = d.get("cursor")
            print(series, path, len(rows), flush=True)
            if not cur or not ms: break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    for c in df.columns:
        if df[c].map(lambda v: isinstance(v, (dict, list))).any(): df[c] = df[c].map(json.dumps)
    df.to_parquet(OUT / f"markets_{series}.parquet", index=False)
    print(series, "markets", len(df), "events", df.event_ticker.nunique(), df.result.value_counts().to_dict())
