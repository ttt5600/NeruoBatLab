"""Fetch Hyperliquid hourly funding history (public info API) for coins -> parquet. Read-only."""
import sys, time, json, urllib.request
import pandas as pd
OUT = sys.argv[1]
for coin in sys.argv[2:]:
    rows, start = [], 1672531200000  # 2023-01-01
    while True:
        req = urllib.request.Request("https://api.hyperliquid.xyz/info", method="POST",
              data=json.dumps({"type": "fundingHistory", "coin": coin, "startTime": start}).encode(),
              headers={"Content-Type": "application/json"})
        b = json.loads(urllib.request.urlopen(req, timeout=30).read())
        if not b: break
        rows += b
        nxt = b[-1]["time"] + 1
        if nxt <= start or len(b) < 2: break
        start = nxt; time.sleep(1.5)
    df = pd.DataFrame(rows).drop_duplicates("time")
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float); df["premium"] = df["premium"].astype(float)
    df[["ts", "fundingRate", "premium"]].to_parquet(f"{OUT}/hl_funding_{coin}.parquet")
    print(coin, len(df), df.ts.min(), df.ts.max(), df.fundingRate.describe().round(7).to_dict())
