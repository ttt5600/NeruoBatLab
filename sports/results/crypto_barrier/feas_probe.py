"""Outcome-free feasibility probe: Deribit DVOL history, Coinbase candles, Kalshi historical event listing (no results read)."""
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from kx_throttle import kget  # noqa: E402

now = int(time.time() * 1000)
for cur in ("BTC", "ETH", "SOL"):
    r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                     params={"currency": cur, "start_timestamp": now - 3 * 86400_000, "end_timestamp": now,
                             "resolution": "3600"}, timeout=20).json()
    d = r.get("result", {}).get("data", [])
    print("DVOL", cur, len(d), d[-1] if d else r.get("error"))
r = requests.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                 params={"currency": "BTC", "start_timestamp": 1704067200000, "end_timestamp": 1704067200000 + 5 * 86400_000,
                         "resolution": "3600"}, timeout=20).json()
print("DVOL 2024-01", len(r.get("result", {}).get("data", [])))
c = requests.get("https://api.exchange.coinbase.com/products/BTC-USD/candles",
                 params={"granularity": 60, "start": "2024-03-01T00:00:00Z", "end": "2024-03-01T04:00:00Z"}, timeout=20)
print("coinbase 1m", c.status_code, len(c.json()), c.json()[:1])
for s in ("KXBTCMAXM", "KXBTCMAXMON"):
    for path in ("/events", "/historical/events"):
        e = kget(path, {"series_ticker": s, "limit": 200}) or {}
        print(path, s, [x.get("event_ticker") for x in e.get("events", [])][:20])
m = kget("/historical/markets", {"event_ticker": "KXBTCMAXMON-BTC-26JAN31", "limit": 100}) or {}
print("hist markets 26JAN31", len(m.get("markets", [])), [x.get("ticker") for x in m.get("markets", [])])
m = kget("/markets", {"event_ticker": "KXBTCMAXMON-BTC-26JAN31", "limit": 100}) or {}
print("live markets 26JAN31", len(m.get("markets", [])))
