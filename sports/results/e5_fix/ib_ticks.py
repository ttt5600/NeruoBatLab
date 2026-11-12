"""Read-only: pull IBKR HMDS TRADES ticks for FX pick contracts (side contract of each pick), clientId 93.

    python results/e5_fix/ib_ticks.py <in.csv with ticker,side,conid,event_date> <out.parquet>
Pages forward from (event_date - 2 d) 1000 ticks at a time. Resumable: skips tickers already in out."""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from ib_async import IB, Contract

src, out = sys.argv[1], Path(sys.argv[2])
picks = pd.read_csv(src)
done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["ticker", "side"])
have = set(zip(done.ticker, done.side))
ib = IB()
ib.connect("127.0.0.1", 4001, clientId=93, timeout=8, readonly=True)
rows = []
for r in picks.itertuples():
    if (r.ticker, r.side) in have:
        continue
    c = Contract(conId=int(r.conid), exchange="FORECASTX")
    c.includeExpired = True
    det = ib.reqContractDetails(c)
    if not det:
        print(r.ticker, r.side, "no details", flush=True)
        rows.append({"ticker": r.ticker, "side": r.side, "ts": None, "price": None, "size": None, "status": "nodetails"})
        continue
    con = det[0].contract
    # HMDS only searches a short window forward of startDateTime (a start 2 days early returned 0 ticks for a
    # contract that traded), so walk 12-hour windows from 00Z D to 12Z D+1.
    w0 = datetime.fromisoformat(r.event_date).replace(tzinfo=timezone.utc) - timedelta(days=0)
    n0 = len(rows)
    seen = set()
    for i in range(3):
        a, b = w0 + timedelta(hours=12 * i), w0 + timedelta(hours=12 * (i + 1))
        start = a
        for _ in range(10):
            t = ib.reqHistoricalTicks(con, startDateTime=start, endDateTime="", numberOfTicks=1000,
                                      whatToShow="TRADES", useRth=False)
            t = [x for x in t if a <= x.time < b]
            for x in t:
                k = (x.time, x.price, x.size)
                rows.append({"ticker": r.ticker, "side": r.side, "ts": x.time, "price": x.price, "size": x.size,
                             "status": "ok"})
                seen.add(k)
            if len(t) < 1000:
                break
            start = t[-1].time + timedelta(seconds=1)
    if len(rows) == n0:
        rows.append({"ticker": r.ticker, "side": r.side, "ts": None, "price": None, "size": None, "status": "noticks"})
    print(r.ticker, r.side, len(rows) - n0, flush=True)
ib.disconnect()
new = pd.DataFrame(rows)
allr = pd.concat([done, new]) if len(done) else new
allr["ts"] = pd.to_datetime(allr["ts"], utc=True)
allr.to_parquet(out)
print("saved", len(allr))
