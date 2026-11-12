"""Read-only: historical TRADES ticks for many FX conids (clientId 93, readonly). Prints contracts with any trades."""
import sys
from datetime import datetime, timedelta, timezone

from ib_async import IB, Contract

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=93, timeout=8, readonly=True)
end = datetime.now(timezone.utc)
for cid in [int(x) for x in sys.argv[1:]]:
    det = ib.reqContractDetails(Contract(conId=cid, exchange="FORECASTX"))
    if not det:
        print(cid, "no details")
        continue
    c = det[0].contract
    t = ib.reqHistoricalTicks(c, startDateTime=end - timedelta(days=2), endDateTime="", numberOfTicks=1000,
                              whatToShow="TRADES", useRth=False)
    print(c.localSymbol, "ticks", len(t), [(x.time.isoformat(), x.price, x.size) for x in t[:6]], flush=True)
ib.disconnect()
