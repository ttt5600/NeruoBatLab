"""Read-only probe: does IBKR serve intraday trade history for ForecastEx contracts? clientId 93, readonly, no orders."""
import sys
from ib_async import IB, Contract

conids = [int(x) for x in sys.argv[1:]]
ib = IB()
ib.connect("127.0.0.1", 4001, clientId=93, timeout=8, readonly=True)
for cid in conids:
    det = ib.reqContractDetails(Contract(conId=cid, exchange="FORECASTX"))
    print(cid, "details:", len(det), det[0].contract if det else None)
    if not det:
        continue
    c = det[0].contract
    for what in ("TRADES", "MIDPOINT", "BID_ASK"):
        try:
            bars = ib.reqHistoricalData(c, endDateTime="", durationStr="3 D", barSizeSetting="5 mins",
                                        whatToShow=what, useRTH=False, formatDate=2, timeout=20)
            print(" ", what, len(bars), bars[:2], bars[-2:])
        except Exception as e:
            print(" ", what, "ERR", e)
    try:
        t = ib.reqHistoricalTicks(c, startDateTime="", endDateTime=ib.reqCurrentTime(), numberOfTicks=100,
                                  whatToShow="TRADES", useRth=False)
        print("  ticks", len(t), t[:3])
    except Exception as e:
        print("  ticks ERR", e)
ib.disconnect()
