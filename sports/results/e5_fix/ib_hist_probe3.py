"""Read-only: TRADES ticks for a heavily traded FX contract by symbol (clientId 93)."""
from datetime import datetime, timedelta, timezone

from ib_async import IB, Contract

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=93, timeout=8, readonly=True)
for strike in (84, 82):
    for right in ("C", "P"):
        det = ib.reqContractDetails(Contract(symbol="UHLAX", secType="OPT", exchange="FORECASTX", currency="USD",
                                             lastTradeDateOrContractMonth="20261111", strike=strike, right=right))
        print(strike, right, "details", len(det), flush=True)
        if not det:
            continue
        c = det[0].contract
        t = ib.reqHistoricalTicks(c, startDateTime=datetime.now(timezone.utc) - timedelta(days=2), endDateTime="",
                                  numberOfTicks=1000, whatToShow="TRADES", useRth=False)
        print(c.localSymbol, "ticks", len(t), [(x.time.isoformat(), x.price, x.size) for x in t[:5]], flush=True)
        b = ib.reqHistoricalData(c, endDateTime="", durationStr="2 D", barSizeSetting="1 hour", whatToShow="TRADES",
                                 useRTH=False, formatDate=2, timeout=20)
        print("  bars", len(b), [(x.date, x.low, x.volume) for x in b[:5]], flush=True)
ib.disconnect()
