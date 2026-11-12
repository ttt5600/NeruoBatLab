"""Read-only: why did ib_ticks.py get no ticks for UHDFW_111026_78 (81 traded per daily file)? clientId 94."""
from datetime import datetime, timedelta, timezone

from ib_async import IB, Contract

ib = IB()
ib.connect("127.0.0.1", 4001, clientId=94, timeout=8, readonly=True)
for sym, strike in (("UHDFW", 78), ("UHMSY", 78)):
    det = ib.reqContractDetails(Contract(symbol=sym, secType="OPT", exchange="FORECASTX", currency="USD",
                                         lastTradeDateOrContractMonth="20261111", strike=strike, right="P"))
    c = det[0].contract
    print(c.localSymbol, c.conId, flush=True)
    for lab, kw in (("start now-2d", dict(startDateTime=datetime.now(timezone.utc) - timedelta(days=2), endDateTime="")),
                    ("start 11-08", dict(startDateTime=datetime(2026, 11, 8, tzinfo=timezone.utc), endDateTime="")),
                    ("end now", dict(startDateTime="", endDateTime=datetime.now(timezone.utc)))):
        t = ib.reqHistoricalTicks(c, numberOfTicks=1000, whatToShow="TRADES", useRth=False, **kw)
        print(" ", lab, len(t), [(x.time.isoformat(), x.price, x.size) for x in t[:3]], flush=True)
ib.disconnect()
