"""pm_full_census: compact gamma events json -> markets parquet (one row per market) + events parquet."""
import json, pandas as pd
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results/pm_full_census"
ev = json.load(open(OUT / "gamma_events_active.json"))
E, M = [], []
f = lambda x: float(x) if x not in (None, "") else None
for e in ev:
    tags = [t.get("label") or t.get("slug") for t in (e.get("tags") or [])]
    tslugs = [t.get("slug") for t in (e.get("tags") or [])]
    ser = (e.get("series") or [{}])[0].get("slug") if e.get("series") else None
    E.append(dict(eid=e["id"], eslug=e.get("slug"), title=e.get("title"), tags="|".join(t for t in tags if t),
                  tslugs="|".join(t for t in tslugs if t), series=ser, endDate=e.get("endDate"), negRisk=e.get("negRisk"),
                  vol=f(e.get("volume")), vol24=f(e.get("volume24hr")), vol1wk=f(e.get("volume1wk")),
                  vol1mo=f(e.get("volume1mo")), liq=f(e.get("liquidity")), oi=f(e.get("openInterest")),
                  n_mkts=len(e.get("markets") or []), restricted=e.get("restricted"), startDate=e.get("startDate")))
    for m in e.get("markets") or []:
        M.append(dict(eid=e["id"], mid=m.get("id"), mslug=m.get("slug"), q=m.get("question"),
                      active=m.get("active"), closed=m.get("closed"), acc=m.get("acceptingOrders"),
                      endDate=m.get("endDate"), gameStart=m.get("gameStartTime"), bid=f(m.get("bestBid")),
                      ask=f(m.get("bestAsk")), last=f(m.get("lastTradePrice")), vol=f(m.get("volumeNum") or m.get("volume")),
                      vol24=f(m.get("volume24hr")), vol1wk=f(m.get("volume1wk")), liq=f(m.get("liquidityNum") or m.get("liquidity")),
                      feesEnabled=m.get("feesEnabled"), feeType=str(m.get("feeType")), sportsMarketType=m.get("sportsMarketType"),
                      clob=m.get("clobTokenIds"), outcomePrices=m.get("outcomePrices"), cid=m.get("conditionId")))
pd.DataFrame(E).to_parquet(OUT / "gamma_events.parquet")
pd.DataFrame(M).to_parquet(OUT / "gamma_markets.parquet")
print(len(E), len(M))
