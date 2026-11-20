"""Quote at decision time from cached hourly candles: last candle with end_period_ts <= T. Placeholder books dropped."""
import json
from pathlib import Path
import pandas as pd
OUT = Path(__file__).resolve().parent
def f(x):
    try: return float(x)
    except (TypeError, ValueError): return None
def quote_at(cands, T):
    c = [x for x in cands if int(x["end_period_ts"]) <= T]
    if not c: return None
    x = max(c, key=lambda z: int(z["end_period_ts"]))
    def cl(k):
        o = x.get(k) or {}
        return f(o.get("close", o.get("close_dollars")))
    return {"end_ts": int(x["end_period_ts"]), "bid": cl("yes_bid"), "ask": cl("yes_ask"), "vol_last": f(x.get("volume", x.get("volume_fp")))}
def ufc_quotes():
    rows = []
    for l in open(OUT / "candles_ufc_dev.jsonl"):
        d = json.loads(l); q = quote_at(d["c"], d["T"])
        r = {"ticker": d["ticker"], "T": d["T"]}
        if q: r.update(q)
        rows.append(r)
    q = pd.DataFrame(rows)
    assert (q.end_ts.dropna() <= q["T"][q.end_ts.notna()]).all()
    q["age_min"] = (q["T"] - q.end_ts) / 60
    ok = q.bid.notna() & q.ask.notna() & (q.ask > q.bid) & (q.bid > 0) & (q.ask < 1)
    q["valid"] = ok
    q["mid"] = (q.bid + q.ask) / 2
    return q
