"""fed_vs_futures: fetch Kalshi Fed-decision markets + hourly candles (dev meetings only) and ZQ=F daily. Read-only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget  # noqa: E402
import pandas as pd  # noqa: E402

R = Path(__file__).resolve().parent.parent / "results" / "fed_vs_futures"
R.mkdir(parents=True, exist_ok=True)
ev_file = R / "events.json"
if not ev_file.exists():
    r = kget("/events", {"series_ticker": "KXFEDDECISION", "limit": 200, "status": "settled"})
    json.dump([e["event_ticker"] for e in r["events"]], open(ev_file, "w"))
DEV = [e for e in json.load(open(ev_file)) if "-26" not in e]  # 2026 meetings sealed


def markets(ev):
    ms = (kget("/historical/markets", {"event_ticker": ev}) or {}).get("markets", [])
    if not ms:
        ms = (kget("/markets", {"event_ticker": ev}) or {}).get("markets", [])
    return ms


out_m, cache = [], R / "candles.json"
cd = json.loads(cache.read_text()) if cache.exists() else {}
for ev in DEV:
    for m in markets(ev):
        out_m.append({k: m.get(k) for k in ["ticker", "event_ticker", "result", "open_time", "close_time", "yes_sub_title", "volume"]})
        tk = m["ticker"]
        if tk in cd:
            continue
        t1 = int(pd.Timestamp(m["close_time"]).timestamp())
        p = {"start_ts": t1 - 20 * 86400, "end_ts": t1, "period_interval": 60}
        cs = (kget(f"/historical/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
        if not cs:
            cs = (kget(f"/series/{tk.split('-')[0]}/markets/{tk}/candlesticks", p) or {}).get("candlesticks", [])
        cd[tk] = cs
        print(tk, len(cs), flush=True)
cache.write_text(json.dumps(cd))
json.dump(out_m, open(R / "markets.json", "w"), indent=1)

import yfinance as yf  # noqa: E402

z = yf.download("ZQ=F", start="2023-01-01", end="2026-01-01", progress=False, auto_adjust=False)
z.columns = [c[0] if isinstance(c, tuple) else c for c in z.columns]
z.to_csv(R / "zq_front.csv")
print("markets", len(out_m), "zq rows", len(z))
