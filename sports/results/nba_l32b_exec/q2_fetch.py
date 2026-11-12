"""Q2 fetch: for every L32b decision (live + backfill, pick or not) pull the KXNBATOTAL ladder (all strikes) and 1-min
candlesticks from 6 h before the L32b decision to tip. Read-only public GETs. Output: q2_markets.parquet, q2_candles.parquet."""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
API = "https://api.elections.kalshi.com/trade-api/v2"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def get(path, params):
    url = f"{API}{path}?{urllib.parse.urlencode(params)}"
    for i in range(8):
        try:
            time.sleep(0.35)
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "x"}), timeout=30) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            if i == 7:
                raise
            time.sleep(4 * (i + 1))


def ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def px(c, k, f="close"):
    d = c.get(k) or {}
    v = d.get(f + "_dollars", d.get(f))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1 else v


L = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_state.json"))
S = json.load(open(Path.home() / ".claude/quant-sports/data/l32b_state.json"))
mrows, crows = [], []
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
for k, r in [(k, r) for k, r in S.items() if r.get("pick")]:
    g = L[k]
    cf = CACHE / (k.replace("|", "_").replace("@", "_at_").replace(" ", "-") + ".json")
    if cf.exists():
        m0, c0 = json.loads(cf.read_text())
        mrows += m0
        crows += c0
        continue
    n_m, n_c = len(mrows), len(crows)
    ev = (g.get("kal_open") or {}).get("event")
    if not ev:
        print("no kalshi event", k)
        continue
    ms = get("/markets", {"event_ticker": ev, "limit": 200}).get("markets", [])
    s0 = min(ts(r["t"]), ts(g["pin_open"]["t"])) - 6 * 3600
    e0 = ts(g["start"])
    for m in ms:
        fs = m.get("floor_strike")
        mrows.append({"key": k, "event": ev, "ticker": m["ticker"], "strike": fs, "result": m.get("result"),
                      "open_time": m.get("open_time"), "status": m.get("status")})
        if fs is None or abs(float(fs) - g["pin_open"]["line"]) > 5.0:
            continue
        try:
            j = get(f"/series/KXNBATOTAL/markets/{m['ticker']}/candlesticks",
                    {"start_ts": s0, "end_ts": e0, "period_interval": 1})
        except Exception as e:  # noqa: BLE001
            print("candle err", m["ticker"], type(e).__name__)
            continue
        for c in j.get("candlesticks", []):
            crows.append({"key": k, "ticker": m["ticker"], "strike": float(fs), "end": c["end_period_ts"],
                          "bid": px(c, "yes_bid"), "ask": px(c, "yes_ask"),
                          "vol": float(c.get("volume_fp") or c.get("volume") or 0)})
    cf.write_text(json.dumps([mrows[n_m:], crows[n_c:]]))
    print(k, ev, len(ms), "markets", flush=True)
pd.DataFrame(mrows).to_parquet(OUT / "q2_markets.parquet", index=False)
pd.DataFrame(crows).to_parquet(OUT / "q2_candles.parquet", index=False)
print("candles", len(crows))
