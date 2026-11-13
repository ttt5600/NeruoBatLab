"""NBA: L32b live decisions (t < 2026-11-11) whose game has no cached K1 candles -> KXNBATOTAL markets + 1-min candles
T-60 min..T for strikes within 5 pts of the Pinnacle open (batch endpoint, sequential, cached per game).
Read-only public GETs. Output: nba_markets.parquet, nba_candles.parquet (new games only)."""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parent / "kalshi_openers_xleague"))
from kx import get  # noqa: E402

EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
S = json.load(open(Path.home() / ".claude/quant-sports/data/l32b_state.json"))
L = json.load(open(Path.home() / ".claude/quant-kalshi/data/open_totals_state.json"))
have = set(pd.read_parquet(OUT.parent / "nba_l32b_exec/q2_candles.parquet").key)
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)


def ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def F(v):
    if v is None:
        return None
    v = float(v)
    return v / 100 if v > 1 else v


mrows, crows = [], []
todo = [r for r in S.values() if r["kind"] == "live" and pd.Timestamp(r["t"]) < pd.Timestamp("2026-11-11", tz="UTC")
        and r["key"] not in have]
print("games to fetch", len(todo), flush=True)
for r in todo:
    k = r["key"]
    cf = CACHE / (k.replace("|", "_").replace("@", "_at_").replace(" ", "-") + ".json")
    if cf.exists():
        m0, c0 = json.loads(cf.read_text())
    else:
        g = L[k]
        ev = g["kal_open"]["event"]
        ms = get("/markets", {"event_ticker": ev, "limit": 200}).get("markets", [])
        m0 = [{"key": k, "event": ev, "ticker": m["ticker"], "strike": m.get("floor_strike"), "result": m.get("result"),
               "status": m.get("status")} for m in ms]
        tk = {m["ticker"]: float(m["floor_strike"]) for m in ms
              if m.get("floor_strike") is not None and abs(float(m["floor_strike"]) - g["pin_open"]["line"]) <= 5.0}
        T = ts(r["t"])
        c0 = []
        tl = list(tk)
        for i in range(0, len(tl), 20):
            j = get("/markets/candlesticks", {"market_tickers": ",".join(tl[i:i + 20]), "start_ts": T - 3600,
                                              "end_ts": T, "period_interval": 1})
            for mm in j.get("markets", []):
                for c in mm.get("candlesticks", []):
                    yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
                    c0.append({"key": k, "ticker": mm["market_ticker"], "strike": tk[mm["market_ticker"]],
                               "end": c["end_period_ts"], "bid": F(yb.get("close_dollars", yb.get("close"))),
                               "ask": F(ya.get("close_dollars", ya.get("close")))})
        cf.write_text(json.dumps([m0, c0]))
        print(k, ev, len(ms), "mk", len(c0), "candles", flush=True)
    mrows += m0
    crows += c0
pd.DataFrame(mrows).to_parquet(OUT / "nba_markets.parquet", index=False)
pd.DataFrame(crows).to_parquet(OUT / "nba_candles.parquet", index=False)
print("done markets", len(mrows), "candles", len(crows))
