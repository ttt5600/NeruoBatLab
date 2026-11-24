"""wx_bot_flow: read-only Kalshi pull for orig-7 KXHIGH events D in [d0, d1]: market metadata, trades and hourly candles in
[14:00Z, 20:00Z) of D-1 (the 06Z/12Z GEFS arrival window). Public endpoints, no auth, no orders.

    ../.venv_quant/bin/python scripts/wx_bot_flow_tape.py 2026-10-08 2026-11-23
Outputs results/wx_bot_flow/{markets,trades,candles}.parquet
"""
import concurrent.futures as cf
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "results" / "wx_bot_flow"
API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHLAX", "KXHIGHDEN", "KXHIGHPHIL"]
S = requests.Session()


def quiet_window():
    """Pause around the live E4 executor runs (22:00 station-local = ~03/04/05/06:00Z) so this read-only pull never
    competes with the live bot for API rate budget."""
    while True:
        n = dt.datetime.now(dt.timezone.utc)
        if (n.minute >= 56 and n.hour in (2, 3, 4, 5)) or (n.minute < 14 and n.hour in (3, 4, 5, 6)):
            time.sleep(20); continue
        return


def get(path, params):
    for a in range(10):
        quiet_window()
        time.sleep(0.2)
        try:
            r = S.get(API + path, params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(5 + 5 * a); continue
        if r.status_code == 429:
            time.sleep(3 + 5 * a); continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"429 x6 {path} {params}")


def ev_ticker(series, D):
    return f"{series}-{D.strftime('%y%b%d').upper()}"


def pull_market(m, D):
    t0 = int(dt.datetime.combine(D - dt.timedelta(days=1), dt.time(14), tzinfo=dt.timezone.utc).timestamp())
    t1 = t0 + 6 * 3600
    trades, cur = [], None
    while True:
        p = {"ticker": m["ticker"], "limit": 1000, "min_ts": t0, "max_ts": t1}
        if cur: p["cursor"] = cur
        j = get("/markets/trades", p)
        trades += j.get("trades", [])
        cur = j.get("cursor")
        if not cur or not j.get("trades"):
            break
    c = get(f"/series/{m['series']}/markets/{m['ticker']}/candlesticks",
            {"start_ts": t0, "end_ts": t1, "period_interval": 60})
    cs = (c or {}).get("candlesticks", [])
    return trades, [dict(ticker=m["ticker"], end_ts=x["end_period_ts"],
                         yes_bid=float(x["yes_bid"].get("close_dollars") or "nan"),
                         yes_ask=float(x["yes_ask"].get("close_dollars") or "nan"),
                         vol=float(x.get("volume_fp") or 0)) for x in cs]


def main():
    d0, d1 = [dt.date.fromisoformat(x) for x in sys.argv[1:3]]
    mk = []
    D = d0
    while D <= d1:
        for s in SERIES:
            j = get("/markets", {"event_ticker": ev_ticker(s, D), "limit": 50})
            for m in (j or {}).get("markets", []):
                mk.append(dict(ticker=m["ticker"], event=m["event_ticker"], series=s, D=D.isoformat(),
                               strike_type=m.get("strike_type"), floor=m.get("floor_strike"), cap=m.get("cap_strike"),
                               open_time=m.get("open_time"), close_time=m.get("close_time"), result=m.get("result"),
                               expiration_value=m.get("expiration_value"), status=m.get("status")))
        D += dt.timedelta(days=1)
    mkd = pd.DataFrame(mk)
    mkd.to_parquet(OUT / "markets.parquet")
    print("markets", mkd.shape, mkd.groupby("series").size().to_dict(), flush=True)
    T, C = [], []
    ck = OUT / "tape_checkpoint.jsonl"
    done = {}
    if ck.exists():
        for ln in open(ck):
            x = json.loads(ln); done[x["ticker"]] = x
    todo = [r for r in mk if r["ticker"] not in done]
    print("checkpointed", len(done), "todo", len(todo), flush=True)
    for x in done.values():
        T += x["trades"]; C += x["candles"]
    with cf.ThreadPoolExecutor(2) as ex, open(ck, "a") as fck:
        futs = {ex.submit(pull_market, r, dt.date.fromisoformat(r["D"])): r["ticker"] for r in todo}
        for n, f in enumerate(cf.as_completed(futs)):
            tr, cs = f.result()
            T += tr; C += cs
            fck.write(json.dumps(dict(ticker=futs[f], trades=tr, candles=cs)) + "\n"); fck.flush()
            if n % 200 == 0:
                print(n, len(T), len(C), flush=True)
    td = pd.DataFrame([dict(ticker=x["ticker"], created_time=x["created_time"], count=float(x["count_fp"]),
                            yes_price=float(x["yes_price_dollars"]), taker_side=x["taker_side"], trade_id=x["trade_id"])
                       for x in T])
    td.to_parquet(OUT / "trades.parquet")
    pd.DataFrame(C).to_parquet(OUT / "candles.parquet")
    print("done trades", td.shape, "candles", len(C), flush=True)


if __name__ == "__main__":
    main()
