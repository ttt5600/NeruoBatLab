"""READ-ONLY weekday liquidity check on SETTLED KX*EVENT markets (notes/econ_event_ladders.md, census only).
Reads ONLY pre-release quotes: hourly candles with end_period_ts <= close - 60 min. Never reads result / post-release
prices (window end is cut before the request, so post-release candles are never fetched)."""
import json, sys, datetime as dt
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from kx_throttle import kget

OUT = Path(__file__).parent.parent / "results" / "econ_event_ladders"
d = json.loads((OUT / "census_raw.json").read_text())
rows = []
for s, v in d.items():
    done = [e for e in v["events"] if e.get("markets") and all(m.get("status") != "active" for m in e["markets"])]
    for e in done:
        for m in e["markets"]:
            close = dt.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
            opn = dt.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
            end = int((close - dt.timedelta(minutes=60)).timestamp())
            start = int(opn.timestamp())
            r = kget(f"/series/{s}/markets/{m['ticker']}/candlesticks",
                     {"start_ts": start, "end_ts": end, "period_interval": 60}) or {}
            cs = [c for c in r.get("candlesticks", []) if c["end_period_ts"] <= end]
            def px(c, side):
                x = (c.get(side) or {}).get("close_dollars")
                return float(x) if x not in (None, "") else None
            last = cs[-1] if cs else None
            # last pre-release candle, and the 'day before' candle (close - 24h) for staleness
            rows.append({"series": s, "event": e["event_ticker"], "ticker": m["ticker"], "strike": m.get("floor_strike"),
                         "close": m["close_time"], "n_candles": len(cs),
                         "bid_T": px(last, "yes_bid") if last else None, "ask_T": px(last, "yes_ask") if last else None,
                         "T_end": last["end_period_ts"] if last else None,
                         "vol_pre": sum(float(c.get("volume_fp") or c.get("volume") or 0) for c in cs)})
    print(s, len(rows), flush=True)
(OUT / "liquidity_pre_release.json").write_text(json.dumps(rows))
