"""Q1 fetch (read-only): ESPN games + book open/close lines, Kalshi settled ladder markets, hourly candles.
Usage: q1_fetch.py LEAGUE [LEAGUE..]   (nfl nhl cfb cbb). Cached per game in cache/. Outputs <lg>_games.parquet,
<lg>_kmarkets.parquet, <lg>_candles.parquet."""
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OUT))
from kx import SERIES, get  # noqa: E402
from sports import espn  # noqa: E402
from sports.http import fetch  # noqa: E402

espn.LEAGUES["cbb"] = "basketball/mens-college-basketball"
_core0 = espn._core
espn._core = lambda lg: (espn.CORE.format(sport="basketball", league="mens-college-basketball") if lg == "cbb" else _core0(lg))
import os  # noqa: E402
LO = date.fromisoformat(os.environ.get("KX_LO", "2026-10-01"))  # forward: KX_LO=2026-11-12 KX_HI=<yesterday> KX_TAG=fwd
HI = date.fromisoformat(os.environ.get("KX_HI", "2026-11-11"))
TAG = os.environ.get("KX_TAG", "")
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
CACHE = OUT / "cache"
CACHE.mkdir(exist_ok=True)
F = lambda v: float(v) if v not in (None, "") else np.nan  # noqa: E731


def ts(x):
    return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))


def espn_games(lg):
    rows, d = [], max(LO, date(2026, 11, 3)) if lg == "cbb" else LO
    while d <= HI:
        p = {"dates": d.strftime("%Y%m%d"), "limit": 400}
        if lg == "cfb":
            p["groups"] = 80
        if lg == "cbb":
            p["groups"] = 50
        for e in fetch(espn.SITE.format(path=espn.LEAGUES[lg]) + "/scoreboard", p, cache=True).get("events", []):
            rows.append(espn.parse_event(lg, e))
        d += timedelta(days=1)
    g = pd.DataFrame(rows).drop_duplicates("event_id")
    g = g[g.status == "STATUS_FINAL"]
    g["date_et"] = pd.to_datetime(g.start_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%y%b%d").str.upper()
    return g


def book(lg, eid):
    L = espn.lines(lg, str(eid), cache=True)
    if not L:
        return {}
    L = sorted(L, key=lambda r: (0 if "Draft" in r["provider"] else 1 if "ESPN" in r["provider"] else 2, r["provider"]))
    out = {}
    for r in L:
        if "tot" not in out and pd.notna(r.get("total_open")) and pd.notna(r.get("total_close")):
            out.update(tot=1, book_tot=r["provider"], total_open=r["total_open"], total_close=r["total_close"],
                       over_open=r["over_odds_open"], under_open=r["under_odds_open"],
                       over_close=r["over_odds_close"], under_close=r["under_odds_close"])
        if "spr" not in out and pd.notna(r.get("home_spread_open")) and pd.notna(r.get("home_spread_close")):
            out.update(spr=1, book_spr=r["provider"], home_spread_open=r["home_spread_open"],
                       home_spread_close=r["home_spread_close"])
    return out


def kalshi_settled(series):
    ms, cur = [], None
    while True:
        p = {"series_ticker": series, "status": "settled", "limit": 1000, "min_close_ts": ts("2026-10-01T00:00Z")}
        if cur:
            p["cursor"] = cur
        d = get("/markets", p)
        ms += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            break
    return ms


def main(lg):
    g = espn_games(lg)
    print(lg, "espn final games", len(g), flush=True)
    bk = []
    for eid in g.event_id:
        cf = CACHE / f"espn_{lg}_{eid}.json"
        if cf.exists():
            b = json.loads(cf.read_text())
        else:
            b = book(lg, eid)
            cf.write_text(json.dumps(b))
        bk.append(b)
    g = pd.concat([g.reset_index(drop=True), pd.DataFrame(bk)], axis=1)
    g["key"] = g.date_et + g.away.fillna("") + g.home.fillna("")
    km = []
    for (l2, mk), s in SERIES.items():
        if l2 != lg:
            continue
        for m in kalshi_settled(s):
            km.append({"mk": mk, "series": s, "event": m["event_ticker"], "ticker": m["ticker"],
                       "strike": F(m.get("floor_strike")), "open_time": m.get("open_time"),
                       "occ": m.get("occurrence_datetime"), "exp_value": m.get("expiration_value"),
                       "result": m.get("result")})
    K = pd.DataFrame(km)
    K["key"] = K.event.str.split("-").str[1]
    K = K[K.key.isin(set(g.key))]
    print(lg, "kalshi events matched", K.groupby("mk").event.nunique().to_dict(), "of espn", len(g), flush=True)
    g.to_parquet(OUT / f"{lg}{TAG}_games.parquet", index=False)
    K.to_parquet(OUT / f"{lg}{TAG}_kmarkets.parquet", index=False)
    # candles: strikes within +-7 of the book open (totals: total_open; spreads: |home line|), hourly, open -> occ
    gi = g.set_index("key")

    def one(ev, kk):
        cf = CACHE / f"c_{ev}.json"
        if cf.exists():
            return json.loads(cf.read_text())
        r = gi.loc[kk.key.iloc[0]]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        mk = kk.mk.iloc[0]
        ref = r.get("total_open") if mk == "total" else abs(r.get("home_spread_open", np.nan))
        if pd.isna(ref):
            cf.write_text("[]")
            return []
        s0 = ts(pd.to_datetime(kk.open_time).min()) - 3600
        e0 = ts(pd.to_datetime(kk.occ).max())
        win = 2.0 if lg == "nhl" else (4.0 if mk == "total" else 5.0)
        rows = []
        sel = kk[kk.strike.notna() & ((kk.strike - ref).abs() <= win)]
        strike = dict(zip(sel.ticker, sel.strike))
        tick = list(strike)
        for i in range(0, len(tick), 20):
            try:
                j = get("/markets/candlesticks", {"market_tickers": ",".join(tick[i:i + 20]), "start_ts": s0,
                                                  "end_ts": e0, "period_interval": 60})
            except Exception as e:  # noqa: BLE001
                print("candle err", ev, type(e).__name__, flush=True)
                continue
            for mm in j.get("markets", []):
                for c in mm.get("candlesticks", []):
                    yb, ya = c.get("yes_bid") or {}, c.get("yes_ask") or {}
                    rows.append({"event": ev, "ticker": mm["market_ticker"], "strike": strike[mm["market_ticker"]],
                                 "end": c["end_period_ts"],
                                 "bid": F(yb.get("close_dollars", yb.get("close"))),
                                 "ask": F(ya.get("close_dollars", ya.get("close"))),
                                 "vol": F(c.get("volume_fp") or c.get("volume") or 0)})
        cf.write_text(json.dumps(rows))
        return rows

    from concurrent.futures import ThreadPoolExecutor
    crows = []
    groups = list(K.groupby("event"))
    with ThreadPoolExecutor(4) as ex:
        for i, rows in enumerate(ex.map(lambda a: one(*a), groups)):
            crows += rows
            if (i + 1) % 25 == 0:
                print(lg, "events", i + 1, "/", len(groups), flush=True)
    C = pd.DataFrame(crows)
    for c in ("bid", "ask"):
        C.loc[C[c] > 1, c] = C.loc[C[c] > 1, c] / 100
    C.to_parquet(OUT / f"{lg}{TAG}_candles.parquet", index=False)
    print(lg, "candles", len(C), "events with candles", C.event.nunique() if len(C) else 0, flush=True)


if __name__ == "__main__":
    for lg in sys.argv[1:]:
        main(lg)
