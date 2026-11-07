"""Builders for the physical-ladder trials SX1 (SpaceX monthly launch count) and SP1 (weekly SPR level), 2026-11-06.

Rules + specs were registered BEFORE this script first ran (sx1_spacex_maker-099dcc58, sp1_spr_maker-f6ebe4f5;
commit 72369aa). Read-only public APIs, no orders.

    python scripts/phys_ladders.py spacex    # LL2 launches + KXSPACEXCOUNT monthly markets, candles, tape -> frame
    python scripts/phys_ladders.py spr       # EIA WCSSTUS1 + KXSPRLVL markets, candles, tape -> frame
    python scripts/phys_ladders.py audit     # settlement audit: our count / EIA value vs Kalshi expiration_value

Output data/phys_ladders/ (gitignored): {spacex,spr}_frame.parquet + _tape.parquet, raw caches.
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "data" / "phys_ladders"
H = {"User-Agent": "Mozilla/5.0"}
ET = ZoneInfo("America/New_York")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV",
                                   "DEC"], 1)}
sys.path.insert(0, str(ROOT))
from edgelab.rules import spacex_count as SX  # noqa: E402
from edgelab.rules import spr_level as SP  # noqa: E402


def _get(url, params=None):
    for i in range(6):
        try:
            r = requests.get(url, params=params, headers=H, timeout=60)
        except requests.RequestException:
            time.sleep(2 + 2 * i)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(1 + 2 * i)
            continue
        if r.status_code == 404:
            return {}
        return r.json()
    return {}


def _iso(s):
    return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())


def markets(series):
    rows = []
    for path, st in (("/historical/markets", None), ("/markets", "settled")):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if st:
                p["status"] = st
            if cur:
                p["cursor"] = cur
            d = _get(API + path, p)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"])].copy()
    vol = df["volume_fp"] if "volume_fp" in df else df.get("volume")
    return pd.DataFrame({
        "ticker": df["ticker"], "event_ticker": df["event_ticker"], "strike_type": df["strike_type"],
        "K": pd.to_numeric(df["floor_strike"], errors="coerce"), "open_ts": df["open_time"].map(_iso),
        "close_ts": df["close_time"].map(_iso), "rules": df["rules_primary"],
        "expiration_value": pd.to_numeric(df["expiration_value"], errors="coerce"),
        "result": df["result"], "volume": pd.to_numeric(vol, errors="coerce")}).reset_index(drop=True)


def _px(c, side):
    v = c.get(side) or {}
    return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")


def candles(series, ev, tickers, t0, t1):
    """hourly candles per ticker; event endpoint first, per-market historical endpoint as fallback."""
    out = {}
    s = t0
    while s < t1:  # chunk: the API caps candles per call
        e = min(t1, s + 4000 * 3600)
        d = _get(f"{API}/series/{series}/events/{ev}/candlesticks", {"start_ts": s, "end_ts": e, "period_interval": 60})
        for tk, cs in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
            out.setdefault(tk, []).extend(cs or [])
        s = e
    for tk in tickers:
        if not out.get(tk):
            d = _get(f"{API}/historical/markets/{tk}/candlesticks", {"start_ts": t0, "end_ts": t1, "period_interval": 60})
            out[tk] = d.get("candlesticks", [])
    return {k: sorted(v, key=lambda c: c["end_period_ts"]) for k, v in out.items()}


def quote(cs, t):
    c = [x for x in cs if x["end_period_ts"] <= t]
    return (_px(c[-1], "yes_bid"), _px(c[-1], "yes_ask")) if c else (np.nan, np.nan)


def tape(tk, t0, t1):
    out = []
    for path in ("/historical/trades", "/markets/trades"):
        cur = None
        while True:
            p = {"ticker": tk, "limit": 1000, "min_ts": t0, "max_ts": t1}
            if cur:
                p["cursor"] = cur
            d = _get(API + path, p)
            for x in d.get("trades", []):
                out.append({"ticker": tk, "ts": _iso(x["created_time"]),
                            "yes_price": float(x.get("yes_price_dollars") or x.get("yes_price", 0)),
                            "no_price": float(x.get("no_price_dollars") or x.get("no_price", 0)),
                            "count": float(x.get("count_fp") or x.get("count", 0)), "trade_id": x.get("trade_id")})
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
    return out


# ------------------------------------------------------------------------------------------------ SpaceX
def launches():
    f = OUT / "ll_spacex.json"
    if f.exists():
        return json.load(open(f))
    url = ("https://ll.thespacedevs.com/2.2.0/launch/previous/?lsp__id=121&net__gte=2024-01-01T00:00:00Z"
           "&limit=100&mode=list&ordering=net")
    rows = []
    while url:
        d = _get(url)
        if not d:
            raise SystemExit("LL2 fetch failed (rate limit 15/h?)")
        rows += [{"name": x["name"], "net": x["net"], "status": (x.get("status") or {}).get("abbrev")}
                 for x in d["results"]]
        url = d.get("next")
    json.dump(rows, open(f, "w"))
    return rows


def _month_bounds(y, m):
    a = datetime(y, m, 1, tzinfo=ET)
    b = datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=ET)
    return int(a.timestamp()), int(b.timestamp())


def spacex():
    OUT.mkdir(parents=True, exist_ok=True)
    L = launches()
    lt = np.array(sorted(_iso(x["net"]) for x in L if x["status"] in ("Success", "Failure", "Partial Failure")),
                  "int64")
    print(len(L), "LL2 launches;", len(lt), "flown; status", pd.Series([x["status"] for x in L]).value_counts().to_dict())
    mk = markets("KXSPACEXCOUNT")
    suf = mk["event_ticker"].str.split("-").str[1]
    mm = suf.str.extract(r"^(\d\d)([A-Z]{3})B?$")
    mk = mk[mm[1].isin(MON)].copy()
    mk["y"], mk["m"] = 2000 + mm.loc[mk.index, 0].astype(int), mm.loc[mk.index, 1].map(MON)
    rows, cand = [], {}
    for ev, g in mk.groupby("event_ticker"):
        y, m = int(g["y"].iloc[0]), int(g["m"].iloc[0])
        ms, me = _month_bounds(y, m)
        t_open = int(g["open_ts"].min())
        cs = candles("KXSPACEXCOUNT", ev, list(g["ticker"]), t_open - 3600, me + 3600)
        day = datetime(y, m, 1, 16, tzinfo=timezone.utc)
        Ts = []
        while int(day.timestamp()) < me:
            t = int(day.timestamp())
            if t >= t_open:
                Ts.append(t)
            day += timedelta(days=1)
        for T in Ts:
            c = int(((lt >= ms) & (lt < T)).sum())
            Ld = int(np.ceil((me - T) / 86400))
            past = lt[(lt >= T - 365 * 86400) & (lt < T)]
            cap = int(max((np.searchsorted(past, s + Ld * 86400, "left") - i for i, s in enumerate(past)), default=0))
            last = int(lt[lt < T].max())
            for r in g.itertuples():
                if T >= r.close_ts:
                    continue
                yb, ya = quote(cs.get(r.ticker, []), T)
                mb, ma = quote(cs.get(r.ticker, []), me - 3600)
                rows.append({"ticker": r.ticker, "event_ticker": ev, "date": f"{y}-{m:02d}-01", "K": r.K,
                             "strike_type": r.strike_type, "close_ts": r.close_ts, "month_end_ts": me, "T": T,
                             "c": c, "L": Ld, "cap": cap, "last_launch_ts": last, "yes_bid": yb, "yes_ask": ya,
                             "mark_yes_bid": mb, "mark_yes_ask": ma, "result": r.result,
                             "expiration_value": r.expiration_value, "volume": r.volume})
                if r.strike_type == "greater" and r.K + 1 - c >= cap + SX.MARGIN:
                    cand[r.ticker] = min(cand.get(r.ticker, T), T)
        print(ev, len(g), "rungs", len(Ts), "decision days", flush=True)
    fr = pd.DataFrame(rows)
    me_of = fr.groupby("ticker")["month_end_ts"].first()
    tp = []
    for tk, t0 in sorted(cand.items()):
        tp += tape(tk, t0, int(me_of[tk]))
    tp = pd.DataFrame(tp, columns=["ticker", "ts", "yes_price", "no_price", "count", "trade_id"])
    fr.to_parquet(OUT / "spacex_frame.parquet")
    tp.to_parquet(OUT / "spacex_tape.parquet")
    print("frame", len(fr), "rows,", fr["ticker"].nunique(), "tickers,", len(cand), "tape tickers,", len(tp), "trades")


# ------------------------------------------------------------------------------------------------ SPR
def eia_spr():
    f = OUT / "WCSSTUS1w.xls"
    if not f.exists():
        r = requests.get("https://www.eia.gov/dnav/pet/hist_xls/WCSSTUS1w.xls", headers=H, timeout=60)
        f.write_bytes(r.content)
    x = pd.read_excel(f, sheet_name="Data 1", skiprows=2)
    x.columns = ["week", "v"]
    x["week"] = pd.to_datetime(x["week"]).dt.normalize()
    x["v"] = x["v"].astype(float) / 1000.0
    return x.dropna().set_index("week")["v"].sort_index()


def spr():
    OUT.mkdir(parents=True, exist_ok=True)
    s = eia_spr()
    dv = s.diff().abs()
    mk = markets("KXSPRLVL")
    rows, cand = [], {}
    for ev, g in mk.groupby("event_ticker"):
        dte = datetime.strptime(ev.split("-")[1], "%y%b%d")
        wtxt = re.search(r"week ending\s+([A-Za-z]+ \d+, \d{4})", g["rules"].iloc[0])
        if not wtxt:  # e.g. KXSPRLVL-26APR01 ("level on April 1"): no WPSR week -> no inputs, no bets
            print(ev, "no 'week ending' in rules -> skipped")
            continue
        W = pd.Timestamp(datetime.strptime(wtxt.group(1), "%B %d, %Y"))
        W0 = W - pd.Timedelta(days=7)
        v0 = float(s.get(W0, np.nan))
        hist = dv[dv.index <= W0].dropna().iloc[-104:]
        dmax = float(hist.max()) if len(hist) == 104 else np.nan
        rel = int(datetime(dte.year, dte.month, dte.day, 14, 30, tzinfo=timezone.utc).timestamp())
        t_open = int(g["open_ts"].min())
        cs = candles("KXSPRLVL", ev, list(g["ticker"]), t_open - 3600, rel + 3600)
        for r in g.itertuples():
            T = r.open_ts + 3600
            yb, ya = quote(cs.get(r.ticker, []), T)
            mb, ma = quote(cs.get(r.ticker, []), rel - 3600)
            rows.append({"ticker": r.ticker, "event_ticker": ev, "date": dte.strftime("%Y-%m-%d"), "K": r.K,
                         "strike_type": r.strike_type, "open_ts": r.open_ts, "close_ts": r.close_ts, "T": T,
                         "rel_ts": rel, "week": W.strftime("%Y-%m-%d"), "v0": v0, "dmax": dmax,
                         "yes_bid": yb, "yes_ask": ya, "mark_yes_bid": mb, "mark_yes_ask": ma, "result": r.result,
                         "expiration_value": r.expiration_value, "volume": r.volume, "v_eia_target": float(s.get(W, np.nan))})
            if np.isfinite(v0) and np.isfinite(dmax) and (r.K >= v0 + dmax + SP.MARGIN or r.K <= v0 - dmax - SP.MARGIN):
                cand[r.ticker] = (T, rel)
        print(ev, "W", W.date(), "v0", v0, "dmax", round(dmax, 3), len(g), "rungs K", g["K"].min(), "-", g["K"].max(),
              flush=True)
    fr = pd.DataFrame(rows)
    tp = []
    for tk, (a, b) in sorted(cand.items()):
        tp += tape(tk, a, b)
    tp = pd.DataFrame(tp, columns=["ticker", "ts", "yes_price", "no_price", "count", "trade_id"])
    fr.to_parquet(OUT / "spr_frame.parquet")
    tp.to_parquet(OUT / "spr_tape.parquet")
    print("frame", len(fr), "rows;", len(cand), "input-selected tickers;", len(tp), "trades")


def audit():
    """Settlement check (descriptive; outcome columns read here only to validate the input sources)."""
    f = pd.read_parquet(OUT / "spacex_frame.parquet")
    L = launches()
    lt = np.array(sorted(_iso(x["net"]) for x in L if x["status"] in ("Success", "Failure", "Partial Failure")))
    ev = f.groupby("event_ticker").agg(date=("date", "first"), me=("month_end_ts", "first"),
                                       val=("expiration_value", "max"))
    for e, r in ev.iterrows():
        y, m = int(r.date[:4]), int(r.date[5:7])
        ms, me = _month_bounds(y, m)
        print("SX", e, "LL2(ET month)", int(((lt >= ms) & (lt < me)).sum()), "kalshi", r.val)
    g = pd.read_parquet(OUT / "spr_frame.parquet").groupby("event_ticker").agg(
        eia=("v_eia_target", "first"), val=("expiration_value", "max"))
    for e, r in g.iterrows():
        print("SP", e, "EIA", r.eia, "kalshi", r.val)


if __name__ == "__main__":
    {"spacex": spacex, "spr": spr, "audit": audit}[sys.argv[1]]()
