"""Kalshi HOURLY temperature markets (KXTEMP*H, KXTEMP*HS): data builder for edgelab trial HT1
(rule + pre-registration: edgelab/rules/kalshi_wx_hourly.py; dataset kalshi_temp_hourly in edgelab/vault.yaml).

    python scripts/kalshi_hourly_temp.py build      # markets, values, forecasts, 1-min quotes, tapes -> data/kalshi_hourly_temp/

Build order keeps downloads small: (1) settled markets of every hourly series; (2) Open-Meteo previous-runs (lead-1)
hourly temperature, 5 models, per city point; (3) the input-only pre-quote filter (model trajectories miss K by > M,
H-2 value public by T) picks candidate EVENTS; (4) 1-min event candlesticks for those events -> quote at T + CLV mark;
(5) the rule's full input selection (incl. quote filter) picks TICKERS; (6) public trade tape (T, H] for those tickers.
No outcome is used to choose what to download. The frame stores `result`/`expiration_value`/`volume` for the vault
to hide.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from edgelab.rules import kalshi_wx_hourly as R  # noqa: E402
from sports import wx  # noqa: E402

OUT = ROOT / "data" / "kalshi_hourly_temp"
H = {"User-Agent": "Mozilla/5.0"}
SERIES = {  # series -> NWS station whose coordinates stand in for the city (forecast CHANGES only)
    "KXTEMPMIAH": "KMIA", "KXTEMPLAXH": "KLAX", "KXTEMPLAXHS": "KLAX", "KXTEMPNYCH": "KNYC", "KXTEMPNYCHS": "KNYC",
    "KXTEMPCHIH": "KMDW", "KXTEMPCHIHS": "KMDW", "KXTEMPAUSH": "KAUS", "KXTEMPDCH": "KDCA", "KXTEMPBOSH": "KBOS",
}
T_OFFSET = 15 * 60
MARK_BEFORE = 5 * 60


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


def markets(series: str) -> pd.DataFrame:
    rows = []
    for path, st in (("/historical/markets", None), ("/markets", "settled")):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if st:
                p["status"] = st
            if cur:
                p["cursor"] = cur
            d = _get(wx.API + path, p)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"])].copy()
    vol = df["volume_fp"] if "volume_fp" in df else df.get("volume")
    out = pd.DataFrame({
        "ticker": df["ticker"], "series": series, "event_ticker": df["event_ticker"],
        "strike_type": df["strike_type"], "K": pd.to_numeric(df["floor_strike"], errors="coerce"),
        "open_ts": df["open_time"].map(_iso), "H": df["close_time"].map(_iso),
        "settle_ts": df["settlement_ts"].map(lambda x: _iso(x) if isinstance(x, str) and x else -1),
        "expiration_value": pd.to_numeric(df["expiration_value"], errors="coerce"),
        "result": df["result"], "volume": pd.to_numeric(vol, errors="coerce")})
    ev = out.groupby("event_ticker").agg(val=("expiration_value", "max"), st=("settle_ts", "max"), H=("H", "first"))
    ev = ev[ev["st"] > 0]
    key = {(series, int(h)): (v, s) for (v, s, h) in zip(ev["val"], ev["st"], ev["H"])}
    prev = [key.get((series, int(h) - 7200), (np.nan, -1)) for h in out["H"]]
    out["v2"] = [p[0] for p in prev]
    out["v2_settle_ts"] = np.array([p[1] for p in prev], "int64")
    out["v2_settle_ts"] = np.where(np.isfinite(out["v2"]), out["v2_settle_ts"], np.iinfo("int64").max)
    out["T"] = out["open_ts"] + T_OFFSET
    out["date"] = pd.to_datetime(out["event_ticker"].str.split("-").str[1].str[:7], format="%y%b%d")
    return out.reset_index(drop=True)


def forecasts(station: str, start: str, end: str) -> pd.DataFrame:
    lat, lon = wx.station_coords(station)
    d = _get(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": ",".join(R.MODELS),
                            "hourly": "temperature_2m_previous_day1", "temperature_unit": "fahrenheit",
                            "timezone": "GMT", "start_date": start, "end_date": end})["hourly"]
    f = pd.DataFrame(d)
    f["ts"] = pd.to_datetime(f.pop("time")).map(lambda t: int(t.tz_localize("UTC").timestamp()))
    f = f.rename(columns={f"temperature_2m_previous_day1_{m}": m for m in R.MODELS})
    return f.set_index("ts")[[m for m in R.MODELS if m in f.columns]]


def _candles(series, ev, t0, t1):
    d = _get(f"{wx.API}/series/{series}/events/{ev}/candlesticks",
             {"start_ts": t0, "end_ts": t1, "period_interval": 1})
    out = {}
    for tk, cs in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
        out[tk] = cs
    return out


def _px(c, side):
    v = c.get(side) or {}
    return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")


def _quote(cs, t):
    c = [x for x in cs if x["end_period_ts"] <= t]
    return (_px(c[-1], "yes_bid"), _px(c[-1], "yes_ask")) if c else (np.nan, np.nan)


def _hist_candles(tk, t0, t1):
    return _get(f"{wx.API}/historical/markets/{tk}/candlesticks",
                {"start_ts": t0, "end_ts": t1, "period_interval": 1}).get("candlesticks", [])


def _tape(tk, t0, t1):
    out = []
    for path in ("/historical/trades", "/markets/trades"):
        cur = None
        while True:
            p = {"ticker": tk, "limit": 1000, "min_ts": t0, "max_ts": t1}
            if cur:
                p["cursor"] = cur
            d = _get(wx.API + path, p)
            for x in d.get("trades", []):
                out.append({"ticker": tk, "ts": _iso(x["created_time"]),
                            "yes_price": float(x.get("yes_price_dollars") or x.get("yes_price", 0)),
                            "no_price": float(x.get("no_price_dollars") or x.get("no_price", 0)),
                            "count": float(x.get("count_fp") or x.get("count", 0)), "taker_side": x.get("taker_side"),
                            "trade_id": x.get("trade_id")})
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
    return out


def build() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fm = OUT / "markets.parquet"
    if fm.exists():
        mk = pd.read_parquet(fm)
    else:
        mk = pd.concat([markets(s) for s in SERIES], ignore_index=True)
        mk.to_parquet(fm)
    print(f"markets: {len(mk)} on {mk.event_ticker.nunique()} events, {mk.series.nunique()} series", flush=True)
    # forecasts -> per-row model deltas d_m = fc(H) - fc(H-2)
    ff = OUT / "forecasts.parquet"
    if ff.exists():
        fc = pd.read_parquet(ff)
    else:
        start = (pd.to_datetime(mk["H"].min(), unit="s") - pd.Timedelta(days=1)).date().isoformat()
        end = pd.to_datetime(mk["H"].max(), unit="s").date().isoformat()
        parts = []
        for st in sorted(set(SERIES.values())):
            f = forecasts(st, start, end)
            f["station"] = st
            parts.append(f.reset_index())
        fc = pd.concat(parts, ignore_index=True)
        fc.to_parquet(ff)
    mk["station"] = mk["series"].map(SERIES)
    for m in R.MODELS:
        lk = {(s, int(t)): v for s, t, v in zip(fc["station"], fc["ts"], fc[m])} if m in fc else {}
        a = np.array([lk.get((s, int(h)), np.nan) for s, h in zip(mk["station"], mk["H"])], float)
        b = np.array([lk.get((s, int(h) - 7200), np.nan) for s, h in zip(mk["station"], mk["H"])], float)
        mk[f"d_{m}"] = a - b
    # pre-quote candidate events: the selection with a dummy always-passing quote
    tmp = mk.copy()
    tmp["yes_bid"], tmp["yes_ask"] = 0.45, 0.50
    pre = R.select(tmp)
    ev = pre[["series", "event_ticker", "open_ts", "H", "T"]].drop_duplicates("event_ticker")
    print(f"candidate events (input-only, pre-quote): {len(ev)} of {mk.event_ticker.nunique()}", flush=True)
    fq = OUT / "quotes.parquet"
    if fq.exists():
        q = pd.read_parquet(fq)
    else:
        need = set(pre["ticker"])

        def one(r):
            cs = _candles(r.series, r.event_ticker, int(r.open_ts) - 60, int(r.H))
            rows = []
            tks = [t for t in mk.loc[mk.event_ticker == r.event_ticker, "ticker"] if t in need]
            for tk in tks:
                c = cs.get(tk)
                if c is None:
                    c = _hist_candles(tk, int(r.open_ts) - 60, int(r.H))
                yb, ya = _quote(c, int(r.T))
                mb, ma = _quote(c, int(r.H) - MARK_BEFORE)
                rows.append({"ticker": tk, "yes_bid": yb, "yes_ask": ya, "mark_yes_bid": mb, "mark_yes_ask": ma})
            return rows

        with ThreadPoolExecutor(8) as ex:
            res = []
            for i, x in enumerate(ex.map(one, ev.itertuples())):
                res += x
                if i % 1000 == 0:
                    print(f"  candles {i}/{len(ev)}", flush=True)
        q = pd.DataFrame(res, columns=["ticker", "yes_bid", "yes_ask", "mark_yes_bid", "mark_yes_ask"])
        q.to_parquet(fq)
    mk = mk.merge(q, on="ticker", how="left")
    sel = R.select(mk)
    print(f"selected tickers (full input selection): {len(sel)}", flush=True)
    ft = OUT / "tape.parquet"
    if not ft.exists():
        with ThreadPoolExecutor(8) as ex:
            res = []
            for i, x in enumerate(ex.map(lambda r: _tape(r.ticker, int(r.T), int(r.H)), sel.itertuples())):
                res += x
                if i % 2000 == 0:
                    print(f"  tape {i}/{len(sel)}", flush=True)
        tp = pd.DataFrame(res, columns=["ticker", "ts", "yes_price", "no_price", "count", "taker_side", "trade_id"])
        tp = tp.drop_duplicates("trade_id") if tp["trade_id"].notna().all() else tp.drop_duplicates()
        tp.to_parquet(ft)
    mk.drop(columns=["station"]).to_parquet(OUT / "frame.parquet")
    print("frame written", OUT / "frame.parquet", flush=True)
    return 0


if __name__ == "__main__":
    if (sys.argv[1:2] or [""])[0] != "build":
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(build())
