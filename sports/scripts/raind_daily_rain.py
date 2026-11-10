"""Daily rain NO when NWS MOS rules rain out -- KXRAINNYC (2021-26) + KXRAINSEA (2021-22) + KXRAIN NYC rows. DEV only.
Pre-registration: notes/raind_daily_rain.md (written before any price / forecast / outcome pull).

    python scripts/raind_daily_rain.py markets   # market list + sealed cutoff (no outcomes printed)
    python scripts/raind_daily_rain.py mos       # IEM MOS archive (GFS + NAM, 18Z runs) -> data/raind/mos.parquet
    python scripts/raind_daily_rain.py frame     # quotes at T, mark, signal; tape for input-true tickers
Frame -> data/raind/rain_frame.parquet + rain_tape.parquet (edgelab loader kalshi_rain_daily_nyc).
"""
from __future__ import annotations

import io
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import requests

from sports import wx
from sports.http import fetch, fetch_many

OUT = ROOT / "data" / "raind"
OUT.mkdir(parents=True, exist_ok=True)
CITY = {"NYC": ("KNYC", "America/New_York"), "SEA": ("KSEA", "America/Los_Angeles")}
# P06 periods (ending hour UTC, day offset vs D) per city; see notes for the climate-day coverage argument
PERIODS = {"NYC": [(6, 0), (12, 0), (18, 0), (0, 1), (6, 1)], "SEA": [(12, 0), (18, 0), (0, 1), (6, 1), (12, 1)]}
THRESH = 5.0


def _all(path, **p):
    out, cur = [], None
    while True:
        q = dict(p, limit=1000)
        if cur:
            q["cursor"] = cur
        d = fetch(f"{wx.API}{path}", q, cache="historical" in path)
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            return out


def _date(ev: str) -> pd.Timestamp:
    return pd.to_datetime(ev.split("-")[1], format="%y%b%d")


def cmd_markets() -> int:
    rows = []
    for s in ("KXRAINNYC", "KXRAINSEA", "KXRAIN"):
        ms = {m["ticker"]: m for m in _all("/historical/markets", series_ticker=s) + _all("/markets", series_ticker=s, status="settled")}
        for m in ms.values():
            tk = m["ticker"]
            if s == "KXRAIN":
                if not tk.endswith("-NYC"):
                    continue
                city = "NYC"
            else:
                city = "NYC" if s == "KXRAINNYC" else "SEA"
                fs = m.get("floor_strike")
                if not (tk.endswith("-T0") or (m.get("strike_type") == "greater" and fs in (0, 0.0))):
                    continue
            if m.get("result") not in ("yes", "no"):
                continue
            rows.append({"series": s, "ticker": tk, "event_ticker": m["event_ticker"], "city": city,
                         "date": _date(m["event_ticker"]), "close_time": m.get("close_time"), "result": m["result"],
                         "rules": (m.get("rules_primary") or "")[:200]})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    dup = df.duplicated(["city", "date"], keep=False)
    print("city-date overlaps (legacy kept):", df[dup][["ticker", "date"]].to_dict("records"))
    df = df.sort_values("series", key=lambda s: s.eq("KXRAIN")).drop_duplicates(["city", "date"]).sort_values(["city", "date"])
    assert not df.duplicated(["city", "date"]).any(), df[df.duplicated(["city", "date"], keep=False)].head()
    df.to_parquet(OUT / "markets.parquet", index=False)
    u = np.sort(df.date.unique())
    cut = pd.Timestamp(u[int(np.floor(0.7 * len(u)))])
    print(df.groupby(["series", "city"]).date.agg(["count", "min", "max"]))
    print(f"unique dates {len(u)}; sealed = dates >= {cut.date()} ({(u >= cut.to_datetime64()).sum()} dates, "
          f"{(df.date >= cut).sum()} markets); dev = {(df.date < cut).sum()} markets")
    return 0


# ------------------------------------------------------------------------------------------------ MOS (IEM archive)
def _mos(station: str, model: str, sts: str, ets: str) -> pd.DataFrame:
    cache = OUT / "mos_cache" / f"{station}_{model}_{sts[:10]}.csv"
    cache.parent.mkdir(exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)
    for i in range(30):
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py",
                         params={"station": station, "model": model, "sts": sts, "ets": ets, "format": "csv"}, timeout=180)
        if r.status_code == 200 and r.text.startswith("runtime"):
            cache.write_text(r.text)
            time.sleep(3)
            return pd.read_csv(io.StringIO(r.text))
        time.sleep(20 + 10 * i)
    raise RuntimeError(f"IEM MOS failed {station} {model} {sts}: {r.status_code} {r.text[:200]}")


def cmd_mos() -> int:
    mk = pd.read_parquet(OUT / "markets.parquet")
    frames = []
    for city, g in mk.groupby("city"):
        st = CITY[city][0]
        months = pd.period_range(g.date.min() - pd.Timedelta(days=2), g.date.max(), freq="Y")
        for model in ("GFS", "NAM"):
            for pm in months:
                a = pm.start_time
                b = pm.end_time.normalize()
                d = _mos(st, model, f"{a:%Y-%m-%d}T00:00Z", f"{b:%Y-%m-%d}T23:59Z")
                d = d[pd.to_datetime(d.runtime).dt.hour == 18]
                d["city"] = city
                frames.append(d[["runtime", "ftime", "model", "p06", "city"]])
            print(city, model, "months", len(months), flush=True)
    m = pd.concat(frames, ignore_index=True)
    m["runtime"] = pd.to_datetime(m.runtime)
    m["ftime"] = pd.to_datetime(m.ftime)
    m.to_parquet(OUT / "mos.parquet", index=False)
    print(f"mos rows {len(m)}; runs {m.groupby(['city', 'model']).runtime.nunique().to_dict()}")
    return 0


def signal(mk: pd.DataFrame, mos: pd.DataFrame) -> pd.DataFrame:
    """Per (city, date): max P06 per model from the 18Z run of D-1 over PERIODS[city]."""
    out = []
    idx = mos.set_index(["city", "model", "runtime", "ftime"]).p06.sort_index()
    for city, date in mk[["city", "date"]].drop_duplicates().itertuples(index=False):
        run = pd.Timestamp(date) - pd.Timedelta(days=1) + pd.Timedelta(hours=18)
        row = {"city": city, "date": date, "runtime": run}
        for model in ("GFS", "NAM"):
            vals = []
            for h, off in PERIODS[city]:
                ft = pd.Timestamp(date) + pd.Timedelta(days=off) + pd.Timedelta(hours=h)
                try:
                    v = idx.loc[(city, model, run, ft)]
                    v = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
                except KeyError:
                    v = np.nan
                vals.append(v)
            row[f"p06max_{model}"] = max(vals) if all(np.isfinite(vals)) else np.nan
        out.append(row)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ frame
def _local_ts(date, tz, hour, day_off):
    d = pd.Timestamp(date) + pd.Timedelta(days=day_off)
    return int(datetime(d.year, d.month, d.day, hour, tzinfo=ZoneInfo(tz)).timestamp())


def _candle_rows(tickers, events, series, t0s, t1s):
    """Hourly candles per ticker in [t0, t1]; event endpoint first, historical per-market fallback."""
    res = {}
    urls = [f"{wx.API}/series/{s}/events/{e}/candlesticks?start_ts={a}&end_ts={b}&period_interval=60"
            for s, e, a, b in zip(series, events, t0s, t1s)]
    miss = []
    for tk, e, d in zip(tickers, events, fetch_many(urls)):
        tks = d.get("market_tickers", [])
        if tk in tks:
            res[tk] = d["market_candlesticks"][tks.index(tk)]
        else:
            miss.append(tk)
    sel = [i for i, tk in enumerate(tickers) if tk in set(miss)]
    urls = [f"{wx.API}/historical/markets/{tickers[i]}/candlesticks?start_ts={t0s[i]}&end_ts={t1s[i]}&period_interval=60"
            for i in sel]
    for i, d in zip(sel, fetch_many(urls)):
        res[tickers[i]] = d.get("candlesticks", [])
    return res


def cmd_frame() -> int:
    mk = pd.read_parquet(OUT / "markets.parquet")
    # Lead constraint 2026-11-09: R1 (kalshi_rain) scored KXRAIN dev 07-15..09-14 and its LOCKBOX 09-15..10-18
    # (lab/lockbox.jsonl line 3). Those dates are dropped from this dataset entirely (never loadable by any split).
    r1 = mk.date.between(pd.Timestamp("2026-07-15"), pd.Timestamp("2026-10-18"))
    print(f"dropping {int(r1.sum())} rows in R1's dev+sealed window 2026-07-15..10-18")
    mk = mk[~r1].reset_index(drop=True)
    mos = pd.read_parquet(OUT / "mos.parquet")
    sig = signal(mk, mos)
    f = mk.merge(sig, on=["city", "date"], how="left", validate="1:1")
    tz = f.city.map(lambda c: CITY[c][1])
    f["T"] = [_local_ts(d, z, 22, -1) for d, z in zip(f.date, tz)]
    f["event_start_ts"] = [_local_ts(d, z, 0, 0) for d, z in zip(f.date, tz)]
    f["close_ts"] = pd.to_datetime(f.close_time, utc=True, format="ISO8601").map(lambda x: int(x.timestamp()))
    # conservative MOS issue time = runtime + 4 h  (traps: build from naive UTC timestamps explicitly)
    f["inputs_ts"] = (pd.to_datetime(f.runtime).dt.tz_localize("UTC") + pd.Timedelta(hours=4)).map(lambda x: int(x.timestamp()))
    assert (f.inputs_ts <= f["T"]).all() and (f["T"] < f.event_start_ts).all() and (f.event_start_ts < f.close_ts).all()
    cs = _candle_rows(list(f.ticker), list(f.event_ticker), list(f.series), list(f["T"] - 6 * 3600),
                      list(f.event_start_ts))
    q = []
    for tk, t, es in zip(f.ticker, f["T"], f.event_start_ts):
        c = cs.get(tk, [])
        a, b = wx._row(tk, "", c, int(t)), wx._row(tk, "", c, int(es))
        q.append({"ticker": tk, "price_ts": a["price_ts"] if a else np.nan, "yes_bid": a["yes_bid"] if a else np.nan,
                  "yes_ask": a["yes_ask"] if a else np.nan, "vol_to_T": a["volume_to_decision"] if a else np.nan,
                  "mark_yes_bid": b["yes_bid"] if b else np.nan, "mark_yes_ask": b["yes_ask"] if b else np.nan})
    f = f.merge(pd.DataFrame(q), on="ticker", how="left", validate="1:1")
    sys.path.insert(0, str(Path.home() / "agent-harness" / "lib"))
    import traps
    for c in ("T", "event_start_ts", "close_ts", "inputs_ts", "price_ts"):
        traps.check_epoch_seconds(f[c], c)
    tt = pd.DataFrame({"inp": pd.to_datetime(f.inputs_ts, unit="s", utc=True), "T": pd.to_datetime(f["T"], unit="s", utc=True),
                       "es": pd.to_datetime(f.event_start_ts, unit="s", utc=True)})
    traps.check_no_lookahead(tt, "inp", "T", strict=False)
    traps.check_no_lookahead(tt, "T", "es")
    pq = f.dropna(subset=["price_ts"])
    assert (pq.price_ts <= pq["T"]).all(), "quote after decision"
    # local 22:00 sanity: T in UTC must be 02Z (EDT) / 03Z (EST) for NYC, 05Z/06Z for SEA
    hrs = pd.to_datetime(f["T"], unit="s", utc=True).dt.hour
    assert hrs[f.city == "NYC"].isin([2, 3]).all() and hrs[f.city == "SEA"].isin([5, 6]).all()
    f.to_parquet(OUT / "rain_frame.parquet", index=False)
    inp = f[[c for c in f.columns if c.startswith("p06max_")]]
    itrue = f.p06max_GFS.notna() & (inp.max(axis=1, skipna=True) <= THRESH)
    print(f"frame {len(f)} rows; GFS sig {f.p06max_GFS.notna().mean():.2f} NAM sig {f.p06max_NAM.notna().mean():.2f}; "
          f"quote at T {f.yes_bid.notna().mean():.2f}; input-true {int(itrue.sum())}")
    tape = []
    sel = f[itrue]
    from scripts_kalshi_temp_maker import _tape  # noqa: E402  (shim below)
    for i, (tk, t, c) in enumerate(zip(sel.ticker, sel["T"], sel.close_ts)):
        tape += _tape(tk, int(t), int(c))
        if i % 100 == 0:
            print(f"  tape {i}/{len(sel)}", flush=True)
    tp = pd.DataFrame(tape, columns=["ticker", "ts", "no_price", "count", "taker_side"])
    tp.to_parquet(OUT / "rain_tape.parquet", index=False)
    print(f"tape {len(tp)} trades on {tp.ticker.nunique()} of {len(sel)} input-true tickers")
    return 0


if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location("scripts_kalshi_temp_maker", ROOT / "scripts" / "kalshi_temp_maker.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scripts_kalshi_temp_maker"] = mod
    spec.loader.exec_module(mod)
    sys.exit({"markets": cmd_markets, "mos": cmd_mos, "frame": cmd_frame}[sys.argv[1]]())
