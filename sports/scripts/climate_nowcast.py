"""climate_nowcast agent (2026-11-20): census of Kalshi climate-index series (global temp, sea ice, CO2, ENSO).
Read-only public API GETs. No orders.
    python scripts/climate_nowcast.py census   -> results/climate_nowcast/census.json + markets_<S>.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "results" / "climate_nowcast"
H = {"User-Agent": "Mozilla/5.0"}
SERIES = ["KXHMONTH", "HMONTH", "KXHMONTHRANGE", "KXTEMPMON", "TEMPMON", "KXGTEMP", "GTEMP", "KXHOTYEAR", "HOTYEAR",
          "KXWARMING", "KXARCTICICEMAX", "ARCTICICEMAX", "KXARCTICICEMIN", "ARCTICICEMIN", "KXARTICICE", "ARTICICE",
          "KXSEAICE", "SEAICE", "KXCO2LEVEL", "KXLANINA", "KXELNINO", "KXANTARCTICICE", "KXANTARCTICICEMAX",
          "KXANTARCTICICEMIN", "KXHOTMONTH", "KXGLOBALTEMP"]


def _get(path, **p):
    for i in range(6):
        try:
            r = requests.get(f"{API}{path}", params=p, headers=H, timeout=60)
        except requests.RequestException:
            time.sleep(2 + 2 * i); continue
        if r.status_code == 429:
            time.sleep(2 + 2 * i); continue
        if r.status_code != 200:
            return {}
        return r.json()
    return {}


def _all(path, key="markets", **p):
    out, cur = [], None
    while True:
        q = dict(p, limit=1000)
        if cur: q["cursor"] = cur
        j = _get(path, **q)
        out += j.get(key, [])
        cur = j.get("cursor")
        if not cur or len(out) > 50000: return out


def census():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for s in SERIES:
        ser = _get(f"/series/{s}").get("series", {})
        ms = {}
        for st in ("settled", "open", "closed"):
            for m in _all("/markets", series_ticker=s, status=st): ms[m["ticker"]] = m
        for m in _all("/historical/markets", series_ticker=s): ms.setdefault(m["ticker"], m)
        if not ms and not ser:
            continue
        json.dump(list(ms.values()), open(OUT / f"markets_{s}.json", "w"))
        evs = {}
        for m in ms.values():
            e = evs.setdefault(m["event_ticker"], {"n": 0, "vol": 0.0, "status": set(), "close": m.get("close_time")})
            e["n"] += 1; e["vol"] += float(m.get("volume_fp") or m.get("volume") or 0); e["status"].add(m.get("status"))
        rows[s] = {"title": ser.get("title"), "freq": ser.get("frequency"), "fee_type": ser.get("fee_type"),
                   "src": [d.get("url") for d in (ser.get("settlement_sources") or [])],
                   "n_mkts": len(ms), "n_events": len(evs),
                   "events": {k: {"n": v["n"], "vol": round(v["vol"]), "status": sorted(map(str, v["status"])),
                                  "close": v["close"]} for k, v in sorted(evs.items())}}
        print(s, len(evs), "events", len(ms), "mkts", round(sum(v["vol"] for v in evs.values())), flush=True)
    json.dump(rows, open(OUT / "census.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "census": census()


# ---------------------------------------------------------------- build (CN1 frame; prereg in notes/climate_nowcast.md)
import numpy as np
import pandas as pd

ERA5 = "https://sites.ecmwf.int/data/climatepulse/data/series/era5_daily_series_2t_global.csv"
NOAA = ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/globe/land_ocean/1/0/"
        "1850-2026/data.csv")
DATA = ROOT / "data" / "climate_nowcast"
FIT = ("2001-01", "2023-12")
INFL = 1.25
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _fetch(url, name):
    DATA.mkdir(parents=True, exist_ok=True)
    p = DATA / name
    if not p.exists():
        r = requests.get(url, headers=H, timeout=120); r.raise_for_status(); p.write_text(r.text)
    return p


def era5_monthly():
    d = pd.read_csv(_fetch(ERA5, "era5_2t_global.csv"), comment="#")
    d["date"] = pd.to_datetime(d["date"]); d["m"] = d["date"].dt.to_period("M")
    g = d.groupby("m").agg(ano=("ano_91-20", "mean"), n=("ano_91-20", "size"))
    return g


def noaa_monthly():
    d = pd.read_csv(_fetch(NOAA, "noaa_land_ocean.csv"), comment="#")
    d.columns = ["Date", "v"]
    d["m"] = pd.PeriodIndex(pd.to_datetime(d["Date"].astype(str), format="%Y%m"), freq="M")
    return d.set_index("m")["v"]


def fit():
    e, n = era5_monthly(), noaa_monthly()
    j = pd.DataFrame({"e": e["ano"], "n": n}).dropna()
    j = j[(j.index >= pd.Period(FIT[0])) & (j.index <= pd.Period(FIT[1]))]
    j["off"] = j["n"] - j["e"]; j["cal"] = j.index.month
    c = j.groupby("cal")["off"].mean()
    res = j["off"] - j["cal"].map(c)
    sd = float(np.sqrt((res ** 2).sum() / (len(res) - 12)))
    return {int(k): float(v) for k, v in c.items()}, sd, len(j)


def _cget(ticker, series, a, b):
    for path in (f"/historical/markets/{ticker}/candlesticks", f"/series/{series}/markets/{ticker}/candlesticks"):
        j = _get(path, start_ts=a, end_ts=b, period_interval=60)
        if j.get("candlesticks"):
            return j["candlesticks"]
    return []


def build():
    c_cal, sd0, nfit = fit()
    sd = sd0 * INFL
    e, n = era5_monthly(), noaa_monthly()
    rows, candles = [], {}
    for s in ("KXHMONTHRANGE", "KXHMONTH"):
        for m in json.load(open(OUT / f"markets_{s}.json")):
            ev = m["event_ticker"]
            mon = pd.Period(pd.to_datetime(ev.split("-")[1], format="%y%b"), freq="M")
            nxt = (mon + 1).to_timestamp().tz_localize("UTC")
            T = nxt + pd.Timedelta(days=3, hours=16)
            ias = nxt + pd.Timedelta(days=2)
            close = pd.Timestamp(m["close_time"])
            es = min(close, nxt + pd.Timedelta(days=7))
            ee = e.loc[mon] if mon in e.index else None
            full = ee is not None and int(ee["n"]) == mon.days_in_month
            pred = float(ee["ano"]) + c_cal[mon.month] if full else np.nan
            pm1 = mon - 1
            pers = float(e.loc[pm1, "ano"]) + c_cal[mon.month] if pm1 in e.index else np.nan
            rec = float(n[(n.index.month == mon.month) & (n.index.year < mon.year)].max())
            lo = m.get("floor_strike"); hi = m.get("cap_strike")
            st = m.get("strike_type")
            rows.append({"ticker": m["ticker"], "event_ticker": ev, "series": s, "month": str(mon),
                         "strike_type": st, "lo": np.nan if lo is None else float(lo),
                         "hi": np.nan if hi is None else float(hi), "rec": rec,
                         "pred": pred, "pred_pers": pers, "sd": sd, "T": T, "inputs_as_of": ias,
                         "event_start": es, "close_time": close, "date": nxt.tz_localize(None),
                         "result": m.get("result"), "expiration_value": m.get("expiration_value")})
            if close > T - pd.Timedelta(hours=6):
                a = int(((T - pd.Timedelta(hours=24)) - EPOCH) // pd.Timedelta(seconds=1))
                b = int((es - EPOCH) // pd.Timedelta(seconds=1))
                candles[m["ticker"]] = _cget(m["ticker"], s, a, b)
                time.sleep(0.15)
    f = pd.DataFrame(rows)
    # ladder centre per event (placebo P1): midpoint of lowest 'less' cap and highest 'greater' floor
    cen = {}
    for evk, g in f[f.series == "KXHMONTHRANGE"].groupby("event_ticker"):
        lc = g.loc[g.strike_type == "less", "hi"].min(); gf = g.loc[g.strike_type == "greater", "lo"].max()
        cen[evk] = (lc + gf) / 2
    f["ladder_centre"] = f["event_ticker"].map(cen)
    DATA.mkdir(parents=True, exist_ok=True)
    f.to_parquet(DATA / "frame.parquet")
    json.dump(candles, open(DATA / "candles.json", "w"))
    json.dump({"c_cal": c_cal, "sd_fit": sd0, "sd_used": sd, "infl": INFL, "fit": FIT, "fit_months": nfit},
              open(DATA / "inputs.json", "w"), indent=1)
    print(len(f), "rows;", sum(1 for v in candles.values() if v), "with candles; sd_fit", round(sd0, 4), "n", nfit)


# ---------------------------------------------------------------- CN2 forward-only paper logger (NOT installed; no launchd)
# Spec (notes/climate_nowcast.md, CN2): run daily ~16:05Z; acts only on day 4 of M+1 (decision) and day 8 (CLV close).
# pred = ERA5 mean of M + trailing offset = mean(NOAA - ERA5) over months M-13..M-2 from the NOAA CSV vintage fetched
# at run time (vintage saved, so first prints accumulate); sd = 0.064. Same edge rule as CN1 (p - ask - fee >= .05).
LOG = DATA / "cn2_paper.jsonl"
CN2_SD = 0.064


def log():
    from datetime import datetime, timezone
    now = pd.Timestamp(datetime.now(timezone.utc))
    DATA.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M")
    era = requests.get(ERA5, headers=H, timeout=120).text; noaa = requests.get(NOAA, headers=H, timeout=120).text
    (DATA / "vintages").mkdir(exist_ok=True)
    (DATA / "vintages" / f"noaa_{stamp}.csv").write_text(noaa)
    e = pd.read_csv(__import__("io").StringIO(era), comment="#"); e["m"] = pd.to_datetime(e["date"]).dt.to_period("M")
    em = e.groupby("m")["ano_91-20"].agg(["mean", "size"])
    n = pd.read_csv(__import__("io").StringIO(noaa), comment="#"); n.columns = ["d", "v"]
    nv = n.set_index(pd.PeriodIndex(pd.to_datetime(n["d"].astype(str), format="%Y%m"), freq="M"))["v"]
    mon = pd.Period(now, freq="M") - 1
    if now.day not in (4, 8) or mon not in em.index or int(em.loc[mon, "size"]) != mon.days_in_month:
        print("not a decision/close day or ERA5 incomplete", now.day); return
    off = (nv - em["mean"]).dropna(); off = off[(off.index <= mon - 2) & (off.index > mon - 14)].mean()
    pred = float(em.loc[mon, "mean"]) + float(off)
    from scipy.stats import norm
    tag = mon.strftime("%y%b").upper()
    out = []
    for s in ("KXHMONTHRANGE", "KXHMONTH"):
        for m in _all("/markets", series_ticker=s, event_ticker=f"{s}-{tag}"):
            lo, hi, st = m.get("floor_strike"), m.get("cap_strike"), m.get("strike_type")
            if s == "KXHMONTH":
                rec = float(nv[(nv.index.month == mon.month) & (nv.index.year < mon.year)].max())
                py = 1 - norm.cdf((rec - pred) / CN2_SD)
            else:
                a = -np.inf if lo is None else lo; b = np.inf if hi is None else hi
                py = norm.cdf((b - pred) / CN2_SD) - norm.cdf((a - pred) / CN2_SD)
            out.append({"ts": now.isoformat(), "kind": "decision" if now.day == 4 else "close", "ticker": m["ticker"],
                        "month": str(mon), "pred": pred, "offset": float(off), "sd": CN2_SD, "p_yes": float(py),
                        "yes_bid": m.get("yes_bid_dollars"), "yes_ask": m.get("yes_ask_dollars"),
                        "close_time": m.get("close_time")})
    with open(LOG, "a") as fh:
        for r in out: fh.write(json.dumps(r) + "\n")
    print(len(out), "rows ->", LOG, "pred", round(pred, 3))


if __name__ == "__main__" and sys.argv[1] in ("build", "fit", "log"):
    {"build": build, "fit": lambda: print(fit()), "log": log}[sys.argv[1]]()
