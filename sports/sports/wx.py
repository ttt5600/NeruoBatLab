"""Kalshi daily-high-temperature markets + archived forecasts, for backtesting.

Each market is a bracket on one city's official daily maximum temperature
(NWS climate report, e.g. CLINYC = Central Park). Each day has ~6 brackets:
two open tails ("less"/"greater") and 2-degree "between" bins; exactly one
pays $1.

Timing is the whole game. A bet is placed at DECISION_HOUR local time on
the day BEFORE, at the ask actually quoted then (hourly candle). The forecast
used is the GFS run from ~24 h before each valid hour (Open-Meteo "previous
runs", lead day 1) -- issued before the decision, never revised after it.

Caveats recorded, not hidden:
* The NWS climate day runs in local STANDARD time; the forecast max here is
  over the local calendar day. In summer they differ by an hour at midnight,
  which rarely moves a daily max.
* Kalshi's taker fee is modelled as 0.07 * p * (1 - p) per contract
  (FEE_RATE); verify against the current fee schedule before real money.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .http import fetch, fetch_many

API = "https://api.elections.kalshi.com/trade-api/v2"
PREV_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
FEE_RATE = 0.07
DECISION_HOUR = 22  # local, on the day before

# series -> (NWS station from the contract rules, local tz)
CITIES = {
    "KXHIGHNY": ("KNYC", "America/New_York"),
    "KXHIGHCHI": ("KMDW", "America/Chicago"),
    "KXHIGHMIA": ("KMIA", "America/New_York"),
    "KXHIGHAUS": ("KAUS", "America/Chicago"),
    "KXHIGHDEN": ("KDEN", "America/Denver"),
    "KXHIGHLAX": ("KLAX", "America/Los_Angeles"),
    "KXHIGHPHIL": ("KPHL", "America/New_York"),
}


def station_coords(station: str) -> tuple[float, float]:
    g = fetch(f"https://api.weather.gov/stations/{station}")["geometry"]["coordinates"]
    return float(g[1]), float(g[0])


def settled_markets(series: str) -> pd.DataFrame:
    """Every settled bracket: archived (/historical) plus recently settled."""
    rows = []
    for path, cache in ((f"{API}/historical/markets", True), (f"{API}/markets", False)):
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000}
            if path.endswith("/markets") and "historical" not in path:
                p["status"] = "settled"
            if cursor:
                p["cursor"] = cursor
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("ticker")
    keep = ["ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike",
            "expiration_value", "result", "open_time", "close_time"]
    df = df[keep].copy()
    # Archived markets record the official value only on the WINNING bracket;
    # the day's other brackets carry "". The value is a property of the day.
    df["expiration_value"] = pd.to_numeric(df["expiration_value"], errors="coerce")
    df["expiration_value"] = df.groupby("event_ticker")["expiration_value"].transform("max")
    df = df[df["result"].isin(["yes", "no"]) & df["expiration_value"].notna()]
    return df.reset_index(drop=True)


def event_date(event_ticker: str) -> pd.Timestamp:
    """'KXHIGHNY-26AUG06' / 'HIGHNY-21AUG06' -> 2026-08-06."""
    # Last segment: one Chicago event is 'HIGHCHI-2-24FEB28'.
    return pd.to_datetime(event_ticker.split("-")[-1], format="%y%b%d")


def decision_ts(day: pd.Timestamp, tz: str) -> int:
    local = datetime(day.year, day.month, day.day, DECISION_HOUR, tzinfo=ZoneInfo(tz)) - timedelta(days=1)
    return int(local.timestamp())


def _row(tk, ev, candles, t):
    c = [x for x in candles if x["end_period_ts"] <= t]
    if not c:
        return None
    last = c[-1]

    def px(side):
        v = last.get(side) or {}
        return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")

    return {"ticker": tk, "event_ticker": ev, "price_ts": last["end_period_ts"],
            "yes_ask": px("yes_ask"), "yes_bid": px("yes_bid"),
            "volume_to_decision": float(sum(float(x.get("volume_fp", x.get("volume", 0)) or 0) for x in c))}


def prices_at_decision(series: str, markets: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Hourly yes bid/ask for every bracket at the decision time.

    One request per day via the event endpoint where it answers; archived days
    (where it returns nothing) fall back to one request per bracket.
    """
    events = sorted(markets["event_ticker"].unique())
    ts = {ev: decision_ts(event_date(ev), tz) for ev in events}
    urls = [f"{API}/series/{series}/events/{ev}/candlesticks"
            f"?start_ts={ts[ev] - 3 * 3600}&end_ts={ts[ev]}&period_interval=60" for ev in events]
    out, missing = [], []
    for ev, d in zip(events, fetch_many(urls)):
        tks = d.get("market_tickers", [])
        if not tks:
            missing.append(ev)
        for tk, candles in zip(tks, d.get("market_candlesticks", [])):
            r = _row(tk, ev, candles, ts[ev])
            if r:
                out.append(r)
    legacy = markets[markets["event_ticker"].isin(missing)]
    urls = [f"{API}/historical/markets/{tk}/candlesticks?start_ts={ts[ev] - 3 * 3600}"
            f"&end_ts={ts[ev]}&period_interval=60" for tk, ev in zip(legacy["ticker"], legacy["event_ticker"])]
    for (tk, ev), d in zip(zip(legacy["ticker"], legacy["event_ticker"]), fetch_many(urls)):
        r = _row(tk, ev, d.get("candlesticks", []), ts[ev])
        if r:
            out.append(r)
    return pd.DataFrame(out)


def forecast_daily_max(lat: float, lon: float, tz: str, start: str, end: str) -> pd.DataFrame:
    """GFS lead-1-day and lead-2-day forecast of the local-day maximum, in F."""
    frames = []
    s = pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        done = e < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
        d = fetch(PREV_RUNS, {
            "latitude": lat, "longitude": lon, "models": "gfs_seamless",
            "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
            "temperature_unit": "fahrenheit", "timezone": tz,
            "start_date": s.date().isoformat(), "end_date": e.date().isoformat()}, cache=done)
        h = pd.DataFrame(d["hourly"])
        h["day"] = pd.to_datetime(h["time"]).dt.normalize()
        frames.append(h.groupby("day")[["temperature_2m_previous_day1",
                                         "temperature_2m_previous_day2"]].max())
        s = e + pd.Timedelta(days=1)
    f = pd.concat(frames)
    f.columns = ["fc_max_lead1", "fc_max_lead2"]
    return f


def build(series: str) -> pd.DataFrame:
    station, tz = CITIES[series]
    lat, lon = station_coords(station)
    mk = settled_markets(series)
    mk["date"] = mk["event_ticker"].map(event_date)
    px = prices_at_decision(series, mk, tz)
    df = mk.merge(px, on=["ticker", "event_ticker"], how="inner")
    fc = forecast_daily_max(lat, lon, tz, str(mk["date"].min().date()), str(mk["date"].max().date()))
    df = df.join(fc, on="date")
    df["city"] = series.replace("KXHIGH", "")
    df["station"] = station
    return df
