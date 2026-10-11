"""Weather AT the game, for any league: what happened, and what was forecast.

    observed  ERA5 reanalysis (Open-Meteo archive), start .. start + duration.
              Hindsight: use it to measure how much weather matters (an oracle),
              never as a betting input.
    fc_d1/d3  the forecast for the same hours issued ~1 / ~3 days before
              (Open-Meteo previous-runs, best-match model). This is what a
              bettor could have known. Archive starts ~2021; earlier games
              get NaN, not a silent fallback to observed.

Venues are geocoded once from ESPN's city/state (Open-Meteo geocoding).
City-level coordinates are ~10 km off a stadium; ERA5 cells are ~25 km, so
it does not matter. Requests are batched per (venue, season): one archive
call and one forecast call cover every game a venue hosted that season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .http import fetch

ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
GEO = "https://geocoding-api.open-meteo.com/v1/search"
VARS = ["precipitation", "rain", "snowfall", "wind_speed_10m", "wind_gusts_10m", "temperature_2m"]
FC_VARS = ["precipitation", "snowfall", "wind_speed_10m", "temperature_2m"]
FC_LEADS = (1, 3)
FC_FIRST = pd.Timestamp("2021-03-23", tz="UTC")  # earliest previous-runs data
DURATION_H = {"nfl": 3, "cfb": 4, "mlb": 3, "nba": 3, "wnba": 2, "nhl": 3}

STATES = dict(AL="Alabama", AK="Alaska", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado",
              CT="Connecticut", DE="Delaware", FL="Florida", GA="Georgia", HI="Hawaii", ID="Idaho",
              IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine",
              MD="Maryland", MA="Massachusetts", MI="Michigan", MN="Minnesota", MS="Mississippi",
              MO="Missouri", MT="Montana", NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey",
              NM="New Mexico", NY="New York", NC="North Carolina", ND="North Dakota", OH="Ohio",
              OK="Oklahoma", OR="Oregon", PA="Pennsylvania", RI="Rhode Island", SC="South Carolina",
              SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia",
              WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming", DC="District of Columbia")


def geocode(city: str | None, state: str | None, country: str | None) -> tuple[float, float] | None:
    if not city:
        return None
    res = fetch(GEO, {"name": city, "count": 10}).get("results", []) or []
    want = STATES.get(state or "", state)
    for r in res:
        if want and r.get("admin1") == want:
            return r["latitude"], r["longitude"]
    for r in res:  # non-US venues (London, Munich, Mexico City, Toronto)
        if country and r.get("country") == country:
            return r["latitude"], r["longitude"]
    return (res[0]["latitude"], res[0]["longitude"]) if res and not state else None


def _hourly(url: str, lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp, vars_: list[str],
            cache: bool) -> pd.DataFrame:
    d = fetch(url, {"latitude": round(lat, 3), "longitude": round(lon, 3), "hourly": ",".join(vars_),
                    "start_date": start.date().isoformat(), "end_date": end.date().isoformat(),
                    "timezone": "UTC", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}, cache=cache)
    h = pd.DataFrame(d.get("hourly", {}))
    if h.empty:
        return h
    h["time"] = pd.to_datetime(h["time"]).dt.tz_localize("UTC")
    return h.set_index("time")


def _window(h: pd.DataFrame, t0: pd.Timestamp, hours: int, sfx: str = "") -> dict:
    w = h.loc[t0.floor("h"): t0.floor("h") + pd.Timedelta(hours=hours)] if len(h) else h
    if w.empty or w.isna().all().all():
        return {}
    g = lambda v: w[v + sfx] if v + sfx in w else pd.Series(dtype=float)
    out = {"precip_mm": g("precipitation").sum(min_count=1), "snow_cm": g("snowfall").sum(min_count=1),
           "wet_hours": float((g("precipitation") >= 0.2).sum()),
           "wind_mean": g("wind_speed_10m").mean(), "temp_mean": g("temperature_2m").mean()}
    if not sfx:
        out.update(rain_mm=g("rain").sum(min_count=1), wind_max=g("wind_speed_10m").max(),
                   gust_max=g("wind_gusts_10m").max())
    return out


def for_games(g: pd.DataFrame, league: str, progress=None) -> pd.DataFrame:
    """Weather columns for every outdoor game in ``g`` (output of espn.games)."""
    g = g[g["indoor"] != True].copy()  # noqa: E712 -- None = unknown, keep it
    hours = DURATION_H.get(league, 3)
    now = pd.Timestamp.now(tz="UTC")
    g["start_utc"] = pd.to_datetime(g["start_utc"], utc=True)
    g["_loc"] = list(zip(g["city"], g["state"], g["country"]))
    coords = {loc: geocode(*loc) for loc in g["_loc"].dropna().unique()}
    rows = []
    for (loc, season), gg in g.groupby(["_loc", "season"]):
        c = coords.get(loc)
        if c is None:
            continue
        t0, t1 = gg["start_utc"].min() - pd.Timedelta(days=1), gg["start_utc"].max() + pd.Timedelta(days=1)
        done = t1 < now - pd.Timedelta(days=7)  # ERA5 lags ~5 days
        obs = _hourly(ARCHIVE, *c, t0, min(t1, now - pd.Timedelta(days=6)), VARS, done) if t0 < now - pd.Timedelta(days=6) else pd.DataFrame()
        fc = pd.DataFrame()
        if t1 >= FC_FIRST:
            fvars = [f"{v}_previous_day{k}" for v in FC_VARS for k in FC_LEADS]
            fc = _hourly(PREV, *c, max(t0, FC_FIRST), min(t1, now), fvars, done)
        for _, r in gg.iterrows():
            row = {"event_id": r["event_id"], "lat": c[0], "lon": c[1]}
            row.update({f"wx_{k}": v for k, v in _window(obs, r["start_utc"], hours).items()})
            for k in FC_LEADS:
                row.update({f"fc_d{k}_{kk}": v for kk, v in _window(fc, r["start_utc"], hours, f"_previous_day{k}").items()})
            rows.append(row)
        if progress:
            progress(len(rows))
    return pd.DataFrame(rows)
