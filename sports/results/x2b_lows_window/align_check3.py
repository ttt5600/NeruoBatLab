"""Same hours, three requests: long-range local tz (original), short-range local tz, short-range GMT."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

from sports import wx
from sports.http import fetch

lat, lon, model, tz, day = 30.18311, -97.67989, "gfs_seamless", "America/Chicago", sys.argv[1]
p = fetch("https://api.weather.gov/stations/KAUS"); lon, lat = p["geometry"]["coordinates"][:2]
base = {"latitude": lat, "longitude": lon, "models": model, "hourly": "temperature_2m_previous_day1",
        "temperature_unit": "fahrenheit"}
out = {}
for name, prm in {"long_local": {"timezone": tz, "start_date": "2025-12-13", "end_date": "2026-10-18"},
                  "short_local": {"timezone": tz, "start_date": "2025-12-12", "end_date": "2026-03-01"},
                  "short_gmt": {"timezone": "GMT", "start_date": "2025-12-12", "end_date": "2026-03-01"}}.items():
    h = pd.DataFrame(fetch(wx.PREV_RUNS, {**base, **prm}, cache=False)["hourly"])
    t = pd.to_datetime(h.time)
    if prm["timezone"] == "GMT":
        t = t.dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    out[name] = pd.Series(h.temperature_2m_previous_day1.values, index=t)
m = pd.DataFrame(out)
print(m.loc[str(day)].to_string())
