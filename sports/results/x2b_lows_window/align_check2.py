"""Refetch with the EXACT original wx_expand params (no cache) and recompute its fcsafe min; compare to the parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import pandas as pd

import wx_expand as X
from sports import wx
from sports.http import fetch

st, tz, model = sys.argv[1], sys.argv[2], sys.argv[3]
df = pd.read_parquet(X.OUT, columns=["date", "station", "kind", f"fcsafe_{model}"]); df["date"] = pd.to_datetime(df.date)
a = df[(df.kind == "low") & (df.station == st)].drop_duplicates("date").set_index("date")
p = fetch(f"https://api.weather.gov/stations/{st}"); lon, lat = p["geometry"]["coordinates"][:2]
s = max(X.MODELS[model], str(a.index.min().date())); e = str(a.index.max().date())
d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model, "hourly": "temperature_2m_previous_day1",
                         "temperature_unit": "fahrenheit", "timezone": tz, "start_date": s, "end_date": e}, cache=False)
h = pd.DataFrame(d["hourly"]); t = pd.to_datetime(h.time); h["day"] = t.dt.normalize()
early = h[t.dt.hour <= 18].groupby("day")["temperature_2m_previous_day1"]
v = early.min().where(early.count() >= 15)
m = a.join(v.rename("refetch"))
m = m[m.index < X.SPLIT]
m["d"] = m[f"fcsafe_{model}"] - m.refetch
print(st, model, "dev days", len(m), "differ", int((m.d.abs() > 0.05).sum()))
print(m[m.d.abs() > 0.05].to_string())
