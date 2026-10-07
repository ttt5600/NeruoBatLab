"""Add multi-model day-ahead forecasts to data/wx_<CITY>.parquet.

    python scripts/add_forecasts.py

For each model, the lead-1-day forecast (issued ~24 h before each valid hour,
i.e. before the 10pm-day-before decision) of the local-day maximum, as
column fc_<model>. Missing where the model's archive does not reach.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sports import wx
from sports.http import fetch

DATA = Path(__file__).resolve().parents[1] / "data"
MODELS = {"ecmwf_ifs025": "2024-03-01", "icon_seamless": "2024-03-01", "gem_seamless": "2024-03-01",
          "ncep_hrrr_conus": "2024-03-01", "ncep_nbm_conus": "2025-03-01", "jma_seamless": "2022-01-01",
          "best_match": "2022-01-01"}


def model_max(model, lat, lon, tz, start, end):
    frames, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        done = e < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
        d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model,
                                 "hourly": "temperature_2m_previous_day1", "temperature_unit": "fahrenheit",
                                 "timezone": tz, "start_date": s.date().isoformat(),
                                 "end_date": e.date().isoformat()}, cache=done)
        h = pd.DataFrame(d.get("hourly", {}))
        if not h.empty:
            h["day"] = pd.to_datetime(h["time"]).dt.normalize()
            # a day counts only if most hours are present
            g = h.groupby("day")["temperature_2m_previous_day1"]
            frames.append(g.max().where(g.count() >= 20))
        s = e + pd.Timedelta(days=1)
    return pd.concat(frames).rename(f"fc_{model}") if frames else None


def main() -> int:
    inv = {k.replace("KXHIGH", ""): k for k in wx.CITIES}
    for f in sorted(DATA.glob("wx_*.parquet")):
        city = f.stem.split("_", 1)[1]
        station, tz = wx.CITIES[inv[city]]
        lat, lon = wx.station_coords(station)
        df = pd.read_parquet(f).drop(columns=[c for c in pd.read_parquet(f).columns
                                              if c.startswith("fc_") and c not in ("fc_max_lead1", "fc_max_lead2")])
        end = str(pd.to_datetime(df["date"]).max().date())
        for m, start in MODELS.items():
            s = model_max(m, lat, lon, tz, start, end)
            if s is not None:
                df = df.join(s, on="date")
        df.to_parquet(f, index=False)
        cov = {c: round(df[c].notna().mean(), 2) for c in df.columns if c.startswith("fc_")}
        print(city, cov, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
