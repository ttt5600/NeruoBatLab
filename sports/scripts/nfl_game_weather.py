"""Hourly weather DURING each outdoor NFL game (kickoff .. +3h), 2006-2025.

    python scripts/nfl_game_weather.py      # -> data/nfl_game_wx.parquet

Source: Open-Meteo historical archive (ERA5 reanalysis, ~25 km grid), so these
are OBSERVED conditions, not what a bettor could have forecast. Anything built
on them is an ORACLE: if perfect knowledge of in-game rain/snow/wind cannot
beat the closing total, a forecast of it cannot either.
Coordinates are approximate stadium locations; at 25 km resolution that is ample.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sports import nfl
from sports.http import fetch

OUT = Path(__file__).resolve().parents[1] / "data" / "nfl_game_wx.parquet"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
VARS = "temperature_2m,precipitation,rain,snowfall,wind_speed_10m,wind_gusts_10m"
STADIUMS = {  # stadium_id -> (lat, lon)
    "ATL97": (33.755, -84.401), "BAL00": (39.278, -76.623), "BOS00": (42.091, -71.264),
    "BUF00": (42.774, -78.787), "CAR00": (35.226, -80.853), "CHI98": (41.862, -87.617),
    "CIN00": (39.095, -84.516), "CLE00": (41.506, -81.700), "DAL00": (32.748, -97.093),
    "DAL99": (32.840, -96.911), "DEN00": (39.744, -105.020), "FRA00": (50.069, 8.645),
    "GER00": (48.219, 11.625), "GNB00": (44.501, -88.062), "HOU00": (29.685, -95.411),
    "IND00": (39.760, -86.164), "JAX00": (30.324, -81.637), "KAN00": (39.049, -94.484),
    "LAX97": (33.864, -118.261), "LAX99": (34.014, -118.288), "LON00": (51.556, -0.280),
    "LON01": (51.456, -0.342), "LON02": (51.604, -0.066), "MEX00": (19.303, -99.150),
    "MIA00": (25.958, -80.239), "MIN98": (44.976, -93.225), "NAS00": (36.166, -86.771),
    "NYC00": (40.812, -74.077), "NYC01": (40.814, -74.074), "OAK00": (37.752, -122.201),
    "PHI00": (39.901, -75.168), "PHO00": (33.528, -112.263), "PIT00": (40.447, -80.016),
    "SAO00": (-23.545, -46.474), "SDG00": (32.783, -117.120), "SEA00": (47.595, -122.332),
    "SFO00": (37.714, -122.386), "SFO01": (37.403, -121.970), "TAM00": (27.976, -82.503),
    "WAS00": (38.908, -76.865),
}


def hourly(lat, lon, start, end):
    d = fetch(ARCHIVE, {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                        "hourly": VARS, "timezone": "UTC", "temperature_unit": "fahrenheit",
                        "wind_speed_unit": "mph", "precipitation_unit": "mm"}, cache=True)
    h = pd.DataFrame(d["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    return h.set_index("time")


def main() -> int:
    raw = nfl.raw_games()
    raw = raw[(raw.season >= 2006) & (raw.season <= 2025) & raw.roof.isin(["outdoors", "open"])]
    g = nfl.from_raw(raw).assign(stadium_id=raw["stadium_id"].to_numpy())
    rows = []
    for sid, grp in g.groupby("stadium_id"):
        lat, lon = STADIUMS[sid]
        for season, sg in grp.groupby("season"):  # one request per stadium-season
            t0, t1 = sg["start_utc"].min(), sg["start_utc"].max() + pd.Timedelta(hours=5)
            h = hourly(lat, lon, t0.date().isoformat(), t1.date().isoformat())
            for _, r in sg.iterrows():
                if pd.isna(r["start_utc"]):
                    continue
                k = r["start_utc"].floor("h")
                w = h.loc[k:k + pd.Timedelta(hours=3)]  # kickoff hour + 3 => ~game length
                if len(w) < 3:
                    continue
                rows.append({"game_id": r["game_id"], "stadium_id": sid,
                             "wx_precip_mm": w["precipitation"].sum(), "wx_rain_mm": w["rain"].sum(),
                             "wx_snow_cm": w["snowfall"].sum(),
                             "wx_wet_hours": int((w["precipitation"] >= 0.3).sum()),
                             "wx_wind_mean": w["wind_speed_10m"].mean(), "wx_wind_max": w["wind_speed_10m"].max(),
                             "wx_gust_max": w["wind_gusts_10m"].max(), "wx_temp_mean": w["temperature_2m"].mean()})
        print(sid, len(grp), flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    print(f"{len(out)} games -> {OUT}; rain games (>=1mm): {(out.wx_precip_mm >= 1).sum()}, "
          f"snow games (>=0.5cm): {(out.wx_snow_cm >= 0.5).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
