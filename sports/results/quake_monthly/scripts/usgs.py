"""Pull USGS ComCat M>=6.0 worldwide 1990-01-01..now (csv). Read-only."""
import io, sys
from pathlib import Path
import pandas as pd, requests

OUT = Path(__file__).resolve().parents[1]
frames = []
for y0 in range(1990, 2027, 6):
    y1 = min(y0 + 6, 2027)
    r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query",
                     params={"format": "csv", "starttime": f"{y0}-01-01", "endtime": f"{y1}-01-01",
                             "minmagnitude": 6.0, "eventtype": "earthquake", "orderby": "time-asc", "limit": 20000},
                     timeout=120)
    r.raise_for_status()
    frames.append(pd.read_csv(io.StringIO(r.text)))
df = pd.concat(frames).drop_duplicates("id")
df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
df.to_parquet(OUT / "usgs_m6.parquet")
print(len(df), df.time.min(), df.time.max())
print(df[df.time >= "2026-10-01"][["time", "mag", "magType", "status", "place"]].to_string())
