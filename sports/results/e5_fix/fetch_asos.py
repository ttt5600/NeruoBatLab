"""IEM ASOS archive (routine + special METARs, tmpf) for every station in the Kalshi proxy and FX pick sets.
Output: results/e5_fix/asos/<STN>.csv (raw), results/e5_fix/asos_obs.parquet (station, valid UTC, tmpf)."""
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

OUT = Path(__file__).parent / "asos"
OUT.mkdir(exist_ok=True)
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}&data=tmpf&year1={y1}&month1={m1}&day1={d1}"
       "&year2={y2}&month2={m2}&day2={d2}&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty&trace=empty"
       "&report_type=3&report_type=4")

KALSHI = ["NYC", "MDW", "MIA", "AUS", "DEN", "LAX", "PHL", "ATL", "BOS", "DFW", "DCA", "EWR", "HOU", "LAS", "MSP",
          "MSY", "OKC", "PHX", "SAN", "SAT", "SDF", "SEA", "SFO", "TTN"]
FX = ["LGA", "BNA", "CLT", "DTW", "JAX", "BKF"]


def get(stn, start, end):
    f = OUT / f"{stn}_{start}_{end}.csv"
    if f.exists() and f.stat().st_size > 1000:
        return f
    a, b = pd.Timestamp(start), pd.Timestamp(end)
    u = URL.format(s=stn, y1=a.year, m1=a.month, d1=a.day, y2=b.year, m2=b.month, d2=b.day)
    for i in range(5):
        try:
            data = urllib.request.urlopen(u, timeout=300).read()
            f.write_bytes(data)
            return f
        except Exception as e:  # IEM rate limits
            print(stn, "retry", i, e, file=sys.stderr)
            time.sleep(10 * (i + 1))
    raise RuntimeError(stn)


jobs = [(s, "2024-02-25", "2026-11-12") for s in KALSHI] + [(s, "2026-10-25", "2026-11-12") for s in FX]
with ThreadPoolExecutor(3) as ex:
    files = list(ex.map(lambda j: get(*j), jobs))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d = d.dropna(subset=["tmpf"])
    dfs.append(d)
o = pd.concat(dfs).rename(columns={"station": "stn"})
o["valid"] = pd.to_datetime(o["valid"], utc=True)
o = o.drop_duplicates(["stn", "valid"]).sort_values(["stn", "valid"])
o.to_parquet(Path(__file__).parent / "asos_obs.parquet")
print(o.groupby("stn").valid.agg(["min", "max", "count"]).to_string())
