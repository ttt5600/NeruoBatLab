"""Refresh IEM ASOS obs for the FX stations, 2026-11-08..11-13 UTC, and merge into asos_obs.parquet."""
import io
import time
import urllib.request
from pathlib import Path

import pandas as pd

from fx_f2 import TZ

OUTD = Path(__file__).parent
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={s}&data=tmpf&year1=2026&month1=11&day1=8"
       "&year2=2026&month2=11&day2=13&tz=Etc/UTC&format=onlycomma&latlon=no&missing=empty&trace=empty"
       "&report_type=3&report_type=4")
dfs = []
for s in TZ:
    for i in range(5):
        try:
            dfs.append(pd.read_csv(io.BytesIO(urllib.request.urlopen(URL.format(s=s), timeout=120).read())))
            break
        except Exception as e:
            print(s, e)
            time.sleep(10 * (i + 1))
    time.sleep(1)
n = pd.concat(dfs).dropna(subset=["tmpf"]).rename(columns={"station": "stn"})
n["valid"] = pd.to_datetime(n["valid"], utc=True)
o = pd.read_parquet(OUTD / "asos_obs.parquet")
o = pd.concat([o, n]).drop_duplicates(["stn", "valid"]).sort_values(["stn", "valid"])
o.to_parquet(OUTD / "asos_obs.parquet")
print(o.valid.max(), len(o))
