"""Download ERCOT NP6-785-ER yearly RTM hub SPP zips (public MIS) for DEV years + current year (forward only).
Sealed 2024-2025 zips are NOT downloaded. Raw zips go to raw/ (untrusted data dir); parsed hub frame -> hubs.parquet."""
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent
RAW = OUT / "raw"
RAW.mkdir(exist_ok=True)
LIST = "https://www.ercot.com/misapp/servlets/IceDocListJsonWS?reportTypeId=13061"
DL = "https://www.ercot.com/misdownload/servlets/mirDownload?doclookupId={}"
YEARS = [int(y) for y in sys.argv[1:]] or list(range(2011, 2024)) + [2026]
HUBS = {"HB_HOUSTON", "HB_NORTH", "HB_WEST"}

docs = requests.get(LIST, timeout=60).json()["ListDocsByRptTypeRes"]["DocumentList"]
byyear = {}
for d in docs:
    d = d["Document"]
    y = int(d["FriendlyName"].split("_")[-1])
    byyear[y] = d["DocID"]
print("years available", sorted(byyear))
frames = []
for y in YEARS:
    if y in (2024, 2025):
        continue  # sealed
    f = RAW / f"RTMLZHBSPP_{y}.zip"
    if not f.exists():
        r = requests.get(DL.format(byyear[y]), timeout=300)
        r.raise_for_status()
        f.write_bytes(r.content)
    with zipfile.ZipFile(f) as z:
        name = z.namelist()[0]
        xl = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=None)
    for sh, df in xl.items():
        df = df[df["Settlement Point Name"].isin(HUBS)]
        frames.append(df)
    print(y, name, sum(len(v) for v in xl.values()))
h = pd.concat(frames, ignore_index=True)
h.to_parquet(OUT / "hubs.parquet")
print(h.head(), h.shape)
