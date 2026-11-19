"""Fetch ALFRED vintages (as of each Wednesday D-1) of OBMMIC30YF and MORTGAGE30US. Read-only, cached per file."""
import time
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "vintages"
OUT.mkdir(exist_ok=True)
S = requests.Session()
S.headers["User-Agent"] = "research-readonly"
weds = pd.date_range("2022-12-28", "2026-11-18", freq="W-WED")
for sid in ["OBMMIC30YF", "MORTGAGE30US"]:
    for w in weds:
        f = OUT / f"{sid}_{w.date()}.csv"
        if f.exists() and f.stat().st_size > 1000:
            continue
        for k in range(4):
            try:
                r = S.get("https://alfred.stlouisfed.org/graph/alfredgraph.csv",
                          params={"id": sid, "vintage_date": str(w.date())}, timeout=60)
                if r.ok and r.text.startswith("observation_date"):
                    f.write_text(r.text)
                    break
            except requests.RequestException:
                pass
            time.sleep(3 * (k + 1))
        time.sleep(0.4)
print("files", len(list(OUT.glob("*.csv"))))
