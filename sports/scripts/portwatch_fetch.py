"""IMF PortWatch Daily_Chokepoints_Data (public ArcGIS) -> data/dl/portwatch_daily.parquet.
The layer is overwritten weekly (lastEditDate Tuesdays ~14:13Z); it has no vintages, so the stored values are the
CURRENT revision. Retries the shared 429 quota."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

D = Path(__file__).resolve().parents[1] / "data" / "dl"
URL = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Chokepoints_Data/FeatureServer/0")
PORTS = {"chokepoint1": "suez", "chokepoint2": "panama", "chokepoint4": "babelmandeb", "chokepoint6": "hormuz"}


def q(params: dict) -> dict:
    for a in range(60):
        r = requests.get(URL + "/query", params={**params, "f": "json"}, timeout=60)
        j = r.json()
        if "error" in j:
            print("retry", j["error"].get("code"), flush=True)
            time.sleep(20)
            continue
        return j
    raise RuntimeError("quota")


if __name__ == "__main__":
    meta = requests.get(URL, params={"f": "json"}, timeout=60).json()
    rows = []
    for pid in PORTS:
        off = 0
        while True:
            j = q({"where": f"portid='{pid}'", "outFields": "*", "orderByFields": "date ASC",
                   "resultOffset": off, "resultRecordCount": 2000})
            fs = [f["attributes"] for f in j.get("features", [])]
            rows += fs
            print(pid, off, len(fs), flush=True)
            if len(fs) < 2000 and not j.get("exceededTransferLimit"):
                break
            off += len(fs)
    df = pd.DataFrame(rows)
    df.to_parquet(D / "portwatch_daily.parquet")
    (D / "portwatch_meta.json").write_text(json.dumps({"editingInfo": meta.get("editingInfo"),
                                                       "fetched_utc": pd.Timestamp.utcnow().isoformat()}))
    print(df.shape, df.columns.tolist())
