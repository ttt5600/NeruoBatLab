"""wx_bot_flow: pull GEFS 0.25 deg TMP 2 m member values at bot city-centre + settlement-station points from NOAA AWS
(noaa-gefs-pds, as-issued, no lookahead: a run's files are the forecast as published). Read-only.

    /tmp/wx_venv/bin/python scripts/wx_bot_flow_gefs.py 2026-10-08 2026-11-23   # target dates D (runs 06Z,12Z of D-1)
Output: results/wx_bot_flow/gefs_points.parquet (run_init, member, fhr, valid, point, tK)
"""
import concurrent.futures as cf
import datetime as dt
import sys
import time
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parents[1] / "results" / "wx_bot_flow"
BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# bot city-centre coords (suislanchez/AwpDemon CITY_CONFIG; AUS/PHIL city centres) and Kalshi settlement stations
POINTS = {
    "NY_bot": (40.7128, -74.0060), "CHI_bot": (41.8781, -87.6298), "MIA_bot": (25.7617, -80.1918),
    "LAX_bot": (34.0522, -118.2437), "DEN_bot": (39.7392, -104.9903), "AUS_bot": (30.2672, -97.7431),
    "PHIL_bot": (39.9526, -75.1652),
    "NY_stn": (40.7789, -73.9692), "CHI_stn": (41.7868, -87.7522), "MIA_stn": (25.7959, -80.2870),
    "LAX_stn": (33.9425, -118.4081), "DEN_stn": (39.8561, -104.6737), "AUS_stn": (30.1945, -97.6699),
    "PHIL_stn": (39.8729, -75.2408),
}
IDX = {k: (int(round((90 - la) / 0.25)), int(round(((lo + 360) % 360) / 0.25))) for k, (la, lo) in POINTS.items()}
S = requests.Session()


def one(init, mem, fhr):
    ymd, hh = init.strftime("%Y%m%d"), init.strftime("%H")
    fn = f"{BASE}/gefs.{ymd}/{hh}/atmos/pgrb2sp25/{mem}.t{hh}z.pgrb2s.0p25.f{fhr:03d}"
    for attempt in range(4):
        try:
            idx = S.get(fn + ".idx", timeout=30)
            if idx.status_code == 404:
                return [("missing", init, mem, fhr)]
            lines = idx.text.strip().split("\n")
            for i, ln in enumerate(lines):
                if ":TMP:2 m above ground:" in ln:
                    a = int(ln.split(":")[1]); b = int(lines[i + 1].split(":")[1]) - 1
                    break
            r = S.get(fn, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
            g = eccodes.codes_new_from_message(r.content)
            v = eccodes.codes_get_values(g).reshape(721, 1440)
            eccodes.codes_release(g)
            valid = init + dt.timedelta(hours=fhr)
            return [(init, mem, fhr, valid, k, float(v[i, j])) for k, (i, j) in IDX.items()]
        except Exception as e:  # retry
            err = e
            time.sleep(2 + attempt * 3)
    return [("error", init, mem, fhr, str(err))]


def main():
    d0, d1 = [dt.date.fromisoformat(x) for x in sys.argv[1:3]]
    mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
    tasks = []
    D = d0
    while D <= d1:
        for rh in (6, 12):
            init = dt.datetime.combine(D - dt.timedelta(days=1), dt.time(rh))
            v0 = dt.datetime.combine(D, dt.time(0))
            for vh in range(0, 34, 3):  # valid 00Z D .. 09Z D+1
                fhr = int((v0 + dt.timedelta(hours=vh) - init).total_seconds() // 3600)
                for m in mems:
                    tasks.append((init, m, fhr))
        D += dt.timedelta(days=1)
    tasks = sorted(set(tasks))
    print(f"{len(tasks)} fields", flush=True)
    rows, bad, t0 = [], [], time.time()
    with cf.ThreadPoolExecutor(24) as ex:
        for n, res in enumerate(ex.map(lambda t: one(*t), tasks)):
            if res and isinstance(res[0][0], str):
                bad.append(res[0])
            else:
                rows.extend(res)
            if n % 1000 == 0:
                print(f"{n}/{len(tasks)} {time.time()-t0:.0f}s bad={len(bad)}", flush=True)
    df = pd.DataFrame(rows, columns=["run_init", "member", "fhr", "valid", "point", "tK"])
    df.to_parquet(OUT / "gefs_points.parquet")
    print("done", df.shape, "bad", len(bad), bad[:5], f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
