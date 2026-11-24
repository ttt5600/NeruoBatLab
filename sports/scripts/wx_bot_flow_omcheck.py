"""wx_bot_flow replication check: Open-Meteo ensemble API (what the bots read) vs our NOAA-GEFS point extraction, same run.
    /tmp/wx_venv/bin/python scripts/wx_bot_flow_omcheck.py   -> results/wx_bot_flow/omcheck.json
"""
import concurrent.futures as cf, datetime as dt, json, sys
from pathlib import Path
import numpy as np, requests
sys.path.insert(0, str(Path(__file__).parent))
import wx_bot_flow_gefs as G

OUT = Path(__file__).resolve().parents[1] / "results" / "wx_bot_flow"
cities = {"NY": (40.7128, -74.0060), "CHI": (41.8781, -87.6298), "MIA": (25.7617, -80.1918),
          "LAX": (34.0522, -118.2437), "DEN": (39.7392, -104.9903)}
D = dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
lj = requests.get("https://openmeteo.s3.amazonaws.com/data_spatial/ncep_gefs025/latest.json").json()
run = dt.datetime.fromisoformat(lj["reference_time"].replace("Z", "")).replace(tzinfo=None)
res = {"om_reference_time": lj["reference_time"], "target": D.isoformat(), "checked_at": dt.datetime.utcnow().isoformat()}
mems = ["gec00"] + [f"gep{i:02d}" for i in range(1, 31)]
v0 = dt.datetime.combine(D, dt.time(0))
tasks = [(run, m, int((v0 + dt.timedelta(hours=vh) - run).total_seconds() // 3600)) for m in mems for vh in range(0, 25, 3)]
F = {}
with cf.ThreadPoolExecutor(16) as ex:
    for t, r in zip(tasks, ex.map(lambda t: G.one(*t), tasks)):
        for x in r:
            F[(t[1], t[2], x[4])] = (x[5] - 273.15) * 9 / 5 + 32
for c, (la, lo) in cities.items():
    j = requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params=dict(
        latitude=la, longitude=lo, daily="temperature_2m_max", hourly="temperature_2m", temperature_unit="fahrenheit",
        start_date=D.isoformat(), end_date=D.isoformat(), models="gfs_seamless")).json()
    hh = j["hourly"]; times = hh["time"]
    dkey = lambda i: "temperature_2m_max" if i == 0 else f"temperature_2m_max_member{i:02d}"
    hkey = lambda i: "temperature_2m" if i == 0 else f"temperature_2m_member{i:02d}"
    om_max = np.array([j["daily"][dkey(i)][0] for i in range(31)])
    fh = [int((v0 + dt.timedelta(hours=vh) - run).total_seconds() // 3600) for vh in range(0, 25, 3)]
    ours = np.array([[F[(m, f, f"{c}_bot")] for f in fh] for m in mems])
    om3 = np.array([[hh[hkey(i)][times.index((v0 + dt.timedelta(hours=vh)).strftime("%Y-%m-%dT%H:%M"))] for vh in range(0, 24, 3)]
                    for i in range(31)])
    om_hmax = np.array([max(hh[hkey(i)]) for i in range(31)])
    d3 = ours[:, :8] - om3
    our_max, our_max8 = ours.max(1), ours[:, :8].max(1)
    res[c] = dict(daily_max_diff_mean=round(float((our_max - om_max).mean()), 2), daily_max_diff_sd=round(float((our_max - om_max).std()), 2),
                  daily_max_absdiff_max=round(float(np.abs(our_max - om_max).max()), 2),
                  daily_max_00_21_diff_mean=round(float((our_max8 - om_max).mean()), 2),
                  om_daily_minus_om_hourlymax=round(float((om_max - om_hmax).mean()), 2),
                  three_hourly_diff_mean=round(float(d3.mean()), 2), three_hourly_diff_sd=round(float(d3.std()), 2),
                  om_head=om_max[:4].round(1).tolist(), ours_head=our_max[:4].round(1).tolist())
    print(c, res[c], flush=True)
json.dump(res, open(OUT / "omcheck.json", "w"), indent=1)
