"""Step-1 diagnosis for X2 lows (DEV only, dates < 2026-03-01). Throwaway analysis script."""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd

import wx_expand as X
from wx_lockin import ob_f, IEM, std_offset
from sports.http import fetch

OUTD = Path(__file__).parent
df = pd.read_parquet(X.OUT); df["date"] = pd.to_datetime(df.date)
df = df[(df.date < X.SPLIT) & (df.kind == "low")]
assert df.date.max() < X.SPLIT
t = X.picks(df)
print("picks", len(t), "units", round(t.pnl.sum(), 2), "roi", round(t.pnl.mean(), 4), "losses", int((t.pnl < 0).sum()))
TZ = {"KAUS": "America/Chicago", "KMDW": "America/Chicago", "KDEN": "America/Denver", "KLAX": "America/Los_Angeles",
      "KMIA": "America/New_York", "KNYC": "America/New_York", "KPHL": "America/New_York"}
obs = []
for st in TZ:
    raw = fetch(IEM, {"station": st[1:], "data": "metar", "year1": 2025, "month1": 12, "day1": 10,
                      "year2": 2026, "month2": 3, "day2": 1, "tz": "Etc/UTC", "format": "onlycomma",
                      "latlon": "no", "report_type": [3, 4]}, binary=True, cache=True)
    o = pd.read_csv(io.BytesIO(raw)).drop_duplicates(["valid", "metar"])
    o["valid"] = pd.to_datetime(o.valid).dt.tz_localize("UTC")
    o["tf"] = o.metar.map(lambda m: ob_f(m)[0]); o["station"] = st
    obs.append(o.dropna(subset=["tf"])[["station", "valid", "tf"]])
obs = pd.concat(obs)
obs = obs[obs.valid < pd.Timestamp("2026-03-01 12:00", tz="UTC")]
obs.to_parquet(OUTD / "obs_dev.parquet")
rows = []
for (st, ev), g in df.drop_duplicates("event_ticker").groupby(["station", "event_ticker"]):
    r = g.iloc[0]; tz = TZ[st]
    lst0 = pd.Timestamp(r.date).tz_localize("UTC") - std_offset(tz, r.date)
    o = obs[(obs.station == st) & (obs.valid >= lst0) & (obs.valid < lst0 + pd.Timedelta(days=1))]
    if o.empty:
        continue
    h = ((o.valid - lst0).dt.total_seconds() / 3600).to_numpy()
    tf = o.tf.to_numpy(); mn = tf.min()
    rows.append(dict(station=st, event_ticker=ev, date=r.date, settle=r.expiration_value, obs_min=mn,
                     hr_first=h[tf == mn].min(), hr_last=h[tf == mn].max(),
                     min_0_18=tf[h < 19].min(), min_19_24=tf[h >= 19].min() if (h >= 19).any() else np.nan, nobs=len(o)))
day = pd.DataFrame(rows)
day.to_parquet(OUTD / "day_dev.parquet")
t.to_parquet(OUTD / "x2_dev_picks.parquet")
print("station-days", len(day), "obs_min==settle", round((day.obs_min == day.settle).mean(), 3))
