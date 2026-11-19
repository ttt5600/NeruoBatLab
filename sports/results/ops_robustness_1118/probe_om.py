"""Time open-meteo E4 forecast requests per city (connect / TLS / first byte / total) to see which phase hangs."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sports import wx  # noqa: E402

MODELS = "gfs_seamless,ecmwf_ifs025,icon_seamless,gem_seamless,ncep_nbm_conus"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
for series, (st, tz) in list(wx.CITIES.items()):
    lat, lon = wx.station_coords(st)
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&models={MODELS}"
           f"&hourly=temperature_2m&temperature_unit=fahrenheit&timezone={tz}&forecast_days=3")
    for k in range(n):
        out = subprocess.run(["curl", "-s", "-o", "/dev/null", "-m", "60", "-w",
                              "%{http_code} conn=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} "
                              "tot=%{time_total} ip=%{remote_ip}", url], capture_output=True, text=True).stdout
        print(series, lat, lon, out, flush=True)
