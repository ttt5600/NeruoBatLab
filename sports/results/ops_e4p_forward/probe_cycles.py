"""Probe which NBS runtimes the IEM MOS archive holds per month (default KNYC). Read-only GETs."""
import csv, io, time, sys
from collections import Counter
import requests
st = sys.argv[1] if len(sys.argv) > 1 else "KNYC"
months = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "2025-06", "2025-09", "2025-12", "2026-02", "2026-03", "2026-04", "2026-06", "2026-08", "2026-10", "2026-11"]
for m in months:
    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py",
                     params={"station": st, "model": "NBS", "sts": f"{m}-01T00:00Z", "ets": f"{m}-28T23:59Z", "format": "csv"},
                     timeout=120)
    rows = list(csv.DictReader(io.StringIO(r.text)))
    rts = sorted({x["runtime"] for x in rows})
    print(m, len(rows), dict(Counter(x[11:13] for x in rts)), rts[:1], rts[-1:], flush=True)
    time.sleep(2)
