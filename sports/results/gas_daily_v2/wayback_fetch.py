"""Background: Wayback CDX for gasprices.aaa.com (2026-03..11), then raw captures (id_) at <= 1 req / 6 s, one per day.
Pinned IP because local DNS returns a non-IA address for web.archive.org. Extract AAA homepage rows. Read-only."""
import json, re, subprocess, sys, time
from pathlib import Path
OUT = Path(__file__).resolve().parent / "wayback"
IP = "207.241.237.3"
def curl(url, dest):
    r = subprocess.run(["curl", "-s", "-m", "60", "--resolve", f"web.archive.org:443:{IP}", "-A", "research-script/0.1 (gas audit)",
                        "-o", str(dest), "-w", "%{http_code}", url], capture_output=True, text=True)
    return r.stdout.strip()
cdx = OUT / "cdx.txt"
for attempt in range(40):
    code = curl("https://web.archive.org/cdx/search/cdx?url=gasprices.aaa.com/&from=20260301&to=20261121&fl=timestamp,statuscode,digest&filter=statuscode:200&collapse=timestamp:8", cdx)
    print(time.strftime("%H:%M:%S"), "cdx", code, flush=True)
    if code == "200" and cdx.stat().st_size > 0 and "<html" not in cdx.read_text()[:100].lower():
        break
    time.sleep(180)
else:
    print("gave up"); sys.exit(1)
rows = [l.split() for l in cdx.read_text().splitlines() if l.strip()]
print("captures (1/day)", len(rows), flush=True)
for ts, st, dg in rows:
    f = OUT / f"{ts}.html"
    if f.exists() and f.stat().st_size > 20000: continue
    for a in range(5):
        code = curl(f"https://web.archive.org/web/{ts}id_/https://gasprices.aaa.com/", f)
        if code == "200": break
        print(ts, "code", code, flush=True); time.sleep(30 * (a + 1))
    time.sleep(6)
print("done", flush=True)
