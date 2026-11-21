"""Fetch Wayback captures of vercel.com/ai-gateway/leaderboards/labs (raw id_) since 2026-09-20 and extract the lab daily
shares shown at capture time -> wayback/labs_panel.csv (cap_ts, day, lab, metric, share)."""
import json, time, os, sys, requests
sys.path.insert(0, os.path.dirname(__file__))
from parse_rsc import rsc_text, day_objects
d = json.load(open("wayback/cdx_leaderboards.json"))[1:]
caps = sorted({r[0] for r in d if r[1].split("?")[0].rstrip("/") == "https://vercel.com/ai-gateway/leaderboards/labs" and r[2] == "200"
               and "?" not in r[1] and r[0] >= "20260920"})
print(len(caps), caps)
rows = []
for c in caps:
    fn = f"wayback/labs_{c}.html"
    if not os.path.exists(fn):
        for a in range(4):
            try:
                r = requests.get(f"https://web.archive.org/web/{c}id_/https://vercel.com/ai-gateway/leaderboards/labs", timeout=90)
                if r.status_code == 200: open(fn, "wb").write(r.content); break
            except requests.RequestException: pass
            time.sleep(5 * (a + 1))
    if not os.path.exists(fn): print("miss", c); continue
    raw = open(fn, "rb").read()
    try:
        import gzip; raw = gzip.decompress(raw)
    except Exception: pass
    txt = rsc_text(raw.decode("utf-8", "ignore"))
    n = 0
    for o in day_objects(txt):
        if "metric" not in o: continue
        for k, v in o["chef_values"]:
            rows.append((c, o["day"][:10], k, o["metric"], v)); n += 1
    print(c, n, flush=True)
import pandas as pd
pd.DataFrame(rows, columns=["cap", "day", "lab", "metric", "share"]).to_csv("wayback/labs_panel.csv", index=False)
