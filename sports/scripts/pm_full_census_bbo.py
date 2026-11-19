"""pm_full_census: PM US per-market bbo (sharesTraded, openInterest, bid/ask, depth). All non-sports + 1/8 random sports. Read-only."""
import json, random, sys, time, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results/pm_full_census"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}
ms = json.load(open(OUT / "pmus_markets_open.json"))
random.seed(7)
pick = [m for m in ms if m.get("category") != "sports" or random.random() < 0.125]
print(len(pick), file=sys.stderr)
S = requests.Session()
def one(m):
    for a in range(5):
        try:
            r = S.get(f"https://gateway.polymarket.us/v1/markets/{m['slug']}/bbo", headers=H, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** a); continue
            if r.status_code != 200: return {"slug": m["slug"], "err": r.status_code}
            d = r.json().get("marketData", {})
            d["slug"] = m["slug"]; d["sampled_w"] = 1 if m.get("category") != "sports" else 8
            return d
        except requests.RequestException:
            time.sleep(2 ** a)
    return {"slug": m["slug"], "err": "fail"}
res = []
t0 = time.time()
with ThreadPoolExecutor(4) as ex:
    for i, d in enumerate(ex.map(one, pick)):
        res.append(d)
        if i % 2000 == 0: print(i, round(time.time() - t0), file=sys.stderr)
json.dump(res, open(OUT / "pmus_bbo.json", "w"))
