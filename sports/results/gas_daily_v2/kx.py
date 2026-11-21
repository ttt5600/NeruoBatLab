"""gas_daily_v2: read-only Kalshi GET helper with disk cache + throttle (no orders)."""
import hashlib, json, time
from pathlib import Path
import requests
API = "https://api.elections.kalshi.com/trade-api/v2"
CACHE = Path(__file__).resolve().parent / "cache"
GAP = 0.2; _last = [0.0]
def get(path, params=None, cache=True):
    params = params or {}
    key = hashlib.sha1((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if cache and f.exists():
        return json.loads(f.read_text())
    for a in range(8):
        w = GAP - (time.time() - _last[0])
        if w > 0: time.sleep(w)
        _last[0] = time.time()
        try:
            r = requests.get(API + path, params=params, timeout=30, headers={"User-Agent": "research-script/0.1"})
            if r.status_code == 429: time.sleep(min(5 * 2 ** a, 60)); continue
            if r.status_code >= 500: time.sleep(2 * (a + 1)); continue
            if r.status_code == 404: return {}
            r.raise_for_status(); d = r.json()
            if cache: f.write_text(json.dumps(d))
            return d
        except requests.RequestException:
            if a == 7: raise
            time.sleep(1.5 * (a + 1))
    return {}
def paged(path, params, key):
    out, cur = [], None
    while True:
        p = dict(params)
        if cur: p["cursor"] = cur
        d = get(path, p)
        out += d.get(key, [])
        cur = d.get("cursor")
        if not cur or not d.get(key): break
    return out
