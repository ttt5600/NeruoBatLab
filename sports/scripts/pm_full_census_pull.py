"""pm_full_census: pull every active Polymarket market (global Gamma events + PM US gateway). Read-only GETs."""
import json, sys, time, requests
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results/pm_full_census"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}

def get(url, params, tries=6):
    for a in range(tries):
        try:
            r = requests.get(url, params=params, headers=H, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** a); continue
            r.raise_for_status(); return r.json()
        except requests.RequestException as e:
            time.sleep(2 ** a)
    raise RuntimeError(url)

def gamma_events():
    out, cur = [], None
    while True:
        p = {"active": "true", "closed": "false", "limit": 500}
        if cur: p["after_cursor"] = cur
        d = get("https://gamma-api.polymarket.com/events/keyset", p)
        ev = d.get("events", [])
        out += ev
        cur = d.get("next_cursor")
        print("gamma", len(out), file=sys.stderr)
        if not ev or not cur: break
        time.sleep(0.12)
    return out

def pmus_markets():
    out, off = [], 0
    while True:
        d = get("https://gateway.polymarket.us/v1/markets", {"limit": 500, "offset": off, "closed": "false"})
        ms = d.get("markets", [])
        out += ms
        print("pmus", off, len(ms), file=sys.stderr)
        if len(ms) < 500: break
        off += 500; time.sleep(0.1)
    return out

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gamma":
        json.dump(gamma_events(), open(OUT / "gamma_events_active.json", "w"))
    else:
        json.dump(pmus_markets(), open(OUT / "pmus_markets_open.json", "w"))
