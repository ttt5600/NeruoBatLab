"""pm_full_census T1 XV-US: PM US (exec at ask) vs global Polymarket (reference) on resolved non-game markets.
build: match + fetch histories -> results/pm_full_census/xvus_panel.parquet ; score: dev only (earliest 60% events)."""
import json, sys, time, requests, numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from pm_full_census_match import match
import polymarket_us as P
OUT = "results/pm_full_census/"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}

def gget(url, params):
    for a in range(6):
        try:
            r = requests.get(url, params=params, headers=H, timeout=60)
            if r.status_code == 200: return r.json()
        except requests.RequestException: pass
        time.sleep(2 ** a)
    return None

def build():
    us = pd.read_parquet(OUT + "pmus_closed_ns.parquet")
    us = us[us.win.isin(["yes", "no"])].copy()
    gev, gmk = [], []
    qs = us.question.dropna().unique()
    print("events to search", len(qs), file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor
    def srch(q):
        return gget("https://gamma-api.polymarket.com/public-search", {"q": q[:120], "events_status": "closed", "limit_per_type": 5, "keep_closed_markets": 1})
    with ThreadPoolExecutor(6) as ex:
        for i, d in enumerate(ex.map(srch, qs)):
            for e in (d or {}).get("events", []) or []:
                gev.append(dict(eid=e["id"], title=e.get("title"), end=e.get("endDate")))
                for m in e.get("markets") or []:
                    gmk.append(dict(eid=e["id"], mid=m.get("id"), q=m.get("question"), clob=m.get("clobTokenIds"),
                                    outcomePrices=m.get("outcomePrices"), closed=m.get("closed")))
            if i % 200 == 0: print(i, len(gev), file=sys.stderr)
    gev = pd.DataFrame(gev).drop_duplicates("eid"); gmk = pd.DataFrame(gmk).drop_duplicates("mid")
    gev.to_parquet(OUT + "xvus_gev.parquet"); gmk.to_parquet(OUT + "xvus_gmk.parquet")
    mm = match(us[["slug", "question", "title", "category", "sub", "mtype", "end", "win"]], gev, gmk)
    mm = mm[mm.jacc >= 0.6]
    mm.to_parquet(OUT + "xvus_match.parquet"); print("matched", len(mm), file=sys.stderr)
    hist(mm)

def one(r):
    rows = []
    try:
        tok = json.loads(r.g_clob)[0]
    except Exception:
        return rows
    try:
        ush = P.price_history(r.slug)
    except Exception:
        return rows
    gh = (gget("https://clob.polymarket.com/prices-history", {"market": tok, "interval": "all", "fidelity": 720}) or {}).get("history", [])
    if not ush or not gh: return rows
    gts = np.array([x["t"] for x in gh]); gp = np.array([float(x["p"]) for x in gh])
    dec = next((int(a) for a, b in zip(gts, gp) if b >= 0.97 or b <= 0.03), None)
    for x in ush:
        t = int(x["timestamp"]); k = np.searchsorted(gts, t, side="right") - 1
        k7 = np.searchsorted(gts, t - 7 * 86400, side="right") - 1
        rows.append(dict(slug=r.slug, question=r.question, category=r.category, end=r.end, win=r.win, t=t,
                         ya=float(x["longPrice"]), na=float(x["shortPrice"]), g=gp[k] if k >= 0 else np.nan,
                         g_t=gts[k] if k >= 0 else np.nan, g7=gp[k7] if k7 >= 0 else np.nan, g_dec=dec, g_q=r.g_q))
    return rows

def hist(mm=None):
    from concurrent.futures import ThreadPoolExecutor
    if mm is None:
        from pm_full_census_a3 import a3
        mm = a3(pd.read_parquet(OUT + "xvus_match.parquet"), pd.read_parquet(OUT + "xvus_gev.parquet"))
    rows = []
    with ThreadPoolExecutor(4) as ex:
        for i, rr in enumerate(ex.map(one, list(mm.itertuples()))):
            rows += rr
            if i % 500 == 0: print("hist", i, len(rows), file=sys.stderr)
    pd.DataFrame(rows).to_parquet(OUT + "xvus_panel.parquet"); print("panel", len(rows), file=sys.stderr)

if __name__ == "__main__":
    {"build": build, "hist": hist}[sys.argv[1]]()
