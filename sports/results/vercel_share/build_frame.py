"""Build the edgelab frame for VS1/VS2 (prereg notes/vercel_share.md). DEV value days only (2026-09-11..2026-10-25); sealed days
are NOT fetched. Model probabilities use only source days <= D-1 (D-2 for P-persist). Read-only Kalshi GETs.
Out: sports/data/vercel_share/{vs_frame.parquet, vs_candles.json, panel.parquet}"""
import json, re, time, sys, os
from datetime import datetime
import numpy as np, pandas as pd, requests
sys.path.insert(0, os.path.dirname(__file__))
from source_panel import SER, load_panel
OUT = os.path.join(os.path.dirname(__file__), "../../data/vercel_share"); os.makedirs(OUT, exist_ok=True)
DEV = ("2026-09-11", "2026-10-25")
API = "https://api.elections.kalshi.com/trade-api/v2"
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
def ts(x): return int((pd.Timestamp(x) - EPOCH) // pd.Timedelta(seconds=1))
def lg(v): v = np.clip(v, 0.01, 99.99) / 100; return np.log(v / (1 - v))
def sg(x): return 100 / (1 + np.exp(-x))
def ar_fit(x, lag):
    y, z = x[lag:], x[:-lag]; ok = np.isfinite(y) & np.isfinite(z); y, z = y[ok], z[ok]
    if len(y) < 30: return None
    b = np.polyfit(z, y, 1)[0] if np.var(z) > 0 else 0.0; b = float(np.clip(b, 0, 1))
    a = y.mean() - b * z.mean(); return a, b, y - a - b * z
def probs(P, s, D, K):
    """p_m, p_persist, p_clim for P(V_D > K + 0.05) using days < D only."""
    d0 = D - pd.Timedelta(days=60); w = P.loc[(P.index >= d0) & (P.index <= D - pd.Timedelta(days=1)), s]
    w = w.reindex(pd.date_range(d0, D - pd.Timedelta(days=1)))
    x = lg(w.values); out = {}
    thr = K + 0.05
    f1 = ar_fit(x, 1); x1 = x[-1]
    out["p_m"] = np.nan if (f1 is None or not np.isfinite(x1)) else ((sg(f1[0] + f1[1] * x1 + f1[2]) > thr).sum() + .5) / (len(f1[2]) + 1)
    f2 = ar_fit(x, 2); x2 = x[-2]
    out["p_persist"] = np.nan if (f2 is None or not np.isfinite(x2)) else ((sg(f2[0] + f2[1] * x2 + f2[2]) > thr).sum() + .5) / (len(f2[2]) + 1)
    lv = w.dropna().values
    out["p_clim"] = np.nan if len(lv) < 30 else ((lv > thr).sum() + .5) / (len(lv) + 1)
    return out
def get(path, params):
    """-> (json, ok). 404 = genuinely absent (ok True, empty). Persistent 429/errors -> ok False (caller must not cache)."""
    for a in range(40):
        try:
            r = requests.get(API + path, params=params, timeout=30)
            if r.status_code == 429:
                try: w = float(r.headers.get("x-ratelimit-reset", 3))
                except ValueError: w = 3
                time.sleep(min(max(w, 0.5) + 0.3, 30)); continue
            if r.status_code == 404: return {}, True
            r.raise_for_status(); return r.json(), True
        except requests.RequestException:
            time.sleep(2 * (a + 1))
    return {}, False
if __name__ == "__main__":
    P = load_panel(); P = P[P.index < "2026-11-21"]; P.to_parquet(f"{OUT}/panel.parquet")
    rows = []
    for s in SER:
        for m in json.load(open(f"mk/{s}.json")):
            if m["status"] != "finalized" or m.get("result") not in ("yes", "no"): continue
            D = datetime.strptime(re.search(r"for (\w+ \d+, \d{4})", m["rules_primary"]).group(1), "%b %d, %Y")
            Ds = D.strftime("%Y-%m-%d")
            if not (DEV[0] <= Ds <= DEV[1]): continue
            D = pd.Timestamp(Ds); K = float(m["floor_strike"])
            r = dict(ticker=m["ticker"], event_ticker=m["event_ticker"], series=s, date=Ds, X=K,
                     T1=pd.Timestamp(Ds + " 05:00", tz="UTC"), close=pd.Timestamp(m["close_time"]), open_time=pd.Timestamp(m["open_time"]),
                     V=P.at[D, s] if D in P.index else np.nan, V1=P.at[D - pd.Timedelta(days=1), s],
                     result=m["result"], expiration_value=m["expiration_value"])
            r.update(probs(P, s, D, K)); rows.append(r)
    F = pd.DataFrame(rows); print("rows", len(F), F.groupby("series").size().to_dict(), flush=True)
    print(F[["p_m", "p_persist", "p_clim"]].describe().round(3).to_string(), flush=True)
    cfile = f"{OUT}/vs_candles.json"
    C = json.load(open(cfile)) if os.path.exists(cfile) else {}
    for i, r in enumerate(F.itertuples(index=False)):
        if r.ticker in C: continue
        p = {"start_ts": ts(r.open_time), "end_ts": ts(r.close) + 3600, "period_interval": 60}
        d, ok1 = get(f"/series/{r.series}/markets/{r.ticker}/candlesticks", p); cs = d.get("candlesticks") or []
        src = "series"; ok2 = True
        if not cs:
            d, ok2 = get(f"/historical/markets/{r.ticker}/candlesticks", p); cs = d.get("candlesticks") or []; src = "historical"
        if ok1 and ok2: C[r.ticker] = cs
        else: print("FETCH FAILED (not cached)", r.ticker, flush=True)
        time.sleep(0.3)
        if i % 50 == 0:
            json.dump(C, open(cfile, "w")); print(i, r.ticker, src, len(cs), flush=True)
    json.dump(C, open(cfile, "w"))
    F["n_candles"] = F.ticker.map(lambda t: len(C.get(t, [])))
    F.to_parquet(f"{OUT}/vs_frame.parquet"); print("saved", len(F), "with candles", (F.n_candles > 0).sum())
