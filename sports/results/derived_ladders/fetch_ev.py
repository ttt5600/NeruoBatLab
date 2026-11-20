"""For a Kalshi series: settled markets -> event start (ESPN, else ticker HHMM for MLB) -> T = start - 60 min ->
event-level 1-min candles over [T-6h, T] -> last quote <= T per market. Read-only, restartable.
usage: fetch_ev.py SERIES [--max-events N]   (N: most recent N events with a start; default all)"""
import sys, json, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget
import starts
OUT = Path(__file__).resolve().parent / "ev"; OUT.mkdir(exist_ok=True)
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
def es(ts): return int((ts - EPOCH) // pd.Timedelta(seconds=1))
def markets(s):
    f = OUT / f"markets_{s}.parquet"
    if f.exists(): return pd.read_parquet(f)
    rows = []
    for path, extra, src in (("/markets", {"status": "settled"}, "live"), ("/historical/markets", {}, "hist")):
        cur = None
        while True:
            p = {"series_ticker": s, "limit": 1000, **extra}
            if cur: p["cursor"] = cur
            j = kget(path, p) or {}
            for m in j.get("markets") or []:
                rows.append({k: m.get(k) for k in ["ticker", "event_ticker", "title", "yes_sub_title", "open_time", "close_time",
                             "occurrence_datetime", "expected_expiration_time", "floor_strike", "cap_strike", "strike_type",
                             "volume_fp", "result", "custom_strike"]} | {"src": src})
            cur = j.get("cursor")
            if not cur or not j.get("markets"): break
    d = pd.DataFrame(rows)
    d["custom_strike"] = d.custom_strike.astype(str)
    d = d.drop_duplicates("ticker")
    d = d[d.result.isin(["yes", "no"])]
    d.to_parquet(f); return d
def mlb_start(evt):
    suf = evt.split("-")[1]
    m = re.match(r"(\d{2}[A-Z]{3}\d{2})(\d{4})", suf)
    if not m: return pd.NaT
    return pd.Timestamp(f"{pd.to_datetime(m.group(1), format='%y%b%d'):%Y-%m-%d} {m.group(2)[:2]}:{m.group(2)[2:]}").tz_localize("America/New_York").tz_convert("UTC")
if __name__ == "__main__":
    s = sys.argv[1]
    mx = int(sys.argv[sys.argv.index("--max-events") + 1]) if "--max-events" in sys.argv else None
    m = markets(s)
    ev = m.groupby("event_ticker").agg(occ=("occurrence_datetime", "first"), src=("src", "first")).reset_index()
    ev["series"] = s
    sf = OUT / f"starts_{s}.parquet"
    if sf.exists(): st = pd.read_parquet(sf)
    else:
        if s.startswith("KXMLB"): ev["start_utc"] = ev.event_ticker.map(mlb_start); ev["how"] = "ticker"
        else:
            sm = starts.match(ev); ev["start_utc"] = ev.event_ticker.map(sm); ev["how"] = "espn"
        ev["occ_m3"] = pd.to_datetime(ev.occ, utc=True) - pd.Timedelta(hours=3)
        ev["off_min"] = (ev.occ_m3 - pd.to_datetime(ev.start_utc, utc=True)).dt.total_seconds() / 60
        st = ev; st.to_parquet(sf)
    agree = (st.off_min.dropna().abs() <= 5).mean() if st.off_min.notna().sum() >= 20 else 0.0
    if agree >= 0.95:   # occurrence_datetime - 3h validated against ESPN on this series -> use it where ESPN didn't match
        fill = st.start_utc.isna() & st.occ_m3.notna()
        st.loc[fill, "start_utc"] = st.loc[fill, "occ_m3"]; st.loc[fill, "how"] = "occ-3h"
    print("ESPN agreement |occ-3h - start| <= 5 min:", round(agree, 3), "| how:", st.how.value_counts().to_dict())
    ok = st.start_utc.notna()
    print(s, "events", len(st), "with start", ok.sum(), "| occ-3h minus start (min):", st.off_min.describe().round(1).to_dict(), flush=True)
    live_ev = set(m[m.src == "live"].event_ticker)   # historical (pre-cutoff) markets have no pre-game candles unless traded
    print("live-src events", len(live_ev & set(st.event_ticker)), "of", len(st), flush=True)
    st = st[ok & st.event_ticker.isin(live_ev)].sort_values("start_utc")
    if mx: st = st.tail(mx)
    cache = OUT / f"q_{s}.jsonl"
    done = {json.loads(l)["event"] for l in cache.open()} if cache.exists() else set()
    with cache.open("a") as f:
        for i, r in enumerate(st.itertuples()):
            if r.event_ticker in done: continue
            T = es(pd.Timestamp(r.start_utc)) - 3600
            j = kget(f"/series/{s}/events/{r.event_ticker}/candlesticks", {"start_ts": T - 6 * 3600, "end_ts": T, "period_interval": 1}) or {}
            q = {}
            for tk, cs in zip(j.get("market_tickers") or [], j.get("market_candlesticks") or []):
                cs = [c for c in cs if int(c["end_period_ts"]) <= T]
                if not cs: q[tk] = None; continue
                c = max(cs, key=lambda c: int(c["end_period_ts"]))
                g = lambda k: (c.get(k) or {}).get("close_dollars")
                q[tk] = {"ask": g("yes_ask"), "bid": g("yes_bid"), "end": int(c["end_period_ts"]), "oi": c.get("open_interest_fp"),
                         "vol6h": sum(float(x.get("volume_fp") or 0) for x in cs), "last": (c.get("price") or {}).get("close_dollars") or (c.get("price") or {}).get("previous_dollars")}
            f.write(json.dumps({"event": r.event_ticker, "T": T, "q": q}) + "\n")
            if i % 50 == 0: f.flush(); print(i, r.event_ticker, len(q), flush=True)
    print("done", s)
