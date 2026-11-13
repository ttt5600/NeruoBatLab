"""LX2 long-dated tail-NO carry on Kalshi (prereg: notes/lx2_tail_no_carry.md, written before any price/outcome).

    python scripts/lx2_tail_no_carry.py markets   # settled market metadata from /historical/markets (cached)
    python scripts/lx2_tail_no_carry.py prices    # daily candle quote at S-90 / S-180 for eligible DEV markets
    python scripts/lx2_tail_no_carry.py dev       # score DEV trials A (T-90) and B (T-180)

READ-ONLY public GETs. No orders. SEALED (settlement >= 2025-10-01) is never priced here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import sports.http as H
from sports.http import fetch

H.MIN_INTERVAL = 0.12


KEEP = ["ticker", "event_ticker", "market_type", "open_time", "close_time", "expected_expiration_time",
        "latest_expiration_time", "settlement_ts", "can_close_early", "result", "status", "title", "yes_sub_title",
        "volume_fp"]


def fetch_many(urls, workers=6, cache=True):
    """sports.http.fetch_many, but a URL that still fails after fetch's retries is retried (slower) not fatal."""
    import time

    def one(u):
        p = H.CACHE_DIR / (H._key(u, None) + ".json")
        if p.exists():  # read pages cached by the first run, but never write new ones when cache=False
            try:
                return json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                pass
        for k in range(6):
            try:
                return fetch(u, cache=cache)
            except Exception as e:  # noqa: BLE001  (429 storms / blips)
                if k == 5:
                    print("GAVE UP", u, e, flush=True)
                    return {"_failed": True}
                time.sleep(20 * (k + 1))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(one, urls))
API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "results" / "lx2_tail_no_carry"
DATA = ROOT / "data" / "lx2_tail_no_carry"
SKIP_FREQ = {"hourly", "fifteen_min", "daily", "weekly"}
CUT = pd.Timestamp("2025-10-01", tz="UTC")
HORIZONS = {"A": 90, "B": 180}
APY = 0.035
CAT = {"Politics": "politics", "Elections": "politics", "World": "politics",
       "Economics": "economics", "Financials": "economics", "Companies": "economics", "Commodities": "economics",
       "Crypto": "economics", "Climate and Weather": "climate", "Sports": "sports"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def _sec(t):  # unit-safe epoch seconds (traps: never astype(int)//1e9)
    return ((t - EPOCH) // pd.Timedelta(seconds=1)).astype("Int64")


def series_list():
    ser, cur = [], None
    while True:
        p = {"limit": 1000, **({"cursor": cur} if cur else {})}
        d = fetch(f"{API}/series", p, cache=False)
        ser += d.get("series", [])
        cur = d.get("cursor")
        if not cur or not d.get("series"):
            return ser


def cmd_markets():
    DATA.mkdir(parents=True, exist_ok=True)
    ser = [s for s in series_list() if s.get("frequency") not in SKIP_FREQ]
    print("series kept", len(ser), flush=True)
    meta = {s["ticker"]: s for s in ser}
    base = f"{API}/historical/markets"
    pending = {s: None for s in meta}
    rows, rnd, dry, failed = [], 0, {}, []
    # First run cached every page and hit 20.7M rows / OOM (a few custom series list millions of short markets).
    # Now: no disk cache, keep only metadata-long-lived binary markets at ingest, and stop a series after 5
    # consecutive pages with none kept (metadata-only rule; no price/outcome involved).
    while pending:
        urls = [base + "?" + urlencode({"series_ticker": s, "limit": 1000, **({"cursor": c} if c else {})})
                for s, c in pending.items()]
        res = fetch_many(urls, cache=False)
        nxt = {}
        for (s, _), d in zip(pending.items(), res):
            if d.get("_failed"):
                failed.append(s)
                continue
            ms = d.get("markets", []) or []
            kept = []
            for m in ms:
                ee, ot = _ts(m.get("expected_expiration_time")), _ts(m.get("open_time"))
                if m.get("market_type") == "binary" and pd.notna(ee) and pd.notna(ot) and ot <= ee - pd.Timedelta(days=90):
                    kept.append({k: m.get(k) for k in KEEP} | {"series": s})
            rows += kept
            dry[s] = 0 if kept else dry.get(s, 0) + 1
            if d.get("cursor") and ms and dry[s] < 5:
                nxt[s] = d["cursor"]
        rnd += 1
        print(f"round {rnd}: {len(pending)} reqs, rows {len(rows)}", flush=True)
        pending = nxt
    print("failed series", len(failed), failed[:20])
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df["category"] = df.series.map(lambda s: meta[s].get("category"))
    df["frequency"] = df.series.map(lambda s: meta[s].get("frequency"))
    df["fee_type"] = df.series.map(lambda s: meta[s].get("fee_type"))
    df["fee_multiplier"] = df.series.map(lambda s: meta[s].get("fee_multiplier"))
    df.to_parquet(DATA / "markets.parquet", index=False)
    print("->", DATA / "markets.parquet", len(df))


def eligible(df):
    df = df[(df.market_type == "binary") & df.result.isin(["yes", "no"])].copy()
    for c in ["open_time", "close_time", "expected_expiration_time", "settlement_ts"]:
        df[c] = _ts(df[c])
    df = df.dropna(subset=["expected_expiration_time", "open_time", "close_time", "settlement_ts"])
    df = df[df.settlement_ts < CUT]  # DEV only
    out = []
    for k, d in HORIZONS.items():
        t = df.expected_expiration_time - pd.Timedelta(days=d)
        e = df[(df.open_time <= t) & (df.close_time > t)].copy()
        e["trial"], e["t"] = k, t[e.index]
        out.append(e)
    return pd.concat(out, ignore_index=True)


def _quote(cs, t):
    best = None
    for c in cs:
        if t - 7 * 86400 < c["end_period_ts"] <= t:
            best = c
    if best is None:
        return np.nan, np.nan, np.nan

    def g(side):
        v = best.get(side) or {}
        return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")
    return g("yes_bid"), g("yes_ask"), best["end_period_ts"]


def cmd_prices():
    df = eligible(pd.read_parquet(DATA / "markets.parquet"))
    df["t_s"] = _sec(df.t).astype(int)
    print("eligible market-trials", len(df), "markets", df.ticker.nunique(), flush=True)
    urls = [f"{API}/historical/markets/{tk}/candlesticks?start_ts={t - 8 * 86400}&end_ts={t}&period_interval=1440"
            for tk, t in zip(df.ticker, df.t_s)]
    res = fetch_many(urls)
    q = [_quote(d.get("candlesticks", []) or [], t) for d, t in zip(res, df.t_s)]
    df["yes_bid"], df["yes_ask"], df["q_end"] = zip(*q) if q else ([], [], [])
    df.to_parquet(DATA / "priced.parquet", index=False)
    print("priced", df.yes_ask.notna().sum(), "->", DATA / "priced.parquet")


def fee(p, mult):
    return 0.07 * mult * p * (1 - p)


def bets(df):
    b = df[(df.yes_bid >= 0.01) & (df.yes_ask <= 0.10) & (df.yes_ask > df.yes_bid)].copy()
    b["price"] = (1 - b.yes_bid).round(4)
    mult = np.where(b.fee_type == "quadratic", b.fee_multiplier.fillna(1).astype(float), 1.0)
    b["fee"] = fee(b.price, mult)
    b["fee_1ct"] = np.ceil(np.round(b.fee * 100, 6)) / 100
    b["cost"] = b.price + b.fee
    b["won"] = (b.result == "no").astype(float)
    b["days"] = (b.settlement_ts - b.t) / pd.Timedelta(days=1)
    b["pnl"] = b.won - b.cost
    b["pnl_1ct"] = b.won - b.price - b.fee_1ct
    b["interest"] = b.cost * APY * b.days / 365
    b["cat"] = b.category.map(CAT).fillna("culture")
    return b


def boot(b, col, cluster, n=10000, seed=0, extra=None):
    g = b.groupby(cluster).agg(p=(col, "sum"), c=("cost", "sum"))
    if extra:
        g["p"] += b.groupby(cluster)[extra].sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    P, C = g.p.to_numpy(), g.c.to_numpy()
    r = P[idx].sum(1) / C[idx].sum(1)
    return float(P.sum() / C.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summarize(b):
    o = {"n": len(b), "events": int(b.event_ticker.nunique()), "series": int(b.series.nunique()),
         "win_rate": float(b.won.mean()), "mean_price": float(b.price.mean()), "mean_days": float(b.days.mean())}
    o["roi_noint"] = boot(b, "pnl", "event_ticker")
    o["roi_int"] = boot(b, "pnl", "event_ticker", extra="interest")
    o["roi_1ct_noint"] = boot(b, "pnl_1ct", "event_ticker")
    o["roi_int_series_cl"] = boot(b, "pnl", "series", extra="interest")
    cd = (b.cost * b.days / 365).sum()
    o["ann_noint"] = float(b.pnl.sum() / cd)
    o["ann_int"] = float((b.pnl.sum() + b.interest.sum()) / cd)
    tot = b.pnl.sum() + b.interest.sum()
    s = (b.pnl + b.interest).groupby(b.series).sum().sort_values(ascending=False)
    o["top_series"] = {k: round(float(v), 3) for k, v in s.head(5).items()}
    o["top_share"] = float(s.iloc[0] / tot) if tot > 0 else None
    o["pnl_noint"], o["interest"] = float(b.pnl.sum()), float(b.interest.sum())
    return o


def cmd_dev():
    b = bets(pd.read_parquet(DATA / "priced.parquet"))
    b.to_csv(OUT / "bets_dev.csv", index=False)
    res = {}
    for k in HORIZONS:
        bk = b[b.trial == k]
        res[k] = {"all": summarize(bk)}
        for c, bc in bk.groupby("cat"):
            if bc.event_ticker.nunique() >= 5:
                res[k][c] = summarize(bc)
    (OUT / "dev_summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    {"markets": cmd_markets, "prices": cmd_prices, "dev": cmd_dev}[sys.argv[1]]()
