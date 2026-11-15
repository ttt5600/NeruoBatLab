"""L22 Kalshi politics favourites (prereg: notes/pol_favourites.md, written before any price/outcome was loaded).

    python scripts/pol_favourites.py meta      # settled Politics/Elections/World market metadata (events endpoint)
    python scripts/pol_favourites.py prices    # daily candle quote at S-7d / S-14d / S-3d for eligible markets
    python scripts/pol_favourites.py score     # trials + calibration

READ-ONLY public GETs via scripts/kx_throttle.py. No orders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from kx_throttle import kget

OUT = ROOT / "results" / "pol_favourites"
DATA = ROOT / "data" / "pol_favourites"
CATS = {"Politics", "Elections", "World"}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
KEEP = ["ticker", "event_ticker", "market_type", "open_time", "close_time", "expected_expiration_time",
        "latest_expiration_time", "settlement_ts", "can_close_early", "result", "status", "title", "yes_sub_title",
        "volume_fp", "volume"]
EKEEP = ["event_ticker", "series_ticker", "category", "title", "sub_title", "strike_date", "strike_period",
         "mutually_exclusive"]


def _ts(s):
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def _sec(t):  # unit-safe epoch seconds (traps: never astype(int)//1e9)
    return ((t - EPOCH) // pd.Timedelta(seconds=1)).astype("Int64")


def _page(path, params, keep):
    rows, cur = [], None
    while True:
        d = kget(path, dict(params, limit=1000, **({"cursor": cur} if cur else {})))
        if not d:
            return rows, False
        for m in d.get("markets", []) or []:
            if keep(m):
                rows.append({k: m.get(k) for k in KEEP})
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            return rows, True


def cmd_meta():
    """Per series (Politics/Elections/World, all frequencies): live settled markets with close >= 2025-01-01
    (min_close_ts honoured) + /historical/markets paged in full (min_close_ts ignored there), filtered at ingest on
    METADATA only: binary, close_time >= 2025-01-01, open_time <= close_time - 3 d."""
    DATA.mkdir(parents=True, exist_ok=True)
    ser = json.loads((ROOT / "results" / "pm_literature_scout" / "series.json").read_text())
    ser = ser if isinstance(ser, list) else ser.get("series", ser)
    ser = [x for x in ser if x.get("category") in ("Politics", "Elections") and x["ticker"].startswith("KX")]
    lo = pd.Timestamp("2025-01-01", tz="UTC")
    t0 = int(_sec(pd.Series([lo]))[0])

    def keep(m):
        c, o = _ts(m.get("close_time")), _ts(m.get("open_time"))
        return (m.get("market_type") == "binary" and pd.notna(c) and pd.notna(o) and c >= lo
                and o <= c - pd.Timedelta(days=3))
    def one(x):
        s, out, bd = x["ticker"], [], []
        for path, prm in (("/markets", {"series_ticker": s, "status": "settled", "min_close_ts": t0}),
                          ("/historical/markets", {"series_ticker": s})):
            r, ok = _page(path, prm, keep)
            if not ok:
                bd.append((s, path))
            for m in r:
                m.update(series=s, category=x.get("category"), frequency=x.get("frequency"),
                         fee_type=x.get("fee_type"), fee_multiplier=x.get("fee_multiplier"), src=path)
            out += r
        return out, bd
    # threads only overlap network latency; every send still takes a kx_throttle slot (<= ~3.3 req/s total)
    from concurrent.futures import ThreadPoolExecutor
    rows, bad = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for i, (r, bd) in enumerate(ex.map(one, ser)):
            rows += r
            bad += bd
            if i % 200 == 0:
                print(f"series {i}/{len(ser)} rows {len(rows)} bad {len(bad)}", flush=True)
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df.to_parquet(DATA / "meta.parquet", index=False)
    json.dump(bad, open(DATA / "meta_bad.json", "w"))
    print("->", DATA / "meta.parquet", len(df), "bad", len(bad), flush=True)


WINDOWS = {"PRIMARY": ("2026-07-01", "2026-11-15"), "PAPER": ("2025-01-01", "2026-06-30")}
TRIALS = {"T7": 7, "T14": 14, "T3": 3}
DAY = 86400


def _window(ts):
    for k, (a, b) in WINDOWS.items():
        if pd.Timestamp(a, tz="UTC") <= ts < pd.Timestamp(b, tz="UTC") + pd.Timedelta(days=1):
            return k
    return None


def eligible(df):
    """Market x trial rows open at t (metadata only). Adds the close-anchored T7 sensitivity row ('C7')."""
    df = df[(df.market_type == "binary")].copy()
    for c in ["open_time", "close_time", "expected_expiration_time", "settlement_ts"]:
        df[c] = _ts(df[c])
    df = df.dropna(subset=["open_time", "close_time", "settlement_ts"])
    df["window"] = df.settlement_ts.map(_window)
    df = df[df.window.notna()]
    out = []
    for k, d in list(TRIALS.items()) + [("C7", 7)]:
        anchor = df.close_time if k == "C7" else df.expected_expiration_time
        t = anchor - pd.Timedelta(days=d)
        e = df[t.notna() & (df.open_time <= t) & (df.close_time > t)].copy()
        e["trial"], e["t"] = k, t[e.index]
        out.append(e)
    return pd.concat(out, ignore_index=True)


def _quote(cs, t):
    best = None
    for c in cs:
        if t - 36 * 3600 < c["end_period_ts"] <= t:
            best = c
    if best is None:
        return np.nan, np.nan, np.nan

    def g(side):
        v = best.get(side) or {}
        return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")
    return g("yes_bid"), g("yes_ask"), best["end_period_ts"]


def _candles(tk, series, src, a, b):
    if src == "/historical/markets":
        d = kget(f"/historical/markets/{tk}/candlesticks", {"start_ts": a, "end_ts": b, "period_interval": 1440})
    else:
        d = kget(f"/series/{series}/markets/{tk}/candlesticks", {"start_ts": a, "end_ts": b, "period_interval": 1440})
    return (d or {}).get("candlesticks", []) or []


def cmd_prices():
    e = eligible(pd.read_parquet(DATA / "meta.parquet"))
    e["t_s"] = _sec(e.t).astype(int)
    print("eligible market-trials", len(e), "markets", e.ticker.nunique(), e.groupby(["window", "trial"]).size().to_dict(),
          flush=True)
    cache = DATA / "candles.jsonl"
    done = {}
    if cache.exists():
        for line in cache.read_text().splitlines():
            r = json.loads(line)
            done[(r["tk"], r["a"], r["b"])] = r["c"]
    reqs = []
    # STAGE env (comma list of window:trialgroup, e.g. PRIMARY:T,PRIMARY:C) limits which rows get candles requested
    # (priority order under the shared rate budget); priced.parquet is always built from everything cached.
    import os
    stage = os.environ.get("STAGE")
    er = e
    if stage:
        keep = set(stage.split(","))
        er = e[[f"{w}:{'C' if t == 'C7' else 'T'}" in keep for w, t in zip(e.window, e.trial)]]
    for tk, g in er.groupby("ticker"):
        ts = sorted(g.t_s)
        # one request per cluster of entry times within 40 days
        grp = [[ts[0]]]
        for x in ts[1:]:
            (grp[-1].append(x) if x - grp[-1][0] <= 40 * DAY else grp.append([x]))
        for gg in grp:
            reqs.append((tk, g.series.iloc[0], g.src.iloc[0], gg[0] - 2 * DAY, gg[-1]))
    print("requests", len(reqs), "cached", sum((r[0], r[3], r[4]) in done for r in reqs), flush=True)
    todo = [r for r in reqs if (r[0], r[3], r[4]) not in done]

    def one(r):
        tk, s, src, a, b = r
        c = _candles(tk, s, src, a, b)
        return r, [{"end_period_ts": x["end_period_ts"], "yes_bid": x.get("yes_bid"), "yes_ask": x.get("yes_ask")}
                   for x in c]
    from concurrent.futures import ThreadPoolExecutor   # 2 threads, every send through kx_throttle.slot
    with open(cache, "a") as fh, ThreadPoolExecutor(max_workers=2) as ex:
        for i, ((tk, s, src, a, b), c) in enumerate(ex.map(one, todo)):
            done[(tk, a, b)] = c
            fh.write(json.dumps({"tk": tk, "a": a, "b": b, "c": c}) + "\n")
            fh.flush()
            if i % 500 == 0:
                print("candles", i, len(todo), flush=True)
    by_tk = {}
    for (tk, a, b), c in done.items():
        by_tk.setdefault(tk, []).extend(c)
    q = [_quote(sorted(by_tk.get(tk, []), key=lambda x: x["end_period_ts"]), t) for tk, t in zip(e.ticker, e.t_s)]
    e["yes_bid"], e["yes_ask"], e["q_end"] = zip(*q)
    e.to_parquet(DATA / "priced.parquet", index=False)
    print("priced", int(e.yes_ask.notna().sum()), "of", len(e), "->", DATA / "priced.parquet", flush=True)


# ------------------------------------------------------------------ scoring
def _logit_fit(x, y, w=None, it=25):
    X = np.column_stack([np.ones_like(x), x])
    b = np.zeros(2)
    w = np.ones_like(x) if w is None else w
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ b))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(2)
        b = b + np.linalg.solve(H, X.T @ (w * (y - p)))
    return b


def calibration(df, n=1000, seed=1):
    d = df[(df.yes_bid >= 0.01) & (df.yes_ask <= 0.99) & (df.yes_ask > df.yes_bid)].copy()
    mid = ((d.yes_bid + d.yes_ask) / 2).clip(0.02, 0.98)
    d["x"], d["y"] = np.log(mid / (1 - mid)), d.yes_pay.astype(float)
    if len(d) < 30:
        return {"n": len(d)}
    b = _logit_fit(d.x.to_numpy(), d.y.to_numpy())
    codes, uniq = pd.factorize(d.event_ticker)
    rng = np.random.default_rng(seed)
    X, Y = d.x.to_numpy(), d.y.to_numpy()
    sl = []
    for _ in range(n):
        cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))
        sl.append(_logit_fit(X, Y, cnt[codes].astype(float))[1])
    return {"n": len(d), "events": len(uniq), "slope": float(b[1]), "intercept": float(b[0]),
            "slope_ci": [float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5))]}


def fee(p, mult):
    return 0.07 * mult * p * (1 - p)


def bets(fr, trial, cats):
    sys.path.insert(0, str(ROOT))
    from edgelab.rules.politics_fav import bets_frame
    b = bets_frame(fr, trial, cats)
    mult = np.where(b.fee_type.isin(["quadratic", "quadratic_with_maker_fees"]),
                    pd.to_numeric(b.fee_multiplier, errors="coerce").fillna(1).astype(float), 1.0)
    b["fee"] = fee(b.price, mult)
    b["fee_1ct"] = np.ceil(np.round(b.fee * 100, 6)) / 100
    b["cost"] = b.price + b.fee
    b["won"] = np.where(b.side == "yes", b.yes_pay, 1 - b.yes_pay)
    b["pnl"] = b.won - b.cost
    b["pnl_1ct"] = b.won - b.price - b.fee_1ct
    b["day"] = b.close_time.dt.strftime("%Y-%m-%d")
    return b


def boot(b, col, cluster, n=10000, seed=0):
    g = b.groupby(cluster).agg(p=(col, "sum"), c=("cost", "sum"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    P, C = g.p.to_numpy(), g.c.to_numpy()
    r = P[idx].sum(1) / C[idx].sum(1)
    return [round(float(P.sum() / C.sum()), 5), round(float(np.percentile(r, 2.5)), 5),
            round(float(np.percentile(r, 97.5)), 5)]


def summarize(b):
    if len(b) == 0:
        return {"n": 0}
    o = {"n": len(b), "events": int(b.event_ticker.nunique()), "series": int(b.series.nunique()),
         "days": int(b.day.nunique()), "win_rate": round(float(b.won.mean()), 4),
         "mean_price": round(float(b.price.mean()), 4), "yes_share": round(float((b.side == "yes").mean()), 3),
         "pnl": round(float(b.pnl.sum()), 3)}
    o["roi_event"] = boot(b, "pnl", "event_ticker")
    o["roi_series"] = boot(b, "pnl", "series")
    o["roi_day"] = boot(b, "pnl", "day")
    o["roi_1ct_event"] = boot(b, "pnl_1ct", "event_ticker")
    s = b.groupby("day").pnl.sum().sort_values(ascending=False)
    tot = b.pnl.sum()
    o["top_days"] = {k: round(float(v), 3) for k, v in s.head(3).items()}
    o["top_day_share"] = round(float(s.iloc[0] / tot), 3) if tot > 0 else None
    ss = b.groupby("series").pnl.sum().sort_values(ascending=False)
    o["top_series"] = {k: round(float(v), 3) for k, v in ss.head(3).items()}
    o["worst_series"] = {k: round(float(v), 3) for k, v in ss.tail(3).items()}
    return o


def cmd_score():
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(DATA / "priced.parquet")
    # scalar-settled binaries (disputes / partial settlement) pay settlement_value_dollars per YES contract
    sv = json.loads((DATA / "scalar_values.json").read_text())
    fr = fr[fr.result.isin(["yes", "no", "scalar"])].copy()
    fr["yes_pay"] = np.where(fr.result == "yes", 1.0, 0.0)
    sc = fr.result == "scalar"
    fr.loc[sc, "yes_pay"] = pd.to_numeric(fr.loc[sc, "ticker"].map(sv), errors="coerce")
    fr = fr.dropna(subset=["yes_pay"])
    res, allb = {}, []
    for w in WINDOWS:
        fw = fr[fr.window == w]
        res[w] = {}
        for tr in ["T7", "T14", "T3", "C7"]:
            b = bets(fw, tr, ("Politics", "Elections"))
            allb.append(b.assign(cats="PE"))
            r = {"PE": summarize(b)}
            for c in ["Politics", "Elections", "World"]:
                r[c] = summarize(bets(fw, tr, (c,)))
            if tr in ("T7", "C7"):
                r["calibration_PE"] = calibration(fw[(fw.trial == tr) & fw.category.isin(["Politics", "Elections"])])
            res[w][tr] = r
            print(w, tr, json.dumps(r["PE"]), flush=True)
    pd.concat(allb).to_csv(OUT / "bets.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print("->", OUT / "summary.json")


if __name__ == "__main__":
    {"meta": cmd_meta, "prices": cmd_prices, "score": cmd_score}[sys.argv[1]]()
