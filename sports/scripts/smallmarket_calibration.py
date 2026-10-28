"""Small / long-tail prediction-market calibration scan (EXPLORATORY, hypothesis-generating; read-only public data).

    python scripts/smallmarket_calibration.py pm_enum      # Polymarket: sampled days -> events/markets (dev window)
    python scripts/smallmarket_calibration.py pm_prices    # Polymarket CLOB price history at fixed horizons
    python scripts/smallmarket_calibration.py pm_spreads   # snapshot of LIVE Polymarket books -> spread proxy
    python scripts/smallmarket_calibration.py k_enum       # Kalshi: sampled series -> settled markets
    python scripts/smallmarket_calibration.py k_candles    # Kalshi hourly candles (bid/ask/volume) at fixed horizons
    python scripts/smallmarket_calibration.py holdout      # COUNT-ONLY census of the untouched holdout window
    python scripts/smallmarket_calibration.py analyze      # tables -> data/smallmarket/tables.json

No orders, no keys, no geo-block workarounds: gamma-api / clob (Polymarket public data) and Kalshi public trade-api.

Design (fixed 2026-10-27 before any price/outcome was looked at):
  DEV window   : markets whose scheduled end is 2025-05-01 .. 2026-04-30 (UTC). Only DEV prices/outcomes are read.
  HOLDOUT      : 2026-05-01 .. 2026-10-20, never priced here (only counted) -> reserved for pre-registered tests.
  Anchor S     : Polymarket = endDate (scheduled); Kalshi = close_time. Decision time t = S - h, h in {168h, 24h, 1h}.
                 A market enters horizon h only if it was still open at t (t < actual close) and has a quote at/before t
                 within 6 h (Polymarket display price) / the last hourly candle with a two-sided quote (Kalshi).
                 Kalshi markets with can_close_early are flagged; results are shown with and without them.
  Price        : Polymarket display price (mid, or last trade if spread > 10c) of outcome-0 token; Kalshi mid of
                 yes_bid/yes_ask candle close. Spread: Kalshi actual quoted; Polymarket = proxy from today's books.
  Liquidity    : Kalshi = contracts traded in the 24 h before t (ex-ante) and lifetime volume; Polymarket = lifetime
                 volumeNum (EX-POST -- a caveat, not tradeable as a filter as-is).
  CIs          : bootstrap over EVENTS (all markets of an event resampled together), 1000 draws.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode, urlparse

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "smallmarket"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path("/Users/jonathanwang/.claude/jobs/588c8b26/tmp/cache")
CACHE.mkdir(parents=True, exist_ok=True)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
DEV0, DEV1 = pd.Timestamp("2025-05-01", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")
HO0, HO1 = pd.Timestamp("2026-05-01", tz="UTC"), pd.Timestamp("2026-10-21", tz="UTC")
HORIZONS = {"7d": 168, "1d": 24, "1h": 1}
RATE = {"gamma-api.polymarket.com": 0.25, "clob.polymarket.com": 0.12, "api.elections.kalshi.com": 0.17}
_lock = threading.Lock()
_last: dict[str, float] = {}
_local = threading.local()


def get(url, params=None, cache=True, retries=8):
    full = url + ("?" + urlencode(params) if params else "")
    path = CACHE / (hashlib.sha256(full.encode()).hexdigest()[:24] + ".json")
    if cache and path.exists():
        return json.loads(path.read_text())
    host = urlparse(url).netloc
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    for a in range(retries):
        with _lock:
            now = time.monotonic()
            slot = max(now, _last.get(host, 0) + RATE.get(host, 0.25))
            _last[host] = slot
        if slot > now:
            time.sleep(slot - now)
        try:
            r = _local.s.get(full, headers={"User-Agent": "Mozilla/5.0 (research; read-only)"}, timeout=60)
        except requests.RequestException:
            time.sleep(min(2 ** a, 30)); continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(2 ** (a + 1), 60)); continue
        if r.status_code in (400, 404, 422):
            return None
        r.raise_for_status()
        d = r.json()
        if cache:
            path.write_text(json.dumps(d))
        return d
    return None


def pmap(fn, xs, workers=6):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


# ----------------------------------------------------------------------------------------------- Polymarket
def pm_category(tags: list[str]) -> str:
    t = set(tags)
    if "Up or Down" in t or "Crypto Prices" in t:
        return "crypto_price"
    if "Esports" in t:
        return "esports"
    if "Sports" in t or "Games" in t:
        return "sports"
    if "Weather" in t or "Daily Temperature" in t:
        return "weather"
    if "Mentions" in t or "Mention Markets" in t:
        return "mentions"
    if "Crypto" in t:
        return "crypto_other"
    if t & {"Politics", "Elections", "Geopolitics", "World", "US Election", "Primaries", "Trump", "Middle East",
            "Ukraine", "Global Elections", "World Elections"}:
        return "politics_world"
    if t & {"Finance", "Equities", "Stocks", "Business", "Economy", "Earnings", "Indicies", "Fed", "Economics",
            "Commodities", "Fed Rates"}:
        return "finance_econ"
    if t & {"Culture", "Movies", "Entertainment", "Music", "Awards", "Tech", "Science", "AI", "Pop Culture",
            "Celebrities", "Twitter", "Elon Musk", "Box Office", "Netflix", "Top Netflix"}:
        return "culture_tech"
    return "other"


def pm_day(day: pd.Timestamp) -> list[dict]:
    a, b = day.strftime("%Y-%m-%dT%H:%M:%SZ"), (day + pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = []
    for off in range(0, 3000, 100):
        r = get(f"{GAMMA}/events", {"closed": "true", "limit": 100, "offset": off, "end_date_min": a, "end_date_max": b})
        if not r:
            break
        out += r
        if len(r) < 100:
            break
    return out


def pm_flatten(evs: list[dict], keep_outcome: bool) -> list[dict]:
    rows = []
    for e in evs:
        tags = [t.get("label", "") for t in (e.get("tags") or [])]
        ms = e.get("markets") or []
        neg = bool(e.get("enableNegRisk") or e.get("negRiskAugmented") or any(m.get("negRisk") for m in ms))
        for m in ms:
            try:
                outs = json.loads(m.get("outcomes") or "[]")
                toks = json.loads(m.get("clobTokenIds") or "[]")
                op = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            except Exception:
                continue
            if len(toks) != 2 or len(op) != 2:
                continue
            r = {"event_id": e["id"], "event_slug": e.get("slug"), "market_id": m["id"], "question": m.get("question"),
                 "outcome0": outs[0] if outs else None, "token0": toks[0], "cat": pm_category(tags),
                 "tags": "|".join(tags), "neg_risk": neg, "n_mkts_event": len(ms),
                 "end": m.get("endDate") or e.get("endDate"), "closed_time": m.get("closedTime"),
                 "start": m.get("startDate"), "game_start": m.get("gameStartTime"),
                 "volume": float(m.get("volumeNum") or 0), "event_volume": float(e.get("volume") or 0),
                 "fees_enabled": bool(m.get("feesEnabled")), "fee_schedule": json.dumps(m.get("feeSchedule")),
                 "uma_status": m.get("umaResolutionStatus")}
            if keep_outcome:
                r["y0"] = 1.0 if op[0] == 1 else (0.0 if op[0] == 0 else np.nan)
            rows.append(r)
    return rows


def cmd_pm_enum() -> int:
    rng = random.Random(20261027)
    days = list(pd.date_range(DEV0, DEV1 - pd.Timedelta(days=1), freq="D"))
    days = sorted(rng.sample(days, 92))  # ~1 day in 4
    allrows = []
    for i, evs in enumerate(pmap(pm_day, days, workers=3)):
        allrows += pm_flatten(evs, keep_outcome=True)
        if i % 10 == 0:
            print(f"  day {days[i].date()}: {len(evs)} events; total markets {len(allrows)}", flush=True)
    df = pd.DataFrame(allrows).drop_duplicates("market_id")
    df["end"] = pd.to_datetime(df["end"], utc=True, errors="coerce", format="ISO8601")
    df = df[(df.end >= DEV0) & (df.end < DEV1)]
    df.to_parquet(OUT / "pm_markets_all.parquet", index=False)
    print(df.groupby("cat").agg(markets=("market_id", "size"), events=("event_id", "nunique")).to_string())
    return 0


def pm_select(df: pd.DataFrame) -> pd.DataFrame:
    """Sample EVENTS per category (cap), keep every binary market of a sampled event (needed for sums)."""
    rng = np.random.default_rng(7)
    caps = {"crypto_price": 400}
    keep = []
    for cat, g in df.groupby("cat"):
        ev = g.groupby("event_id").agg(n=("market_id", "size")).reset_index()
        ev = ev[ev.n <= 40]  # very large events (e.g. 100-name fields) dropped for request cost
        ev = ev.sample(frac=1, random_state=int(rng.integers(1e9)))
        cap = caps.get(cat, 4000)  # markets per category; whole events kept (needed for sums)
        ev = ev[ev.n.cumsum() <= cap]
        keep.append(g[g.event_id.isin(ev.event_id)])
    return pd.concat(keep)


def pm_hist(row, anchor: int | None = None) -> dict | None:
    """anchor (unix s) overrides endDate as S; the request URL (hence cache) is always keyed on endDate."""
    end = int(row.end.timestamp())
    d = get(f"{CLOB}/prices-history", {"market": row.token0, "startTs": end - 8 * 86400, "endTs": end + 3600,
                                         "fidelity": 60})
    if not d or not d.get("history"):
        return {"market_id": row.market_id, "n_pts": 0}
    h = d["history"]
    ts = np.array([x["t"] for x in h]); ps = np.array([x["p"] for x in h], dtype=float)
    out = {"market_id": row.market_id, "n_pts": len(h)}
    closed = pd.to_datetime(row.closed_time, utc=True, errors="coerce")
    cts = closed.timestamp() if pd.notna(closed) else np.inf
    S = end if anchor is None else min(end, anchor)
    for k, hh in HORIZONS.items():
        t = S - hh * 3600
        if t < end - 8 * 86400:
            continue
        if t >= cts:
            continue
        j = np.searchsorted(ts, t, side="right") - 1
        if j >= 0 and t - ts[j] <= 6 * 3600:
            out[f"p_{k}"] = ps[j]
    return out


def cmd_pm_prices() -> int:
    df = pd.read_parquet(OUT / "pm_markets_all.parquet")
    df = df[df.y0.notna()]
    sel = pm_select(df)
    print(f"selected {len(sel)} markets / {sel.event_id.nunique()} events", flush=True)
    rows = []
    rs = list(sel.itertuples())
    for i in range(0, len(rs), 600):
        rows += [r for r in pmap(pm_hist, rs[i:i + 600], workers=6) if r]
        print(f"  {i + 600}/{len(rs)}", flush=True)
    p = pd.DataFrame(rows)
    sel.merge(p, on="market_id", how="left").to_parquet(OUT / "pm_priced.parquet", index=False)
    return 0


def cmd_pm_spreads() -> int:
    """Today's ACTIVE markets: best bid/ask + lifetime volume -> spread proxy by category x volume x price."""
    rows = []
    for off in range(0, 30000, 100):
        r = get(f"{GAMMA}/events", {"closed": "false", "active": "true", "limit": 100, "offset": off}, cache=False)
        if not r:
            break
        for e in r:
            tags = [t.get("label", "") for t in (e.get("tags") or [])]
            for m in e.get("markets") or []:
                bb, ba = m.get("bestBid"), m.get("bestAsk")
                if bb is None or ba is None or not m.get("acceptingOrders"):
                    continue
                rows.append({"cat": pm_category(tags), "volume": float(m.get("volumeNum") or 0), "bid": float(bb),
                             "ask": float(ba), "end": m.get("endDate")})
        if len(r) < 100:
            break
    s = pd.DataFrame(rows)
    s.to_parquet(OUT / "pm_live_books.parquet", index=False)
    print(len(s), "live markets with a book")
    return 0


# ----------------------------------------------------------------------------------------------- Kalshi
def k_category(c: str) -> str:
    return {"Elections": "politics", "Politics": "politics", "World": "politics", "Financials": "finance_econ",
            "Economics": "finance_econ", "Companies": "finance_econ", "Commodities": "finance_econ",
            "Climate and Weather": "weather", "Science and Technology": "sci_tech", "Health": "sci_tech",
            "Entertainment": "entertainment", "Social": "entertainment", "Culture": "entertainment",
            "Sports": "sports", "Mentions": "mentions", "Crypto": "crypto", "Transportation": "other",
            "Exotics": "other"}.get(c, "other")


def k_series_markets(s: str) -> list[dict]:
    rows = []
    for path, extra in ((f"{KAPI}/historical/markets", {}), (f"{KAPI}/markets", {"status": "settled"})):
        cur = None
        for _ in range(60):
            p = {"series_ticker": s, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = get(path, p, cache=True) or {}
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    return rows


def cmd_k_enum() -> int:
    ser = (get(f"{KAPI}/series", cache=True) or {})["series"]
    sd = pd.DataFrame([{"series": s["ticker"], "category": s.get("category"), "frequency": s.get("frequency"),
                        "fee_type": s.get("fee_type"), "fee_mult": float(s.get("fee_multiplier") or 1)} for s in ser])
    sd = sd[~sd.frequency.isin(["hourly", "fifteen_min"]) & ~sd.series.str.contains("TEST|MVE")]
    sd["cat"] = sd.category.map(k_category)
    rng = np.random.default_rng(11)
    pick = []
    for cat, g in sd.groupby("cat"):
        n = min(len(g), 220)
        pick.append(g.sample(n, random_state=int(rng.integers(1e9))))
    sd = pd.concat(pick)
    print(f"sampled {len(sd)} series", sd.cat.value_counts().to_dict(), flush=True)
    rows = []
    res = pmap(k_series_markets, list(sd.series), workers=4)
    meta = sd.set_index("series")
    for s, ms in zip(sd.series, res):
        for m in ms:
            if m.get("result") not in ("yes", "no") or m.get("mve_collection_ticker"):
                continue
            rows.append({"series": s, "cat": meta.loc[s, "cat"], "category": meta.loc[s, "category"],
                         "frequency": meta.loc[s, "frequency"], "fee_mult": meta.loc[s, "fee_mult"],
                         "fee_type": meta.loc[s, "fee_type"], "ticker": m["ticker"], "event_ticker": m["event_ticker"],
                         "close_time": m.get("close_time"), "open_time": m.get("open_time"),
                         "exp_time": m.get("expected_expiration_time"), "can_close_early": bool(m.get("can_close_early")),
                         "strike_type": m.get("strike_type"), "volume": float(m.get("volume_fp") or m.get("volume") or 0),
                         "result": m["result"]})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    for c in ("close_time", "open_time", "exp_time"):
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce", format="ISO8601")
    ho = df[(df.close_time >= HO0) & (df.close_time < HO1)].drop(columns=["result"])
    ho.to_parquet(OUT / "k_holdout_census.parquet", index=False)  # outcome column dropped on purpose
    df = df[(df.close_time >= DEV0) & (df.close_time < DEV1)]
    df["n_mkts_event"] = df.groupby("event_ticker").ticker.transform("size")
    df.to_parquet(OUT / "k_markets_all.parquet", index=False)
    print(df.groupby("cat").agg(markets=("ticker", "size"), events=("event_ticker", "nunique"),
                                series=("series", "nunique")).to_string())
    return 0


def k_event_meta(ev: str) -> dict:
    d = get(f"{KAPI}/events/{ev}", cache=True) or {}
    e = d.get("event") or {}
    return {"event_ticker": ev, "mutually_exclusive": e.get("mutually_exclusive")}


def k_select(df: pd.DataFrame) -> pd.DataFrame:
    """Per series up to 6 random events; per event all markets if <= 15 (sums), else 3 random markets;
    then at most 3,500 markets per category (series visited in random order, whole events kept)."""
    rng = np.random.default_rng(5)
    keep = []
    for cat, gc in df.groupby("cat"):
        n = 0
        sers = list(gc.series.unique())
        rng.shuffle(sers)
        for s in sers:
            g = gc[gc.series == s]
            evs = g.event_ticker.unique()
            if len(evs) > 6:
                evs = rng.choice(evs, 6, replace=False)
            for e in evs:
                ge = g[g.event_ticker == e]
                ge = ge if len(ge) <= 15 else ge.sample(3, random_state=int(rng.integers(1e9)))
                keep.append(ge)
                n += len(ge)
            if n >= 3500:
                break
    return pd.concat(keep)


def _n(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def k_candles(row, anchor: int | None = None) -> dict:
    """anchor (unix s, >= own close) = scheduled event close; t = anchor - h must precede this market's own close."""
    close = int(row.close_time.timestamp())
    S = close if anchor is None else max(close, anchor)
    start = max(close - 9 * 86400, int(row.open_time.timestamp()) if pd.notna(row.open_time) else 0)
    d = get(f"{KAPI}/historical/markets/{row.ticker}/candlesticks",
            {"start_ts": start, "end_ts": close, "period_interval": 60}) or {}
    cs = d.get("candlesticks") or []
    out = {"ticker": row.ticker, "n_candles": len(cs)}
    if not cs:
        return out
    ts = np.array([c["end_period_ts"] for c in cs])
    g = lambda c, side: _n((c.get(side) or {}).get("close_dollars", (c.get(side) or {}).get("close")))
    bid = np.array([g(c, "yes_bid") for c in cs]); ask = np.array([g(c, "yes_ask") for c in cs])
    vol = np.array([_n(c.get("volume_fp", c.get("volume"))) for c in cs]); vol = np.nan_to_num(vol)
    for k, hh in HORIZONS.items():
        t = S - hh * 3600
        if t >= close:  # market already closed at t (early close) -> not tradeable then
            continue
        j = np.searchsorted(ts, t, side="right") - 1
        if j < 0 or t - ts[j] > 6 * 3600:
            continue
        out[f"bid_{k}"], out[f"ask_{k}"] = bid[j], ask[j]
        out[f"v24_{k}"] = float(vol[(ts > t - 86400) & (ts <= t)].sum())
    return out


def cmd_k_candles() -> int:
    df = pd.read_parquet(OUT / "k_markets_all.parquet")
    sel = k_select(df)
    print(f"selected {len(sel)} markets / {sel.event_ticker.nunique()} events", flush=True)
    evs = list(sel.event_ticker.unique())
    meta = pd.DataFrame(pmap(k_event_meta, evs, workers=4))
    rows = []
    rs = list(sel.itertuples())
    for i in range(0, len(rs), 1000):
        rows += pmap(k_candles, rs[i:i + 1000], workers=5)
        print(f"  {min(i + 1000, len(rs))}/{len(rs)}", flush=True)
    c = pd.DataFrame(rows)
    sel.merge(meta, on="event_ticker", how="left").merge(c, on="ticker", how="left").to_parquet(
        OUT / "k_priced.parquet", index=False)
    return 0


def cmd_holdout() -> int:
    """Count-only: Polymarket markets per category on 12 random holdout days (outcomes never parsed)."""
    rng = random.Random(99)
    days = sorted(rng.sample(list(pd.date_range(HO0, HO1 - pd.Timedelta(days=1), freq="D")), 12))
    rows = []
    for evs in pmap(pm_day, days, workers=3):
        rows += pm_flatten(evs, keep_outcome=False)
    df = pd.DataFrame(rows).drop_duplicates("market_id")
    df.to_parquet(OUT / "pm_holdout_census.parquet", index=False)
    print(df.groupby("cat").agg(markets=("market_id", "size"), events=("event_id", "nunique")).to_string())
    return 0


def cmd_reprice() -> int:
    """Re-derive horizon prices from the CACHED responses with ex-ante anchors (fixed after the first audit,
    before any outcome was tabulated): Polymarket S = min(endDate, gameStartTime) -- 1h before a sports endDate was
    often post-game; Kalshi S = latest close_time among the event's settled markets (= scheduled close; a market that
    closed early, e.g. a mention said, enters a horizon only if it was still open at S - h)."""
    p = pd.read_parquet(OUT / "pm_priced.parquet")
    p = p[[c for c in p.columns if not (c.startswith("p_") or c == "n_pts")]]
    gs = pd.to_datetime(p.game_start, utc=True, errors="coerce")
    anchors = [int(g.timestamp()) if pd.notna(g) else None for g in gs]
    rows = pmap(lambda ra: pm_hist(*ra), list(zip(p.itertuples(), anchors)), workers=8)
    p.merge(pd.DataFrame([r for r in rows if r]), on="market_id", how="left").to_parquet(OUT / "pm_priced2.parquet", index=False)
    k = pd.read_parquet(OUT / "k_priced.parquet")
    k = k[[c for c in k.columns if not (c.split("_")[0] in ("bid", "ask", "v24") or c == "n_candles")]]
    allm = pd.read_parquet(OUT / "k_markets_all.parquet")
    S = allm.groupby("event_ticker").close_time.max()
    k["anchor"] = k.event_ticker.map(S)
    rows = pmap(lambda ra: k_candles(*ra), [(r, int(r.anchor.timestamp())) for r in k.itertuples()], workers=8)
    k.merge(pd.DataFrame(rows), on="ticker", how="left").to_parquet(OUT / "k_priced2.parquet", index=False)
    print("reprice done", len(p), len(k))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "reprice":
        sys.exit(cmd_reprice())
    sys.exit({"pm_enum": cmd_pm_enum, "pm_prices": cmd_pm_prices, "pm_spreads": cmd_pm_spreads,
              "k_enum": cmd_k_enum, "k_candles": cmd_k_candles, "holdout": cmd_holdout}[cmd]())
