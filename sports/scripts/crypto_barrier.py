"""crypto_barrier DEV pipeline (notes/crypto_barrier.md). Descriptive only: no trade-rule scoring on DEV.

    python scripts/crypto_barrier.py markets   # DEV events (close <= 2026-04-10) -> results/crypto_barrier/markets.parquet
    python scripts/crypto_barrier.py candles   # Kalshi hourly candles per market (cache candles/)
    python scripts/crypto_barrier.py ext       # Deribit DVOL hourly + Coinbase hourly candles (BTC, ETH)
    python scripts/crypto_barrier.py frame     # daily 16:00 UTC rows with M0/M1/M2 + mid (outcome kept hidden)
    python scripts/crypto_barrier.py mech      # mechanics check: Kalshi early close vs first Coinbase minute crossing
    python scripts/crypto_barrier.py calib     # calibration table + Brier (reads outcomes; run last)
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402
import crypto_barrier_model as M  # noqa: E402

R = ROOT / "results" / "crypto_barrier"
DEV_END = "2026-04-10"
SERIES = {"KXBTCMAXM": ("BTC", "max"), "KXETHMAXM": ("ETH", "max"), "KXBTCMAXMON": ("BTC", "max"),
          "KXBTCMINMON": ("BTC", "min"), "KXETHMAXMON": ("ETH", "max"), "KXETHMINMON": ("ETH", "min")}
T_HOUR_UTC = 16
_S = requests.Session()


def ts(s) -> int:
    return int(pd.Timestamp(s).timestamp())     # tz-aware ISO -> unix s (no datetime64 // 1e9 trap)


def num(c: dict, side: str, key: str) -> float:
    d = c.get(side) or {}
    if d.get(key + "_dollars") is not None:
        return float(d[key + "_dollars"])
    v = d.get(key)
    if v is None:
        return np.nan
    if isinstance(v, str):
        return float(v)                                         # historical endpoint: dollar strings ("0.1500")
    return float(v) / 100                                       # legacy integer cents


def window_end(rules: str) -> int:
    """Scheduled end of the measurement window from rules_primary (NOT close_time, which moves earlier on a touch)."""
    import re
    m = re.search(r"(?:on|by|before)\s+([A-Z][a-z]+\.? \d{1,2}, \d{4})(.{0,30})", rules)
    if not m:
        raise ValueError(rules)
    day = pd.Timestamp(m.group(1).replace(".", "")).tz_localize("America/New_York")
    if re.search(r"12:00\s*AM", m.group(2), re.I):
        return int(day.timestamp())
    return int((day + pd.Timedelta(hours=23, minutes=59, seconds=59)).timestamp())


# ------------------------------------------------------------------ Kalshi
def cmd_markets():
    cen = pd.read_csv(ROOT / "results" / "ot1_extension" / "census_longdated.csv", index_col=0)
    cen = cen[cen.series.isin(SERIES) & (cen.close <= DEV_END)]
    rows = []
    for ev, r in cen.iterrows():
        ms = (kget("/historical/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])
        if not ms:
            ms = (kget("/markets", {"event_ticker": ev, "limit": 200}) or {}).get("markets", [])
        for m in ms:
            rows.append({"event": ev, "series": r.series, "ticker": m["ticker"], "strike_type": m.get("strike_type"),
                         "floor_strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
                         "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                         "expiration_time": m.get("expiration_time"), "rules_primary": m.get("rules_primary"),
                         "volume": m.get("volume"), "result": m.get("result"), "status": m.get("status")})
        print(ev, len(ms), flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(R / "markets.parquet")
    # outcome-free print: strike fields + one rules text per event
    for ev, g in d.groupby("event"):
        print(ev, g.strike_type.unique(), sorted(g.floor_strike.dropna().tolist())[:3], sorted(g.cap_strike.dropna().tolist())[:3])
        print("   ", (g.rules_primary.iloc[0] or "")[:220].replace("\n", " "))


def cmd_candles():
    d = pd.read_parquet(R / "markets.parquet")
    cache = R / "candles"
    cache.mkdir(exist_ok=True)
    for r in d.itertuples():
        f = cache / f"{r.ticker}.json"
        if f.exists():
            continue
        cs = []
        a, b = ts(r.open_time), ts(r.close_time)
        for path in (f"/historical/markets/{r.ticker}/candlesticks", f"/series/{r.series}/markets/{r.ticker}/candlesticks"):
            t0 = a
            cs = []
            while t0 < b:                                 # <= ~5000 candles per call: chunk 120 days
                t1 = min(b, t0 + 120 * 86400)
                x = kget(path, {"start_ts": t0, "end_ts": t1, "period_interval": 60}) or {}
                cs += x.get("candlesticks", [])
                t0 = t1
            if cs:
                break
        f.write_text(json.dumps(cs))
        print(r.ticker, len(cs), flush=True)


# ------------------------------------------------------------------ Deribit DVOL + Coinbase
def dvol(cur: str, t0: int, t1: int) -> pd.DataFrame:
    out = []
    a = t0
    while a < t1:
        b = min(t1, a + 30 * 86400)
        for _ in range(5):
            try:
                j = _S.get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                           params={"currency": cur, "start_timestamp": a * 1000, "end_timestamp": b * 1000,
                                   "resolution": "3600"}, timeout=30).json()
                break
            except Exception:  # noqa: BLE001
                time.sleep(2)
        out += j["result"]["data"]
        a = b
    d = pd.DataFrame(out, columns=["ts_ms", "o", "h", "l", "c"]).drop_duplicates("ts_ms").sort_values("ts_ms")
    d["start"] = d.ts_ms // 1000                         # candle START (unix s); close is known at start + 3600
    return d


def coinbase(product: str, t0: int, t1: int, gran: int = 3600) -> pd.DataFrame:
    out = []
    a = t0
    step = 300 * gran
    while a < t1:
        b = min(t1, a + step)
        for k in range(6):
            r = _S.get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                       params={"granularity": gran, "start": datetime.fromtimestamp(a, timezone.utc).isoformat(),
                               "end": datetime.fromtimestamp(b, timezone.utc).isoformat()}, timeout=30)
            if r.status_code == 200:
                out += r.json()
                break
            time.sleep(1 + k)
        time.sleep(0.12)
        a = b
    d = pd.DataFrame(out, columns=["start", "low", "high", "open", "close", "vol"]).drop_duplicates("start")
    return d.sort_values("start").reset_index(drop=True)


def cmd_ext():
    t0, t1 = ts("2022-02-15T00:00:00Z"), ts("2026-04-12T00:00:00Z")
    for cur in ("BTC", "ETH"):
        dvol(cur, ts("2024-01-01T00:00:00Z"), t1).to_parquet(R / f"dvol_{cur}.parquet")
        coinbase(f"{cur}-USD", t0, t1).to_parquet(R / f"cb1h_{cur}.parquet")
        print(cur, "ok", flush=True)


# ------------------------------------------------------------------ frame
def cmd_frame():
    d = pd.read_parquet(R / "markets.parquet")
    d["asset"] = d.series.map(lambda s: SERIES[s][0])
    d["kind"] = d.series.map(lambda s: SERIES[s][1])
    d["strike"] = np.where(d.kind == "max", d.floor_strike, d.cap_strike).astype(float)
    bad = d.strike.isna() | ~d.strike_type.isin(["greater", "less", "greater_or_equal", "less_or_equal"])
    print("strike fields:", d.groupby(["kind", "strike_type"]).size().to_dict(), "unparsed", int(bad.sum()))
    d = d[~bad]
    ext = {c: (pd.read_parquet(R / f"dvol_{c}.parquet"), pd.read_parquet(R / f"cb1h_{c}.parquet")) for c in ("BTC", "ETH")}
    rows = []
    for ev, g in d.groupby("event"):
        a = g.iloc[0]
        open_ts, close_ts = ts(g.open_time.min()), max(window_end(x) for x in g.rules_primary)
        cands = {t: json.loads((R / "candles" / f"{t}.json").read_text()) for t in g.ticker}
        day0 = pd.Timestamp(open_ts, unit="s", tz="UTC").normalize()
        for day in pd.date_range(day0, pd.Timestamp(close_ts, unit="s", tz="UTC"), freq="D"):
            T = int(day.timestamp()) + T_HOUR_UTC * 3600
            if T <= open_ts or T >= close_ts:
                continue
            dv, cb = ext[a.asset]
            dvr = dv[dv.start + 3600 <= T]
            cbr = cb[cb.start + 3600 <= T]
            if dvr.empty or cbr.empty or cbr.start.iloc[-1] + 3600 != T or T - (dvr.start.iloc[-1] + 3600) > 3 * 3600:
                continue
            sigma = float(dvr.c.iloc[-1]) / 100
            S = float(cbr.close.iloc[-1])
            hist = cbr[cbr.start >= T - M.BOOT_DAYS * 86400]
            blocks = M.hourly_blocks(hist.start.values, hist.open.values, hist.high.values, hist.low.values, hist.close.values)
            tau_ev = None
            ext2 = None
            for r in g.itertuples():
                c_ts = ts(r.close_time)
                if not (ts(r.open_time) < T < c_ts):
                    continue                              # market not open at T (early-closed or not yet listed)
                pre = [x for x in cands[r.ticker] if x["end_period_ts"] <= T]
                if not pre:
                    continue
                yb, ya = num(pre[-1], "yes_bid", "close"), num(pre[-1], "yes_ask", "close")
                tau = (window_end(r.rules_primary) - T) / (365 * 86400)
                # tau runs to the scheduled END of the measurement window (parsed from the rules), not the
                # market's own close_time (which moves earlier on a touch -> would leak the outcome)
                dd = M.dist(r.kind, S, r.strike)
                row = {"event": ev, "series": r.series, "ticker": r.ticker, "asset": r.asset, "kind": r.kind,
                       "strike": r.strike, "T": T, "S": S, "sigma": sigma, "tau_d": tau * 365, "d": dd,
                       "yes_bid": yb, "yes_ask": ya, "quote_age_h": (T - pre[-1]["end_period_ts"]) / 3600,
                       "result": r.result}
                if dd > 0:
                    if ext2 is None or tau_ev != tau:
                        ext2 = M.m2_extremes(blocks, tau, seed=T)
                        tau_ev = tau
                    row.update(p0=M.p_m0(dd, sigma, tau) if r.kind == "max" else M.p_m0_min(dd, sigma, tau),
                               p1=M.p_m1(dd, sigma, tau), p2=M.p_m2(ext2, r.kind, dd), nblocks=len(blocks))
                rows.append(row)
        print(ev, len(rows), flush=True)
    f = pd.DataFrame(rows)
    f.to_parquet(R / "frame.parquet")
    print(len(f), "rows;", f.ticker.nunique(), "markets;", f.event.nunique(), "events;",
          "touched_proxy rows", int((f.d <= 0).sum()))


# ------------------------------------------------------------------ mechanics check
def cmd_mech():
    d = pd.read_parquet(R / "markets.parquet")
    d["asset"] = d.series.map(lambda s: SERIES[s][0])
    d["kind"] = d.series.map(lambda s: SERIES[s][1])
    d["strike"] = np.where(d.kind == "max", d.floor_strike, d.cap_strike).astype(float)
    end = d.rules_primary.map(window_end)
    early = d[(d.result == "yes") & (d.close_time.map(ts) < end - 3600)]
    out = []
    for r in early.itertuples():
        c = ts(r.close_time)
        m = coinbase(f"{r.asset}-USD", c - 3 * 86400, c + 3600, gran=60)
        v = (m.open + m.high + m.low + m.close) / 4
        hit = (v > r.strike) if r.kind == "max" else (v < r.strike)
        tick = (m.high > r.strike) if r.kind == "max" else (m.low < r.strike)
        f1 = int(m.start[hit].iloc[0]) if hit.any() else None
        f2 = int(m.start[tick].iloc[0]) if tick.any() else None
        out.append({"ticker": r.ticker, "close": r.close_time, "first_proxy_min": f1, "first_tick": f2,
                    "lag_proxy_h": (c - f1) / 3600 if f1 else None, "lag_tick_h": (c - f2) / 3600 if f2 else None})
        print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(R / "mech.csv", index=False)


# ------------------------------------------------------------------ calibration (outcomes)
def cmd_calib():
    f = pd.read_parquet(R / "frame.parquet")
    f = f[(f.d > 0) & f.yes_bid.notna() & f.yes_ask.notna() & (f.yes_ask > f.yes_bid)].copy()
    f["mid"] = (f.yes_bid + f.yes_ask) / 2
    f["y"] = (f.result == "yes").astype(float)
    lines = [f"rows {len(f)}  markets {f.ticker.nunique()}  events {f.event.nunique()}  months "
             f"{pd.to_datetime(f['T'], unit='s').dt.to_period('M').nunique()}  (rows are autocorrelated within market)"]
    bins = [0, .02, .05, .10, .20, .35, .50, .65, .80, .90, .95, .98, 1.0001]
    for key in ("p1", "p2", "mid"):
        f["b"] = pd.cut(f[key], bins, right=False)
        t = f.groupby("b", observed=True).agg(n=("y", "size"), markets=("ticker", "nunique"), p0=("p0", "mean"),
                                              p1=("p1", "mean"), p2=("p2", "mean"), mid=("mid", "mean"), real=("y", "mean"))
        lines += ["", f"binned by {key}:", t.round(4).to_string()]
    br = {k: float(((f[k] - f.y) ** 2).mean()) for k in ("p0", "p1", "p2", "mid")}
    lines += ["", "Brier (all rows): " + json.dumps({k: round(v, 5) for k, v in br.items()})]
    # by event (cluster-level view): mean Brier per event, and mean signed (model - mid)
    g = f.groupby("event").apply(lambda x: pd.Series({k: ((x[k] - x.y) ** 2).mean() for k in ("p0", "p1", "p2", "mid")}
                                                     | {"n": len(x), "p1_minus_mid": (x.p1 - x.mid).mean(),
                                                        "p2_minus_mid": (x.p2 - x.mid).mean()}))
    lines += ["", "by event:", g.round(4).to_string()]
    wins = {k: int((g[k] < g["mid"]).sum()) for k in ("p0", "p1", "p2")}
    lines += ["", f"events where model Brier < mid Brier (of {len(g)}): {wins}"]
    for kind in ("max", "min"):
        s = f[f.kind == kind]
        if len(s):
            lines.append(f"{kind}: rows {len(s)} mean p1 {s.p1.mean():.4f} p2 {s.p2.mean():.4f} mid {s.mid.mean():.4f} "
                         f"real {s.y.mean():.4f}")
    txt = "\n".join(lines)
    (R / "calibration.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    {"markets": cmd_markets, "candles": cmd_candles, "ext": cmd_ext, "frame": cmd_frame, "mech": cmd_mech,
     "calib": cmd_calib}[sys.argv[1]]()
