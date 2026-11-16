"""CB1 crypto barrier FORWARD paper logger (READ-ONLY; notes/crypto_barrier.md; edgelab trial cb1_crypto_barrier_taker).

    python scripts/crypto_barrier_forward.py tick [--force-now]   # decide once per UTC day from 16:00 UTC
    python scripts/crypto_barrier_forward.py settle                # settle logged picks (GET /markets/{ticker})
    python scripts/crypto_barrier_forward.py status                # counts

Decide (order fixed by the prereg; all model inputs strictly older than the Kalshi GET):
  1. Deribit DVOL (BTC, ETH) hourly candles -> last CLOSE with candle end <= now
  2. Coinbase <ASSET>-USD hourly candles over the trailing 730 d (bootstrap M2), complete hours only
  3. Coinbase /ticker spot
  4. Kalshi GET /events?series_ticker=S&status=open&with_nested_markets=true per series (live yes_bid / yes_ask)
  Models + rule = scripts/crypto_barrier_model.py (frozen). Window end parsed from rules_primary
  (scripts/crypto_barrier.py window_end), never from close_time.
Output (append-only) ~/.claude/quant-sports/data/kx_forward/crypto_barrier.jsonl:
  day    one per UTC date: status, timing, counts, inputs
  pick   one per rule-firing TRIAL market (first firing only): side, ask, p's, edge, status bet | off_time
  settle one per pick once the market is settled: result
and crypto_barrier_quotes.jsonl: every rung of every series (trial + diagnostic SOL / yearly) with quote and P's.
--force-now writes to *_manual.jsonl only. No orders, no keys, no portfolio endpoints.
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
sys.path.insert(0, str(HERE))
from kx_throttle import kget  # noqa: E402
import crypto_barrier_model as M  # noqa: E402
from crypto_barrier import window_end  # noqa: E402

TRIAL = {"KXBTCMAXMON": ("BTC", "max"), "KXBTCMINMON": ("BTC", "min"), "KXETHMAXMON": ("ETH", "max"),
         "KXETHMINMON": ("ETH", "min"), "KXBTCMAXM": ("BTC", "max")}
DIAG = {"KXSOLMAXMON": ("SOL", "max"), "KXSOLMINMON": ("SOL", "min"), "KXBTCMAXY": ("BTC", "max"),
        "KXBTCMINY": ("BTC", "min"), "KXETHMAXY": ("ETH", "max"), "KXETHMINY": ("ETH", "min")}
DECIDE_H = 16
BET_WINDOW_MIN = 30
FWD_START = "2026-11-16"
OUT = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"
SUFFIX = ""
_S = requests.Session()
_S.headers["User-Agent"] = "quant-sports-readonly-paper"


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def path(name: str) -> Path:
    return OUT / f"{name}{SUFFIX}.jsonl"


def append(p: Path, rows: list[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_js) + "\n")


def _js(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _get(url, params, tries=5):
    for k in range(tries):
        try:
            r = _S.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1 + k)
    raise RuntimeError(f"GET failed {url} {params}")


def dvol_last(cur: str, now: float) -> tuple[float, int]:
    j = _get("https://www.deribit.com/api/v2/public/get_volatility_index_data",
             {"currency": cur, "start_timestamp": int((now - 6 * 3600) * 1000), "end_timestamp": int(now * 1000),
              "resolution": "3600"})
    rows = [r for r in j["result"]["data"] if r[0] // 1000 + 3600 <= now]
    if not rows:
        raise RuntimeError(f"no closed DVOL candle for {cur}")
    r = max(rows, key=lambda x: x[0])
    return float(r[4]) / 100, int(r[0] // 1000 + 3600)


def cb_hourly(product: str, now: float) -> pd.DataFrame:
    end = int(now // 3600 * 3600)                 # last COMPLETE hour ends here
    t0 = end - (M.BOOT_DAYS + 2) * 86400
    out, a = [], t0
    while a < end:
        b = min(end, a + 300 * 3600)
        out += _get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                    {"granularity": 3600, "start": iso(a), "end": iso(b)})
        time.sleep(0.12)
        a = b
    d = pd.DataFrame(out, columns=["start", "low", "high", "open", "close", "vol"]).drop_duplicates("start")
    d = d[d.start + 3600 <= end].sort_values("start").reset_index(drop=True)
    return d


def cb_spot(product: str) -> tuple[float, float]:
    j = _get(f"https://api.exchange.coinbase.com/products/{product}/ticker", {})
    return float(j["price"]), pd.Timestamp(j["time"]).timestamp()


def q(m: dict, k: str) -> float | None:
    v = m.get(k + "_dollars")
    if v is not None:
        return float(v)
    v = m.get(k)
    return None if v is None else float(v) / 100


def strike_of(m: dict, kind: str) -> float | None:
    v = m.get("floor_strike") if kind == "max" else m.get("cap_strike")
    return None if v is None else float(v)


def decide_day(now: float, uday: str) -> dict:
    t_start = time.time()
    inputs = {}
    for cur in ("BTC", "ETH"):
        sig, sig_end = dvol_last(cur, now)
        inputs[cur] = {"sigma": sig, "dvol_end": sig_end}
    for cur in ("BTC", "ETH", "SOL"):
        h = cb_hourly(f"{cur}-USD", now)
        hist = h[h.start >= h.start.max() + 3600 - M.BOOT_DAYS * 86400]
        blocks = M.hourly_blocks(hist.start.values, hist.open.values, hist.high.values, hist.low.values, hist.close.values)
        spot, spot_t = cb_spot(f"{cur}-USD")
        inputs.setdefault(cur, {}).update(blocks=blocks, nblocks=len(blocks), spot=spot, spot_t=spot_t,
                                          hist_end=int(h.start.max() + 3600))
    inputs_as_of = max([v.get("dvol_end", 0) for v in inputs.values()] + [v["spot_t"] for v in inputs.values()]
                       + [v["hist_end"] for v in inputs.values()])
    prior = {r["ticker"] for r in read(path("crypto_barrier")) if r.get("kind") == "pick"}
    quotes, picks, nser = [], [], 0
    seed = int(now // 3600 * 3600)
    for ser, (cur, kind) in {**TRIAL, **DIAG}.items():
        k_start = time.time()
        if k_start <= inputs_as_of:
            time.sleep(inputs_as_of - k_start + 0.01)
            k_start = time.time()
        evs = (kget("/events", {"series_ticker": ser, "status": "open", "with_nested_markets": "true"}) or {}).get("events", [])
        k_end = time.time()
        nser += 1
        trial = ser in TRIAL
        for ev in evs:
            ext = {}
            for m in ev.get("markets", []):
                row = {"kind": "quote", "uday": uday, "series": ser, "event": ev["event_ticker"], "ticker": m["ticker"],
                       "trial": trial, "asset": cur, "barrier": kind, "status_mkt": m.get("status"),
                       "yes_bid": q(m, "yes_bid"), "yes_ask": q(m, "yes_ask"), "quote_time": iso(k_start),
                       "decided_at": iso(k_end), "close_time": m.get("close_time")}
                try:
                    wend = window_end(m.get("rules_primary") or "")
                except Exception:  # noqa: BLE001
                    row["skip"] = "window_parse"
                    quotes.append(row)
                    continue
                B = strike_of(m, kind)
                S = inputs[cur]["spot"]
                tau = (wend - k_start) / (365 * 86400)
                row.update(window_end=iso(wend), strike=B, spot=S, tau_d=tau * 365)
                if B is None or m.get("status") != "active" or tau <= 0:
                    row["skip"] = "not_active_or_no_strike"
                    quotes.append(row)
                    continue
                d = M.dist(kind, S, B)
                row["d"] = d
                if d <= 0:
                    row["skip"] = "touched_proxy"
                    quotes.append(row)
                    continue
                if round(tau, 9) not in ext:
                    ext[round(tau, 9)] = M.m2_extremes(inputs[cur]["blocks"], tau, seed=seed)
                p2 = M.p_m2(ext[round(tau, 9)], kind, d)
                sig = inputs[cur].get("sigma")
                p0 = p1 = None
                if sig:
                    p0 = M.p_m0(d, sig, tau) if kind == "max" else M.p_m0_min(d, sig, tau)
                    p1 = M.p_m1(d, sig, tau)
                row.update(sigma=sig, p0=p0, p1=p1, p2=p2)
                dec = M.decide(row["yes_bid"], row["yes_ask"], p1, p2, tau * 365) if (trial and p1 is not None) else None
                row["fires"] = bool(dec)
                quotes.append(row)
                if dec and m["ticker"] not in prior:
                    late = (k_start - (pd.Timestamp(uday, tz="UTC").timestamp() + DECIDE_H * 3600)) / 60
                    picks.append({**{k: row[k] for k in ("series", "event", "ticker", "asset", "barrier", "strike", "spot",
                                                         "sigma", "tau_d", "d", "p0", "p1", "p2", "yes_bid", "yes_ask",
                                                         "quote_time", "decided_at", "window_end")},
                                  "kind": "pick", "date": uday, "side": dec["side"], "ask": dec["ask"],
                                  "p_cons": dec["p_cons"], "edge": dec["edge"], "fee": M.fee(dec["ask"]),
                                  "inputs_as_of": iso(inputs_as_of),
                                  "cluster": "crypto_month|" + pd.Timestamp(wend - 1, unit="s", tz="UTC")
                                  .tz_convert("America/New_York").strftime("%Y-%m"),
                                  "status": "bet" if 0 <= late < BET_WINDOW_MIN else "off_time", "late_min": late})
    append(path("crypto_barrier_quotes"), quotes)
    append(path("crypto_barrier"), picks)
    day = {"kind": "day", "uday": uday, "status": "decided", "started": iso(t_start), "finished": iso(time.time()),
           "inputs_as_of": iso(inputs_as_of), "series": nser, "quotes": len(quotes),
           "scored": sum(1 for r in quotes if "p2" in r), "fires": sum(1 for r in quotes if r.get("fires")),
           "picks": len(picks), "bets": sum(1 for p in picks if p["status"] == "bet"),
           "inputs": {c: {k: v for k, v in x.items() if k != "blocks"} for c, x in inputs.items()}}
    append(path("crypto_barrier"), [day])
    return day


def cmd_tick(force: bool):
    now = time.time()
    uday = datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")
    if not force:
        if uday < FWD_START or datetime.fromtimestamp(now, timezone.utc).hour < DECIDE_H:
            return
        if any(r.get("kind") == "day" and r.get("uday") == uday for r in read(path("crypto_barrier"))):
            return
    d = decide_day(now, uday)
    print(json.dumps({k: v for k, v in d.items() if k != "inputs"}))


def cmd_settle():
    rs = read(path("crypto_barrier"))
    done = {r["ticker"] for r in rs if r.get("kind") == "settle"}
    todo = [r for r in rs if r.get("kind") == "pick" and r["ticker"] not in done]
    out = []
    for r in todo:
        m = (kget(f"/markets/{r['ticker']}") or {}).get("market", {})
        if m.get("result") in ("yes", "no") and m.get("status") in ("settled", "finalized", "determined"):
            out.append({"kind": "settle", "ticker": r["ticker"], "result": m["result"], "status_mkt": m.get("status"),
                        "close_time": m.get("close_time"), "settled_seen": iso(time.time())})
    append(path("crypto_barrier"), out)
    if out:
        print(f"settled {len(out)} of {len(todo)} open picks")


def cmd_status():
    rs = read(path("crypto_barrier"))
    k = pd.Series([r.get("kind") for r in rs]).value_counts().to_dict()
    days = [r for r in rs if r.get("kind") == "day"]
    print(json.dumps({"rows": k, "last_day": {x: days[-1].get(x) for x in ("uday", "status", "quotes", "fires", "picks", "bets")}
                      if days else None}))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if "--force-now" in sys.argv:
        SUFFIX = "_manual"
    if cmd == "tick":
        cmd_tick("--force-now" in sys.argv)
    elif cmd == "settle":
        cmd_settle()
    elif cmd == "status":
        cmd_status()
    else:
        raise SystemExit(f"unknown {cmd}")
