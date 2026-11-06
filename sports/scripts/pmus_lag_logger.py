"""LIVE Kalshi -> Polymarket US lag logger (paper only, never orders). Forward data for pmus_lag_t1/t2 (notes/pmus_lag.md).

    python scripts/pmus_lag_logger.py run [--minutes 120] [--start-at 2026-11-07T00:05Z] [--leagues nba,nhl,cbb,cfb,nfl]
    python scripts/pmus_lag_logger.py report [FILES...]        # jumps, PM US ask staleness at +1/3/5/10 s, paper EV

Feeds:
  Kalshi  -- WebSocket wss://api.elections.kalshi.com/trade-api/ws/v2, channels `trade` + `ticker`, signed with the
             user's existing key (KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH read from ~/.claude/quant-kalshi/.env, read
             only, never modified). Market data only; no order endpoint is touched. If the WS fails -> REST poll
             /markets?tickers= every 2 s (public).
  PM US   -- the Markets WebSocket needs an API key we do not have (docs websocket/overview), so: public REST
             gateway /v1/markets/{slug}/bbo, round-robin ~8 req/s total (< 20/s public limit). Responses are served
             through Cloudflare with `Cache-Control: max-age=30`, so a view can be up to ~30 s old. Every record keeps
             Date, Age and cf-cache-status; as_of = Date - Age is the time the view was generated. No cache busting.

Records (jsonl data/pmus_lag/live/<UTC date>.jsonl):
  k:   recv, src (ws_trade | ws_ticker | rest), ticker, price (trade yes price) or yes_bid/yes_ask, ts (exchange ts)
  p:   recv, slug, as_of, age, cf, ask (YES best ask), bid (YES best bid), long/short quotes, sample_ts (lastPriceSample)
  map: kalshi ticker (PM-long team) <-> PM US slug for each matched live game
Jump (same as the historical frame): Kalshi trade YES price differs >= 4c from the last trade <= 10 s earlier,
0.05-0.95, one per game per 60 s. Report: for k in 1,3,5,10 s, the freshest PM US view RECEIVED by t+k --
staleness = (t+k) - as_of, whether as_of > t (the view could reflect the jump), side ask move vs the last view
generated <= t, and paper EV = buy the side at the t+3 s view's ask + fee if that view is post-jump, mark-to-market at
the first view generated >= t+60 / t+300 (mid); settlement EV in a later report run (PM US /settlement).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import glob
import json
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import requests

OUT = ROOT / "data" / "pmus_lag" / "live"
OUT.mkdir(parents=True, exist_ok=True)
ENV = Path.home() / ".claude" / "quant-kalshi" / ".env"
KAPI = "https://api.elections.kalshi.com/trade-api/v2"
KWS = "wss://api.elections.kalshi.com/trade-api/ws/v2"
G = "https://gateway.polymarket.us/v1"
H = {"User-Agent": "Mozilla/5.0 (research; read-only)"}
KSER = {"nba": "KXNBAGAME", "nhl": "KXNHLGAME", "nfl": "KXNFLGAME", "cfb": "KXNCAAFGAME", "cbb": "KXNCAAMBGAME",
        "mlb": "KXMLBGAME"}
FEE = 0.0695
JUMP_C, JUMP_S, DEDUP_S = 0.04, 10, 60

_lock = threading.Lock()
_fh = None


def log(rec: dict) -> None:
    global _fh
    with _lock:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        p = OUT / f"{day}.jsonl"
        if _fh is None or _fh.name != str(p):
            _fh = open(p, "a")
        _fh.write(json.dumps(rec) + "\n")
        _fh.flush()


# ------------------------------------------------------------------ matching (same code logic as pmus_lag.py)
def live_games(leagues, now=None, horizon_h=3.0) -> list[dict]:
    from pmus_lag import _codes, _norm
    now = now or pd.Timestamp.now(tz="UTC")
    out = []
    for lg in leagues:
        evs = requests.get(f"https://gateway.polymarket.us/v2/leagues/{lg}/events", params={"limit": 300},
                           headers=H, timeout=30).json().get("events", [])
        km = []
        cursor = None
        while True:
            p = {"series_ticker": KSER[lg], "status": "open", "limit": 1000}
            if cursor:
                p["cursor"] = cursor
            d = requests.get(f"{KAPI}/markets", params=p, headers=H, timeout=30).json()
            km += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
        kev = {}
        for m in km:
            kev.setdefault(m["event_ticker"], []).append(m)
        for e in evs:
            st = pd.Timestamp(e["startTime"])
            if e.get("closed") or not (now - pd.Timedelta(hours=4) <= st <= now + pd.Timedelta(hours=horizon_h)):
                continue
            ml = [m for m in e.get("markets", []) if m.get("marketType") == "moneyline" and m["slug"].startswith("aec-")]
            if not ml:
                continue
            sides = ml[0].get("marketSides") or []
            L = [s for s in sides if s.get("long") is True]
            S = [s for s in sides if s.get("long") is False]
            if len(L) != 1 or len(S) != 1:
                continue
            ta, tb = L[0].get("team") or {}, S[0].get("team") or {}
            ca, cb = _codes(ta.get("abbreviation"), ta.get("displayAbbreviation")), _codes(tb.get("abbreviation"), tb.get("displayAbbreviation"))
            etd = st.tz_convert("America/New_York")
            days = {(etd + pd.Timedelta(days=d)).strftime("%y%b%d").upper() for d in (0, -1)}
            cands = []
            for et, ms in kev.items():
                if len(ms) != 2 or et.split("-", 1)[1][:7] not in days:
                    continue
                for a, b in ((ms[0], ms[1]), (ms[1], ms[0])):
                    ka, kb = a["ticker"].split("-")[-1], b["ticker"].split("-")[-1]
                    if (ka in ca or _norm(a.get("yes_sub_title")) == _norm(ta.get("safeName") or ta.get("name"))) and \
                       (kb in cb or _norm(b.get("yes_sub_title")) == _norm(tb.get("safeName") or tb.get("name"))):
                        cands.append((et, a["ticker"], b["ticker"]))
            if len(cands) == 1:
                out.append({"lg": lg, "slug": ml[0]["slug"], "start": e["startTime"], "ticker": cands[0][1],
                            "opp_ticker": cands[0][2], "event_ticker": cands[0][0]})
    return out


# ------------------------------------------------------------------ Kalshi feed
def _env() -> dict:
    e = {}
    for ln in ENV.read_text().splitlines():
        if "=" in ln and not ln.lstrip().startswith("#"):
            k, v = ln.split("=", 1)
            if k.strip() in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH"):
                e[k.strip()] = v.strip().strip('"').strip("'")
    return e


def _ws_headers() -> dict:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519, padding, rsa
    e = _env()
    key = serialization.load_pem_private_key(Path(e["KALSHI_PRIVATE_KEY_PATH"]).expanduser().read_bytes(), password=None)
    ts = str(int(time.time() * 1000))
    msg = (ts + "GET" + "/trade-api/ws/v2").encode()
    if isinstance(key, ed25519.Ed25519PrivateKey):
        sig = key.sign(msg)
    elif isinstance(key, rsa.RSAPrivateKey):
        sig = key.sign(msg, padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
                       hashes.SHA256())
    else:
        raise ValueError("unsupported key")
    return {"KALSHI-ACCESS-KEY": e["KALSHI_KEY_ID"], "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode()}


def _f(x, cents=False):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v / 100 if cents else v


async def kalshi_ws(tickers: list[str], stop: float, state: dict) -> None:
    import websockets
    while time.time() < stop:
        try:
            async with websockets.connect(KWS, additional_headers=_ws_headers(), ping_interval=10) as ws:
                await ws.send(json.dumps({"id": 1, "cmd": "subscribe",
                                          "params": {"channels": ["trade", "ticker"], "market_tickers": tickers}}))
                state["kws"] = "up"
                log({"t": "info", "recv": time.time(), "msg": f"kalshi ws subscribed {len(tickers)}"})
                while time.time() < stop:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        continue
                    now = time.time()
                    m = json.loads(raw)
                    typ, msg = m.get("type"), m.get("msg") or {}
                    if typ == "trade":
                        px = _f(msg.get("yes_price_dollars")) if msg.get("yes_price_dollars") is not None else _f(msg.get("yes_price"), True)
                        log({"t": "k", "src": "ws_trade", "recv": now, "ticker": msg.get("market_ticker"), "price": px,
                             "count": msg.get("count_fp", msg.get("count")), "taker": msg.get("taker_side"), "ts": msg.get("ts")})
                    elif typ == "ticker":
                        yb = _f(msg.get("yes_bid_dollars")) if msg.get("yes_bid_dollars") is not None else _f(msg.get("yes_bid"), True)
                        ya = _f(msg.get("yes_ask_dollars")) if msg.get("yes_ask_dollars") is not None else _f(msg.get("yes_ask"), True)
                        log({"t": "k", "src": "ws_ticker", "recv": now, "ticker": msg.get("market_ticker"), "yes_bid": yb,
                             "yes_ask": ya, "ts": msg.get("ts")})
                    elif typ in ("error", "subscribed"):
                        log({"t": "info", "recv": now, "msg": raw[:300]})
        except Exception as ex:   # noqa: BLE001
            state["kws"] = f"down {type(ex).__name__}: {ex}"
            log({"t": "info", "recv": time.time(), "msg": f"kalshi ws error {type(ex).__name__}: {str(ex)[:200]}"})
            state["ws_fail"] = state.get("ws_fail", 0) + 1
            if state["ws_fail"] >= 3:
                return
            await asyncio.sleep(5)


def kalshi_rest(tickers: list[str], stop: float) -> None:
    last = {}
    while time.time() < stop:
        try:
            d = requests.get(f"{KAPI}/markets", params={"tickers": ",".join(tickers)}, headers=H, timeout=10).json()
            now = time.time()
            for m in d.get("markets", []):
                rec = (m.get("last_price_dollars"), m.get("yes_bid_dollars"), m.get("yes_ask_dollars"))
                if last.get(m["ticker"]) != rec:
                    last[m["ticker"]] = rec
                    log({"t": "k", "src": "rest", "recv": now, "ticker": m["ticker"], "price": _f(rec[0]),
                         "yes_bid": _f(rec[1]), "yes_ask": _f(rec[2])})
        except Exception as ex:   # noqa: BLE001
            log({"t": "info", "recv": time.time(), "msg": f"kalshi rest error {ex}"[:200]})
        time.sleep(2)


# ------------------------------------------------------------------ PM US feed
def pm_poll(slugs: list[str], stop: float, rate: float = 8.0) -> None:
    s = requests.Session()
    i = 0
    while time.time() < stop:
        slug = slugs[i % len(slugs)]
        i += 1
        t0 = time.time()
        try:
            r = s.get(f"{G}/markets/{slug}/bbo", headers=H, timeout=10)
            now = time.time()
            md = (r.json() or {}).get("marketData", {})
            date = r.headers.get("Date")
            dts = parsedate_to_datetime(date).timestamp() if date else None
            age = float(r.headers.get("Age", 0) or 0)
            g = lambda k: _f((md.get(k) or {}).get("value"))  # noqa: E731
            lps = md.get("lastPriceSample") or {}
            log({"t": "p", "recv": now, "slug": slug, "status": r.status_code, "date": dts, "age": age,
                 "cf": r.headers.get("cf-cache-status"), "as_of": None if dts is None else dts - age,
                 "ask": g("bestAsk"), "bid": g("bestBid"), "long": g("longQuote"), "short": g("shortQuote"),
                 "last": g("lastTradePx"), "sample_ts": lps.get("ts"), "state": md.get("state")})
        except Exception as ex:   # noqa: BLE001
            log({"t": "info", "recv": time.time(), "msg": f"pm error {slug} {ex}"[:200]})
        time.sleep(max(0.0, 1.0 / rate - (time.time() - t0)))


def cmd_run(a) -> int:
    if a.start_at:
        t = pd.Timestamp(a.start_at).timestamp()
        while time.time() < t:
            time.sleep(min(60, t - time.time()))
    stop = time.time() + a.minutes * 60
    leagues = a.leagues.split(",")
    games, refreshed = [], 0.0
    log({"t": "info", "recv": time.time(), "msg": f"start minutes={a.minutes} leagues={leagues}"})
    while time.time() < stop:
        games = live_games(leagues, horizon_h=a.horizon)
        for gm in games:
            log({"t": "map", "recv": time.time(), **gm})
        print(f"{datetime.now(timezone.utc):%H:%M:%S} matched {len(games)} games", flush=True)
        if games:
            break
        time.sleep(300)
    if not games:
        return 1
    seg_stop = min(stop, time.time() + 45 * 60)       # re-match every 45 min (new tip-offs)
    while True:
        tickers = sorted({g["ticker"] for g in games} | {g["opp_ticker"] for g in games})
        slugs = [g["slug"] for g in games]
        state = {}
        th = threading.Thread(target=pm_poll, args=(slugs, seg_stop), daemon=True)
        th.start()
        if a.no_ws:
            kalshi_rest(tickers, seg_stop)
        else:
            asyncio.run(kalshi_ws(tickers, seg_stop, state))
            if time.time() < seg_stop:          # ws gave up -> public REST fallback
                log({"t": "info", "recv": time.time(), "msg": "falling back to kalshi REST"})
                kalshi_rest(tickers, seg_stop)
        th.join()
        if time.time() >= stop:
            break
        new = live_games(leagues, horizon_h=a.horizon)
        for gm in new:
            log({"t": "map", "recv": time.time(), **gm})
        games = new or games
        seg_stop = min(stop, time.time() + 45 * 60)
    log({"t": "info", "recv": time.time(), "msg": "stop"})
    return 0


# ------------------------------------------------------------------ report
def _load(files):
    recs = []
    for f in files:
        with open(f) as fh:
            for ln in fh:
                try:
                    recs.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return recs


def cmd_report(a) -> int:
    files = a.files or sorted(glob.glob(str(OUT / "*.jsonl")))
    recs = _load(files)
    mp = {}
    for r in recs:
        if r.get("t") == "map":
            mp[r["ticker"]] = r
    k = pd.DataFrame([r for r in recs if r.get("t") == "k"])
    p = pd.DataFrame([r for r in recs if r.get("t") == "p" and r.get("status") == 200])
    print(f"files {len(files)}; games mapped {len(mp)}; kalshi records {len(k)}; pm views {len(p)}")
    if not len(k) or not len(p):
        return 0
    print("kalshi sources:", k.src.value_counts().to_dict())
    print("pm cf-cache-status:", p.cf.value_counts().to_dict(), "| age s quantiles",
          np.percentile(p.age, [10, 50, 90]).round(1).tolist(), "| recv - as_of median",
          round(float((p.recv - p.as_of).median()), 2))
    tr = k[(k.src.isin(["ws_trade", "rest"])) & k.price.notna() & k.ticker.isin(list(mp))].sort_values("recv")
    jumps = []
    for tk, g in tr.groupby("ticker"):
        ts, px = g.recv.to_numpy(float), g.price.to_numpy(float)
        last = -1e18
        for i in range(len(ts)):
            if ts[i] < last + DEDUP_S or not (0.05 <= px[i] <= 0.95):
                continue
            j = np.searchsorted(ts, ts[i] - JUMP_S, side="right") - 1
            if j >= 0 and abs(px[i] - px[j]) >= JUMP_C - 1e-9:
                jumps.append({"ticker": tk, "slug": mp[tk]["slug"], "t": ts[i], "p": px[i], "sgn": np.sign(px[i] - px[j])})
                last = ts[i]
    J = pd.DataFrame(jumps)
    print(f"jumps (>= {JUMP_C*100:.0f}c in {JUMP_S}s, Kalshi trades, PM-long ticker): {len(J)} in {J.slug.nunique() if len(J) else 0} games")
    if not len(J):
        return 0
    P = {s: g.sort_values("recv") for s, g in p.groupby("slug")}
    rows = []
    for j in J.itertuples():
        g = P.get(j.slug)
        if g is None:
            continue
        side_ask = lambda x: x["ask"] if j.sgn > 0 else (None if x["bid"] is None or pd.isna(x["bid"]) else 1 - x["bid"])  # noqa: E731
        pre = g[g.as_of <= j.t]
        if not len(pre):
            continue
        a0 = side_ask(pre.iloc[-1])
        r = {"slug": j.slug, "t": j.t, "sgn": j.sgn, "k_side": j.p if j.sgn > 0 else 1 - j.p, "ask0": a0}
        for kk in (1, 3, 5, 10):
            v = g[g.recv <= j.t + kk]
            v = v.iloc[-1] if len(v) else None
            if v is None:
                continue
            r[f"stale{kk}"] = j.t + kk - v.as_of
            r[f"post{kk}"] = bool(v.as_of > j.t)
            r[f"ask{kk}"] = side_ask(v)
        for hh in (60, 300):
            w = g[g.as_of >= j.t + hh]
            if len(w):
                x = w.iloc[0]
                mid = (x["ask"] + x["bid"]) / 2 if x["ask"] is not None and x["bid"] is not None else np.nan
                r[f"mid{hh}"] = mid if j.sgn > 0 else 1 - mid
        rows.append(r)
    R = pd.DataFrame(rows)
    print(f"jumps with a PM US view before t: {len(R)}")
    for kk in (1, 3, 5, 10):
        if f"stale{kk}" not in R:
            continue
        s = R[f"stale{kk}"].dropna()
        mv = (R[f"ask{kk}"] - R["ask0"]).dropna()
        print(f"  t+{kk:>2}s: view age median {s.median():.1f}s p90 {s.quantile(.9):.1f}s | post-jump view share "
              f"{R[f'post{kk}'].mean():.2f} | side ask move vs pre-jump view mean {mv.mean()*100:+.2f}c (n {len(mv)})")
    b = R[R.get("post3", False) == True].copy()  # noqa: E712
    if len(b):
        b["cost"] = b.ask3 + FEE * b.ask3 * (1 - b.ask3)
        for hh in (60, 300):
            if f"mid{hh}" in b:
                c = (b[f"mid{hh}"] - b.cost).dropna()
                print(f"  paper EV (buy t+3 s post-jump view ask + fee) vs mid +{hh}s: {c.mean()*100:+.2f}c/contract n {len(c)}")
        st = (b.k_side - b.cost)
        print(f"  Kalshi side price - PM US cost at t+3 s: mean {st.mean()*100:+.2f}c; share >= 2c {(st >= .02).mean():.2f}")
    else:
        print("  no jump had a post-jump PM US view by t+3 s (cache) -> paper EV not computable")
    R.to_csv(OUT / "report_jumps.csv", index=False)
    print(f"rows -> {OUT / 'report_jumps.csv'}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--minutes", type=float, default=120)
    r.add_argument("--start-at", default=None)
    r.add_argument("--leagues", default="nba,nhl,cbb,cfb,nfl")
    r.add_argument("--no-ws", action="store_true")
    r.add_argument("--horizon", type=float, default=3.0)
    q = sp.add_parser("report")
    q.add_argument("files", nargs="*")
    a = ap.parse_args()
    raise SystemExit(cmd_run(a) if a.cmd == "run" else cmd_report(a))
