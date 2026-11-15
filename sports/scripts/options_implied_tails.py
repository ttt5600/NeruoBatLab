"""options_implied_tails: Kalshi daily crypto / index ladders vs options-implied tail probabilities.
Pre-registration: notes/options_implied_tails.md (written before any quote / outcome was fetched).

    python scripts/options_implied_tails.py markets   # DEV market lists (result kept in a hidden column)
    python scripts/options_implied_tails.py deribit   # Deribit option trades [T-60m, T) per DEV crypto day (cached)
    python scripts/options_implied_tails.py implied   # options-implied P(YES) per market (inputs only)
    python scripts/options_implied_tails.py candles   # Kalshi hourly candles for markets with a tail side (cached)
    python scripts/options_implied_tails.py frame     # edgelab frame + tape -> data/opt_tails/
Read-only GETs only (Kalshi via kx_throttle, Deribit public history host). No orders.
"""
from __future__ import annotations

import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import norm, t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "opt_tails"
D.mkdir(parents=True, exist_ok=True)
CRYPTO = ["KXBTCD", "KXETHD"]
INDEX = ["KXINX", "KXNASDAQ100", "KXINXU", "KXNASDAQ100U"]
DEV_END = pd.Timestamp("2026-04-10")
IDX_START = pd.Timestamp("2024-02-01")
ET = "America/New_York"
DERIBIT = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
_S = requests.Session()


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


# ------------------------------------------------------------------ markets
def series_markets(series: str) -> list[dict]:
    out, cur = [], None
    while True:
        p = {"series_ticker": series, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = kget("/historical/markets", p) or {}
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            break
    return out


LEGACY = {"KXINX": "INX", "KXNASDAQ100": "NASDAQ100"}


def event_markets(job):
    s, d = job
    tag = d.strftime("%y") + d.strftime("%b").upper() + d.strftime("%d")
    cands = [f"{s}-{tag}17"] if s in CRYPTO else [f"{s}-{tag}H1600"] + ([f"{LEGACY[s]}-{tag}"] if s in LEGACY else [])
    cache = D / "mkts"
    cache.mkdir(exist_ok=True)
    f = cache / f"{s}_{tag}.json"
    if f.exists():
        return s, json.loads(f.read_text())
    keep = ("ticker", "event_ticker", "strike_type", "floor_strike", "cap_strike", "open_time", "close_time",
            "result", "status")
    for ev in cands:
        try:
            ms = (kget("/historical/markets", {"event_ticker": ev, "limit": 1000}) or {}).get("markets", [])
        except Exception as e:  # noqa: BLE001 -- a 5xx after retries: record nothing for this name, do not cache
            print("ERR", ev, e, flush=True)
            return s, []
        if ms:
            ms = [{k: m.get(k) for k in keep} for m in ms]
            f.write_text(json.dumps(ms))
            return s, ms
    f.write_text("[]")
    return s, []


def cmd_markets():
    rows = []
    jobs = [(s, d) for s in CRYPTO for d in pd.date_range("2024-10-24", DEV_END)]
    jobs += [(s, d) for s in INDEX for d in pd.bdate_range(IDX_START, DEV_END)]
    print(len(jobs), "event lookups", flush=True)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(event_markets, jobs))
    for s, ms in res:
        n0 = len(rows)
        for m in ms:
            ct = pd.Timestamp(m["close_time"]).tz_convert(ET)
            ev = m["event_ticker"]
            if s in CRYPTO and not (ct.hour == 17 and ct.minute == 0 and ev.endswith("17")):
                continue
            if s in INDEX and not (ct.hour == 16 and ct.minute == 0):
                continue
            dte = pd.Timestamp(ct.date())
            if dte > DEV_END or (s in INDEX and dte < IDX_START):
                continue
            rows.append({"series": s, "ticker": m["ticker"], "event_ticker": ev, "date": dte,
                         "strike_type": m.get("strike_type"), "floor": _num(m.get("floor_strike")),
                         "cap": _num(m.get("cap_strike")), "open_ts": int(pd.Timestamp(m["open_time"]).timestamp()),
                         "close_ts": int(pd.Timestamp(m["close_time"]).timestamp()),
                         "result": m.get("result"), "status": m.get("status")})
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df.to_parquet(D / "markets.parquet")
    print(df.groupby("series").agg(n=("ticker", "size"), ev=("event_ticker", "nunique"), d0=("date", "min"),
                                    d1=("date", "max")))


def decision_ts(date: pd.Timestamp) -> int:
    return int(pd.Timestamp(date.date()).tz_localize(ET).replace(hour=11).timestamp())


# ------------------------------------------------------------------ Deribit
def deribit_window(cur: str, t0_ms: int, t1_ms: int) -> list[dict]:
    out, start = [], t0_ms
    for _ in range(200):
        for i in range(6):
            try:
                r = _S.get(DERIBIT, params={"currency": cur, "kind": "option", "start_timestamp": start,
                                            "end_timestamp": t1_ms - 1, "count": 1000, "include_old": "true",
                                            "sorting": "asc"}, timeout=60)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 * (i + 1))
        res = r.json()["result"]
        tr = res["trades"]
        out += tr
        if not res.get("has_more") or not tr:
            break
        start = tr[-1]["timestamp"]  # may repeat the boundary ms; dedupe by trade_id below
    seen, ded = set(), []
    for x in out:
        if x["trade_id"] not in seen:
            seen.add(x["trade_id"])
            ded.append(x)
    return ded


def cmd_deribit():
    days = list(pd.date_range("2024-10-24", DEV_END))
    cache = D / "deribit"
    cache.mkdir(exist_ok=True)
    jobs = []
    for dd in days:
        T = decision_ts(pd.Timestamp(dd))
        for cur in ("BTC", "ETH"):
            f = cache / f"{cur}_{pd.Timestamp(dd).date()}.json"
            if not f.exists():
                jobs.append((cur, T, f))
    print(len(jobs), "Deribit windows to fetch", flush=True)

    def go(j):
        cur, T, f = j
        tr = deribit_window(cur, (T - 3600) * 1000, T * 1000)
        f.write_text(json.dumps([{k: x.get(k) for k in ("timestamp", "instrument_name", "price", "mark_price", "iv",
                                                         "index_price", "amount")} for x in tr]))
        return len(tr)

    with ThreadPoolExecutor(3) as ex:
        for i, n in enumerate(ex.map(go, jobs)):
            if i % 100 == 0:
                print(i, n, flush=True)


# ------------------------------------------------------------------ implied probabilities
def b76_call(F, K, sig, tau):
    sig = np.maximum(sig, 1e-6)
    sd = sig * np.sqrt(tau)
    d1 = (np.log(F / K) + 0.5 * sd * sd) / sd
    return F * norm.cdf(d1) - K * norm.cdf(d1 - sd)


def b76_put(F, K, sig, tau):
    return b76_call(F, K, sig, tau) - (F - K)


def implied_vol(price, F, K, tau, call):
    f = b76_call if call else b76_put
    lo, hi = 1e-3, 10.0
    if not (f(F, K, lo, tau) <= price <= f(F, K, hi, tau)):
        return np.nan
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(F, K, mid, tau) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}


def _expiry(inst: str) -> pd.Timestamp:
    s = inst.split("-")[1]
    day, mon, yy = int(s[:-5]), MON[s[-5:-2]], 2000 + int(s[-2:])
    return pd.Timestamp(year=yy, month=mon, day=day, hour=8, tz="UTC")


def smile(trades: list[dict], close_ts: int, T: int):
    """-> (F_T, tau_D, xs, sigs) or None. Pre-registered construction (notes, 'Options-implied P(YES)')."""
    if not trades:
        return None
    tr = sorted([x for x in trades if x["timestamp"] < T * 1000], key=lambda x: x["timestamp"])
    if not tr:
        return None
    FT = float(tr[-1]["index_price"])
    exps = sorted({_expiry(x["instrument_name"]) for x in tr})
    tgt = [e for e in exps if e.timestamp() >= close_ts]
    if not tgt:
        return None
    e = tgt[0]
    tauD = (e.timestamp() - T) / (365 * 86400)
    last = {}
    for x in tr:
        if _expiry(x["instrument_name"]) == e:
            last[x["instrument_name"]] = x
    pts = {}
    for inst, x in last.items():
        _, _, k, cp = inst.split("-")
        K, F = float(k), float(x["index_price"])
        call = cp == "C"
        if (call and K < F) or ((not call) and K >= F):
            continue
        tau_i = (e.timestamp() * 1000 - x["timestamp"]) / (365 * 86400 * 1000)
        iv = implied_vol(float(x["mark_price"]) * F, F, K, tau_i, call)
        if np.isfinite(iv):
            pts.setdefault(K, []).append(iv)
    if not pts:
        return None
    Ks = np.array(sorted(pts))
    sig = np.array([np.mean(pts[k]) for k in Ks])
    xs = np.log(Ks / FT)
    if len(xs) < 5 or not ((xs < 0).any() and (xs > 0).any()):
        return None
    return FT, tauD, xs, sig


def p_above_crypto(K, sm, tauK):
    FT, tauD, xs, sig = sm
    def sigK(k):
        xd = np.log(k / FT) * math.sqrt(tauD / tauK)
        if xd < xs[0] or xd > xs[-1]:
            return np.nan
        return float(np.interp(xd, xs, sig))
    h = 0.001 * K
    s1, s2 = sigK(K - h), sigK(K + h)
    if not (np.isfinite(s1) and np.isfinite(s2)):
        return np.nan
    return float(np.clip(-(b76_call(FT, K + h, s2, tauK) - b76_call(FT, K - h, s1, tauK)) / (2 * h), 0, 1))


T4_SCALE = math.sqrt(2.0)  # sd of Student-t(4) = sqrt(4/2)


def p_above_index(K, S, sd):
    z = np.log(K / S) / sd
    return float(student_t.sf(z * T4_SCALE, 4))


def p_yes(row, pa):
    st = row["strike_type"]
    if st in ("greater", "greater_or_equal"):
        return pa(row["floor"])
    if st == "less":
        return 1 - pa(row["cap"])
    if st == "between":
        return pa(row["floor"]) - pa(row["cap"])
    return np.nan


def yahoo_inputs():
    f = D / "yahoo_hourly.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import yfinance as yf
    out = {}
    for t in ["^VIX1D", "^VIX", "^VXN", "^SPX", "^NDX"]:
        x = yf.download(t, period="730d", interval="1h", progress=False, auto_adjust=False)
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
        x.index = x.index.tz_convert(ET)
        x = x[x.index < pd.Timestamp("2026-04-11", tz=ET)]
        out[t] = x["Close"]
    y = pd.DataFrame(out)
    y.to_parquet(f)
    return y


def cmd_implied(sig_scale: float = 1.0, out_name: str = "implied.parquet"):
    m = pd.read_parquet(D / "markets.parquet")
    rows = []
    # crypto
    for (s, dd), g in m[m.series.isin(CRYPTO)].groupby(["series", "date"]):
        cur = "BTC" if s == "KXBTCD" else "ETH"
        f = D / "deribit" / f"{cur}_{pd.Timestamp(dd).date()}.json"
        T = decision_ts(pd.Timestamp(dd))
        close = int(g["close_ts"].iloc[0])
        sm = smile(json.loads(f.read_text()), close, T) if f.exists() else None
        if sm is not None and sig_scale != 1.0:
            sm = (sm[0], sm[1], sm[2], sm[3] * sig_scale)
        tauK = (close - T) / (365 * 86400)
        for _, r in g.iterrows():
            p = p_yes(r, lambda k: p_above_crypto(k, sm, tauK)) if sm is not None else np.nan
            rows.append({"ticker": r["ticker"], "T": T, "p_yes": p, "spot": sm[0] if sm else np.nan,
                         "inputs_as_of": T - 1})
    # index
    y = yahoo_inputs()
    for (s, dd), g in m[m.series.isin(INDEX)].groupby(["series", "date"]):
        T = decision_ts(pd.Timestamp(dd))
        bar = pd.Timestamp(pd.Timestamp(dd).date()).tz_localize(ET).replace(hour=9, minute=30)
        sd, S = np.nan, np.nan
        vbar = bar.replace(minute=0)  # Yahoo ^VIX hourly bars start on the hour (CT feed); 09:00 ET bar ends 10:00
        if bar in y.index:
            v = y.loc[bar]
            sig1 = v["^VIX1D"] / 100 / math.sqrt(252)
            nd = s in ("KXNASDAQ100", "KXNASDAQ100U")
            if nd:
                vix = y.loc[vbar, "^VIX"] if vbar in y.index else np.nan
                sig1 *= v["^VXN"] / vix
            S = v["^NDX"] if nd else v["^SPX"]
            sd = sig1 * math.sqrt(5 / 6.5) * sig_scale
        for _, r in g.iterrows():
            ok = bar in y.index and np.isfinite(sd) and np.isfinite(S)
            pp = p_yes(r, lambda k: p_above_index(k, S, sd)) if ok else np.nan
            rows.append({"ticker": r["ticker"], "T": T, "p_yes": pp, "spot": S if ok else np.nan,
                         "inputs_as_of": int(bar.timestamp()) + 3600})
    out = pd.DataFrame(rows)
    out.to_parquet(D / out_name)
    print(out.p_yes.notna().sum(), "of", len(out), "markets with an implied P(YES)")


def tail_candidates(imp: pd.DataFrame) -> pd.DataFrame:
    """AMENDMENT 2: tail candidates (p_T <= .10), only the 3 closest to the money per (event, side of spot)."""
    p = imp["p_yes"]
    c = imp[(p <= 0.10) | (p >= 0.90)].copy()
    c["pT"] = np.where(c.p_yes <= 0.10, c.p_yes, 1 - c.p_yes)
    ctr = np.where(c.strike_type.isin(["greater", "greater_or_equal"]), c["floor"],
                   np.where(c.strike_type == "less", c["cap"], (c["floor"] + c["cap"]) / 2))
    c["up"] = ctr > c["spot"]
    c = c.sort_values("pT", ascending=False, kind="stable")
    c = c.groupby(["event_ticker", "up"], sort=False).head(3)
    return c.drop(columns=["pT", "up"]).sort_index()


# ------------------------------------------------------------------ Kalshi candles
def cmd_candles():
    m = pd.read_parquet(D / "markets.parquet").merge(pd.read_parquet(D / "implied.parquet"), on="ticker")
    c = tail_candidates(m)
    cache = D / "candles"
    cache.mkdir(exist_ok=True)
    todo = [r for r in c.itertuples() if not (cache / f"{r.ticker}.json").exists()]
    print(len(c), "tail-candidate markets;", len(todo), "to fetch", flush=True)

    def go(r):
        d = kget(f"/historical/markets/{r.ticker}/candlesticks",
                 {"start_ts": int(r.open_ts), "end_ts": int(r.close_ts), "period_interval": 60}) or {}
        (cache / f"{r.ticker}.json").write_text(json.dumps(d.get("candlesticks", [])))

    with ThreadPoolExecutor(4) as ex:
        for i, _ in enumerate(ex.map(go, todo)):
            if i % 500 == 0:
                print(i, flush=True)


def _px(c, side, key):
    v = (c.get(side) or {}).get(key)
    if v is None:
        v = (c.get(side) or {}).get(key + "_dollars")
        return _num(v)
    return _num(v) / 100 if isinstance(v, (int, float)) and v > 1.0 else _num(v)


def cmd_frame():
    m = pd.read_parquet(D / "markets.parquet").merge(pd.read_parquet(D / "implied.parquet"), on="ticker")
    c = tail_candidates(m).copy()
    rows, tape = [], []
    for r in c.itertuples():
        f = D / "candles" / f"{r.ticker}.json"
        cs = json.loads(f.read_text()) if f.exists() else []
        pre = [x for x in cs if x["end_period_ts"] <= r.T]
        mk = [x for x in cs if x["end_period_ts"] <= r.close_ts - 3600]
        yb = _px(pre[-1], "yes_bid", "close") if pre else np.nan
        ya = _px(pre[-1], "yes_ask", "close") if pre else np.nan
        qa = pre[-1]["end_period_ts"] if pre else -1
        mb = _px(mk[-1], "yes_bid", "close") if mk else np.nan
        ma = _px(mk[-1], "yes_ask", "close") if mk else np.nan
        rows.append({"yes_bid": yb, "yes_ask": ya, "quote_ts": qa, "mark_yes_bid": mb, "mark_yes_ask": ma})
        for x in cs:
            if r.T < x["end_period_ts"] <= r.close_ts:
                vol = _num(x.get("volume", x.get("volume_fp")))
                tape.append({"ticker": r.ticker, "ts": x["end_period_ts"], "yes_low": _px(x, "price", "low"),
                             "yes_high": _px(x, "price", "high"), "volume": vol})
    q = pd.DataFrame(rows, index=c.index)
    fr = pd.concat([c, q], axis=1)
    fr["asset_class"] = np.where(fr.series.isin(CRYPTO), "crypto", "index")
    fr["date"] = pd.to_datetime(fr["date"])
    fr.to_parquet(D / "frame.parquet")
    pd.DataFrame(tape).to_parquet(D / "tape.parquet")
    print(len(fr), "frame rows;", fr.yes_bid.notna().sum(), "with a quote;", len(tape), "tape candles")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "implied" and len(sys.argv) > 2:
        cmd_implied(float(sys.argv[2]), f"implied_s{sys.argv[2]}.parquet")
    else:
        {"markets": cmd_markets, "deribit": cmd_deribit, "implied": cmd_implied, "candles": cmd_candles,
         "frame": cmd_frame}[cmd]()
