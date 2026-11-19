"""E4 probabilistic rule C / C-agree forward maker paper logger (READ-ONLY: public GETs; no keys, no orders).

Pre-registration + forward judge: sports/notes/ops_e4p_forward.md (lead notes/e4_probabilistic.md, critic critic_e4p).
edgelab trials: e4p_c_maker_fwd, e4p_cagree_maker_fwd (family kalshi_weather_temp), adapter edgelab/adapters/e4p_fwd.py.

    python scripts/e4p_paper.py tick                 # decide every city whose local clock is in [22:00, 22:50) on D-1
    python scripts/e4p_paper.py settle               # station outcome + trade-through fill + result once settled
    python scripts/e4p_paper.py status               # counts, fills, loss tests, SPRT, P(hit) reliability
    python scripts/e4p_paper.py dry [--day D] [--out F]   # decide NOW for event day D ignoring the window -> scratch file
    python scripts/e4p_paper.py fitcheck             # refit dev months with this code and compare to the dev fits.json

Output (append-only): ~/.claude/quant-sports/data/kx_forward/e4p.jsonl (override E4P_LOG)
  kind city_day  one per (series, D): forecasts, NBS run, n brackets, status decided | missed | no_event | few_models
  kind bracket   one per tradeable bracket decided: phit, e4_own, e4_live_at_T, sel_c, sel_cagree, book, q, status, taker24
  kind settle    one per C-selected bracket once settled: result, tt_fill, fill_ts, n10, pnl_unit, e4_live_final
  kind station   one per (series, D) once settled: expiration_value (forward training row for later monthly fits)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from scipy import optimize, stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kx_throttle import kget  # noqa: E402
import om_fetch  # noqa: E402

QS = Path.home() / ".claude" / "quant-sports" / "data" / "kx_forward"
LOG = Path(os.environ.get("E4P_LOG", str(QS / "e4p.jsonl")))
FITS = Path(os.environ.get("E4P_FITS", str(QS / "e4p_fits.json")))
TRAIN = Path(os.environ.get("E4P_TRAIN", str(QS / "e4p_train_dev.parquet")))
KD = Path(os.environ.get("E4P_KD", str(Path.home() / ".claude" / "quant-kalshi" / "data")))
# series -> (station, tz, lat, lon); copied from kalshi_paper.PAPER_CITIES + api.weather.gov coords (2026-11-16)
CITIES = {
    "KXHIGHNY": ("KNYC", "America/New_York", 40.78333, -73.96667),
    "KXHIGHCHI": ("KMDW", "America/Chicago", 41.78417, -87.75528),
    "KXHIGHMIA": ("KMIA", "America/New_York", 25.79056, -80.31639),
    "KXHIGHAUS": ("KAUS", "America/Chicago", 30.18304, -97.67987),
    "KXHIGHDEN": ("KDEN", "America/Denver", 39.84658, -104.65622),
    "KXHIGHLAX": ("KLAX", "America/Los_Angeles", 33.93806, -118.38889),
    "KXHIGHPHIL": ("KPHL", "America/New_York", 39.87327, -75.22678),
    "KXHIGHTATL": ("KATL", "America/New_York", 33.64028, -84.42694),
    "KXHIGHTBOS": ("KBOS", "America/New_York", 42.36056, -71.01056),
    "KXHIGHTDAL": ("KDFW", "America/Chicago", 32.89743, -97.02196),
    "KXHIGHTDC": ("KDCA", "America/New_York", 38.84833, -77.03417),
    "KXHIGHTEWR": ("KEWR", "America/New_York", 40.6825, -74.16944),
    "KXHIGHTHOU": ("KHOU", "America/Chicago", 29.6375, -95.2825),
    "KXHIGHTLV": ("KLAS", "America/Los_Angeles", 36.07188, -115.1634),
    "KXHIGHTMIN": ("KMSP", "America/Chicago", 44.88306, -93.22889),
    "KXHIGHTNOLA": ("KMSY", "America/Chicago", 29.99278, -90.25083),
    "KXHIGHTOKC": ("KOKC", "America/Chicago", 35.38861, -97.60028),
    "KXHIGHTPHX": ("KPHX", "America/Phoenix", 33.427799, -112.003465),
    "KXHIGHTSAN": ("KSAN", "America/Los_Angeles", 32.73361, -117.18306),
    "KXHIGHTSATX": ("KSAT", "America/Chicago", 29.53278, -98.46361),
    "KXHIGHTSDF": ("KSDF", "America/Kentucky/Louisville", 38.17406, -85.7365),
    "KXHIGHTSEA": ("KSEA", "America/Los_Angeles", 47.44472, -122.31361),
    "KXHIGHTSFO": ("KSFO", "America/Los_Angeles", 37.61961, -122.36558),
    "KXHIGHTTTN": ("KTTN", "America/New_York", 40.27639, -74.81639),
}
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
MARGIN = 3.0
T_HOUR, WIN_MIN, NBS_WAIT_MIN = 22, 50, 40
PSTAR, AGREE_YES_BID = 0.02, 0.05
MAX_IMPROVE = 0.05
TAKER_CT = 24
FEE = 0.07
FORWARD_FROM = "2026-11-17"     # first event day D (decision 22:00 local on 2026-11-16)
CLOSE_H = 10
IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
OM = "https://api.open-meteo.com/v1/forecast"
H = {"User-Agent": "quant-sports-readonly-paper"}


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def pts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def jl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def append(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_js) + "\n")


def _js(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def cent_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def cent_ceil(x: float) -> float:
    return math.ceil(round(x * 100, 6)) / 100


def retry(fn, tries=4, wait=5.0):
    for i in range(tries):
        try:
            return fn()
        except (requests.RequestException, ValueError, KeyError):
            if i == tries - 1:
                raise
            time.sleep(wait * (i + 1))


# ------------------------------------------------------------------ EMOS (verbatim port of scripts/e4_probabilistic.py)
def _unpack(th, nbs):
    if nbs:
        b0, b1, b2, c0, c1, c2, d = th
    else:
        b0, b1, c0, c2, d = th
        b2, c1 = 0.0, -50.0
    return b0, b1, b2, c0, c1, c2, d


def _musig(th, X, nbs):
    b0, b1, b2, c0, c1, c2, d = _unpack(th, nbs)
    mu = b0 + b1 * X["m"] + b2 * (X["txn"].fillna(0) if nbs else 0)
    sig = np.sqrt(np.exp(c0) + np.exp(c1) * (X["xnd"].fillna(0) ** 2 if nbs else 0) + np.exp(c2) * X["s2"])
    return np.asarray(mu, float), np.asarray(sig, float), 2 + np.exp(d)


def fit(sd: pd.DataFrame, nbs: bool):
    X = sd[sd.txn.notna() & sd.xnd.notna()] if nbs else sd
    y = X.expiration_value.to_numpy(float)

    def nll(th):
        mu, sig, df = _musig(th, X, nbs)
        return -sst.t.logpdf(y, df, mu, sig).sum()

    th0 = [0, 1, 0, np.log(4), np.log(1), np.log(1), np.log(5)] if nbs else [0, 1, np.log(4), np.log(1), np.log(5)]
    th0 = np.array(th0, float)
    if nbs:
        th0[1], th0[2] = .5, .5
    return optimize.minimize(nll, th0, method="L-BFGS-B").x


def p_hit(th, X, nbs):
    st, f, c = X.strike_type.to_numpy(), X.floor_strike.to_numpy(float), X.cap_strike.to_numpy(float)
    mu, sig, df = _musig(th, X, nbs)
    F = lambda v: sst.t.cdf((v - mu) / sig, df)  # noqa: E731
    return np.where(st == "between", F(c + .5) - F(f - .5), np.where(st == "greater", 1 - F(f + .5), F(c - .5)))


def train_frame(before: str) -> pd.DataFrame:
    """Dev station-days + this logger's settled forward station-days dated before `before` (YYYY-MM-01)."""
    d = pd.read_parquet(TRAIN)[["station", "date", "m", "s2", "txn", "xnd", "expiration_value"]]
    d["src"] = "dev"
    rs = jl(LOG)
    cd = {(r["series"], r["event_day"]): r for r in rs if r.get("kind") == "city_day" and r.get("status") == "decided"}
    fw = []
    for r in rs:
        if r.get("kind") != "station" or r.get("expiration_value") is None or r["event_day"] >= before:
            continue
        c = cd.get((r["series"], r["event_day"]))
        if not c or c.get("n_mod", 0) < 3:
            continue
        fw.append({"station": c["station"], "date": pd.Timestamp(r["event_day"]), "m": c["m"], "s2": c["s2"],
                   "txn": c.get("txn"), "xnd": c.get("xnd"), "expiration_value": float(r["expiration_value"]), "src": "fwd"})
    if fw:
        d = pd.concat([d, pd.DataFrame(fw)], ignore_index=True)
    d["txn"], d["xnd"] = pd.to_numeric(d.txn, errors="coerce"), pd.to_numeric(d.xnd, errors="coerce")
    return d


def month_fit(month: str) -> dict:
    """Frozen per-month fit: computed once (first need) on data before the month, then never rewritten."""
    fits = json.loads(FITS.read_text()) if FITS.exists() else {}
    if month in fits:
        return fits[month]
    tr = train_frame(f"{month}-01")
    fits[month] = {"fitted_at": iso(now_utc()), "n_train": int(len(tr)), "n_fwd": int((tr.src == "fwd").sum()),
                   "n_train_nbs": int((tr.txn.notna() & tr.xnd.notna()).sum()),
                   "nbs": [float(x) for x in fit(tr, True)], "nonbs": [float(x) for x in fit(tr, False)]}
    FITS.parent.mkdir(parents=True, exist_ok=True)
    tmp = FITS.with_suffix(".tmp")
    tmp.write_text(json.dumps(fits, indent=1))
    tmp.replace(FITS)
    return fits[month]


# ------------------------------------------------------------------ inputs
def live_forecasts(lat, lon, tz, day, since=None, src=None) -> dict:
    """= kalshi_paper.live_forecasts: max over hours <= 18 local on `day`, >= 15 hourly values.
    Same endpoint + params via om_fetch (ops_robustness_1118): 30 s TLS-handshake timeout, and on failure the
    same-params response fetched at/after `since` (this city's 22:00 window start; E4 fetches it at :03) is used."""
    r, s = om_fetch.fetch_hourly(lat, lon, tz, MODELS, headers=H, since=since)
    if src is not None:
        src.append(s)
    times = [datetime.fromisoformat(t) for t in r["time"]]
    keep = [i for i, t in enumerate(times) if t.date() == day and t.hour <= 18]
    out = {}
    for m in MODELS:
        v = [r.get(f"temperature_2m_{m}", [None] * len(times))[i] for i in keep]
        v = [x for x in v if x is not None]
        out[m] = max(v) if len(v) >= 15 else None
    return out


def nbs(station: str, day) -> dict | None:
    """NBS run on D-1 (19Z if present, else 18Z): txn/xnd at ftime (D+1) 00Z. None if not in the IEM archive yet."""
    d1 = day - timedelta(days=1)
    r = requests.get(IEM, timeout=60, params={"station": station, "model": "NBS", "sts": f"{d1:%Y-%m-%d}T18:00Z",
                                              "ets": f"{d1:%Y-%m-%d}T19:00Z", "format": "csv"})
    r.raise_for_status()
    ft = f"{day + timedelta(days=1):%Y-%m-%d} 00:00:00"
    best = None
    for x in csv.DictReader(io.StringIO(r.text)):
        if x.get("ftime") != ft or not x.get("txn") or not x.get("xnd"):
            continue
        hr = x["runtime"][11:13]
        if hr in ("18", "19") and (best is None or hr > best["run_hour"]):
            best = {"nbs_run": x["runtime"], "run_hour": hr, "txn": float(x["txn"]), "xnd": float(x["xnd"])}
    return best


def e4_live_tickers() -> set[str]:
    return {r["ticker"] for r in jl(KD / "kalshi_paper.jsonl")}


def e4_rejected(st, fl, ca, fcs) -> bool:
    """= kalshi_paper.rejected (margin 3F, all models)."""
    hi, lo = max(fcs), min(fcs)
    if st == "greater":
        return hi + MARGIN < fl + 1
    if st == "less":
        return lo - MARGIN > ca - 1
    return hi + MARGIN < fl or lo - MARGIN > ca


def book(ticker: str) -> dict:
    d = kget(f"/markets/{ticker}/orderbook") or {}
    ob = d.get("orderbook_fp") or d.get("orderbook") or {}
    lvl = lambda rows: sorted(((float(p) if float(p) <= 1 else float(p) / 100, float(q)) for p, q in (rows or [])),  # noqa: E731
                              reverse=True)
    ys, ns = lvl(ob.get("yes_dollars") or ob.get("yes")), lvl(ob.get("no_dollars") or ob.get("no"))
    out = {"yes_bids": ys[:8], "no_bids": ns[:5], "yes_depth_total": sum(q for _, q in ys)}
    if ys:
        out["no_ask"] = round(1 - ys[0][0], 4)
    if ns:
        out["no_bid"] = round(ns[0][0], 4)
    return out


def taker24(yes_bids: list) -> dict:
    """Descriptive: buy 24 NO by hitting YES bids (best first); per-ORDER fee rounded up to the cent."""
    need, n, notional, fq = float(TAKER_CT), 0.0, 0.0, 0.0
    for p, q in yes_bids:
        take = min(need - n, q)
        if take <= 0:
            break
        P = round(1 - p, 4)
        n += take
        notional += take * P
        fq += take * P * (1 - P)
    if n <= 0:
        return {"taker_ct": 0}
    fee = cent_ceil(FEE * fq)
    return {"taker_ct": n, "taker_avg_no": round(notional / n, 5), "taker_fee_order": fee,
            "taker_cost_per_ct": round((notional + fee) / n, 5), "taker_worst_no": round(1 - [p for p, _ in yes_bids][0], 4)
            if yes_bids else None}


_FEE: dict = {}


def fee_type(series: str):
    if series not in _FEE:
        _FEE[series] = ((kget(f"/series/{series}") or {}).get("series") or {}).get("fee_type")
    return _FEE[series]


def tape(ticker: str, t0: int, t1: int) -> list[tuple[int, float, float, str]]:
    out = []
    for p in ("/markets/trades", "/historical/trades"):
        cur = None
        while True:
            prm = {"ticker": ticker, "min_ts": int(t0), "max_ts": int(t1), "limit": 1000}
            if cur:
                prm["cursor"] = cur
            d = kget(p, prm)
            if not d:
                break
            out += d.get("trades", [])
            cur = d.get("cursor")
            if not cur or not d.get("trades"):
                break
        if out:
            break
    rows = []
    for r in out:
        npx = r.get("no_price_dollars")
        npx = float(npx) if npx is not None else float(r.get("no_price")) / 100
        cnt = r.get("count_fp")
        cnt = float(cnt) if cnt is not None else float(r.get("count", 0))
        rows.append((int(pts(r["created_time"]).timestamp()), npx, cnt, r.get("taker_side")))
    return sorted(set(rows))


# ------------------------------------------------------------------ decision
def decide_city(series: str, day, t: datetime, allow_no_nbs: bool, force: bool = False) -> list[dict] | None:
    """Rows for one (series, D). None = wait (NBS not yet published and still before the no-NBS cutoff)."""
    station, tz, lat, lon = CITIES[series]
    ev = f"{series}-{day.strftime('%y%b%d').upper()}"
    base = {"kind": "city_day", "series": series, "station": station, "event": ev, "event_day": day.isoformat(),
            "decided_at": iso(t), "local": t.astimezone(ZoneInfo(tz)).isoformat(timespec="seconds")}
    nb = retry(lambda: nbs(station, day))
    if nb is None and not allow_no_nbs:
        return None
    ms = retry(lambda: (kget("/markets", {"event_ticker": ev, "limit": 100}) or {}).get("markets", []))
    if not ms:
        return [{**base, "status": "no_event"}]
    since = None if force else t.astimezone(ZoneInfo(tz)).replace(hour=T_HOUR, minute=0, second=0, microsecond=0)
    src: list[str] = []
    fc = retry(lambda: live_forecasts(lat, lon, tz, day, since=since, src=src))
    vals = [v for v in fc.values() if v is not None]
    cd = {**base, "forecasts": fc, "fc_source": src[-1] if src else None, "n_mod": len(vals),
          **(nb or {"nbs_run": None, "txn": None, "xnd": None})}
    if len(vals) < 3:
        return [{**cd, "status": "few_models"}]
    m, s2 = float(np.mean(vals)), float(np.var(vals))
    month = (day - timedelta(days=1)).strftime("%Y-%m")   # month of the decision (D-1 local)
    ft = month_fit(month)
    has = nb is not None
    th = np.array(ft["nbs"] if has else ft["nonbs"], float)
    live = e4_live_tickers()
    rows = []
    for mk in ms:
        yb, ya = float(mk.get("yes_bid_dollars") or 0), float(mk.get("yes_ask_dollars") or 0)
        if not (yb > 0 and ya < 1 and ya - yb <= 0.10 and ya > 0.01):
            continue
        fl = float(mk["floor_strike"]) if mk.get("floor_strike") is not None else np.nan
        ca = float(mk["cap_strike"]) if mk.get("cap_strike") is not None else np.nan
        X = pd.DataFrame({"m": [m], "s2": [s2], "txn": [nb["txn"] if has else np.nan], "xnd": [nb["xnd"] if has else np.nan],
                          "strike_type": [mk.get("strike_type")], "floor_strike": [fl], "cap_strike": [ca]})
        ph = float(p_hit(th, X, has)[0])
        e4o = bool(e4_rejected(mk.get("strike_type"), fl, ca, vals))
        e4l = mk["ticker"] in live
        sel_c = (ph <= PSTAR) and not e4o and not e4l
        r = {"kind": "bracket", "ticker": mk["ticker"], "series": series, "event_day": day.isoformat(), "decided_at": iso(t),
             "strike_type": mk.get("strike_type"), "floor": fl, "cap": ca, "yes_bid": yb, "yes_ask": ya,
             "close_time": mk.get("close_time"), "market_status": mk.get("status"), "phit": round(ph, 6), "has_nbs": has,
             "fit_month": month, "e4_own": e4o, "e4_live_at_T": e4l, "sel_c": sel_c,
             "sel_cagree": bool(sel_c and yb < AGREE_YES_BID)}
        if sel_c:
            r.update(maker(mk, series))
        rows.append(r)
    return [{**cd, "status": "decided", "m": m, "s2": s2, "fit_month": month, "n_tradeable": len(rows),
             "n_c": sum(r["sel_c"] for r in rows), "n_cagree": sum(r["sel_cagree"] for r in rows)}] + rows


def maker(mk: dict, series: str) -> dict:
    tk = mk["ticker"]
    out = {}
    if mk.get("status") not in ("active", "open"):
        return {"status": "skip_market_not_active"}
    ft = fee_type(series)
    out["fee_type"] = ft
    b = retry(lambda: book(tk))
    out.update({"book": b, "book_ts": iso(now_utc()), **taker24(b.get("yes_bids") or [])})
    if "no_ask" not in b or "no_bid" not in b:
        return {**out, "status": "skip_no_book"}
    ask, bid = b["no_ask"], b["no_bid"]
    q = round(min(cent_floor((ask + bid) / 2), ask - 0.01), 4)
    out.update({"q": q, "no_ask": ask, "no_bid": bid, "spread": round(ask - bid, 4)})
    if ft != "quadratic":
        return {**out, "status": "skip_fee_type"}
    if q <= 0 or ask - q > MAX_IMPROVE + 1e-9:
        return {**out, "status": "skip_maker_spread"}
    return {**out, "status": "bid", "side": "no"}


def cmd_tick(a) -> int:
    t = now_utc()
    have = {(r["series"], r["event_day"]) for r in jl(LOG) if r.get("kind") == "city_day"}
    rows, waits, fails = [], [], []
    for series, (station, tz, _, _) in CITIES.items():
        loc = t.astimezone(ZoneInfo(tz))
        day = (loc + timedelta(days=1)).date()
        if day.isoformat() < FORWARD_FROM or (series, day.isoformat()) in have:
            continue
        mins = (loc.hour - T_HOUR) * 60 + loc.minute
        if mins < 0:
            continue
        if mins >= WIN_MIN:
            rows.append({"kind": "city_day", "series": series, "station": station, "event_day": day.isoformat(),
                         "decided_at": iso(t), "local": loc.isoformat(timespec="seconds"), "status": "missed"})
            continue
        try:
            out = decide_city(series, day, t, allow_no_nbs=mins >= NBS_WAIT_MIN)
        except Exception as ex:  # noqa: BLE001 -- one city's failure must not stop the others; retried next tick
            fails.append(f"{series}: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        if out is None:
            waits.append(series)
            continue
        rows += out
    append(rows, LOG)
    missed = [r for r in rows if r["kind"] == "city_day" and r["status"] == "missed"]
    if missed:  # one row per (series, D) ever, so this alerts once per lost city-day
        om_fetch.notify("e4p MISSED city-day", ", ".join(f"{r['series']} D={r['event_day']}" for r in missed)
                        + ": no decision in the 22:00-22:50 window")
    for r in rows:
        if r["kind"] == "city_day":
            print(f"{r['series']} D={r['event_day']} {r['status']} nbs={r.get('nbs_run')} tradeable={r.get('n_tradeable')} "
                  f"C={r.get('n_c')} Cagree={r.get('n_cagree')}")
        elif r.get("sel_c"):
            print(f"  C {r['ticker']} phit={r['phit']:.4f} yb={r['yes_bid']} status={r.get('status')} q={r.get('q')} "
                  f"taker24={r.get('taker_cost_per_ct')}")
    for w in waits:
        print(f"wait NBS {w}")
    for f in fails:
        print(f"FAIL {f}")
    if not rows and not waits and not fails:
        print("idle: no city in its 22:00 window")
    return 3 if fails else 0


def cmd_dry(a) -> int:
    t = now_utc()
    day = datetime.strptime(a.day, "%Y-%m-%d").date() if a.day else (t + timedelta(days=1)).date()
    rows = []
    for series in (a.series.split(",") if a.series else CITIES):
        try:
            out = decide_city(series, day, t, allow_no_nbs=True) or []
        except Exception as ex:  # noqa: BLE001
            print(f"FAIL {series}: {type(ex).__name__}: {str(ex)[:120]}")
            continue
        rows += [{**r, "dry": True} for r in out]
    append(rows, Path(a.out))
    for r in rows:
        if r["kind"] == "city_day":
            print(f"{r['series']} D={r['event_day']} {r['status']} nbs={r.get('nbs_run')} txn={r.get('txn')} xnd={r.get('xnd')} "
                  f"m={r.get('m')} tradeable={r.get('n_tradeable')} C={r.get('n_c')} Cagree={r.get('n_cagree')}")
        elif r.get("sel_c") or r.get("e4_own") or r.get("e4_live_at_T"):
            print(f"  {r['ticker']} phit={r['phit']:.4f} yb={r['yes_bid']} e4own={r['e4_own']} e4live={r['e4_live_at_T']} "
                  f"C={r['sel_c']} CA={r['sel_cagree']} status={r.get('status')} q={r.get('q')} taker24={r.get('taker_cost_per_ct')}")
    print(f"{len(rows)} dry rows -> {a.out}")
    return 0


# ------------------------------------------------------------------ settle
def cmd_settle(a) -> int:
    rs = jl(LOG)
    done_tk = {r["ticker"] for r in rs if r.get("kind") == "settle"}
    done_st = {(r["series"], r["event_day"]) for r in rs if r.get("kind") == "station"}
    live = e4_live_tickers()
    rows = []
    for c in rs:  # station outcomes (forward training rows): one per decided city-day
        if c.get("kind") != "city_day" or c.get("status") != "decided" or (c["series"], c["event_day"]) in done_st:
            continue
        ms = (kget("/markets", {"event_ticker": c["event"], "limit": 100}) or {}).get("markets", [])
        ev = [m.get("expiration_value") for m in ms if m.get("status") in ("finalized", "settled")
              and m.get("expiration_value") not in (None, "")]
        if ms and len(ev) == len(ms):
            vals = {float(v) for v in ev}
            rows.append({"kind": "station", "series": c["series"], "event_day": c["event_day"], "ts": iso(now_utc()),
                         "expiration_value": vals.pop() if len(vals) == 1 else None, "ev_conflict": len(vals) > 0})
    for p in rs:
        if p.get("kind") != "bracket" or not p.get("sel_c") or p["ticker"] in done_tk:
            continue
        tk = p["ticker"]
        m = (kget(f"/markets/{tk}") or {}).get("market") or {}
        res = m.get("result")
        if m.get("status") not in ("finalized", "settled") or res not in ("yes", "no"):
            continue
        row = {"kind": "settle", "ticker": tk, "ts": iso(now_utc()), "result": res, "close_time": m.get("close_time"),
               "e4_live_final": tk in live, "status_at_T": p.get("status")}
        if p.get("status") == "bid":
            t0, t1 = int(pts(p["decided_at"]).timestamp()), int(pts(m.get("close_time") or p["close_time"]).timestamp())
            tp = tape(tk, t0 + 1, t1)
            q = float(p["q"])
            tt = [(ts, px, n) for ts, px, n, _ in tp if px < q - 1e-9]
            cum, n10 = 0.0, None
            for ts, px, n, _ in tp:
                if px <= q + 1e-9:
                    cum += n
                    if cum >= 10 and n10 is None:
                        n10 = ts
            nos = [(ts, px) for ts, px, _, sd in tp if sd == "no" and ts <= t0 + CLOSE_H * 3600]
            win = res == "no"
            row.update({"tt_fill": bool(tt), "fill_ts": iso(datetime.fromtimestamp(tt[0][0], timezone.utc)) if tt else None,
                        "tt_volume": sum(n for _, _, n in tt), "n10_fill": n10 is not None, "prints": len(tp),
                        "close_no": nos[-1][1] if nos else None, "pnl_unit": round((1 - q) / q if win else -1.0, 6)})
            if p.get("taker_cost_per_ct"):
                tc = float(p["taker_cost_per_ct"])
                row["taker_pnl_unit"] = round((1 - tc) / tc if win else -1.0, 6)
        rows.append(row)
    append(rows, LOG)
    print(f"{sum(r['kind'] == 'station' for r in rows)} station outcomes, {sum(r['kind'] == 'settle' for r in rows)} settles")
    return 0


# ------------------------------------------------------------------ status / judge
def poisson_binom_le(p: list[float], k: int) -> float:
    dist = [1.0]
    for pi in p:
        nd = [0.0] * (len(dist) + 1)
        for j, v in enumerate(dist):
            nd[j] += v * (1 - pi)
            nd[j + 1] += v * pi
        dist = nd
    return float(sum(dist[: k + 1]))


def judge(fills: list[tuple[float, bool, str]]) -> dict:
    """fills = (q, lost, event_day). SPRT (bar_calibration_v2) + Poisson-binomial loss count (prereg ops_e4p_forward.md)."""
    A, B = math.log(0.9 / 0.005), math.log(0.1 / 0.995)
    if not fills:
        return {"fills": 0, "A": round(A, 3), "B": round(B, 3), "llr_adj": 0.0, "state": "too-early"}
    llr = sum(math.log(0.7) if lost else math.log((1 - 0.7 * (1 - q)) / q) for q, lost, _ in fills)
    nights = len({d for _, _, d in fills})
    adj = llr / (1 + len(fills) / nights * 0.031)
    L = sum(l for _, l, _ in fills)
    p0 = [1 - q for q, _, _ in fills]
    pnl = [(1 - q) / q if not l else -1.0 for q, l, _ in fills]
    pv = poisson_binom_le(p0, L)
    st = "KILL (SPRT)" if adj <= B else "H1 (ask user; no self-promotion)" if adj >= A else "continue"
    n = len(fills)
    if st == "continue" and n >= 60:
        if L >= sum(p0):
            st = "KILL (review: losses >= break-even expectation)"
        elif pv < 0.025 and sum(pnl) > 0:
            st = "holds (review p < .025)"
        elif n >= 120:
            st = "KILL (not holds at 120)"
    return {"fills": n, "losses": L, "exp_losses_H0": round(sum(p0), 3), "p_le": round(pv, 5), "units": round(sum(pnl), 3),
            "roi": round(sum(pnl) / n, 5), "llr_adj": round(adj, 3), "A": round(A, 3), "B": round(B, 3), "state": st}


def summary(rs: list[dict]) -> dict:
    br = [r for r in rs if r.get("kind") == "bracket"]
    st = {r["ticker"]: r for r in rs if r.get("kind") == "settle"}
    cd = [r for r in rs if r.get("kind") == "city_day"]
    out = {"city_days": len(cd), "status": {}, "brackets": len(br)}
    for r in cd:
        out["status"][r["status"]] = out["status"].get(r["status"], 0) + 1
    for var, key in (("C", "sel_c"), ("C_agree", "sel_cagree")):
        sel = [r for r in br if r.get(key)]
        bids = [r for r in sel if r.get("status") == "bid"]
        s = [(r, st[r["ticker"]]) for r in bids if r["ticker"] in st and not st[r["ticker"]].get("e4_live_final")]
        f = [(float(r["q"]), x["result"] != "no", r["event_day"]) for r, x in s if x.get("tt_fill")]
        uf = [x["result"] != "no" for r, x in s if not x.get("tt_fill")]
        tk = [x.get("taker_pnl_unit") for r, x in s if x.get("taker_pnl_unit") is not None]
        out[var] = {"selected": len(sel), "bids": len(bids), "settled_bids": len(s), "unfilled": len(uf),
                    "unfilled_losses": sum(uf), "taker24_desc": {"n": len(tk), "units": round(sum(tk), 3),
                                                                 "losses": sum(1 for x in tk if x == -1.0)},
                    "judge": judge(f)}
    return out


def cmd_status(a) -> int:
    rs = jl(LOG)
    print(f"{LOG}: {len(rs)} rows")
    print(json.dumps(summary(rs), indent=1))
    # P(hit) reliability on settled brackets with a station outcome (all tradeable decided brackets)
    so = {(r["series"], r["event_day"]): r.get("expiration_value") for r in rs if r.get("kind") == "station"}
    rel = []
    for r in rs:
        if r.get("kind") != "bracket":
            continue
        y = so.get((r["series"], r["event_day"]))
        if y is None:
            continue
        stp, f, c = r["strike_type"], r["floor"], r["cap"]
        hit = (f <= y <= c) if stp == "between" else (y >= f + 1) if stp == "greater" else (y <= c - 1)
        rel.append((r["phit"], int(hit)))
    if rel:
        d = pd.DataFrame(rel, columns=["p", "h"])
        print(d.groupby(pd.cut(d.p, [0, .005, .01, .02, .05, .1, .2, .5, 1.0001], right=False), observed=True)
              .agg(n=("h", "size"), mean_p=("p", "mean"), hit=("h", "mean")))
    return 0


def cmd_fitcheck(a) -> int:
    """Port check: refit the dev months that used the full E1 dev set (2026-01) and compare to dev fits.json."""
    dev = json.loads(Path(a.fits).read_text())
    d = pd.read_parquet(TRAIN)
    tr = d[d.cohort == "E1"]
    for nb, key in ((True, "nbs"), (False, "nonbs")):
        th = fit(tr, nb)
        print(key, "port", np.round(th, 4).tolist(), "dev", dev["2026-01"][key], "max|diff|",
              float(np.max(np.abs(np.round(th, 4) - np.array(dev["2026-01"][key])))))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tick").set_defaults(fn=cmd_tick)
    sub.add_parser("settle").set_defaults(fn=cmd_settle)
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("dry")
    p.add_argument("--day")
    p.add_argument("--series")
    p.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "results" / "ops_e4p_forward" / "dry.jsonl"))
    p.set_defaults(fn=cmd_dry)
    p = sub.add_parser("fitcheck")
    p.add_argument("--fits", default=str(Path(__file__).resolve().parents[1] / "results" / "e4_probabilistic" / "fits.json"))
    p.set_defaults(fn=cmd_fitcheck)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
