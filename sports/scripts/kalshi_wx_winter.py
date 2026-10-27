"""Kalshi MONTHLY rain / snow totals: observed-to-date + forecast "unreachable / already-locked" rule. DEV only.

    python scripts/kalshi_wx_winter.py build          # settled ladders + ACIS daily obs + forecasts + hourly quotes (cached)
    python scripts/kalshi_wx_winter.py audit          # ACIS month total vs Kalshi settlement (data check, no P&L)
    python scripts/kalshi_wx_winter.py dev [--ledger] # score the pre-registered trials on DEV rows only

Why (2026-10-27): the only proven edge is E1/E2/E4 daily-high consensus-reject (buy NO on brackets all models miss
by > 3F, decided 22:00 local the night before). Daily lows (X2) and daily rain NO (R1) are DEAD. This is a genuinely
different mechanism: MONTHLY accumulation ladders ("total precip/snow in <city> in <month> strictly greater than X
in"), where the observed month-to-date total is a hard state variable and only the last few days are uncertain.

Series (Kalshi public API, category "Climate and Weather", census 2026-10-27):
  RAIN  : every ^KXRAIN[A-Z]+M$ series with settled events (NYC since 2024-03; CHI since 2025-07; LAX HOU SEA SFO
          MIA DAL DEN AUS since 2025-11; a few one-month newcomers). Ladder of `greater` strikes, 1..10 in.
          Station = the CLIxxx in rules_primary ("Central Park" -> NYC). Settlement NWS CLI / The Weather Company.
  SNOW  : the *SNOWM* monthly ladders (NYC DC PHL CHI BOS DET SEA DEN SLC LAX DAL HOU SFO AUS MIA, 2025-12..2026-03)
          + legacy KXSNOWNYM (NYC, 2021-12..2025-02). Ladder of `greater` strikes (0.1 .. 35 in).
          Station = fixed map below (city's NWS climate station); ACIS id. KXJACWSNOWM dropped (ACIS has no snow at JAC).

PRE-REGISTERED 2026-10-27, written before any price, forecast or observation for these series was pulled
(only the series list, rules text and strike ladders had been looked at):

  Data   : daily observed precip/snowfall from NOAA ACIS StnData (keyless; T = 0, M = missing). A city-month with
           ANY missing ACIS day in the month is dropped (data rule, outcome-blind).
  Window : decision times T_d = 22:00 station-local on each day d with  end - 6 <= d <= end  (end = last day of
           the month). State at T_d:
             OBS_d = observed month total over days 1..d-1 (day d-1's climate report is out by the morning of d)
             F_m,d = model m's forecast total over days d..end, where day d+j uses Open-Meteo previous-runs
                     <var>_previous_day(j+1)  (lead j+1 days: every value was issued >= 24 h before its hour, so before
                     T_d; j+1 <= 7 is why the window is the last 7 days). Models gfs_seamless, ecmwf_ifs025,
                     icon_seamless, gem_seamless, ncep_nbm_conus; a model counts only if every hour of days d..end is
                     present; need >= 3 models. FMAX = max over models.
             RAIN: var = precipitation (inches).  SNOW: var = precipitation x 10 (all liquid as snow at 10:1, an
                     upper-bound proxy; Open-Meteo's snowfall field is missing for ecmwf/icon at long leads).
  Quote  : last hourly candle ending <= T_d; tradeable = yes_bid > 0, yes_ask < 1, spread <= 10c (frozen from E1).
           Taker fee 0.07*p*(1-p) (series fee_type quadratic, no maker fee).
  Trials (strike X, market YES iff month total > X):
    W1 rain_unreachable_NO : X - OBS_d > 2*FMAX + 0.50  -> buy NO at 1 - yes_bid (taker), if yes_ask > 0.01
    W2 rain_locked_YES     : OBS_d > X                  -> buy YES at yes_ask (taker), if yes_ask <= 0.98
    S1 snow_unreachable_NO : X - OBS_d > 2*FMAX + 1.0   -> buy NO at 1 - yes_bid (taker), if yes_ask > 0.01
    S2 snow_locked_YES     : OBS_d > X                  -> buy YES at yes_ask, if yes_ask <= 0.98
    W1m / S1m (maker)      : same picks as W1 / S1, executed as E4 M3: resting NO bid at q = min(floor_cent(NO mid),
                             NO ask - 1c) from T_d to close; filled iff >= 1 contract printed at no_price <= q after
                             T_d (public tape); unfilled = no bet. Maker fee 0.
  One bet per market: the FIRST d in the window at which the rule fires on a tradeable quote; held to settlement.
  Stake 1 unit of cost per bet. Score: units, ROI, loss rate (exact binomial CI), one-sided cluster bootstrap over
  city-months (event_ticker) of total units, p = P(boot units <= 0). Halves = chronological split of DEV events.
  6 trials -> bar p < 0.05/6 = 0.0083 each [amended before the ledger write, after the first dev print: the cluster
  bootstrap gives p = 0 for any loss-free sample, so a trial passes only if BOTH the bootstrap p and the market-null
  p (`null_p`: every bet wins with prob = its all-in cost) are < bar -- a strictly harder test, never easier] (W1m/S1m are judged as their own trials, not as deltas).
  Split  : RAIN DEV = months < 2026-07; SEALED = 2026-07+ (never loaded by `dev`).
           SNOW DEV = months < 2026-03 (incl. legacy 2021-25); SEALED = 2026-03 (+ the coming winter forward).
  Ledger : lab/ledger.jsonl, dataset kalshi_wx_winter.
Caveats written down first: precipitation climate day = local STANDARD time (1 h off in DST; irrelevant to snow
months); TWC vs ACIS values may differ (the `audit` step measures it on settled ladders); the upper bound
"FMAX" misses convective / lake-effect surprises -- the margins are the only protection and the dev result measures
whether they suffice. Capacity: hourly candle volume around picks (no historical depth).
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from sports import wx
from sports.http import fetch, fetch_many

OUT = ROOT / "data" / "kalshi_wx_winter.parquet"
OBS = ROOT / "data" / "kalshi_wx_winter_obs.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_wx_winter"
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
SPLIT = {"rain": pd.Timestamp("2026-07-01"), "snow": pd.Timestamp("2026-03-01")}
FEE = wx.FEE_RATE
TRIALS = ["W1_rain_unreachable_NO", "W2_rain_locked_YES", "S1_snow_unreachable_NO", "S2_snow_locked_YES",
          "W1m_rain_unreachable_NO_maker", "S1m_snow_unreachable_NO_maker"]
SNOW = {  # series -> ACIS station id (NWS climate station of the city)
    "KXNYCSNOWM": "NYC", "KXSNOWNYM": "NYC", "KXDCSNOWM": "DCA", "KXPHILSNOWM": "PHL", "KXCHISNOWM": "ORD",
    "KXBOSSNOWM": "BOS", "KXDETSNOWM": "DTW", "KXSEASNOWM": "SEA", "KXDENSNOWMB": "DEN", "KXDENSNOWM": "DEN",
    "KXSLCSNOWM": "SLC", "KXLAXSNOWM": "LAX", "KXDALSNOWM": "DFW", "KXHOUSNOWM": "IAH", "KXSFOSNOWM": "SFO",
    "KXAUSSNOWM": "AUS", "KXMIASNOWM": "MIA",
}


# ------------------------------------------------------------------------------------------------ markets
def settled(series: str) -> pd.DataFrame:
    rows = []
    for path, extra in ((f"{wx.API}/historical/markets", {}), (f"{wx.API}/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache="historical" in path)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.result.isin(["yes", "no"]) & (df.get("strike_type") == "greater")].copy()
    if df.empty:
        return df
    df["X"] = pd.to_numeric(df.floor_strike, errors="coerce")
    df["month"] = df.event_ticker.str.split("-").str[-1].map(lambda s: pd.to_datetime(s[:5], format="%y%b"))
    df["series"] = series
    return df[["series", "ticker", "event_ticker", "month", "X", "result", "close_time", "rules_primary"]]


def rain_series() -> list[str]:
    d = fetch(f"{wx.API}/series", {"limit": 10000}, cache=False)["series"]
    return sorted(x["ticker"] for x in d if re.match(r"^KXRAIN[A-Z]+M$", x["ticker"])
                  and x.get("category") == "Climate and Weather")


def rain_station(rules: str) -> str | None:
    if "Central Park" in rules:
        return "NYC"
    g = re.search(r"CLI([A-Z]{3})", rules)
    return g.group(1) if g else None


# ------------------------------------------------------------------------------------------------ obs / forecasts
def acis(sid: str, start: str, end: str) -> pd.DataFrame:
    q = {"sid": sid, "sdate": start, "edate": end, "elems": "pcpn,snow", "meta": "name,ll"}
    d = fetch("https://data.rcc-acis.org/StnData", {"params": json.dumps(q)},
              cache=pd.Timestamp(end) < pd.Timestamp.today().normalize() - pd.Timedelta(days=10))
    if "data" not in d:
        return pd.DataFrame()

    def v(x):
        return 0.0 if x == "T" else (np.nan if x in ("M", "S", "") else float(str(x).rstrip("A")))
    df = pd.DataFrame([{"date": pd.Timestamp(r[0]), "pcpn": v(r[1]), "snow": v(r[2])} for r in d["data"]])
    df.attrs["ll"] = d.get("meta", {}).get("ll")
    return df


def station_meta(sid: str) -> tuple[float, float, str]:
    p = fetch(f"https://api.weather.gov/stations/K{sid}")
    lon, lat = p["geometry"]["coordinates"][:2]
    return lat, lon, p["properties"]["timeZone"]


def forecasts(lat, lon, tz, end: pd.Timestamp) -> pd.DataFrame:
    """-> rows (d, model, F) = model total over days d..end with lead (day-d)+1, for d in end-6..end. Inches."""
    start = end - pd.Timedelta(days=6)
    hv = ",".join(f"precipitation_previous_day{k}" for k in range(1, 8))
    d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": ",".join(MODELS), "hourly": hv,
                             "precipitation_unit": "inch", "timezone": tz, "start_date": str(start.date()),
                             "end_date": str(end.date())},
              cache=end < pd.Timestamp.today().normalize() - pd.Timedelta(days=3))
    h = pd.DataFrame(d.get("hourly", {}))
    if h.empty:
        return pd.DataFrame(columns=["d", "model", "F"])
    h["day"] = pd.to_datetime(h["time"]).dt.normalize()
    rows = []
    for m in MODELS:
        for dd in pd.date_range(start, end):
            tot, ok = 0.0, True
            for day in pd.date_range(dd, end):
                k = (day - dd).days + 1
                col = f"precipitation_previous_day{k}_{m}"
                if col not in h:
                    ok = False
                    break
                x = h.loc[h.day == day, col]
                if len(x) < 23 or x.isna().any():
                    ok = False
                    break
                tot += float(x.sum())
            rows.append({"d": dd, "model": m, "F": tot if ok else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ quotes
def decision_ts(day: pd.Timestamp, tz: str) -> int:
    return int(datetime(day.year, day.month, day.day, 22, tzinfo=ZoneInfo(tz)).timestamp())


def candles(series: str, ev: str, tickers: list[str], t0: int, t1: int) -> dict[str, list]:
    d = fetch(f"{wx.API}/series/{series}/events/{ev}/candlesticks",
              {"start_ts": t0, "end_ts": t1, "period_interval": 60})
    out = dict(zip(d.get("market_tickers", []), d.get("market_candlesticks", [])))
    miss = [t for t in tickers if not out.get(t)]
    if miss:
        for t in miss:
            for base in (f"{wx.API}/historical/markets/{t}", f"{wx.API}/series/{series}/markets/{t}"):
                try:
                    out[t] = fetch(f"{base}/candlesticks", {"start_ts": t0, "end_ts": t1, "period_interval": 60}
                                   ).get("candlesticks", [])
                    break
                except Exception:  # noqa: BLE001 -- 404: market settled early / not archived
                    out[t] = []
    return out


def _px(c, side):
    v = c.get(side) or {}
    return pd.to_numeric(v.get("close_dollars", v.get("close")), errors="coerce")


# ------------------------------------------------------------------------------------------------ build
def cmd_build() -> int:
    mk = []
    for s in rain_series():
        df = settled(s)
        if len(df):
            df["kind"] = "rain"
            df["sid"] = df.rules_primary.map(rain_station)
            mk.append(df)
    for s, sid in SNOW.items():
        df = settled(s)
        if len(df):
            df["kind"] = "snow"
            df["sid"] = sid
            mk.append(df)
    mk = pd.concat(mk, ignore_index=True).drop(columns=["rules_primary"])
    mk = mk[mk.sid.notna() & mk.X.notna()]
    print(f"settled ladders: {mk.event_ticker.nunique()} city-months, {len(mk)} markets; "
          f"{mk.groupby('kind').event_ticker.nunique().to_dict()}", flush=True)
    rows, obs_rows = [], []
    meta = {}
    for (series, ev), g in mk.groupby(["series", "event_ticker"]):
        sid, kind, month = g.sid.iloc[0], g.kind.iloc[0], g.month.iloc[0]
        end = month + pd.offsets.MonthEnd(0)
        if sid not in meta:
            try:
                meta[sid] = station_meta(sid)
            except Exception as ex:  # noqa: BLE001
                print(f"  {sid}: no NWS station meta ({ex}); skip", flush=True)
                meta[sid] = None
        if meta[sid] is None:
            continue
        lat, lon, tz = meta[sid]
        ob = acis(sid, str(month.date()), str(end.date()))
        if ob.empty:
            print(f"  {ev}: no ACIS data", flush=True)
            continue
        var = "pcpn" if kind == "rain" else "snow"
        n_missing = int(ob[var].isna().sum())
        obs_rows.append({"series": series, "event_ticker": ev, "sid": sid, "kind": kind, "month": month,
                         "acis_total": float(ob[var].sum()), "acis_missing": n_missing,
                         "acis_pcpn_total": float(ob.pcpn.sum())})
        fc = forecasts(lat, lon, tz, end)
        if kind == "snow":
            fc["F"] = fc["F"] * 10.0
        t_first = decision_ts(end - pd.Timedelta(days=6), tz) - 3 * 3600
        t_last = decision_ts(end, tz)
        cs = candles(series, ev, g.ticker.tolist(), t_first, t_last)
        for dd in pd.date_range(end - pd.Timedelta(days=6), end):
            T = decision_ts(dd, tz)
            obs_d = float(ob.loc[ob.date < dd, var].sum()) if n_missing == 0 else np.nan
            f = fc[fc.d == dd].set_index("model")["F"]
            for _, m in g.iterrows():
                c = [x for x in cs.get(m.ticker, []) if x["end_period_ts"] <= T]
                last = c[-1] if c else None
                rows.append({"series": series, "event_ticker": ev, "ticker": m.ticker, "kind": kind, "sid": sid,
                             "tz": tz, "month": month, "end": end, "d": dd, "days_left": (end - dd).days, "T": T,
                             "price_ts": last["end_period_ts"] if last else np.nan,
                             "X": m.X, "result": m.result, "close_time": m.close_time, "obs_d": obs_d,
                             "acis_missing": n_missing,
                             **{f"fc_{k}": f.get(k, np.nan) for k in MODELS},
                             "yes_bid": _px(last, "yes_bid") if last else np.nan,
                             "yes_ask": _px(last, "yes_ask") if last else np.nan,
                             "vol_window": float(sum(float(x.get("volume_fp", x.get("volume", 0)) or 0)
                                                     for x in cs.get(m.ticker, [])))})
        print(f"  {ev:22s} {sid} {kind} markets {len(g)} acis_missing {n_missing} "
              f"fc models d=end-6: {fc[(fc.d == end - pd.Timedelta(days=6))].F.notna().sum()}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    pd.DataFrame(obs_rows).to_parquet(OBS, index=False)
    print(f"-> {OUT} ({len(out)} rows), {OBS}")
    return 0


def cmd_audit() -> int:
    """ACIS month total vs settlement: for each settled market, YES should equal (acis_total > X)."""
    df = pd.read_parquet(OUT).drop_duplicates("ticker")
    ob = pd.read_parquet(OBS)
    df = df.merge(ob[["event_ticker", "acis_total"]], on="event_ticker")
    df = df[df.acis_missing == 0]
    pred = np.where(df.acis_total > df.X + 1e-9, "yes", "no")
    df["agree"] = pred == df.result
    print(f"markets {len(df)}  agreement {df.agree.mean():.4f}")
    bad = df[~df.agree]
    print(bad.groupby(["kind", "series"]).size().to_string() if len(bad) else "no mismatches")
    print(bad[["ticker", "X", "acis_total", "result"]].head(40).to_string())
    by = df.groupby("event_ticker").agree.all()
    print(f"city-months fully consistent: {by.mean():.3f} ({by.sum()}/{len(by)})")
    return 0


# ------------------------------------------------------------------------------------------------ scoring
def close_s(x: pd.Series) -> pd.Series:
    ct = pd.to_datetime(x, format="ISO8601", utc=True)
    return (ct - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)


def picks(df: pd.DataFrame, trial: str) -> pd.DataFrame:
    kind = "rain" if trial[0] == "W" else "snow"
    d = df[(df.kind == kind) & (df.acis_missing == 0)].copy()
    d = d[(d.yes_bid > 0) & (d.yes_ask < 1) & ((d.yes_ask - d.yes_bid) <= 0.10 + 1e-9)]
    # guard (added before scoring, after seeing that Kalshi settles LOCKED ladders early and close_time records it):
    # the market must still be open at T_d, else the last candle is a pre-settlement quote nobody could trade.
    d = d[d["T"] < close_s(d.close_time)]
    cols = [f"fc_{m}" for m in MODELS]
    n = d[cols].notna().sum(1)
    fmax = d[cols].max(1)
    gap = d.X - d.obs_d
    if trial.startswith(("W1", "S1")):
        margin = 0.50 if kind == "rain" else 1.0
        s = d[(n >= 3) & (gap > 2 * fmax + margin) & (d.yes_ask > 0.01)].copy()
        s["side"] = "no"
    else:
        s = d[(d.obs_d > d.X + 1e-9) & (d.yes_ask <= 0.98)].copy()
        s["side"] = "yes"
    s = s.sort_values("d").drop_duplicates("ticker", keep="first")
    s["won"] = (s.result == s.side).astype(int)
    if (s.side == "no").all():
        a = 1 - s.yes_bid
    else:
        a = s.yes_ask
    s["taker_cost"] = a + FEE * a * (1 - a)
    return s


def tape(tk: str, t0: int) -> pd.DataFrame:
    out = []
    for path in ("historical/trades", "markets/trades"):
        cur = None
        while True:
            p = {"ticker": tk, "min_ts": t0, "limit": 1000}
            if cur:
                p["cursor"] = cur
            d = fetch(f"{wx.API}/{path}", p, cache=True)
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
        rows.append({"ts": int(pd.Timestamp(r["created_time"]).timestamp()), "no_price": npx,
                     "count": float(r.get("count_fp") or r.get("count") or 0)})
    return pd.DataFrame(rows, columns=["ts", "no_price", "count"])


def maker(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    no_ask = 1 - s.yes_bid
    mid = 1 - (s.yes_bid + s.yes_ask) / 2
    s["q"] = np.round(np.minimum(np.floor(np.round(mid * 100, 6)) / 100, no_ask - 0.01), 4)
    close = close_s(s.close_time)
    filled = []
    for tk, T, q, c in zip(s.ticker, s["T"], s.q, close):
        t = tape(tk, int(T))
        m = (t.ts > T) & (t.ts <= c) & (t.no_price <= q + 1e-9)
        filled.append(bool(t.loc[m, "count"].sum() >= 1))
    s["filled"] = filled
    s["maker_cost"] = s.q
    return s


def score(s: pd.DataFrame, cost: str, bet: pd.Series | None = None, rng=None) -> dict:
    rng = rng or np.random.default_rng(0)
    b = s if bet is None else s[bet.to_numpy()]
    pnl = np.where(b.won == 1, 1 / b[cost] - 1, -1.0)
    ev = b.event_ticker.to_numpy()
    u, inv = np.unique(ev, return_inverse=True) if len(ev) else (np.array([]), np.array([], int))
    sx = np.bincount(inv, weights=pnl, minlength=len(u)) if len(u) else np.zeros(0)
    boots = sx[rng.integers(0, len(u), size=(4000, len(u)))].sum(1) if len(u) else np.zeros(1)
    k = int((b.won == 0).sum())
    n = len(b)
    lo, hi = (stats.binomtest(k, n).proportion_ci(method="exact") if n else (np.nan, np.nan))
    months = sorted(b.month.unique())
    half = months[len(months) // 2] if months else None
    h1 = pnl[(b.month < half).to_numpy()] if half is not None else np.array([])
    h2 = pnl[(b.month >= half).to_numpy()] if half is not None else np.array([])
    return {"bets": int(n), "city_months": int(len(u)), "losses": k, "loss_rate": k / n if n else np.nan,
            "loss_ci": [float(lo), float(hi)], "units": float(pnl.sum()), "roi": float(pnl.mean()) if n else 0.0,
            "avg_cost": float(b[cost].mean()) if n else np.nan,
            "breakeven_loss_rate": float(1 - b[cost].mean()) if n else np.nan,
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)),
            "p": float(np.mean(boots <= 0)),
            "half1": [float(h1.mean()) if len(h1) else None, int(len(h1))],
            "half2": [float(h2.mean()) if len(h2) else None, int(len(h2))],
            "median_window_vol": float(b.vol_window.median()) if n else 0.0}


def null_p(b: pd.DataFrame, cost: str, n=20000, seed=1) -> tuple[float, float]:
    """Market null (protocol rule 5): each bet wins with prob = its all-in cost (price is fair). Returns
    (P(sim units >= observed) with bets independent, same with ONE bet per city-month = the lowest-margin pick).
    Added before the ledger write because the cluster bootstrap returns p = 0 whenever a sample has no losses."""
    rng = np.random.default_rng(seed)
    out = []
    for bb in (b, b.assign(m=b.X - b.obs_d).sort_values("m").drop_duplicates("event_ticker")):
        if not len(bb):
            out.append(1.0)
            continue
        c = bb[cost].to_numpy(float)
        obs = np.where(bb.won == 1, 1 / c - 1, -1.0).sum()
        w = rng.random((n, len(c))) < c
        sim = np.where(w, 1 / c - 1, -1.0).sum(1)
        out.append(float(np.mean(sim >= obs - 1e-12)))
    return out[0], out[1]


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df = df[((df.kind == "rain") & (df.month < SPLIT["rain"])) | ((df.kind == "snow") & (df.month < SPLIT["snow"]))]
    tr = df.drop_duplicates("ticker")
    print(f"DEV: rain {tr[tr.kind == 'rain'].event_ticker.nunique()} city-months / {(tr.kind == 'rain').sum()} markets; "
          f"snow {tr[tr.kind == 'snow'].event_ticker.nunique()} city-months / {(tr.kind == 'snow').sum()} markets")
    q = df[(df.yes_bid > 0) & (df.yes_ask < 1)]
    print(f"  tradeable quote share in window: {len(q) / max(len(df), 1):.3f}; "
          f"rows w/ >=3 models: {(df[[f'fc_{m}' for m in MODELS]].notna().sum(1) >= 3).mean():.3f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r.get("dataset") == DATASET}
    recs, res = [], {}
    for name in TRIALS:
        base = name.replace("m_", "_") if "m_" in name else name
        s = picks(df, base[:2] + base[2:].replace("_maker", ""))
        if name.endswith("_maker"):
            s = maker(s)
            r = score(s, "maker_cost", s.filled)
            r["fill_rate"] = float(s.filled.mean()) if len(s) else np.nan
            r["loss_rate_unfilled"] = float(1 - s.loc[~s.filled, "won"].mean()) if (~s.filled).any() else np.nan
            r["p_null"], r["p_null_1per_cm"] = null_p(s[s.filled], "maker_cost")
        else:
            r = score(s, "taker_cost")
            r["p_null"], r["p_null_1per_cm"] = null_p(s, "taker_cost")
        res[name] = (s, r)
        sha = hashlib.sha256(("kalshi_wx_winter:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:32s} bets {r['bets']:4d} ({r['city_months']} city-months) losses {r['losses']} "
              f"[{r['loss_ci'][0]:.3f},{r['loss_ci'][1]:.3f}] BE {r['breakeven_loss_rate']:.3f} units {r['units']:+7.2f} "
              f"ROI {r['roi']:+.2%} CI [{r['ci_lo']:+.1f},{r['ci_hi']:+.1f}] p {r['p']:.4f} halves {r['half1']} {r['half2']}"
              + (f" fill {r['fill_rate']:.2f}" if 'fill_rate' in r else "")
              + f" | market-null p {r['p_null']:.4f} (1/city-month {r['p_null_1per_cm']:.4f})")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_wx_winter_{name}",
                     "sha": sha, "dataset": DATASET, "trial": name,
                     "hypothesis": f"pre-registered in scripts/kalshi_wx_winter.py ({name})",
                     "source": "Kalshi monthly rain/snow ladders, ACIS obs + Open-Meteo previous-runs (2026-10-27)",
                     "status": "ok", "study_trials": len(TRIALS), "bonferroni_bar": 0.05 / len(TRIALS), **r,
                     "passed": bool(max(r["p"], r["p_null"]) < 0.05 / len(TRIALS) and r["units"] > 0)})
    for name, (s, r) in res.items():
        if len(s):
            print(f"\n{name}: by days_left / series")
            if "filled" in s:
                s = s[s.filled]
            c = s["maker_cost"] if name.endswith("_maker") else s["taker_cost"]
            pn = np.where(s.won == 1, 1 / c - 1, -1.0)
            print(s.assign(pnl=pn).groupby("days_left").pnl.agg(["size", "sum"]).round(2).T.to_string())
            print(s.assign(pnl=pn).groupby("series").pnl.agg(["size", "sum"]).round(2).T.to_string())
            if (s.won == 0).any():
                print("losers:", s.loc[s.won == 0, ["ticker", "d", "X", "obs_d", "yes_bid", "yes_ask"]].to_string())
    pd.concat([s.assign(trial=n) for n, (s, _) in res.items() if len(s)]).to_parquet(
        ROOT / "data" / "kalshi_wx_winter_dev_picks.parquet", index=False)
    print(f"\n{DATASET} trials {len(shas)} -> bar p < {0.05 / len(TRIALS):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "audit": cmd_audit, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
