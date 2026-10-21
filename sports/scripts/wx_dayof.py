"""DAY-OF consensus-reject on Kalshi daily-HIGH markets (QUEUE v2 N1). DEV only.

    python scripts/wx_dayof.py build
    python scripts/wx_dayof.py dev [--ledger]

Idea: the two validated rules decide at 22:00 the night before with lead-1 forecasts. Kalshi high-temp markets trade
until the afternoon of the day, and the overnight model runs are sharper -> more brackets can be rejected with the
same frozen 3F rule = more bets/month.

Leak-free design (2026-10-19, before any day-of price or forecast was pulled):
  forecast : Open-Meteo SINGLE-RUNS archive (exists from ~2026-04; probed 10-19), the 00Z run of day D for
             gfs025, ecmwf_ifs025, icon_global, gem_seamless, ncep_nbm_conus; daily max over the station-local
             calendar day D (gfs025 returns null temperature_2m in single-runs -> gfs_seamless, swapped
             10-19 before any build). 00Z runs are published by ~07Z at the latest (ECMWF), i.e. before 06:00 local in every
             US zone used here (06:00 EDT = 10Z ... 06:00 PDT = 13Z).
  decision : 06:00 station-local on day D; last hourly candle ending at or before; tradeable quote
             (yes_bid > 0, yes_ask < 1, spread <= 10c), yes_ask > 0.01; buy NO at 1 - yes_bid; fee 0.07*p*(1-p).
  rule     : FROZEN reject logic of kalshi_reject_safe_check (>= 3 models, 3F margin).
  cities   : the 24 paper cities (orig 7 + X1 17).
  split    : DEV = dates < 2026-08-01; SEALED = 2026-08-01..2026-10-18 (user's yes).
  D1 dayof_all  : every day-of pick
  D2 dayof_new  : day-of picks whose bracket the NIGHT-BEFORE rule did not pick (lead-1 fcsafe max, 22:00 prices)
                  -- the genuinely new volume
  2 trials -> bar p < 0.025, one-sided bootstrap over station-days.
Disclosed caveat: Apr-Oct 2026 is inside the night-before rules' already-scored sealed windows. Day-of is a different
decision time + forecast, but its outcomes overlap days whose night-before result we have seen; D2 removes the
overlapping brackets, so D2 is the cleaner test.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports import wx
from sports.http import fetch, fetch_many
from kalshi_paper import PAPER_CITIES
from wx_expand import model_ext, score

OUT = ROOT / "data" / "wx_dayof.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_temp_dayof"
START, SPLIT = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-08-01")
RUN_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_global", "gem_seamless", "ncep_nbm_conus"]
LEAD1 = {"gfs_seamless": "2021-03-01", "ecmwf_ifs025": "2024-03-01", "icon_seamless": "2024-03-01",
         "gem_seamless": "2024-03-01", "ncep_nbm_conus": "2025-03-01"}
SINGLE = "https://single-runs-api.open-meteo.com/v1/forecast"
MARGIN = 3.0
TRIALS = ["D1_dayof_all", "D2_dayof_new"]


def run_max(cities: dict, coords: dict, day: pd.Timestamp, model: str) -> dict:
    """Max over each city's local day D from the 00Z run of D (one call, all cities)."""
    names = list(cities)
    try:  # an unavailable run answers HTTP 400 ("requested model run is not available") -> no data that day
        d = fetch(SINGLE, {"latitude": ",".join(str(coords[c][0]) for c in names),
                           "longitude": ",".join(str(coords[c][1]) for c in names), "hourly": "temperature_2m",
                           "models": model, "run": f"{day.date()}T00:00", "forecast_days": 2, "temperature_unit": "fahrenheit",
                           "timezone": "GMT"}, cache=True)
    except Exception as e:
        if "400" in str(e):
            return {}
        if "429" in str(e):  # Open-Meteo free tier counts each location: ~5k/h, 10k/day -> wait and retry
            import time
            print(f"  429 at {model} {day.date()}; sleeping 10 min", flush=True)
            time.sleep(600)
            return run_max(cities, coords, day, model)
        raise
    if isinstance(d, dict) and d.get("error"):
        return {}
    d = d if isinstance(d, list) else [d]
    out = {}
    for c, loc in zip(names, d):
        h = pd.DataFrame(loc.get("hourly", {}))
        if h.empty:
            continue
        t = pd.to_datetime(h["time"]).dt.tz_localize("UTC").dt.tz_convert(cities[c][1])
        v = h.loc[(t.dt.date == day.date()).to_numpy(), "temperature_2m"].dropna()
        if len(v) >= 20:
            out[c] = float(v.max())
    return out


def prices_at(series, mk, tz, hour, day_offset):
    """yes bid/ask at `hour` local on (event date + day_offset) -- same fetch path as wx.prices_at_decision."""
    events = sorted(mk.event_ticker.unique())
    ts = {}
    for ev in events:
        dd = wx.event_date(ev) + pd.Timedelta(days=day_offset)
        ts[ev] = int(datetime(dd.year, dd.month, dd.day, hour, tzinfo=ZoneInfo(tz)).timestamp())
    urls = [f"{wx.API}/series/{series}/events/{ev}/candlesticks?start_ts={ts[ev] - 3 * 3600}&end_ts={ts[ev]}&period_interval=60"
            for ev in events]
    out, missing = [], []
    for ev, d in zip(events, fetch_many(urls)):
        if not d.get("market_tickers"):
            missing.append(ev)
        for tk, c in zip(d.get("market_tickers", []), d.get("market_candlesticks", [])):
            r = wx._row(tk, ev, c, ts[ev])
            if r:
                out.append(r)
    # archived events answer nothing on the event endpoint -> per-market /historical fallback (as wx.prices_at_decision;
    # missing this dropped every event before ~2026-08 in the first build, 10-21)
    leg = mk[mk.event_ticker.isin(missing)]
    urls = [f"{wx.API}/historical/markets/{tk}/candlesticks?start_ts={ts[ev] - 3 * 3600}&end_ts={ts[ev]}&period_interval=60"
            for tk, ev in zip(leg.ticker, leg.event_ticker)]
    for (tk, ev), d in zip(zip(leg.ticker, leg.event_ticker), fetch_many(urls)):
        r = wx._row(tk, ev, d.get("candlesticks", []), ts[ev])
        if r:
            out.append(r)
    return pd.DataFrame(out)


def cmd_build() -> int:
    coords = {s: wx.station_coords(st) for s, (st, tz) in PAPER_CITIES.items()}
    days = pd.date_range(START, pd.Timestamp("2026-10-18"))
    fc = {}
    for m in RUN_MODELS:
        n = 0
        for day in days:
            for c, v in run_max(PAPER_CITIES, coords, day, m).items():
                fc[(c, day)] = {**fc.get((c, day), {}), f"run_{m}": v}
                n += 1
        print(f"{m}: {n} city-days", flush=True)
    fcdf = pd.DataFrame([{"series": c, "date": d, **v} for (c, d), v in fc.items()])
    frames = []
    for s, (st, tz) in PAPER_CITIES.items():
        mk = wx.settled_markets(s)
        mk["date"] = mk.event_ticker.map(wx.event_date)
        mk = mk[(mk.date >= START) & (mk.date <= pd.Timestamp("2026-10-18"))]
        if mk.empty:
            continue
        day = prices_at(s, mk, tz, 6, 0)
        nb = prices_at(s, mk, tz, 22, -1).rename(columns={"yes_ask": "nb_yes_ask", "yes_bid": "nb_yes_bid"})
        df = mk.merge(day[["ticker", "yes_ask", "yes_bid"]], on="ticker", how="inner")
        df = df.merge(nb[["ticker", "nb_yes_ask", "nb_yes_bid"]], on="ticker", how="left")
        lat, lon = coords[s]
        for m, start in LEAD1.items():
            f = model_ext(m, lat, lon, tz, max(start, str(df.date.min().date())), str(df.date.max().date()), low=False)
            if f is not None:
                df = df.join(f, on="date")
        df["series"] = s
        frames.append(df)
        print(f"{s}: {len(df)} brackets {df.event_ticker.nunique()} days", flush=True)
    out = pd.concat(frames, ignore_index=True).merge(fcdf, on=["series", "date"], how="left")
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rows); day-of forecast coverage "
          f"{ {m: round(out['run_' + m].notna().mean(), 2) for m in RUN_MODELS} }")
    return 0


def _rej(d, cols):
    n = d[cols].notna().sum(1)
    hi, lo = d[cols].max(1), d[cols].min(1)
    r = np.where(d.strike_type == "greater", hi + MARGIN < d.floor_strike + 1,
        np.where(d.strike_type == "less", lo - MARGIN > d.cap_strike - 1,
                 (hi + MARGIN < d.floor_strike) | (lo - MARGIN > d.cap_strike)))
    return r & (n >= 3)


def picks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    v = df.expiration_value
    mine = np.select([df.strike_type == "greater", df.strike_type == "less", df.strike_type == "between"],
                     [v > df.floor_strike, v < df.cap_strike, (v >= df.floor_strike) & (v <= df.cap_strike)], default=np.nan)
    d = df[mine.astype(float) == (df.result == "yes").astype(float)].copy()
    day = _rej(d, [c for c in d.columns if c.startswith("run_")]) & (d.yes_bid > 0) & (d.yes_ask < 1) & \
        ((d.yes_ask - d.yes_bid) <= 0.10) & (d.yes_ask > 0.01)
    nb = _rej(d, [c for c in d.columns if c.startswith("fcsafe_")]) & (d.nb_yes_bid > 0) & (d.nb_yes_ask < 1) & \
        ((d.nb_yes_ask - d.nb_yes_bid) <= 0.10) & (d.nb_yes_ask > 0.01)
    s = d[day & (~nb if name == "D2_dayof_new" else True)].copy()
    cost = (1 - s.yes_bid) + wx.FEE_RATE * (1 - s.yes_bid) * s.yes_bid
    s["pnl"] = np.where(s.result == "no", 1 / cost - 1, -1.0)
    s["stake"] = 1.0
    return s


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date < SPLIT]
    print(f"DEV {df.series.nunique()} series, {df.event_ticker.nunique()} station-days, {df.date.min().date()}..{df.date.max().date()}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = picks(df, name)
        r = score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("wx_dayof:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        months = max(1, (t.date.max() - t.date.min()).days / 30.4) if len(t) else 1
        print(f"{name:13s} bets {r['bets']:5d} (~{r['bets'] / months:.0f}/mo) losses {r['losses']}  units {r['units']:+7.2f}  "
              f"ROI {r['roi']:+6.2%}  95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  avg NO {(1 - t.yes_bid).mean():.3f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_dayof_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/wx_dayof.py ({name})",
                     "source": "consensus-reject at 06:00 day-of", "status": "ok", "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
