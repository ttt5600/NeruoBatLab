"""Kalshi daily RAIN markets (KXRAIN "Where will it rain daily", ~35 US cities): model-consensus rule. DEV only.

    python scripts/kalshi_rain.py build            # settled city-days + price at decision + 5-model forecasts (cached)
    python scripts/kalshi_rain.py dev [--ledger]

Contract: YES iff total precipitation at the city's climate station (The Weather Company) on the day is strictly
> 0 in; trace and missing count as 0. Series starts 2026-08 (KXRAIN-26AUG01). Liquid: Oct-19 WPB 33k contracts.

Pre-registered 2026-10-19, before any KXRAIN price or forecast was pulled (ORCHESTRATION Q4):
  decision : 22:00 America/New_York the day before, for EVERY city (western cities decide earlier = still
             pre-event); last hourly candle ending at or before; tradeable quote = yes_bid > 0, yes_ask < 1,
             spread <= 10c. Fee 0.07*p*(1-p).
  forecast : Open-Meteo previous-runs, LEAD-2 (temperature-free: precipitation_previous_day2), summed over the
             station-local calendar day. Lead-2 values for day D were issued ~48 h before each hour -> all before
             the decision (lead-1 hours 22-23 would not be). Models gfs_seamless, ecmwf_ifs025, icon_seamless,
             gem_seamless, ncep_nbm_conus; need >= 3 with a full day (>= 20 hours).
  R1 dry_no   : every available model's daily total < 0.01 in -> buy NO at 1 - yes_bid, if yes_ask > 0.01
  R2 wet_yes  : every available model's daily total >= 0.10 in -> buy YES at the ask
  split : DEV = dates < 2026-09-15; SEALED = 2026-09-15+ (never loaded by `dev`; user's yes)
  2 trials -> bar p < 0.025, one-sided bootstrap over DAYS (cities on one day share weather systems).
Caveat: dev is ~6 weeks; underpowered by design -- dev decides only whether the sealed test is worth asking for.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports import wx
from sports.http import fetch

SERIES = "KXRAIN"
OUT = ROOT / "data" / "kalshi_rain.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_rain"
SPLIT = pd.Timestamp("2026-09-15")
MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "ncep_nbm_conus"]
TRIALS = ["R1_dry_no", "R2_wet_yes"]


def settled() -> pd.DataFrame:
    rows = []
    for path, extra in ((f"{wx.API}/historical/markets", {}), (f"{wx.API}/markets", {"status": "settled"})):
        cur = None
        while True:
            p = {"series_ticker": SERIES, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache="historical" in path)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.result.isin(["yes", "no"])].copy()
    df["date"] = df.event_ticker.map(wx.event_date)
    df["city"] = df.ticker.str.split("-").str[-1]
    df["station"] = df.rules_primary.str.extract(r"at CLI([A-Z]{3})", expand=False)
    return df[["ticker", "event_ticker", "date", "city", "station", "result", "yes_sub_title"]]


def locate(station: str, name: str):
    try:
        p = fetch(f"https://api.weather.gov/stations/K{station}")
        lon, lat = p["geometry"]["coordinates"][:2]
        return lat, lon, p["properties"]["timeZone"], f"K{station}"
    except Exception:
        g = fetch("https://geocoding-api.open-meteo.com/v1/search", {"name": name, "count": 5, "countryCode": "US"})
        r = g["results"][0]
        return r["latitude"], r["longitude"], r["timezone"], f"geo:{name}"


def model_total(model, lat, lon, tz, start, end) -> pd.Series | None:
    d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model, "hourly": "precipitation_previous_day2",
                             "precipitation_unit": "inch", "timezone": tz, "start_date": start, "end_date": end},
              cache=pd.Timestamp(end) < pd.Timestamp.today().normalize() - pd.Timedelta(days=3))
    h = pd.DataFrame(d.get("hourly", {}))
    if h.empty:
        return None
    h["day"] = pd.to_datetime(h["time"]).dt.normalize()
    g = h.groupby("day")["precipitation_previous_day2"]
    return g.sum(min_count=1).where(g.count() >= 20).rename(f"fc_{model}")


def cmd_build() -> int:
    mk = settled()
    print(f"settled {len(mk)} city-days, {mk.city.nunique()} cities, {mk.date.min().date()}..{mk.date.max().date()}")
    px = wx.prices_at_decision(SERIES, mk, "America/New_York")
    df = mk.merge(px, on=["ticker", "event_ticker"], how="inner")
    frames = []
    for city, g in df.groupby("city"):
        lat, lon, tz, src = locate(g.station.iloc[0], g.yes_sub_title.iloc[0])
        g = g.copy()
        for m in MODELS:
            s = model_total(m, lat, lon, tz, str(g.date.min().date()), str(g.date.max().date()))
            if s is not None:
                g = g.join(s, on="date")
        g["loc_src"] = src
        frames.append(g)
        print(f"  {city:5s} {src:12s} {tz:20s} days {len(g)}  fc coverage "
              f"{ {m[:4]: round(g.get('fc_' + m, pd.Series(dtype=float)).notna().mean(), 2) for m in MODELS} }", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rows)")
    return 0


def trial(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df[(df.yes_bid > 0) & (df.yes_ask < 1) & ((df.yes_ask - df.yes_bid) <= 0.10)].copy()
    cols = [c for c in d.columns if c.startswith("fc_")]
    n = d[cols].notna().sum(1)
    if name == "R1_dry_no":
        s = d[(n >= 3) & (d[cols].max(1) < 0.01) & (d.yes_ask > 0.01)].copy()
        cost = (1 - s.yes_bid) + wx.FEE_RATE * (1 - s.yes_bid) * s.yes_bid
        won = s.result == "no"
    else:
        s = d[(n >= 3) & (d[cols].min(1) >= 0.10)].copy()
        cost = s.yes_ask + wx.FEE_RATE * s.yes_ask * (1 - s.yes_ask)
        won = s.result == "yes"
    s["pnl"] = np.where(won, 1 / cost - 1, -1.0)
    s["price"] = cost
    return s


def score(t, rng) -> dict:
    pm = t.groupby("date")["pnl"].sum().to_numpy()
    boots = np.array([rng.choice(pm, len(pm)).sum() for _ in range(3000)]) if len(pm) else np.zeros(1)
    return {"bets": int(len(t)), "staked": float(len(t)), "units": float(t.pnl.sum()), "roi": float(t.pnl.mean()) if len(t) else 0.0,
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
            "days": int(len(pm)), "losses": int((t.pnl < 0).sum())}


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date < SPLIT]
    y = (df.result == "yes").astype(float)
    mid = (df.yes_bid + df.yes_ask) / 2
    print(f"DEV {df.city.nunique()} cities, {df.date.nunique()} days, {len(df)} city-days; base rain rate {y.mean():.3f}; "
          f"Brier mid {np.mean((mid - y) ** 2):.4f}")
    cols = [c for c in df.columns if c.startswith("fc_")]
    for c in cols:
        ok = df[c].notna()
        print(f"  {c:18s} coverage {ok.mean():.2f}  P(rain | fc<0.01) {y[ok & (df[c] < .01)].mean():.3f}  "
              f"P(rain | fc>=0.10) {y[ok & (df[c] >= .10)].mean():.3f}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = trial(df, name)
        r = score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("kalshi_rain:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        print(f"{name:10s} bets {r['bets']:4d} over {r['days']} days, losses {r['losses']}, units {r['units']:+6.2f}, "
              f"ROI {r['roi']:+6.2%}, 95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], p {r['p']:.4f}, avg cost {t.price.mean():.3f}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_rain_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_rain.py ({name})",
                     "source": "model consensus (kalshi temp analogue)", "status": "ok", "trial_number": len(shas), **r})
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
