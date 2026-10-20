"""Consensus-reject on the NEW Kalshi temperature series (14 more daily highs, 27 daily lows). DEV only.

    python scripts/wx_expand.py build      # settled brackets + 22:00-day-before prices + 5-model forecasts (cached)
    python scripts/wx_expand.py dev [--ledger]

Why (ORCHESTRATION Q1, 2026-10-19): the one lockbox-passed rule (kalshi_reject_safe_check, 7 high-temp cities,
sealed +1.45%/1,844) trades 7 of the 48 US temperature series Kalshi lists; more series = more bets/month at the
same stake, which is the only evidence-backed path to $100/month.

RULE FROZEN (identical to lab/strategies/kalshi_reject_safe_check.py; nothing fitted or tuned here):
  per model in {gfs_seamless, ecmwf_ifs025, icon_seamless, gem_seamless, ncep_nbm_conus}: lead-1-day forecast,
  local-day max (HIGH series) or min (LOW series) over hours <= 18:00 only (all issued before the decision);
  reject the bracket if even the most extreme available model (>= 3 available) misses it by > 3F
  (greater: hi + 3 < floor + 1; less: lo - 3 > cap - 1; between: hi + 3 < floor or lo - 3 > cap);
  buy NO at 1 - yes_bid if yes_ask > 0.01; tradeable quotes only (yes_bid > 0, yes_ask < 1, spread <= 10c).
  Decision = 22:00 station-local the day before (hourly candle ending at or before). Fee 0.07*p*(1-p).
Pre-registered 2026-10-19, before any price or forecast was pulled:
  split : DEV = dates < 2026-03-01; SEALED = 2026-03-01+ (never loaded by `dev`; user's yes required)
  X1 new_highs : the 14 KXHIGH* series not already traded
  X2 lows      : the 27 KXLOW* series (forecast MIN instead of max; same margin; LST-vs-local-day caveat below)
  2 trials -> bar p < 0.025 each, one-sided bootstrap over station-days. Stake 1 unit of cost per bet.
Caveats: NWS/TWC climate day is local STANDARD time; forecasts use the local calendar day (1 h off in DST).
For lows that hour sits at midnight, where a late-evening cold front can set the low -- the 3F margin is the
only protection; the dev result measures whether it is enough. Settlement switched to The Weather Company
2026-08-14 (all sealed) and matches NWS 3,080/3,081 days (scripts/wx_source_check.py).
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

OUT = ROOT / "data" / "wx_expand.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_temp_expand"
SPLIT = pd.Timestamp("2026-03-01")
TRADED = set(wx.CITIES)
MODELS = {"gfs_seamless": "2021-03-01", "ecmwf_ifs025": "2024-03-01", "icon_seamless": "2024-03-01",
          "gem_seamless": "2024-03-01", "ncep_nbm_conus": "2025-03-01"}
MARGIN = 3.0
TRIALS = ["X1_new_highs", "X2_lows"]


def series_list() -> list[str]:
    d = fetch(f"{wx.API}/series", {"limit": 5000}, cache=False)["series"]
    s = sorted(x["ticker"] for x in d if re.match(r"^KX(HIGH|LOW)T?[A-Z]{2,5}$", x["ticker"])
               and x.get("category") == "Climate and Weather")
    return [t for t in s if t not in TRADED and fetch(f"{wx.API}/markets", {"series_ticker": t, "status": "open", "limit": 1},
                                                      cache=False).get("markets")]


def station_of(series: str) -> str | None:
    m = fetch(f"{wx.API}/markets", {"series_ticker": series, "status": "open", "limit": 1}, cache=False)["markets"][0]
    g = re.search(r"\(CLI([A-Z]{3})\)", m.get("rules_primary", ""))
    return f"K{g.group(1)}" if g else None


def model_ext(model, lat, lon, tz, start, end, low: bool):
    frames, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        done = e < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
        d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model,
                                 "hourly": "temperature_2m_previous_day1", "temperature_unit": "fahrenheit",
                                 "timezone": tz, "start_date": s.date().isoformat(), "end_date": e.date().isoformat()}, cache=done)
        h = pd.DataFrame(d.get("hourly", {}))
        if not h.empty:
            t = pd.to_datetime(h["time"])
            h["day"] = t.dt.normalize()
            early = h[t.dt.hour <= 18].groupby("day")["temperature_2m_previous_day1"]
            v = early.min() if low else early.max()
            frames.append(v.where(early.count() >= 15).rename(f"fcsafe_{model}"))
        s = e + pd.Timedelta(days=1)
    return pd.concat(frames) if frames else None


def cmd_build() -> int:
    frames = []
    for s in series_list():
        st = station_of(s)
        if not st:
            print(f"{s}: no station in rules, skipped")
            continue
        p = fetch(f"https://api.weather.gov/stations/{st}")
        lon, lat = p["geometry"]["coordinates"][:2]
        tz = p["properties"]["timeZone"]
        mk = wx.settled_markets(s)
        if mk.empty:
            print(f"{s}: no settled markets")
            continue
        mk["date"] = mk["event_ticker"].map(wx.event_date)
        px = wx.prices_at_decision(s, mk, tz)
        df = mk.merge(px, on=["ticker", "event_ticker"], how="inner")
        low = s.startswith("KXLOW")
        for m, start in MODELS.items():
            f = model_ext(m, lat, lon, tz, max(start, str(df.date.min().date())), str(df.date.max().date()), low)
            if f is not None:
                df = df.join(f, on="date")
        v = df["expiration_value"]
        mine = np.select([df.strike_type == "greater", df.strike_type == "less", df.strike_type == "between"],
                         [v > df.floor_strike, v < df.cap_strike, (v >= df.floor_strike) & (v <= df.cap_strike)], default=np.nan)
        df["result_mismatch"] = (mine.astype(float) != (df.result == "yes").astype(float)) & ~np.isnan(mine.astype(float))
        df["series"], df["station"], df["kind"] = s, st, "low" if low else "high"
        print(f"{s} {st} {tz}: {len(df)} priced brackets, {df.event_ticker.nunique()} days "
              f"{df.date.min().date()}..{df.date.max().date()}, dev days {df[df.date < SPLIT].event_ticker.nunique()}, "
              f"mismatch {int(df.result_mismatch.sum())}", flush=True)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rows)")
    return 0


def picks(df: pd.DataFrame) -> pd.DataFrame:
    d = df[~df.result_mismatch & (df.yes_bid > 0) & (df.yes_ask < 1) & ((df.yes_ask - df.yes_bid) <= 0.10)].copy()
    cols = [c for c in d.columns if c.startswith("fcsafe_")]
    n = d[cols].notna().sum(1)
    hi, lo = d[cols].max(1), d[cols].min(1)
    rej = np.where(d.strike_type == "greater", hi + MARGIN < d.floor_strike + 1,
          np.where(d.strike_type == "less", lo - MARGIN > d.cap_strike - 1,
                   (hi + MARGIN < d.floor_strike) | (lo - MARGIN > d.cap_strike)))
    s = d[rej & (n >= 3) & (d.yes_ask > 0.01)].copy()
    cost = (1 - s.yes_bid) + wx.FEE_RATE * (1 - s.yes_bid) * s.yes_bid
    s["stake"] = 1.0
    s["pnl"] = np.where(s.result == "no", 1 / cost - 1, -1.0)
    return s


def score(t: pd.DataFrame, rng) -> dict:
    t = t.assign(day=t.series + t.date.astype(str))
    pm = t.groupby("day")["pnl"].sum()
    boots = np.array([rng.choice(pm.to_numpy(), len(pm)).sum() for _ in range(3000)])
    q = t.date.dt.year.astype(str) + "Q" + t.date.dt.quarter.astype(str)
    per = t.groupby(q)["pnl"].sum().round(2)
    return {"bets": int(len(t)), "staked": float(len(t)), "units": float(t.pnl.sum()), "roi": float(t.pnl.mean()) if len(t) else 0.0,
            "ci_lo": float(np.quantile(boots, .025)), "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)),
            "folds_positive": int((per > 0).sum()), "per_fold": per.to_dict(), "losses": int((t.pnl < 0).sum())}


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date < SPLIT]  # sealed rows never enter dev
    print(f"DEV {df.series.nunique()} series, {df.event_ticker.nunique()} station-days, {df.date.min().date()}..{df.date.max().date()}")
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    for name in TRIALS:
        t = picks(df[df.kind == ("high" if name == "X1_new_highs" else "low")])
        r = score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("wx_expand:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        months = max(1, (t.date.max() - t.date.min()).days / 30.4) if len(t) else 1
        print(f"{name:13s} bets {r['bets']:5d} (~{r['bets'] / months:.0f}/mo) losses {r['losses']}  units {r['units']:+7.2f}  "
              f"ROI {r['roi']:+6.2%}  95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  quarters+ "
              f"{r['folds_positive']}/{len(r['per_fold'])}  avg NO {(1 - t.yes_bid).mean():.3f}")
        by = t.groupby("series").pnl.agg(["size", "sum"]).round(2)
        print("   by series:", {k: f"{v['sum']:+.2f}/{int(v['size'])}" for k, v in by.iterrows()})
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"wx_expand_{name}", "sha": sha,
                     "dataset": DATASET, "hypothesis": f"pre-registered in scripts/wx_expand.py ({name}); rule frozen from "
                     "kalshi_reject_safe_check", "source": "replication on new series", "status": "ok",
                     "trial_number": len(shas), **r})
    print(f"{DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print("appended to ledger")
    return 0


def cmd_sealed() -> int:
    """ONE-TIME sealed score, user's explicit yes given 2026-10-19 ("yes" to scoring X1_new_highs on its held-back data).
    Written before any sealed row was scored. Same rule and scorer as dev; rows with date >= SPLIT only.
    PASS iff one-sided bootstrap p < 0.05 / (#kalshi_temp_expand promotions incl. this one) and units > 0. Refuses a re-score."""
    if "--confirm" not in sys.argv or "X1_new_highs" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm X1_new_highs")
        return 1
    lock = ROOT / "lab" / "lockbox.jsonl"
    prior = [json.loads(l) for l in lock.read_text().splitlines() if l.strip()]
    sha = hashlib.sha256(("wx_expand:X1_new_highs").encode()).hexdigest()[:12]
    if any(r["sha"] == sha for r in prior):
        print("X1_new_highs already scored on the lockbox; refusing")
        return 1
    df = pd.read_parquet(OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.date >= SPLIT]
    t = picks(df[df.kind == "high"])
    r = score(t, np.random.default_rng(0))
    n_prom = sum(x["dataset"] == DATASET for x in prior) + 1
    bar = 0.05 / n_prom
    verdict = "PASS" if (r["p"] < bar and r["units"] > 0) else "FAIL"
    with lock.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "wx_expand_X1_new_highs",
                            "sha": sha, "dataset": DATASET, "promotion_number": n_prom, **r, "verdict": verdict}, default=str) + "\n")
    print(f"SEALED X1_new_highs: {df.date.min().date()}..{df.date.max().date()}  bets {r['bets']}  losses {r['losses']}  "
          f"units {r['units']:+.2f}  ROI {r['roi']:+.2%}  95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  "
          f"-> {verdict} (bar {bar:.4f})")
    print("by series/city:", t.groupby("series").pnl.agg(["size", "sum"]).round(2).to_dict("index"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev, "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())
