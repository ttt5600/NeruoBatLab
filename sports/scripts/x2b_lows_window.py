"""X2b: consensus-reject on Kalshi daily-LOW series with the forecast min over the FULL settlement window.

    python scripts/x2b_lows_window.py dev [--ledger]
    python scripts/x2b_lows_window.py sealed [--ledger]     # once, only after a dev PASS is in the ledger

Pre-registered in notes/x2_lows_window.md (commit d54f8dd) before any X2b number existed. Single change vs X2
(scripts/wx_expand.py, rule frozen there and not modified): per model, forecast min over hourly values at
LST 00:00..24:00 inclusive of the target day (25 values, >= 20 required) instead of local hours <= 18:00.
Leak control: valid times <= decision (22:00 local day before) + 20 h use temperature_2m_previous_day1 (as the
frozen rule); later valid times use temperature_2m_previous_day2. Everything else = wx_expand.picks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import wx_expand as X
from sports import wx
from sports.http import fetch

LOCK = ROOT / "lab" / "lockbox.jsonl"
OUTD = ROOT / "results" / "x2b_lows_window"
NAME = "wx_expand_X2b_lows_window"
HALF = {"dev": pd.Timestamp("2026-01-21"), "sealed": pd.Timestamp("2026-06-24")}
SEALED_END = pd.Timestamp("2026-10-18")
NBOOT = 4000


class _BuildOffset(dict):
    def __missing__(self, tz):
        return datetime(2026, 10, 19, 12, tzinfo=ZoneInfo(tz)).utcoffset()


BUILD_OFFSET = _BuildOffset()


def std_offset(tz: str, day: pd.Timestamp) -> timedelta:
    dt = datetime(day.year, day.month, day.day, 12, tzinfo=ZoneInfo(tz))
    return dt.utcoffset() - (dt.dst() or timedelta(0))


def hourly(model, lat, lon, start, end) -> pd.DataFrame:
    frames, s = [], pd.Timestamp(start)
    while s <= pd.Timestamp(end):
        e = min(s + pd.Timedelta(days=364), pd.Timestamp(end))
        done = e < pd.Timestamp.today().normalize() - pd.Timedelta(days=3)
        d = fetch(wx.PREV_RUNS, {"latitude": lat, "longitude": lon, "models": model,
                                 "hourly": "temperature_2m_previous_day1,temperature_2m_previous_day2",
                                 "temperature_unit": "fahrenheit", "timezone": "GMT",
                                 "start_date": s.date().isoformat(), "end_date": e.date().isoformat()}, cache=done)
        h = pd.DataFrame(d.get("hourly", {}))
        if not h.empty:
            frames.append(h)
        s = e + pd.Timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    h = pd.concat(frames).drop_duplicates("time")
    h["t"] = pd.to_datetime(h["time"]).dt.tz_localize("UTC")
    return h.set_index("t").sort_index()


def window_min(h: pd.DataFrame, dates, tz: str) -> pd.DataFrame:
    """-> per date: new full-LST-window min, and the old local<=18 day1 min (alignment check)."""
    z, rows = ZoneInfo(tz), []
    d1, d2 = h["temperature_2m_previous_day1"], h["temperature_2m_previous_day2"]
    for day in dates:
        lst0 = pd.Timestamp(day).tz_localize("UTC") - std_offset(tz, day)
        win = h.index[(h.index >= lst0) & (h.index <= lst0 + pd.Timedelta(hours=24))]
        dec = pd.Timestamp(datetime(day.year, day.month, day.day, wx.DECISION_HOUR, tzinfo=z) - timedelta(days=1)).tz_convert("UTC")
        use1 = win <= dec + pd.Timedelta(hours=20)
        v = pd.concat([d1.loc[win[use1]], d2.loc[win[~use1]]])
        new = v.min() if v.notna().sum() >= 20 else np.nan
        # Emulate the cached build exactly: Open-Meteo labels a whole response with ONE utc_offset (the offset in
        # force on the request date; wx_expand build ran 2026-10-19), not the DST-correct offset per date.
        loc = (h.index + BUILD_OFFSET[tz]).tz_localize(None)
        old_m = (loc.normalize() == pd.Timestamp(day)) & (loc.hour <= 18)
        o = d1[old_m]
        old = o.min() if o.notna().sum() >= 15 else np.nan
        rows.append({"date": pd.Timestamp(day), "new": new, "old": old, "n_day2": int((~use1).sum())})
    return pd.DataFrame(rows).set_index("date")


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for st, g in df.groupby("station"):
        p = fetch(f"https://api.weather.gov/stations/{st}")
        lon, lat = p["geometry"]["coordinates"][:2]
        tz = p["properties"]["timeZone"]
        dates = sorted(g.date.unique())
        g = g.copy()
        for m, start in X.MODELS.items():
            lo = max(pd.Timestamp(start), pd.Timestamp(dates[0]))
            ds = [d for d in dates if pd.Timestamp(d) >= lo]
            col = f"fcsafe_{m}"
            if not ds:
                g[col] = np.nan
                continue
            h = hourly(m, lat, lon, ds[0] - pd.Timedelta(days=1), ds[-1] + pd.Timedelta(days=1))
            if h.empty:
                g[col] = np.nan
                continue
            w = window_min(h, [pd.Timestamp(d) for d in ds], tz)
            old_cached = g[col].copy()
            chk = pd.DataFrame({"cached": old_cached.values, "recomp": g.date.map(w["old"]).values}).dropna()
            bad = (chk.cached - chk.recomp).abs() > 0.05
            print(f"  {st} {m}: alignment old-window recompute vs cached n={len(chk)} mismatches={int(bad.sum())}", flush=True)
            if bad.mean() > 0.01:
                raise SystemExit(f"alignment check failed for {st} {m}")
            g[col] = g.date.map(w["new"]).values
            g[f"old_{m}"] = old_cached.values
        g["tz"] = tz
        out.append(g)
    return pd.concat(out, ignore_index=True)


def score(t: pd.DataFrame, split: pd.Timestamp) -> dict:
    rng = np.random.default_rng(0)
    t = t.assign(day=t.series + t.date.astype(str))
    g = t.groupby("day").pnl.agg(["sum", "size"])
    s, n = g["sum"].to_numpy(), g["size"].to_numpy()
    idx = rng.integers(0, len(g), (NBOOT, len(g)))
    bu, bn = s[idx].sum(1), n[idx].sum(1)
    cost = (1 - t.yes_bid) + wx.FEE_RATE * (1 - t.yes_bid) * t.yes_bid
    h1, h2 = t[t.date < split], t[t.date >= split]
    months = max(1.0, (t.date.max() - t.date.min()).days / 30.4)
    return {"bets": int(len(t)), "staked": float(len(t)), "units": float(t.pnl.sum()), "roi": float(t.pnl.mean()),
            "roi_lo": float(np.quantile(bu / bn, .025)), "roi_hi": float(np.quantile(bu / bn, .975)),
            "ci_lo": float(np.quantile(bu, .025)), "ci_hi": float(np.quantile(bu, .975)), "p": float(np.mean(bu <= 0)),
            "half1_units": float(h1.pnl.sum()), "half1_bets": int(len(h1)), "half2_units": float(h2.pnl.sum()),
            "half2_bets": int(len(h2)), "losses": int((t.pnl < 0).sum()), "loss_rate": float((t.pnl < 0).mean()),
            "breakeven_loss_rate": float(1 - cost.mean()), "avg_no": float((1 - t.yes_bid).mean()),
            "picks_per_month": float(len(t) / months), "station_days": int(len(g))}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("dev", "sealed"):
        print(__doc__)
        return 1
    led = [json.loads(l) for l in X.LEDGER.read_text().splitlines() if l.strip()]
    sha = hashlib.sha256(f"x2b_lows_window:{mode}".encode()).hexdigest()[:12]
    if mode == "sealed":
        dev = [r for r in led if r.get("strategy") == NAME and r.get("split") == "dev"]
        if not dev or dev[-1].get("verdict") != "PASS":
            print("sealed is gated on a dev PASS in the ledger; refusing")
            return 1
        lock = [json.loads(l) for l in LOCK.read_text().splitlines() if l.strip()]
        if any(r["sha"] == sha for r in lock):
            print("X2b sealed already scored; refusing")
            return 1
    df = pd.read_parquet(X.OUT)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df.kind == "low"]
    df = df[df.date < X.SPLIT] if mode == "dev" else df[(df.date >= X.SPLIT) & (df.date <= SEALED_END)]
    print(f"{mode.upper()}: {df.series.nunique()} series, {df.event_ticker.nunique()} station-days, "
          f"{df.date.min().date()}..{df.date.max().date()}")
    d = build(df)
    OUTD.mkdir(parents=True, exist_ok=True)
    t = X.picks(d)
    cols = [c for c in t.columns if c.startswith("fcsafe_")]
    hi = t[cols].max(axis=1)
    t["side"] = np.where(t.strike_type == "greater", "above", np.where(t.strike_type == "less", "below",
                         np.where(hi + X.MARGIN < t.floor_strike, "above", "below")))
    t.drop(columns=[c for c in t.columns if c.startswith("old_")]).to_csv(OUTD / f"bets_{mode}.csv", index=False)
    r = score(t, HALF[mode])
    if mode == "dev":
        ok = r["p"] < 0.05 and r["half1_units"] > 0 and r["half2_units"] > 0
    else:
        ok = r["p"] < 0.05 and r["roi"] > 0 and r["half1_units"] > 0 and r["half2_units"] > 0
    r["verdict"] = "PASS" if ok else "FAIL"
    if r["roi"] > 0.10 and r["bets"] > 200:
        print("TOO-GOOD GUARD TRIPPED: inspect rows before reporting")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, indent=1))
    print("by station:", t.groupby("station").pnl.agg(["size", "sum"]).round(2).to_dict("index"))
    print("by side:", t.groupby("side").pnl.agg(["size", "sum"]).round(2).to_dict("index"))
    print("by month:", t.groupby(t.date.dt.strftime("%Y-%m")).pnl.agg(["size", "sum"]).round(2).to_dict("index"))
    # overlap with the X2 (old-window) picks on the same rows
    o = d.copy()
    for m in X.MODELS:
        if f"old_{m}" in o:
            o[f"fcsafe_{m}"] = o[f"old_{m}"]
    t0 = X.picks(o)
    both = set(t.ticker) & set(t0.ticker)
    print(f"X2 old-window picks on same rows: {len(t0)} (units {t0.pnl.sum():+.2f}); kept {len(both)}, "
          f"dropped {len(t0) - len(both)} (their units {t0[~t0.ticker.isin(both)].pnl.sum():+.2f}), new {len(t) - len(both)}")
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": NAME + ("_SEALED" if mode == "sealed" else ""),
           "sha": sha, "dataset": X.DATASET, "split": mode, "window": f"{df.date.min().date()}..{df.date.max().date()}",
           "hypothesis": "pre-registered in notes/x2_lows_window.md (commit d54f8dd): X2 rule with forecast min over full "
                         "LST 00-24 settlement window (day1 to decision+20h, day2 after)",
           "source": "diagnosis of X2 dev losses", "status": "ok", **r}
    if "--ledger" in sys.argv:
        with X.LEDGER.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if mode == "sealed":
            with LOCK.open("a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        print("appended to ledger" + (" + lockbox" if mode == "sealed" else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
