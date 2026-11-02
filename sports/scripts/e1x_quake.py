"""E1X-Q: Kalshi KXBIGGESTQUAKE (daily max worldwide USGS magnitude) vs USGS-catalog climatology.

Pre-registration: notes/e1x_quake.md (committed before any Kalshi quote was joined to a forecast or outcome).

Stages:
  usgs     -- USGS ComCat M>=4.5 worldwide 2000-01-01..now (csv, yearly chunks) -> data/e1x/usgs.parquet
  clim     -- p_clim(K) = share of UTC days 2000-2025 with daily max M >= K (and per-day max table)
  candles  -- hourly candles for every settled KXBIGGESTQUAKE daily market (no outcomes used)
  build    -- decision rows at 23:00 UTC on D-1 with p_clim, quotes, ETAS guard flag (NO outcomes)
  score    -- trials T1/T2 on --split dev|sealed; ledger rows
  settle   -- descriptive: USGS daily max vs Kalshi expiration_value
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "e1x"
D.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "lab" / "ledger.jsonl"
SERIES = "KXBIGGESTQUAKE"
USGS = "https://earthquake.usgs.gov/fdsnws/event/1/query"
TAKER = 0.07
MARGIN = 0.03
SEED = 20261101
NBOOT = 10_000
DEC_HOUR_BEFORE = 1  # decision = 23:00 UTC on D-1 (one hour before the UTC day starts)


def fee(p):
    return TAKER * p * (1 - p)


# ------------------------------------------------------------------ usgs
def cmd_usgs() -> int:
    parts = []
    for y in range(2000, 2027):
        for h in (("01-01", "07-01"), ("07-01", "01-01")):
            st = f"{y}-{h[0]}"
            en = f"{y + 1 if h[1] == '01-01' else y}-{h[1]}"
            for a in range(8):
                r = requests.get(USGS, params={"format": "csv", "starttime": st, "endtime": en, "minmagnitude": 4.5,
                                               "orderby": "time-asc", "limit": 20000}, timeout=120)
                if r.status_code == 200:
                    break
                time.sleep(5 * (a + 1))
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            parts.append(df[["time", "latitude", "longitude", "mag", "magType", "id", "type"]])
            print(st, en, len(df), flush=True)
            time.sleep(1.0)
    u = pd.concat(parts).drop_duplicates("id")
    u["time"] = pd.to_datetime(u.time, format="ISO8601", utc=True)
    u = u[u.type == "earthquake"]
    u.to_parquet(D / "usgs.parquet")
    print("events", len(u), u.time.min(), u.time.max())
    return 0


def daily_max(u: pd.DataFrame) -> pd.Series:
    day = u.time.dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    return u.groupby(day).mag.max()


KS = [5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4, 6.6, 6.8, 7.0]


def clim_table(u: pd.DataFrame, start="2000-01-01", end="2025-12-31") -> pd.Series:
    dm = daily_max(u)
    days = pd.date_range(start, end, freq="D")
    dm = dm.reindex(days).fillna(0.0)
    # magnitudes are reported to 0.1 (sometimes 0.01); compare after rounding to 2 dp to avoid float edges
    m = dm.round(2).to_numpy()
    return pd.Series({k: float((m >= k - 1e-9).mean()) for k in KS})


def cmd_clim() -> int:
    u = pd.read_parquet(D / "usgs.parquet")
    c = clim_table(u)
    print("p_clim (2000-2025 UTC days):")
    print(c.round(4).to_string())
    # stability: by 5-year block
    for a, b in (("2000", "2004"), ("2005", "2009"), ("2010", "2014"), ("2015", "2019"), ("2020", "2025")):
        print(a, b, clim_table(u, f"{a}-01-01", f"{b}-12-31").round(3).to_dict())
    c.to_json(D / "p_clim.json")
    return 0


# ------------------------------------------------------------------ candles
def cmd_candles() -> int:
    df = pd.read_parquet(D / f"{SERIES}_markets.parquet")
    df = df[df.event_ticker.str.match(rf"{SERIES}-\d\d[A-Z]{{3}}\d\d$")]
    out = {}
    cache = D / "quake_candles.json"
    if cache.exists():
        out = json.loads(cache.read_text())
    for _, m in df.iterrows():
        if m.ticker in out or m.result not in ("yes", "no"):
            continue
        op = int(pd.Timestamp(m.open_time).timestamp())
        cl = int(pd.Timestamp(m.close_time).timestamp())
        p = {"start_ts": op - 3600, "end_ts": cl + 3600, "period_interval": 60}
        if m["_src"] == "/historical/markets":
            r = get(f"/historical/markets/{m.ticker}/candlesticks", p)
        else:
            r = get(f"/series/{SERIES}/markets/{m.ticker}/candlesticks", p)
        out[m.ticker] = r.get("candlesticks", [])
    cache.write_text(json.dumps(out))
    print("markets with candles", sum(1 for v in out.values() if v), "of", len(out))
    return 0


def _px(c: dict, side: str):
    x = c.get(side, {})
    v = x.get("close_dollars", x.get("close"))
    if v is None:
        return np.nan
    v = float(v)
    return v / 100 if v > 1.5 else v


# ------------------------------------------------------------------ build
def cmd_build() -> int:
    df = pd.read_parquet(D / f"{SERIES}_markets.parquet")
    df = df[df.event_ticker.str.match(rf"{SERIES}-\d\d[A-Z]{{3}}\d\d$")].copy()
    df["day"] = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%d%b%y")
    cs = json.loads((D / "quake_candles.json").read_text())
    pc = pd.read_json(D / "p_clim.json", typ="series")
    pc.index = pc.index.astype(float)
    u = pd.read_parquet(D / "usgs.parquet")
    rows = []
    for _, m in df.iterrows():
        if m.ticker not in cs:
            continue
        t = pd.Timestamp(m.day, tz="UTC") - pd.Timedelta(hours=DEC_HOUR_BEFORE)
        ts = int(t.timestamp())
        c = [x for x in cs[m.ticker] if x["end_period_ts"] <= ts]
        yb = _px(c[-1], "yes_bid") if c else np.nan
        ya = _px(c[-1], "yes_ask") if c else np.nan
        prior = u[(u.time > t - pd.Timedelta(hours=72)) & (u.time <= t)]
        rows.append(dict(ticker=m.ticker, event=m.event_ticker, day=m.day, K=float(m.floor_strike),
                         p=float(pc.loc[round(float(m.floor_strike), 1)]), yes_bid=yb, yes_ask=ya,
                         etas_guard=bool((prior.mag >= 7.0).any()), dec_ts=t))
    b = pd.DataFrame(rows)
    b.to_parquet(D / "quake_build.parquet")
    print(len(b), "rows;", b.event.nunique(), "days;", "quote present", b.yes_bid.notna().mean().round(3),
          "guard days", b[b.etas_guard].event.nunique())
    return 0


# ------------------------------------------------------------------ score
def split_days(days: pd.Series):
    u = np.sort(days.unique())
    n_dev = int(round(0.7 * len(u)))
    return set(u[:n_dev]), set(u[n_dev:])


def select(b: pd.DataFrame, trial: str) -> pd.DataFrame:
    x = b[(~b.etas_guard) & b.yes_bid.notna()].copy()
    x["no_ask"] = 1 - x.yes_bid
    x = x[(x.yes_bid >= 0.01) & (x.no_ask <= 0.99)]
    x["edge"] = (1 - x.p) - x.no_ask - fee(x.no_ask)
    x = x[x.edge >= MARGIN]
    if trial == "T2":
        x = x[x.K >= 6.4]
    return x


def boot(x: pd.DataFrame, rng) -> tuple:
    g = x.groupby("event").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
    ev = g.index.to_numpy()
    roi = g.pnl.sum() / g.cost.sum()
    res = np.empty(NBOOT)
    for i in range(NBOOT):
        s = rng.integers(0, len(ev), len(ev))
        res[i] = g.pnl.to_numpy()[s].sum() / g.cost.to_numpy()[s].sum()
    return roi, np.percentile(res, 2.5), np.percentile(res, 97.5), float((res <= 0).mean())


def cmd_score(split: str) -> int:
    b = pd.read_parquet(D / "quake_build.parquet")
    # frozen split (prereg): DEV = event days 2026-09-02..2026-10-14; SEALED = 2026-10-15..2026-11-01
    if split == "dev":
        b = b[(b.day >= "2026-09-02") & (b.day <= "2026-10-14")]
    else:
        b = b[(b.day >= "2026-10-15") & (b.day <= "2026-11-01")]
    res = pd.read_parquet(D / f"{SERIES}_markets.parquet")[["ticker", "event_ticker", "result", "expiration_value"]]
    full = res.groupby("event_ticker").result.apply(lambda r: bool(r.isin(["yes", "no"]).all()))
    n0 = b.event.nunique()
    b = b[b.event.map(full).fillna(False).astype(bool)]
    print(f"{split}: {b.event.nunique()} complete event days used of {n0} in window")
    b = b.merge(res.drop(columns="event_ticker"), on="ticker", how="left")
    assert b.result.isin(["yes", "no"]).all(), b[~b.result.isin(["yes", "no"])][["ticker", "result"]]
    rng = np.random.default_rng(SEED)
    k = 2
    out = []
    for trial in ("T1", "T2"):
        x = select(b, trial)
        if x.empty:
            print(trial, "0 bets")
            continue
        x["cost"] = x.no_ask + fee(x.no_ask)
        x["pnl"] = np.where(x.result == "no", 1.0, 0.0) - x.cost
        roi, lo, hi, p = boot(x, rng)
        dd = np.sort(x.day.unique())
        h1 = x[x.day.isin(dd[: len(dd) // 2])]
        h2 = x[x.day.isin(dd[len(dd) // 2:])]
        r1 = h1.pnl.sum() / h1.cost.sum()
        r2 = h2.pnl.sum() / h2.cost.sum()
        passed = bool(p < 0.05 / k and r1 > 0 and r2 > 0)
        row = dict(trial=trial, split=split, bets=len(x), days=x.event.nunique(), losses=int((x.result == "yes").sum()),
                   roi=round(roi, 4), ci=[round(lo, 4), round(hi, 4)], p=round(p, 4), half1=round(r1, 4),
                   half2=round(r2, 4), mean_no_ask=round(x.no_ask.mean(), 3), mean_p=round(x.p.mean(), 3),
                   pnl_units=round(x.pnl.sum(), 3), passed=passed)
        print(json.dumps(row))
        out.append(row)
        x.to_csv(D / f"quake_bets_{trial}_{split}.csv", index=False)
        if roi > 0.10 and len(x) > 200:
            print("TOO-GOOD GUARD: inspect rows before believing")
    import subprocess
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    with LEDGER.open("a") as f:
        for i, r in enumerate(out, 1):
            f.write(json.dumps(dict(
                ts=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), strategy=f"e1x_quake_{r['trial']}",
                sha=sha, dataset="e1x_quake",
                hypothesis=f"NO on KXBIGGESTQUAKE rung if (1-p_clim)-no_ask-fee >= {MARGIN} at 23:00 UTC D-1"
                           + (", K>=6.4" if r["trial"] == "T2" else "") + "; skip day if M>=7 in prior 72h",
                source="agent E1-pattern search 2026-11-01 (notes/e1x_quake.md)", status="ok",
                split=("DEV 2026-09-02..10-14" if split == "dev" else "SEALED 2026-10-15..11-01"),
                trial_number=i, study_trials=k, **r)) + "\n")
    return 0


def cmd_settle() -> int:
    u = pd.read_parquet(D / "usgs.parquet")
    dm = daily_max(u)
    df = pd.read_parquet(D / f"{SERIES}_markets.parquet")
    df = df[df.event_ticker.str.match(rf"{SERIES}-\d\d[A-Z]{{3}}\d\d$")].copy()
    df["day"] = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%d%b%y")
    df["ev"] = pd.to_numeric(df.expiration_value, errors="coerce")
    e = df.groupby("day").ev.median().to_frame()
    e["usgs"] = dm.reindex(e.index)
    e["diff"] = (e.usgs - e.ev).round(2)
    print(e.to_string())
    print("days with |diff| > 0.1:", int((e["diff"].abs() > 0.1).sum()), "of", int(e.ev.notna().sum()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    fn = {"usgs": cmd_usgs, "clim": cmd_clim, "candles": cmd_candles, "build": cmd_build,
          "settle": cmd_settle}.get(a.stage)
    sys.exit(cmd_score(a.split) if a.stage == "score" else fn())
