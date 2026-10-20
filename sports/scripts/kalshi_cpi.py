"""Kalshi CPI ladders vs the Cleveland Fed inflation nowcast (QUEUE v2 N2). DEV only.

    python scripts/kalshi_cpi.py build
    python scripts/kalshi_cpi.py dev [--ledger]

Markets: KXCPI (headline MoM), KXCPICORE (core MoM), KXCPIYOY (headline YoY), KXCPICOREYOY (core YoY); each event is a
ladder of "above X%" rungs (0.1 pp steps), YES iff the BLS single-decimal first print > X. close_time = 8:25 ET on
release day (KXCPI-26SEP: 2026-10-14T12:29Z).
Nowcast: Cleveland Fed nowcast_month.json / nowcast_year.json (the page's chart data, 2026-10-19): one chart per
target month (from 2013-07), daily published nowcasts + the first official print ("Actual ...").

Pre-registered 2026-10-19, before any Kalshi CPI price was pulled:
  decision : close_time - 48 h (two days before release); last hourly candle ending at or before; tradeable quote
             (yes_bid > 0, yes_ask < 1, spread <= 10c); fee 0.07*p*(1-p).
  nowcast  : latest vintage dated <= decision date - 1 day (published by then).
  band     : sd of (first print - nowcast at the same horizon) per measure, estimated ONLY on target months
             2013-07..2021-07 (before Kalshi CPI markets existed -> no betting data); K = 3 * sd, frozen.
  C1 band  : rung with strike X: if nowcast + K < X + 0.05 (print can't round above X) buy NO at 1 - yes_bid
             (if yes_ask > 0.01); if nowcast - K > X + 0.05 buy YES at the ask (if yes_ask < 0.99).
  split    : DEV = target months 2021-08..2024-12; SEALED = 2025-01+ (never loaded by `dev`; user's yes)
  1 trial -> bar p < 0.05, one-sided bootstrap over EVENTS (rungs of one release are one draw).
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch, fetch_many
from kalshi_longshot import API, _px

OUT = ROOT / "data" / "kalshi_cpi.parquet"
NOW = ROOT / "data" / "cleveland_nowcast.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_cpi"
NC = "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_{}.json"
SERIES = {"KXCPI": ("month", "CPI Inflation"), "KXCPICORE": ("month", "Core CPI Inflation"),
          "KXCPIYOY": ("year", "CPI Inflation"), "KXCPICOREYOY": ("year", "Core CPI Inflation")}
BAND_END = pd.Period("2021-07", "M")
DEV_END = pd.Period("2024-12", "M")
TRIALS = ["C1_band"]


def nowcasts() -> pd.DataFrame:
    rows = []
    for kind in ("month", "year"):
        for ch in fetch(NC.format(kind), cache=False):
            tgt = pd.Period(ch["chart"]["subcaption"], "M")
            labels = [c["label"] for c in ch["categories"][0]["category"]]
            dates = []
            for lab in labels:
                m = re.fullmatch(r"(\d\d)/(\d\d)", lab)
                if not m:
                    dates.append(None)
                    continue
                mo, dy = int(m.group(1)), int(m.group(2))
                yr = tgt.year + (1 if mo < tgt.month else 0)
                dates.append(pd.Timestamp(yr, mo, dy))
            for ds in ch["dataset"]:
                name = ds.get("seriesname", "")
                for dt, pt in zip(dates, ds["data"]):
                    v = pd.to_numeric(pt.get("value"), errors="coerce")
                    if dt is not None and v == v:
                        rows.append({"kind": kind, "target": str(tgt), "date": dt, "series": name, "value": float(v)})
    return pd.DataFrame(rows)


def settled(series: str) -> pd.DataFrame:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.result.isin(["yes", "no"]) & (df.strike_type == "greater")].copy()
    tm = pd.to_datetime(df.event_ticker.str.split("-").str[1].str[:5], format="%y%b", errors="coerce")
    df = df[tm.notna()].copy()
    df["target"] = tm[tm.notna()].dt.to_period("M").astype(str)
    close = pd.to_datetime(df.close_time, utc=True, format="ISO8601")
    df["decision_ts"] = ((close - pd.Timedelta(hours=48) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype(int)
    df["series"] = series
    return df[["series", "ticker", "event_ticker", "target", "floor_strike", "result", "expiration_value", "decision_ts"]]


def prices(series: str, mk: pd.DataFrame) -> pd.DataFrame:
    ev = mk.groupby("event_ticker").decision_ts.min()
    urls = [f"{API}/series/{series}/events/{e}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}&period_interval=60" for e, t in ev.items()]
    got, missing = {}, []
    for (e, t), d in zip(ev.items(), fetch_many(urls)):
        tks = d.get("market_tickers", [])
        if not tks:
            missing.append(e)
        for tk, cs in zip(tks, d.get("market_candlesticks", [])):
            got[tk] = _px(cs, t)
    leg = mk[mk.event_ticker.isin(missing)]
    urls = [f"{API}/historical/markets/{tk}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}&period_interval=60"
            for tk, t in zip(leg.ticker, leg.decision_ts)]
    for (tk, t), d in zip(zip(leg.ticker, leg.decision_ts), fetch_many(urls)):
        got[tk] = _px(d.get("candlesticks", []), t)
    out = mk.copy()
    out["yes_bid"] = out.ticker.map(lambda k: got.get(k, (np.nan, np.nan))[0])
    out["yes_ask"] = out.ticker.map(lambda k: got.get(k, (np.nan, np.nan))[1])
    return out


def cmd_build() -> int:
    nc = nowcasts()
    nc.to_parquet(NOW, index=False)
    print(f"nowcasts {len(nc)} rows, targets {nc.target.min()}..{nc.target.max()}")
    frames = []
    for s, (kind, name) in SERIES.items():
        mk = settled(s)
        px = prices(s, mk)
        g = nc[(nc.kind == kind) & (nc.series == name)]
        dec = pd.to_datetime(px.decision_ts, unit="s").dt.normalize()
        px["nowcast"] = [g[(g.target == t) & (g.date <= d - pd.Timedelta(days=1))].sort_values("date").value.iloc[-1]
                         if len(g[(g.target == t) & (g.date <= d - pd.Timedelta(days=1))]) else np.nan
                         for t, d in zip(px.target, dec)]
        print(f"{s}: {len(px)} rungs, {px.event_ticker.nunique()} events {px.target.min()}..{px.target.max()}, "
              f"priced {px.yes_ask.notna().mean():.2f}, nowcast {px.nowcast.notna().mean():.2f}", flush=True)
        frames.append(px)
    pd.concat(frames, ignore_index=True).to_parquet(OUT, index=False)
    return 0


def band_sd(nc: pd.DataFrame) -> dict:
    """sd of (first print - nowcast ~2 days before the print) on targets <= 2021-07 only."""
    out = {}
    for s, (kind, name) in SERIES.items():
        g = nc[(nc.kind == kind)]
        errs = []
        for t, gt in g.groupby("target"):
            if pd.Period(t, "M") > BAND_END:
                continue
            act = gt[gt.series == "Actual " + name]
            fc = gt[gt.series == name].sort_values("date")
            if act.empty or fc.empty:
                continue
            rel = act.date.min()
            f = fc[fc.date <= rel - pd.Timedelta(days=3)]
            if len(f):
                errs.append(round(act.value.iloc[0], 1) - f.value.iloc[-1])
        out[s] = (float(np.std(errs, ddof=1)), len(errs))
    return out


def trial(df: pd.DataFrame, K: dict) -> pd.DataFrame:
    d = df[df.yes_bid.notna() & (df.yes_bid > 0) & (df.yes_ask < 1) & ((df.yes_ask - df.yes_bid) <= 0.10) & df.nowcast.notna()].copy()
    k = d.series.map(K)
    no = (d.nowcast + k < d.floor_strike + 0.05) & (d.yes_ask > 0.01)
    yes = (d.nowcast - k > d.floor_strike + 0.05) & (d.yes_ask < 0.99)
    rows = []
    for side, sel in (("no", no), ("yes", yes)):
        s = d[sel].copy()
        c = (1 - s.yes_bid) if side == "no" else s.yes_ask
        cost = c + 0.07 * c * (1 - c)
        won = (s.result == side)
        s["pnl"] = np.where(won, 1 / cost - 1, -1.0)
        s["side"] = side
        rows.append(s)
    return pd.concat(rows)


def cmd_dev() -> int:
    nc = pd.read_parquet(NOW)
    sd = band_sd(nc)
    K = {s: 3 * v for s, (v, n) in sd.items()}
    print("band (targets 2013-07..2021-07):", {s: f"sd {v:.3f} n {n} -> K {3 * v:.3f}" for s, (v, n) in sd.items()})
    df = pd.read_parquet(OUT)
    df = df[df.target.map(lambda t: pd.Period(t, "M") <= DEV_END)]
    print(f"DEV {df.event_ticker.nunique()} events, {len(df)} rungs, targets {df.target.min()}..{df.target.max()}")
    t = trial(df, K)
    ev = t.groupby("event_ticker").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)]) if len(ev) else np.zeros(1)
    r = {"bets": int(len(t)), "events": int(len(ev)), "staked": float(len(t)), "units": float(t.pnl.sum()),
         "roi": float(t.pnl.mean()) if len(t) else 0.0, "ci_lo": float(np.quantile(boots, .025)),
         "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)), "losses": int((t.pnl < 0).sum())}
    print(f"C1_band bets {r['bets']} over {r['events']} events, losses {r['losses']}, units {r['units']:+.2f}, ROI {r['roi']:+.2%}, "
          f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], p {r['p']:.4f}")
    print(t.groupby(["series", "side"]).pnl.agg(["size", "sum", "mean"]).round(3).to_string())
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        shas = {x["sha"] for x in prior if x.get("status") == "ok" and x["dataset"] == DATASET}
        sha = hashlib.sha256(b"kalshi_cpi:C1_band").hexdigest()[:12]
        shas.add(sha)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "kalshi_cpi_C1_band",
                                "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in scripts/kalshi_cpi.py (C1_band)",
                                "source": "Cleveland Fed nowcast band", "status": "ok", "trial_number": len(shas), **r,
                                "K": K}, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
