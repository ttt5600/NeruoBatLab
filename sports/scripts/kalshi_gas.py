"""Kalshi DAILY US gas price (KXAAAGASD, AAA national average) vs a random-walk band (QUEUE v2 N6). DEV only.

    python scripts/kalshi_gas.py build
    python scripts/kalshi_gas.py dev [--ledger]

Contract: event for day D ("KXAAAGASD-26OCT19") trades on D-1 (open 12:00Z D-1, close 03:59Z D = 23:59 ET D-1),
rungs "above X" every $0.005, YES iff AAA's US regular average for D is strictly > X. Settlement value =
expiration_value (AAA, published early on D). AAA's value for D-1 is public during D-1.

Pre-registered 2026-10-19, before any KXAAAGASD price was pulled:
  history  : AAA daily series rebuilt from Kalshi's own settled expiration_values (one per event day).
  decision : 20:00 America/New_York on D-1 (00:00-01:00Z D; < close); last hourly candle ending at or before;
             tradeable quote (yes_bid > 0, yes_ask < 1, spread <= 10c); fee 0.07*p*(1-p).
  known    : AAA(D-1) = settled value of the D-1 event (published on D-1, before the decision).
  band     : K = 3 * sd of daily changes AAA(t) - AAA(t-1) over the 60 calendar days ending D-2 (strictly past;
             >= 30 changes required); if AAA(D-1) is missing, no bet.
  G1 band  : rung X: AAA(D-1) + K < X -> buy NO at 1 - yes_bid (if yes_ask > 0.01);
             AAA(D-1) - K > X -> buy YES at the ask (if yes_ask < 0.99).
  split    : DEV = event days < 2025-07-01; SEALED = 2025-07-01+ (user's yes)
  1 trial -> bar p < 0.05, one-sided bootstrap over event days.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from kalshi_cpi import prices, settled as _settled_unused  # noqa: F401  (prices() is generic over series)
from sports.http import fetch
from kalshi_longshot import API

SERIES = "KXAAAGASD"
OUT = ROOT / "data" / "kalshi_gas.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
DATASET = "kalshi_gas"
SPLIT = pd.Timestamp("2025-07-01")


def settled() -> pd.DataFrame:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cur = None
        while True:
            p = {"series_ticker": SERIES, "limit": 1000, **extra}
            if cur:
                p["cursor"] = cur
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cur = d.get("cursor")
            if not cur or not d.get("markets"):
                break
    df = pd.DataFrame(rows).drop_duplicates("ticker")
    df = df[df.result.isin(["yes", "no"]) & (df.strike_type == "greater")].copy()
    df["day"] = pd.to_datetime(df.event_ticker.str.split("-").str[1], format="%y%b%d", errors="coerce")
    df = df[df.day.notna()].copy()
    df["value"] = pd.to_numeric(df.expiration_value, errors="coerce")
    df["value"] = df.groupby("event_ticker").value.transform("max")
    et = ZoneInfo("America/New_York")
    df["decision_ts"] = [int(datetime(d.year, d.month, d.day, 20, tzinfo=et).timestamp()) - 86400 for d in df.day]
    df["series"] = SERIES
    return df[["series", "ticker", "event_ticker", "day", "floor_strike", "result", "value", "decision_ts"]]


def cmd_build() -> int:
    mk = settled()
    px = prices(SERIES, mk)
    aaa = mk.groupby("day").value.first().sort_index()
    px["prev"] = px.day.map(lambda d: aaa.get(d - pd.Timedelta(days=1), np.nan))
    ch = aaa.diff()[aaa.index.to_series().diff() == pd.Timedelta(days=1)]
    def k_for(d):
        w = ch[(ch.index <= d - pd.Timedelta(days=2)) & (ch.index > d - pd.Timedelta(days=62))]
        return 3 * float(w.std(ddof=1)) if len(w) >= 30 else np.nan
    ks = {d: k_for(d) for d in px.day.unique()}
    px["K"] = px.day.map(ks)
    px.to_parquet(OUT, index=False)
    print(f"{len(px)} rungs, {px.event_ticker.nunique()} days {px.day.min().date()}..{px.day.max().date()}; priced "
          f"{px.yes_ask.notna().mean():.2f}; prev known {px.prev.notna().mean():.2f}; K known {px.K.notna().mean():.2f}; "
          f"median K ${px.K.median():.4f}; median |daily change| ${ch.abs().median():.4f}")
    return 0


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df = df[df.day < SPLIT]
    d = df[df.yes_bid.notna() & (df.yes_bid > 0) & (df.yes_ask < 1) & ((df.yes_ask - df.yes_bid) <= 0.10) & df.prev.notna() & df.K.notna()]
    print(f"DEV {df.event_ticker.nunique()} days, {len(df)} rungs; tradeable with band {len(d)}")
    out = []
    for side, sel in (("no", (d.prev + d.K < d.floor_strike) & (d.yes_ask > 0.01)), ("yes", (d.prev - d.K > d.floor_strike) & (d.yes_ask < 0.99))):
        s = d[sel].copy()
        c = (1 - s.yes_bid) if side == "no" else s.yes_ask
        cost = c + 0.07 * c * (1 - c)
        s["pnl"] = np.where(s.result == side, 1 / cost - 1, -1.0)
        s["side"] = side
        out.append(s)
    t = pd.concat(out)
    ev = t.groupby("event_ticker").pnl.sum().to_numpy()
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(ev, len(ev)).sum() for _ in range(3000)]) if len(ev) else np.zeros(1)
    r = {"bets": int(len(t)), "events": int(len(ev)), "staked": float(len(t)), "units": float(t.pnl.sum()),
         "roi": float(t.pnl.mean()) if len(t) else 0.0, "ci_lo": float(np.quantile(boots, .025)),
         "ci_hi": float(np.quantile(boots, .975)), "p": float(np.mean(boots <= 0)), "losses": int((t.pnl < 0).sum())}
    print(f"G1_band bets {r['bets']} over {r['events']} days, losses {r['losses']}, units {r['units']:+.2f}, ROI {r['roi']:+.2%}, "
          f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}], p {r['p']:.4f}")
    if len(t):
        print(t.groupby("side").agg(n=("pnl", "size"), units=("pnl", "sum"), avg_cost=("yes_ask", "mean")).round(3).to_string())
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
        shas = {x["sha"] for x in prior if x.get("status") == "ok" and x["dataset"] == DATASET}
        sha = hashlib.sha256(b"kalshi_gas:G1_band").hexdigest()[:12]
        shas.add(sha)
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "kalshi_gas_G1_band",
                                "sha": sha, "dataset": DATASET, "hypothesis": "pre-registered in scripts/kalshi_gas.py (G1_band)",
                                "source": "random-walk band on AAA daily", "status": "ok", "trial_number": len(shas), **r}, default=str) + "\n")
        print("appended to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
