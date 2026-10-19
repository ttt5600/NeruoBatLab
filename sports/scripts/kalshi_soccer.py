"""Kalshi SOCCER 1X2 game markets: are Kalshi's pre-kickoff soccer prices calibrated? DEV only.

    python scripts/kalshi_soccer.py build            # settled markets + hourly price ~2 h before kickoff (cached)
    python scripts/kalshi_soccer.py dev [--ledger]   # calibration table + pre-registered trials, DEV only

Why: football-data (soccer_fd.py) shows soft books lag Pinnacle; Kalshi is the one US-legal venue we can
automate, and its soccer game markets trade $1-6M per EPL outcome (KXEPLGAME-26OCT18NFOARS).
Each match = 3 binary markets (home wins / Tie / away wins), 90 min + stoppage.

Decision time (leak check 2026-10-19): occurrence_datetime / expected_expiration_time are NOT kickoff
(NFO-ARS: occurrence 18:30Z, market closed 17:25Z at full time -> kickoff ~15:30Z). close_time = when the
result was declared (can_close_early), so we price at close_time - 4 h <= kickoff - ~2 h (a match cannot end
sooner than ~1 h 50 after kickoff). Events whose close_time is not before expected_expiration_time did not
close at full time -> dropped (close would not anchor kickoff). Quote must be two-sided, spread <= 5c.
Costs: taker fee 0.07*p*(1-p) per contract.

Pre-registered 2026-10-19, before any price was fetched:
  split : DEV = events dated < 2026-03-01; SEALED = 2026-03-01 onward (never loaded by `dev`)
  S1 draw_yes   : buy YES on every Tie market at the ask (recreational money under-bets draws)
  S2 fav60_yes  : buy YES on the favourite (highest mid of the 3) when its mid >= 0.60, at the ask
  S3 tail_no    : buy NO on any outcome with YES ask <= 0.12, at 1 - yes_bid (favourite-longshot bias)
  3 trials -> bar p < 0.0167, one-sided bootstrap over EVENTS. Sealed pass = p < 0.05/#promoted and units > 0.
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

from sports.http import fetch
from kalshi_longshot import API, LEDGER, ok_quote, pnl, prices, score

OUT = ROOT / "data" / "kalshi_soccer_prices.parquet"
DATASET = "kalshi_soccer"
SERIES = ["KXEPLGAME", "KXLALIGAGAME", "KXSERIEAGAME", "KXBUNDESLIGAGAME", "KXLIGUE1GAME", "KXEFLCHAMPIONSHIPGAME",
          "KXEREDIVISIEGAME", "KXUCLGAME", "KXMLSGAME", "KXBUNDESLIGA2GAME", "KXLALIGA2GAME", "KXSCOTTISHPREMGAME",
          "KXEFLL1GAME", "KXLIGUE2GAME", "KXBRASILEIROGAME"]
SPLIT = pd.Timestamp("2026-03-01")
TRIALS = ["S1_draw_yes", "S2_fav60_yes", "S3_tail_no"]
TS0 = pd.Timestamp("1970-01-01", tz="UTC")


def settled(series: str) -> pd.DataFrame:
    rows = []
    for path, extra, cache in ((f"{API}/historical/markets", {}, True), (f"{API}/markets", {"status": "settled"}, False)):
        cursor = None
        while True:
            p = {"series_ticker": series, "limit": 1000, **extra}
            if cursor:
                p["cursor"] = cursor
            d = fetch(path, p, cache=cache)
            rows += d.get("markets", [])
            cursor = d.get("cursor")
            if not cursor or not d.get("markets"):
                break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates("ticker")
    df = df[df["result"].isin(["yes", "no"]) & df["close_time"].notna() & df["expected_expiration_time"].notna()].copy()
    close = pd.to_datetime(df["close_time"], utc=True, format="ISO8601")
    exp = pd.to_datetime(df["expected_expiration_time"], utc=True, format="ISO8601")
    df["closed_at_ft"] = close < exp
    df["series"] = series
    df["decision_ts"] = ((close - pd.Timedelta(hours=4) - TS0) // pd.Timedelta(seconds=1)).astype(int)
    df["event_date"] = pd.to_datetime(df["event_ticker"].str.split("-").str[1].str[:7], format="%y%b%d", errors="coerce")
    home = df["rules_primary"].str.extract(r"wins the (.+?) vs ", expand=False)
    df["side"] = np.where(df["ticker"].str.endswith("-TIE"), "D", np.where(df["yes_sub_title"] == home, "H", "A"))
    keep = ["series", "ticker", "event_ticker", "event_date", "decision_ts", "closed_at_ft", "result", "side", "yes_sub_title"]
    return df[keep].dropna(subset=["event_date"])


def cmd_build() -> int:
    frames = []
    for s in SERIES:
        mk = settled(s)
        if mk.empty:
            print(f"{s}: no settled markets")
            continue
        n_all = mk.event_ticker.nunique()
        bad = mk.groupby("event_ticker").closed_at_ft.transform("min") == False  # noqa: E712
        mk = mk[~bad]
        n3 = mk.groupby("event_ticker").ticker.transform("size")
        mk = mk[n3 == 3]
        px = prices(s, mk)
        print(f"{s}: {n_all} events, {mk.event_ticker.nunique()} closed at FT with 3 markets, priced "
              f"{px.yes_ask.notna().sum()}/{len(px)}", flush=True)
        frames.append(px)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df)} rows)")
    return 0


def trial_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    d = df[ok_quote(df)].copy()
    d["mid"] = (d.yes_bid + d.yes_ask) / 2
    if name == "S1_draw_yes":
        s = d[d.side == "D"]
        cost, won = s.yes_ask.to_numpy(), (s.result == "yes").to_numpy()
    elif name == "S2_fav60_yes":
        fav = d.groupby("event_ticker")["mid"].transform("max") == d["mid"]
        s = d[fav & (d.mid >= 0.60)].drop_duplicates("event_ticker")
        cost, won = s.yes_ask.to_numpy(), (s.result == "yes").to_numpy()
    else:
        s = d[d.yes_ask <= 0.12]
        cost, won = 1 - s.yes_bid.to_numpy(), (s.result == "no").to_numpy()
    p, c = pnl(cost, won)
    return pd.DataFrame({"event": s.event_ticker.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                         "pnl": p, "stake": c})


def calibration(df: pd.DataFrame):
    d = df[ok_quote(df)].copy()
    d["mid"] = (d.yes_bid + d.yes_ask) / 2
    d["b"] = pd.cut(d.mid, [0, .05, .12, .20, .30, .40, .50, .60, .75, .90, 1.0])
    d["y"] = (d.result == "yes").astype(float)
    print(f"{'side':4s} {'bucket':12s} {'n':>6s} {'mid':>6s} {'hit':>6s} {'hit-mid':>8s}")
    for (k, b), g in d.groupby(["side", "b"], observed=True):
        print(f"{k:4s} {str(b):12s} {len(g):6d} {g.mid.mean():6.3f} {g.y.mean():6.3f} {g.y.mean() - g.mid.mean():+8.3f}")
    s = d.groupby("event_ticker").agg(n=("mid", "size"), sm=("mid", "sum"), y=("y", "sum"))
    full = s[s.n == 3]
    print(f"events with 3 good quotes: {len(full)}; sum of mids median {full.sm.median():.3f}; "
          f"exactly one winner {np.mean(full.y == 1):.3f}")
    print(f"Brier of mids {np.mean((d.mid - d.y) ** 2):.4f}  (leak check: a pre-kickoff soccer 1X2 should sit ~0.18-0.20)")


def cmd_dev() -> int:
    df = pd.read_parquet(OUT)
    df = df[df.event_date < SPLIT]  # sealed rows never enter dev
    print(f"DEV rows {len(df)} ({df.event_ticker.nunique()} events), {df.event_date.min():%Y-%m-%d}..{df.event_date.max():%Y-%m-%d}\n")
    calibration(df)
    prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == DATASET}
    recs = []
    print()
    for name in TRIALS:
        t = trial_frame(df, name)
        res = score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("kalshi_soccer:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        by = t.groupby("series").apply(lambda g: f"{g.pnl.sum() / g.stake.sum():+.1%}/{len(g)}", include_groups=False)
        print(f"{name:13s} bets {res['bets']:5d} events {res['events']:5d}  units {res['units']:+7.1f}  ROI {res['roi']:+6.2%}  "
              f"95%CI [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}]  p {res['p']:.4f}  quarters+ {res['folds_positive']}/{len(res['per_fold'])}")
        print(f"              by series: {dict(by)}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_soccer_{name}",
                     "sha": sha, "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_soccer.py ({name})",
                     "source": "draw bias / favourite-longshot bias", "status": "ok", "trial_number": len(shas), **res})
    print(f"\n{DATASET} trials: {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"appended {len(recs)} trials to ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
