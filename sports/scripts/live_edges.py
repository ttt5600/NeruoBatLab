"""Live edge scanner: Kalshi game contracts vs Pinnacle's no-vig price. Paper only.

    python scripts/live_edges.py scan                 # one pass, all leagues; prints edges, logs snapshot
    python scripts/live_edges.py loop --every 10      # scan every N minutes (keep >= 5)
    python scripts/live_edges.py report               # how did the flagged bets do? CLV + settled P&L

Every scan writes the full matched table to data/live/kalshi/<ts>.parquet, so the
closing price of every contract is on disk. A FLAG is the first time a contract
side clears EV_MIN (conservative de-vig, after Kalshi's fee) before the game starts;
it is a paper bet of 1 contract at the ask seen then.

How we will know the edge is real, in order of speed:
  1. CLV  -- did Pinnacle's fair price at its LAST pre-game snapshot move toward us?
             (positive mean CLV over a few hundred flags = real edge, per Buchdahl)
  2. P&L  -- settled Kalshi results; slow and noisy, but it is the money.
No real orders are placed by this script.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports import live

OUT = ROOT / "data" / "live"
SNAP = OUT / "kalshi"
FLAGS = OUT / "flags.jsonl"
EV_MIN = 0.02
MIN_LEAD_MIN = 10   # ignore games starting within 10 minutes (stale-quote risk)
MAX_PRICE, MIN_PRICE = 0.97, 0.03


def scan(verbose: bool = True) -> pd.DataFrame:
    ts = live.now()
    frames = []
    for lg in live.LEAGUES:
        try:
            p, k = live.pinnacle(lg), live.kalshi(lg)
        except Exception as e:  # one league down must not stop the others
            print(f"  {lg}: fetch failed ({e.__class__.__name__}: {str(e)[:80]})")
            continue
        m = live.kalshi_edges(live.match_kalshi(p, k))
        if verbose:
            print(f"  {lg}: pinnacle {p['pin_id'].nunique() if len(p) else 0} games, "
                  f"kalshi {k['event'].nunique() if len(k) else 0} events, matched {m['event'].nunique() if len(m) else 0}")
        if len(m):
            frames.append(m.assign(league=lg))
    if not frames:
        return pd.DataFrame()
    m = pd.concat(frames, ignore_index=True).assign(ts=ts)
    SNAP.mkdir(parents=True, exist_ok=True)
    m.to_parquet(SNAP / f"{ts:%Y%m%dT%H%M%S}.parquet", index=False)
    pre = m[(m["start"] - ts).dt.total_seconds() > MIN_LEAD_MIN * 60]
    flagged = set()
    if FLAGS.exists():
        flagged = {(f["ticker"], f["side"]) for f in map(json.loads, FLAGS.read_text().splitlines())}
    new = []
    for side in ("yes", "no"):
        px = pre[f"{side}_ask"]
        hit = pre[(pre[f"ev_{side}"] >= EV_MIN) & px.between(MIN_PRICE, MAX_PRICE)]
        for _, r in hit.iterrows():
            if (r["ticker"], side) in flagged:
                continue
            rec = {"ts": ts.isoformat(), "league": r["league"], "ticker": r["ticker"], "side": side,
                   "team": r["team"], "home": r["home"], "away": r["away"], "start": r["start"].isoformat(),
                   "ask": float(r[f"{side}_ask"]), "fee": float(r[f"fee_{side}"]), "ev": float(r[f"ev_{side}"]),
                   "fair": float(r["fair"]), "pfair": float(r["pfair"]), "pin_price": float(r["pin_price"])}
            new.append(rec)
    if new:
        with FLAGS.open("a") as fh:
            for rec in new:
                fh.write(json.dumps(rec) + "\n")
    if verbose:
        best = pre.assign(ev=pre[["ev_yes", "ev_no"]].max(axis=1)).sort_values("ev", ascending=False)
        cols = ["league", "team", "home", "away", "start", "fair", "pfair", "yes_bid", "yes_ask", "ev_yes", "ev_no"]
        print(f"\n{ts:%Y-%m-%d %H:%M} UTC  top prices vs Pinnacle no-vig (EV after fee, conservative):")
        with pd.option_context("display.width", 220):
            print(best[cols].head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\nnew flags (EV >= {EV_MIN:.0%}): {len(new)}")
        for r in new:
            print(f"  {r['league']} {r['ticker']} {r['side'].upper()} @ {r['ask']:.2f}  EV {r['ev']:+.1%}  "
                  f"({r['away']} at {r['home']}, {r['start'][:16]})")
    return m


def report() -> None:
    if not FLAGS.exists():
        print("no flags yet")
        return
    fl = pd.DataFrame(map(json.loads, FLAGS.read_text().splitlines()))
    snaps = pd.concat([pd.read_parquet(f, columns=["ticker", "ts", "start", "fair", "pfair", "yes_ask", "no_ask"])
                       for f in sorted(SNAP.glob("*.parquet"))], ignore_index=True)
    snaps = snaps[snaps["ts"] < snaps["start"]].sort_values("ts")
    close = snaps.groupby("ticker").last()
    fl["fair_close"] = fl["ticker"].map(close["fair"])
    fl["close_ts"] = fl["ticker"].map(close["ts"])
    # CLV per $1 risked: closing fair value of what we bought minus what we paid
    win_p = np.where(fl["side"] == "yes", fl["fair_close"], 1 - fl["fair_close"])
    cost = fl["ask"] + fl["fee"]
    fl["clv"] = (win_p - cost) / cost
    # settlement from Kalshi
    res = {}
    for t in fl["ticker"].unique():
        try:
            m = live._get(f"{live.KALSHI}/markets/{t}")["market"]
            res[t] = m.get("result")
        except Exception:
            res[t] = None
    fl["result"] = fl["ticker"].map(res)
    done = fl["result"].isin(["yes", "no"])
    won = (fl["result"] == fl["side"])
    fl["pnl"] = np.where(done, np.where(won, 1 - cost, -cost), np.nan) / cost
    pd.set_option("display.width", 220)
    print(f"{len(fl)} flags; with closing snapshot {fl['fair_close'].notna().sum()}; settled {done.sum()}")
    c = fl["clv"].dropna()
    if len(c):
        print(f"CLV per $1: mean {c.mean():+.2%}  (t = {c.mean() / (c.std(ddof=1) / np.sqrt(len(c))) if len(c) > 1 else float('nan'):.2f}, n={len(c)})"
              f"  | entry EV claimed {fl['ev'].mean():+.2%}")
    if done.any():
        s = fl[done]
        print(f"settled P&L per $1: {s['pnl'].sum():+.2f} on {len(s)} bets, ROI {s['pnl'].mean():+.2%}")
    print(fl.groupby("league").agg(n=("ev", "size"), ev=("ev", "mean"), clv=("clv", "mean"),
                                   settled=("pnl", "count"), roi=("pnl", "mean")).round(3).to_string())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["scan", "loop", "report"])
    ap.add_argument("--every", type=float, default=10.0, help="minutes between scans (loop)")
    a = ap.parse_args()
    if a.cmd == "scan":
        scan()
    elif a.cmd == "report":
        report()
    else:
        every = max(a.every, 5.0)
        while True:
            try:
                scan(verbose=False)
                print(f"{live.now():%H:%M} scan ok", flush=True)
            except Exception as e:
                print(f"{live.now():%H:%M} scan failed: {e}", flush=True)
            time.sleep(every * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
