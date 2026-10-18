"""Does resting orders on Kalshi sports earn the spread net of adverse selection? Symmetric ladder test. DEV only.

    python scripts/kalshi_maker_ladder.py build          # candles T-8h..T-1h for dev total/spread ladders (cached)
    python scripts/kalshi_maker_ladder.py dev [--ledger]

The winner maker result (kalshi_maker.py MK1 +9.8%) used a price band chosen after seeing dev calibration,
and its mirror (MK2) lost. This asks the mechanism question with NO directional choice.
Pre-registered 2026-10-18, before any ladder candle was built:
  T = expected_expiration - 3 h (2025 dev convention, verified). Quote at T-3h (last candle ending <= T-3h),
  two-sided, spread <= 5c, rung mid in [0.35, 0.65].
  MK3 maker_ladder_yes : rest YES bid at b; filled iff a trade prints < b in (T-3h, T-1h]; cost b + maker fee
  MK4 maker_ladder_no  : rest NO bid at 1 - a (a = YES ask); filled iff a trade prints > a in the window; cost 1-a + fee
  Maker fee 0.0175*p*(1-p). Dataset kalshi_sports (trials 9-10). Bootstrap over events (rungs of a game correlate).
  Reading: both > 0 -> resting orders earn the spread here; one side only -> directional noise.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from sports.http import fetch_many
import kalshi_longshot as K
import kalshi_maker as M

OUT = ROOT / "data" / "kalshi_maker_ladder_candles.parquet"


def cmd_build() -> int:
    px = pd.read_parquet(K.OUT)
    w = px[(px.kind == "ladder") & (px.event_date < K.SPLIT)].copy()
    w["t_start"] = w.decision_ts + 3600
    rows = []
    for series, mk in w.groupby("series"):
        ev = mk.groupby("event_ticker")["t_start"].min()
        urls = [f"{K.API}/series/{series}/events/{e}/candlesticks?start_ts={t - 8 * 3600}&end_ts={t - 3600}&period_interval=60"
                for e, t in ev.items()]
        missing = []
        for (e, t), d in zip(ev.items(), fetch_many(urls)):
            tks = d.get("market_tickers", [])
            if not tks:
                missing.append(e)
            for tk, cs in zip(tks, d.get("market_candlesticks", [])):
                r = M._parse(tk, cs, t)
                if r:
                    rows.append(r)
        leg = mk[mk.event_ticker.isin(missing)]
        urls = [f"{K.API}/historical/markets/{tk}/candlesticks?start_ts={t - 8 * 3600}&end_ts={t - 3600}&period_interval=60"
                for tk, t in zip(leg.ticker, leg.t_start)]
        for (tk, t), d in zip(zip(leg.ticker, leg.t_start), fetch_many(urls)):
            r = M._parse(tk, d.get("candlesticks", []), t)
            if r:
                rows.append(r)
        print(f"{series}: {len(ev)} events", flush=True)
    out = w.drop(columns=["yes_bid", "yes_ask"]).merge(pd.DataFrame(rows), on="ticker", how="inner")
    out.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(out)} rungs)")
    return 0


def cmd_dev() -> int:
    d = pd.read_parquet(OUT)
    d = d[d.event_date < K.SPLIT]
    d = d[d.bid_post.notna() & d.ask_post.notna() & (d.bid_post > 0) & (d.ask_post < 1) & (d.ask_post - d.bid_post <= 0.05)].copy()
    d["mid"] = (d.bid_post + d.ask_post) / 2
    d = d[(d.mid >= 0.35) & (d.mid <= 0.65)]
    print(f"DEV ladder rungs in band {len(d)} ({d.event_ticker.nunique()} events); "
          f"YES fill {(d.min_trade_after < d.bid_post).mean():.1%}, NO fill {(d.max_trade_after > d.ask_post).mean():.1%}")
    prior = [json.loads(l) for l in K.LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == K.DATASET}
    recs = []
    for name in ("maker_ladder_yes", "maker_ladder_no"):
        if name.endswith("yes"):
            s = d[d.min_trade_after < d.bid_post]
            price, won = s.bid_post, s.result == "yes"
        else:
            s = d[d.max_trade_after > d.ask_post]
            price, won = 1 - s.ask_post, s.result == "no"
        cost = price + M.MAKER_FEE * price * (1 - price)
        t = pd.DataFrame({"event": s.event_ticker.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                          "pnl": np.where(won, 1 - cost, -cost), "stake": cost.to_numpy()})
        r = K.score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("kalshi_maker_ladder:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        by = t.groupby("series").apply(lambda g: f"{g.pnl.sum() / g.stake.sum():+.1%}/{len(g)}", include_groups=False)
        print(f"{name:17s} fills {r['bets']:5d} events {r['events']:4d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  quarters+ {r['folds_positive']}/{len(r['per_fold'])}")
        print(f"                  by series: {dict(by)}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_{name}", "sha": sha,
                     "dataset": K.DATASET, "hypothesis": f"pre-registered in scripts/kalshi_maker_ladder.py ({name})",
                     "source": "market-making mechanism test", "status": "ok", "trial_number": len(shas), **r})
    print(f"{K.DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with K.LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
