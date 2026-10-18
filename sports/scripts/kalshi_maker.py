"""Kalshi sports game winners: MAKER side (rest a bid, get filled, hold to settlement). DEV only.

    python scripts/kalshi_maker.py build           # hourly candles T-4h..T-1h for dev winner markets (cached)
    python scripts/kalshi_maker.py dev [--ledger]

Bürgi, Deng & Whelan (2025): makers earn more than takers on Kalshi. Taker tests (kalshi_longshot.py)
showed 20-50c underdogs hit +2.7pp above price but fees/spread ate it (+1.4%, p .21).
Pre-registered 2026-10-18, before building any candles:
  T = scheduled start (= expected_expiration - 3 h, valid for 2025 dev events, checked twice).
  At T-3h rest a bid for YES at the then-best yes_bid b (last hourly candle ending <= T-3h).
  FILLED iff some trade in the candles ending in (T-3h, T-1h] printed at a price STRICTLY below b
  (trade-through: conservative about queue position; captures adverse selection). Unfilled = no bet.
  Cost = b + maker fee 0.0175*b*(1-b); payoff 1 if YES settles yes.
  MK1 maker_dog2050 : the underdog market with mid in [0.20, 0.50) at T-3h
  MK2 maker_fav5080 : the favourite market with mid in [0.50, 0.80) at T-3h (mirror control)
Dataset kalshi_sports (trials 7-8). Dev = events dated < 2026-01-01. Bootstrap over events.
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

OUT = ROOT / "data" / "kalshi_maker_candles.parquet"
OUT_SEALED = ROOT / "data" / "kalshi_maker_candles_sealed.parquet"
MAKER_FEE = 0.0175


def _num(v):
    return pd.to_numeric(v, errors="coerce")


def _parse(tk, candles, t_start):
    t_post, t_end = t_start - 3 * 3600, t_start - 3600
    pre = [c for c in candles if c["end_period_ts"] <= t_post]
    win = [c for c in candles if t_post < c["end_period_ts"] <= t_end]
    if not pre:
        return None
    last = pre[-1]
    g = lambda c, side, f: _num((c.get(side) or {}).get(f + "_dollars", (c.get(side) or {}).get(f)))
    lows = [g(c, "price", "low") for c in win]
    lows = [x for x in lows if pd.notna(x)]
    highs = [x for x in (g(c, "price", "high") for c in win) if pd.notna(x)]
    return {"ticker": tk, "bid_post": g(last, "yes_bid", "close"), "ask_post": g(last, "yes_ask", "close"),
            "min_trade_after": min(lows) if lows else np.nan, "max_trade_after": max(highs) if highs else np.nan}


def cmd_build(sealed: bool = False) -> int:
    px = pd.read_parquet(K.OUT)
    if not sealed:
        w = px[(px.kind == "winner") & (px.event_date < K.SPLIT)].copy()
        w["t_start"] = w.decision_ts + 3600  # decision_ts = exp - 4h = start - 1h for 2025 events
        dest = OUT
    else:  # 2026: verified real start times only (kalshi_start_times.py), never exp-based (leaks in 2026)
        st = pd.read_parquet(ROOT / "data" / "kalshi_event_starts.parquet").dropna(subset=["start_utc"])
        w = px[(px.kind == "winner") & (px.event_date >= K.SPLIT)].merge(st[["event_ticker", "start_utc"]], on="event_ticker")
        w["t_start"] = ((pd.to_datetime(w.start_utc, utc=True) - pd.Timestamp("1970-01-01", tz="UTC"))
                        // pd.Timedelta(seconds=1)).astype(int)
        w = w.drop(columns=["start_utc"])
        dest = OUT_SEALED
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
                r = _parse(tk, cs, t)
                if r:
                    rows.append(r)
        leg = mk[mk.event_ticker.isin(missing)]
        urls = [f"{K.API}/historical/markets/{tk}/candlesticks?start_ts={t - 8 * 3600}&end_ts={t - 3600}&period_interval=60"
                for tk, t in zip(leg.ticker, leg.t_start)]
        for (tk, t), d in zip(zip(leg.ticker, leg.t_start), fetch_many(urls)):
            r = _parse(tk, d.get("candlesticks", []), t)
            if r:
                rows.append(r)
        print(f"{series}: {len(ev)} events", flush=True)
    c = pd.DataFrame(rows)
    out = w.drop(columns=["yes_bid", "yes_ask"]).merge(c, on="ticker", how="inner")
    out.to_parquet(dest, index=False)
    print(f"-> {dest} ({len(out)} markets){' -- prices only, NOT scored' if sealed else ''}")
    return 0


def cmd_dev() -> int:
    d = pd.read_parquet(OUT)
    d = d[d.event_date < K.SPLIT]
    d = d[d.bid_post.notna() & d.ask_post.notna() & (d.bid_post > 0) & (d.ask_post - d.bid_post <= 0.05)]
    d["mid"] = (d.bid_post + d.ask_post) / 2
    two = d.groupby("event_ticker")["ticker"].transform("count") == 2
    d = d[two].copy()
    d["is_fav"] = d.groupby("event_ticker")["mid"].transform("max") == d["mid"]
    d["filled"] = d.min_trade_after < d.bid_post
    print(f"DEV winner markets {len(d)} ({d.event_ticker.nunique()} events); fill rate {d.filled.mean():.1%}")
    prior = [json.loads(l) for l in K.LEDGER.read_text().splitlines() if l.strip()]
    shas = {r["sha"] for r in prior if r.get("status") == "ok" and r["dataset"] == K.DATASET}
    recs = []
    for name, sel in (("maker_dog2050", (~d.is_fav) & (d.mid >= 0.20) & (d.mid < 0.50)),
                      ("maker_fav5080", d.is_fav & (d.mid >= 0.50) & (d.mid < 0.80))):
        s = d[sel & d.filled]
        cost = s.bid_post + MAKER_FEE * s.bid_post * (1 - s.bid_post)
        pnl = np.where(s.result == "yes", 1 - cost, -cost)
        t = pd.DataFrame({"event": s.event_ticker.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                          "pnl": pnl, "stake": cost.to_numpy()})
        r = K.score(t, np.random.default_rng(0))
        sha = hashlib.sha256(("kalshi_maker:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        by = t.groupby("series").apply(lambda g: f"{g.pnl.sum() / g.stake.sum():+.1%}/{len(g)}", include_groups=False)
        print(f"{name:14s} eligible {int(sel.sum()):5d} filled {r['bets']:5d}  units {r['units']:+7.1f}  ROI {r['roi']:+6.2%}  "
              f"95%CI [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}]  p {r['p']:.4f}  quarters+ {r['folds_positive']}/{len(r['per_fold'])}")
        print(f"               by series: {dict(by)}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_{name}", "sha": sha,
                     "dataset": K.DATASET, "hypothesis": f"pre-registered in scripts/kalshi_maker.py ({name})",
                     "source": "Burgi, Deng & Whelan 2025 (makers vs takers)", "status": "ok", "trial_number": len(shas), **r})
    print(f"{K.DATASET} trials {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with K.LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
    return 0


def cmd_sealed() -> int:
    """ONE-TIME sealed score of maker_dog2050 (2026 events, verified start). Needs the user's explicit yes.
    Pre-registered 2026-10-18 before any sealed candle existed: same rule as dev; PASS iff p < 0.05 / n_promoted
    (kalshi_sports promotions incl. this one) and units > 0."""
    if "--confirm" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm")
        return 1
    name = "maker_dog2050"
    sha = hashlib.sha256(("kalshi_maker:" + name).encode()).hexdigest()[:12]
    prior = [json.loads(l) for l in K.LOCKBOX_LOG.read_text().splitlines() if l.strip()]
    if any(r["sha"] == sha for r in prior):
        print("already scored on the lockbox; refusing")
        return 1
    d = pd.read_parquet(OUT_SEALED)
    d = d[d.event_date >= K.SPLIT]
    d = d[d.bid_post.notna() & d.ask_post.notna() & (d.bid_post > 0) & (d.ask_post - d.bid_post <= 0.05)].copy()
    d["mid"] = (d.bid_post + d.ask_post) / 2
    d = d[d.groupby("event_ticker")["ticker"].transform("count") == 2]
    d["is_fav"] = d.groupby("event_ticker")["mid"].transform("max") == d["mid"]
    s = d[(~d.is_fav) & (d.mid >= 0.20) & (d.mid < 0.50) & (d.min_trade_after < d.bid_post)]
    cost = s.bid_post + MAKER_FEE * s.bid_post * (1 - s.bid_post)
    t = pd.DataFrame({"event": s.event_ticker.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                      "pnl": np.where(s.result == "yes", 1 - cost, -cost), "stake": cost.to_numpy()})
    r = K.score(t, np.random.default_rng(0))
    n_prom = 1 + sum(x["dataset"] == K.DATASET for x in prior)
    bar = 0.05 / n_prom
    with K.LOCKBOX_LOG.open("a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_{name}",
                            "sha": sha, "dataset": K.DATASET, "promotion_number": n_prom, **r}, default=str) + "\n")
    print(f"SEALED {name}: filled {r['bets']} units {r['units']:+.1f} ROI {r['roi']:+.2%} p {r['p']:.4f} "
          f"quarters+ {r['folds_positive']}/{len(r['per_fold'])} -> {'PASS' if r['p'] < bar and r['units'] > 0 else 'FAIL'} (bar {bar:.4f})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "build-sealed": lambda: cmd_build(sealed=True), "dev": cmd_dev,
                      "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())
