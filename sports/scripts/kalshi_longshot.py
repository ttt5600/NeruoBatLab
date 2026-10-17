"""Favorite-longshot bias on Kalshi SPORTS markets (game winners + total/spread ladders).

    python scripts/kalshi_longshot.py build            # settled markets + hourly price 1 h before start (cached)
    python scripts/kalshi_longshot.py dev [--ledger]   # calibration table + the 5 pre-registered trials, DEV only
    python scripts/kalshi_longshot.py sealed --confirm # ONE-TIME score on 2026+ events (user's yes required)

Decision time: Kalshi sets expected_expiration_time = scheduled start + 3 h (checked on
NFL 2025-09-04 DAL@PHI and NBA 2025-12-25 CLE@NYK). We price at expiration - 4 h,
i.e. ~1 h before start, from the last hourly candle ending at or before then; if a
league's offset is longer than 3 h the decision is only earlier (safe). Quote must be
two-sided with spread <= 5c. Costs: taker fee 0.07*p*(1-p) per contract.

Pre-registered 2026-10-17, before any price was looked at:
  split : DEV = events dated < 2026-01-01; SEALED = 2026-01-01 onward (not loaded by `dev`)
  G1 fav85   winners: back the favorite when its implied prob (mid) >= 0.85, best execution
             (YES on its market at the ask, or NO on the opponent's at 1 - bid)
  G2 fav65   winners: favorite mid in [0.65, 0.85)
  G3 dog15   winners: back the underdog when its mid <= 0.15
  L1 tailno  ladders: buy NO on any rung with YES ask <= 0.10
  L2 headyes ladders: buy YES on any rung with YES ask >= 0.90
  5 trials -> bar p < 0.01, one-sided bootstrap over EVENTS (rungs of one game are not independent).
  Sealed pass = p < 0.05 / (#trials promoted) and units > 0.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from sports.http import fetch, fetch_many

API = "https://api.elections.kalshi.com/trade-api/v2"
OUT = ROOT / "data" / "kalshi_sports_prices.parquet"
LEDGER = ROOT / "lab" / "ledger.jsonl"
LOCKBOX_LOG = ROOT / "lab" / "lockbox.jsonl"
DATASET = "kalshi_sports"
WINNERS = ["KXNFLGAME", "KXNBAGAME", "KXMLBGAME", "KXNHLGAME", "KXNCAAFGAME", "KXNCAABGAME", "KXWNBAGAME"]
LADDERS = ["KXNFLTOTAL", "KXNFLSPREAD", "KXNBATOTAL", "KXNBASPREAD", "KXMLBTOTAL", "KXMLBSPREAD",
           "KXNHLTOTAL", "KXNHLSPREAD", "KXNCAAFTOTAL", "KXNCAAFSPREAD"]
SPLIT = pd.Timestamp("2026-01-01")
FEE = 0.07
TRIALS = ["G1_fav85", "G2_fav65", "G3_dog15", "L1_tailno", "L2_headyes"]


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
    df = df[df["result"].isin(["yes", "no"]) & df["expected_expiration_time"].notna()].copy()
    df["series"] = series
    df["decision_ts"] = (pd.to_datetime(df["expected_expiration_time"], utc=True) - pd.Timedelta(hours=4)).astype("int64") // 10**9
    # event date from the ticker: KXNFLGAME-25SEP04DALPHI -> 2025-09-04
    df["event_date"] = pd.to_datetime(df["event_ticker"].str.split("-").str[1].str[:7], format="%y%b%d", errors="coerce")
    keep = ["series", "ticker", "event_ticker", "event_date", "decision_ts", "result", "floor_strike", "yes_sub_title"]
    return df[[c for c in keep if c in df]].dropna(subset=["event_date"])


def _px(candles, t):
    c = [x for x in candles if x["end_period_ts"] <= t]
    if not c:
        return np.nan, np.nan
    last = c[-1]
    g = lambda side: pd.to_numeric((last.get(side) or {}).get("close_dollars"), errors="coerce")
    return g("yes_bid"), g("yes_ask")


def prices(series: str, mk: pd.DataFrame) -> pd.DataFrame:
    ev = mk.groupby("event_ticker")["decision_ts"].min()
    urls = [f"{API}/series/{series}/events/{e}/candlesticks?start_ts={t - 6 * 3600}&end_ts={t}&period_interval=60"
            for e, t in ev.items()]
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
    frames = []
    for s in WINNERS + LADDERS:
        mk = settled(s)
        if mk.empty:
            print(f"{s}: no settled markets")
            continue
        px = prices(s, mk)
        print(f"{s}: {len(mk)} markets, {mk.event_ticker.nunique()} events, priced {px.yes_ask.notna().sum()}", flush=True)
        frames.append(px.assign(kind="winner" if s in WINNERS else "ladder"))
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(OUT, index=False)
    print(f"-> {OUT} ({len(df)} rows)")
    return 0


# ---------------------------------------------------------------- bets
def fee(p):
    return FEE * p * (1 - p)


def ok_quote(d):
    return d.yes_bid.notna() & d.yes_ask.notna() & (d.yes_ask - d.yes_bid <= 0.05) & (d.yes_bid > 0) & (d.yes_ask < 1)


def winner_bets(w: pd.DataFrame) -> pd.DataFrame:
    """One row per (event, side backed), with the cheaper of the two ways to back that side."""
    w = w[ok_quote(w)].copy()
    w["mid"] = (w.yes_bid + w.yes_ask) / 2
    rows = []
    for e, g in w.groupby("event_ticker"):
        if len(g) != 2:  # two-team markets only (skips ties/3-way)
            continue
        a, b = g.iloc[0], g.iloc[1]
        for me, opp in ((a, b), (b, a)):
            cost = min(me.yes_ask, 1 - opp.yes_bid)
            rows.append({"event": e, "series": me.series, "event_date": me.event_date, "mid": me.mid,
                         "cost": cost, "won": me.result == "yes"})
    return pd.DataFrame(rows)


def pnl(cost, won):
    c = cost + fee(cost)
    return np.where(won, 1 - c, -c), c


def trial_frame(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name.startswith("G"):
        wb = winner_bets(df[df.kind == "winner"])
        fav = wb.groupby("event")["mid"].transform("max") == wb["mid"]
        sel = {"G1_fav85": fav & (wb.mid >= 0.85), "G2_fav65": fav & (wb.mid >= 0.65) & (wb.mid < 0.85),
               "G3_dog15": (~fav) & (wb.mid <= 0.15)}[name]
        s = wb[sel]
        p, c = pnl(s.cost.to_numpy(), s.won.to_numpy())
        return pd.DataFrame({"event": s.event.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                             "pnl": p, "stake": c})
    lad = df[(df.kind == "ladder") & ok_quote(df)]
    if name == "L1_tailno":
        s = lad[lad.yes_ask <= 0.10]
        cost, won = 1 - s.yes_bid.to_numpy(), (s.result == "no").to_numpy()
    else:
        s = lad[lad.yes_ask >= 0.90]
        cost, won = s.yes_ask.to_numpy(), (s.result == "yes").to_numpy()
    p, c = pnl(cost, won)
    return pd.DataFrame({"event": s.event_ticker.to_numpy(), "series": s.series.to_numpy(), "date": s.event_date.to_numpy(),
                         "pnl": p, "stake": c})


def score(t: pd.DataFrame, rng) -> dict:
    ev = t.groupby("event").agg(pnl=("pnl", "sum"), stake=("stake", "sum"))
    boots = np.array([ev.sample(len(ev), replace=True, random_state=rng.integers(1 << 31))["pnl"].sum()
                      for _ in range(3000)])
    q = pd.PeriodIndex(pd.to_datetime(t.date), freq="Q").astype(str)
    per = t.groupby(q)["pnl"].sum().round(2)
    return {"bets": int(len(t)), "events": int(len(ev)), "staked": float(t.stake.sum()), "units": float(t.pnl.sum()),
            "roi": float(t.pnl.sum() / t.stake.sum()) if len(t) else 0.0,
            "ci_lo": float(np.quantile(boots, 0.025)), "ci_hi": float(np.quantile(boots, 0.975)),
            "p": float(np.mean(boots <= 0)), "folds_positive": int((per > 0).sum()), "per_fold": per.to_dict()}


def calibration(df: pd.DataFrame):
    d = df[ok_quote(df)].copy()
    d["mid"] = (d.yes_bid + d.yes_ask) / 2
    d["b"] = pd.cut(d.mid, [0, .05, .10, .20, .35, .50, .65, .80, .90, .95, 1.0])
    d["y"] = (d.result == "yes").astype(float)
    print(f"{'kind':7s} {'bucket':12s} {'n':>6s} {'mid':>6s} {'hit':>6s} {'hit-mid':>8s}")
    for (k, b), g in d.groupby(["kind", "b"], observed=True):
        print(f"{k:7s} {str(b):12s} {len(g):6d} {g.mid.mean():6.3f} {g.y.mean():6.3f} {g.y.mean() - g.mid.mean():+8.3f}")


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
        sha = hashlib.sha256(("kalshi_longshot:" + name).encode()).hexdigest()[:12]
        shas.add(sha)
        by = t.groupby("series").apply(lambda g: f"{g.pnl.sum() / g.stake.sum():+.1%}/{len(g)}", include_groups=False)
        print(f"{name:11s} bets {res['bets']:6d} events {res['events']:5d}  units {res['units']:+8.1f}  ROI {res['roi']:+6.2%}  "
              f"95%CI [{res['ci_lo']:+.1f}, {res['ci_hi']:+.1f}]  p {res['p']:.4f}  quarters+ {res['folds_positive']}/{len(res['per_fold'])}")
        print(f"            by series: {dict(by)}")
        recs.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_longshot_{name}",
                     "sha": sha, "dataset": DATASET, "hypothesis": f"pre-registered in scripts/kalshi_longshot.py ({name})",
                     "source": "favorite-longshot bias, prediction-market literature", "status": "ok",
                     "trial_number": len(shas), **res})
    print(f"\n{DATASET} trials: {len(shas)} -> bar p < {0.05 / len(shas):.4f}")
    if "--ledger" in sys.argv:
        with LEDGER.open("a") as f:
            for r in recs:
                f.write(json.dumps(r, default=str) + "\n")
        print(f"appended {len(recs)} trials to ledger")
    return 0


def cmd_sealed() -> int:
    if "--confirm" not in sys.argv:
        print("sealed scoring is the user's decision; pass --confirm")
        return 1
    names = [a for a in sys.argv[2:] if a in TRIALS]
    if not names:
        print(f"name the promoted trial(s): {TRIALS}")
        return 1
    prior = [json.loads(l) for l in LOCKBOX_LOG.read_text().splitlines() if l.strip()]
    df = pd.read_parquet(OUT)
    df = df[df.event_date >= SPLIT]
    n_prom = sum(r["dataset"] == DATASET for r in prior) + len(names)
    for name in names:
        sha = hashlib.sha256(("kalshi_longshot:" + name).encode()).hexdigest()[:12]
        if any(r["sha"] == sha for r in prior):
            print(f"{name} already scored on the lockbox; refusing")
            continue
        res = score(trial_frame(df, name), np.random.default_rng(0))
        bar = 0.05 / n_prom
        with LOCKBOX_LOG.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": f"kalshi_longshot_{name}",
                                "sha": sha, "dataset": DATASET, "promotion_number": n_prom, **res}, default=str) + "\n")
        print(f"SEALED {name}: bets {res['bets']} units {res['units']:+.1f} ROI {res['roi']:+.2%} p {res['p']:.4f} -> "
              f"{'PASS' if res['p'] < bar and res['units'] > 0 else 'FAIL'} (bar {bar:.4f})")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"build": cmd_build, "dev": cmd_dev, "sealed": cmd_sealed}.get(cmd, lambda: print(__doc__) or 1)())
