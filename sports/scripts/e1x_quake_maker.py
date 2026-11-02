"""E1X-Q amendment A1: maker NO bid 1c inside the YES ask on KXBIGGESTQUAKE, filled from the public trade tape.
Prereg: notes/e1x_quake.md (A1). Stages: trades -> score --split dev|sealed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e1_family_scan import get  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "e1x"
LEDGER = ROOT / "lab" / "ledger.jsonl"
SERIES = "KXBIGGESTQUAKE"
SEED = 20261102
NBOOT = 10_000
MARGIN = 0.03


def cmd_trades() -> int:
    b = pd.read_parquet(D / "quake_build.parquet")
    cache = D / "quake_trades.json"
    out = json.loads(cache.read_text()) if cache.exists() else {}
    for tk, ts in zip(b.ticker, b.dec_ts):
        if tk in out:
            continue
        tr, cur = [], None
        while True:
            p = {"ticker": tk, "min_ts": int(pd.Timestamp(ts).timestamp()), "limit": 1000}
            if cur:
                p["cursor"] = cur
            r = get("/markets/trades", p)
            tr += r.get("trades", [])
            cur = r.get("cursor")
            if not cur or not r.get("trades"):
                break
        out[tk] = tr
    cache.write_text(json.dumps(out))
    print("tickers", len(out), "trades", sum(len(v) for v in out.values()))
    return 0


def _yp(t: dict) -> float:
    v = t.get("yes_price_dollars", t.get("yes_price"))
    v = float(v)
    return v / 100 if v > 1.5 else v


def cmd_score(split: str) -> int:
    b = pd.read_parquet(D / "quake_build.parquet")
    if split == "dev":
        b = b[(b.day >= "2026-09-02") & (b.day <= "2026-10-14")]
    else:
        b = b[(b.day >= "2026-10-15") & (b.day <= "2026-11-01")]
    res = pd.read_parquet(D / f"{SERIES}_markets.parquet")[["ticker", "event_ticker", "result"]]
    full = res.groupby("event_ticker").result.apply(lambda r: bool(r.isin(["yes", "no"]).all()))
    n0 = b.event.nunique()
    b = b[b.event.map(full).fillna(False).astype(bool)].merge(res.drop(columns="event_ticker"), on="ticker")
    print(f"{split}: {b.event.nunique()} complete days of {n0}")
    trades = json.loads((D / "quake_trades.json").read_text())
    x = b[(~b.etas_guard) & b.yes_bid.notna() & b.yes_ask.notna()].copy()
    x = x[(x.yes_ask - x.yes_bid >= 0.02 - 1e-9) & (x.yes_ask <= 0.99)]
    x["yes_offer"] = (x.yes_ask - 0.01).round(2)
    x["q"] = (1 - x.yes_offer).round(2)
    x = x[(1 - x.p) - x.q >= MARGIN]
    filled = []
    for tk, off, ts in zip(x.ticker, x.yes_offer, x.dec_ts):
        t0 = pd.Timestamp(ts)
        ok = False
        for t in trades.get(tk, []):
            tt = pd.Timestamp(t["created_time"])
            if tt > t0 and t.get("taker_side") == "yes" and _yp(t) >= off - 1e-9:
                ok = True
                break
        filled.append(ok)
    x["filled"] = filled
    x["lose"] = x.result == "yes"
    rng = np.random.default_rng(SEED)
    k = 2
    rows = []
    for trial in ("M1", "M2"):
        y = x if trial == "M1" else x[x.K >= 6.4]
        f = y[y.filled].copy()
        posted = len(y)
        if f.empty:
            print(trial, "posted", posted, "0 fills")
            continue
        f["cost"] = f.q
        f["pnl"] = np.where(f.result == "no", 1.0, 0.0) - f.cost
        g = f.groupby("event").agg(pnl=("pnl", "sum"), cost=("cost", "sum"))
        roi = g.pnl.sum() / g.cost.sum()
        bs = np.empty(NBOOT)
        for i in range(NBOOT):
            s = rng.integers(0, len(g), len(g))
            bs[i] = g.pnl.to_numpy()[s].sum() / g.cost.to_numpy()[s].sum()
        p = float((bs <= 0).mean())
        dd = np.sort(f.day.unique())
        h1, h2 = f[f.day.isin(dd[: len(dd) // 2])], f[f.day.isin(dd[len(dd) // 2:])]
        r1, r2 = h1.pnl.sum() / h1.cost.sum(), h2.pnl.sum() / h2.cost.sum()
        row = dict(trial=trial, posted=posted, fills=len(f), fill_rate=round(len(f) / posted, 3),
                   days=int(f.event.nunique()), losses=int(f.lose.sum()),
                   loss_rate_filled=round(f.lose.mean(), 3),
                   loss_rate_unfilled=round(y[~y.filled].lose.mean(), 3) if (~y.filled).any() else None,
                   mean_p_filled=round(f.p.mean(), 3), mean_q=round(f.q.mean(), 3), roi=round(roi, 4),
                   ci=[round(np.percentile(bs, 2.5), 4), round(np.percentile(bs, 97.5), 4)], p=round(p, 4),
                   half1=round(r1, 4), half2=round(r2, 4), pnl_units=round(f.pnl.sum(), 3),
                   passed=bool(p < 0.05 / k and r1 > 0 and r2 > 0))
        print(json.dumps(row))
        if roi > 0.10 and len(f) > 200:
            print("TOO-GOOD GUARD: inspect rows")
        rows.append(row)
        f.to_csv(D / f"quake_maker_{trial}_{split}.csv", index=False)
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    with LEDGER.open("a") as fh:
        for i, r in enumerate(rows, 1):
            fh.write(json.dumps(dict(
                ts=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), strategy=f"e1x_quake_{r['trial']}",
                sha=sha, dataset="e1x_quake",
                hypothesis="maker NO bid 1c inside YES ask at 23:00 UTC D-1 if (1-p_clim)-q >= .03, tape fills"
                           + (", K>=6.4" if r["trial"] == "M2" else ""),
                source="agent E1-pattern search 2026-11-01 (notes/e1x_quake.md A1)", status="ok",
                split=("DEV 2026-09-02..10-14" if split == "dev" else "SEALED 2026-10-15..11-01"),
                trial_number=i, study_trials=k, **r)) + "\n")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--split", default="dev")
    a = ap.parse_args()
    sys.exit(cmd_trades() if a.stage == "trades" else cmd_score(a.split))
