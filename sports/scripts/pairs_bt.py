"""Statistical arbitrage / pairs trading backtest (user 2026-10-21: "coke and pepsi"). DEV only.

    python scripts/pairs_bt.py dev [--ledger]

Classic distance/z-score pairs rule (Gatev, Goetzmann & Rouwenhorst 2006), daily closes, yfinance adjusted prices.
Pre-registered 2026-10-21 before any price was downloaded:
  pairs    : KO/PEP, XOM/CVX, V/MA, HD/LOW, UPS/FDX, MCD/YUM, JPM/BAC, GOOGL/MSFT is NOT included (not a substitute pair)
  spread   : log(A) - beta*log(B), beta = OLS on the trailing 252 trading days (re-fit daily, no look-ahead)
  signal   : z = (spread - mean_252) / sd_252; open when |z| >= 2 (short the rich leg, long the cheap leg, dollar-neutral,
             $1 per leg), close when |z| <= 0.5 or after 20 trading days (time stop); one position per pair at a time
  costs    : 5 bps per leg per side (4 legs per round trip = 20 bps) + 0 borrow fee (optimistic; noted)
  split    : DEV = 2006-01-01..2018-12-31; SEALED = 2019-01-01..2025-12-31 (user's yes; never loaded by `dev`)
  T1 pairs_z2 : equal-weight daily P&L across the 7 pairs (each pair's P&L in $ per $1 per leg)
  1 trial -> bar p < 0.05 (stocks bar is shared with stocks_ledger in spirit; reported, not mixed); HAC (Newey-West 10)
  t-stat on daily P&L; also Sharpe and per-pair totals.
Prior: published pairs profits fell from ~11%/yr (1962-2002) to ~0 after 2002 (Do & Faff 2010) -- expect little.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd

PAIRS = [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("HD", "LOW"), ("UPS", "FDX"), ("MCD", "YUM"), ("JPM", "BAC")]
CACHE = ROOT / "data" / "stocks" / "pairs_prices.parquet"
LEDGER = ROOT / "lab" / "stocks_ledger.jsonl"
COST = 0.0005
DEV_END = "2018-12-31"


def prices() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    tk = sorted({t for p in PAIRS for t in p})
    px = yf.download(tk, start="2004-01-01", end="2025-12-31", auto_adjust=True, progress=False)["Close"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_parquet(CACHE)
    return px


def run_pair(a: pd.Series, b: pd.Series) -> pd.Series:
    la, lb = np.log(a), np.log(b)
    df = pd.DataFrame({"a": la, "b": lb}).dropna()
    ra, rb = df.a.diff(), df.b.diff()
    pnl = pd.Series(0.0, index=df.index)
    pos, days, beta_open = 0, 0, 1.0
    va, vb = df.a.to_numpy(), df.b.to_numpy()
    for i in range(252, len(df)):
        wa, wb = va[i - 252:i], vb[i - 252:i]   # strictly past window
        beta = np.cov(wa, wb)[0, 1] / np.var(wb, ddof=1)
        s_hist = wa - beta * wb
        mu, sd = s_hist.mean(), s_hist.std(ddof=1)
        if pos != 0:  # P&L of yesterday's position over today's move: dollar-neutral, $1 per leg
            pnl.iloc[i] = pos * (np.expm1(ra.iloc[i]) - np.expm1(rb.iloc[i]))
            days += 1
        z = (va[i] - beta * vb[i] - mu) / sd if sd > 0 else 0.0
        if pos == 0 and abs(z) >= 2:
            pos, days = (-1 if z > 0 else 1), 0      # z>0: A rich -> short A, long B
            pnl.iloc[i] -= 2 * COST                   # open both legs
        elif pos != 0 and (abs(z) <= 0.5 or days >= 20):
            pos = 0
            pnl.iloc[i] -= 2 * COST                   # close both legs
    return pnl


def hac_t(x: np.ndarray, lags: int = 10) -> float:
    x = x - x.mean()
    n = len(x)
    g0 = np.dot(x, x) / n
    s = g0 + 2 * sum((1 - k / (lags + 1)) * np.dot(x[k:], x[:-k]) / n for k in range(1, lags + 1))
    return float((x.mean() + 0) if s <= 0 else 0)  # placeholder (overwritten below)


def cmd_dev() -> int:
    px = prices()
    px = px[px.index <= DEV_END]
    per = {}
    for a, b in PAIRS:
        per[f"{a}/{b}"] = run_pair(px[a], px[b])
    P = pd.DataFrame(per).fillna(0.0)
    P = P[P.index >= "2006-01-01"]
    port = P.mean(axis=1)
    x = port.to_numpy()
    n = len(x)
    d = x - x.mean()
    s = np.dot(d, d) / n + 2 * sum((1 - k / 11) * np.dot(d[k:], d[:-k]) / n for k in range(1, 11))
    t = x.mean() / np.sqrt(s / n)
    sharpe = x.mean() / x.std(ddof=1) * np.sqrt(252)
    ann = x.mean() * 252
    from scipy.stats import norm
    p = float(1 - norm.cdf(t))
    print(f"T1_pairs_z2 DEV 2006-2018: annual P&L {ann:+.2%} per $1/leg (equal-weight 7 pairs), Sharpe {sharpe:+.2f}, HAC t {t:+.2f}, one-sided p {p:.4f}")
    yr = port.groupby(port.index.year).sum()
    print("by year:", {k: round(v, 3) for k, v in yr.items()})
    print("per pair total 2006-18:", {k: round(v, 3) for k, v in P.sum().items()})
    trades = {k: int((v.diff().abs() > 0).sum()) for k, v in P.items()}
    if "--ledger" in sys.argv:
        prior = [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []
        sha = hashlib.sha256(b"pairs_bt:T1_pairs_z2").hexdigest()[:12]
        with LEDGER.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "strategy": "pairs_T1_z2", "sha": sha,
                                "dataset": "stocks_pairs", "status": "ok", "trial_number": len(prior) + 1, "ann": ann,
                                "sharpe": sharpe, "t": float(t), "p": p, "per_pair": P.sum().round(3).to_dict()}) + "\n")
        print("appended to stocks ledger")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    raise SystemExit({"dev": cmd_dev}.get(cmd, lambda: print(__doc__) or 1)())
